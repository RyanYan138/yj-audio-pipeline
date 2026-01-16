# speaker_consistency_filter.py
# Purpose: speaker-consistency filtering for segmented TSV
# Input TSV columns (at least): seg_id,audio_id,path,start_sec,end_sec,duration_sec,...
# Output TSV: original cols + spk_id,sim_spk,sim_file,keep_reason
#
# Dependencies:
#   pip install speechbrain torchaudio soundfile
#
# Usage example:
#   python speaker_consistency_filter.py \
#     --in_tsv  /CDShare3/Huawei_Encoder_Proj/codes/jiahao/SE_dns_mos/out_librispeech/dns_filtered_q70.tsv \
#     --out_tsv /CDShare3/Huawei_Encoder_Proj/codes/jiahao/SE_dns_mos/out_librispeech/spk_filtered.tsv \
#     --cache_db /CDShare3/Huawei_Encoder_Proj/codes/jiahao/SE_dns_mos/out_librispeech/spk_emb_cache.sqlite \
#     --device cuda:0 \
#     --min_sim_spk 0.60 --min_sim_file 0.65 --min_dur 1.0

import argparse
import csv
import os
import sqlite3
from array import array
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch

# audio i/o
def load_segment_audio(path: str, start_sec: float, end_sec: float) -> Tuple[torch.Tensor, int]:
    """
    Return: (wav[1, T] float32, sr)
    Tries soundfile first, falls back to torchaudio.
    """
    try:
        import soundfile as sf
        with sf.SoundFile(path) as f:
            sr = int(f.samplerate)
            start_frame = max(0, int(round(start_sec * sr)))
            end_frame = max(start_frame + 1, int(round(end_sec * sr)))
            total_frames = len(f)
            if start_frame >= total_frames:
                return torch.zeros(1, 0, dtype=torch.float32), sr
            end_frame = min(end_frame, total_frames)
            f.seek(start_frame)
            frames = end_frame - start_frame
            x = f.read(frames, dtype="float32", always_2d=True)  # [T, C]
        if x.shape[1] > 1:
            x = x.mean(axis=1, keepdims=True)  # [T, 1]
        x = torch.from_numpy(x.squeeze(1)).unsqueeze(0)  # [1, T]
        return x, sr
    except Exception:
        import torchaudio
        wav, sr = torchaudio.load(path)  # [C, T]
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        start = max(0, int(round(start_sec * sr)))
        end = max(start + 1, int(round(end_sec * sr)))
        end = min(end, wav.shape[1])
        return wav[:, start:end].contiguous(), int(sr)


def resample_to_16k(wav: torch.Tensor, sr: int) -> torch.Tensor:
    if sr == 16000:
        return wav
    import torchaudio
    return torchaudio.functional.resample(wav, sr, 16000)


def parse_spk_id(audio_id: str, path: str) -> str:
    # First try audio_id pattern: dev-clean_1272_128104_1272-128104-0000
    if audio_id:
        parts = audio_id.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            return parts[1]

    # Fallback: parse LibriSpeech-like path .../dev-clean/1272/128104/xxx.flac
    p = Path(path)
    parts = list(p.parts)
    for i, name in enumerate(parts):
        if name in {
            "dev-clean", "dev-other",
            "test-clean", "test-other",
            "train-clean-100", "train-clean-360", "train-other-500"
        }:
            if i + 1 < len(parts):
                return parts[i + 1]
    return "UNKNOWN"


def ensure_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS embeddings ("
        "seg_id TEXT PRIMARY KEY,"
        "dim INTEGER NOT NULL,"
        "data BLOB NOT NULL"
        ")"
    )
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def db_get(conn: sqlite3.Connection, seg_id: str) -> Optional[torch.Tensor]:
    cur = conn.execute("SELECT dim, data FROM embeddings WHERE seg_id=?", (seg_id,))
    row = cur.fetchone()
    if row is None:
        return None
    dim, blob = int(row[0]), row[1]
    arr = array("f")
    arr.frombytes(blob)
    if len(arr) != dim:
        return None
    return torch.tensor(arr, dtype=torch.float32)


def db_put(conn: sqlite3.Connection, seg_id: str, emb: torch.Tensor):
    emb = emb.detach().cpu().float().contiguous().view(-1)
    arr = array("f", emb.tolist())
    blob = arr.tobytes()
    conn.execute(
        "INSERT OR REPLACE INTO embeddings(seg_id, dim, data) VALUES (?,?,?)",
        (seg_id, emb.numel(), sqlite3.Binary(blob))
    )


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(p=2) + eps)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    # expects both 1D
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(torch.dot(a, b).clamp(-1.0, 1.0).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_tsv", required=True)
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--cache_db", required=True)

    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model", default="speechbrain/spkrec-ecapa-voxceleb")

    ap.add_argument("--min_dur", type=float, default=1.0)

    # filtering thresholds
    ap.add_argument("--min_sim_spk", type=float, default=0.60)
    ap.add_argument("--min_sim_file", type=float, default=0.65)

    ap.add_argument("--add_scores_tsv", default=None,
                    help="optional: write a TSV with all rows + sim columns (before filtering)")

    ap.add_argument("--torch_threads", type=int, default=4)

    args = ap.parse_args()
    torch.set_num_threads(max(1, args.torch_threads))

    # load speaker embedding model
    from speechbrain.pretrained import EncoderClassifier
    classifier = EncoderClassifier.from_hparams(
        source=args.model,
        run_opts={"device": args.device},
    )

    conn = ensure_db(args.cache_db)

    # pass1: compute embeddings + accumulate centroids
    sum_spk: Dict[str, torch.Tensor] = {}
    cnt_spk: Dict[str, int] = {}
    sum_file: Dict[str, torch.Tensor] = {}
    cnt_file: Dict[str, int] = {}

    total_rows = 0
    computed = 0
    skipped_short = 0

    with open(args.in_tsv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            total_rows += 1
            seg_id = row.get("seg_id", "").strip()
            audio_id = row.get("audio_id", "").strip()
            path = row.get("path", "").strip()
            try:
                start_sec = float(row.get("start_sec", "0") or "0")
                end_sec = float(row.get("end_sec", "0") or "0")
                dur = float(row.get("duration_sec", "0") or "0")
            except Exception:
                continue

            if dur < args.min_dur:
                skipped_short += 1
                continue

            emb = db_get(conn, seg_id)
            if emb is None:
                wav, sr = load_segment_audio(path, start_sec, end_sec)
                if wav.numel() == 0:
                    continue
                wav = resample_to_16k(wav, sr)
                # speechbrain expects [B, T]
                with torch.no_grad():
                    e = classifier.encode_batch(wav.to(args.device))  # [B, 1, D] or [B, D]
                emb = e.squeeze().detach().cpu().float()
                emb = l2_normalize(emb)
                db_put(conn, seg_id, emb)
                computed += 1

            spk_id = parse_spk_id(audio_id, path)

            # init sums
            if spk_id not in sum_spk:
                sum_spk[spk_id] = torch.zeros_like(emb)
                cnt_spk[spk_id] = 0
            if audio_id not in sum_file:
                sum_file[audio_id] = torch.zeros_like(emb)
                cnt_file[audio_id] = 0

            sum_spk[spk_id] += emb
            cnt_spk[spk_id] += 1

            sum_file[audio_id] += emb
            cnt_file[audio_id] += 1

            # commit periodically
            if computed % 200 == 0:
                conn.commit()

    conn.commit()

    # build centroids (normalized)
    spk_centroid: Dict[str, torch.Tensor] = {}
    file_centroid: Dict[str, torch.Tensor] = {}

    for k, s in sum_spk.items():
        c = s / max(1, cnt_spk.get(k, 1))
        spk_centroid[k] = l2_normalize(c)

    for k, s in sum_file.items():
        c = s / max(1, cnt_file.get(k, 1))
        file_centroid[k] = l2_normalize(c)

    # pass2: compute sims + filter
    with open(args.in_tsv, "r", encoding="utf-8") as fin:
        reader = csv.DictReader(fin, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        extra_cols = ["spk_id", "sim_spk", "sim_file", "keep_reason"]
        out_fields = fieldnames + [c for c in extra_cols if c not in fieldnames]

        fout_all = None
        writer_all = None
        if args.add_scores_tsv:
            os.makedirs(str(Path(args.add_scores_tsv).parent), exist_ok=True)
            fout_all = open(args.add_scores_tsv, "w", encoding="utf-8", newline="")
            writer_all = csv.DictWriter(fout_all, fieldnames=out_fields, delimiter="\t")
            writer_all.writeheader()

        os.makedirs(str(Path(args.out_tsv).parent), exist_ok=True)
        with open(args.out_tsv, "w", encoding="utf-8", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=out_fields, delimiter="\t")
            writer.writeheader()

            kept = 0
            removed = 0
            removed_spk = 0
            removed_file = 0
            removed_missing = 0
            removed_short = 0

            for row in reader:
                seg_id = row.get("seg_id", "").strip()
                audio_id = row.get("audio_id", "").strip()
                path = row.get("path", "").strip()

                try:
                    dur = float(row.get("duration_sec", "0") or "0")
                except Exception:
                    dur = 0.0

                if dur < args.min_dur:
                    removed += 1
                    removed_short += 1
                    row.update({"spk_id": "", "sim_spk": "", "sim_file": "", "keep_reason": "drop_short"})
                    if writer_all:
                        writer_all.writerow(row)
                    continue

                emb = db_get(conn, seg_id)
                if emb is None:
                    removed += 1
                    removed_missing += 1
                    row.update({"spk_id": "", "sim_spk": "", "sim_file": "", "keep_reason": "drop_no_emb"})
                    if writer_all:
                        writer_all.writerow(row)
                    continue

                spk_id = parse_spk_id(audio_id, path)
                c_spk = spk_centroid.get(spk_id)
                c_file = file_centroid.get(audio_id)

                sim_spk = cosine(emb, c_spk) if c_spk is not None else -1.0
                sim_file = cosine(emb, c_file) if c_file is not None else -1.0

                ok_spk = (sim_spk >= args.min_sim_spk) if c_spk is not None else True
                ok_file = (sim_file >= args.min_sim_file) if c_file is not None else True

                keep_reason = "keep"
                if not ok_spk and not ok_file:
                    keep_reason = "drop_spk_and_file"
                    removed += 1
                    removed_spk += 1
                    removed_file += 1
                elif not ok_spk:
                    keep_reason = "drop_spk"
                    removed += 1
                    removed_spk += 1
                elif not ok_file:
                    keep_reason = "drop_file"
                    removed += 1
                    removed_file += 1
                else:
                    kept += 1

                row.update({
                    "spk_id": spk_id,
                    "sim_spk": f"{sim_spk:.4f}",
                    "sim_file": f"{sim_file:.4f}",
                    "keep_reason": keep_reason
                })

                if writer_all:
                    writer_all.writerow(row)
                if keep_reason == "keep":
                    writer.writerow(row)

        if fout_all:
            fout_all.close()

    conn.close()

    print("=== Speaker consistency filter done ===")
    print(f"Input rows: {total_rows}")
    print(f"Embeddings computed this run: {computed} (cache used for others)")
    print(f"Skipped short in pass1 (<{args.min_dur}s): {skipped_short}")
    print(f"Output written to: {args.out_tsv}")
    if args.add_scores_tsv:
        print(f"All-rows-with-scores TSV: {args.add_scores_tsv}")
    print("Tip: If you drop too much, lower thresholds: e.g., --min_sim_file 0.60 --min_sim_spk 0.55")


if __name__ == "__main__":
    main()
