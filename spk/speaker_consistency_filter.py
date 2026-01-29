#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
speaker_consistency_filter.py (AUTO spk_id version)
Purpose: speaker-consistency filtering for segmented TSV.
- Keep existing naming style (spk_id + sim_spk + sim_file + keep_reason).
- If dirty/unlabeled data cannot provide speaker id, auto-create spk_id per-file via embedding clustering.

Input TSV columns (at least):
  seg_id, audio_id, path, start_sec, end_sec, duration_sec, ...

Output TSV:
  original cols + spk_id, sim_spk, sim_file, keep_reason

Dependencies:
  pip install speechbrain torchaudio soundfile huggingface_hub

Usage example:
  python speaker_consistency_filter.py \
    --in_tsv  dns_filtered_q70.tsv \
    --out_tsv spk_filtered.tsv \
    --cache_db spk_emb_cache.sqlite \
    --device cuda:0 \
    --min_sim_spk 0.60 --min_sim_file 0.65 --min_dur 1.0 \
    --cluster_sim 0.74 \
    --spk_id_col spk_id \
    --spk_patterns "\\bS\\d{3,6}\\b" "\\bid\\d{3,6}\\b"

Notes:
- --model 可以给 HF repo（可能联网/走缓存），也可以直接给本地目录（完全离线）
  例如：--model /Work21/xxx/speechbrain_models/spkrec-ecapa-voxceleb
"""

import argparse
import csv
import os
import re
import sqlite3
from array import array
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import torch


# =========================
# Audio I/O
# =========================
def load_segment_audio(path: str, start_sec: float, end_sec: float) -> Tuple[torch.Tensor, int]:
    """Return: (wav[1, T] float32, sr). Try soundfile first, fallback torchaudio."""
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
            x = x.mean(axis=1, keepdims=True)
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


# =========================
# SQLite Cache (embeddings + meta + spkmap)
# =========================
def ensure_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS embeddings ("
        "seg_id TEXT PRIMARY KEY,"
        "dim INTEGER NOT NULL,"
        "data BLOB NOT NULL"
        ")"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS meta ("
        "seg_id TEXT PRIMARY KEY,"
        "file_id TEXT NOT NULL,"
        "start_sec REAL NOT NULL,"
        "spk_raw TEXT NOT NULL,"
        "spk_final TEXT NOT NULL"
        ")"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_meta_file ON meta(file_id);")
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    return conn


def db_get_emb(conn: sqlite3.Connection, seg_id: str) -> Optional[torch.Tensor]:
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


def db_put_emb(conn: sqlite3.Connection, seg_id: str, emb: torch.Tensor):
    emb = emb.detach().cpu().float().contiguous().view(-1)
    arr = array("f", emb.tolist())
    blob = arr.tobytes()
    conn.execute(
        "INSERT OR REPLACE INTO embeddings(seg_id, dim, data) VALUES (?,?,?)",
        (seg_id, emb.numel(), sqlite3.Binary(blob))
    )


def db_upsert_meta(conn: sqlite3.Connection, seg_id: str, file_id: str, start_sec: float, spk_raw: str):
    # 初始 spk_final 先等于 spk_raw，后面如果是 UNKNOWN 会被自动替换
    conn.execute(
        "INSERT OR REPLACE INTO meta(seg_id, file_id, start_sec, spk_raw, spk_final) VALUES (?,?,?,?,?)",
        (seg_id, file_id, float(start_sec), spk_raw, spk_raw)
    )


def db_update_spk_final(conn: sqlite3.Connection, seg_id: str, spk_final: str):
    conn.execute("UPDATE meta SET spk_final=? WHERE seg_id=?", (spk_final, seg_id))


# =========================
# Math
# =========================
def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(p=2) + eps)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(torch.dot(a, b).clamp(-1.0, 1.0).item())


# =========================
# Speaker ID Parsing (dataset-agnostic)
# =========================
DEFAULT_SPK_PATTERNS: List[str] = [
    r"\bS\d{3,6}\b",
    r"\bid\d{3,6}\b",
    r"\bspk(?:er|r)?[_\-]?\d{1,6}\b",
    r"\bspeaker[_\-]?\d{1,6}\b",
    r"\b\d{3,6}[_\-]\d{3,6}\b",
    r"(?<!\d)\d{3,6}(?!\d)",
]


def _search_patterns(s: str, patterns: List[str]) -> Optional[str]:
    if not s:
        return None
    for pat in patterns:
        m = re.search(pat, s)
        if m:
            return m.group(0)
    return None


def parse_spk_id(audio_id: str, path: str, extra_patterns: Optional[List[str]] = None) -> str:
    patterns = list(DEFAULT_SPK_PATTERNS)
    if extra_patterns:
        patterns = list(extra_patterns) + patterns

    audio_id = (audio_id or "").strip()
    path = (path or "").strip()

    hit = _search_patterns(audio_id, patterns)
    if hit:
        return hit

    p = Path(path)
    parts = [x for x in p.parts if x not in ("/", "")]
    for token in reversed(parts):
        hit = _search_patterns(token, patterns)
        if hit:
            return hit

    for i, name in enumerate(parts):
        if name in {"dev", "train", "test"} and i + 1 < len(parts):
            candidate = parts[i + 1]
            hit = _search_patterns(candidate, patterns)
            if hit:
                return hit
            if candidate:
                return candidate

    for i, name in enumerate(parts):
        if name in {
            "dev-clean", "dev-other",
            "test-clean", "test-other",
            "train-clean-100", "train-clean-360", "train-other-500"
        } and i + 1 < len(parts):
            return parts[i + 1]

    return "UNKNOWN"


def get_file_id(audio_id: str, path: str) -> str:
    audio_id = (audio_id or "").strip()
    if audio_id:
        return audio_id
    p = Path(path or "")
    if p.name:
        return p.stem
    return "UNKNOWN_FILE"


def sanitize_id(s: str, max_len: int = 80) -> str:
    s = (s or "").strip()
    if not s:
        return "UNKNOWN"
    s = re.sub(r"[^0-9A-Za-z_\-\.]+", "_", s)
    return s[:max_len] if len(s) > max_len else s


# =========================
# AUTO spk_id: greedy clustering per file (UNKNOWN only)
# =========================
def auto_assign_spk_ids_per_file(
    conn: sqlite3.Connection,
    file_id: str,
    cluster_sim: float,
    auto_prefix: str = "AUTO",
):
    """
    对某个 file 内 spk_raw == UNKNOWN 的 segments 进行贪心聚类，生成 spk_final：
      AUTO_<file_id>_p000, AUTO_<file_id>_p001, ...
    已有明确 spk_raw 的保持不动。
    """
    cur = conn.execute(
        "SELECT seg_id, start_sec, spk_raw FROM meta WHERE file_id=? ORDER BY start_sec ASC",
        (file_id,)
    )
    rows = cur.fetchall()
    if not rows:
        return

    # 只对 UNKNOWN 做聚类，其他保持 spk_final=spk_raw
    unknown_seg_ids: List[str] = []
    for seg_id, _st, spk_raw in rows:
        if (spk_raw or "") == "UNKNOWN":
            unknown_seg_ids.append(seg_id)

    if not unknown_seg_ids:
        return

    safe_fid = sanitize_id(file_id)
    prefix = f"{auto_prefix}_{safe_fid}"

    centroids: List[torch.Tensor] = []
    # 为保证顺序稳定：按 start_sec 顺序遍历 unknown
    for seg_id, _st, spk_raw in rows:
        if spk_raw != "UNKNOWN":
            continue
        emb = db_get_emb(conn, seg_id)
        if emb is None:
            # 没 emb 的就先不动，后面 pass2 会 drop_no_emb
            continue

        emb = l2_normalize(emb)

        if not centroids:
            centroids.append(emb.clone())
            spk_final = f"{prefix}_p000"
            db_update_spk_final(conn, seg_id, spk_final)
            continue

        sims = [cosine(emb, c) for c in centroids]
        best_k = int(max(range(len(sims)), key=lambda i: sims[i]))
        best_sim = sims[best_k]

        if best_sim >= cluster_sim:
            # 归并更新中心（用加和再归一化，够用且快）
            centroids[best_k] = l2_normalize(centroids[best_k] + emb)
            spk_final = f"{prefix}_p{best_k:03d}"
            db_update_spk_final(conn, seg_id, spk_final)
        else:
            centroids.append(emb.clone())
            new_k = len(centroids) - 1
            spk_final = f"{prefix}_p{new_k:03d}"
            db_update_spk_final(conn, seg_id, spk_final)


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_tsv", required=True)
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--cache_db", required=True)

    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--model", default="speechbrain/spkrec-ecapa-voxceleb",
                    help="HF repo id or local directory (offline).")

    ap.add_argument("--min_dur", type=float, default=1.0)
    ap.add_argument("--min_sim_spk", type=float, default=0.60)
    ap.add_argument("--min_sim_file", type=float, default=0.65)

    # 新增：自动造 spk_id 的聚类阈值（只对 UNKNOWN 生效）
    ap.add_argument("--cluster_sim", type=float, default=0.74,
                    help="Greedy clustering threshold for AUTO spk_id (UNKNOWN only).")

    ap.add_argument("--add_scores_tsv", default=None,
                    help="optional: write all rows + sim columns (before filtering)")

    ap.add_argument("--torch_threads", type=int, default=4)

    ap.add_argument("--spk_id_col", default="",
                    help="If TSV already has speaker id column, set it here (e.g., spk_id). Empty means auto-parse.")
    ap.add_argument("--spk_patterns", nargs="*", default=None,
                    help=r"Extra regex patterns to extract spk_id from audio_id/path. Example: \bS\d{3,6}\b")

    ap.add_argument("--force_offline", type=int, default=0,
                    help="1=force offline mode for HF hub (set HF_HUB_OFFLINE=1 etc).")

    args = ap.parse_args()
    torch.set_num_threads(max(1, args.torch_threads))

    if args.force_offline == 1:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    # ---- load speaker embedding model ----
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except Exception:
        from speechbrain.pretrained import EncoderClassifier

    model_source = args.model
    is_local = os.path.isdir(model_source)

    classifier = EncoderClassifier.from_hparams(
        source=model_source,
        run_opts={"device": args.device},
    )

    conn = ensure_db(args.cache_db)

    # pass1: compute embeddings + write meta(spk_raw) + list files
    total_rows = 0
    computed = 0
    skipped_short = 0
    committed = 0

    spk_id_col = (args.spk_id_col or "").strip()
    seen_files: Dict[str, int] = {}

    with open(args.in_tsv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            total_rows += 1
            seg_id = (row.get("seg_id") or "").strip()
            if not seg_id:
                continue

            audio_id = (row.get("audio_id") or "").strip()
            path = (row.get("path") or "").strip()

            try:
                start_sec = float(row.get("start_sec", "0") or "0")
                end_sec = float(row.get("end_sec", "0") or "0")
                dur = float(row.get("duration_sec", "0") or "0")
            except Exception:
                continue

            if dur < args.min_dur:
                skipped_short += 1
                continue

            emb = db_get_emb(conn, seg_id)
            if emb is None:
                wav, sr = load_segment_audio(path, start_sec, end_sec)
                if wav.numel() == 0:
                    continue
                wav = resample_to_16k(wav, sr)
                with torch.no_grad():
                    e = classifier.encode_batch(wav.to(args.device))
                emb = e.squeeze().detach().cpu().float()
                emb = l2_normalize(emb)
                db_put_emb(conn, seg_id, emb)
                computed += 1

            # spk_raw：优先用列，其次正则解析；失败就 UNKNOWN
            spk_raw = ""
            if spk_id_col:
                spk_raw = (row.get(spk_id_col) or "").strip()
            if not spk_raw:
                spk_raw = parse_spk_id(audio_id, path, extra_patterns=args.spk_patterns)
            if not spk_raw:
                spk_raw = "UNKNOWN"

            file_id = get_file_id(audio_id, path)
            seen_files[file_id] = 1

            db_upsert_meta(conn, seg_id, file_id, start_sec, spk_raw)
            committed += 1

            if committed % 300 == 0:
                conn.commit()

    conn.commit()

    # pass1.5: for dirty/unlabeled (UNKNOWN) -> auto-create spk_final per-file
    # 只对 UNKNOWN 生效，已有 spk_raw 的保持命名风格不变
    for file_id in seen_files.keys():
        auto_assign_spk_ids_per_file(
            conn=conn,
            file_id=file_id,
            cluster_sim=float(args.cluster_sim),
            auto_prefix="AUTO",
        )
    conn.commit()

    # build centroids (spk_final + file_id)
    sum_spk: Dict[str, torch.Tensor] = {}
    cnt_spk: Dict[str, int] = {}
    sum_file: Dict[str, torch.Tensor] = {}
    cnt_file: Dict[str, int] = {}

    cur = conn.execute("SELECT seg_id, file_id, spk_final FROM meta")
    for seg_id, file_id, spk_final in cur.fetchall():
        emb = db_get_emb(conn, seg_id)
        if emb is None:
            continue
        emb = l2_normalize(emb)

        if spk_final not in sum_spk:
            sum_spk[spk_final] = torch.zeros_like(emb)
            cnt_spk[spk_final] = 0
        if file_id not in sum_file:
            sum_file[file_id] = torch.zeros_like(emb)
            cnt_file[file_id] = 0

        sum_spk[spk_final] += emb
        cnt_spk[spk_final] += 1

        sum_file[file_id] += emb
        cnt_file[file_id] += 1

    spk_centroid: Dict[str, torch.Tensor] = {}
    file_centroid: Dict[str, torch.Tensor] = {}

    for k, s in sum_spk.items():
        c = s / max(1, cnt_spk.get(k, 1))
        spk_centroid[k] = l2_normalize(c)

    for k, s in sum_file.items():
        c = s / max(1, cnt_file.get(k, 1))
        file_centroid[k] = l2_normalize(c)

    # pass2: compute sims + filter + write
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
                seg_id = (row.get("seg_id") or "").strip()
                audio_id = (row.get("audio_id") or "").strip()
                path = (row.get("path") or "").strip()

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

                emb = db_get_emb(conn, seg_id)
                if emb is None:
                    removed += 1
                    removed_missing += 1
                    row.update({"spk_id": "", "sim_spk": "", "sim_file": "", "keep_reason": "drop_no_emb"})
                    if writer_all:
                        writer_all.writerow(row)
                    continue

                # 取 file_id / spk_final（脏数据已被自动造 ID）
                file_id = get_file_id(audio_id, path)
                cur2 = conn.execute("SELECT spk_final FROM meta WHERE seg_id=?", (seg_id,))
                r2 = cur2.fetchone()
                spk_id = r2[0] if r2 and r2[0] else "UNKNOWN"

                c_spk = spk_centroid.get(spk_id)
                c_file = file_centroid.get(file_id)

                sim_spk = cosine(l2_normalize(emb), c_spk) if c_spk is not None else -1.0
                sim_file = cosine(l2_normalize(emb), c_file) if c_file is not None else -1.0

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

    print("=== Speaker consistency filter (AUTO spk_id) done ===")
    print(f"Input rows: {total_rows}")
    print(f"Embeddings computed this run: {computed} (cache used for others)")
    print(f"Skipped short in pass1 (<{args.min_dur}s): {skipped_short}")
    print(f"Output written to: {args.out_tsv}")
    if args.add_scores_tsv:
        print(f"All-rows-with-scores TSV: {args.add_scores_tsv}")
    print("Tip: dirty data too mixed? raise --cluster_sim (e.g. 0.78) and/or raise thresholds.")


if __name__ == "__main__":
    main()
 