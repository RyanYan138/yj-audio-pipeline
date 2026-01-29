#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import sqlite3
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


@dataclass
class Row:
    raw: Dict[str, str]


def read_tsv(path: str) -> Tuple[List[str], List[Row]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        header = reader.fieldnames or []
        rows = [Row(r) for r in reader]
    return header, rows


def write_tsv(path: str, header: List[str], rows: List[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def to_float(x: Optional[str]) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return float(s)
    except Exception:
        return None


def load_segment_audio(path: str, start_sec: float, end_sec: float) -> Tuple[np.ndarray, int]:
    """Read audio segment [start_sec, end_sec] using SoundFile seek/read."""
    with sf.SoundFile(path, "r") as f:
        sr = f.samplerate
        start = max(0, int(start_sec * sr))
        end = max(start, int(end_sec * sr))
        n = max(0, end - start)
        f.seek(start)
        audio = f.read(n, dtype="float32", always_2d=True)  # (n, ch)
    # to mono
    if audio.ndim == 2 and audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    else:
        audio = audio.reshape(-1)
    return audio, sr


def ensure_sr(audio: np.ndarray, sr: int, target_sr: int = 16000) -> np.ndarray:
    if sr == target_sr:
        return audio
    # resample with polyphase
    g = np.gcd(sr, target_sr)
    up = target_sr // g
    down = sr // g
    return resample_poly(audio, up, down).astype(np.float32)


def init_db(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True) if os.path.dirname(db_path) else None
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS lid_cache (audio_id TEXT PRIMARY KEY, lang TEXT, prob REAL)"
    )
    conn.commit()
    return conn


def db_get(conn: sqlite3.Connection, audio_id: str) -> Optional[Tuple[str, float]]:
    cur = conn.execute("SELECT lang, prob FROM lid_cache WHERE audio_id=?", (audio_id,))
    row = cur.fetchone()
    if row is None:
        return None
    return str(row[0]), float(row[1])


def db_put(conn: sqlite3.Connection, audio_id: str, lang: str, prob: float) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO lid_cache(audio_id, lang, prob) VALUES(?,?,?)",
        (audio_id, lang, float(prob)),
    )
    conn.commit()


def pick_probe_row(rows: List[Row]) -> Row:
    """Choose the row with the largest duration as probe."""
    best = None
    best_d = -1.0
    for r in rows:
        d = to_float(r.raw.get("duration_sec")) or 0.0
        if d > best_d:
            best_d = d
            best = r
    return best if best is not None else rows[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_tsv", required=True)
    ap.add_argument("--out_scores", required=True)
    ap.add_argument("--out_filtered", required=True)
    ap.add_argument("--cache_db", required=True)

    ap.add_argument("--model_id", default="Systran/faster-whisper-large-v3",
                    help="HF repo id or local path for faster-whisper")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute_type", default="float16")

    ap.add_argument("--target_lang", default="en",
                help="(deprecated) single target lang, kept for backward compatibility")
    ap.add_argument("--target_langs", nargs="*", default=None,
                help="Multiple target langs, e.g. --target_langs en zh")
    ap.add_argument("--min_lang_prob", type=float, default=0.90)

    ap.add_argument("--probe_max_sec", type=float, default=12.0,
                    help="Cap probe segment length to reduce cost")
    ap.add_argument("--min_probe_sec", type=float, default=1.0,
                    help="If probe is too short, mark as too_short_for_lid")

    args = ap.parse_args()
    def _parse_target_langs(args) -> List[str]:
        # 优先用 --target_langs；没给就回退到 --target_lang
        if args.target_langs is None or len(args.target_langs) == 0:
            raw = [args.target_lang]
        else:
            raw = args.target_langs

        # 允许用户写成 "en,zh" 这种
        out = []
        for x in raw:
            if x is None:
                continue
            for t in str(x).replace(",", " ").split():
                t = t.strip().lower()
                if t:
                    out.append(t)
        # 去重但保序
        seen = set()
        uniq = []
        for t in out:
            if t not in seen:
                uniq.append(t)
                seen.add(t)
        return uniq

    target_langs = _parse_target_langs(args)

    def _lang_hit(lang: str, targets: List[str]) -> bool:
        # faster-whisper 通常给 "en"/"zh"；也可能出现 "zh-cn" 这类，做个 base 匹配
        lang = (lang or "").strip().lower()
        if not lang:
            return False
        if lang in targets:
            return True
        base = lang.split("-")[0]
        return base in targets


    header, rows = read_tsv(args.in_tsv)
    if not rows:
        print("Empty input TSV.", file=sys.stderr)
        write_tsv(args.out_scores, header + ["lang", "lang_prob", "lang_keep_reason"], [])
        write_tsv(args.out_filtered, header + ["lang", "lang_prob", "lang_keep_reason"], [])
        return

    # group by audio_id
    by_aid: Dict[str, List[Row]] = {}
    for r in rows:
        aid = (r.raw.get("audio_id") or "").strip()
        by_aid.setdefault(aid, []).append(r)

    # init cache
    conn = init_db(args.cache_db)

    # load model
    from faster_whisper import WhisperModel
    model = WhisperModel(args.model_id, device=args.device, compute_type=args.compute_type)

    # detect lang per audio_id
    lang_map: Dict[str, Tuple[str, float, str]] = {}  # aid -> (lang, prob, reason)
    total = len(by_aid)
    done = 0

    for aid, rs in by_aid.items():
        done += 1

        cached = db_get(conn, aid)
        if cached is not None:
            lang, prob = cached
            lang_map[aid] = (lang, prob, "cached")
            continue

        probe = pick_probe_row(rs)
        path = probe.raw.get("path") or ""
        s = to_float(probe.raw.get("start_sec")) or 0.0
        e = to_float(probe.raw.get("end_sec")) or 0.0
        dur = max(0.0, e - s)
        if dur < args.min_probe_sec:
            lang_map[aid] = ("", 0.0, "too_short_for_lid")
            db_put(conn, aid, "", 0.0)
            continue

        # cap probe length
        if dur > args.probe_max_sec:
            e = s + args.probe_max_sec

        try:
            audio, sr = load_segment_audio(path, s, e)
            audio = ensure_sr(audio, sr, 16000)

            # Run a very light transcribe to get language info
            segments, info = model.transcribe(
                audio,
                beam_size=1,
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=False,
            )
            lang = (info.language or "").strip()
            prob = float(getattr(info, "language_probability", 0.0) or 0.0)

            lang_map[aid] = (lang, prob, "ok")
            db_put(conn, aid, lang, prob)

        except Exception as ex:
            lang_map[aid] = ("", 0.0, f"lid_error:{type(ex).__name__}")
            db_put(conn, aid, "", 0.0)

        if done % 200 == 0 or done == total:
            print(f"[LID] {done}/{total} audio_ids processed", file=sys.stderr)

    # build outputs
    out_header = header.copy()
    for c in ["lang", "lang_prob", "lang_keep_reason"]:
        if c not in out_header:
            out_header.append(c)

    scores_rows: List[Dict[str, str]] = []
    filtered_rows: List[Dict[str, str]] = []

    for r in rows:
        aid = (r.raw.get("audio_id") or "").strip()
        lang, prob, status = lang_map.get(aid, ("", 0.0, "missing"))
        keep_reason = "keep"

        if status.startswith("lid_error"):
            keep_reason = "lid_error"
        elif status == "too_short_for_lid":
            keep_reason = "too_short_for_lid"
        else:
            if not _lang_hit(lang, target_langs):
                keep_reason = "not_target_lang"
            elif prob < args.min_lang_prob:
                keep_reason = "low_lang_prob"
            else:
                keep_reason = "keep"
            

        rr = dict(r.raw)
        rr["lang"] = lang
        rr["lang_prob"] = f"{prob:.4f}"
        rr["lang_keep_reason"] = keep_reason
        scores_rows.append(rr)

        if keep_reason == "keep":
            filtered_rows.append(rr)

    write_tsv(args.out_scores, out_header, scores_rows)
    write_tsv(args.out_filtered, out_header, filtered_rows)

    # summary
    kept = len(filtered_rows)
    print(f"[DONE] input_rows={len(rows)} kept_rows={kept} keep_ratio={kept/len(rows):.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
