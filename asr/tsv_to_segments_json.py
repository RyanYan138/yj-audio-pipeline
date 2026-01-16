#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, csv, json
from collections import OrderedDict

def ffloat(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_tsv", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    by_audio = OrderedDict()

    with open(args.in_tsv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            aid = (r.get("audio_id") or "").strip()
            if not aid:
                continue

            if aid not in by_audio:
                by_audio[aid] = {
                    "audio_id": aid,
                    "path": (r.get("path") or "").strip(),
                    "segments": []
                }

            by_audio[aid]["segments"].append({
                "seg_id": (r.get("seg_id") or "").strip(),
                "start_sec": ffloat(r.get("start_sec"), 0.0),
                "end_sec": ffloat(r.get("end_sec"), 0.0),
                "duration_sec": ffloat(r.get("duration_sec"), 0.0),

                # 这些是可选保留字段（你 TSV 里有就带上）
                "lang": (r.get("lang") or "").strip(),
                "lang_prob": ffloat(r.get("lang_prob"), 0.0),

                "mos_sig": ffloat(r.get("mos_sig"), 0.0),
                "mos_bak": ffloat(r.get("mos_bak"), 0.0),
                "mos_ovr": ffloat(r.get("mos_ovr"), 0.0),

                "spk_id": (r.get("spk_id") or "").strip(),
                "sim_spk": ffloat(r.get("sim_spk"), 0.0),
                "sim_file": ffloat(r.get("sim_file"), 0.0),

                "keep_reason": (r.get("keep_reason") or "").strip(),
                "lang_keep_reason": (r.get("lang_keep_reason") or "").strip(),
            })

    items = list(by_audio.values())
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False)

    print(f"Wrote {len(items)} audio items to {args.out_json}")

if __name__ == "__main__":
    main()
