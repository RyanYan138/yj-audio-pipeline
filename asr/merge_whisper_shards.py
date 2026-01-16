#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path

def norm_f(x, nd=3):
    try:
        return round(float(x), nd)
    except Exception:
        return None

def make_key(d):
    # 用更稳的联合键：audio_id + start/end + seg_id
    return (
        (d.get("audio_id") or "").strip(),
        norm_f(d.get("start_sec")),
        norm_f(d.get("end_sec")),
        (d.get("seg_id") or "").strip(),
    )

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg_json", required=True, help="原始 segments json（决定输出顺序）")
    ap.add_argument("--inputs", nargs="+", required=True, help="各 shard 输出 json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    seg_items = load_json(args.seg_json)

    # 1) 先构造“期望顺序”的 key 列表（逐 segment）
    expected_keys = []
    expected_metas = []
    for item in seg_items:
        audio_id = item.get("audio_id")
        path = item.get("path") or item.get("source")
        for seg in item.get("segments", []):
            meta = {
                "audio_id": audio_id,
                "path": path,
                "seg_id": seg.get("seg_id"),
                "start_sec": seg.get("start_sec"),
                "end_sec": seg.get("end_sec"),
                "duration_sec": seg.get("duration_sec"),
                # 额外字段如果 seg_json 里有，也保留下来（可选）
                "lang": seg.get("lang"),
                "lang_prob": seg.get("lang_prob"),
            }
            expected_metas.append(meta)
            expected_keys.append(make_key(meta))

    # 2) 读入所有 shard 结果，建立 key -> result 的映射
    result_map = {}
    dup = 0
    total_read = 0
    for p in args.inputs:
        arr = load_json(p)
        total_read += len(arr)
        for r in arr:
            k = make_key(r)
            if k in result_map:
                dup += 1
            result_map[k] = r

    # 3) 按 expected_keys 顺序输出，缺的就打标
    merged = []
    missing = 0
    for meta, k in zip(expected_metas, expected_keys):
        r = result_map.get(k)
        if r is None:
            missing += 1
            out = meta.copy()
            out["text"] = ""
            out["error"] = "missing_from_shards"
        else:
            # 以 shard 输出为主，但补齐 meta 里可能存在的字段
            out = meta.copy()
            out.update(r)
        merged.append(out)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"[merge] expected_segments={len(expected_keys)}")
    print(f"[merge] read_from_shards={total_read}, dup_keys={dup}")
    print(f"[merge] missing={missing}")
    print(f"[merge] wrote: {out_path}")

if __name__ == "__main__":
    main()
