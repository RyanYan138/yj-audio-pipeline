import argparse
import json
import os
import glob
import csv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vad-json",
        type=str,
        required=True,
        help="Path to VAD json file, e.g. librispeech_silero_vad_segments_mp.json",
    )
    parser.add_argument(
        "--shard-dir",
        type=str,
        required=True,
        help="Directory containing dns_from_vad_*_*.tsv files",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to merged output tsv (in desired json order)",
    )
    return parser.parse_args()


def build_seg_order_from_json(vad_json_path):
    """
    根据原始 VAD json，按顺序生成 seg_id 列表：
    test-other_..._0000_seg000, test-other_..._0000_seg001, ...
    """
    print(f"Loading VAD json from: {vad_json_path}")
    with open(vad_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    order = []
    for item in data:
        audio_id = item["audio_id"]
        segments = item.get("segments", [])
        for i, _seg in enumerate(segments):
            seg_id = f"{audio_id}_seg{i:03d}"
            order.append(seg_id)
    print(f"Total segments in VAD json: {len(order)}")
    return order


def load_dns_shards(shard_dir):
    """
    读取所有 dns_from_vad_*_*.tsv，构建 seg_id -> row 的映射。
    """
    pattern = os.path.join(shard_dir, "dns_from_vad_*_*.tsv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise RuntimeError(f"No shard files matched: {pattern}")

    print("Found shard files:")
    for f in files:
        print("  ", f)

    seg2row = {}
    header = None

    for idx, fpath in enumerate(files):
        with open(fpath, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            file_header = next(reader)  # first line is header
            if header is None:
                header = file_header
            else:
                if file_header != header:
                    print(f"Warning: header mismatch in {fpath}")

            for row in reader:
                if not row:
                    continue
                seg_id = row[0]
                seg2row[seg_id] = row

    print(f"Loaded {len(seg2row)} rows from shards.")
    return header, seg2row


def main():
    args = parse_args()

    seg_order = build_seg_order_from_json(args.vad_json)
    header, seg2row = load_dns_shards(args.shard_dir)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"Writing merged tsv (json order) to: {args.output}")

    missing = 0
    written = 0

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)

        for seg_id in seg_order:
            row = seg2row.get(seg_id)
            if row is None:
                # 有可能因为 min_dur 等筛过，缺一些 segment，这里可以选择跳过
                missing += 1
                continue
            writer.writerow(row)
            written += 1

    print(f"Done. Written rows: {written}, missing segments (not found in shards): {missing}")


if __name__ == "__main__":
    main()
