#!/usr/bin/env python3
"""把 tar_pipeline 输出的 labels.json 转成 WeNet 训练格式（train/dev/test.jsonl）"""
import json, re, random, argparse
from pathlib import Path
from collections import Counter

PUNCS = set("，。！？；：、（）()【】[]《》\"'''—…,.!?;:")

def norm_text(text):
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    chars = []
    for ch in text:
        if ch in PUNCS: continue
        if ch.isspace(): chars.append(" ")
        elif "一" <= ch <= "鿿": chars.append(ch)
        else: chars.append(ch.lower())
    return re.sub(r"\s+", " ", "".join(chars)).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--target_langs", nargs="*", default=["zh"])
    ap.add_argument("--min_lang_prob", type=float, default=0.90)
    ap.add_argument("--min_mos_ovr", type=float, default=0.0)
    ap.add_argument("--min_mos_sig", type=float, default=0.0)
    ap.add_argument("--min_mos_bak", type=float, default=0.0)
    ap.add_argument("--dev_ratio",  type=float, default=0.02)
    ap.add_argument("--test_ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=20260605)
    args = ap.parse_args()

    data = json.load(open(args.labels, encoding="utf-8"))
    print(f"读入 {len(data)} 条")

    rows = []
    skip_notext = skip_nowav = skip_dur = skip_lang = skip_mos = 0
    for i, item in enumerate(data):
        raw_text = item.get("transcribe", {}).get("funasr-nano", "") or item.get("text", "")
        text = norm_text(raw_text)
        if not text:
            skip_notext += 1; continue

        wav = item.get("wav")
        if not wav:
            skip_nowav += 1; continue

        dur = float(item.get("duration") or item.get("duration_sec") or 0)
        if dur < 1.0 or dur > 30.0:
            skip_dur += 1; continue

        lang = item.get("text_lang") or item.get("lang", "")
        lang_prob = float(item.get("lang_prob") or 0)
        if args.target_langs and lang not in args.target_langs:
            skip_lang += 1; continue
        if lang_prob < args.min_lang_prob:
            skip_lang += 1; continue

        mos_ovr = item.get("mos_ovr")
        mos_sig = item.get("mos_sig")
        mos_bak = item.get("mos_bak")
        if mos_ovr is not None and float(mos_ovr) < args.min_mos_ovr:
            skip_mos += 1; continue
        if mos_sig is not None and float(mos_sig) < args.min_mos_sig:
            skip_mos += 1; continue
        if mos_bak is not None and float(mos_bak) < args.min_mos_bak:
            skip_mos += 1; continue

        rows.append({
            "key":          item.get("wav_uuid", f"utt_{i:08d}"),
            "wav":          wav,
            "text":         text,
            "lang":         lang,
            "duration_sec": dur,
            "mos_ovr":      mos_ovr,
            "mos_sig":      mos_sig,
            "mos_bak":      mos_bak,
            "lang_prob":    lang_prob or None,
            "sim_spk":      None,
            "source_path":  item.get("tar_path"),
        })

    print(f"过滤后 {len(rows)} 条 (跳过: 无文本={skip_notext} 无wav={skip_nowav} 时长={skip_dur} 语言={skip_lang} MOS={skip_mos})")
    total_h = sum(r["duration_sec"] for r in rows) / 3600
    print(f"总时长: {total_h:.2f}h")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n = len(rows)
    n_test = max(1, int(n * args.test_ratio))
    n_dev  = max(1, int(n * args.dev_ratio))
    test  = rows[:n_test]
    dev   = rows[n_test:n_test+n_dev]
    train = rows[n_test+n_dev:]

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for split, sdata in [("train", train), ("dev", dev), ("test", test)]:
        p = out / f"{split}.jsonl"
        p.open("w").writelines(json.dumps(x, ensure_ascii=False)+"\n" for x in sdata)
        dur = sum(x["duration_sec"] for x in sdata)
        print(f"  {split}: {len(sdata)} 条, {dur/3600:.2f}h -> {p}")

    vocab_chars = Counter()
    for x in train: vocab_chars.update(x["text"])
    vocab = ["<pad>","<unk>","|"] + [c for c,_ in vocab_chars.most_common() if c != " "]
    (out/"vocab.json").write_text(json.dumps({c:i for i,c in enumerate(vocab)}, ensure_ascii=False, indent=2))
    print(f"  vocab: {len(vocab)} chars")

if __name__ == "__main__":
    main()
