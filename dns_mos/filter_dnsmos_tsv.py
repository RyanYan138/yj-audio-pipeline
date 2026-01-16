# filter_dnsmos_tsv_stream.py (no pandas / no numpy) - memory optimized
import argparse
import csv
import math

NUM_COLS = ["start_sec", "end_sec", "duration_sec", "mos_sig", "mos_bak", "mos_ovr"]

def to_float(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return float(s)
    except Exception:
        return None

def quantile(values, p):
    """
    Linear interpolation quantile like numpy/pandas default (type=7-ish).
    values: list of floats, must be non-empty
    p in [0,1]
    """
    vs = sorted(values)
    n = len(vs)
    if n == 1:
        return vs[0]
    p = min(1.0, max(0.0, p))
    h = (n - 1) * p
    lo = int(math.floor(h))
    hi = int(math.ceil(h))
    if lo == hi:
        return vs[lo]
    return vs[lo] + (vs[hi] - vs[lo]) * (h - lo)

def row_pass_basic(r, args):
    # required keys
    if r.get("path") in (None, ""):
        return False

    # parse numeric columns (in-place)
    for c in NUM_COLS:
        r[c] = to_float(r.get(c))

    if r["start_sec"] is None or r["end_sec"] is None or r["duration_sec"] is None or r["mos_ovr"] is None:
        return False

    # basic sanity
    if not (args.min_dur <= r["duration_sec"] <= args.max_dur):
        return False
    if not (r["end_sec"] > r["start_sec"]):
        return False

    # optional constraints
    if args.min_mos_sig is not None:
        if r["mos_sig"] is None or r["mos_sig"] < args.min_mos_sig:
            return False
    if args.min_mos_bak is not None:
        if r["mos_bak"] is None or r["mos_bak"] < args.min_mos_bak:
            return False

    return True

def format_row(r):
    rr = dict(r)
    for c in NUM_COLS:
        if rr.get(c) is None:
            rr[c] = ""
        else:
            rr[c] = str(rr[c])
    return rr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_tsv", required=True)
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--min_dur", type=float, default=2.0)
    ap.add_argument("--max_dur", type=float, default=20.0)
    ap.add_argument("--keep_quantile", type=float, default=None,
                    help="e.g. 0.7 means keep top 70% by mos_ovr")
    ap.add_argument("--min_mos_ovr", type=float, default=None)
    ap.add_argument("--min_mos_sig", type=float, default=None)
    ap.add_argument("--min_mos_bak", type=float, default=None)
    args = ap.parse_args()

    # 0) read header only
    with open(args.in_tsv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise RuntimeError("Empty header in input tsv")

    # 1) if keep_quantile: first pass collect mos_ovr_list only (no storing rows)
    thr = None
    if args.keep_quantile is not None:
        mos_ovr_list = []
        with open(args.in_tsv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for r in reader:
                if row_pass_basic(r, args):
                    mos_ovr_list.append(r["mos_ovr"])

        if not mos_ovr_list:
            # write empty with header
            with open(args.out_tsv, "w", newline="", encoding="utf-8") as fo:
                writer = csv.DictWriter(fo, delimiter="\t", fieldnames=fieldnames)
                writer.writeheader()
            print("[WARN] no rows kept after basic filtering, wrote empty:", args.out_tsv)
            return

        q = float(args.keep_quantile)
        thr = quantile(mos_ovr_list, 1.0 - q)
        print(f"[INFO] keep_quantile={q}, mos_ovr_threshold={thr:.4f}")

    # 2) second pass: apply full filter, store only final kept rows for sorting
    kept = []
    with open(args.in_tsv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            if not row_pass_basic(r, args):
                continue

            # mos_ovr thresholding
            if thr is not None:
                if r["mos_ovr"] is None or r["mos_ovr"] < thr:
                    continue
            elif args.min_mos_ovr is not None:
                if r["mos_ovr"] is None or r["mos_ovr"] < float(args.min_mos_ovr):
                    continue

            kept.append(format_row(r))

    if thr is None and args.min_mos_ovr is None:
        print(f"[WARN] no mos_ovr threshold set, kept={len(kept)}")
    else:
        print(f"[INFO] kept={len(kept)}")

    if not kept:
        with open(args.out_tsv, "w", newline="", encoding="utf-8") as fo:
            writer = csv.DictWriter(fo, delimiter="\t", fieldnames=fieldnames)
            writer.writeheader()
        print("[WARN] no rows kept after mos filtering, wrote empty:", args.out_tsv)
        return

    # 3) sort for determinism (audio_id is string; start/end are strings now but comparable as float in your old code)
    # keep same semantics: you sorted by (audio_id, start_sec, end_sec) with start/end as floats.
    def k(r):
        return (r.get("audio_id", ""), to_float(r.get("start_sec")), to_float(r.get("end_sec")))
    kept.sort(key=k)

    # 4) write tsv
    with open(args.out_tsv, "w", newline="", encoding="utf-8") as fo:
        writer = csv.DictWriter(fo, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    print("[OK] wrote:", args.out_tsv)

if __name__ == "__main__":
    main()
