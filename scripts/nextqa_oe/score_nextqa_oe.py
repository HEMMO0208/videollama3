#!/usr/bin/env python3
"""Score NExT-QA Open-Ended predictions using WUPS metrics.

All metric code (metrics.py, stopwords.txt) is vendored into this directory
from https://github.com/doc-doc/NExT-OE — no external repository required.

Input:
  --pred-path   flat JSONL produced by infer_nextqa_oe_jsonl.py
                OR .nested.json in {video_id: {qid: pred_text}} format

  --ref-csv     OE reference CSV (e.g. dataset/nextqa/openend/val.csv)
                Columns: video, qid, answer, type

  --add-ref     (optional) JSON with additional reference answers
                {video_id: {qid: text}} — e.g. NExT-OE's
                add_reference_answer_val.json (absent for val split)

Output:
  .wups.json alongside --pred-path with per-type WUPS@0 / WUPS@0.9
  Tab-separated table printed to stdout (matches NExT-OE format)

Dependencies:
  pip install nltk pywsd pandas
  python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt'); nltk.download('punkt_tab')"
  (pywsd is optional — falls back to simple lowercasing if absent)
"""
import argparse
import json
import os.path as osp
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Local vendored imports (same directory as this script)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from metrics import get_wups  # noqa: E402  (vendored in scripts/nextqa_oe/)

# ---------------------------------------------------------------------------
# Stop-word removal — uses pywsd lemmatisation if available, else lowercase
# ---------------------------------------------------------------------------
_SW_PATH = SCRIPT_DIR / "stopwords.txt"
with open(_SW_PATH) as _f:
    _STOPWORDS = {line.strip() for line in _f if line.strip()}

try:
    from pywsd.utils import lemmatize_sentence as _lemmatize

    def remove_stop(sentence: str) -> str:
        words = _lemmatize(sentence)
        return " ".join(w for w in words if w not in _STOPWORDS)

except ImportError:
    print(
        "[warn] pywsd not installed — stop-word removal uses simple tokenisation. "
        "Install with: pip install pywsd",
        file=sys.stderr,
    )

    from nltk.tokenize import word_tokenize as _word_tokenize

    def remove_stop(sentence: str) -> str:  # type: ignore[misc]
        words = _word_tokenize(sentence.lower())
        return " ".join(w for w in words if w not in _STOPWORDS)


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------

def load_predictions(pred_path: str) -> dict:
    """Return {video_id: {qid: pred_text}} regardless of input format."""
    path = Path(pred_path)
    if path.suffix == ".json":
        with open(path) as f:
            return json.load(f)

    # Flat JSONL from infer_nextqa_oe_jsonl.py
    nested: dict = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            vid  = str(rec.get("video_id", ""))
            qid  = str(rec.get("qid", rec.get("id", "")))
            pred = rec.get("pred_text") or ""
            nested.setdefault(vid, {})[qid] = pred
    return nested


# ---------------------------------------------------------------------------
# Evaluation logic (mirrors NExT-OE eval_oe.py)
# ---------------------------------------------------------------------------
TYPE_GROUPS = {
    "C": ["CW", "CH"],
    "T": ["TN", "TC"],   # TP is merged into TN (same as NExT-OE)
    "D": ["DB", "DC", "DL", "DO"],
}
ALL_TYPES = ["CW", "CH", "TN", "TC", "DB", "DC", "DL", "DO"]


def evaluate(preds: dict, ref_csv_path: str, add_ref_path: str | None = None) -> dict:
    refer = pd.read_csv(ref_csv_path)

    add_ref: dict | None = None
    if add_ref_path and osp.exists(add_ref_path):
        with open(add_ref_path) as f:
            add_ref = json.load(f)

    wups0:  dict = {t: 0.0 for t in ALL_TYPES}
    wups9:  dict = {t: 0.0 for t in ALL_TYPES}
    counts: dict = {t: 0   for t in ALL_TYPES}

    skipped = 0
    for _, row in refer.iterrows():
        video  = str(row["video"])
        qid    = str(row["qid"])
        ans    = str(row["answer"])
        qtype  = str(row["type"])
        if qtype == "TP":
            qtype = "TN"   # merge TP → TN as in NExT-OE

        if qtype not in ALL_TYPES:
            continue

        if video not in preds or qid not in preds[video]:
            skipped += 1
            continue

        pred_raw = preds[video][qid]
        gt_ans   = remove_stop(ans)
        pred_ans = remove_stop(pred_raw)

        extra_gt: str | None = None
        if add_ref and video in add_ref and qid in add_ref[video]:
            extra_gt = remove_stop(add_ref[video][qid])

        if qtype in ("DC", "DB"):
            # Binary / count: exact match after stop-word removal
            if extra_gt is not None:
                score = 1.0 if pred_ans in (gt_ans, extra_gt) else 0.0
            else:
                score = 1.0 if pred_ans == gt_ans else 0.0
            cur0 = cur9 = score
        else:
            if extra_gt is not None:
                cur0 = max(get_wups(pred_ans, gt_ans, 0),   get_wups(pred_ans, extra_gt, 0))
                cur9 = max(get_wups(pred_ans, gt_ans, 0.9), get_wups(pred_ans, extra_gt, 0.9))
            else:
                cur0 = get_wups(pred_ans, gt_ans, 0)
                cur9 = get_wups(pred_ans, gt_ans, 0.9)

        wups0[qtype]  += cur0
        wups9[qtype]  += cur9
        counts[qtype] += 1

    if skipped:
        print(f"[warn] {skipped} reference rows had no prediction.", file=sys.stderr)

    # Per-type averages (×100)
    per_type_0: dict = {}
    per_type_9: dict = {}
    for t in ALL_TYPES:
        n = counts[t]
        per_type_0[t] = (wups0[t] / n * 100) if n else 0.0
        per_type_9[t] = (wups9[t] / n * 100) if n else 0.0

    def group_avg(group_types):
        total_score = sum(wups0[t] for t in group_types)
        total_count = sum(counts[t] for t in group_types)
        return (total_score / total_count * 100) if total_count else 0.0

    wups0_C = group_avg(TYPE_GROUPS["C"])
    wups0_T = group_avg(TYPE_GROUPS["T"])
    wups0_D = group_avg(TYPE_GROUPS["D"])

    total_score  = sum(wups0.values())
    total_count  = sum(counts.values())
    wups0_all    = (total_score  / total_count * 100) if total_count else 0.0
    wups9_all    = (sum(wups9.values()) / total_count * 100) if total_count else 0.0

    return {
        "per_type_wups0": per_type_0,
        "per_type_wups9": per_type_9,
        "group_wups0":    {"C": wups0_C, "T": wups0_T, "D": wups0_D},
        "wups0_all":      wups0_all,
        "wups9_all":      wups9_all,
        "counts":         counts,
        "skipped":        skipped,
    }


def print_table(metrics: dict) -> None:
    p0 = metrics["per_type_wups0"]
    g0 = metrics["group_wups0"]
    print("CW\tCH\tWUPS_C\tTPN\tTC\tWUPS_T\tDB\tDC\tDL\tDO\tWUPS_D\tWUPS")
    print(
        "{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t"
        "{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(
            p0["CW"], p0["CH"], g0["C"],
            p0["TN"], p0["TC"], g0["T"],
            p0["DB"], p0["DC"], p0["DL"], p0["DO"], g0["D"],
            metrics["wups0_all"],
        )
    )
    print(f"\nWUPS@0.9 (overall): {metrics['wups9_all']:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Score NExT-QA OE predictions with WUPS metrics."
    )
    parser.add_argument(
        "--pred-path", required=True,
        help="Flat JSONL from infer_nextqa_oe_jsonl.py, "
             "or .nested.json in {video_id:{qid:pred}} format.",
    )
    parser.add_argument(
        "--ref-csv", required=True,
        help="OE reference CSV (e.g. dataset/nextqa/openend/val.csv). "
             "Must have columns: video, qid, answer, type.",
    )
    parser.add_argument(
        "--add-ref", default=None,
        help="(optional) Path to additional-reference JSON "
             "(NExT-OE add_reference_answer_*.json). "
             "Only exists for the test split.",
    )
    parser.add_argument(
        "--output-path", default=None,
        help="Where to write the metrics JSON. "
             "Defaults to <pred-path>.wups.json.",
    )
    args = parser.parse_args()

    preds   = load_predictions(args.pred_path)
    metrics = evaluate(preds, args.ref_csv, args.add_ref)

    print_table(metrics)

    out_path = (
        Path(args.output_path)
        if args.output_path
        else Path(args.pred_path).with_suffix("").with_suffix(".wups.json")
    )
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\nWrote metrics to {out_path}")


if __name__ == "__main__":
    main()
