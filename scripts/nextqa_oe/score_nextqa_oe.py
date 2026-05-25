#!/usr/bin/env python3
"""Score NExT-QA Open-Ended predictions using WUPS metrics.

Wraps NExT-OE's eval_oe.py logic with flexible path arguments,
so it can be run from anywhere without depending on NExT-OE's
working directory.

Input:
  --pred-path   flat JSONL produced by infer_nextqa_oe_jsonl.py
                OR .nested.json in {video_id: {qid: pred_text}} format

  --ref-csv     OE reference CSV (e.g. dataset/nextqa/openend/val.csv)
                Columns: video, qid, answer, type

  --add-ref     (optional) JSON with additional reference answers
                {video_id: {qid: text}} — e.g. NExT-OE's
                add_reference_answer_val.json (absent for val split)

Output:
  .metrics.json alongside --pred-path with per-type WUPS@0 / WUPS@0.9
  Tab-separated table printed to stdout (matches NExT-OE format)

Dependencies (same as NExT-OE):
  pip install nltk pywsd
  python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt')"
"""
import argparse
import json
import os
import os.path as osp
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Locate NExT-OE and import its metric helpers
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
NEXTOE_DIR = (SCRIPT_DIR / "../../../NExT-OE").resolve()

if NEXTOE_DIR.is_dir():
    sys.path.insert(0, str(NEXTOE_DIR))
    try:
        from metrics import get_wups
        from pywsd.utils import lemmatize_sentence

        def _load_stopwords():
            sw_path = NEXTOE_DIR / "stopwords.txt"
            with open(sw_path) as f:
                return set(line.strip() for line in f if line.strip())

        _STOPWORDS = _load_stopwords()

        def remove_stop(sentence):
            words = lemmatize_sentence(sentence)
            return " ".join(w for w in words if w not in _STOPWORDS)

    except ImportError as e:
        print(f"[warn] Could not import from NExT-OE ({e}); falling back to basic WUPS.")
        from metrics import get_wups  # still try metrics.py

        def remove_stop(sentence):  # type: ignore[misc]
            return sentence.lower().strip()

else:
    print(f"[warn] NExT-OE not found at {NEXTOE_DIR}. Attempting local import.")
    try:
        from metrics import get_wups  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "Cannot find NExT-OE metrics.py. "
            f"Expected at {NEXTOE_DIR} or on PYTHONPATH."
        )

    def remove_stop(sentence):  # type: ignore[misc]
        return sentence.lower().strip()


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------

def load_predictions(pred_path: str) -> dict[str, dict[str, str]]:
    """Return {video_id: {qid: pred_text}} regardless of input format."""
    path = Path(pred_path)
    if path.suffix == ".json":
        with open(path) as f:
            return json.load(f)

    # Flat JSONL from infer_nextqa_oe_jsonl.py
    nested: dict[str, dict[str, str]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            vid = str(rec.get("video_id", ""))
            qid = str(rec.get("qid", rec.get("id", "")))
            pred = rec.get("pred_text") or ""
            nested.setdefault(vid, {})[qid] = pred
    return nested


# ---------------------------------------------------------------------------
# Evaluation logic (mirrors NExT-OE eval_oe.py)
# ---------------------------------------------------------------------------
TYPE_GROUPS = {
    "C": ["CW", "CH"],
    "T": ["TN", "TC"],  # TP is merged into TN
    "D": ["DB", "DC", "DL", "DO"],
}
ALL_TYPES = ["CW", "CH", "TN", "TC", "DB", "DC", "DL", "DO"]


def evaluate(preds, ref_csv_path, add_ref_path=None):
    refer = pd.read_csv(ref_csv_path)

    add_ref: dict | None = None
    if add_ref_path and osp.exists(add_ref_path):
        with open(add_ref_path) as f:
            add_ref = json.load(f)

    # Accumulate per-type scores
    wups0: dict[str, float] = {t: 0.0 for t in ALL_TYPES}
    wups9: dict[str, float] = {t: 0.0 for t in ALL_TYPES}
    counts: dict[str, int] = {t: 0 for t in ALL_TYPES}

    skipped = 0
    for _, row in refer.iterrows():
        video = str(row["video"])
        qid   = str(row["qid"])
        ans   = str(row["answer"])
        qtype = str(row["type"])
        if qtype == "TP":
            qtype = "TN"  # merge TP → TN as in NExT-OE

        if qtype not in ALL_TYPES:
            continue

        if video not in preds or qid not in preds[video]:
            skipped += 1
            continue

        pred_raw = preds[video][qid]
        gt_ans   = remove_stop(ans)
        pred_ans = remove_stop(pred_raw)

        # Additional reference (only for test split in NExT-OE)
        extra_gt = None
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
        print(f"[warn] {skipped} reference rows had no prediction.")

    # Per-type averages (×100)
    per_type_0: dict[str, float] = {}
    per_type_9: dict[str, float] = {}
    for t in ALL_TYPES:
        n = counts[t]
        per_type_0[t] = (wups0[t] / n * 100) if n else 0.0
        per_type_9[t] = (wups9[t] / n * 100) if n else 0.0

    # Group averages
    def group_avg(group_types, score_dict, count_dict):
        total_score = sum(score_dict[t] for t in group_types)
        total_count = sum(count_dict[t] for t in group_types)
        return (total_score / total_count * 100) if total_count else 0.0

    wups0_C = group_avg(TYPE_GROUPS["C"], wups0, counts)
    wups0_T = group_avg(TYPE_GROUPS["T"], wups0, counts)
    wups0_D = group_avg(TYPE_GROUPS["D"], wups0, counts)

    total_score = sum(wups0.values())
    total_count = sum(counts.values())
    wups0_all   = (total_score / total_count * 100) if total_count else 0.0

    total_score9 = sum(wups9.values())
    wups9_all    = (total_score9 / total_count * 100) if total_count else 0.0

    return {
        "per_type_wups0": per_type_0,
        "per_type_wups9": per_type_9,
        "group_wups0": {"C": wups0_C, "T": wups0_T, "D": wups0_D},
        "wups0_all": wups0_all,
        "wups9_all": wups9_all,
        "counts": counts,
        "skipped": skipped,
    }


def print_table(metrics):
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

    preds = load_predictions(args.pred_path)
    metrics = evaluate(preds, args.ref_csv, args.add_ref)

    print_table(metrics)

    out_path = Path(args.output_path) if args.output_path else \
               Path(args.pred_path).with_suffix("").with_suffix(".wups.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\nWrote metrics to {out_path}")


if __name__ == "__main__":
    main()
