#!/usr/bin/env python3
"""Convert NExT-QA Open-Ended CSV to VideoLLaMA3 SFT JSONL.

OE format differs from MCQ:
  - No a0~a4 options columns
  - Answer is free-form text (not an option index)
  - Instruction asks for a short phrase/word rather than a letter
"""
import argparse
import json
import random
from pathlib import Path

import pandas as pd


def _video_path(video_id, video_map):
    key = str(video_id)
    mapped = video_map.get(key, key)
    mapped = str(mapped)
    return mapped if Path(mapped).suffix else f"{mapped}.mp4"


def build_records(csv_path, map_json_path):
    records = pd.read_csv(csv_path)
    with open(map_json_path, "r", encoding="utf-8") as f:
        video_map = json.load(f)

    for row in records.itertuples(index=False):
        row_dict = row._asdict()
        question = str(row_dict["question"]).strip()
        answer   = str(row_dict["answer"]).strip()

        prompt = (
            "<video>\n"
            f"Question: {question}\n"
            "Answer the question with a short phrase or word."
        )

        qid = row_dict.get("qid", 0)
        yield {
            "id": str(qid),
            "video": [_video_path(row_dict["video"], video_map)],
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt",   "value": answer},
            ],
            "metadata": {
                "source":        "nextqa_oe",
                "video_id":      str(row_dict["video"]),
                "question_type": str(row_dict.get("type", "")),
                "answer_text":   answer,
            },
        }


def main():
    parser = argparse.ArgumentParser(
        description="Convert NExT-QA OE CSV to VideoLLaMA3 SFT JSONL."
    )
    parser.add_argument("--csv-path",     required=True)
    parser.add_argument("--map-json-path", required=True)
    parser.add_argument("--output-path",  required=True)
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="If set, randomly sample this many records (for val_mini splits).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_records = list(build_records(args.csv_path, args.map_json_path))

    if args.sample is not None and args.sample < len(all_records):
        rng = random.Random(args.seed)
        all_records = rng.sample(all_records, args.sample)

    with output_path.open("w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_records)} records to {output_path}")


if __name__ == "__main__":
    main()
