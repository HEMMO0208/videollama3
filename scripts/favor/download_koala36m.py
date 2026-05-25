#!/usr/bin/env python3
"""
Download Koala36M clips for FAVOR-Train from YouTube using yt-dlp.

Usage:
    python scripts/favor/download_koala36m.py \
        --sft_json dataset/favor/sft.json \
        --out_dir /home/hmkang/project/videollama3/FAVOR/videos/FAVOR-Train \
        [--workers 4] [--log_file logs/download_koala36m.log]
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sft_json", default="dataset/favor/sft.json")
    p.add_argument("--out_dir", default="/home/hmkang/project/videollama3/FAVOR/videos/FAVOR-Train")
    p.add_argument("--workers", type=int, default=4,
                   help="병렬 다운로드 수 (YouTube rate limit 주의, 4~8 권장)")
    p.add_argument("--log_file", default="logs/download_koala36m.log")
    p.add_argument("--retry", type=int, default=2, help="실패 시 재시도 횟수")
    return p.parse_args()


def download_one(entry: dict, out_dir: str, retry: int) -> tuple[str, bool, str]:
    """
    Returns (video_name, success, reason)
    """
    video_name = entry["video_name"]
    out_path = os.path.join(out_dir, video_name)

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return video_name, True, "already_exists"

    url = entry["url"]
    start = float(entry["start"])
    end = float(entry["end"])
    duration = end - start

    # yt-dlp: 해당 구간만 다운로드
    # --download-sections "*START-END" 은 초 단위 구간 지정
    # -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4" 로 mp4 우선
    # --force-keyframes-at-cuts 로 정확한 cut
    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "--download-sections", f"*{start}-{end}",
        "--force-keyframes-at-cuts",
        "-f", "bestvideo[ext=mp4][height<=480]+bestaudio[ext=m4a]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4/best",
        "--merge-output-format", "mp4",
        "-o", out_path,
        url,
    ]

    for attempt in range(retry + 1):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,  # 2분 타임아웃
            )
            if result.returncode == 0 and os.path.exists(out_path):
                return video_name, True, "downloaded"
            # 실패 이유 추출
            err = (result.stderr or result.stdout or "unknown error").strip().splitlines()
            err_msg = err[-1] if err else "unknown"
            if attempt < retry:
                time.sleep(2 ** attempt)  # 지수 백오프
        except subprocess.TimeoutExpired:
            err_msg = "timeout"
            if attempt < retry:
                time.sleep(2)

    return video_name, False, err_msg


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(args.log_file, encoding="utf-8"),
        ],
    )
    log = logging.getLogger()

    # Koala36M 항목만 추출
    with open(args.sft_json) as f:
        sft = json.load(f)
    entries = [e for e in sft if e["subset"] == "Koala36M"]
    log.info(f"Koala36M 총 {len(entries)}개 항목")

    # 이미 존재하는 파일 집계
    already = sum(
        1 for e in entries
        if os.path.exists(os.path.join(args.out_dir, e["video_name"]))
        and os.path.getsize(os.path.join(args.out_dir, e["video_name"])) > 0
    )
    todo = len(entries) - already
    log.info(f"이미 완료: {already}개 | 다운로드 필요: {todo}개")

    if todo == 0:
        log.info("모두 완료됨. 종료.")
        return

    # 병렬 다운로드
    success_count = already
    fail_count = 0
    fail_log = []
    done = already

    log.info(f"workers={args.workers} 로 다운로드 시작...")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one, e, args.out_dir, args.retry): e
            for e in entries
        }
        for fut in as_completed(futures):
            video_name, ok, reason = fut.result()
            done += 1
            if ok:
                if reason != "already_exists":
                    success_count += 1
                    if success_count % 100 == 0 or done % 500 == 0:
                        log.info(f"[{done}/{len(entries)}] ✓ {video_name}")
            else:
                fail_count += 1
                fail_log.append({"video_name": video_name, "reason": reason})
                log.warning(f"[{done}/{len(entries)}] ✗ {video_name}: {reason}")

            # 100개마다 요약
            if done % 100 == 0:
                log.info(f"  진행: {done}/{len(entries)} | 성공: {success_count} | 실패: {fail_count}")

    log.info(f"\n=== 완료 ===")
    log.info(f"성공: {success_count} / {len(entries)}")
    log.info(f"실패: {fail_count}")

    if fail_log:
        fail_path = args.log_file.replace(".log", "_failed.json")
        with open(fail_path, "w") as f:
            json.dump(fail_log, f, ensure_ascii=False, indent=2)
        log.info(f"실패 목록: {fail_path}")


if __name__ == "__main__":
    main()
