#!/usr/bin/env python3
"""
Download Koala36M clips for FAVOR-Train from YouTube.

Strategy: yt-dlp로 직접 스트림 URL만 얻고, ffmpeg로 직접 seek+trim.
yt-dlp가 ffmpeg를 내부 호출할 때 SIGSEGV가 나는 경우의 우회책.

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

# 영구 실패 키워드 (private, deleted 등 → 재시도 의미 없음)
PERMANENT_ERRORS = (
    "Private video",
    "This video is not available",
    "Video unavailable",
    "has been removed",
    "This video has been",
    "Sign in to confirm your age",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sft_json", default="dataset/favor/sft.json")
    p.add_argument("--out_dir",
                   default="/home/hmkang/project/videollama3/FAVOR/videos/FAVOR-Train")
    p.add_argument("--workers", type=int, default=4,
                   help="병렬 다운로드 수 (YouTube rate limit 주의, 4~8 권장)")
    p.add_argument("--log_file", default="logs/download_koala36m.log")
    p.add_argument("--retry", type=int, default=2, help="실패 시 재시도 횟수")
    return p.parse_args()


def get_stream_url(youtube_url: str, timeout: int = 30) -> str | None:
    """
    yt-dlp -g 로 직접 스트림 URL 획득 (ffmpeg 호출 없음).
    단일 pre-merged mp4 스트림 우선 → ffmpeg merge 불필요.
    """
    cmd = [
        "yt-dlp", "-g",
        "--quiet", "--no-warnings",
        # pre-merged 단일 스트림 우선 (YouTube format 18=360p, 22=720p)
        # merge 불필요하므로 ffmpeg 호출이 없어짐
        "-f", "18/22/best[ext=mp4][height<=480]/best[ext=mp4]/best",
        "--no-playlist",
        youtube_url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0 or not result.stdout.strip():
        output = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(output or "yt-dlp returned no URL")
    # 여러 줄일 경우 첫 번째 URL (video stream)
    return result.stdout.strip().splitlines()[0]


def ffmpeg_trim(stream_url: str, start: float, end: float,
                out_path: str, timeout: int = 120) -> None:
    """
    ffmpeg로 스트림 URL에서 직접 seek+trim.
    -ss를 -i 앞에 배치 → keyframe 기반 빠른 seek.
    -c copy → re-encoding 없이 stream copy.
    """
    duration = end - start
    tmp_path = out_path + ".part.mp4"  # .mp4 필수: ffmpeg가 확장자로 muxer 결정
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", stream_url,
        "-t", str(duration),
        "-c", "copy",
        "-avoid_negative_ts", "1",
        tmp_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        # 실패 시 임시 파일 정리
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(
            f"ffmpeg exit code {result.returncode}: "
            + (result.stderr or "").strip().splitlines()[-1:][0] if (result.stderr or "").strip() else ""
        )
    os.rename(tmp_path, out_path)


def download_one(entry: dict, out_dir: str, retry: int) -> tuple[str, bool, str]:
    """Returns (video_name, success, reason)"""
    video_name = entry["video_name"]
    out_path = os.path.join(out_dir, video_name)

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return video_name, True, "already_exists"

    url = entry["url"]
    start = float(entry["start"])
    end = float(entry["end"])

    err_msg = "unknown"
    for attempt in range(retry + 1):
        try:
            # Step 1: yt-dlp로 스트림 URL 획득 (ffmpeg 호출 없음)
            stream_url = get_stream_url(url, timeout=30)

            # Step 2: ffmpeg로 직접 trim
            ffmpeg_trim(stream_url, start, end, out_path, timeout=120)

            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                return video_name, True, "downloaded"
            err_msg = "output file missing after ffmpeg"

        except RuntimeError as e:
            err_msg = str(e)
            if any(kw in err_msg for kw in PERMANENT_ERRORS):
                return video_name, False, err_msg  # 재시도 의미 없음
        except subprocess.TimeoutExpired:
            err_msg = "timeout"
        except Exception as e:
            err_msg = str(e)

        if attempt < retry:
            time.sleep(2 ** attempt)  # 지수 백오프

    return video_name, False, err_msg


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    log_dir = os.path.dirname(args.log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

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

    with open(args.sft_json) as f:
        sft = json.load(f)
    entries = [e for e in sft if e["subset"] == "Koala36M"]
    log.info(f"Koala36M 총 {len(entries)}개 항목")

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

    success_count = already
    fail_count = 0
    fail_log = []
    done = already

    log.info(f"workers={args.workers} 로 다운로드 시작 (yt-dlp URL 획득 → ffmpeg trim)...")

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
