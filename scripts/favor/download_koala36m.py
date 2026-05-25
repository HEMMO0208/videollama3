#!/usr/bin/env python3
"""
Download Koala36M clips for FAVOR-Train from YouTube.

Strategy: yt-dlp로 스트림 URL 획득 → PyAV(Python libav 바인딩)로 trim.
ffmpeg 서브프로세스를 전혀 사용하지 않으므로 서버 ffmpeg SIGSEGV 완전 우회.

Usage:
    pip install av yt-dlp
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

try:
    import av
except ImportError:
    import subprocess as _sp
    _sp.check_call([sys.executable, "-m", "pip", "install", "av", "-q"])
    import av

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


def get_stream_url(youtube_url: str, timeout: int = 30) -> str:
    """yt-dlp -g 로 직접 스트림 URL 획득 (ffmpeg 호출 없음)."""
    cmd = [
        "yt-dlp", "-g",
        "--quiet", "--no-warnings",
        "-f", "18/22/best[ext=mp4][height<=480]/best[ext=mp4]/best",
        "--no-playlist",
        youtube_url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0 or not result.stdout.strip():
        output = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(output or "yt-dlp returned no URL")
    return result.stdout.strip().splitlines()[0]


def pyav_trim(stream_url: str, start: float, end: float, out_path: str) -> None:
    """
    PyAV(libav Python 바인딩)로 URL에서 직접 seek+decode+re-encode.
    ffmpeg 서브프로세스 불사용 → 서버 ffmpeg 크래시 완전 우회.
    """
    tmp_path = out_path + ".part.mp4"
    try:
        with av.open(stream_url) as inp:
            v_in = inp.streams.video[0] if inp.streams.video else None
            a_in = inp.streams.audio[0] if inp.streams.audio else None
            if v_in is None:
                raise RuntimeError("no video stream found")

            # seek: av.time_base = Fraction(1, 1000000) → microseconds로 변환
            inp.seek(int(start * 1_000_000), any_frame=False)

            # mp4 muxer는 타임스탬프 interleaving에 엄격 → EINVAL 발생 가능
            # matroska(mkv)는 더 관대하고 ffprobe/ffmpeg 모두 자동 감지함
            with av.open(tmp_path, "w", format="matroska") as out:
                v_out = out.add_stream("libx264", rate=v_in.average_rate)
                v_out.width   = v_in.codec_context.width  // 2 * 2  # libx264: 짝수 필수
                v_out.height  = v_in.codec_context.height // 2 * 2
                v_out.pix_fmt = "yuv420p"
                v_out.options = {"preset": "ultrafast", "crf": "28"}

                a_out = None
                if a_in:
                    a_out = out.add_stream("aac")
                    a_out.sample_rate = a_in.codec_context.sample_rate
                    a_out.layout      = a_in.codec_context.layout

                streams = (v_in,) + ((a_in,) if a_in else ())
                for frame in inp.decode(*streams):
                    t = frame.time
                    if t is None or t < start - 0.1:
                        continue
                    if t > end + 0.1:
                        break
                    frame.pts = None  # encoder가 timestamp 자동 관리 (0 기준 reset)
                    if isinstance(frame, av.VideoFrame):
                        for pkt in v_out.encode(frame):
                            out.mux(pkt)
                    elif isinstance(frame, av.AudioFrame) and a_out:
                        for pkt in a_out.encode(frame):
                            out.mux(pkt)

                # flush
                for pkt in v_out.encode(None):
                    out.mux(pkt)
                if a_out:
                    for pkt in a_out.encode(None):
                        out.mux(pkt)

        os.rename(tmp_path, out_path)

    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


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

            # Step 2: PyAV로 trim → ffmpeg subprocess 없음
            pyav_trim(stream_url, start, end, out_path)

            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                return video_name, True, "downloaded"
            err_msg = "output file missing after pyav"

        except RuntimeError as e:
            err_msg = str(e)
            if any(kw in err_msg for kw in PERMANENT_ERRORS):
                return video_name, False, err_msg
        except subprocess.TimeoutExpired:
            err_msg = "yt-dlp timeout"
        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"

        if attempt < retry:
            time.sleep(2 ** attempt)

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

    log.info(f"workers={args.workers} 로 다운로드 시작 (yt-dlp URL → PyAV trim)...")

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
