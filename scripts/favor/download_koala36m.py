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

# ── 에러 분류 ──────────────────────────────────────────────────────────────────

# 영구 실패: 비디오 자체 문제 (삭제, 비공개, 지역차단 등) → 재시도 의미 없음
PERMANENT_ERRORS = (
    "Private video",
    "This video is not available",
    "Video unavailable",
    "has been removed",
    "This video has been",
    "Sign in to confirm your age",
    "Only images are available",  # storyboard만 존재 = 삭제/지역차단
    # NOTE: "Requested format is not available"는 n challenge 실패 시도 발생하므로
    #       여기서 제외 — 아래 N_CHALLENGE_KEYWORDS로 별도 처리
)

# n challenge 실패 키워드 (Node.js / EJS 문제)
N_CHALLENGE_KEYWORDS = (
    "n challenge solving failed",
    "n challenge",
)

# 쿠키 만료 / 봇 감지 키워드
COOKIE_KEYWORDS = (
    "Sign in to confirm you're not a bot",
    "Sign in to confirm",
    "Use --cookies",
    "cookies-from-browser",
)

# preflight 테스트용 항상 존재하는 공개 영상 (Rick Astley)
_PREFLIGHT_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sft_json", default="dataset/favor/sft.json")
    p.add_argument("--out_dir",
                   default="/home/hmkang/project/videollama3/FAVOR/videos/FAVOR-Train")
    p.add_argument("--workers", type=int, default=4,
                   help="병렬 다운로드 수 (YouTube rate limit 주의, 4~8 권장)")
    p.add_argument("--log_file", default="logs/download_koala36m.log")
    p.add_argument("--retry", type=int, default=2, help="실패 시 재시도 횟수")
    p.add_argument("--cookies", default="/home/hmkang/project/videollama3/yt_cookies.txt",
                   help="yt-dlp에 전달할 쿠키 파일 경로 (없으면 무시)")
    p.add_argument("--node_bin", default=None,
                   help="node 바이너리 경로 (없으면 PATH에서 자동 탐색)")
    p.add_argument("--skip_preflight", action="store_true",
                   help="시작 시 node/쿠키 동작 확인 건너뜀")
    return p.parse_args()


def _classify_yt_error(output: str) -> str:
    """yt-dlp 에러 출력을 사람이 읽기 쉬운 메시지로 변환."""
    if any(kw in output for kw in COOKIE_KEYWORDS):
        return f"[쿠키 만료/봇 감지] 쿠키 파일을 갱신하세요 — {output[:120]}"
    if any(kw in output for kw in N_CHALLENGE_KEYWORDS):
        return f"[n challenge 실패] Node.js 경로·버전을 확인하세요 — {output[:120]}"
    return output


def _build_ytdlp_cmd(tmp_path: str, cookies: str | None,
                     node_bin: str, extra_flags: list | None = None) -> list:
    """공통 yt-dlp 커맨드 생성."""
    cmd = [
        "yt-dlp",
        "--quiet",        # 다운로드 진행 막대 억제
        # --no-warnings 제거: 경고(n challenge, 쿠키 등)를 stderr에서 캡처해야 함
        "--js-runtimes", f"node:{node_bin}",
        "--remote-components", "ejs:github",
        "-f", ("18/22"
               "/best[ext=mp4][vcodec!=none][acodec!=none][height<=480]"
               "/best[ext=mp4][vcodec!=none][acodec!=none]"
               "/best[vcodec!=none][acodec!=none]"
               "/bestvideo[ext=mp4][height<=480]"
               "/bestvideo[ext=mp4]"),
        "--no-playlist",
    ]
    if extra_flags:
        cmd.extend(extra_flags)
    if tmp_path:
        cmd += ["--no-part", "-o", tmp_path]
    if cookies and os.path.exists(cookies):
        cmd += ["--cookies", cookies]
    return cmd


def preflight_check(node_bin: str, cookies: str | None, log) -> bool:
    """
    시작 전 Node.js / EJS / 쿠키 동작 여부를 확인.
    실패 시 원인(n challenge vs 쿠키)을 명확히 로그.
    Returns True if OK, False if problems detected.
    """
    log.info(f"[preflight] Node.js / 쿠키 동작 확인 중 ({_PREFLIGHT_URL}) ...")
    cmd = _build_ytdlp_cmd(tmp_path="", cookies=cookies, node_bin=node_bin,
                           extra_flags=["--simulate"])
    # tmp_path="" → -o "" 가 들어가면 안 되므로 직접 제거
    cmd = [x for x in cmd if x not in ("--no-part", "-o", "")]
    cmd.append(_PREFLIGHT_URL)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        log.warning("[preflight] 타임아웃 — 네트워크 상태 불량. 다운로드는 계속 시도합니다.")
        return True  # timeout은 결정적 실패가 아님

    output = (result.stderr + "\n" + result.stdout).strip()

    if result.returncode == 0:
        log.info("[preflight] ✓ Node.js n challenge + 쿠키 정상 동작")
        return True

    if any(kw in output for kw in COOKIE_KEYWORDS):
        log.error("=" * 60)
        log.error("[preflight] ✗ 쿠키 만료 또는 봇 감지!")
        log.error("  → yt_cookies.txt 를 브라우저에서 다시 내보내야 합니다.")
        log.error(f"  쿠키 경로: {cookies}")
        log.error("  방법: 브라우저 확장 'Get cookies.txt LOCALLY' 사용")
        log.error("=" * 60)
        return False

    if any(kw in output for kw in N_CHALLENGE_KEYWORDS):
        log.error("=" * 60)
        log.error("[preflight] ✗ YouTube n challenge 실패!")
        log.error("  → Node.js 가 동작하지 않거나 EJS 스크립트를 다운받지 못했습니다.")
        log.error(f"  node_bin: {node_bin}")
        log.error(f"  확인: {node_bin} --version")
        log.error("=" * 60)
        return False

    # 그 외 (지역차단, 비공개 등 테스트 URL 문제) — 비결정적, 경고만
    log.warning(f"[preflight] 경고 (비결정적 실패, 무시): {output[:200]}")
    return True


def download_full_video(youtube_url: str, tmp_path: str,
                        cookies: str | None = None, timeout: int = 300,
                        node_bin: str | None = None) -> None:
    """
    yt-dlp로 영상 전체를 로컬에 저장 (ffmpeg 호출 없음).
    에러 메시지에 원인(쿠키/n challenge/포맷 없음)을 명시.
    """
    import shutil
    if node_bin is None:
        node_bin = shutil.which("node") or "node"

    cmd = _build_ytdlp_cmd(tmp_path, cookies, node_bin)
    cmd.append(youtube_url)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raw = (result.stderr + "\n" + result.stdout).strip()
        raise RuntimeError(_classify_yt_error(raw) if raw else "yt-dlp download failed")

    if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
        raise RuntimeError("yt-dlp output file missing or empty")


def pyav_trim(src_path: str, start: float, end: float, out_path: str) -> None:
    """
    PyAV(libav Python 바인딩)로 로컬 파일에서 seek+decode+re-encode.
    ffmpeg 서브프로세스 불사용 → 서버 ffmpeg 크래시 완전 우회.
    로컬 파일 사용으로 moov-at-end 문제도 해결.
    """
    tmp_path = out_path + ".part.mp4"
    try:
        with av.open(src_path) as inp:
            v_in = inp.streams.video[0] if inp.streams.video else None
            a_in = inp.streams.audio[0] if inp.streams.audio else None
            if v_in is None:
                raise RuntimeError("no video stream found")

            inp.seek(int(start * 1_000_000), any_frame=False)

            with av.open(tmp_path, "w", format="matroska") as out:
                v_out = out.add_stream("libx264", rate=v_in.average_rate)
                v_out.width   = v_in.codec_context.width  // 2 * 2
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
                    frame.pts = None
                    if isinstance(frame, av.VideoFrame):
                        for pkt in v_out.encode(frame):
                            out.mux(pkt)
                    elif isinstance(frame, av.AudioFrame) and a_out:
                        for pkt in a_out.encode(frame):
                            out.mux(pkt)

                for pkt in v_out.encode(None):
                    out.mux(pkt)
                if a_out:
                    for pkt in a_out.encode(None):
                        out.mux(pkt)

        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
            raise RuntimeError("no frames written (seek overshot or empty segment)")
        os.rename(tmp_path, out_path)

    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def download_one(entry: dict, out_dir: str, retry: int,
                 cookies: str | None = None,
                 node_bin: str | None = None) -> tuple[str, bool, str]:
    """Returns (video_name, success, reason)"""
    video_name = entry["video_name"]
    out_path = os.path.join(out_dir, video_name)

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return video_name, True, "already_exists"

    url = entry["url"]
    start = float(entry["start"])
    end = float(entry["end"])

    tmp_full = f"/tmp/{video_name}.full.mp4"

    err_msg = "unknown"
    for attempt in range(retry + 1):
        try:
            download_full_video(url, tmp_full, cookies=cookies, timeout=300,
                                node_bin=node_bin)
            pyav_trim(tmp_full, start, end, out_path)

            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                return video_name, True, "downloaded"
            err_msg = "output file missing after pyav"

        except RuntimeError as e:
            err_msg = str(e)
            if any(kw in err_msg for kw in PERMANENT_ERRORS):
                return video_name, False, err_msg
            # 쿠키 에러도 재시도해도 의미 없음 → 즉시 실패
            if err_msg.startswith("[쿠키 만료/봇 감지]"):
                return video_name, False, err_msg
        except subprocess.TimeoutExpired:
            err_msg = "yt-dlp timeout"
        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
        finally:
            if os.path.exists(tmp_full):
                os.remove(tmp_full)

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

    # ── node 바이너리 탐색 ────────────────────────────────────────────────────
    import shutil
    node_bin = args.node_bin or shutil.which("node") or "node"
    if shutil.which(node_bin):
        log.info(f"Node.js: {node_bin} ({subprocess.check_output([node_bin, '--version'], text=True).strip()})")
    else:
        log.error(f"[경고] Node.js 미발견: '{node_bin}' — n challenge 실패 예상")

    # ── 쿠키 파일 확인 ────────────────────────────────────────────────────────
    cookies = args.cookies if os.path.exists(args.cookies) else None
    if cookies:
        mtime = os.path.getmtime(cookies)
        age_h = (time.time() - mtime) / 3600
        log.info(f"쿠키 파일: {cookies} (수정 {age_h:.1f}시간 전)")
        if age_h > 48:
            log.warning(f"[쿠키 주의] 파일이 {age_h:.0f}시간 전에 생성됨 — 만료됐을 수 있음")
    else:
        log.error(f"[쿠키 없음] {args.cookies} 미존재 — 봇 감지로 대부분 실패 예상")

    # ── preflight check ───────────────────────────────────────────────────────
    if not args.skip_preflight:
        ok = preflight_check(node_bin, cookies, log)
        if not ok:
            log.error("preflight 실패. 문제를 해결한 뒤 재실행하세요.")
            log.error("(건너뛰려면 --skip_preflight 플래그 사용)")
            sys.exit(1)

    log.info(f"workers={args.workers} 로 다운로드 시작 (yt-dlp → PyAV trim)...")

    success_count = already
    fail_count = 0
    fail_log = []
    done = already

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one, e, args.out_dir, args.retry, cookies,
                        node_bin): e
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
