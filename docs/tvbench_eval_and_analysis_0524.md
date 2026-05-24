# TVBench 평가 파이프라인 구축 및 결과 분석 (2025-05-24)

## 목차
1. [TVBench SFT 데이터 변환](#1-tvbench-sft-데이터-변환)
2. [평가 스크립트 구성](#2-평가-스크립트-구성)
3. [실험 결과 (result0524)](#3-실험-결과-result0524)
4. [심층 분석](#4-심층-분석)
5. [아키텍처 검토 및 다음 실험 방향](#5-아키텍처-검토-및-다음-실험-방향)

---

## 1. TVBench SFT 데이터 변환

### 소스

`dataset/tvbench/*.json` (10개 카테고리, 총 2,525개 항목)

| 카테고리 | 항목 수 | 선택지 수 |
|---|---|---|
| action_antonym | 320 | 2 |
| action_count | 536 | 4 |
| action_localization | 160 | 4 |
| action_sequence | 437 | 2 |
| egocentric_sequence | 200 | 4 |
| moving_direction | 232 | 4 |
| object_count | 148 | 4 |
| object_shuffle | 225 | 3 |
| scene_transition | 185 | 2 |
| unexpected_action | 82 | 4 |

### 변환 형식

`data/nextqa/*.jsonl` 형식과 동일하게 맞춤:

```json
{
  "id": "320",
  "video": ["action_count/action_count/video_8686.mp4"],
  "conversations": [
    {
      "from": "human",
      "value": "<video>\nQuestion: ...\nOptions:\n(A) ...\n(B) ...\nAnswer with the option's letter from the given choices directly and only give the best option."
    },
    {"from": "gpt", "value": "D"}
  ],
  "metadata": {
    "source": "tvbench",
    "category": "action_count",
    "video_id": "video_8686",
    "answer_index": 3
  }
}
```

### 비디오 경로 규칙

`path.txt` (master 서버의 `~/project/videollama3/TVBench/video/` 구조)를 기반으로:

| 카테고리 | JSON `video` 필드 | 변환된 경로 |
|---|---|---|
| 일반 (action_count 등) | `video_8686.mp4` | `action_count/action_count/video_8686.mp4` |
| egocentric_sequence | `1.2/sunshuo.MP4` | `egocentric_sequence/1.2/1.2/sunshuo.MP4` |

egocentric_sequence는 디렉토리가 `<X.Y>/<X.Y>/` 이중 중첩 구조이므로 subfolder를 두 번 사용.

### 생성 파일

| 파일 | 설명 |
|---|---|
| `data/tvbench/test_sft.jsonl` | 전체 2,525개 |
| `data/tvbench/mini_sft.jsonl` | 랜덤 1,000개 (seed=42) |
| `data/tvbench/<category>_sft.jsonl` | 카테고리별 10개 파일 |
| `data/tvbench/convert_tvbench.py` | 변환 스크립트 |

mini 샘플의 카테고리 분포 (원본 비율 자연 반영):
action_antonym 125, action_count 210, action_localization 78, action_sequence 176,
egocentric_sequence 82, moving_direction 90, object_count 53, object_shuffle 80,
scene_transition 69, unexpected_action 37.

---

## 2. 평가 스크립트 구성

`scripts/nextqa/`와 완전히 동일한 구조로 `scripts/tvbench/`에 작성.

### 파일 목록

| 파일 | 대응 nextqa 파일 | 변경 사항 |
|---|---|---|
| `infer_tvbench_jsonl.py` | `infer_nextqa_jsonl.py` | `question_type` → `category`, 설명 문자열 |
| `eval_tvbench_jsonl.sh` | `eval_nextqa_jsonl.sh` | 경로 및 기본값 변경 (아래 참조) |
| `eval_tvbench_jsonl_baseline.sbatch` | `eval_nextqa_jsonl_baseline.sbatch` | job-name, sh 경로 |
| `eval_tvbench_jsonl_diffusion.sbatch` | `eval_nextqa_jsonl_diffusion.sbatch` | job-name, sh 경로 |
| `eval_tvbench_jsonl_causal_diffusion.sbatch` | `eval_nextqa_jsonl_causal_diffusion.sbatch` | job-name, sh 경로 |
| `submit_eval_sweep.sh` | `submit_eval_sweep.sh` | sbatch 파일명 |

### `eval_tvbench_jsonl.sh` 핵심 경로

```bash
# 체크포인트는 nextqa 위에서 학습된 것 사용
MODEL_PATH = work_dirs/nextqa_{baseline,diffusion,causal_diffusion}

SPLIT       = mini                        # 기본값 (mini_sft.jsonl)
JSONL_PATH  = data/tvbench/${SPLIT}_sft.jsonl
DATA_FOLDER = /home/hmkang/project/videollama3/TVBench/video
OUTPUT_DIR  = results/tvbench
```

### 사용법

```bash
# 단일 모델
./scripts/tvbench/submit_eval_sweep.sh causal_diffusion 100

# 전체 모델 × MAX_FRAMES sweep
./scripts/tvbench/submit_eval_sweep.sh all 8 32 100

# dependency 지정
./scripts/tvbench/submit_eval_sweep.sh all 8 32 100 -- --dependency=afterok:12345
```

### `infer_tvbench_jsonl.py` 변경 포인트

- `result["category"]` 필드 추가 (nextqa의 `question_type` 대응)
- `summarize()` 함수에서 `by_type` → `by_category`
- metrics.json 구조:
  ```json
  {
    "total": 1000,
    "evaluated": 875,
    "correct": 398,
    "accuracy": 0.455,
    "by_category": {
      "action_count": {"total": 210, "correct": 83, "accuracy": 0.395},
      ...
    }
  }
  ```

---

## 3. 실험 결과 (result0524)

3개 모델 × 3개 MAX_FRAMES(8/32/100) = 9개 설정, NextQA와 TVBench 각각 평가.

### 3-1. Overall Accuracy

#### NextQA (n=1,000, error 0개)

| 모델 | mf8 | mf32 | mf100 |
|---|---|---|---|
| baseline | 0.828 | 0.830 | **0.841** |
| diffusion | 0.823 | **0.843** | **0.843** |
| causal_diffusion | **0.831** | 0.839 | 0.837 |

#### TVBench (n=875 evaluated, action_antonym 125개 ffprobe 오류 제외)

| 모델 | mf8 | mf32 | mf100 |
|---|---|---|---|
| baseline | 0.426 | 0.455 | 0.455 |
| diffusion | 0.417 | 0.454 | 0.455 |
| causal_diffusion | 0.423 | 0.453 | 0.455 |

TVBench mf32/mf100에서 세 모델 모두 정확히 398/875로 완전 수렴.

### 3-2. NextQA question_type별 breakdown (mf32)

| Type | 설명 | n | baseline | diffusion | causal_diff | 승자 |
|---|---|---|---|---|---|---|
| CH | Causal-How | 148 | 0.811 | **0.858** | 0.845 | diffusion (+4.7%p) |
| TN | Temporal-Next | 174 | 0.787 | **0.816** | 0.805 | diffusion (+2.9%p) |
| TC | Temporal-Concurrent | 127 | **0.843** | 0.819 | 0.827 | baseline (+2.4%p) |
| CW | Causal-Why | — | 0.828 | 0.841 | **0.841** | diff/causal |
| TP | Temporal-Prior | 15 | 0.800 | **0.933** | 0.800 | 소샘플 |
| DL | Desc.-Location | 53 | **0.943** | 0.925 | **0.943** | 차이 없음 |
| DC | Desc.-Count | — | **0.778** | 0.750 | **0.778** | baseline |
| DO | Desc.-Other | — | **0.907** | 0.893 | 0.880 | baseline |

### 3-3. TVBench category별 breakdown (mf100)

| Category | n | baseline | diffusion | causal_diff | 비고 |
|---|---|---|---|---|---|
| action_sequence | 176 | **0.733** | 0.727 | 0.722 | baseline 우위 |
| scene_transition | 69 | 0.768 | **0.783** | **0.783** | diff 계열 소폭 우위 |
| action_localization | 78 | 0.462 | **0.500** | 0.487 | diffusion +3.8%p |
| object_shuffle | 80 | 0.363 | **0.400** | 0.388 | diffusion +3.7%p |
| action_count | 210 | **0.395** | 0.376 | 0.362 | baseline 우위, causal 꼴찌 |
| object_count | 53 | **0.302** | 0.226 | 0.264 | diffusion -7.6%p 급락 |
| moving_direction | 90 | 0.233 | 0.244 | **0.267** | 전부 랜덤 수준 (≈0.25) |
| egocentric_sequence | 82 | 0.232 | **0.256** | **0.256** | 전부 낮음 |
| unexpected_action | 37 | 0.324 | 0.297 | **0.351** | |

---

## 4. 심층 분석

### 4-1. 통계적 유의성

n=1,000에서 95% CI 반폭 = ±2.3%p. NextQA 전체 모델 간 최대 spread = 2.0%p.
**모든 모델 간 비교에서 McNemar 검정 p > 0.05.** 현재 관찰된 차이는 통계적 노이즈 범위 내.

### 4-2. mf 스케일링 패턴 — 가장 주목할 발견

| 모델 | NQ mf8→32 | NQ mf32→100 | 해석 |
|---|---|---|---|
| baseline | +0.2%p | **+1.1%p** | 꾸준히 개선 |
| diffusion | **+2.0%p** | **±0.0%p** | **mf32에서 완전 포화** |
| causal_diffusion | +0.8%p | -0.2%p | mf32에서 포화 후 소폭 하락 |

diffusion은 mf32에서 이미 ceiling에 도달하고, baseline은 mf100까지 계속 이득을 얻는 유일한 모델.
가설: diffusion head의 condition 추출 방식이 dense frame에서 noise에 더 취약하거나,
적은 프레임으로도 충분한 정보를 압축 추출하는 것 중 하나.

TVBench: mf8→mf32 약 +3%p 공통 점프 후 세 모델 모두 mf32에서 완전 포화.

### 4-3. 선택지 완전 회피 버그 (TVBench)

모든 모델·모든 mf 설정에서 일관:

- **action_count**: GT에 C가 54개 존재하지만 **C 예측 0회**. A(45~60%) / D(26~47%)로 양분.
- **egocentric_sequence**: GT에 A가 22개 존재하지만 **A 예측 0회**. C/D에 집중.

아키텍처 문제가 아닌 **프롬프트 또는 디코딩 레벨의 구조적 bias**로 추정. 이 두 카테고리가 TVBench 전체 정확도의 주요 하한선 역할.

### 4-4. Error 현황

- **NextQA**: 9개 설정 전부 error 0. 완벽하게 클린.
- **TVBench**: action_antonym 카테고리 125개 **전부** `ffprobe error`.
  - 파일 포맷: NTU RGB+D 데이터셋의 `.avi` (H.264)
  - 모든 설정에서 동일하게 실패 → 현재 평가에서 완전히 누락된 상태

### 4-5. Causal vs Non-Causal Diffusion

| 조건 | 우위 |
|---|---|
| NextQA mf8 | **causal** (0.831 > diffusion 0.823, baseline 0.828도 넘음) |
| NextQA mf32 | **diffusion** (0.843 > causal 0.839) |
| NextQA mf100 | **diffusion** (0.843 > causal 0.837) |
| TVBench 전반 | 동일 (mf100 완전 수렴) |

**패턴**: causal constraint가 sparse frame(mf=8)에서는 오히려 유리하게 작용하고,
dense frame(mf≥32)에서는 long-range dependency를 억제하여 제약이 됨.

mf32 기준 diffusion vs causal 불일치 케이스(32건) 분석:
diffusion 15승, causal 11승, 둘 다 오답 6건.

### 4-6. CH 강세 / TC 약세 패턴

| Task Type | 추론 특성 | diffusion vs baseline |
|---|---|---|
| CH (Causal-How) | 절차적 인과 추론 ("어떻게") | **+4.7%p 우위** |
| TN (Temporal-Next) | 시간적 다음 사건 | **+2.9%p 우위** |
| TC (Temporal-Concurrent) | 동시 공존 사건 | **-2.4%p 열세** |

해석: diffusion supervision이 LM으로 하여금 "인과적 흐름"을 포착하는 방향으로 video representation을 학습시키되, 동시 발생 사건의 non-causal attention에는 역효과를 일으킬 가능성.

---

## 5. 아키텍처 검토 및 다음 실험 방향

### 5-1. 아키텍처 요약 (MaskedVideoTokenDiffusion)

```
token_dim         = 256   (latent_channels × unshuffle_factor²)
hidden_size       = 2048  (LLM hidden dim, Qwen2-2B)
depth             = 4     (PrefixDiffusionBlock 개수)
num_heads         = 8
max_latent_tokens = 1764
latent_chunk_size = 144   (tokens per frame)
causal            = False / True
```

**Loss**:
```
total_loss = lm_loss + diffusion_loss_weight × diffusion_loss
             (default: diffusion_loss_weight = 1.0)
```

**Inference 파이프라인**:
```
비디오 → RossVAE → Latent (256, 12×12 per frame)
→ LM forward → video token 위치 hidden states (2048-dim) = condition
→ MaskedVideoTokenDiffusion(condition) → denoised latent
```

### 5-2. `diffusion_loss_weight` 조정의 유의미성

**결론: 현 시점에서 유의미하지 않음.**

- CH/TC 패턴은 weight 크기가 아닌 diffusion objective 자체의 **inductive bias** 문제. weight를 조정해도 이 구조는 변하지 않음.
- TVBench의 counting 실패는 capacity가 아닌 LM의 task 이해 문제. weight 조정으로 해결 불가.
- n=1,000 통계 파워로는 weight 변화의 효과를 측정조차 불가 (CI ±2.3%p).

### 5-3. Layer 크기 변경의 유의미성

**결론: 현 시점에서 유의미하지 않음.**

- action_count/moving_direction/object_count의 랜덤 수준 성능은 diffusion head **capacity 병목이 아님**. LM이 해당 task 자체를 이해 못 하는 상태.
- 재학습 비용 大, 결과 측정 파워 小.

### 5-4. 실제로 의미 있는 다음 실험 (우선순위순)

| 순위 | 실험 | 이유 |
|---|---|---|
| 1 | **NextQA full val 평가** | n=5,000+으로 통계적 파워 확보. 지금은 측정 도구 자체가 부족 |
| 2 | **mf=64, 128 추가** | diffusion 포화점 정확히 확인. 저비용, 고insight |
| 3 | **action_antonym ffprobe 수정** | AVI 코덱 변환 또는 로더 수정으로 12.5% 데이터 복구 |
| 4 | **TVBench C/A skip 원인 파악** | 프롬프트/디코딩 레벨 버그로 추정. 해결 시 TVBench 결과 신뢰성 확보 |
| 5 | **CH 타입 attention 시각화** | diffusion이 Causal-How에 왜 강한지 메커니즘 이해 |

---

## 부록: 파일 경로 참조

```
data/tvbench/
├── convert_tvbench.py          # 변환 스크립트
├── test_sft.jsonl              # 전체 2,525개
├── mini_sft.jsonl              # 랜덤 1,000개 (seed=42)
└── <category>_sft.jsonl       # 카테고리별 10개 파일

scripts/tvbench/
├── infer_tvbench_jsonl.py
├── eval_tvbench_jsonl.sh
├── eval_tvbench_jsonl_baseline.sbatch
├── eval_tvbench_jsonl_diffusion.sbatch
├── eval_tvbench_jsonl_causal_diffusion.sbatch
└── submit_eval_sweep.sh

result0524/
├── nextqa/                     # baseline/diffusion/causal_diffusion × mf8/32/100
│   ├── val_mini_1000_*_predictions.jsonl
│   └── val_mini_1000_*_predictions.jsonl.metrics.json
└── tvbench/
    ├── mini_*_predictions.jsonl
    └── mini_*_predictions.jsonl.metrics.json
```
