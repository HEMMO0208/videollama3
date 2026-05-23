# Diffusion Supervision Experiment Analysis

## 1. 현재까지 결과 (Stage 3 base)

### 1-1. 전체 accuracy

| 설정 | default frames | mf8 | mf32 |
|------|---------------|-----|------|
| baseline | 85.0% | 82.2% | 84.7% |
| diffusion | **85.1%** | 82.2% | 84.7% |

### 1-2. 타입별 (default frames)

| type | n | baseline | diffusion | diff |
|------|---|----------|-----------|------|
| TN | 174 | 81.0% | 82.2% | +1.1% |
| CW | 372 | **84.4%** | 83.3% | -1.1% |
| CH | 148 | 86.5% | **88.5%** | +2.1% |
| DL | 53 | **98.1%** | 96.2% | -1.9% |
| TC | 127 | 83.5% | 83.5% | 0.0% |
| TP | 15 | **93.3%** | 86.7% | -6.7% |
| DC | 36 | 75.0% | **80.6%** | +5.6% |
| DO | 75 | 90.7% | 90.7% | 0.0% |

---

## 2. Training Dynamics 분석

### 2-1. Loss 구조

HF Trainer의 `train/loss`는 `gradient_accumulation_steps(=8)`배로 스케일된 값으로 로깅됨.
`lm_loss`, `diffusion_loss`는 마지막 micro-batch 1개의 unscaled 값 → 직접 합산 불가.

실제 per-sample 평균:
- `lm_loss` ≈ 0.007 (near-zero, 시작부터 거의 수렴)
- `diffusion_loss` ≈ 0.295 (전체 loss의 99%)

### 2-2. 핵심 문제: Diffusion Loss Flat

| 단계 | 초기 | 최종 | 감소율 |
|------|------|------|--------|
| pretrain (LLM frozen) | 6.81 | 0.46 | 93.3% |
| joint finetune (Stage 3 base) | 0.32 | 0.32 | **0%** |

Stage 3 base에서 시작하면 LM loss ≈ 0 → LLM이 표현을 바꿀 이유 없음 → diffusion head도 local minimum 고착.

---

## 3. 실험 방향

### 3-1. Image-aligned 모델(Stage 2)에서 재실험 [진행 중]

Stage 3 base의 근본 문제: 모델이 이미 수렴 상태 → diffusion signal이 개입할 여지 없음.

Image base(VideoLLaMA3-2B-Image)에서 시작하면:
- LM loss 높은 상태에서 시작 → gradient signal 강함
- 표현이 형성되는 과정에서 diffusion이 함께 개입 → visual info 유지 강제 가능

sbatch 파일 3개 (`finetune_nextqa_baseline.sbatch`, `finetune_nextqa_diffusion.sbatch`, `pretrain_nextqa_diffusion.sbatch`) 모두 `VideoLLaMA3-2B-Image`로 변경 완료.

---

## 4. 체크포인트 분석 계획

### Layer 1: 성능 숫자 해부

**기본 accuracy 비교**
```
Image-baseline vs Image-diffusion vs Stage3-baseline vs Stage3-diffusion
핵심 질문: Image base에서 diffusion이 유의미하게 도움이 되는가?
```

**비디오 길이별 breakdown**

NExT-QA 메타데이터로 영상 길이를 join해 short(<30s) / medium(30-60s) / long(>60s) 별 accuracy 비교.
Diffusion이 긴 영상에서 더 이득이면 "시간 정보 유지" 가설 확인.

---

### Layer 2: Training Log 비교 [우선순위 최고]

Image base에서도 diffusion_loss가 flat인가?
- flat → 문제가 head LR 또는 구조에 있음
- 같이 내려감 → "표현 형성 과정에 개입" 가설 확인

`lm_loss`와 `diffusion_loss` 곡선을 Stage 3 실험과 직접 비교.

---

### Layer 3: Representation 해부

**Linear Probe**

동일한 비디오를 두 모델(baseline vs diffusion)에 입력 후,
LLM 레이어별 video token hidden state에 linear classifier 학습:

- probe 1: 프레임 내 객체 분류 (object presence)
- probe 2: 프레임 순서 맞추기 (temporal order)
- probe 3: 모션 방향 (static / forward / backward)

Diffusion 모델이 더 높으면 표현이 실제로 더 풍부하다는 직접 증거.

**CKA (Centered Kernel Alignment)**

두 모델의 레이어별 hidden state similarity 시각화.
어느 레이어부터 표현이 갈라지는지 확인.

---

### Layer 4: Attention 해부

레포 내 기존 스크립트(`visualize_nextqa_*_attn.py`) 활용:

```
baseline 틀리고 diffusion 맞춘 문제
  → correct-option→video attention 비교
  → diffusion 모델이 정답 관련 프레임을 더 attend하는가?

diffusion 틀리고 baseline 맞춘 문제
  → 반대 방향 분석
```

---

### Layer 5: Diffusion Head 복원 품질

Fine-tune 후 head로 실제 프레임을 복원해 SSIM / LPIPS 측정:

| checkpoint | reconstruction quality |
|------------|----------------------|
| pretrain 후 | ? |
| Stage3 joint finetune 후 | ? |
| Image joint finetune 후 | ? |

Stage 3 실험에서 diffusion_loss flat(0.32)이었으므로 복원 품질도 변화 없을 것으로 예상.
Image base 실험에서 loss가 내려가면 복원 품질도 같이 올라가는지 검증.

---

## 5. 우선순위 요약

| 분석 | 난이도 | 인사이트 |
|------|--------|---------|
| Training log (lm+diff 곡선) | 낮음 | **최고** — 학습 dynamics 즉시 확인 |
| 비디오 길이별 accuracy | 낮음 | 중간 |
| Attention 비교 (기존 스크립트 활용) | 낮음 | **높음** |
| Diffusion 복원 품질 (SSIM/LPIPS) | 중간 | **높음** |
| Linear probe | 높음 | **최고** — 표현 변화 직접 증거 |
