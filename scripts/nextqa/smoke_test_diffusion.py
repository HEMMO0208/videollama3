#!/usr/bin/env python3
"""
Smoke test: diffusion 파이프라인 각 단계별 finite 확인
Usage: python scripts/nextqa/smoke_test_diffusion.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import numpy as np

HIDDEN_SIZE = 2048   # Qwen2 2B
LATENT_CH   = 16     # Flux VAE latent channels
CHUNK_SIZE  = 4
N_CHUNKS    = 3      # 테스트할 chunk 수

def tag(ok): return "✓" if ok else "✗ FAIL"
def finite(t, name):
    ok = torch.isfinite(t).all().item()
    if not ok:
        nan = t.isnan().sum().item()
        inf = t.isinf().sum().item()
        print(f"    → NaN={nan}, Inf={inf}, range=[{t[torch.isfinite(t)].min():.3f}, {t[torch.isfinite(t)].max():.3f}]")
    return ok

# ──────────────────────────────────────────────────────────
print("=" * 60)
print("1. Preprocessing: pad_square_resize_frame + prepare_vae_frame")
print("=" * 60)
from videollama3.diffusion import pad_square_resize_frame, prepare_vae_frame, chunk_frames_for_diffusion

shapes = [(720, 1280, 3), (1080, 1920, 3), (480, 640, 3), (336, 336, 3)]
for shape in shapes:
    frame = np.random.randint(0, 255, shape, dtype=np.uint8)
    sq    = pad_square_resize_frame(frame, 336)
    vf    = prepare_vae_frame(sq, 384)
    ok    = sq.shape == (336, 336, 3) and finite(vf, f"vae_frame{shape}")
    print(f"  {tag(ok)} {shape} → pad_sq={sq.shape} → vae={tuple(vf.shape)}")

# ──────────────────────────────────────────────────────────
print()
print("=" * 60)
print("2. diffusion_target_geometry  (vae_size=384, spatial=12)")
print("=" * 60)
from videollama3.diffusion import diffusion_target_geometry
geo = diffusion_target_geometry(384, 12, LATENT_CH, CHUNK_SIZE)
print(f"  unshuffle_factor  = {geo.unshuffle_factor}  (expect 4)")
print(f"  token_dim         = {geo.token_dim}  (expect {LATENT_CH * 16})")
print(f"  tokens_per_frame  = {geo.tokens_per_frame}  (expect 144)")
print(f"  tokens_per_chunk  = {geo.tokens_per_chunk}  (expect 576)")
assert geo.unshuffle_factor == 4 and geo.tokens_per_frame == 144 and geo.tokens_per_chunk == 576

# ──────────────────────────────────────────────────────────
print()
print("=" * 60)
print("3. MaskedVideoTokenDiffusion — forward / diffusion_loss")
print("=" * 60)
from videollama3.diffusion import MaskedVideoTokenDiffusion

head = MaskedVideoTokenDiffusion(
    token_dim=geo.token_dim,
    hidden_size=HIDDEN_SIZE,
    depth=4,
    max_latent_tokens=geo.tokens_per_chunk,
    latent_chunk_size=geo.tokens_per_frame,
).eval()

B, N, D = 1, geo.tokens_per_chunk, geo.token_dim  # 1, 576, 256

for dtype_name, cond_dtype, tgt_dtype in [
    ("float32 / float32", torch.float32, torch.float32),
    ("bfloat16 / float32", torch.bfloat16, torch.float32),
    ("bfloat16 / bfloat16", torch.bfloat16, torch.bfloat16),
]:
    cond   = torch.randn(B, N, HIDDEN_SIZE).to(cond_dtype)
    target = torch.randn(B, N, D).to(tgt_dtype)
    with torch.no_grad():
        loss = head.diffusion_loss(cond, target)
    ok = finite(loss, f"loss_{dtype_name}")
    print(f"  {tag(ok)} diffusion_loss ({dtype_name}) = {loss.item():.4f}")

# ──────────────────────────────────────────────────────────
print()
print("=" * 60)
print("4. restore_tokens — 압축 비율별 finite 확인")
print("=" * 60)
for keep_ratio in [1.0, 0.5, 0.1, 0.0]:
    kept_count   = int(N * keep_ratio)
    kept_tokens  = torch.randn(kept_count, HIDDEN_SIZE).bfloat16()
    kept_indices = torch.randperm(N)[:kept_count]
    restored     = head.restore_tokens(kept_tokens, kept_indices, full_length=N)
    ok = restored.shape == (N, HIDDEN_SIZE) and finite(restored, f"restored_{keep_ratio}")
    print(f"  {tag(ok)} keep={keep_ratio*100:.0f}%  shape={tuple(restored.shape)}")

# ──────────────────────────────────────────────────────────
print()
print("=" * 60)
print("5. end-to-end: restore → diffusion_loss (N_CHUNKS 배치)")
print("=" * 60)
restored_list = []
target_list   = []
for _ in range(N_CHUNKS):
    kept_count   = np.random.randint(1, N)
    kept_tokens  = torch.randn(kept_count, HIDDEN_SIZE).bfloat16()
    kept_indices = torch.randperm(N)[:kept_count]
    restored     = head.restore_tokens(kept_tokens, kept_indices, full_length=N)
    target_np    = np.random.randint(0, 255, (CHUNK_SIZE, 3, 384, 384), dtype=np.uint8).astype(np.float32) / 255.0
    target_t     = torch.tensor(target_np).bfloat16()
    restored_list.append(restored)
    target_list.append(target_t.view(CHUNK_SIZE, 3, 384, 384))

# MaskedVideoTokenDiffusion은 이미 VAE-encoded target을 받는다
# 여기서는 mock VAE output으로 대체: (576, 256)
mock_targets = [torch.randn(N, D).bfloat16() for _ in range(N_CHUNKS)]
r_stack = torch.stack(restored_list).bfloat16()       # [N_CHUNKS, 576, hidden]
t_stack = torch.stack(mock_targets).bfloat16()         # [N_CHUNKS, 576, 256]
loss = head.diffusion_loss(r_stack, t_stack)
ok = finite(loss, "e2e_loss")
print(f"  {tag(ok)} e2e diffusion_loss = {loss.item():.4f}")

# ──────────────────────────────────────────────────────────
print()
print("=" * 60)
print("6. diffusion_head 파라미터 dtype / 초기화 범위")
print("=" * 60)
for name, p in head.named_parameters():
    if p.numel() == 0: continue
    ok = finite(p.data.float(), name)
    print(f"  {tag(ok)} {name:40s}  dtype={p.dtype}  range=[{p.data.float().min():.3f}, {p.data.float().max():.3f}]")

print()
print("=" * 60)
print("Done.")
