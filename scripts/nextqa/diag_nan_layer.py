#!/usr/bin/env python3
"""
NaN layer diagnostic: loads the real model, feeds synthetic 336x336 frames,
runs the full diffusion forward pass, and reports exactly which transformer
layer first produces non-finite values.

Usage:
    python scripts/nextqa/diag_nan_layer.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH  = "checkpoints/VideoLLaMA3-2B"
VAE_PATH    = "pretrained_vae"
DEVICE      = "cuda:0"   # CUDA_VISIBLE_DEVICES=4 maps to cuda:0 inside process
DTYPE       = torch.bfloat16

N_FRAMES    = 4       # one diffusion chunk
SIGLIP_SIZE = 336
VAE_SIZE    = 384
MERGE_SIZE  = 2
PATCH_SIZE  = 14

print("=" * 60)
print("Loading model …")
print("=" * 60)

from videollama3.diffusion import (
    RossVAE, MaskedVideoTokenDiffusion, diffusion_target_geometry,
    pad_square_resize_frame, prepare_vae_frame, chunk_frames_for_diffusion,
)

# Use the checkpoint's own model class (trust_remote_code=True) to avoid
# the key-mismatch corruption that occurs with our Videollama3Qwen2ForCausalLM wrapper.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=DTYPE,
    trust_remote_code=True,
).to(DEVICE).eval()

# Patch in diffusion config attrs that aren't in the base checkpoint
model.config.diffusion_chunk_size      = N_FRAMES
model.config.diffusion_target_spatial  = 12
model.config.diffusion_pixel_unshuffle = 4

from videollama3.model.videollama3_encoder.image_processing_videollama3 import Videollama3ImageProcessor as VideoLLaMA3ImageProcessor

img_proc  = VideoLLaMA3ImageProcessor.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print("Model loaded. Creating synthetic frames …")

# ── 1. synthetic video: N_FRAMES random 336×336 RGB frames ──────────────
rng = np.random.default_rng(42)
raw_frames = [rng.integers(0, 255, (SIGLIP_SIZE, SIGLIP_SIZE, 3), dtype=np.uint8)
              for _ in range(N_FRAMES)]

# ── 2. build pixel_values via the image processor directly ────────────────
from PIL import Image as PILImage
pil_frames = [PILImage.fromarray(f) for f in raw_frames]

# Pass frames as one video: images=[[f1,f2,f3,f4]] → grid_size=(4,H,W)
img_out     = img_proc(images=[pil_frames], merge_size=MERGE_SIZE, return_tensors="pt")
pixel_values = img_out["pixel_values"].to(DEVICE, DTYPE)
grid_sizes   = img_out["grid_sizes"].to(DEVICE)    # (1, 3): [[4, 24, 24]]
merge_sizes  = img_out["merge_sizes"].to(DEVICE)   # (1,): [2]

# Compute number of image tokens after token compression
t_v, gh, gw = int(grid_sizes[0, 0]), int(grid_sizes[0, 1]), int(grid_sizes[0, 2])
m            = int(merge_sizes[0])
n_img_tokens = t_v * (gh // m) * (gw // m)        # 4 * 12 * 12 = 576

# Build input_ids: image token placeholders followed by a short text
image_token_id = model.config.image_token_index
text_ids       = tokenizer.encode("Describe the video.", add_special_tokens=False)
input_ids      = torch.tensor(
    [image_token_id] * n_img_tokens + text_ids, dtype=torch.long
).unsqueeze(0).to(DEVICE)
attention_mask = torch.ones_like(input_ids)

print(f"  pixel_values: {tuple(pixel_values.shape)}  dtype={pixel_values.dtype}")
print(f"  grid_sizes:   {tuple(grid_sizes.shape) if grid_sizes is not None else None}")
print(f"  input_ids:    {tuple(input_ids.shape)}")

# ── 3. check mm_features (encoder + projector) independently ──────────────
print()
print("=" * 60)
print("3. Checking SigLIP encoder output (before projector) …")
print("=" * 60)

def report_finite(t, label):
    fin = torch.isfinite(t)
    ok = fin.all().item()
    nan_n = t.isnan().sum().item()
    inf_n = t.isinf().sum().item()
    vals = t[fin]
    rng = f"[{vals.min():.3f}, {vals.max():.3f}]" if vals.numel() > 0 else "all-nan/inf"
    print(f"  {'✓' if ok else '✗'} {label}:  finite={ok}  nan={nan_n}  inf={inf_n}  range={rng}")
    return ok

with torch.no_grad():
    # Encoder only (before projector)
    enc_out = model.get_model().get_vision_encoder()(
        pixel_values=pixel_values,
        grid_sizes=grid_sizes,
        merge_sizes=merge_sizes,
    )
report_finite(enc_out, "encoder output")

# Now encoder + projector
with torch.no_grad():
    mm_feats = model.encode_images(pixel_values, grid_sizes, merge_sizes)
print()
print("=" * 60)
print("3b. Checking mm_features (encoder + projector) …")
print("=" * 60)
report_finite(mm_feats, "mm_features")

# ── 4. full LLM forward with output_hidden_states=True ─────────────────────
print()
print("=" * 60)
print("4. Full forward pass (output_hidden_states=True, use_cache=False) …")
print("=" * 60)

fwd_kwargs = dict(
    input_ids=input_ids,
    attention_mask=attention_mask,
    pixel_values=pixel_values,
    grid_sizes=grid_sizes,
    modals=["video"],
    output_hidden_states=True,
    use_cache=False,
)
# merge_sizes kwarg name may vary
try:
    fwd_kwargs["merge_sizes"] = merge_sizes
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=DTYPE):
        outputs = model(**fwd_kwargs)
except TypeError:
    fwd_kwargs.pop("merge_sizes")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=DTYPE):
        outputs = model(**fwd_kwargs)

print(f"  LM loss: {outputs.loss}")
all_hs = outputs.hidden_states  # tuple: (embed, layer1, …, layerN)
print(f"  Hidden states returned: {len(all_hs)} layers (0=embed, -1=last)")

first_nan_layer = -1
for i, hs in enumerate(all_hs):
    if not torch.isfinite(hs).all():
        nan_n = hs.isnan().sum().item()
        inf_n = hs.isinf().sum().item()
        finite_vals = hs[torch.isfinite(hs)]
        rng_str = f"[{finite_vals.min():.2f}, {finite_vals.max():.2f}]" if finite_vals.numel() else "all-nan"
        print(f"\n  ✗ FIRST NaN at hidden_states[{i}]  "
              f"nan={nan_n}  inf={inf_n}  finite_range={rng_str}")
        if i == 0:
            print("    → layer 0 = embedding output → NaN comes from mm_features/projector")
        else:
            print(f"    → NaN introduced inside transformer layer {i}")
        first_nan_layer = i
        break

if first_nan_layer == -1:
    print("  ✓ All hidden states finite — NaN must come from diffusion head forward")
    # ── 5. check diffusion head ──────────────────────────────────────────
    print()
    print("=" * 60)
    print("5. Checking diffusion head with real condition tokens …")
    print("=" * 60)
    vae = RossVAE(VAE_PATH).to(DEVICE)
    vae.eval()
    geo = diffusion_target_geometry(VAE_SIZE, 12, vae.latent_channels, N_FRAMES)
    head = MaskedVideoTokenDiffusion(
        token_dim=geo.token_dim,
        hidden_size=model.config.hidden_size,
        depth=4,
        max_latent_tokens=geo.tokens_per_chunk,
        latent_chunk_size=geo.tokens_per_frame,
    ).to(DEVICE).eval()

    hidden_last = all_hs[-1]          # [1, seq, hidden]
    image_token_id = model.config.image_token_index
    video_mask = (input_ids == image_token_id)[0]  # [seq]
    cond_tokens = hidden_last[0][video_mask]        # [N_video_tokens, hidden]
    print(f"  video token count: {cond_tokens.shape[0]}, expected ~{N_FRAMES * (SIGLIP_SIZE//PATCH_SIZE//MERGE_SIZE)**2}")

    restored = head.restore_tokens(
        cond_tokens.float(),
        torch.arange(cond_tokens.shape[0]),
        full_length=geo.tokens_per_chunk,
    )
    # encode dummy VAE target
    dummy_frames_t = torch.stack([
        prepare_vae_frame(f, VAE_SIZE) for f in raw_frames
    ]).to(DEVICE)  # [4, 3, 384, 384]
    with torch.no_grad():
        posterior = vae.encode(dummy_frames_t).latent_dist
        latent = (posterior.sample() - vae.shift_factor) * vae.scaling_factor
        latent = torch.nn.functional.pixel_unshuffle(latent, 4)
        k, c, h, w = latent.shape
        targets = latent.permute(0, 2, 3, 1).reshape(k * h * w, c)
    print(f"  VAE targets finite: {torch.isfinite(targets).all().item()}  shape={tuple(targets.shape)}")
    loss = head.diffusion_loss(
        restored.unsqueeze(0),
        targets.unsqueeze(0).float(),
    )
    print(f"  diffusion_loss = {loss.item():.4f}  finite={torch.isfinite(loss).item()}")

print()
print("=" * 60)
print("Done.")
