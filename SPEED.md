# MPS Speed Optimization Log

## Standing Instructions
- **Continue tirelessly without asking for user prompt** — keep optimizing in a loop
- Commit each time a good improvement is reached
- Complexity increase must be justified by speed results
- Re-read this file at each context compaction
- Take notes on all results, what worked, what didn't

## Current Baseline (2026-02-06)
- Decoder: 43.2 ms/token
- Encoder: ~2.4s (test_speech.wav, 3.6s audio)
- Theoretical floor: ~23 ms/token (300 GB/s bandwidth, 6.9 GB weights)
- Remaining overhead: ~79 command buffer round-trips per token

## Already Optimized
- Fused QKV: 3 matmuls → 1 command buffer (saves 52 round-trips/token)
- Fused FFN: w1+w3+silu+mul+w2 → 1 command buffer (saves 52 round-trips/token)
- Persistent decoder buffers: eliminates 11 malloc/free per token
- Pre-warm weight cache: all bf16→f16 conversion at load time (8.4 GB GPU)
- GPU batched encoder attention: causal_softmax shader + strided MPS views
- Fused encoder QKV + FFN (reusing decoder fused functions)

## Optimization Attempts

### Attempt 1: Fused wo+FFN + Fused norm+QKV + Fused logits (SUCCESS)
- **Fused wo+FFN**: wo projection + residual + rms_norm + ada_scale + SwiGLU FFN + residual in 1 cmd buf
  - Eliminates separate wo cmd buf per layer
  - Result: 43 → ~40 ms/step
- **Fused norm+QKV**: rms_norm + 3 QKV matmuls in 1 cmd buf (replaces separate norm + fused QKV)
  - Marginal additional improvement (CPU rms_norm was fast)
- **Fused logits**: final rms_norm + logits matmul + argmax_f32 shader in 1 cmd buf
  - Avoids 512KB logits download when only argmax needed
  - Result: ~40 → ~38 ms/step
- **Prefill timing split**: separate prefill_ms from per-step timing for accurate measurement
- **Combined: 43.2 → 38 ms/step (12% improvement)**
- Command buffers per token: 79 → 53 (26×2 per layer + 1 logits)
- New shaders: ada_scale_mul, argmax_f32

### Attempt 2: Cross-layer fusion — persistent GPU x (SUCCESS)
- Keep x on GPU buffer across all 26 layers (no CPU round-trip between layers)
- Fuse wo_ffn[i] + norm_qkv[i+1] into a single command buffer
- Helper functions: encode_wo_ffn_steps(), encode_norm_qkv_steps()
- New API: decoder_start/end, decoder_norm_qkv, decoder_wo_ffn_next_qkv, decoder_wo_ffn_logits
- Command buffers per token: 53 → 27 (1 + 25 fused + 1 final)
- **Result: 38 → 32.5 ms/step (test_speech), 39 → 35.7 ms/step (jfk)**
- **Cumulative from baseline: 43.2 → 32.5 ms/step (25% faster)**

### Attempt 3: Monolithic decoder step — 1 command buffer (SUCCESS)
- All 26 layers + logits in a SINGLE Metal command buffer
- 3 new Metal compute shaders: rope_apply, kv_cache_copy, decoder_attention
- KV cache allocated with MTLResourceStorageModeShared (zero-copy CPU/GPU)
- GPU RoPE: applies rotary embeddings on Q and K (interleaved pair convention)
- GPU KV cache write: copies K/V to shared cache at correct layer/position offset
- GPU decoder attention: one threadgroup (128 threads) per head, cooperative dot products
  - Online softmax (single pass): avoids second scan over keys
  - First attempt with 1-thread-per-head: 77.9 ms/step (TERRIBLE — sequential per-key scan)
  - Fixed with 128-thread cooperative kernel: 26.6 ms/step (great improvement)
- Command buffers per token: 27 → 1
- **Result: 32.5 → 26.6 ms/step (test_speech), 35.7 → 28.7 ms/step (jfk)**
- **Cumulative from baseline: 43.2 → 26.6 ms/step (38% faster)**

### Attempt 4: SIMD attention (SUCCESS)
- Replaced threadgroup barrier reductions with simd_sum() in decoder_attention kernel
- 4 SIMD groups of 32 threads, simd_sum for dot product, cross-SIMD via threadgroup
- **Result: 26.6 → 25.3 ms/step (test_speech)**

### Attempt 5: Merged weight matrices (SUCCESS)
- Merged w1+w3 into single MPS matmul in FFN (saves 1 MPS encode per layer)
  - get_merged_f16_2(): concatenates two bf16 weight buffers into one f16 buffer
  - bufGate now sized hidden*2, silu/mul use buffer offsets
- Merged Q+K+V into single MPS matmul in QKV projection (saves 2 MPS encodes per layer)
  - get_merged_f16_3(): concatenates three bf16 weight buffers into one f16 buffer
  - Single bufQKV output, downstream RoPE/KV-cache/attention use buffer offsets
- Total: 3 fewer MPS matmul encodes per layer × 26 layers = 78 fewer encodes
- **Result: 25.3 → 23.7 ms/step (test_speech), ~24.9 ms/step (jfk)**
- **Cumulative from baseline: 43.2 → 23.7 ms/step (45% faster)**

### Next targets
- Theoretical floor: ~23 ms (300 GB/s, 6.9 GB weights)
- Gap: 23.7 - 23 = 0.7ms (very close to memory bandwidth limit)
- Ideas: encoder speedups, attention kernel optimizations for longer sequences

## MLX Credits
- If any optimization ideas or kernel code are taken from Apple MLX
  (https://github.com/ml-explore/mlx), proper credits must be added to
  both the README and the relevant source file.
