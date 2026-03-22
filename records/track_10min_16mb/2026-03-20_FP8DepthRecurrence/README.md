# FP8 Training Submission

## Approach

Custom train_gpt.py built from scratch combining:

**Training speed:**
1. **FP8 matmuls** — all linear layers use float8_e4m3fn forward, float8_e5m2 backward via `torch._scaled_mm` with custom autograd ops. ~2x H100 throughput.
2. **torch.compile(fullgraph=True)** — fused Triton kernels for non-FP8 ops.

**Architecture:**
3. **3× MLP** — hidden=1536 (funded by int6 compression savings).
4. **Value embeddings** — 5 banks of (vocab, kv_dim) added to attention V per layer.
5. **Multi-token prediction** — 2 extra heads predict t+2, t+3 with annealed weights.
6. **Depth recurrence** — loop blocks K times for more effective depth, fewer stored params.
7. **SmearGate** — learned gate blending current + previous token embedding.
8. **BigramHash** — 4096-bucket hash table for token pair patterns.
9. **Orthogonal init** — with output projection scaling by 1/sqrt(2*effective_layers).

**Compression:**
10. **Int6 quantization** — [-32,31] per-row scaling, fp16 embedding passthrough.
11. **zstd-22 compression** — when available, zlib-9 fallback.

**Evaluation:**
12. **Sliding window eval** — stride=64, each token scored with 960+ context.

## Progress

| Step | Feature | Status |
|------|---------|--------|
| 1 | FP8 training | Done |
| 2 | Sliding window eval | Done |
| 3 | Value embeddings | Done |
| 4 | Multi-token prediction | Done |
| 5 | Depth recurrence | Done |
| 6 | torch.compile(fullgraph=True) | Done |
| 7 | Phase 1: int6, 3×MLP, SmearGate, BigramHash, ortho init, grad clip | Done |
| 8 | Phase 2: hyperparameter tuning on H100 | TODO |

## Local Test Results (RTX 2000 Ada, eager mode, 20 steps)

Config: 6L×2 recurrence, 512 dim, 3× MLP, 5 value banks, 2 extra heads, BigramHash(4096)

- **17.6M params**
- **4.8MB artifact** (int6+zlib) — 11.2MB headroom in 16MB budget
- **Val BPB: 4.80** at step 20 (untrained — needs full H100 run)

## Key Findings

### FP8 Implementation
- Custom ops via `torch.library.custom_op` with `register_autograd` and `register_fake`
- Transposed weight storage (in, out) for efficient `_scaled_mm` column-major layout
- Static scales: x_s=100/448, w_s=1.6/448, grad_s=grad_scale*0.75/448
- `_scaled_mm` requires all dimensions divisible by 16 — FP8Linear pads when needed
- `@torch.compile` on inner op impls + `torch.compile(model, fullgraph=True)` — works together
- No `torch.autocast` — model is bf16 natively

### Initialization
- Orthogonal init in fp32, cast to bf16 (orthogonal_ requires fp32 for QR decomposition)
- Output projections scaled by 1/sqrt(2*effective_layers) per muP
- Embedding: normal(std=0.005) for tied embeddings
- Value embeddings: 0.01 * randn
- BigramHash scale: 0.05, embedding std=0.02
- SmearGate: sigmoid(3.0) ≈ 0.95 initial gate value

### Quantization
- Int6 [-32,31] per-row scaling for large 2D weight matrices
- FP16 passthrough for tied embeddings (dual-use means int6 errors compound)
- Small tensors (<65536 elements) kept in fp16
- Control tensors (attn_scale, mlp_scale, resid_mix, q_gain) kept in fp32
- zstd level 22 when available, zlib level 9 fallback

### Compromises
- zstd not in competition image — falls back to zlib (~5% larger artifact)
- Int6 stored in int8 containers — compression handles the wasted bits
- Orthogonal matrix slightly non-orthogonal after bf16 cast
- Output projection scaling + FP8 static scales may cause slow early convergence
- BigramHash projection uses FP8Linear — may be overkill for 128→512 matmul

## Competition Landscape (as of 2026-03-21)

Current SOTA: **1.1428 BPB** (10L Int5-MLP + BigramHash + SWA)

Top techniques across all submissions:
- Int5/Int6 + zstd-22
- 3× MLP expansion
- BigramHash (4096-10240 buckets)
- SmearGate + orthogonal init
- SWA (every 50 steps, last 40% of warmdown)
- Muon WD=0.04, momentum=0.99
- Sequence length 2048
- Sliding window eval stride=64

Our unique contribution: **FP8 matmuls for ~2× training throughput.** No other submission uses FP8. Combined with depth recurrence and the proven techniques above, we aim to get more training steps per 10 minutes than any competitor.

## Next Steps: Phase 2 (H100 experiments)

Hyperparameter tuning (~20 experiments, 2 min each, ~$2):
- LR: 0.02, 0.03, 0.04
- Warmdown: 1200, 3000, 5000, 20000
- Muon momentum: 0.95 vs 0.99
- Sequence length: 1024 vs 2048
- Architecture: layers × recurrence × width
- Batch size: 524K vs 786K
- SWA and QAT

## Architecture

- 6 layers × 2 recurrence = 12 effective depth
- 512 dim × 8 heads × 4 KV heads × 3× MLP (1536 hidden)
- 5 value embedding banks, 2 extra prediction heads
- SmearGate + BigramHash(4096, dim=128→512)
- Tied embeddings (vocab 1024)
- RoPE, RMSNorm, logit softcap, grad clip 0.3
- ~17.6M params, ~4.8MB artifact

## Included Files

- `train_gpt.py` (code snapshot used for the run)
- `train.log` (exact remote training log)
- `submission.json` (leaderboard metadata)
