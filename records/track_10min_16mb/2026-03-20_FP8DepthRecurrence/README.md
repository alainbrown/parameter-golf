# FP8 Training Submission

## Approach

Custom train_gpt.py built from scratch. All linear layer matmuls use FP8 (float8_e4m3fn forward, float8_e5m2 backward) via `torch._scaled_mm` with custom autograd ops adapted from modded-nanogpt. ~2× H100 throughput vs bf16.

### Features

| Feature | Description |
|---------|-------------|
| FP8 matmuls | Custom ops via `torch.library.custom_op`, tensor scales, `torch.compile(fullgraph=True)` |
| Value embeddings | 5 banks of (vocab, kv_dim) added to attention V per layer |
| BigramHash | 4096-bucket token-pair hash table (dim=128 → model_dim) |
| Depth recurrence | Loop blocks K times for more effective depth, fewer stored params |
| 3× MLP | Hidden dim = 1536 (funded by int6 compression) |
| Orthogonal init | fp32 QR decomposition, cast to bf16. Output projections scaled by 1/√(2×eff_layers) |
| Int6 quantization | Per-row [-32,31] with fp16 embedding passthrough |
| Sliding window eval | Overlapping windows (stride=64), each token scored with 960+ context |
| Muon optimizer | Newton-Schulz orthogonalization for 2D weights, Adam for scalars/embeddings |
| SWA | Stochastic weight averaging during warmdown (toggleable) |
| QAT | STE fake-quantization to int6 during training (toggleable) |
| Grad clip | 0.3 default |
| zstd/zlib | zstd-22 when available, zlib-9 fallback |

### Removed
- **Multi-token prediction**: CUDA driver error on H100 when FP8 padding interacts with non-16-aligned sequence slicing from extra head targets. Removed.
- **SmearGate**: Experiments showed it hurts BPB (see below). Disabled by default.

### Feature Flags (env vars)
`VE=1/0`, `SMEAR=0/1`, `BIGRAM=1/0`, `QAT=0/1`, `SWA_EVERY=0/50`

## Experiment Results

### Setup
- 1×H100 SXM 80GB on RunPod
- 5 minutes training per experiment (`MAX_WALLCLOCK_SECONDS=300`)
- No warmup (`WARMUP_STEPS=0`)
- Fast eval (`EVAL_STRIDE=960`)
- SOTA hyperparameter defaults (LR=0.02, warmdown=3000, momentum=0.99, seq=2048, batch=786K)

### Architecture (sorted by BPB)

| Run | Config | BPB | Steps | Artifact | Quant Gap |
|-----|--------|-----|-------|----------|-----------|
| **AR1** | **6L×2 recur, 512 dim** | **3.474** | **128** | **4.6MB** | **0.065** |
| AR5 | 9L, 512 dim | 4.127 | 81 | 6.0MB | 0.117 |
| AR3 | 6L×2 recur, 640 dim | 4.413 | 98 | 6.0MB | 0.400 |
| AR2 | 10L, 512 dim | 4.847 | 66 | 6.3MB | 0.172 |
| AR4 | 6L×3 recur, 512 dim | 5.422 | 69 | 4.4MB | 0.807 |

**Finding**: 6L×2 recurrence dominates. Gets most steps (fastest per step), lowest BPB, smallest quant gap. Wider (640) and deeper recurrence (6L×3) are both worse. Standard 9L/10L can't compete — too few steps in 5 min.

### Feature Ablations (base = AR1 with all features on)

| Run | Change | BPB | Steps | Quant Gap | Impact |
|-----|--------|-----|-------|-----------|--------|
| **NO_SMEAR** | **SMEAR=0** | **3.294** | **123** | **0.157** | **Best result — SmearGate hurts** |
| AR1 | All on | 3.474 | 128 | 0.065 | Baseline |
| NO_BIGRAM | BIGRAM=0 | 3.493 | 125 | 0.075 | BigramHash helps slightly |
| NO_VE | VE=0 | 4.708 | 72 | 0.335 | VE helps a lot |

**Findings**:
- **SmearGate hurts**: Disabling it improves BPB from 3.474 to 3.294. SmearGate should be off.
- **Value embeddings essential**: Removing VE degrades BPB by 1.2+ and reduces steps (slower model).
- **BigramHash marginal**: Small improvement (0.02 BPB). Worth keeping — low overhead.

### Unique Hyperparameters

| Run | Change | BPB | Steps | Quant Gap |
|-----|--------|-----|-------|-----------|
| AR1 | Baseline (4096 buckets, 5 VE banks) | 3.474 | 128 | 0.065 |
| BG_10k | BIGRAM_BUCKETS=10240 | 3.481 | 126 | 0.067 |
| VE_3 | NUM_VALUE_BANKS=3 | 3.544 | 114 | 0.081 |
| VE_8 | NUM_VALUE_BANKS=8 | 3.562 | 112 | 0.088 |

**Findings**:
- **BigramHash 10k ≈ 4k**: More buckets doesn't help. Keep 4096.
- **5 VE banks is optimal**: 3 banks worse (fewer capacity), 8 banks worse (more params = slower steps, worse quant).

### SWA + QAT

| Run | Config | BPB | Steps | Quant Gap |
|-----|--------|-----|-------|-----------|
| SWA_on | SWA_EVERY=50 SWA_START_FRAC=0.4 | 3.553 | 124 | 0.073 |
| SQ_both | SWA + QAT | 3.574 | 122 | 0.076 |
| QAT_on | QAT=1 | 4.064 | 99 | 0.209 |

**Findings**:
- **SWA doesn't help in 5 min**: Only 2 snapshots collected — not enough warmdown steps. May help at full 10-min scale.
- **QAT hurts**: Adds overhead (fewer steps: 99 vs 128) and worsens quant gap. FP8 + QAT is a bad combination — the FP8 noise during training conflicts with QAT's fake quantization.
- **SQ_both**: Similar to SWA alone. QAT's overhead partially offset by SWA averaging.

### Full Results Summary (sorted by BPB)

| Rank | Run | BPB | Steps | Artifact | Quant Gap |
|------|-----|-----|-------|----------|-----------|
| 1 | **NO_SMEAR** | **3.294** | 123 | 4.6MB | 0.157 |
| 2 | AR1 (all on) | 3.474 | 128 | 4.6MB | 0.065 |
| 3 | NO_MTP* | 3.476 | 125 | 4.6MB | 0.064 |
| 4 | MTP_3h* | 3.475 | 127 | 4.6MB | 0.064 |
| 5 | BG_10k | 3.481 | 126 | 4.8MB | 0.067 |
| 6 | NO_BIGRAM | 3.493 | 125 | 4.4MB | 0.075 |
| 7 | VE_3 | 3.544 | 114 | 4.2MB | 0.081 |
| 8 | SWA_on | 3.553 | 124 | -- | 0.073 |
| 9 | VE_8 | 3.562 | 112 | 5.1MB | 0.088 |
| 10 | SQ_both | 3.574 | 122 | -- | 0.076 |
| 11 | QAT_on | 4.064 | 99 | -- | 0.209 |
| 12 | AR5 (9L) | 4.127 | 81 | 6.0MB | 0.117 |
| 13 | AR3 (6L×2, 640) | 4.413 | 98 | 6.0MB | 0.400 |
| 14 | NO_VE | 4.708 | 72 | 3.6MB | 0.335 |
| 15 | AR2 (10L) | 4.847 | 66 | 6.3MB | 0.172 |
| 16 | AR4 (6L×3) | 5.422 | 69 | 4.4MB | 0.807 |

*MTP not in script — these are duplicate baselines.

## Optimal Config

```
NUM_LAYERS=6 RECURRENCE_LOOPS=2 MODEL_DIM=512 MLP_MULT=3
VE=1 SMEAR=0 BIGRAM=1 QAT=0 SWA_EVERY=0
MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03
MUON_MOMENTUM=0.99 WARMDOWN_ITERS=3000 TRAIN_SEQ_LEN=2048
TRAIN_BATCH_TOKENS=786432 GRAD_CLIP_NORM=0.3
NUM_VALUE_BANKS=5 BIGRAM_BUCKETS=4096
```

Best BPB: 3.294 (5 min, 1×H100). SmearGate off, everything else on. No SWA or QAT.

## Pipeline

1. Setup (DDP, seeds, data, tokenizer)
2. Build GPT with feature flags
3. `torch.compile(fullgraph=True)`
4. DDP wrap
5. Muon + Adam optimizer setup
6. Warmup (compile priming, then reset weights)
7. Training loop (FP8 forward/backward → Muon/Adam → LR warmdown → SWA collection)
8. SWA averaging (if enabled)
9. Pre-quant eval (standard, full val set)
10. Int6 quantization → zstd/zlib compress
11. Dequantize roundtrip → sliding window eval → final BPB

## Configuration

TODO: Fill in after final run.

## Command

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Key Metrics

TODO: Fill in from `train.log` after final run.

## Included Files

- `train_gpt.py` (code snapshot used for the run)
- `train.log` (exact remote training log)
- `submission.json` (leaderboard metadata)
