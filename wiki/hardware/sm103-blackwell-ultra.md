---
id: hw-sm103-blackwell-ultra
title: "SM103 Blackwell Ultra (B300 / GB300)"
type: hardware
architectures: [sm103, sm103a, sm100]
tags: [nvfp4, fp4, fp8, block-scale, tcgen05, tmem, tma, clc, 2sm-cooperative, gemm, moe, attention, mla]
confidence: source-reported
evidence_basis:
  - source_id: doc-blackwell-ultra-sm103
    evidence_type: official-doc
  - source_id: pr-flashinfer-2303
    evidence_type: upstream-code
  - source_id: pr-cutlass-3124
    evidence_type: upstream-code
related:
  - migration-sm100-to-sm103
  - hw-nvfp4
  - hw-tcgen05-mma
  - hw-tmem
  - kernel-nvfp4-gemm
  - kernel-flashmla
  - kernel-fused-moe
sources:
  - doc-blackwell-ultra-sm103
  - doc-cutlass-changelog-sm100
  - pr-flashinfer-2303
  - pr-flashinfer-2404
  - pr-flashinfer-4063
  - pr-flashinfer-3888
  - pr-flashinfer-1850
  - pr-cutlass-3124
  - pr-vllm-30484
  - pr-vllm-38730
  - pr-sglang-9807
  - pr-sglang-21906
aliases:
  - SM103
  - B300
  - GB300
  - "Blackwell Ultra"
  - sm_103
  - sm_103a
  - "CC 10.3"
---

# SM103 Blackwell Ultra (B300 / GB300)

## Overview

**SM103** (compute capability 10.3) is the architecture ID for **NVIDIA Blackwell Ultra** GPUs:

| Product | What it is |
|---|---|
| **B300** | Blackwell Ultra SXM GPU |
| **GB300** | Grace + B300 Superchip (NVLink-C2C) |
| **GB300 NVL72** | 72-GPU rack system |

SM103 keeps the **same programming model** as SM100 (tcgen05, TMEM, TMA, CLC, NVFP4 block-scaled MMA) but changes **peaks, capacity, SFU throughput, and which kernel specializations win**. Treating SM103 as "just another SM100" loses large NVFP4 gains and can hit **forward-compat hangs**.

## Spec Deltas vs SM100 (B200)

| Axis | SM100 (B200) | SM103 (B300/GB300) | Kernel impact |
|---|---|---|---|
| NVFP4 dense peak | ~10 PFLOPS | **~15 PFLOPS (1.5×)** | Prefer NVFP4 paths; higher FP4/BF16 ratio (~6× vs ~4×) |
| HBM3e | ~192 GB | **up to 288 GB** | Larger single-GPU models / batches |
| SMs | ~148 | **up to 160** | Tile schedulers / split-KV must use SM count correctly |
| SFU / softmax | baseline | **~2× SFU**, accel. attention | Softmax-heavy attention more compute-friendly |
| TMEM / SM | 256 KB | 256 KB | Same TMEM budget math |
| ISA family | tcgen05 / TMA / CLC | same family | Still recompile for `sm_103` / `sm_103a` |

Source synthesis: [doc-blackwell-ultra-sm103](../../sources/docs/blackwell-ultra-sm103.md).

## What Stays the Same

```
Same as SM100:
  tcgen05.mma (+ block-scale NVFP4/MXFP4 variants)
  TMEM 256KB/SM (tcgen05.alloc/ld/st/dealloc)
  TMA bulk async loads (cp.async.bulk.tensor)
  CLC persistent tile scheduling
  2-SM cooperative MMA (cta_group::2)
  Warp specialization + multi-stage pipelines
```

You still write Blackwell kernels the SM100 way. SM103 is a **specialization and validation** problem, not a new ISA rewrite.

## What Changes for Kernels

### 1. Architecture-specific NVFP4 schedules

SM103's higher NVFP4/BF16 ratio is only approachable with **SM103 tile/cluster/scheduler specializations**. FlashInfer keeps SM100 configs and adds SM103 templates; autotune picks per shape ([pr-flashinfer-2303](../../sources/prs/flashinfer/PR-2303.md)):

```
Llama-3.1-70B style GEMM (N=8192, K=28672) on SM103 — NVFP4 TFLOP/s
  batch=1024:  5659 → 6341 after SM103 schedules
  batch=4096:  6160 → 6898
  batch=8192:  5939 → 6653
```

Native SM103 NVFP4 mainloops use larger K (e.g. **CTA_K=768**) than many SM100 defaults.

### 2. Epilogue store width and scale fusion

On B300, widening the NVFP4 epilogue output store to **256-bit** (`STG.256`) plus paired `mul.f32x2` scale yields **~1.12–1.16×** geomean over the prior SM103 epilogue ([pr-flashinfer-4063](../../sources/prs/flashinfer/PR-4063.md)).

```cpp
// Conceptual SM103 no-SMEM NVFP4 epilogue differences
// - C/src alignment: 128-bit (unchanged)
// - D/out alignment: 256-bit  → enables STG.256
// - scale: mul.f32x2 pairs when arch >= SM100
using OutputAlignment = cutlass::Int<32>; // 256-bit / 8B element path dependent
```

### 3. Backend heuristics differ from B200

| Workload | SM100 (B200) often prefers | SM103 (B300) often prefers |
|---|---|---|
| FP4 dense GEMM | cuDNN (CUDA 13 + cuDNN 9.15+) | **CUTLASS** ([pr-flashinfer-2404](../../sources/prs/flashinfer/PR-2404.md)) |
| MLA decode | cute-dsl / cutlass / trtllm mix | **cute-dsl is valid and frequently fastest** ([pr-flashinfer-3888](../../sources/prs/flashinfer/PR-3888.md)) |
| Vision attention | FA4 | **triton_attn** until FA4 validated ([pr-sglang-25570](../../sources/prs/sglang/PR-25570.md)) |
| TRTLLM-gen attention | enabled | **often gated off** (hang under high concurrency) |

### 4. Dedicated scale-factor TMA warp (grouped / MoE)

SM103 CuTeDSL grouped block-scaled GEMM uses a **7-warp** layout with warp 6 as a dedicated scale TMA warp, separate from A/B TMA ([pr-cutlass-3124](../../sources/prs/cutlass/PR-3124.md)):

```
Warp roles (SM103 grouped blockscaled GEMM sketch)
  warps 0-3 : MMA / compute
  warp 4    : epilogue / other
  warp 5    : TMA A/B
  warp 6    : TMA scale factors   <-- SM103 Ultra blockscaled pattern
  + StaticPersistentGroupTileScheduler for multi-shape groups
```

### 5. 2-SM cluster accounting

For 2-SM MLA, budget split-KV by **cluster slots** (`sm_count / (B * cluster_ctas)`), not `sm_count / B`. Over-splitting doubles LSE reduction work ([pr-flashinfer-3888](../../sources/prs/flashinfer/PR-3888.md)).

### 6. Feature-guard portability

`sm_100a`-only guards on FP4 `cvt` / quant kernels **fail on sm_103a**. Use family macros covering both ([pr-sglang-9807](../../sources/prs/sglang/PR-9807.md)):

```cpp
// Bad: SM100a-only
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && (__CUDA_ARCH__ < 1030)
// ... e2m1 cvt ...
#endif

// Better: CUTLASS / family checks that include SM103a
// (see SGLang nvfp4_quant.cuh approach in pr-sglang-9807)
```

### 7. Forward-compat is not free

TRTLLM attention kernels that "work on SM10x" have hung on GB300 at large batch (99% SM util, 0% DRAM bandwidth). Frameworks gate TRTLLM attention to **exact SM100** and fall back on SM103 ([pr-vllm-38730](../../sources/prs/vllm/PR-38730.md), [pr-sglang-21906](../../sources/prs/sglang/PR-21906.md)).

## Attention Perf Snapshot (B300)

FlashInfer Blackwell CUTLASS FMHA on B300 / CUDA 13 ([pr-flashinfer-1850](../../sources/prs/flashinfer/PR-1850.md)):

| Config | Peak TFLOP/s (non-causal) |
|---|---:|
| head_dim=128, long seq | ~1658 |
| head_dim=64, long seq | ~872 |

## Compile / Toolchain

```bash
# CUDA 13+ recommended for full SM103
nvcc -arch=sm_103a ...          # arch-accelerated features
# or multi-arch serving wheels
nvcc -gencode arch=compute_100,code=sm_100 \
     -gencode arch=compute_103,code=sm_103 ...

# Triton: older bundled ptxas may lack sm_103a — point at system CUDA
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
```

CUTLASS timeline: SM103 blockscaled Ultra FP4 in **4.2.0**; GB300 batched FP4 Ultra in **4.4.0**; CuTeDSL SM103 grouped blockscaled GEMM in **2026-07** ([pr-cutlass-3124](../../sources/prs/cutlass/PR-3124.md)). Changelog: [doc-cutlass-changelog-sm100](../../sources/docs/cutlass-changelog-sm100.md).

## Checklist: Shipping a Kernel on GB300

```
[ ] Build/test with sm_103 / sm_103a (not only sm_100a)
[ ] Include SM103 in feature guards for FP4 cvt / quant
[ ] Provide SM103 tile/cluster/schedule candidates for NVFP4 GEMM
[ ] Consider 256-bit epilogue stores + f32x2 scale fusion
[ ] Separate TMA warp for block scales on Ultra blockscaled / MoE
[ ] Autotune backend per arch (CUTLASS vs cuDNN vs cute-dsl)
[ ] For 2-SM kernels, schedule by cluster slots
[ ] Gate non-forward-compat cubins (TRTLLM-gen attention) to exact SM100
[ ] Re-validate FA4 / MLA / MoE paths — do not assume SM100 defaults
```

## Related

- [migration-sm100-to-sm103](../migration/sm100-to-sm103.md) — porting guide
- [hw-nvfp4](nvfp4.md) — NVFP4 format
- [kernel-nvfp4-gemm](../kernels/nvfp4-gemm.md) — NVFP4 GEMM case study
- [hw-tcgen05-mma](tcgen05-mma.md) — shared MMA model
