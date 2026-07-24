---
id: doc-blackwell-ultra-sm103
title: "NVIDIA Blackwell Ultra (SM103 / B300 / GB300) Hardware Overview"
url: https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/
source_category: official-doc
architectures: [sm103, sm103a, sm100]
tags: [nvfp4, fp4, fp8, fp6, block-scale, tcgen05, tmem, tma, clc, attention, gemm, moe]
retrieved_at: 2026-07-23
---

# NVIDIA Blackwell Ultra (SM103 / B300 / GB300)

## Overview

Blackwell Ultra is NVIDIA's inference-weighted follow-on to base Blackwell (SM100 / B200 / GB200). The GPU compute capability is **SM103 (CC 10.3)**. Product SKUs:

| SKU | Form | Role |
|---|---|---|
| **B300** | SXM GPU | Datacenter GPU (Blackwell Ultra die) |
| **GB300** | Grace + B300 Superchip | CPU–GPU coherent pairing via NVLink-C2C |
| **GB300 NVL72** | Rack scale | 72× B300 + 36× Grace, NVLink 5 fabric |

SM103 is **not** a new instruction set family relative to SM100: it keeps tcgen05, TMEM, TMA, CLC, and NVFP4 block-scaled MMA. The ultra SKU changes **peak rates, memory capacity, SFU throughput, and recommended tile/schedule specializations**.

## Hardware Specs (peak / flagship SKU)

| Feature | B200 (SM100) | B300 / GB300 (SM103) | Delta |
|---|---|---|---|
| Compute capability | 10.0 | **10.3** | — |
| SMs (flagship) | ~148 | **up to 160** | more SMs |
| HBM3e capacity | ~192 GB | **up to 288 GB** | +50% memory |
| Dense NVFP4 peak | ~10 PFLOPS | **~15 PFLOPS** | **1.5×** NVFP4 |
| NVFP4 / BF16 ratio (spec) | ~4× | **~6×** | higher FP4 advantage |
| Tensor cores / SM | 4 (5th gen) | 4 (5th gen + 2nd-gen TE) | same family |
| TMEM / SM | 256 KB | **256 KB** | same programming model |
| SFU / attention path | baseline | **~2× SFU** + accelerated softmax | attention-friendly |
| NVLink | NVLink 5 | NVLink 5 (1.8 TB/s/GPU) | same gen |
| Die interconnect | dual-die NV-HBI | dual-die NV-HBI (~10 TB/s) | same approach |

Numbers above summarize NVIDIA's public Blackwell Ultra materials (flagship SKUs; exact SM/HBM counts vary by product).

## Kernel-Programming Implications

### 1. Same PTX surface, different optima

- Compile targets: `sm_103` / `sm_103a` (CUDA 13+).
- `sm_100a`-only feature guards break on SM103a — use family checks that include both `sm_100a` and `sm_103a` (see SGLang fp4 quant fix `pr-sglang-9807`).
- Forward-compat is **not** free: some SM100 cubins / TRT-LLM-gen attention paths hang or misbehave on SM103 under high concurrency (`pr-vllm-38730`, `pr-sglang-21906`).

### 2. NVFP4 is the headline workload

- Spec peak favors NVFP4 more aggressively than B200 (6× vs BF16 vs ~4×).
- Production stacks ship **SM103-specific** CUTLASS tile shapes, cluster shapes, and schedulers (`pr-flashinfer-2303`) and later epilogue Store256 + `mul.f32x2` paths (`pr-flashinfer-4063`).
- Backend heuristics diverge: on SM103 CUTLASS often beats cuDNN for FP4 GEMM; on SM100 cuDNN can still win (`pr-flashinfer-2404`).

### 3. Larger tiles and dedicated scale warps

- SM103 dense block-scaled kernels commonly use larger K tiles (e.g. native `CTA_K=768` in FlashInfer SM103 NVFP4) and a **dedicated scale-factor TMA warp** separate from A/B TMA warps (CUTLASS CuTeDSL SM103 grouped blockscaled GEMM, `pr-cutlass-3124`).

### 4. Attention / MLA

- Doubled SFU helps softmax-heavy attention; B300 FMHA has published 1600+ TFLOPS/s non-causal head_dim=128 configs (`pr-flashinfer-1850`).
- Cute-DSL MLA decode is valid and often **fastest** on SM103 — do not omit it from SM103 backend tables (`pr-flashinfer-3888`).
- 2-SM cluster MLA must budget split-KV by **cluster slots**, not raw SM count (same PR).

### 5. Memory capacity changes host policy

- 288 GB HBM enables larger single-GPU footprints (e.g. DeepSeek-V3.1-NVFP4 on one GB300 with heavy CPU offload still used in early vLLM bring-up, `pr-vllm-30484`).

## Software Timeline (kernel-relevant)

| Date | Event |
|---|---|
| 2025-08/09 | SGLang enables SM100 FP8/FP4 kernels on SM103; FlashInfer initial SM103/110/120/121 support |
| 2025-09 | CUTLASS 4.2.0: SM103 blockscaled Ultra FP4 dense/grouped examples |
| 2025-11/12 | FlashInfer SM103 MoE DSL; vLLM initial SM103 support |
| 2026-01/02 | FlashInfer SM103 NVFP4 schedulers; CUTLASS 4.4.0 GB300 batched FP4 Ultra GEMM |
| 2026-04 | TRTLLM attention hang workarounds on (G)B300 (exact-SM100 gating) |
| 2026-05 | SGLang defaults vision attn to triton on B300 (FA4 not yet validated) |
| 2026-07 | CUTLASS CuTeDSL SM103 grouped blockscaled GEMM; FlashInfer MLA cute-dsl on SM103; SM103 NVFP4 Store256 epilogue |

## Related Wiki Pages

- `hw-sm103-blackwell-ultra` — synthesized hardware page
- `migration-sm100-to-sm103` — porting checklist SM100 → SM103
- `hw-nvfp4`, `kernel-nvfp4-gemm` — NVFP4 format and GEMM
- `hw-tcgen05-mma`, `hw-tmem` — shared instruction/memory model
