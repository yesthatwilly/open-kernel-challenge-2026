# muse_prefill_layer findings

Regenerated 2026-10-24T015224Z by `churn.py index`. Read this before writing a hypothesis for muse_prefill_layer.

## Current best

none recorded yet (run `churn.py best-init`).

## By area

## Open follow-ups

- none

## Do not retry (rejected / duplicates)

- none

## Queue and in flight

- queue: experts-dequant-grouped-mm-cutlass [w13 marlin + w2 marlin (GEMM engine: CUTLASS grouped GEMM instead of Triton)] p4 queued
- queue: experts-dequant-overlap-attention [w13 marlin + w2 marlin (dequant placement on a side stream)] p3 queued
- queue: experts-dequant-threshold-turn4 [w13 marlin + w2 marlin (Marlin vs dequant crossover)] p4 queued
- queue: experts-dequant-triton-moe-16k [w13 marlin + w2 marlin (expert grouped GEMM engine at 16K tokens)] p2 queued
- queue: experts-triton-w4a16-inkernel-fp4 [w13 marlin + w2 marlin (W4A16 engine with in-kernel NVFP4 dequant, for the turn shapes)] p5 queued
- queue: fa3-batched-mode [FA3 call] p4 queued
- queue: fa3-contiguous-kv-gather [FA3 call] p3 queued
- queue: moe-align-side-stream [MoE align/dispatch] p2 queued
- queue: qk-norm-fp8-cache-store-fusion [pre-attention norm+rope + cache store] p4 queued
- queue: sglang-pipeline-parity [whole-layer parity (SGLang pipeline) + routing remap] p1 queued
