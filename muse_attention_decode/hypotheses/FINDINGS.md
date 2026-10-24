# muse_attention_decode findings

Regenerated 2026-10-24T132213Z by `churn.py index`. Read this before writing a hypothesis for muse_attention_decode.

## Current best

1.2686x [1.2496,1.2877] from job `bd64bf9f9e904f02` (check), master commit `bfb7bac32c`, hypothesis paired-heads-launch-cleanup-ab-tune-batch-gate

Where the time goes (per scored shape, top kernels):

- batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_1 page=1: yours 16.6µs vs SGLang 20.8µs (SoL 0.1µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 8.2µs 52.3%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 2.6µs 16.3%, _pre_attention_kernel 2.1µs 13.4%
- batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_1 page=1: yours 28.1µs vs SGLang 31.6µs (SoL 4.3µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 14.5µs 60.3%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 7.0µs 29.4%, _pre_attention_kernel 1.4µs 5.9%
- batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_16 page=1 spread=agentic: yours 20.9µs vs SGLang 27.1µs (SoL 1.6µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 9.9µs 55.5%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 3.2µs 17.8%, _pre_attention_kernel 1.8µs 10.1%
- batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_16 page=1 spread=agentic: yours 156.2µs vs SGLang 200.8µs (SoL 63.2µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 150.4µs 93.3%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 4.1µs 2.6%, void flash::prepare_varlen_num_blocks_kernel<1, true>(int... 3.1µs 1.9%
- batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_32 page=1 spread=agentic: yours 25.0µs vs SGLang 31.4µs (SoL 3.3µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 13.3µs 61.8%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 3.1µs 14.4%, _pre_attention_kernel 2.2µs 10.2%
- batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_32 page=1 spread=agentic: yours 298.5µs vs SGLang 384.6µs (SoL 126.4µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 291.2µs 96.3%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 3.8µs 1.2%, void flash::prepare_varlen_num_blocks_kernel<2, true>(int... 3.5µs 1.1%
- batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_1 page=1 q_len=8: yours 17.2µs vs SGLang 23.1µs (SoL 0.1µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 7.9µs 48.5%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 3.8µs 23.7%, void flash::prepare_varlen_num_blocks_kernel<1, false>(in... 1.7µs 10.7%
- batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_1 page=1 q_len=8: yours 30.2µs vs SGLang 36.6µs (SoL 4.3µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 15.1µs 54.1%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 7.4µs 26.6%, void flash::prepare_varlen_num_blocks_kernel<1, true>(int... 2.8µs 9.9%
- batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_16 page=1 q_len=8 spread=agentic: yours 22.9µs vs SGLang 29.5µs (SoL 1.9µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 10.8µs 52.5%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 3.9µs 18.9%, _pre_attention_kernel 2.8µs 13.3%
- batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_16 page=1 q_len=8 spread=agentic: yours 163.3µs vs SGLang 206.0µs (SoL 63.4µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 152.1µs 91.0%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 6.8µs 4.1%, void flash::prepare_varlen_num_blocks_kernel<1, true>(int... 2.9µs 1.7%
- batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_32 page=1 q_len=8 spread=agentic: yours 28.9µs vs SGLang 34.7µs (SoL 3.7µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 15.6µs 57.5%, _pre_attention_kernel 4.2µs 15.3%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 4.1µs 15.1%
- batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_32 page=1 q_len=8 spread=agentic: yours 301.7µs vs SGLang 390.0µs (SoL 126.8µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 286.7µs 93.9%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 6.9µs 2.3%, _pre_attention_kernel 4.2µs 1.4%

## By area

### FA3 call (head packing)
- **fa3-hdim128-head-pairing** → gain (gain), 1.1706x — gained on full decode_16, decode_32, verify_16, verify_32; lost on full decode_1, verify_1. On full-attention layers only (config window == 0, decode and verify) call flash_attn_with_kvcache on a head-paired view of the cache so each token's two KV hea

### FA3 call (head packing) gate
- **paired-heads-launch-cleanup-ab-tune-batch-gate** → neutral (unproven-vs-best), 1.2686x — gained on full decode_1, verify_1. From the parent's branch (paired-heads-launch-cleanup built on master), change only the pairing predicate to paired = (window == 0 and hkv % 2 == 0 and batch > 

### attention boundary
- **paired-heads-launch-cleanup** → gain (gain), 1.2436x — gained on full decode_1, decode_16, decode_32, verify_1, verify_16. Stacked on fa3-hdim128-head-pairing (and using fused-preattention-triton's kernel, which must have landed too): remove the five pad/un-pad launches the pairing 

### page table (sliding)
- **fa3-window-trimmed-page-table** → regression (regression), 1.1411x — lost on sliding decode_1, decode_16, decode_32, verify_1, verify_16, verify_32. On sliding-window layers only (config window != 0) hand FA3 a window-trimmed view of each request instead of the whole 80K-token context. Let W = window - 1 + L

### pre-attention norm+rope
- **fused-preattention-triton** → gain (gain), 1.1274x — gained on sliding decode_1, decode_16, decode_32, verify_1, verify_16, verify_32; full decode_1, decode_16, decode_32, verify_1, verify_16, verify_32. Replace the five pre-attention launches of the parity pipeline (FusedQKNorm (+RoPE), k .to(FP8), v .to(FP8), store_cache, q .to(FP8)) with one Triton kernel, gr
- **preattention-group-tiled-programs** → neutral (within-ci), 1.2614x — gained on sliding verify_32; lost on sliding decode_1. Regrid master _pre_attention_kernel from one 1-warp program per (token, head) ((tokens, Hq+Hkv) = 18 programs per token, 4608 at verify_32) to one program per (

### sliding-layer attention (batch-1 decode gate)
- **sliding-fused-triton-decode1-gate** → neutral (unproven-vs-best), 1.2931x — gained on sliding decode_1. Port _sliding_fused_kernel and its helpers (_counters, _num_splits, _norm_rope, _gpu_fence, _online_update, _gate_epilogue, LOG2E, _sliding_forward) verbatim fr

### sliding-layer attention (whole-path rewrite)
- **sliding-single-launch-fused-attention** → regression (regression), 0.9452x — gained on sliding decode_1; lost on sliding decode_16, decode_32, verify_1, verify_16, verify_32. Stage 1 of a rewrite of the sliding-layer path. Built on hyp/muse_attention_decode/fused-preattention-triton (1.127x, job dfee6f2939974bad). On sliding layers (

## Open follow-ups

- paired-heads-launch-cleanup-ab-tune-batch-gate: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: paired-heads-launch-cleanup-ab-tune-batch-gate`) that adds a further change in the same area is the only way to learn more (identi
- sliding-fused-triton-decode1-gate: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: sliding-fused-triton-decode1-gate`) that adds a further change in the same area is the only way to learn more (identical code is n
- preattention-group-tiled-programs: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: preattention-group-tiled-programs`) that adds a further change in the same area is the only way to learn more (identical code is never re

## Do not retry (rejected / duplicates)

- none

## Queue and in flight

- queue: sglang-path-parity [SGLang path parity] p1 hold
- queue: fa3-num-splits-full-decode [FA3 call] p4 hold
- queue: fa3-num-splits-sliding [FA3 call] p3 hold
- queue: triton-window-attention-sliding [FA3 call] p5 hold
- queue: fa3-drop-cu-seqlens-decode [FA3 call] p5 queued
- queue: fa3-scheduler-metadata-full-verify [FA3 call] p4 hold
- queue: triton-flash-decode-full-both-heads [FA3 call] p5 hold
- queue: fused-sliding-preattn-into-window-kernel [attention boundary] p4 hold
- queue: fused-sliding-normgate-epilogue [output norm-gate] p5 hold
- queue: fa3-batched-q-verify [FA3 call (q layout)] p4 hold
- queue: fa3-num-splits-full-verify [FA3 call] p5 hold
- queue: fa3-single-split-sliding [FA3 call] p4 queued
- queue: fused-full-combine-normgate [output norm-gate] p5 hold
- queue: stream-fork-preattention [launch overhead] p4 hold
- queue: triton-flash-decode-deep-pipeline [FA3 call (Triton pipeline)] p4 hold
- queue: triton-flash-decode-full-verify [FA3 call] p4 hold
- running: fa3-scheduler-metadata-side-stream [FA3 call (schedule, off critical path)] p2 running
