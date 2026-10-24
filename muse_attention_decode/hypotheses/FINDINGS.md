# muse_attention_decode findings

Regenerated 2026-10-24T141733Z by `churn.py index`. Read this before writing a hypothesis for muse_attention_decode.

## Current best

1.2931x [1.2737,1.3212] from job `295727fc68e14093` (check), master commit `b8a0e4db5c`, hypothesis sliding-fused-triton-decode1-gate

Where the time goes (per scored shape, top kernels):

- batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_1 page=1: yours 13.1µs vs SGLang 20.7µs (SoL 0.1µs) · _sliding_fused_kernel 10.5µs 100.0%
- batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_1 page=1: yours 27.0µs vs SGLang 30.9µs (SoL 4.3µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 14.8µs 60.7%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 7.2µs 29.5%, _pre_attention_kernel 1.3µs 5.5%
- batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_16 page=1 spread=agentic: yours 20.7µs vs SGLang 26.8µs (SoL 1.6µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 9.9µs 55.2%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 3.2µs 17.9%, void flash::prepare_varlen_num_blocks_kernel<1, false>(in... 1.9µs 10.4%
- batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_16 page=1 spread=agentic: yours 156.8µs vs SGLang 200.3µs (SoL 63.2µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 150.4µs 93.2%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 4.4µs 2.7%, void flash::prepare_varlen_num_blocks_kernel<1, true>(int... 3.0µs 1.9%
- batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_32 page=1 spread=agentic: yours 24.9µs vs SGLang 31.0µs (SoL 3.3µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 13.3µs 62.0%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 3.3µs 15.4%, _pre_attention_kernel 1.9µs 8.7%
- batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_32 page=1 spread=agentic: yours 298.5µs vs SGLang 382.9µs (SoL 126.4µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 292.0µs 96.0%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 4.8µs 1.6%, void flash::prepare_varlen_num_blocks_kernel<2, true>(int... 3.4µs 1.1%
- batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_1 page=1 q_len=8: yours 17.0µs vs SGLang 22.6µs (SoL 0.1µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 7.7µs 48.0%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 3.7µs 23.1%, void flash::prepare_varlen_num_blocks_kernel<1, false>(in... 1.8µs 11.2%
- batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_1 page=1 q_len=8: yours 29.7µs vs SGLang 35.8µs (SoL 4.3µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 14.9µs 53.1%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 7.7µs 27.4%, void flash::prepare_varlen_num_blocks_kernel<1, true>(int... 2.8µs 9.9%
- batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_16 page=1 q_len=8 spread=agentic: yours 22.6µs vs SGLang 29.1µs (SoL 1.9µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 10.7µs 51.5%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 4.2µs 20.2%, _pre_attention_kernel 2.7µs 13.0%
- batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_16 page=1 q_len=8 spread=agentic: yours 163.1µs vs SGLang 204.1µs (SoL 63.4µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 152.5µs 91.3%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 6.3µs 3.8%, void flash::prepare_varlen_num_blocks_kernel<1, true>(int... 3.2µs 1.9%
- batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_32 page=1 q_len=8 spread=agentic: yours 28.6µs vs SGLang 34.3µs (SoL 3.7µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 15.1µs 56.4%, _pre_attention_kernel 4.1µs 15.3%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 4.1µs 15.2%
- batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_32 page=1 q_len=8 spread=agentic: yours 302.2µs vs SGLang 387.6µs (SoL 126.8µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 292.7µs 94.1%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 6.8µs 2.2%, _pre_attention_kernel 4.1µs 1.3%

## By area

### FA3 call (head packing)
- **fa3-hdim128-head-pairing** → gain (gain), 1.1706x — gained on full decode_16, decode_32, verify_16, verify_32; lost on full decode_1, verify_1. On full-attention layers only (config window == 0, decode and verify) call flash_attn_with_kvcache on a head-paired view of the cache so each token's two KV hea

### FA3 call (head packing) gate
- **paired-heads-launch-cleanup-ab-tune-batch-gate** → neutral (unproven-vs-best), 1.2686x — gained on full decode_1, verify_1. From the parent's branch (paired-heads-launch-cleanup built on master), change only the pairing predicate to paired = (window == 0 and hkv % 2 == 0 and batch > 

### FA3 call (schedule, off critical path)
- **fa3-scheduler-metadata-side-stream** → regression (regression), 1.2245x — lost on sliding decode_1, decode_16, decode_32, verify_1, verify_16, verify_32; full verify_1. Wherever master calls flash_attn_with_kvcache with scheduler_metadata=None (every sliding shape, full paired shapes at batch > 1, full verify_1), FA3 runs flash

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
- queue: fa3-drop-cu-seqlens-decode [FA3 call] p5 hold
- queue: fa3-scheduler-metadata-full-verify [FA3 call] p4 hold
- queue: triton-flash-decode-full-both-heads [FA3 call] p5 hold
- queue: fused-sliding-preattn-into-window-kernel [attention boundary] p4 hold
- queue: fused-sliding-normgate-epilogue [output norm-gate] p5 hold
- queue: fa3-batched-q-verify [FA3 call (q layout)] p4 hold
- queue: fa3-num-splits-full-verify [FA3 call] p5 hold
- queue: fa3-single-split-sliding [FA3 call] p4 hold
- queue: fused-full-combine-normgate [output norm-gate] p5 hold
- queue: stream-fork-preattention [launch overhead] p4 hold
- queue: triton-flash-decode-deep-pipeline [FA3 call (Triton pipeline)] p4 hold
- queue: triton-flash-decode-full-verify [FA3 call] p4 hold
- queue: full-fused-triton-decode1-gate [full-layer attention (batch-1 decode gate)] p3 queued
- queue: sliding-fused-triton-decode1-splits-32 [sliding-layer attention (batch-1 decode, split depth)] p4 queued
- running: full-decode-cuda-gather-splitkv [FA3 call (full-layer decode gather)] p1 running
- running: sliding-fused-triton-decode-batched-deep-splits [sliding-layer attention (batched decode, split depth)] p2 running
