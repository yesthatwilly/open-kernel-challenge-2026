# muse_attention_decode findings

Regenerated 2026-10-24T173236Z by `churn.py index`. Read this before writing a hypothesis for muse_attention_decode.

## Current best

1.3949x [1.3740,1.4158] from job `698d4cd3838c4a63` (submit), master commit `0a56d22c1f`, hypothesis sliding-deep-splits-on-cuda-splitkv

Where the time goes (per scored shape, top kernels):

- batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_1 page=1: yours 13.2µs vs SGLang 20.8µs (SoL 0.1µs) · _sliding_fused_kernel 10.4µs 100.0%
- batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_1 page=1: yours 27.3µs vs SGLang 31.2µs (SoL 4.3µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 14.6µs 60.5%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 7.1µs 29.5%, _pre_attention_kernel 1.3µs 5.6%
- batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_16 page=1 spread=agentic: yours 15.3µs vs SGLang 26.7µs (SoL 1.6µs) · _sliding_fused_kernel 12.9µs 100.0%
- batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_16 page=1 spread=agentic: yours 122.8µs vs SGLang 200.8µs (SoL 63.2µs) · (anonymous namespace)::splitkv_attn_kernel(unsigned char ... 111.8µs 91.5%, (anonymous namespace)::splitkv_combine_kernel(float const... 8.2µs 6.7%, _pre_attention_kernel 2.1µs 1.8%
- batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_32 page=1 spread=agentic: yours 24.2µs vs SGLang 31.2µs (SoL 3.3µs) · _sliding_fused_kernel 22.1µs 100.0%
- batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_32 page=1 spread=agentic: yours 241.9µs vs SGLang 384.0µs (SoL 126.4µs) · (anonymous namespace)::splitkv_attn_kernel(unsigned char ... 245.9µs 95.9%, (anonymous namespace)::splitkv_combine_kernel(float const... 8.4µs 3.3%, _pre_attention_kernel 2.0µs 0.8%
- batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_1 page=1 q_len=8: yours 16.7µs vs SGLang 22.7µs (SoL 0.1µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 7.6µs 47.7%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 3.9µs 24.3%, void flash::prepare_varlen_num_blocks_kernel<1, false>(in... 1.7µs 10.7%
- batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_1 page=1 q_len=8: yours 30.0µs vs SGLang 36.0µs (SoL 4.3µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 15.0µs 54.0%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 7.6µs 27.4%, void flash::prepare_varlen_num_blocks_kernel<1, true>(int... 2.7µs 9.6%
- batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_16 page=1 q_len=8 spread=agentic: yours 22.7µs vs SGLang 29.0µs (SoL 1.9µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 10.7µs 51.7%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 4.1µs 19.8%, _pre_attention_kernel 2.7µs 13.1%
- batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_16 page=1 q_len=8 spread=agentic: yours 162.0µs vs SGLang 204.5µs (SoL 63.4µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 151.1µs 91.4%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 6.0µs 3.6%, void flash::prepare_varlen_num_blocks_kernel<1, true>(int... 3.0µs 1.8%
- batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_32 page=1 q_len=8 spread=agentic: yours 28.7µs vs SGLang 34.3µs (SoL 3.7µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 14.5µs 55.0%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 4.3µs 16.4%, _pre_attention_kernel 4.1µs 15.7%
- batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_32 page=1 q_len=8 spread=agentic: yours 300.8µs vs SGLang 388.3µs (SoL 126.8µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 290.3µs 94.0%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 7.1µs 2.3%, _pre_attention_kernel 4.1µs 1.3%

## By area

### FA3 call (full-layer decode gather)
- **full-decode-cuda-gather-splitkv** → gain (gain), 1.3468x — gained on full decode_16, decode_32; lost on sliding decode_1. Stage 1 of a rewrite of full-attention decode, built on master (sliding-fused-triton-decode1-gate). Gate (config/shape only): cfg window == 0 and q_len == 1 and

### FA3 call (head packing)
- **fa3-hdim128-head-pairing** → gain (gain), 1.1706x — gained on full decode_16, decode_32, verify_16, verify_32; lost on full decode_1, verify_1. On full-attention layers only (config window == 0, decode and verify) call flash_attn_with_kvcache on a head-paired view of the cache so each token's two KV hea

### FA3 call (head packing) gate
- **paired-heads-launch-cleanup-ab-tune-batch-gate** → neutral (unproven-vs-best), 1.2686x — gained on full decode_1, verify_1. From the parent's branch (paired-heads-launch-cleanup built on master), change only the pairing predicate to paired = (window == 0 and hkv % 2 == 0 and batch > 

### FA3 call (schedule, off critical path)
- **fa3-scheduler-metadata-side-stream** → regression (regression), 1.2245x — lost on sliding decode_1, decode_16, decode_32, verify_1, verify_16, verify_32; full verify_1. Wherever master calls flash_attn_with_kvcache with scheduler_metadata=None (every sliding shape, full paired shapes at batch > 1, full verify_1), FA3 runs flash

### attention boundary
- **paired-heads-launch-cleanup** → gain (gain), 1.2436x — gained on full decode_1, decode_16, decode_32, verify_1, verify_16. Stacked on fa3-hdim128-head-pairing (and using fused-preattention-triton's kernel, which must have landed too): remove the five pad/un-pad launches the pairing 

### full-layer attention (batch-1 decode gate)
- **full-fused-triton-decode1-gate** → neutral (within-ci), 1.3417x — lost on full decode_1. Built on master. Route full-attention batch-1 decode (config window == 0 and batch == 1 and q_len == 1) through master _sliding_fused_kernel via _sliding_forwar

### page table (sliding)
- **fa3-window-trimmed-page-table** → regression (regression), 1.1411x — lost on sliding decode_1, decode_16, decode_32, verify_1, verify_16, verify_32. On sliding-window layers only (config window != 0) hand FA3 a window-trimmed view of each request instead of the whole 80K-token context. Let W = window - 1 + L

### pre-attention norm+rope
- **fused-preattention-triton** → gain (gain), 1.1274x — gained on sliding decode_1, decode_16, decode_32, verify_1, verify_16, verify_32; full decode_1, decode_16, decode_32, verify_1, verify_16, verify_32. Replace the five pre-attention launches of the parity pipeline (FusedQKNorm (+RoPE), k .to(FP8), v .to(FP8), store_cache, q .to(FP8)) with one Triton kernel, gr
- **preattention-group-tiled-programs** → neutral (within-ci), 1.2614x — gained on sliding verify_32; lost on sliding decode_1. Regrid master _pre_attention_kernel from one 1-warp program per (token, head) ((tokens, Hq+Hkv) = 18 programs per token, 4608 at verify_32) to one program per (

### sliding attention, batched decode, split depth
- **sliding-deep-splits-on-cuda-splitkv** → gain (gain), 1.3949x — gained on sliding decode_1, decode_16, decode_32, verify_1; full decode_32. Stack sliding-fused-triton-decode-batched-deep-splits (1.309x on the pre-CUDA master; sliding decode_16 gained 33%, decode_32 +3%, every other shape flat) onto 

### sliding-layer attention (batch-1 decode gate)
- **sliding-fused-triton-decode1-gate** → neutral (unproven-vs-best), 1.2931x — gained on sliding decode_1. Port _sliding_fused_kernel and its helpers (_counters, _num_splits, _norm_rope, _gpu_fence, _online_update, _gate_epilogue, LOG2E, _sliding_forward) verbatim fr

### sliding-layer attention (batch-1 decode, split depth)
- **sliding-fused-triton-decode1-splits-32** → neutral (unproven-vs-best), 1.3520x — no shape moved outside noise. Built on master. One config-only change in _sliding_forward: at batch 1 and q_len 1 (the only shape master routes to _sliding_fused_kernel) use S = 32 splits in

### sliding-layer attention (batched decode, split depth)
- **sliding-fused-triton-decode-batched-deep-splits** → neutral (unproven-vs-best), 1.3087x — gained on sliding decode_16. Built on master. Two edits, both config-only. (1) In kernel_fn widen the fused sliding gate from (window and batch == 1 and q_len == 1) to (window and q_len == 

### sliding-layer attention (whole-path rewrite)
- **sliding-single-launch-fused-attention** → regression (regression), 0.9452x — gained on sliding decode_1; lost on sliding decode_16, decode_32, verify_1, verify_16, verify_32. Stage 1 of a rewrite of the sliding-layer path. Built on hyp/muse_attention_decode/fused-preattention-triton (1.127x, job dfee6f2939974bad). On sliding layers (

## Open follow-ups

- paired-heads-launch-cleanup-ab-tune-batch-gate: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: paired-heads-launch-cleanup-ab-tune-batch-gate`) that adds a further change in the same area is the only way to learn more (identi
- sliding-fused-triton-decode1-gate: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: sliding-fused-triton-decode1-gate`) that adds a further change in the same area is the only way to learn more (identical code is n
- preattention-group-tiled-programs: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: preattention-group-tiled-programs`) that adds a further change in the same area is the only way to learn more (identical code is never re
- sliding-fused-triton-decode-batched-deep-splits: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: sliding-fused-triton-decode-batched-deep-splits`) that adds a further change in the same area is the only way to learn more (ident
- full-fused-triton-decode1-gate: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: full-fused-triton-decode1-gate`) that adds a further change in the same area is the only way to learn more (identical code is never re-ru
- sliding-fused-triton-decode1-splits-32: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: sliding-fused-triton-decode1-splits-32`) that adds a further change in the same area is the only way to learn more (identical code

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
- queue: sliding-verify-cuda-streamk-window [sliding-layer attention (verify)] p2 queued
- queue: full-verify1-cuda-streamk-batch1 [full-layer attention (batch-1 verify gate)] p2 queued
- running: full-verify-cuda-streamk-splitkv [FA3 call (full-layer verify gather)] p1 running
- running: full-decode-cuda-fp8-qk-mma [FA3 call (full-layer decode gather)] p3 running
- running: sliding-cuda-cluster-dsmem-onelaunch [sliding-layer attention (whole-path rewrite, CUDA cluster)] p1 running
- running: full-cuda-onelaunch-warpspec-bulk-gather [full-layer attention (whole-path rewrite, warp-specialized one launch)] p2 running
