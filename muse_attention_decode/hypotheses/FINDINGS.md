# muse_attention_decode findings

Regenerated 2026-10-24T115933Z by `churn.py index`. Read this before writing a hypothesis for muse_attention_decode.

## Current best

1.2436x [1.2250,1.2623] from job `2846122f764043da` (check), master commit `fa95f6d9e5`, hypothesis paired-heads-launch-cleanup

Where the time goes (per scored shape, top kernels):

- batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_1 page=1: yours 16.3µs vs SGLang 20.5µs (SoL 0.1µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 8.2µs 54.5%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 2.5µs 16.8%, void flash::prepare_varlen_num_blocks_kernel<1, false>(in... 1.8µs 12.1%
- batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_1 page=1: yours 30.5µs vs SGLang 31.3µs (SoL 4.3µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 13.2µs 45.9%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 10.5µs 36.3%, void flash::prepare_varlen_num_blocks_kernel<1, true>(int... 2.8µs 9.5%
- batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_16 page=1 spread=agentic: yours 20.6µs vs SGLang 27.0µs (SoL 1.6µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 9.9µs 55.2%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 3.2µs 18.1%, void flash::prepare_varlen_num_blocks_kernel<1, false>(in... 1.8µs 10.2%
- batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_16 page=1 spread=agentic: yours 155.9µs vs SGLang 200.8µs (SoL 63.2µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 147.4µs 92.6%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 4.8µs 3.0%, void flash::prepare_varlen_num_blocks_kernel<1, true>(int... 3.1µs 2.0%
- batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_32 page=1 spread=agentic: yours 25.1µs vs SGLang 31.2µs (SoL 3.3µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 13.3µs 62.1%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 3.3µs 15.3%, _pre_attention_kernel 1.8µs 8.4%
- batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_32 page=1 spread=agentic: yours 298.0µs vs SGLang 385.5µs (SoL 126.4µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 291.8µs 96.3%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 4.0µs 1.3%, void flash::prepare_varlen_num_blocks_kernel<2, true>(int... 3.4µs 1.1%
- batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_1 page=1 q_len=8: yours 17.2µs vs SGLang 22.9µs (SoL 0.1µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 7.7µs 48.1%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 3.8µs 23.9%, void flash::prepare_varlen_num_blocks_kernel<1, false>(in... 1.8µs 10.9%
- batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_1 page=1 q_len=8: yours 36.4µs vs SGLang 36.2µs (SoL 4.3µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 18.1µs 50.2%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 12.4µs 34.4%, void flash::prepare_varlen_num_blocks_kernel<1, true>(int... 2.8µs 7.7%
- batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_16 page=1 q_len=8 spread=agentic: yours 22.4µs vs SGLang 29.0µs (SoL 1.9µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 10.9µs 52.0%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 4.2µs 19.9%, _pre_attention_kernel 2.7µs 12.9%
- batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_16 page=1 q_len=8 spread=agentic: yours 163.5µs vs SGLang 205.1µs (SoL 63.4µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 156.0µs 91.4%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 6.2µs 3.6%, void flash::prepare_varlen_num_blocks_kernel<1, true>(int... 3.2µs 1.9%
- batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_32 page=1 q_len=8 spread=agentic: yours 29.3µs vs SGLang 34.9µs (SoL 3.7µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 15.6µs 56.6%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 4.2µs 15.4%, _pre_attention_kernel 4.2µs 15.3%
- batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_32 page=1 q_len=8 spread=agentic: yours 302.6µs vs SGLang 390.2µs (SoL 126.8µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 290.1µs 93.7%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 7.7µs 2.5%, _pre_attention_kernel 4.1µs 1.3%

## By area

### FA3 call (head packing)
- **fa3-hdim128-head-pairing** → gain (gain), 1.1706x — gained on full decode_16, decode_32, verify_16, verify_32; lost on full decode_1, verify_1. On full-attention layers only (config window == 0, decode and verify) call flash_attn_with_kvcache on a head-paired view of the cache so each token's two KV hea

### FA3 call (head packing) gate
- **paired-heads-launch-cleanup-ab-tune-batch-gate** → neutral (unproven-vs-best), 1.2686x — gained on full decode_1, verify_1. From the parent's branch (paired-heads-launch-cleanup built on master), change only the pairing predicate to paired = (window == 0 and hkv % 2 == 0 and batch > 

### attention boundary
- **paired-heads-launch-cleanup** → gain (gain), 1.2436x — gained on full decode_1, decode_16, decode_32, verify_1, verify_16. Stacked on fa3-hdim128-head-pairing (and using fused-preattention-triton's kernel, which must have landed too): remove the five pad/un-pad launches the pairing 

### pre-attention norm+rope
- **fused-preattention-triton** → gain (gain), 1.1274x — gained on sliding decode_1, decode_16, decode_32, verify_1, verify_16, verify_32; full decode_1, decode_16, decode_32, verify_1, verify_16, verify_32. Replace the five pre-attention launches of the parity pipeline (FusedQKNorm (+RoPE), k .to(FP8), v .to(FP8), store_cache, q .to(FP8)) with one Triton kernel, gr

### sliding-layer attention (whole-path rewrite)
- **sliding-single-launch-fused-attention** → regression (regression), 0.9452x — gained on sliding decode_1; lost on sliding decode_16, decode_32, verify_1, verify_16, verify_32. Stage 1 of a rewrite of the sliding-layer path. Built on hyp/muse_attention_decode/fused-preattention-triton (1.127x, job dfee6f2939974bad). On sliding layers (

## Open follow-ups

- paired-heads-launch-cleanup-ab-tune-batch-gate: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: paired-heads-launch-cleanup-ab-tune-batch-gate`) that adds a further change in the same area is the only way to learn more (identi

## Do not retry (rejected / duplicates)

- none

## Queue and in flight

- queue: sglang-path-parity [SGLang path parity] p1 hold
- queue: fa3-num-splits-full-decode [FA3 call] p4 hold
- queue: fa3-num-splits-sliding [FA3 call] p3 hold
- queue: triton-window-attention-sliding [FA3 call] p5 queued
- queue: fa3-drop-cu-seqlens-decode [FA3 call] p5 queued
- queue: fa3-scheduler-metadata-full-verify [FA3 call] p4 queued
- queue: triton-flash-decode-full-both-heads [FA3 call] p5 queued
- queue: fused-sliding-preattn-into-window-kernel [attention boundary] p4 queued
- queue: fused-sliding-normgate-epilogue [output norm-gate] p5 queued
- queue: fa3-batched-q-verify [FA3 call (q layout)] p4 queued
- queue: fa3-num-splits-full-verify [FA3 call] p5 queued
- queue: fa3-single-split-sliding [FA3 call] p4 queued
- queue: fused-full-combine-normgate [output norm-gate] p5 queued
- queue: stream-fork-preattention [launch overhead] p4 queued
- queue: triton-flash-decode-deep-pipeline [FA3 call (Triton pipeline)] p4 queued
- queue: triton-flash-decode-full-verify [FA3 call] p4 queued
- running: fa3-window-trimmed-page-table [page table (sliding)] p3 running
