# muse_attention_decode findings

Regenerated 2026-10-24T095815Z by `churn.py index`. Read this before writing a hypothesis for muse_attention_decode.

## Current best

1.1284x [1.1115,1.1454] from job `c297440e53334f34` (submit), master commit `3e98bd00dc`, hypothesis fused-preattention-triton

Where the time goes (per scored shape, top kernels):

- batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_1 page=1: yours 16.8µs vs SGLang 20.9µs (SoL 0.1µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 8.2µs 54.6%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 2.5µs 16.8%, void flash::prepare_varlen_num_blocks_kernel<1, false>(in... 1.8µs 12.1%
- batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_1 page=1: yours 27.9µs vs SGLang 31.4µs (SoL 4.3µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 14.6µs 60.6%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 7.1µs 29.6%, _pre_attention_kernel 1.3µs 5.5%
- batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_16 page=1 spread=agentic: yours 20.8µs vs SGLang 27.0µs (SoL 1.6µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 9.9µs 55.4%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 3.4µs 18.8%, void flash::prepare_varlen_num_blocks_kernel<1, false>(in... 1.8µs 10.2%
- batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_16 page=1 spread=agentic: yours 194.4µs vs SGLang 200.1µs (SoL 63.2µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 186.2µs 96.5%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 3.3µs 1.7%, _pre_attention_kernel 2.0µs 1.0%
- batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_32 page=1 spread=agentic: yours 25.1µs vs SGLang 31.4µs (SoL 3.3µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 13.1µs 61.5%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 3.0µs 14.3%, _pre_attention_kernel 2.0µs 9.2%
- batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_32 page=1 spread=agentic: yours 377.9µs vs SGLang 383.5µs (SoL 126.4µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 493.3µs 98.5%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 3.1µs 0.6%, _pre_attention_kernel 2.7µs 0.5%
- batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_1 page=1 q_len=8: yours 17.1µs vs SGLang 22.8µs (SoL 0.1µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 7.7µs 47.9%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 3.9µs 24.0%, void flash::prepare_varlen_num_blocks_kernel<1, false>(in... 1.8µs 11.1%
- batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_1 page=1 q_len=8: yours 30.0µs vs SGLang 36.4µs (SoL 4.3µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 14.9µs 54.2%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 7.3µs 26.6%, void flash::prepare_varlen_num_blocks_kernel<1, true>(int... 2.8µs 10.0%
- batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_16 page=1 q_len=8 spread=agentic: yours 22.5µs vs SGLang 29.3µs (SoL 1.9µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 10.6µs 52.1%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 3.9µs 19.2%, _pre_attention_kernel 2.7µs 13.3%
- batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_16 page=1 q_len=8 spread=agentic: yours 198.3µs vs SGLang 205.5µs (SoL 63.4µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 190.7µs 94.3%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 4.1µs 2.0%, void flash::prepare_varlen_num_blocks_kernel<1, true>(int... 3.1µs 1.5%
- batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_32 page=1 q_len=8 spread=agentic: yours 28.7µs vs SGLang 34.3µs (SoL 3.7µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 14.9µs 55.8%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 4.3µs 16.2%, _pre_attention_kernel 4.1µs 15.4%
- batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_32 page=1 q_len=8 spread=agentic: yours 383.1µs vs SGLang 388.9µs (SoL 126.8µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 375.5µs 96.5%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 4.2µs 1.1%, _pre_attention_kernel 4.1µs 1.1%

## By area

### pre-attention norm+rope
- **fused-preattention-triton** → gain (gain), 1.1274x — gained on sliding decode_1, decode_16, decode_32, verify_1, verify_16, verify_32; full decode_1, decode_16, decode_32, verify_1, verify_16, verify_32. Replace the five pre-attention launches of the parity pipeline (FusedQKNorm (+RoPE), k .to(FP8), v .to(FP8), store_cache, q .to(FP8)) with one Triton kernel, gr

## Open follow-ups

- none

## Do not retry (rejected / duplicates)

- none

## Queue and in flight

- queue: sglang-path-parity [SGLang path parity] p1 hold
- queue: fa3-num-splits-full-decode [FA3 call] p4 queued
- queue: fa3-num-splits-sliding [FA3 call] p3 queued
- queue: triton-window-attention-sliding [FA3 call] p5 queued
- queue: fa3-drop-cu-seqlens-decode [FA3 call] p5 queued
- queue: fa3-scheduler-metadata-full-verify [FA3 call] p4 queued
- queue: triton-flash-decode-full-both-heads [FA3 call] p5 queued
- queue: fused-sliding-preattn-into-window-kernel [attention boundary] p4 queued
- queue: fused-sliding-normgate-epilogue [output norm-gate] p5 queued
- queue: paired-heads-launch-cleanup [attention boundary] p3 queued
- queue: fa3-batched-q-verify [FA3 call (q layout)] p4 queued
- queue: fa3-num-splits-full-verify [FA3 call] p5 queued
- queue: fa3-single-split-sliding [FA3 call] p4 queued
- queue: fa3-window-trimmed-page-table [page table (sliding)] p3 queued
- queue: fused-full-combine-normgate [output norm-gate] p5 queued
- queue: stream-fork-preattention [launch overhead] p4 queued
- queue: triton-flash-decode-deep-pipeline [FA3 call (Triton pipeline)] p4 queued
- queue: triton-flash-decode-full-verify [FA3 call] p4 queued
- running: fa3-hdim128-head-pairing [FA3 call (head packing)] p2 running
- running: sliding-single-launch-fused-attention [sliding-layer attention (whole-path rewrite)] p1 running
