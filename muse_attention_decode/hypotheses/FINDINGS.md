# muse_attention_decode findings

Regenerated 2026-10-24T103713Z by `churn.py index`. Read this before writing a hypothesis for muse_attention_decode.

## Current best

1.1706x [1.1530,1.1924] from job `facc479992e24a35` (check), master commit `86dc7133cb`, hypothesis fa3-hdim128-head-pairing

Where the time goes (per scored shape, top kernels):

- batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_1 page=1: yours 16.6µs vs SGLang 20.7µs (SoL 0.1µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 7.8µs 51.1%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 2.5µs 16.1%, _pre_attention_kernel 2.2µs 14.2%
- batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_1 page=1: yours 38.7µs vs SGLang 31.7µs (SoL 4.3µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 13.2µs 36.5%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 10.3µs 28.5%, void at::native::elementwise_kernel<128, 4, at::native::g... 3.5µs 9.6%
- batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_16 page=1 spread=agentic: yours 20.8µs vs SGLang 26.8µs (SoL 1.6µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 9.7µs 55.5%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 3.1µs 17.9%, void flash::prepare_varlen_num_blocks_kernel<1, false>(in... 1.8µs 10.3%
- batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_16 page=1 spread=agentic: yours 165.2µs vs SGLang 200.0µs (SoL 63.2µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 147.7µs 88.0%, void at::native::elementwise_kernel<128, 4, at::native::g... 4.4µs 2.6%, void at::native::elementwise_kernel<128, 4, at::native::g... 4.4µs 2.6%
- batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_32 page=1 spread=agentic: yours 25.5µs vs SGLang 31.7µs (SoL 3.3µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 13.2µs 62.3%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 3.2µs 15.0%, _pre_attention_kernel 1.8µs 8.3%
- batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_32 page=1 spread=agentic: yours 307.0µs vs SGLang 382.6µs (SoL 126.4µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 286.1µs 92.9%, void at::native::elementwise_kernel<128, 4, at::native::g... 4.6µs 1.5%, void at::native::elementwise_kernel<128, 4, at::native::g... 4.5µs 1.5%
- batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_1 page=1 q_len=8: yours 17.4µs vs SGLang 22.7µs (SoL 0.1µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 7.9µs 48.5%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 3.8µs 23.7%, void flash::prepare_varlen_num_blocks_kernel<1, false>(in... 1.8µs 11.2%
- batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_1 page=1 q_len=8: yours 46.0µs vs SGLang 36.4µs (SoL 4.3µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 18.3µs 40.4%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 12.5µs 27.6%, void at::native::elementwise_kernel<128, 4, at::native::g... 4.3µs 9.4%
- batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_16 page=1 q_len=8 spread=agentic: yours 22.8µs vs SGLang 29.2µs (SoL 1.9µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 10.5µs 53.9%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 3.1µs 16.1%, _pre_attention_kernel 2.7µs 13.8%
- batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_16 page=1 q_len=8 spread=agentic: yours 172.9µs vs SGLang 204.8µs (SoL 63.4µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 158.0µs 86.8%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 6.0µs 3.3%, void at::native::elementwise_kernel<128, 4, at::native::g... 4.9µs 2.7%
- batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_32 page=1 q_len=8 spread=agentic: yours 28.8µs vs SGLang 34.2µs (SoL 3.7µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 14.9µs 56.5%, _pre_attention_kernel 4.1µs 15.6%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 4.1µs 15.4%
- batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_32 page=1 q_len=8 spread=agentic: yours 309.4µs vs SGLang 388.8µs (SoL 126.8µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 286.8µs 91.2%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 7.0µs 2.2%, void at::native::elementwise_kernel<128, 4, at::native::g... 4.9µs 1.6%

## By area

### FA3 call (head packing)
- **fa3-hdim128-head-pairing** → gain (gain), 1.1706x — gained on full decode_16, decode_32, verify_16, verify_32; lost on full decode_1, verify_1. On full-attention layers only (config window == 0, decode and verify) call flash_attn_with_kvcache on a head-paired view of the cache so each token's two KV hea

### pre-attention norm+rope
- **fused-preattention-triton** → gain (gain), 1.1274x — gained on sliding decode_1, decode_16, decode_32, verify_1, verify_16, verify_32; full decode_1, decode_16, decode_32, verify_1, verify_16, verify_32. Replace the five pre-attention launches of the parity pipeline (FusedQKNorm (+RoPE), k .to(FP8), v .to(FP8), store_cache, q .to(FP8)) with one Triton kernel, gr

### sliding-layer attention (whole-path rewrite)
- **sliding-single-launch-fused-attention** → regression (regression), 0.9452x — gained on sliding decode_1; lost on sliding decode_16, decode_32, verify_1, verify_16, verify_32. Stage 1 of a rewrite of the sliding-layer path. Built on hyp/muse_attention_decode/fused-preattention-triton (1.127x, job dfee6f2939974bad). On sliding layers (

## Open follow-ups

- none

## Do not retry (rejected / duplicates)

- none

## Queue and in flight

- queue: sglang-path-parity [SGLang path parity] p1 hold
- queue: fa3-num-splits-full-decode [FA3 call] p4 queued
- queue: fa3-num-splits-sliding [FA3 call] p3 hold
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
