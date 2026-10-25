# muse_attention_decode findings

Regenerated 2026-10-25T151051Z by `churn.py index`. Read this before writing a hypothesis for muse_attention_decode.

## Current best

1.5017x [1.4791,1.5304] from job `252abb7f11ec48c3` (check), master commit `7e12891512`, hypothesis restack-sliding-decode-sk-window-on-cluster-master

Where the time goes (per scored shape, top kernels):

- batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_1 page=1: yours 11.9µs vs SGLang 20.3µs (SoL 0.1µs) · (anonymous namespace)::sliding_cluster_kernel(__nv_bfloat... 10.8µs 100.0%
- batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_1 page=1: yours 22.4µs vs SGLang 31.2µs (SoL 4.3µs) · void (anonymous namespace)::splitkv_attn_sk_kernel<4, tru... 19.0µs 93.0%, _pre_attention_kernel 1.4µs 7.0%
- batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_16 page=1 spread=agentic: yours 15.5µs vs SGLang 26.9µs (SoL 1.6µs) · _sliding_fused_kernel 13.0µs 100.0%
- batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_16 page=1 spread=agentic: yours 111.1µs vs SGLang 200.8µs (SoL 63.2µs) · void (anonymous namespace)::splitkv_attn_sk_kernel<4, tru... 109.0µs 98.0%, _pre_attention_kernel 2.2µs 2.0%
- batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_32 page=1 spread=agentic: yours 17.7µs vs SGLang 31.3µs (SoL 3.3µs) · void (anonymous namespace)::splitkv_attn_sk_kernel<4, tru... 14.0µs 88.3%, _pre_attention_kernel 1.9µs 11.7%
- batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_32 page=1 spread=agentic: yours 227.9µs vs SGLang 384.6µs (SoL 126.4µs) · void (anonymous namespace)::splitkv_attn_sk_kernel<4, tru... 227.7µs 99.0%, _pre_attention_kernel 2.4µs 1.0%
- batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_1 page=1 q_len=8: yours 13.2µs vs SGLang 22.6µs (SoL 0.1µs) · void (anonymous namespace)::splitkv_attn_sk_kernel<4, tru... 10.0µs 85.7%, _pre_attention_kernel 1.7µs 14.3%
- batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_1 page=1 q_len=8: yours 30.5µs vs SGLang 36.5µs (SoL 4.3µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 14.9µs 54.2%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 7.3µs 26.6%, void flash::prepare_varlen_num_blocks_kernel<1, true>(int... 2.8µs 10.0%
- batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_16 page=1 q_len=8 spread=agentic: yours 22.8µs vs SGLang 29.3µs (SoL 1.9µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 10.7µs 51.7%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 4.0µs 19.2%, _pre_attention_kernel 2.8µs 13.5%
- batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_16 page=1 q_len=8 spread=agentic: yours 163.0µs vs SGLang 205.0µs (SoL 63.4µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 154.5µs 91.7%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 5.9µs 3.5%, void flash::prepare_varlen_num_blocks_kernel<1, true>(int... 3.0µs 1.8%
- batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_32 page=1 q_len=8 spread=agentic: yours 29.2µs vs SGLang 34.7µs (SoL 3.7µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 15.0µs 56.1%, _pre_attention_kernel 4.2µs 15.7%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 4.2µs 15.6%
- batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_32 page=1 q_len=8 spread=agentic: yours 302.3µs vs SGLang 389.8µs (SoL 126.8µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 293.1µs 93.8%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 7.7µs 2.5%, _pre_attention_kernel 4.2µs 1.3%

## By area

### FA3 call (full-layer decode gather)
- **full-decode-cuda-gather-splitkv** → gain (gain), 1.3468x — gained on full decode_16, decode_32; lost on sliding decode_1. Stage 1 of a rewrite of full-attention decode, built on master (sliding-fused-triton-decode1-gate). Gate (config/shape only): cfg window == 0 and q_len == 1 and
- **full-decode-cuda-fp8-qk-mma** → neutral (within-ci), 1.3439x — gained on sliding decode_1. Built on master (full-decode-cuda-gather-splitkv). One change inside splitkv_attn_kernel (full-layer decode at 1 < batch <= 64, the only kernel touched): comput
- **splitkv-full-line-gather-mapping** → neutral (unproven-vs-best), 1.4160x — gained on full decode_16, decode_32; lost on sliding decode_1. One change inside splitkv_attn_kernel's issue_tile lambda (master full decode, 1 < batch <= 64): remap threads so that each warp-wide cp.async instruction cover
- **splitkv-chunk2048-batch32** → neutral (within-ci), 1.3838x — no shape moved outside noise. Shape-gated chunk size for master's CUDA split-KV path: at batch >= 32 use 2048-token work items instead of 1024, halving the partials written by kernel A and m

### FA3 call (full-layer verify gather)
- **full-verify-cuda-streamk-splitkv** → regression (regression), 1.2314x — gained on full decode_32; lost on full verify_16, verify_32. Stage 2 of the full-attention rewrite, built on master (stage 1 full-decode-cuda-gather-splitkv landed). Extend the CUDA split-KV path to full-layer VERIFY (q_l

### FA3 call (head packing)
- **fa3-hdim128-head-pairing** → gain (gain), 1.1706x — gained on full decode_16, decode_32, verify_16, verify_32; lost on full decode_1, verify_1. On full-attention layers only (config window == 0, decode and verify) call flash_attn_with_kvcache on a head-paired view of the cache so each token's two KV hea

### FA3 call (head packing) gate
- **paired-heads-launch-cleanup-ab-tune-batch-gate** → neutral (unproven-vs-best), 1.2686x — gained on full decode_1, verify_1. From the parent's branch (paired-heads-launch-cleanup built on master), change only the pairing predicate to paired = (window == 0 and hkv % 2 == 0 and batch > 

### FA3 call (head packing, sliding verify)
- **sliding-verify-fa3-head-pairing** → regression (regression), 1.3475x — lost on full decode_32; sliding verify_1, verify_16, verify_32. Config-only change in kernel_fn: extend master head-pairing predicate to sliding verify. Today paired = window == 0 and hkv % 2 == 0 and batch > 1 and not split

### FA3 call (schedule, off critical path)
- **fa3-scheduler-metadata-side-stream** → regression (regression), 1.2245x — lost on sliding decode_1, decode_16, decode_32, verify_1, verify_16, verify_32; full verify_1. Wherever master calls flash_attn_with_kvcache with scheduler_metadata=None (every sliding shape, full paired shapes at batch > 1, full verify_1), FA3 runs flash

### attention boundary
- **paired-heads-launch-cleanup** → gain (gain), 1.2436x — gained on full decode_1, decode_16, decode_32, verify_1, verify_16. Stacked on fa3-hdim128-head-pairing (and using fused-preattention-triton's kernel, which must have landed too): remove the five pad/un-pad launches the pairing 

### full + sliding decode attention (stack of two pending kernels)
- **stack-streamk-ticket-cluster-c16-c4** → neutral (beats-base-not-best), 1.4266x — gained on full decode_16, decode_32; sliding decode_32; lost on sliding verify_1. Stack the two pending (user-held, not on master) decode wins on current master; they touch disjoint shapes and code paths (REWRITE_READ_2026-10-25 '01:36Z harve

### full-layer attention (batch-1 decode gate)
- **full-fused-triton-decode1-gate** → neutral (within-ci), 1.3417x — lost on full decode_1. Built on master. Route full-attention batch-1 decode (config window == 0 and batch == 1 and q_len == 1) through master _sliding_fused_kernel via _sliding_forwar
- **full-decode1-cuda-splitkv-chunk320** → neutral (unproven-vs-best), 1.3954x — gained on full decode_1; lost on full decode_32; sliding verify_1. Route FULL-layer decode at batch 1 (window == 0, q_len 1, tp8, d 64, page 1) through master's existing CUDA split-KV path with a smaller work chunk, instead of 
- **full-decode1-cuda-splitkv-chunk640** → neutral (within-ci), 1.3805x — gained on full decode_1; lost on full decode_32; sliding verify_1. Same mechanism as full-decode1-cuda-splitkv-chunk320, with fewer, longer items. Route FULL-layer decode at batch 1 (window == 0, q_len 1, tp8, d 64, page 1) thr

### full-layer attention (batch-1 decode, splitkv stream-K kernel)
- **full-decode1-streamk-s3-two-level-ticket** → neutral (unproven-vs-best), 1.4322x — gained on full decode_1; lost on sliding verify_1. Stage 3 of the full-decode rewrite, built on the stage-2 branch hyp/muse_attention_decode/full-decode-splitkv-s2-streamk-ticket-inproc (splitkv_attn_sk_kernel: 

### full-layer attention (batch-1 decode, stream-K kernel merge)
- **full-decode1-sk4-cluster-dsmem-merge** → neutral (unproven-vs-best), 1.4388x — gained on sliding verify_1; lost on sliding decode_1. Stage 4 of the full-decode rewrite, built on master (stage 3 two-level ticket landed, c3c1b01). Full decode_1 is 22.6 us with the kernel at 18.6 us (0.89 TB/s) 

### full-layer attention (batch-1 decode, stream-K prologue fusion)
- **full-decode1-sk-two-level-fused-preattn** → neutral (within-ci), 1.4411x — gained on full decode_1. Full decode_1 is 22.6 us shape for 18.6 us of stream-K kernel + 1.4 us _pre_attention_kernel: ~2.6 us of the row is the second graph node and its launch gap. PD

### full-layer attention (batch-1 verify, stream-K two-level kernel, token grouping)
- **full-verify1-sk-two-level-token-group** → neutral (beats-base-not-best), 1.4714x — gained on sliding decode_1, verify_1. On the token-group branch (sliding-verify-sk-pseudo-s2-token-group). Full verify_1 is 30.5 us shape: FA3 15.4 + combine 7.0 + prepare 2.8 + pre-attn 1.5 + norm-

### full-layer attention (decode 1<B<=64 prologue) on top of the sliding restack
- **restack-sk-pseudo-cluster-plus-v3-fused-preattn** → gain (gain), 1.4920x — gained on sliding decode_1, decode_32, verify_1. Exactly restack-sk-pseudo-b1-cluster-c16-c4 (pseudo-decode sliding verify B=1 + sliding cluster C=16 at B=1 and C=4 at B=17..32, on master c3c1b01), PLUS the st

### full-layer attention (decode 1<B<=64, stream-K ticket epilogue)
- **full-decode-sk-s4-deferred-ticket-arrive** → failed (test:{\"batch\": 1, \"ctx\": 10, \"heads\": \"tp8\", \"layer\": \"sliding\", \"seed\": 6893170448766904337}). Built on master (c3c1b01). Master's <4 stages, ticket> kernel runs 112.3-113.5 us at B=16 vs 106.0-106.7 us for the same stream-K kernel with no ticket (3.07-3.

### full-layer attention (decode, splitkv kernel A/B)
- **full-decode-splitkv-s2-streamk-ticket-inproc** → neutral (unproven-vs-best), 1.4171x — gained on full decode_16, decode_32. Stage 2 of the full-decode rewrite, built on master (stage 1 kernels splitkv_attn_kernel A + splitkv_combine_kernel B in load_inline module muse_full_decode_spl

### full-layer attention (decode, splitkv stream-K kernel prologue)
- **full-decode-splitkv-s3-fused-preattn** → neutral (within-ci), 1.4283x — no shape moved outside noise. Stage 3 of the full-decode rewrite, built on the stage-2 branch hyp/muse_attention_decode/full-decode-splitkv-s2-streamk-ticket-inproc (stream-K over 64-token t

### full-layer attention (verify, splitkv A8/B8)
- **full-verify-wgmma-s3-warpgroup-per-kvhead-inproc** → neutral (beats-base-not-best), 1.3801x — gained on sliding decode_1, decode_16, decode_32; full verify_16, verify_32; lost on full decode_32. Stage 3 of the full-attention rewrite, built on the stage-2 code (branch hyp/muse_attention_decode/full-verify-cuda-streamk-splitkv, impl_commit ee7dd20, rebase

### full-layer attention (verify, wgmma stream-K kernel, multi-tile Q-fragment fix)
- **full-verify-wgmma-s5-qfrag-reload** → neutral (beats-base-not-best), 1.4345x — gained on full decode_1, decode_16, decode_32. Stage 5 on branch hyp/muse_attention_decode/full-verify-wgmma-s4-multitile-fix (job 51b0d5c110f3459c). s4's bisect: every multi-tile segment (>=2 tiles, even B1

### full-layer attention (verify, wgmma stream-K kernel, multi-tile correctness)
- **full-verify-wgmma-s4-multitile-fix** → neutral (within-ci), 1.3758x — no shape moved outside noise. Stage 4 (correctness fix) of the full-verify wgmma rewrite, built on the stage-3 commit (hyp/muse_attention_decode/full-verify-wgmma-s3-warpgroup-per-kvhead-inp

### full-layer attention (whole-path rewrite, warp-specialized one launch)
- **full-cuda-onelaunch-warpspec-bulk-gather** → regression (regression), 0.8238x — lost on full decode_16, decode_32; sliding verify_1. Stage 1 of a rewrite of the FULL-layer path, built on master (stage 1 of the first CUDA kernel, full-decode-cuda-gather-splitkv, has landed; no depends_on). Why

### launch overhead
- **splitkv-pdl-attn-to-combine** → neutral (within-ci), 1.3929x — lost on full decode_32. Programmatic dependent launch between master's kernel A (splitkv_attn_kernel) and kernel B (splitkv_combine_kernel), full decode 1 < batch <= 64. (1) In splitkv

### launch overhead (full decode_1, pre-attention -> stream-K kernel)
- **full-decode1-sk-pdl-from-preattn** → neutral (unproven-vs-best), 1.4571x — gained on full decode_32; sliding verify_1. Built on master (stream-K two-level ticket at full decode B=1, landed c3c1b01). Full decode_1 is 22.6 us shape vs pre-attn 1.4 + sk kernel 18.6 us of kernels, s

### output norm-gate
- **paired-norm-gate-token-tiled** → neutral (within-ci), 1.3911x — lost on full decode_32. Regrid master _paired_norm_gate_kernel (the paired full-layer epilogue, which now runs only at full verify_16/32) from one 1-warp program per (token, query head

### page table (sliding)
- **fa3-window-trimmed-page-table** → regression (regression), 1.1411x — lost on sliding decode_1, decode_16, decode_32, verify_1, verify_16, verify_32. On sliding-window layers only (config window != 0) hand FA3 a window-trimmed view of each request instead of the whole 80K-token context. Let W = window - 1 + L

### pre-attention norm+rope
- **fused-preattention-triton** → gain (gain), 1.1274x — gained on sliding decode_1, decode_16, decode_32, verify_1, verify_16, verify_32; full decode_1, decode_16, decode_32, verify_1, verify_16, verify_32. Replace the five pre-attention launches of the parity pipeline (FusedQKNorm (+RoPE), k .to(FP8), v .to(FP8), store_cache, q .to(FP8)) with one Triton kernel, gr
- **preattention-group-tiled-programs** → neutral (within-ci), 1.2614x — gained on sliding verify_32; lost on sliding decode_1. Regrid master _pre_attention_kernel from one 1-warp program per (token, head) ((tokens, Hq+Hkv) = 18 programs per token, 4608 at verify_32) to one program per (
- **preattention-tiled-gate-large-tokens** → neutral (within-ci), 1.3863x — gained on sliding verify_32; lost on full decode_32. Restack preattention-group-tiled-programs (neutral at 1.261x on an older master; its patch is muse_attention_decode/hypotheses/done/neutral/preattention-group-t

### sliding attention, batched decode, split depth
- **sliding-deep-splits-on-cuda-splitkv** → gain (gain), 1.3949x — gained on sliding decode_1, decode_16, decode_32, verify_1; full decode_32. Stack sliding-fused-triton-decode-batched-deep-splits (1.309x on the pre-CUDA master; sliding decode_16 gained 33%, decode_32 +3%, every other shape flat) onto 

### sliding-layer attention (batch-1 decode gate)
- **sliding-fused-triton-decode1-gate** → neutral (unproven-vs-best), 1.2931x — gained on sliding decode_1. Port _sliding_fused_kernel and its helpers (_counters, _num_splits, _norm_rope, _gpu_fence, _online_update, _gate_epilogue, LOG2E, _sliding_forward) verbatim fr

### sliding-layer attention (batch-1 decode, split depth)
- **sliding-fused-triton-decode1-splits-32** → neutral (unproven-vs-best), 1.3520x — no shape moved outside noise. Built on master. One config-only change in _sliding_forward: at batch 1 and q_len 1 (the only shape master routes to _sliding_fused_kernel) use S = 32 splits in

### sliding-layer attention (batched decode, KV row gather)
- **sliding-decode-paired-kvhead-rows** → regression (regression), 1.2865x — lost on sliding decode_16, decode_32, verify_1; full decode_32. One change inside master _sliding_fused_kernel and its launch in _sliding_forward, sliding decode at batch > 1 only (q_len 1; config-only predicate pair = lengt

### sliding-layer attention (batched decode, gather prologue)
- **sliding-fused-hoist-first-tile** → regression (regression), 1.3449x — lost on sliding decode_1, decode_16, decode_32; full decode_32. In master _sliding_fused_kernel, issue the first old-key tile's gather before the query work so the KV round trip overlaps the q load, RMSNorm and RoPE. Move th

### sliding-layer attention (batched decode, split depth)
- **sliding-fused-triton-decode-batched-deep-splits** → neutral (unproven-vs-best), 1.3087x — gained on sliding decode_16. Built on master. Two edits, both config-only. (1) In kernel_fn widen the fused sliding gate from (window and batch == 1 and q_len == 1) to (window and q_len == 
- **sliding-decode32-splits-8** → neutral (unproven-vs-best), 1.3973x — gained on sliding decode_32; lost on full decode_32. Config-only change in _sliding_forward: at length == 1 use splits = 16 if batch <= 16 else 8 (today 16 at every batch). At batch 32 that is 512 programs of 256 

### sliding-layer attention (batched decode, tile size)
- **sliding-decode-blockn128-single-tile** → neutral (within-ci), 1.3834x — gained on sliding decode_1; lost on full decode_32. Config-only change in _sliding_forward (master sliding decode at every batch, 16 splits): pass BLOCK_N=128 instead of 64 when length == 1. With S = 16 a split c

### sliding-layer attention (decode, cluster kernel)
- **sliding-cluster-s2-decode-c4-pdl-inproc** → neutral (unproven-vs-best), 1.4001x — gained on sliding decode_32, verify_1, verify_16, verify_32; lost on sliding decode_1, decode_16. Stage 2 of the sliding-layer cluster rewrite, built on the stage-1 code (branch hyp/muse_attention_decode/sliding-cuda-cluster-dsmem-onelaunch, impl_commit 5938

### sliding-layer attention (decode, stream-K windowed kernel)
- **sliding-decode-sk-window-route** → neutral (unproven-vs-best), 1.4973x — gained on sliding decode_32; full decode_32. On branch hyp/muse_attention_decode/sliding-verify-sk-pseudo-decode-s1. The PSEUDO kernel is already a windowed stream-K decode kernel (per-row lo = max(0, sp -

### sliding-layer attention (decode, stream-K windowed kernel) restack
- **restack-sliding-decode-sk-window-on-cluster-master** → neutral (unproven-vs-best), 1.5017x — gained on sliding decode_32. Restack sliding-decode-sk-window-route (done/neutral 1.4973x [1.475,1.520], job 118c6de9704146f3; built on the unlanded s1 branch, base blob 76492911) onto curr

### sliding-layer attention (restack of two measured row wins: verify B=1 pseudo-decode + decode cluster C=16/C=4)
- **restack-sk-pseudo-b1-cluster-c16-c4** → neutral (unproven-vs-best), 1.4870x — gained on sliding decode_1, decode_32, verify_1. Restack onto current master (c3c1b01, full decode stream-K v2 at 1<B<=64 + two-level ticket at B=1) the two measured sliding-row wins that touch disjoint shapes

### sliding-layer attention (verify, stream-K pseudo-decode kernel)
- **sliding-verify-sk-pseudo-decode-s1** → neutral (unproven-vs-best), 1.4705x — gained on sliding verify_1; lost on sliding decode_1. Stage 1 of a new sliding-verify line, built on master (which now carries the full-decode stream-K kernels: splitkv_attn_sk_kernel single-level ticket at 1<B<=64

### sliding-layer attention (verify, stream-K pseudo-decode kernel, token grouping)
- **sliding-verify-sk-pseudo-s2-token-group** → regression (regression), 1.4251x — lost on sliding verify_1. Stage 2 of the pseudo-decode line, on branch hyp/muse_attention_decode/sliding-verify-sk-pseudo-decode-s1. Measured: the PSEUDO kernel won B=1 (15.4 vs FA3 19.6

### sliding-layer attention (whole-path rewrite)
- **sliding-single-launch-fused-attention** → regression (regression), 0.9452x — gained on sliding decode_1; lost on sliding decode_16, decode_32, verify_1, verify_16, verify_32. Stage 1 of a rewrite of the sliding-layer path. Built on hyp/muse_attention_decode/fused-preattention-triton (1.127x, job dfee6f2939974bad). On sliding layers (

### sliding-layer attention (whole-path rewrite, CUDA cluster)
- **sliding-cuda-cluster-dsmem-onelaunch** → regression (regression), 1.2206x — gained on sliding decode_1, decode_16; lost on sliding verify_1, verify_16, verify_32. Stage 1 of a rewrite of the whole SLIDING-layer path, built on master (no depends_on). Why: the sliding layer runs 46x per step in every load, and it costs 13.9

### split-KV combine
- **splitkv-combine-wide-mlp** → neutral (within-ci), 1.3836x — lost on full decode_32. Rework splitkv_combine_kernel (master full decode, 1 < batch <= 64) for more loads in flight; grid (B, 16) and the math stay. Today each (request, head) block h

## Open follow-ups

- paired-heads-launch-cleanup-ab-tune-batch-gate: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: paired-heads-launch-cleanup-ab-tune-batch-gate`) that adds a further change in the same area is the only way to learn more (identi
- sliding-fused-triton-decode1-gate: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: sliding-fused-triton-decode1-gate`) that adds a further change in the same area is the only way to learn more (identical code is n
- preattention-group-tiled-programs: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: preattention-group-tiled-programs`) that adds a further change in the same area is the only way to learn more (identical code is never re
- sliding-fused-triton-decode-batched-deep-splits: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: sliding-fused-triton-decode-batched-deep-splits`) that adds a further change in the same area is the only way to learn more (ident
- full-fused-triton-decode1-gate: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: full-fused-triton-decode1-gate`) that adds a further change in the same area is the only way to learn more (identical code is never re-ru
- sliding-fused-triton-decode1-splits-32: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: sliding-fused-triton-decode1-splits-32`) that adds a further change in the same area is the only way to learn more (identical code
- full-decode-cuda-fp8-qk-mma: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: full-decode-cuda-fp8-qk-mma`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).
- full-decode1-cuda-splitkv-chunk320: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: full-decode1-cuda-splitkv-chunk320`) that adds a further change in the same area is the only way to learn more (identical code is 
- full-decode1-cuda-splitkv-chunk640: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: full-decode1-cuda-splitkv-chunk640`) that adds a further change in the same area is the only way to learn more (identical code is never r
- splitkv-full-line-gather-mapping: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: splitkv-full-line-gather-mapping`) that adds a further change in the same area is the only way to learn more (identical code is ne
- splitkv-combine-wide-mlp: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: splitkv-combine-wide-mlp`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).
- paired-norm-gate-token-tiled: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: paired-norm-gate-token-tiled`) that adds a further change in the same area is the only way to learn more (identical code is never re-run)
- preattention-tiled-gate-large-tokens: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: preattention-tiled-gate-large-tokens`) that adds a further change in the same area is the only way to learn more (identical code is never
- sliding-decode-blockn128-single-tile: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: sliding-decode-blockn128-single-tile`) that adds a further change in the same area is the only way to learn more (identical code is never
- sliding-decode32-splits-8: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: sliding-decode32-splits-8`) that adds a further change in the same area is the only way to learn more (identical code is never re-
- splitkv-chunk2048-batch32: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: splitkv-chunk2048-batch32`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).
- splitkv-pdl-attn-to-combine: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: splitkv-pdl-attn-to-combine`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).
- full-decode-splitkv-s2-streamk-ticket-inproc: rewrite stage 2 of `full-layer decode at 1<batch<=64: splitkv_attn_kernel A + splitkv_combine_kernel B (stage 1, master)`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the f
- full-verify-wgmma-s3-warpgroup-per-kvhead-inproc: rewrite stage 3 of `full-layer verify at 1<batch<=64: FA3 main + prepare_varlen_num_blocks + FlashAttnFwdCombine + norm-gate (master); stage-2 splitkv_verify_kernel A8 + splitkv_verify_combine B8`: read the replaced kern
- sliding-cluster-s2-decode-c4-pdl-inproc: rewrite stage 2 of `sliding layer decode at tp8/page 1, 1<=batch<=32: master Triton _sliding_fused_kernel (stage-1 sliding_cluster_kernel<1>)`: read the replaced kernels' own us per shape (profile top_kernels) against th
- full-decode-splitkv-s3-fused-preattn: rewrite stage 3 of `_pre_attention_kernel launch on full decode 1<B<=64 (folded into splitkv_attn_sk_kernel)`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and ch
- full-decode1-streamk-s3-two-level-ticket: rewrite stage 3 of `FA3 main + FlashAttnFwdCombine + sglang rms_norm_gate on full decode batch 1`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Mi
- stack-streamk-ticket-cluster-c16-c4: beats-base-not-best: the change is not proven either way; a stacked hypothesis (`depends_on: stack-streamk-ticket-cluster-c16-c4`) that adds a further change in the same area is the only way to learn more (identical code
- full-verify-wgmma-s4-multitile-fix: rewrite stage 4 of `full-layer verify at 1<batch<=64: FA3 main + prepare_varlen_num_blocks + FlashAttnFwdCombine + norm-gate (master paired FA3)`: read the replaced kernels' own us per shape (profile top_kernels) against
- full-decode1-sk4-cluster-dsmem-merge: rewrite stage 4 of `full decode batch 1: master splitkv_attn_sk_kernel two-level global ticket (grid SM)`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check 
- sliding-verify-sk-pseudo-decode-s1: rewrite stage 1 of `sliding verify (q_len 8, page 1, tp8, 1<=B<=32): prepare_varlen_num_blocks + FA3 main + FlashAttnFwdCombine + sglang rms_norm_gate`: read the replaced kernels' own us per shape (profile top_kernels) a
- full-decode1-sk-pdl-from-preattn: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: full-decode1-sk-pdl-from-preattn`) that adds a further change in the same area is the only way to learn more (identical code is ne
- full-decode1-sk-two-level-fused-preattn: rewrite stage 4 of `full decode batch 1: _pre_attention_kernel launch + master splitkv_attn_sk_kernel<4, two-level ticket>`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the
- full-verify-wgmma-s5-qfrag-reload: rewrite stage 5 of `full-layer verify at 1<batch<=64: FA3 main + prepare_varlen_num_blocks + FlashAttnFwdCombine + norm-gate (master paired FA3)`: read the replaced kernels' own us per shape (profile top_kernels) against
- restack-sk-pseudo-b1-cluster-c16-c4: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: restack-sk-pseudo-b1-cluster-c16-c4`) that adds a further change in the same area is the only way to learn more (identical code is
- full-verify1-sk-two-level-token-group: rewrite stage 3 of `full verify batch 1 (q_len 8): prepare_varlen_num_blocks + FA3 main + FlashAttnFwdCombine + norm-gate`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the 
- sliding-decode-sk-window-route: rewrite stage 2 of `sliding decode (q_len 1, page 1, tp8, 2<=B<=32): master Triton _sliding_fused_kernel (16 splits, in-kernel pre-attention)`: read the replaced kernels' own us per shape (profile top_kernels) against th
- restack-sliding-decode-sk-window-on-cluster-master: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: restack-sliding-decode-sk-window-on-cluster-master`) that adds a further change in the same area is the only way to learn more (id
- full-decode-sk-s4-deferred-ticket-arrive: rewrite stage 4 of `full decode 1<B<=64: master splitkv_attn_sk_kernel<4, ticket> single-level in-stream ticket epilogue`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the f

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
- queue: splitkv-l2-128b-prefetch-hint [FA3 call (full-layer decode gather)] p5 hold
- queue: full-decode-splitkv-s2-streamk-ticket [full-layer attention (decode, splitkv kernel A/B)] p1 hold
- queue: full-verify-wgmma-s3-warpgroup-per-kvhead [full-layer attention (verify, splitkv A8/B8)] p2 hold
- queue: sliding-cluster-s2-decode-c4-pdl [sliding-layer attention (decode, cluster kernel)] p2 hold
- queue: full-verify-wgmma-s4-sliding-window [sliding-layer attention (verify, wgmma stream-K kernel)] p2 hold
- queue: full-verify1-wgmma-s4-batch1-grid-sm [full-layer attention (batch-1 verify, wgmma stream-K kernel)] p2 hold
- queue: sliding-decode-paired-kvhead-rows-control-unpaired-s8 [sliding-layer attention (batched decode, split depth)] p2 queued
- queue: sliding-cluster-s3-verify-attn-only-phase-timed [sliding-layer attention (verify, cluster kernel)] p2 hold
- queue: sliding-verify1-triton-pseudo-decode [sliding-layer attention (batch-1 verify, Triton fused decode kernel)] p2 hold
- queue: full-verify1-sk-pseudo-decode [full-layer attention (batch-1 verify, stream-K pseudo-decode)] p3 hold
- queue: sliding-decode-sk-s3-fused-preattn-rope [sliding-layer attention (decode, stream-K windowed kernel, pre-attention fold)] p2 queued
- queue: preattention-tiled-every-verify-block [pre-attention norm+rope] p2 queued
- running: epilogue-gate-l2-prefetch-cuda [output norm-gate (gate operand prefetch in the CUDA epilogues)] p2 running
