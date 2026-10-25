# muse_prefill_layer · 2026-10-25T044421Z · check · moe-pre-post-norm-triton-multitoken

Hypothesis: `muse_prefill_layer/hypotheses/running/2026-10-24T224425Z-moe-pre-post-norm-triton-multitoken.md` (branch `hyp/muse_prefill_layer/moe-pre-post-norm-triton-multitoken`).

- **Job:** `3fb4e7237a604fa3` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `945363f`

## Goal
Built on master (635cfee). Two kernels at the ends of the MoE block, at every T. (1) PRE-EXPERT NORM: replace state['pre_expert_norm'](expert_input) (MuseSpark1xRMSNorm -> F.rms_norm, profile row at::native vectorized_layer_norm: 6.6/17.8/79 us at T=753/3012/16384 vs byte floors ~2.6/10/56 us) with a Triton row kernel _pre_expert_rmsnorm_kernel: grid NUM_SMS*4 persistent programs over rows, each row [4096] bf16 loaded once, fp32 sum of squares, y = x * rsqrt(mean + 1e-5) * w_eff, stored bf16 into a fresh [T, 4096] tensor; w_eff is the module's effective (1 + w) weight taken from the SGLang module in prepare() (fp32). (2) POST-EXPERT NORM + WEIGHTED SUM: replace fused_post_expert_norm_weighted_sum(y, topk_weights, norm_weight=state['post_expert_w'], eps=POST_NORM_EPS, weight_first=True, slot_expert_ids=local_ids) in both _experts_marlin and _experts_triton with a Triton kernel modelled on muse_moe_w4a16 master's _post_norm_combine_kernel (copy it from muse_moe_w4a16/submission.py, where it gained 1.130x at small M) generalized to many tokens per program: grid NUM_SMS*4 persistent programs over tokens; per token loop k in 0..7, skip slots whose local id is -1 (never read their y row, which may be unwritten), for local slots load y[t*8+k] [4096] bf16, z = y * post_expert_w (weight_first=True: multiply by (1 + w) before normalizing, as master's call does), z *= rsqrt(mean(z^2) + 1e-8), acc += topk_weight[t, k] * z, all fp32; store acc bf16 into a fresh [T, 4096] tensor. Copy moe's kernel's exact order of operations; do not re-derive. prepare(): both kernels vs the SGLang calls on synthetic inputs at T = 753 and 16384 (pre-norm max diff <= 1 bf16 ulp; combine rel L2 <= 1e-3 with NaN in every remote y row, proving they are never read); on failure log '[moe-ends] OFF: <reason>' and keep master's calls, else 'ON'.

## What changed, and where
Built on the worktree HEAD 7e9d65c, which already contains a81d19c (stack-tonight-five-prefill), so no 3-way merge was needed. Only `muse_prefill_layer/submission.py` changed (+302/-7). The `## Adversarial review` overrides the spec where they differ:

- New Triton kernels `_pre_expert_rmsnorm_kernel` (persistent rows, 4096 wide, fp32 math, effective weight from `pre_expert_norm._get_effective_weight()`) and `_post_combine_kernel` (persistent tokens; variants HOIST = one masked [8, 4096] load of every local slot's row before the 8 reductions, or a per-slot loop). Remote slots are masked out of the load, so their y rows are never read. `_pre_norm` and `_post_combine` dispatch by token class (753 up to 1024, 3012 up to 4096, else 16384); `_experts_marlin` and `_experts_triton` both call `_post_combine`.
- Review fix 1 (timing gate): `_moe_ends_setup()` in prepare() graph-times master vs new per kernel, per class T = 753/3012/16384, cold L2 (288 MB flush before every replay), interleaved, median of 25. It tries pre num_warps (4, 8) and combine variants ((hoist, 16), (loop, 8), (hoist, 8)); a variant is enabled only at ratio <= 0.98 and after passing its checks. Prints `[moe-ends] pre|post T=... master X us new Y us ratio ...` and a `PATH CHOSEN` line per kernel and class. Any exception keeps master's call.
- Review fix 2 (double add): moe's kernel adds 1 in-kernel to the raw weight; this kernel takes `state['post_expert_w']` (already 1 + w, bf16) and adds nothing. Verified in the self-check against master's call and an independent fp32 reference.
- Review fix 3: no bf16-ulp gate. Gates are rel L2 vs master <= 2e-3, max |diff| <= 2 bf16 spacings (spacing of max(|ref|, 2^-8 x row max), so near-zero values do not read as huge ulp counts), and error vs an fp32 reference <= 1.25 x master's own error + 1e-4. All numbers are printed.
- Review fix 4: the remote-row probe runs against the NEW combine (NaN vs zero in remote y rows, finite and bitwise-equal output) for every variant, at each class's T. If the combine is ON at the first-turn class, `state['zero_nonlocal_w2']` is forced False (the new kernel's own probe decided); otherwise master's `_weighted_sum_skips_remote` result stands.
- Deviation: the combine's per-token loop is replaced by the hoisted-load variant first (review guidance), with the loop variant as a candidate. Judge on the four replaced profile rows (master: post-expert combine 122 us, pre-expert norm 77 us at T=16384), not the headline.
- Nothing could be run on this node (no GPU/torch/Triton); no subprocess, no CUDA load_inline, no sm_90a PTX, so the CUDA-specific brief items do not apply (self-check launches still print before and after with a synchronize).

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn adds only Triton launches and `torch.empty` on shapes; no .item/.cpu/synchronize (all syncs are in prepare(); lint reports them as prepare-only warnings).
- [x] Nothing derived from `data` is cached across calls: variants are picked in prepare() from synthetic inputs and the token-count class (a shape).
- [x] Weights: no repacking; `pre_w_eff` is a lossless fp32 copy of the module's effective weight, `post_expert_w` is unchanged.
- [x] Activations BF16 where SGLang keeps BF16 (kernel I/O bf16, fp32 math inside as in SGLang's kernels); KV cache FP8 untouched.
- [x] Attention untouched: n/a, no attention change.
- [x] No timing patches, harness frames, module edits or reference import; the flush buffer is a prepare-time local.
- [x] All work happens inside the timed call; no output caching or replay.

okc_lint: `OK with warnings: muse_prefill_layer/submission.py` (host-sync warnings only, all in prepare(); one dtype warning for the uint8 prepare-time flush buffer). `python3 -I -m py_compile` passed; __pycache__ deleted.

## Risks / what could fail
low
Small and below the headline noise floor (~0.5% per shape); judge it on the four replaced rows (Triton rows reproduce within ~1%). Port of moe's post-norm kernel; its all-M variant was neutral at moe's tiny M, but prefill's M is 753-16384, where SGLang's kernel runs at ~2.2 TB/s. Composes with every expert-engine and dispatch hypothesis in the queue (it reads y by slot and skips remote slots, which is also what the local-routes align needs).

## Result (after the run)
- job `3fb4e7237a604fa3` (check), status done, passed True, tests 5/5
- speedup 1.0714x, 95% CI [1.0553, 1.0875], score 361590.0309265246 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.074386115190733, "ci": [1.058270323462872, 1.090501906918594], "job": "502f2eab95164781"}; base result: muse_prefill_layer/hypotheses/done/neutral/stack-tonight-five-prefill.result.json
- **outcome: neutral** (within-ci)
- files: `muse_prefill_layer/hypotheses/done/neutral/moe-pre-post-norm-triton-multitoken.result.json`, `muse_prefill_layer/hypotheses/done/neutral/moe-pre-post-norm-triton-multitoken.patch`; branch `hyp/muse_prefill_layer/moe-pre-post-norm-triton-multitoken`
