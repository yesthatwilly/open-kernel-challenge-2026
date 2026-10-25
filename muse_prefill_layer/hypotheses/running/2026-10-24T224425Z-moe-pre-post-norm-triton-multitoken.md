---
problem: muse_prefill_layer
slug: moe-pre-post-norm-triton-multitoken
created: 2026-10-24T224425Z
base: master
area: pre-expert norm + post-expert norm/combine (the MoE elementwise ends)
change: >
  Built on master (635cfee). Two kernels at the ends of the MoE block, at every T. (1) PRE-EXPERT
  NORM: replace state['pre_expert_norm'](expert_input) (MuseSpark1xRMSNorm -> F.rms_norm, profile
  row at::native vectorized_layer_norm: 6.6/17.8/79 us at T=753/3012/16384 vs byte floors
  ~2.6/10/56 us) with a Triton row kernel _pre_expert_rmsnorm_kernel: grid NUM_SMS*4 persistent
  programs over rows, each row [4096] bf16 loaded once, fp32 sum of squares, y = x * rsqrt(mean +
  1e-5) * w_eff, stored bf16 into a fresh [T, 4096] tensor; w_eff is the module's effective (1 +
  w) weight taken from the SGLang module in prepare() (fp32). (2) POST-EXPERT NORM + WEIGHTED SUM:
  replace fused_post_expert_norm_weighted_sum(y, topk_weights, norm_weight=state['post_expert_w'],
  eps=POST_NORM_EPS, weight_first=True, slot_expert_ids=local_ids) in both _experts_marlin and
  _experts_triton with a Triton kernel modelled on muse_moe_w4a16 master's
  _post_norm_combine_kernel (copy it from muse_moe_w4a16/submission.py, where it gained 1.130x at
  small M) generalized to many tokens per program: grid NUM_SMS*4 persistent programs over tokens;
  per token loop k in 0..7, skip slots whose local id is -1 (never read their y row, which may be
  unwritten), for local slots load y[t*8+k] [4096] bf16, z = y * post_expert_w (weight_first=True:
  multiply by (1 + w) before normalizing, as master's call does), z *= rsqrt(mean(z^2) + 1e-8),
  acc += topk_weight[t, k] * z, all fp32; store acc bf16 into a fresh [T, 4096] tensor. Copy moe's
  kernel's exact order of operations; do not re-derive. prepare(): both kernels vs the SGLang
  calls on synthetic inputs at T = 753 and 16384 (pre-norm max diff <= 1 bf16 ulp; combine rel L2
  <= 1e-3 with NaN in every remote y row, proving they are never read); on failure log '[moe-ends]
  OFF: <reason>' and keep master's calls, else 'ON'.
expected:
  - shapes: "*load=first_turn*"
    direction: faster
    why: "pre-norm 79 us -> ~60 us, post-norm/combine 122 us -> ~70 us (16384 local y rows + output, ~268 MB): ~-0.5% per shape"
  - shapes: "*load=turn_4*"
    direction: faster
    why: "17.8 + 24.4 us -> ~12 + ~12 us: ~-0.5%"
  - shapes: "*load=turn_1*"
    direction: faster
    why: "6.6 + 8.3-8.9 us -> ~4 + ~5 us: ~-0.5%"
risk: low
depends_on: null
priority: 5
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-25T043904Z
review_companions: null
stage: implemented
claimed_at: 2026-10-25T043904Z
base_sha: 7e9d65cd4721de1932a376a8f32f7db7343f97ef
base_blob: db8a3344398bca79ea5cafeb580f701c055fd475
branch: hyp/muse_prefill_layer/moe-pre-post-norm-triton-multitoken
worktree: .worktrees/muse_prefill_layer/moe-pre-post-norm-triton-multitoken
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-25T044421Z
impl_commit: 945363f1bf69506c2161e1672f20f66b3b4796bf
submission_sha256: 5cb0bb62dfc10d7db34371668bd08462620233b7946e2cfb2a7d0ff0c17c6908
norm_sha256: 0c02bc77d8b21aeefde72af80fa702fa538e5f22deb7e8786489b6fd3bd3c8c8
submission_blob: cab1f9ab2b7d6180afac73a43cbcf68cf1b6eb07
lint_warnings: 23
attempt_file: 2026-10-25T044421Z-check-moe-pre-post-norm-triton-multitoken.md
prepared_at: 2026-10-25T044421Z
---

# moe-pre-post-norm-triton-multitoken

## Change

Built on master (635cfee). Two kernels at the ends of the MoE block, at every T. (1) PRE-EXPERT NORM: replace state['pre_expert_norm'](expert_input) (MuseSpark1xRMSNorm -> F.rms_norm, profile row at::native vectorized_layer_norm: 6.6/17.8/79 us at T=753/3012/16384 vs byte floors ~2.6/10/56 us) with a Triton row kernel _pre_expert_rmsnorm_kernel: grid NUM_SMS*4 persistent programs over rows, each row [4096] bf16 loaded once, fp32 sum of squares, y = x * rsqrt(mean + 1e-5) * w_eff, stored bf16 into a fresh [T, 4096] tensor; w_eff is the module's effective (1 + w) weight taken from the SGLang module in prepare() (fp32). (2) POST-EXPERT NORM + WEIGHTED SUM: replace fused_post_expert_norm_weighted_sum(y, topk_weights, norm_weight=state['post_expert_w'], eps=POST_NORM_EPS, weight_first=True, slot_expert_ids=local_ids) in both _experts_marlin and _experts_triton with a Triton kernel modelled on muse_moe_w4a16 master's _post_norm_combine_kernel (copy it from muse_moe_w4a16/submission.py, where it gained 1.130x at small M) generalized to many tokens per program: grid NUM_SMS*4 persistent programs over tokens; per token loop k in 0..7, skip slots whose local id is -1 (never read their y row, which may be unwritten), for local slots load y[t*8+k] [4096] bf16, z = y * post_expert_w (weight_first=True: multiply by (1 + w) before normalizing, as master's call does), z *= rsqrt(mean(z^2) + 1e-8), acc += topk_weight[t, k] * z, all fp32; store acc bf16 into a fresh [T, 4096] tensor. Copy moe's kernel's exact order of operations; do not re-derive. prepare(): both kernels vs the SGLang calls on synthetic inputs at T = 753 and 16384 (pre-norm max diff <= 1 bf16 ulp; combine rel L2 <= 1e-3 with NaN in every remote y row, proving they are never read); on failure log '[moe-ends] OFF: <reason>' and keep master's calls, else 'ON'.

## Expected

- `*load=first_turn*` faster: pre-norm 79 us -> ~60 us, post-norm/combine 122 us -> ~70 us (16384 local y rows + output, ~268 MB): ~-0.5% per shape
- `*load=turn_4*` faster: 17.8 + 24.4 us -> ~12 + ~12 us: ~-0.5%
- `*load=turn_1*` faster: 6.6 + 8.3-8.9 us -> ~4 + ~5 us: ~-0.5%

## Risk

Small and below the headline noise floor (~0.5% per shape); judge it on the four replaced rows (Triton rows reproduce within ~1%). Port of moe's post-norm kernel; its all-M variant was neutral at moe's tiny M, but prefill's M is 753-16384, where SGLang's kernel runs at ~2.2 TB/s. Composes with every expert-engine and dispatch hypothesis in the queue (it reads y by slot and skips remote slots, which is also what the local-routes align needs).

## Adversarial review

Reviewed 2026-10-25T043904Z by an adversarial subagent (model opus).

**Verdict: proceed.** Not a duplicate: norm-boundaries-persistent-multirow replaced the two 8192-wide FusedPostNormResidualPreNorm boundaries, and its 1.0716 came from master's own modules, because every Triton variant timed 12-39% slower in prepare. The decode and moe ports are per-token kernels for T<32, and moe's all-M variant lost or tied SGLang at bs=128/256. The expected ~0.3-0.5% per shape is well under the ±1.5% per-shape noise and the ±1.6% headline CI, so it can only be judged on the four replaced profile rows, and the spec must add a prepare-time timing gate and fix a likely weight double-add.

### Justification
Profile rows confirmed on master (archive/stack-tonight-five-prefill/c4d14d62f11e4aaf.profile.txt): _post_expert_norm_weighted_sum_kernel 8.2-8.9 / 23.8 / 119-130 us and at::native vectorized_layer_norm (pre_expert_norm) 6.6-6.8 / 18.0 / 77-79 us at T=753 / 3012 / 16384. Byte floors at first_turn: post-combine reads ~1/8 of 16384x8 slots (local routes) x 8 KB plus writes 134 MB, about 268 MB, so 56 us at 4.8 TB/s. The incumbent runs at ~2.2 TB/s, so this kernel has real headroom (practical ~4.0 TB/s means ~67 us, saving ~55 us). The pre-norm also moves 268 MB but already runs at ~3.4-3.5 TB/s, so the realistic saving is ~8-12 us. At turn_1 the inputs are L2-resident (pooler output is 6 MB) and both rows are latency-bound, so the HBM floors quoted (2.6 us) do not apply. Savings there are a few us at most. Total: ~65 us of 14.3-21.7 ms at first_turn (0.3-0.45%), ~15 us of 3.3-4.9 ms at turn_4, ~5-6 us of 1.1-1.5 ms at turn_1. Each is below the ±1.5% per-shape noise, so the headline cannot show it. That is acceptable only because both rows sit in the per-run profile.txt top list, which is archived for every run, so the rows are readable. WEAKEST LINK: the evidence that a hand-written Triton row kernel beats the incumbents at prefill M is negative. (a) In norm-boundaries-persistent-multirow's own prepare sweep (archive/norm-boundaries-persistent-multirow/46afe489e2754c83.log), rows-nw4, rows-nw8 and persistent-nw8 were 1.12-1.39x SLOWER than SGLang's CUDA kernel at every T, so the run executed master's modules. (b) The kernel to be copied, moe's _post_norm_combine_kernel, ran 4.5 us vs SGLang's 4.2 at bs=128 and 6.2 vs 6.4 at bs=256 (post-norm-triton-all-m). So it does not beat SGLang's Triton kernel as M grows, and the 1.130x cited is a T<32 result that does not transfer. The persistent multi-token restructuring is the untested part that has to deliver the gain. One structural point helps it: SGLang's kernel reaches only ~2.2 TB/s, likely from per-slot serialization, and the copied kernel's per-token loop over 8 slots with a scalar branch serializes 8 dependent row loads. For bandwidth, the new kernel should issue all local slots' loads before the reductions (moe's post-norm-combine-hoisted-slot-loads idea), not loop slot by slot.

### Confounders on the current kernel
- stack-tonight-five-prefill (landed, current master a81d19c): its w2-nonlocal-zero-store-off part turns the w2 zero store off at T>4096 because _weighted_sum_skips_remote() proved SGLang's combine never reads remote y rows. The new combine replaces that kernel, so it inherits the precondition. The spec's NaN-in-remote-rows check covers it, but _weighted_sum_skips_remote still probes only SGLang's kernel. If the new kernel is ON, the probe (or a copy of it) must run against the new kernel, or a remote-row read becomes a silent correctness bug at first_turn only. The spec also says 'master (635cfee)'. Build on a81d19c and compare rows against c4d14d62f11e4aaf / 502f2eab95164781.
- experts-hybrid-marlin-turn-triton-first-turn (landed): the combine has two call sites with different y producers. Marlin's moe_wna16_marlin_gemm output view at T<=4096, and the Triton engine's cache3 (a view into the shared cache13 buffer) at T>4096. topk_weights is cast to fp32 only on the Triton path. The kernel must accept both and take strides or contiguity explicitly. Otherwise a turn-shape result and a first_turn result may reflect different bugs or copies rather than the kernel.
- router-hilo-one-gemm-n512 / moe-align parts of the stack: no interaction. They change routing-side launches that finish before _experts, and the combine's input set (local slots) is identical by construction.
- sglang-pipeline-parity: defines post_expert_w as the module's effective (1 + w) weight in bf16. See rules_flags: moe's kernel adds 1.0 in-kernel to the ORIGINAL weight, so copying it verbatim while passing post_expert_w gives (2 + w). That is a numerical confounder that row-normalization partly masks.

### Interference and A/B needs
No companion. Both replaced kernels have their own rows in the archived profile.txt of every run (top list), so a gain or loss is read directly from four rows per shape against the c4d14d62f11e4aaf rows above, and nothing landed shares their launch or memory path except the remote-row precondition, which is a correctness question for prepare(), not an A/B question. Reverting any landed patch would either remove the y producer or change the remote-row contract, making the comparison worse, not cleaner. The pre-norm vs post-combine split does not need a separate run: they are separate rows. The run as specified does have one interpretability gap: it has no timing fallback. Given norm-boundaries' 12-39% Triton loss and moe's tie at bs=256, the run must carry its own prepare-time evidence. For each kernel separately and each T class (753, 3012, 16384), CUDA-graph-time the incumbent vs the new kernel with an L2 flush, median of >=25, and print '[moe-ends] pre|post T=... master X us new Y us'. Enable each kernel per T class only if <= 0.98x. Otherwise keep master's call for that kernel and class, and log which path was chosen. Then even an 'all OFF' run tells us something, as norm-boundaries' log did.

### Open questions
- Does a persistent multi-token combine that issues all local slots' row loads before the 8 reductions get the 16384 row below ~80 us? SGLang's Triton kernel sits at ~2.2 TB/s, and moe's per-token loop tied it at bs=256.
- Is pre-expert norm worth replacing at all? At first_turn the incumbent is ~70% of the HBM floor, and at turn_1 it is L2- and latency-bound. If its prepare timing is not <= 0.98x, keep F.rms_norm and let the post-combine stand alone.
- Longer term, the pre-norm belongs in the pooler GEMM's epilogue or the expert GEMM's A-load, and the combine in the w2 epilogue. Those fusions remove the 268 MB round trips that a standalone kernel can only shave.


## Implementation notes

Built on the worktree HEAD 7e9d65c, which already contains a81d19c (stack-tonight-five-prefill), so no 3-way merge was needed. Only `muse_prefill_layer/submission.py` changed (+302/-7). The `## Adversarial review` overrides the spec where they differ:

- New Triton kernels `_pre_expert_rmsnorm_kernel` (persistent rows, 4096 wide, fp32 math, effective weight from `pre_expert_norm._get_effective_weight()`) and `_post_combine_kernel` (persistent tokens; variants HOIST = one masked [8, 4096] load of every local slot's row before the 8 reductions, or a per-slot loop). Remote slots are masked out of the load, so their y rows are never read. `_pre_norm` and `_post_combine` dispatch by token class (753 up to 1024, 3012 up to 4096, else 16384); `_experts_marlin` and `_experts_triton` both call `_post_combine`.
- Review fix 1 (timing gate): `_moe_ends_setup()` in prepare() graph-times master vs new per kernel, per class T = 753/3012/16384, cold L2 (288 MB flush before every replay), interleaved, median of 25. It tries pre num_warps (4, 8) and combine variants ((hoist, 16), (loop, 8), (hoist, 8)); a variant is enabled only at ratio <= 0.98 and after passing its checks. Prints `[moe-ends] pre|post T=... master X us new Y us ratio ...` and a `PATH CHOSEN` line per kernel and class. Any exception keeps master's call.
- Review fix 2 (double add): moe's kernel adds 1 in-kernel to the raw weight; this kernel takes `state['post_expert_w']` (already 1 + w, bf16) and adds nothing. Verified in the self-check against master's call and an independent fp32 reference.
- Review fix 3: no bf16-ulp gate. Gates are rel L2 vs master <= 2e-3, max |diff| <= 2 bf16 spacings (spacing of max(|ref|, 2^-8 x row max), so near-zero values do not read as huge ulp counts), and error vs an fp32 reference <= 1.25 x master's own error + 1e-4. All numbers are printed.
- Review fix 4: the remote-row probe runs against the NEW combine (NaN vs zero in remote y rows, finite and bitwise-equal output) for every variant, at each class's T. If the combine is ON at the first-turn class, `state['zero_nonlocal_w2']` is forced False (the new kernel's own probe decided); otherwise master's `_weighted_sum_skips_remote` result stands.
- Deviation: the combine's per-token loop is replaced by the hoisted-load variant first (review guidance), with the loop variant as a candidate. Judge on the four replaced profile rows (master: post-expert combine 122 us, pre-expert norm 77 us at T=16384), not the headline.
- Nothing could be run on this node (no GPU/torch/Triton); no subprocess, no CUDA load_inline, no sm_90a PTX, so the CUDA-specific brief items do not apply (self-check launches still print before and after with a synchronize).

## Rules check

- [x] CUDA-graph capturable: kernel_fn adds only Triton launches and `torch.empty` on shapes; no .item/.cpu/synchronize (all syncs are in prepare(); lint reports them as prepare-only warnings).
- [x] Nothing derived from `data` is cached across calls: variants are picked in prepare() from synthetic inputs and the token-count class (a shape).
- [x] Weights: no repacking; `pre_w_eff` is a lossless fp32 copy of the module's effective weight, `post_expert_w` is unchanged.
- [x] Activations BF16 where SGLang keeps BF16 (kernel I/O bf16, fp32 math inside as in SGLang's kernels); KV cache FP8 untouched.
- [x] Attention untouched: n/a, no attention change.
- [x] No timing patches, harness frames, module edits or reference import; the flush buffer is a prepare-time local.
- [x] All work happens inside the timed call; no output caching or replay.

okc_lint: `OK with warnings: muse_prefill_layer/submission.py` (host-sync warnings only, all in prepare(); one dtype warning for the uint8 prepare-time flush buffer). `python3 -I -m py_compile` passed; __pycache__ deleted.
