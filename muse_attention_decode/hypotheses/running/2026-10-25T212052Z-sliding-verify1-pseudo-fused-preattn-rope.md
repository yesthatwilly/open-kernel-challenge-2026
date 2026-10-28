---
problem: muse_attention_decode
slug: sliding-verify1-pseudo-fused-preattn-rope
created: 2026-10-25T212052Z
base: master
area: sliding-layer attention (batch-1 verify, PSEUDO stream-K kernel, pre-attention fold)
change: >
  DIAGNOSIS action 3(ii). Stacked on merge-perkvhead-v3chain-sv32-tiled (build on its branch,
  whatever its score; the change touches only the sliding verify B=1 band). Sliding verify_1
  (weight 46, ~0.70%/us, the second-highest rate on the board) is 13.2 us of shape for
  splitkv_attn_sk_kernel PSEUDO 9.9 us + _pre_attention_kernel 1.5 us: about 1.8 us is the second
  graph node and its gap. Fold the pre-attention into the PSEUDO kernel so B=1 sliding verify is
  ONE launch, as combo-b1's V3 did for full decode_1 (FUSED prologue; 18.98 vs V1 21.41 us at its
  gate) and as skF does for full decode B>1. Put it in a NEW load_inline module muse_sv1_fused_v1
  (helper head + a copy of the PSEUDO kernel template with one extra template flag FUSED; the
  landed _EXT2 PSEUDO instantiations stay untouched, as skF was kept apart). FUSED=true at TOK=8:
  (a) q: when a CTA starts a segment of pseudo-row r = 8b + j, it loads token j's raw bf16 q row
  (16 heads x 64, 2 KB) from qkv (riding the page-index cp.async group), applies exactly what
  _pre_attention_kernel does with ROPE=True, PAIRED=False (its RMSNorm and eps, interleaved GPT-J
  RoPE pair (2i, 2i+1) at angle positions[8b+j] * inv_freq[i] from master _inv_freq, then the e4m3
  cast with the same scale handling) and builds the q fragments the non-fused path reads from
  smem. Use accurate sincosf. (b) new K/V rows: the 8 new keys (token j's own key = the last key
  of pseudo-row j, cache slot cache_loc[8b+j], in the PSEUDO kernel's own sp_r convention) are
  built by every CTA whose tile range contains any of them, from qkv (k: RMSNorm then RoPE then /
  k_scale -> e4m3; v: / v_scale -> e4m3), and written over the landed ring stage in smem after
  cp_wait and before that tile's S MMA (one __syncthreads), never read back through cp.async;
  exactly one CTA (the owner of pseudo-row 7's last tile, checked on the host like
  _skf_owner_check) stores the 8 rows to k_cache/v_cache. Other CTAs may read the stale slots by
  cp.async; they overwrite them in smem first. kc/vc become non-const in this module only. Route:
  sliding verify, window 2048, q_len 8, B == 1 only, config-only predicate; bands 2..32 unchanged.
  GATE at B=1 x 80000: interleaved [FUSED PSEUDO, routed variant and grid spec] vs [pre-attn +
  master's routed PSEUDO variant/grid] (and print every grid spec p4/p8/sm for FUSED); route only
  at >= 1.03x. SELF-CHECK vs an fp32 windowed causal reference WITH RoPE: B=1 lens 1, 7, 8, 63,
  2047, 2048, 2055, 80000, kv 0.25, grids 3/p4/p8/sm, coherent data; compare the stored new cache
  rows with _pre_attention_kernel's rows by rel L2 (< 2e-2; never bytewise: LESSONS, Triton
  sin/cos differ in 1-6% of E4M3 bytes at ~5000 rad) and check other rows unchanged; two-state
  alternating capture-once replay with refilled inputs and NaN-poisoned outputs. Any miss: B=1
  stays on [pre-attn + PSEUDO] with a PATH CHOSEN line. New CUDA: first measurement is an okc
  check. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation
  sized by input values (shape/config only), no data-dependent Python control flow, no prints, no
  try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8
  E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes, no
  timing in kernel_fn. The judge prepares TWO states and rotates them: no module/static global may
  hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (self-resetting
  int32 ticket arrays, zeroed once and returned to 0 by every launch, are fine). BUILD:
  TORCH_CUDA_ARCH_LIST=9.0a only, -std=c++20 whenever extra_cuda_cflags are overridden, -lineinfo,
  in-process canary, sync + print after each self-check launch, print regs/local bytes/CTAs per SM
  and refuse a variant with local bytes > 0. Gates compare printed interleaved cold-L2
  graph-replay timings (256 MB flush, empty graph subtracted) and route only when the new path
  WINS by the stated margin; PATH CHOSEN prints both numbers.
expected:
  - shapes: "*layer=sliding load=verify_1 *"
    direction: faster
    why: 13.2 us shape = PSEUDO 9.9 + pre-attn 1.5 + ~1.8 node gap; one launch with a ~0.5 us prologue should give <= 11.2 us (floor 0.1); ~-2 us at 0.70%/us is about +1.4% headline
  - shapes: "*layer=sliding load=verify_16 *"
    direction: flat
    why: B == 1 only
  - shapes: "*layer=sliding load=verify_32 *"
    direction: flat
    why: B == 1 only (SV32 tiled from the base)
  - shapes: "*layer=full *"
    direction: flat
    why: untouched beyond the base
  - shapes: "*load=decode_* *"
    direction: flat
    why: untouched beyond the base
risk: medium
depends_on: merge-perkvhead-v3chain-sv32-tiled
priority: 1
kind: primary
controls_for: null
prior: "full-decode1-sk-two-level-fused-preattn: that fold was on full decode_1 (weight 16, no RoPE, one new row, gained 22.4->21.6 within CI); this folds RoPE pre-attention for 8 query tokens and 8 new K/V rows into the TOK=8 PSEUDO kernel on sliding verify_1 (weight 46, 0.70%/us). Queued sliding-decode-sk-s3-fused-preattn-rope does the TOK=1 decode case only"
review: proceed
status: running
review_at: 2026-10-25T212856Z
review_companions: null
stage: implemented
claimed_at: 2026-10-28T145301Z
base_sha: 47926f5e7d8a192184e21177901be2034d2d4930
base_blob: 2eb03f76ceefa9bf81e28616ced3b33be3a70443
branch: hyp/muse_attention_decode/sliding-verify1-pseudo-fused-preattn-rope
worktree: .worktrees/muse_attention_decode/sliding-verify1-pseudo-fused-preattn-rope
mode: null
restack_of: merge-perkvhead-v3chain-sv32-tiled
stale_base: null
implemented_at: 2026-10-28T150621Z
impl_commit: d73edd948dd3a847b00a08207be4d06635a30025
submission_sha256: 368304ce28e7eba549f470074d2ec91cdca6493d7a4b10b3f7b66553d8460df1
norm_sha256: f3d236e5225affd8c4eb1f5a6f2e0111788b83bcc9ac81c9e71f203866aba478
submission_blob: 7e7938b33b2d532aab7c1f0989f5b95081eee05d
lint_warnings: 101
---

# sliding-verify1-pseudo-fused-preattn-rope

## Change

DIAGNOSIS action 3(ii). Stacked on merge-perkvhead-v3chain-sv32-tiled (build on its branch, whatever its score; the change touches only the sliding verify B=1 band). Sliding verify_1 (weight 46, ~0.70%/us, the second-highest rate on the board) is 13.2 us of shape for splitkv_attn_sk_kernel PSEUDO 9.9 us + _pre_attention_kernel 1.5 us: about 1.8 us is the second graph node and its gap. Fold the pre-attention into the PSEUDO kernel so B=1 sliding verify is ONE launch, as combo-b1's V3 did for full decode_1 (FUSED prologue; 18.98 vs V1 21.41 us at its gate) and as skF does for full decode B>1. Put it in a NEW load_inline module muse_sv1_fused_v1 (helper head + a copy of the PSEUDO kernel template with one extra template flag FUSED; the landed _EXT2 PSEUDO instantiations stay untouched, as skF was kept apart). FUSED=true at TOK=8: (a) q: when a CTA starts a segment of pseudo-row r = 8b + j, it loads token j's raw bf16 q row (16 heads x 64, 2 KB) from qkv (riding the page-index cp.async group), applies exactly what _pre_attention_kernel does with ROPE=True, PAIRED=False (its RMSNorm and eps, interleaved GPT-J RoPE pair (2i, 2i+1) at angle positions[8b+j] * inv_freq[i] from master _inv_freq, then the e4m3 cast with the same scale handling) and builds the q fragments the non-fused path reads from smem. Use accurate sincosf. (b) new K/V rows: the 8 new keys (token j's own key = the last key of pseudo-row j, cache slot cache_loc[8b+j], in the PSEUDO kernel's own sp_r convention) are built by every CTA whose tile range contains any of them, from qkv (k: RMSNorm then RoPE then / k_scale -> e4m3; v: / v_scale -> e4m3), and written over the landed ring stage in smem after cp_wait and before that tile's S MMA (one __syncthreads), never read back through cp.async; exactly one CTA (the owner of pseudo-row 7's last tile, checked on the host like _skf_owner_check) stores the 8 rows to k_cache/v_cache. Other CTAs may read the stale slots by cp.async; they overwrite them in smem first. kc/vc become non-const in this module only. Route: sliding verify, window 2048, q_len 8, B == 1 only, config-only predicate; bands 2..32 unchanged. GATE at B=1 x 80000: interleaved [FUSED PSEUDO, routed variant and grid spec] vs [pre-attn + master's routed PSEUDO variant/grid] (and print every grid spec p4/p8/sm for FUSED); route only at >= 1.03x. SELF-CHECK vs an fp32 windowed causal reference WITH RoPE: B=1 lens 1, 7, 8, 63, 2047, 2048, 2055, 80000, kv 0.25, grids 3/p4/p8/sm, coherent data; compare the stored new cache rows with _pre_attention_kernel's rows by rel L2 (< 2e-2; never bytewise: LESSONS, Triton sin/cos differ in 1-6% of E4M3 bytes at ~5000 rad) and check other rows unchanged; two-state alternating capture-once replay with refilled inputs and NaN-poisoned outputs. Any miss: B=1 stays on [pre-attn + PSEUDO] with a PATH CHOSEN line. New CUDA: first measurement is an okc check. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes, no timing in kernel_fn. The judge prepares TWO states and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, are fine). BUILD: TORCH_CUDA_ARCH_LIST=9.0a only, -std=c++20 whenever extra_cuda_cflags are overridden, -lineinfo, in-process canary, sync + print after each self-check launch, print regs/local bytes/CTAs per SM and refuse a variant with local bytes > 0. Gates compare printed interleaved cold-L2 graph-replay timings (256 MB flush, empty graph subtracted) and route only when the new path WINS by the stated margin; PATH CHOSEN prints both numbers.

## Expected

- `*layer=sliding load=verify_1 *` faster: 13.2 us shape = PSEUDO 9.9 + pre-attn 1.5 + ~1.8 node gap; one launch with a ~0.5 us prologue should give <= 11.2 us (floor 0.1); ~-2 us at 0.70%/us is about +1.4% headline
- `*layer=sliding load=verify_16 *` flat: B == 1 only
- `*layer=sliding load=verify_32 *` flat: B == 1 only (SV32 tiled from the base)
- `*layer=full *` flat: untouched beyond the base
- `*load=decode_* *` flat: untouched beyond the base

## Risk

Weighted effect ~+1.4% alone, above the margin; stacked on the merge for a submit-worthy pair (~+2.5% over the best). Risk: the 8-row ownership across CTAs (several CTAs hold new keys); RoPE accuracy (compare by rel L2).

## Adversarial review

Reviewed 2026-10-25T212856Z by an adversarial subagent (model opus).

**Verdict: proceed.** Plausible, cleanly attributable single-row change: it removes a 1.5 us node from a single-kernel-dominated 13.2 us row worth 0.70%/us, and only the sliding verify_1 row plus an in-prepare interleaved gate decide it, so no companion is needed. The ~2 us estimate is optimistic. The real risks are the import/compile clock (this adds a 6th load_inline module on a base that was told not to gain one) and the multi-CTA new-key overwrite logic.

### Justification
Mechanism: FINDINGS profiles sliding verify_1 at 13.2 us shape = splitkv_attn_sk_kernel PSEUDO 9.9 us + _pre_attention_kernel 1.5 us. The floor is 0.1 us, so the row is entirely latency/launch-bound and deleting a graph node is the right kind of lever (LESSONS: on 5-15 us rows, deleting a node can matter more than kernel speed). Headline arithmetic checks out: the verify_1 load is about 46*13.2 + 16*30.3 = 1092 us, so -2 us x 46 is about -8.4% of that load, about +1.4% geomean over 6 loads. The expected shapes and directions match TASK weights: only sliding verify_1 moves, everything else is flat beyond the base. WEAKEST LINK: the '~1.8 us node gap' is shape-minus-kernel-sum, and that residual is not a clean gap measure. Single-kernel rows show similar residuals: sliding decode_1 12.1 vs 10.6 = 1.5 us with one kernel, sliding decode_16 15.5 vs 12.7 = 2.8 us, full decode_16 110.8 vs 110.3 = 0.5 us. So most of the 1.8 us likely stays after the fold. What reliably goes is the 1.5 us pre-attn kernel plus a fraction of a microsecond of inter-node gap, minus the in-kernel prologue on the critical path. At B=1 with ~256 tiles over ~132 CTAs (~2 tiles per CTA), nearly every CTA starts a segment and pays the q load + RMSNorm + RoPE before its first S MMA, and new-key CTAs also build K/V rows between cp_wait and the MMA. A realistic outcome is -0.8 to -1.8 us (+0.5 to +1.2% headline), still enough to clear a 1.03x gate (about 0.4 us). Precedent is mixed. The fold at full decode_1 alone was 22.4 -> 21.6 us within CI (full-decode1-sk-two-level-fused-preattn). V2 (fused pre alone) in combo-b1 was 0.883 of master. At B>=2 the fold was do-not-try (0.988x/1.014x), but there the kernel is 100-220 us; here it is 9.9 us. PRIOR: the cited prior is real and the difference is real (RoPE, 8 query tokens, 8 new rows read by many CTAs vs one owner-only row, sliding window). The prior field omits two related attempts. sliding-single-launch-fused-attention (Triton single-launch sliding with pre-attn folded; regression 0.945, lost verify_1) is a whole-design Triton rewrite and does not invalidate this targeted CUDA fold. fused-sliding-preattn-into-window-kernel (queued, hold) targeted the FA3/Triton window path. This is not on the do-not-try list: FUSED_PRE B>=2 is about full decode, and this differs in batch, layer and kernel scale. The new-key design departs from skF. In skF exactly one CTA reads the new row, so it is stored before the ring. Here new key m lies in pseudo-rows m..7, so up to 8 rows' tail tiles (possibly two tiles each when seq-8 straddles a 64-key boundary) are read by several CTAs, hence the smem overwrite. The design is sound only if every CTA overwrites every new-key position it loaded, identified by key index >= seq_b - 8 (not by slot), and the self-check asserts cache_loc[j] == page_table[0, seq-8+j] as _skf_case does.

### Confounders on the current kernel
- merge-perkvhead-v3chain-sv32-tiled (the base, unmeasured, not landed): it moves full decode_1 (V3 chain) and sliding verify_32 (SV32 tiled), so the HEADLINE of this run mixes both changes. It does not touch the sliding verify_1 path (B=1 verify keeps master's pre-attn and PSEUDO route), so the sliding verify_1 row and the in-prepare PATH CHOSEN numbers stay attributable. Its separate run gives the base headline to difference against. It also adds a 5th load_inline module, which this change's 6th module compounds (see open questions).
- restack-sk-pseudo-cluster-plus-v3-fused-preattn (landed the B=1 sliding verify PSEUDO route this change modifies): it is the incumbent, not a confounder of the gain, but its route/grid at B=1 is picked by a prepare-time selector. The FUSED gate must compare against master's routed variant/grid from the SAME run, as specified, and print every FUSED grid spec. Otherwise a grid-selection flip could pass for a fold gain.
- fused-preattention-triton (the _pre_attention_kernel being folded): the numeric reference for the new cache rows. Triton sin/cos vs accurate sincosf differ in 1-6% of E4M3 bytes at large positions (~80000 * inv_freq rad here), so a byte-exact cache comparison would falsely fail. The hypothesis correctly uses rel L2 < 2e-2. No timing confound.

### Interference and A/B needs
No companion is warranted. The change has one routed effect, the sliding verify_1 row at B=1. No other landed hypothesis or base change touches that row's launch sequence, and the hypothesis prints an interleaved cold-L2 head-to-head of [FUSED PSEUDO] vs [pre-attn + master's routed PSEUDO] in prepare. A gain or a loss on that row is therefore readable from the row and the PATH CHOSEN line alone. An ab-revert of the base merge would only disentangle the headline, which the merge's own run already does. An ab-revert of the PSEUDO route is meaningless because the fold is defined on it. A 'q-only fold' control (fold q, keep a separate tiny K/V-store kernel) would split the prologue cost from the K/V-overwrite cost, but it re-adds a node, so a fused-vs-unfused difference would stay ambiguous. It is better done as in-prepare DIAG timings (FUSED with K/V build disabled on a correctness-irrelevant timing-only instantiation) than as a scored run.

### Open questions
- Import budget: with merge-perkvhead-v3chain-sv32-tiled already at 5 modules, could FUSED PSEUDO be built inside an existing module, or shared with the queued sliding-decode-sk-s3-fused-preattn-rope (TOK=1), so the two sliding folds do not each add a module? At minimum, what is the deadline at which muse_sv1_fused_v1 is skipped?
- Does the raw bf16 K/V source for the 8 new rows (8 tokens x 2 kv heads x 64 x (k,v), ~4 KB) get prefetched in the prologue cp.async group, or loaded at the overwrite point between cp_wait and the S MMA? The latter puts a global-load latency on the critical path of every new-key CTA.
- Is the ~1.8 us 'node gap' real? Single-kernel rows show 0.5-2.8 us shape-minus-kernel residuals, so the realistic gain may be nearer -1 us. Print the gate's unfused and FUSED medians so the expected-vs-measured gap can be booked either way.
- The new-key positions are identified by key index >= seq_b - 8. Does the self-check include a len where seq-8 straddles a 64-key tile boundary and the window-start edge cases (lens 2047/2048/2055 are listed; add a len with (seq-8) % 64 == 60)?
- Is the 5.473 q scale (and the e4m3 cast of q at the same point as _pre_attention_kernel) reproduced exactly, and are k_scale/v_scale handled for the kv_scale 0.015625 test (the B=1 route is config-gated to page 1, which no B=1 q_len 8 test hits, so only the self-check covers it)?


## Implementation notes

Built 2026-10-28 in the worktree (base 47926f5e, blob 2eb03f76). The change still applies: master's B=1 sliding verify is `_pre_attention_kernel` + PSEUDO `splitkv_attn_sk_kernel` (the `_sw` route is B=2..32 only), and no fused module existed. The base merge (merge-perkvhead-v3chain-sv32-tiled) is already on master. Diff: `submission.py` +943 lines, 3 hunks.

- **New section** before the import block: module `muse_sv1_fused_v1` (`_fl1` suffix when the live `_EXT2` uses the full-line gather). Its source is `_SKF_HEAD` (the helper head) plus `splitkv_attn_sv1_kernel`, a copy of the stream-K template with one extra flag `FUSED`. Only `<4, ticket, {single, two-level}, PSEUDO, TOK 8, FUSED>` is instantiated, as twins of variants 4/5. The landed `_EXT2` instantiations are untouched. Layout `LayF<4>` is the PSEUDO layout with a 2 KB raw bf16 q slot per stage, a float2 RoPE table [8][32] and the new rows [2][8][128], 106016 B in total. A static_assert checks that 2 CTAs fit per SM.
- **(a) q:** the segment's first tile cp.asyncs pseudo-row r's raw bf16 q row from qkv (rides the tile's group). At consume time the kernel does scaleless RMSNorm (eps), interleaved RoPE pairs qi*2+8s and +1, and the e4m3 cast, then builds `qa`. The (cos, sin) values for the 8 rows x 32 pairs are computed once per CTA in the prologue, overlapped with the page-index wait. The angle is `__fmul_rn((float)positions[r], inv_freq[i])` with master's `_inv_freq`.
- **(b) new K/V:** a CTA builds the 8 rows into smem in the prologue if its tile range holds any new key, or if it is the owner. New key m sits at local position `cnt_j - j - 1 + m` of row j >= m. The rows are k: RMSNorm, RoPE, / k_scale, e4m3, and v: / v_scale, e4m3. In the loop, a tile that holds new keys is identified by key index (not slot). It gets the built rows written over the landed swizzled stage after cp_wait, followed by one `__syncthreads` before the S MMA. The owner is CTA gx-1, because at B=1 flat tile T-1 is pseudo-row 7's last tile. The owner stores the 8 rows to k_cache/v_cache at cache_loc. `_sv1_owner_check` verifies this on the host. kc/vc are non-const only in this module.
- **Route:** `kernel_fn` gets one config-only branch after `_sw_gate`: `_SV1_ROUTE and _sv1_gate(cfg)` (window 2048, q_len 8, page 1, tp8, RoPE, B == 1). `_SV1_ROUTE` is a MappingProxyType frozen at import.
- **Setup** (`_sv1_setup`) runs last, after `_sw_setup` and before the PATH re-print. It needs `_PS_ROUTE[1]`. Start deadlines on the import clock are build 400 s, checks 440 s, gate 455 s and replay 470 s, which answers the review's deadline question. The order is: attrs for both twins, printed next to the landed PSEUDO twins (cubin 90; **local bytes > 0 refused**; warning if under 2 CTAs/SM); then the canary and self-check (B=1 lens 1, 3, 7, 8, 60, 63, 1980, 2047, 2048, 2055, 80000; kv 1 and 0.25; grids 3/p4/p8/SM and the routed grid; both twins). Each self-check case asserts `cache_loc[m] == page_table[0, seq-8+m]` and the owner, compares the output with the FP32 RoPE reference (< 2e-2) and with the master pair (< 3e-2), compares the new rows with `_pre_attention_kernel`'s by rel L2 (< 2e-2), and checks that other rows are unchanged. Every launch is announced and synced. Then come the tickets check, and then the gate at B=1 x 80000. The gate first runs each path eagerly for correctness, then times kernel_fn's real incumbent [pre-attn + PSEUDO routed variant/grid] against FUSED at the routed variant and spec, plus FUSED at p4/p8/sm (printed only), in one `_interleaved_us` call (cold L2, empty graph subtracted). It routes at >= 1.03x, and the PATH CHOSEN line prints both numbers. Last is the two-state `_alt_check` through kernel_fn (refilled inputs, NaN-poisoned outputs, lens 80000/974/1950/...). Any miss leaves B=1 on pre-attn + PSEUDO with one PATH CHOSEN line, and tickets are re-zeroed.
- **Deviation:** "accurate sincosf" is implemented as `rope_sincos`, which writes out CUDA's own sinf/cosf fast path: 3-constant Cody-Waite fma reduction by pi/2 and the Cephes minimax kernels, valid for |a| < 105615 (inv_freq <= 1, positions about 8e4 here). A library `sincosf` compiles in the Payne-Hanek slow path, which has a local-memory frame, and that would trip the spec's own "refuse local bytes > 0" rule. The accuracy is fp32-level and is checked by rel L2 against Triton's rows and the torch reference.
- **Deviation:** the spec's "riding the page-index cp.async group" is implemented as riding the segment tile's cp.async group, as skF does. The raw K/V source (about 4 KB) is loaded in the prologue (the review's open question), not at the overwrite point.
- **Not done:** no q-only/no-KV DIAG instantiation. The review called it optional, and it would cost compile time.
- **Uncertain:** register pressure on top of the PSEUDO kernel's ~121-128 regs, since the q build adds RoPE math. If ptxas spills, the attrs check refuses it, which falls back to master and spends one check for nothing. The import clock at this phase's start is unknown. If the earlier phases run past 400 s, the phase is skipped (printed).

## Rules check

Checked against `git -C .worktrees/muse_attention_decode/sliding-verify1-pseudo-fused-preattn-rope diff`:

- [x] CUDA-graph capturable: the kernel_fn path (`_sv1_gate` -> `_sv1_forward` -> `_sv1_launch`) has no `.item()`/`.cpu()`/`synchronize()`/prints/try. Allocations are sized by grid (config + SM count) and shapes only. The one branch is config-only.
- [x] Nothing derived from `data` is cached across calls. The only module state is `_SV1_EXT`, the frozen `_SV1_ROUTE`, the shared self-resetting int32 PSEUDO tickets (zeroed at import and returned to 0 by every launch) and `_inv_freq` (from config theta, as master). The C++ statics are bool flags. No device pointer, TMA map or state-sized buffer is held.
- [x] Weights: n/a: attention problem, no weights. Activations stay bf16 (qkv in, bf16 out).
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8 e4m3 (new rows / k_scale, / v_scale); queries rounded to FP8 E4M3 exactly once (same point as `_pre_attention_kernel`); softmax and merges fp32.
- [x] Attention sees every key: same per-pseudo-row window [max(0, p-2047), p] and causal ranges as the landed PSEUDO kernel. New keys are replaced in smem, not skipped. No top-k, sparse or block skip.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network. load_inline at import only (torch's own nvcc build, as the 5 existing modules do).
- [x] All work happens inside the timed call; no output caching or replay. The import-time self-check/gate/replay use synthetic inputs only.
- [x] Build: TORCH_CUDA_ARCH_LIST=9.0a (restored after), `-std=c++20`, `-lineinfo`, `-O3`; in-process canary; sync + print after each self-check launch; regs/local/CTAs per SM printed; local > 0 refused.
- [x] Two judge states: the owner CTA writes only cache_loc rows of the state passed in; `_alt_check` two-state capture-once replay runs before routing.

Lint (`okc_lint.py ... --problem muse_attention_decode`): `OK with warnings`. There are 0 errors. The warnings are all pre-existing `host-sync` lines in prepare-time code outside kernel_fn's call graph, plus the new section's own prepare-time syncs (`_sv1_case`, `_sv1_validate`, `_sv1_gate_b1`) and one pre-existing `dtype` warning at L7646. `python3 -I -m py_compile`: OK (`__pycache__` removed).
