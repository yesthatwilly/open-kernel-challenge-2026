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
stage: claimed
claimed_at: 2026-10-28T145301Z
base_sha: 47926f5e7d8a192184e21177901be2034d2d4930
base_blob: 2eb03f76ceefa9bf81e28616ced3b33be3a70443
branch: hyp/muse_attention_decode/sliding-verify1-pseudo-fused-preattn-rope
worktree: .worktrees/muse_attention_decode/sliding-verify1-pseudo-fused-preattn-rope
mode: null
restack_of: merge-perkvhead-v3chain-sv32-tiled
stale_base: null
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

