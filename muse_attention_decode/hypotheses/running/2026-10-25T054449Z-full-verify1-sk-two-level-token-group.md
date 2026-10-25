---
problem: muse_attention_decode
slug: full-verify1-sk-two-level-token-group
created: 2026-10-25T054449Z
base: master
area: full-layer attention (batch-1 verify, stream-K two-level kernel, token grouping)
change: >
  On the token-group branch (sliding-verify-sk-pseudo-s2-token-group). Full verify_1 is 30.5 us
  shape: FA3 15.4 + combine 7.0 + prepare 2.8 + pre-attn 1.5 + norm-gate. At B=1 the full
  two-level decode kernel is latency-bound (18.6 us for 20 MB, 0.89 TB/s), so adding rows to each
  K/V tile is nearly free there, unlike B>1. Route full verify B=1 (window == 0, q_len 8, tp8,
  page 1) through the two-level ticket kernel with TG in {2,4,8} token groups (window = 0: lo_j =
  0, sp_j = cache_seqlens-7+j, causal per token on the last tile; R = 8/TG blocks over 80000 keys,
  each K/V tile loaded once and consumed by TG warps of 16 rows each). TG=8 reads the 20 MB once;
  MMA work is 8x decode's, ~2.6 GFLOP, a few us on mma.sync. Two-level ticket groups are per
  block. Gate: interleaved graph replay of each TG x grid {SM, 2xSM} vs master FA3 path (prepare +
  main + combine + norm-gate; pre-attention excluded from both), route at >= 1.05x. Supersedes the
  queued full-verify1-sk-pseudo-decode (that is TG=1 here; include TG=1 in the sweep for the
  record). RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation
  sized by input values (shape/config only), no data-dependent Python control flow, no prints, no
  try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8
  E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes. The
  judge prepares TWO states (setup copies: 2) and rotates them: no module/static global may hold a
  device pointer into a state's tensors, a TMA map or a state-sized buffer (master's
  self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, hold no data
  and stay as they are). Every new CUDA route self-checks in prepare against an fp32 reference
  with two states and alternating graph replays with refilled inputs (capture once, refill A,
  replay, refill B, replay, compare each), and on any miss falls back to master's path for that
  band with a PATH CHOSEN line naming the failing case. Build as the landed TUs
  (TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, in-process canary, sync+print after each self-check
  launch). Gates route a band only when the printed interleaved graph-replay timing (cold L2, 256
  MB flush, empty-graph baseline subtracted) shows the new path winning by the stated margin; PATH
  CHOSEN prints both numbers.
expected:
  - shapes: "*layer=full load=verify_1 *"
    direction: faster
    why: 30.5 us shape, ~27 us replaced; TG=8 reads 20 MB once on the latency-bound two-level grid (decode_1 does it in 18.6 us); M2 shape <= 26 us, M3 <= 23 us
  - shapes: "*layer=full load=verify_16 *"
    direction: flat
    why: B>1 untouched (MMA-bound, wgmma line owns it)
  - shapes: "*layer=sliding *"
    direction: flat
    why: as in the parent stage
risk: high
depends_on: sliding-verify-sk-pseudo-s2-token-group
priority: 3
kind: rewrite
controls_for: null
rewrite_stage: 3
replaces: "full verify batch 1 (q_len 8): prepare_varlen_num_blocks + FA3 main + FlashAttnFwdCombine + norm-gate"
review: proceed
status: running
review_at: 2026-10-25T054747Z
review_companions: null
stage: implemented
claimed_at: 2026-10-25T084919Z
base_sha: 39fb54fe3f47f42f763160b2e09426e294bdbe7a
base_blob: 7fd30a78be18303b7c935dc944c31ff33649bed0
branch: hyp/muse_attention_decode/full-verify1-sk-two-level-token-group
worktree: .worktrees/muse_attention_decode/full-verify1-sk-two-level-token-group
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-25T090903Z
impl_commit: 990bf2fe52e2cc0943c18559531970d9a67f2c2b
submission_sha256: 404a5c90967355a77eceb12855212df3d9518e8dc01be95ec5a3011f19475633
norm_sha256: 6e37eb509d458b2cf40dde59667d4cb698e6af7d647d17281d847e7fd9613cb5
submission_blob: 2a3da06aac723bdc92ec4648642f25ec1c2ccc2c
lint_warnings: 41
attempt_file: 2026-10-25T090903Z-check-full-verify1-sk-two-level-token-group.md
prepared_at: 2026-10-25T090903Z
---

# full-verify1-sk-two-level-token-group

## Change

On the token-group branch (sliding-verify-sk-pseudo-s2-token-group). Full verify_1 is 30.5 us shape: FA3 15.4 + combine 7.0 + prepare 2.8 + pre-attn 1.5 + norm-gate. At B=1 the full two-level decode kernel is latency-bound (18.6 us for 20 MB, 0.89 TB/s), so adding rows to each K/V tile is nearly free there, unlike B>1. Route full verify B=1 (window == 0, q_len 8, tp8, page 1) through the two-level ticket kernel with TG in {2,4,8} token groups (window = 0: lo_j = 0, sp_j = cache_seqlens-7+j, causal per token on the last tile; R = 8/TG blocks over 80000 keys, each K/V tile loaded once and consumed by TG warps of 16 rows each). TG=8 reads the 20 MB once; MMA work is 8x decode's, ~2.6 GFLOP, a few us on mma.sync. Two-level ticket groups are per block. Gate: interleaved graph replay of each TG x grid {SM, 2xSM} vs master FA3 path (prepare + main + combine + norm-gate; pre-attention excluded from both), route at >= 1.05x. Supersedes the queued full-verify1-sk-pseudo-decode (that is TG=1 here; include TG=1 in the sweep for the record). RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes. The judge prepares TWO states (setup copies: 2) and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (master's self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, hold no data and stay as they are). Every new CUDA route self-checks in prepare against an fp32 reference with two states and alternating graph replays with refilled inputs (capture once, refill A, replay, refill B, replay, compare each), and on any miss falls back to master's path for that band with a PATH CHOSEN line naming the failing case. Build as the landed TUs (TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, in-process canary, sync+print after each self-check launch). Gates route a band only when the printed interleaved graph-replay timing (cold L2, 256 MB flush, empty-graph baseline subtracted) shows the new path winning by the stated margin; PATH CHOSEN prints both numbers.

## Expected

- `*layer=full load=verify_1 *` faster: 30.5 us shape, ~27 us replaced; TG=8 reads 20 MB once on the latency-bound two-level grid (decode_1 does it in 18.6 us); M2 shape <= 26 us, M3 <= 23 us
- `*layer=full load=verify_16 *` flat: B>1 untouched (MMA-bound, wgmma line owns it)
- `*layer=sliding *` flat: as in the parent stage

## Risk

Depends on stage 2's token-group code; if stage 2 fails to build or self-check, this waits. Risk: 8 row groups per tile need 8 warps or 2 groups per warp; register pressure at TG=8 (print spills, refuse a spill).

## Milestones

Rewrite stage 3 of `full verify batch 1 (q_len 8): prepare_varlen_num_blocks + FA3 main + FlashAttnFwdCombine + norm-gate`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T054747Z by an adversarial subagent (model opus).

**Verdict: proceed.** The design is sound for full verify_1. All 8 tokens share the key range [0, sp_7), so TG=8 adds no extra keys and reads the 20 MB once, and the per-band printed gate makes the full verify_1 row attributable without a companion. The win is thin, about 26 us replaced against a decode_1 kernel that already takes 18.6 us. The likely limits are the 8x larger merge and per-warp MMA serialization, not DRAM.

### Justification
Profile (FINDINGS): full verify_1 is 30.5 us. FA3 main 15.4 + combine 7.0 + prepare 2.8 + norm-gate ~1.1, so ~26.3 us is replaced, plus pre-attn 1.5. The floor is 4.3 us and the weight is 16 in the verify_1 load. The landed two-level sk kernel does the same 20 MB gather for decode_1 in 18.6 us (0.89 TB/s, latency-bound), so a TG=8 variant at about 19-23 us beats the incumbent by the 1.05x gate (<= ~25 us). For window = 0 the token-group union is exact: lo_j = 0 for every token, and tokens differ only in the last 7 keys (masked on the final tile). TG=8 therefore has no extra key range, unlike the sliding s2 case. That makes the full layer a better fit for token grouping than the sliding parent. MMA: 8 tok x 16 heads x 80000 keys x 64 x 4 = 2.6 GFLOP, about 20 MFLOP per SM at grid SM. At about 2.5-3 kFLOP/clk/SM for mma.sync that is ~3.5-4.5 us of tensor work if perfectly overlapped. It is not free on a kernel whose critical path is ~9.5 tiles per CTA. WEAKEST LINK: the landed kernel's 8 warps split each 64-key tile among themselves (smem ACC [WARPS][G][D] plus a cross-warp merge). TG=8 with one row group per warp makes each warp do the whole tile's QK and PV for its token, 8x the per-warp MMA chain per tile. At 2 CTAs/SM it can serialize with the gather instead of hiding under it. Second weakest: every CTA's partial grows from 4 KB to 32 KB (8 tok x 16 x 64 fp32), so the two-level merge moves about 4.2 MB of partials at grid SM. Each finalizing CTA reads about 12 x 32 KB per level. The stage-3/4 diagnostics priced the decode merge alone at several us, so an 8x merge could eat the whole margin unless the finalize spreads tokens over warps. Milestone 3 (<= 1.5x floor = 6.5 us) is unreachable for this design, as it is for decode_1. M2 is the realistic stage target, and it is readable from the profile (the sk kernel row replaces the prepare_varlen, FA3, combine and norm-gate rows). The expected flats (verify_16/32, decode, sliding relative to the parent) are consistent with the gate.

### Confounders on the current kernel
- sliding-verify-sk-pseudo-s2-token-group (depends_on, not yet run): this run inherits its sliding verify routing. Judge sliding rows against the parent's own result, not master/BEST. If the parent has not been harvested first, the headline delta is not attributable to this change. Only the full verify_1 row (named in PATH CHOSEN) belongs to this hypothesis.
- sliding-verify-sk-pseudo-decode-s1 (grandparent, neutral 1.4705x, impl 6f95ce7): the PSEUDO template, separate _ps_* ticket buffers, Lay<STAGES,256,true> and the gate constant window == 2048 come from it. The full route needs a window = 0 / lo = 0 path. If it is added by loosening the shared PSEUDO predicate rather than a separate full-verify gate, sliding rows can change too.
- full-decode1-streamk-s3-two-level-ticket (landed c3c1b01): the two-level kernel being extended. TWO/PSEUDO=false instantiations must stay byte-identical. Compare the printed sk_attrs with the landed 121 regs / 0 local / 95264 B / 2 CTAs per SM, or full decode_1 (weight 16) can drift and look like interference.
- fused-preattention-triton / paired-heads-launch-cleanup-ab-tune-batch-gate (landed): full verify_1 is unpaired (batch > 1 gate). The sk path must consume the PAIRED=False q layout and the 8 new K/V rows already written by _pre_attention_kernel. Excluding pre-attn from both sides of the gate timing is correct.
- fa3-hdim128-head-pairing (landed): affects only B>1 full shapes. It is not a confounder for the B=1 route, but it means the incumbent FA3 call at verify_1 is the unpaired one.

### Interference and A/B needs
No companion. The route is decided in prepare from printed interleaved graph-replay timings of each TG x grid variant against master's FA3 path, with a 1.05x margin. The full verify_1 row is therefore either the new kernel (named with TG and grid) or byte-identical FA3, and TG=1 in the sweep is the built-in control for token grouping. A gain or loss on that row is attributable from the log alone. The ambiguity in this run is inheritance from the unrun parent on sliding rows, not anything a revert or tune companion would resolve. Sequencing (harvest the parent first) fixes it for free. Checks are scarce (8 left, 8 queued), so spend none on companions.

### Open questions
- BUILDER: TG=8 warp mapping. Either keep the landed key-split across 8 warps and loop over 8 row groups per warp (q fragments in smem), or give one row group per warp. Print both regs/local and the per-tile cycle split if cheap. The second doubles per-warp MMA latency per tile, the first keeps the landed pipeline shape.
- BUILDER: finalize/merge with 8 tokens. Spread the two-level merge over warps by token (warp w merges token w) so the 32 KB partial merge parallelizes 8-way. Include grid in {SM/2, SM, 2xSM} so the slot count per merge stays small; the merge, not DRAM, is the likely ceiling.
- Is full-verify1-cuda-streamk-batch1 (queued p2, stacked on the regressed full-verify-cuda-streamk-splitkv) still worth a check? It targets the same row with the 64-row A8 tiles that LESSONS calls MMA-bound. Never stack it with this. Consider retiring it along with the superseded full-verify1-sk-pseudo-decode.
- If stage 2 (sliding s2) fails to build or self-check, this hypothesis should rebase onto s1 (which has PSEUDO with TG=1 implicitly) rather than wait. The full-layer TG code does not need the sliding TG gate to land.


## Implementation notes

Base check: worktree HEAD is 39fb54fe ("impl: sliding-verify-sk-pseudo-s2-token-group"), the parent's impl commit; master's submission.py is unchanged since (only md commits after it), so no 3-way merge was needed. Edited only `muse_attention_decode/submission.py` in the worktree (about +770 lines, mostly the Python gate/self-check; the CUDA change is about 220 lines).

What changed
- Kernel: `splitkv_attn_sk_kernel` gets a 6th template parameter `FV` (0 = landed/PSEUDO, unchanged). New instantiations (module built with `-DSK_PSEUDO=1 -DSK_FULLV=1`): variant 10 = TG 1, 11 = TG 2, 12 = TG 4 (FV=1, reuse the existing TG=1 / token-pair code with `lo = 0`), 13 = TG 8 one 64-key pass per tile, 14 = TG 8 two 32-key passes per tile. All two-level ticket, 4 stages, grid-agnostic.
- Window: a separate FULLV constant `FV_WINDOW = 1 << 30` replaces `PS_WINDOW` only through new template parameters (`load_prefix_pseudo<TG, WIN>` and a `constexpr WINV`); the shared PSEUDO predicate / `_ps_gate` and the sliding 2048 are untouched (the full layer has window 0, lo = 0 for every token, not 2048). The Python route is a new predicate `_fv_gate(cfg)` (window 0, q_len 8, batch 1, page 1, tp8, no RoPE) and a new branch in kernel_fn after the pseudo branch.
- TG = 8 kernel branch (`else if constexpr (QONE)`): warp w = kv head w & 1, token pair w >> 1 (tokens 2p, 2p+1 in mma rows 0-7 / 8-15, as the TG 2/4 code), all 64 keys of each tile, so there is no cross-warp key merge. Partials go from registers through a per-warp smem stage (ring stages st+1, st+2, which hold only consumed tiles at the last tile; stride 66 floats + 1 per 32 dims, conflict-free) to the slot rows as coalesced stores. The first-level and second-level two-level merges are spread over warps by token (warp w merges and finalizes token w; its 32 lanes cover 2 heads x 64 dims per step so `finalize_store`'s 16-lane shuffles still work).
- q for TG 8 uses one smem slot (Lay got a `QS` template parameter, default = STAGES, so every existing layout is unchanged; FULLV TG 8 is 8 KB for q instead of 4 x 8 KB, ~99 KB total, 2 CTAs/SM). The q load is 2 chunks per thread (512 chunks, 256 threads).
- Fixed in passing (no effect on landed instantiations): the `prb` shift for `TG == 8` (`b >> 0`; the old expression gave 1).
- Host: `sk_tg / sk_two / sk_smem / sk_pick / sk_prepare` extended; `sk_attn` TORCH_CHECKs FULLV is batch 1. Build chain `_build_sk`: FULLV+PSEUDO module first; on failure the parent's PSEUDO-only module; then the landed source. So a FULLV compile error cannot take the parent's sliding route down.
- New Python: `_fv_launch` (workspace `(grid + 8/TG) * TG` slot rows, group slots `8/TG * 32 * TG`, all from config / grid only), own self-resetting ticket arrays `_fv_ticket` (16) and `_fv_gticket` (256), zeroed once at import, `_fv_fa3` (master's unpaired full path, `window_size=(-1,-1)`; the sliding `_ps_fa3` would pass (-1, 0)), `_fv_synthetic/_fv_prepare_q/_fv_reference/_fv_plant/_fv_case`, `_fv_validate`, `_fv_time_interleaved`, `_fv_gate_time`, `_fv_two_state_check`, `_fv_setup` (called after `_ps_setup` in its own try/except).

Review points applied
- "Full layer window = 0, so lo = 0 for every token; do NOT reuse the sliding 2048": done as above (FV_WINDOW; no 2048 anywhere on the FULLV path). The review's own text already says the same.
- "Separate full-verify gate rather than loosening the shared PSEUDO predicate": `_fv_gate` / `_FV_ROUTE`, `_ps_gate` and `_PS_ROUTE` are unchanged.
- "Self-check at full ctx > 2048 (80000 and ~5000) against fp32 reference": `_fv_validate` runs lens 80000 and 5003 at grids SM/2, SM, 2xSM (plain, kv_scale 0.25 and planted coherent-style keys that make the softmax peaked, including keys inside the 8 new rows so the last tile's per-token causal mask matters), plus lens 0, 3, 56, 57, 4089 (tile edges, S = 8..4097) and grid 3. TG 1 runs a subset. Rel L2 < 2e-2 against `_fv_reference` (fp32, e4m3 q, no RoPE, full causal), each launch announced before and synced + printed after.
- BUILDER open question "TG=8 warp mapping, print regs/local for the options": mapping A (pair per warp, built) is variant 13 (8 live n8 key blocks) and variant 14 (two 32-key passes, fewer live registers); `_fv_validate` prints regs / local / smem / CTAs-per-SM for both (and refuses any variant that spills: no local memory). Mapping B (landed key split across 4 warps per kv head, looping 4 token pairs per warp) is not built: it needs 4 x 32 fp32 accumulator registers per thread under the 128 cap, a guaranteed spill; this reasoning is printed by `_fv_layout_lines`. I could not compile or run anything here (no GPU, torch or nvcc), so the printed regs/local lines are the first real numbers.
- BUILDER open question "spread the two-level merge over warps by token": done (warp w merges token w at both levels). It does not reduce per-thread merge work (128 (token, head) pairs x 16 lanes / 256 threads = 8 per thread either way); it removes the token loop dependency and uses `#pragma unroll 2` over the head loop for L2 latency overlap. The grid sweep covers {SM/2, SM, 2xSM} and prints CTAs, tiles per CTA, group sizes, partial MB per grid.
- Grids {SM/2, SM, 2xSM}: swept for TG 2/4/8 (TG 1 at SM only, for the record); gate prints every timing.
- "TWO/PSEUDO=false instantiations byte-identical": every change on those instantiations is a constant folded to the same value (`WIN` default = PS_WINDOW, `WINV`, `QONE = false` => the original q-load statement is kept verbatim in the `else` arm, `Lay QS = STAGES`), and the TG=1 and generic TG 2/4 code is textually untouched. `_fv_setup` prints `sk_attrs` of variants 0 and 3 against the landed 121 regs / 0 local / 95264 B / 2 CTAs per SM and flags DRIFT; I cannot verify the SASS offline.
- "workspace sized from config only": see `_fv_launch`. "tickets self-resetting": the last arriver of each group / block resets its own counter as in the landed kernel; the self-check prints the ticket sum after the cases and after the replay checks and drops the route if non-zero.
- "two-state rule self-check": `_fv_two_state_check` prepares two states, captures kernel_fn once on each, then replays A, B, A, B with in-place refills of fresh inputs each time (lens plans 80000 / 5003 / 79321 / 4999, then kv_scale 0.25 with 3 / 70000 / 56 / 12345), output poisoned with NaN, compared with the fp32 reference of that state's current inputs. No device pointer, tensor map or workspace is held in a module global; the only globals are the self-resetting int32 ticket arrays.
- "kernel_fn clean": the new branch is `if fullv: return _full_verify1_sk(...)`: one config-only predicate, one launch, no prints, no sync, no try/except, no input-dependent allocation (`torch.empty` sized by grid / config), nothing cached from inputs.
- Gate: interleaved cold-L2 graph replays (256 MB flush, round-robin with alternating order, 21 reps, median) of every candidate, master's FA3 path (prepare + main + combine + norm-gate, pre-attention on neither side) and an empty-graph baseline (one trivial kernel). Routed only if the best candidate wins by >= 1.05x BOTH raw and baseline-subtracted (raw is the stricter when the new path is faster). PATH CHOSEN prints both numbers.

Deviations / things to know
- TG = 1 (variant 10) is in the sweep for the record only; it reads the 20 MB context 8 times (L2-served after the first) and is expected to lose.
- Single-level ticket FULLV variants were not built (the review and decode_1 data say two-level at B = 1).
- Inherited bug in the PARENT (not touched, outside this stage's area): the not-routed branch of the final PATH CHOSEN loop in `_ps_setup` evaluates `i.get(tgs, ) or none` and `i[t_a]` (undefined names, missing quotes). If any sliding band is not routed, `_ps_setup` raises NameError after setting `_PS_ROUTE`, and the caller's except runs `_PS_ROUTE.clear()`, which silently sends every sliding verify band back to FA3. Judge the parent's own run for this before trusting sliding rows here.
- Compile risk: the FULLV module has 5 more instantiations of a big template (compile time up) and I could only g++ -fsyntax-only the kernel text against a CUDA shim (all three define sets parse and instantiate; nothing was run on a GPU).

## Rules check

- [x] CUDA-graph capturable: kernel_fn's new branch is one config-only predicate plus one `_EXT2.sk_attn` launch (the ticket arrays and `torch.empty` workspaces are shape/config sized); the self-checks `.item()` / `synchronize()` only in prepare-time functions (okc_lint lists them as outside kernel_fn's call graph). Checked in `git diff`.
- [x] Nothing derived from `data` is cached across calls: the diff adds no cache; `_FV_ROUTE` holds (variant, grid) from the device SM count only; tickets hold no data.
- [x] n/a: no weights in this problem.
- [x] Activations BF16 (out bf16); KV cache FP8 e4m3 read as before; q is the pre-attention e4m3 query (no coarser); softmax / merges fp32; P is fp16 for the PV MMA as in the landed kernels.
- [x] Attention sees every key: window 0, every token attends keys 0..p (per-token causal mask on the last tile), checked against the fp32 full-causal reference at 80000 / 5003 and tile edges; no top-k, sparse, block-skip or weight skipping.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network (grep of the diff for subprocess / fork / multiprocessing / Popen: none; the build uses torch load_inline in-process as the landed TUs).
- [x] All work inside the timed call; no output caching or replay.
- [x] sm_90a: the module is built with `TORCH_CUDA_ARCH_LIST=9.0a` (restored after) and `-lineinfo`; the arch flags are printed; this stage adds no wgmma / setmaxnreg (mma.sync only, as the landed kernels).
- [x] Self-check announces each launch before and syncs + prints after; every spin-free (no mbarriers; tickets are not waited on); any miss or unsupported shape lands on master's FA3 path with a PATH CHOSEN line; nothing raises.
- [x] Two-state rule: no module / static global holds a device pointer, TMA map or state buffer (only the self-resetting int32 tickets); self-check alternates two captured states with refilled inputs.

okc_lint output:

```
OK with warnings: muse_attention_decode/submission.py
(only host-sync warnings, all in prepare-time functions outside kernel_fn's call graph)
```

## Pre-flight audit

Audited 2026-10-25 by an opus subagent (read-only plus emulation; there is no GPU, torch or nvcc on the build node). **Verdict: dispatch-with-fixes.**

Verified
- Hand fix of the parent's quote-stripped f-string in `_ps_setup`'s not-routed PATH CHOSEN branch (about lines 3447-3449): the quotes are restored, and `'t_a'` / `'fa3_a'` are always set together in `_ps_gate_band`. A symtable/AST pass over the whole file (every name referenced in any scope, including 268 f-strings, checked against module scope and builtins) finds no undefined names. The same pass on the parent commit flags exactly `tgs`, `none`, `fa3_u`, `t_a` and `fa3_a` in `_ps_setup`, so the checker works. A separate pass found no except handler that reads a local assigned only inside its try.
- Full path: `load_prefix_pseudo<TG, FV_WINDOW>` and `WINV` give lo = 0 for every token, and the masks are t < S - 7 + j, which matches reference.py (keys 0..p). The self-check covers lens 80000 and 5003 at grids SM/2, SM and 2xSM, plus coherent plants inside the 8 new rows and tile edges 0/3/56/57/4089.
- TG=8 kernel mapping: g = warp & 1 and pp = warp >> 1 give 8 distinct (kv head, token pair) pairs. QK/PV fragments, the V dim mapping (p = (qi*2+e)*8 + nb), the staging (only consumed ring stages, no cp.async in flight at the last tile, 8448 B per stage), the slot/ws/gacc indexing, the ticket sizes and the per-token finalize rows were all checked. With gx = 1 it goes through the ticket with ngrp = 1, which is correct.
- Separate gate: `_fv_gate` requires window 0, q_len 8, batch 1, page 1, 16/2/64 heads, rope_theta 0 (reference.py sets 0.0 on full layers) and a validated route. `_ps_gate`/`_PS_ROUTE` are unchanged, so the sliding routes are untouched. Among the judge's tests only the scored full verify_1 shape routes; the B=1 1M test is page 64.
- Landed instantiations: variants 0-9 fold every change to the original constants (QS = STAGES, WIN = PS_WINDOW, original q-load arm, prb unchanged for TG 1/2/4). The printed variant 0/3 attributes witness this.
- No module/static global holds a device pointer: there are only the self-resetting int32 `_FV_TICKETS`/`_FV_GTICKETS` and floats in `_FV_INFO`. The two-state self-check alternates A/B with in-place refills under a NaN poison.
- Every miss falls back without raising: build failure, attributes/spill, self-check, gate and two-state each land on FA3 with a PATH CHOSEN line, and `_fv_setup` is wrapped at module level. kernel_fn's new branch is one config-only predicate plus one launch, with no sync, print, try or input-sized allocation. okc_lint reports only prepare-time host-sync warnings.

Fixes
- `_fv_plant` wrote planted keys with an FP8 advanced-index assignment (`kc[rows, gh] = ... .to(FP8)`, an index_put on float8_e4m3fn that no earlier run had exercised). If that raised, `_fv_validate` would throw and every FULLV variant would drop to FA3, so the run would be wasted. It now writes through a uint8 byte view, which is bit-identical.
- Added `PATH CHOSEN: full verify_1 FA3 (fallback: stage-2 module unavailable)` for the case where the splitkv or stage-2 module fails before `_fv_setup` runs. Before this, that fallback printed no full verify_1 line.

Residual risk: the uncompiled CUDA (5 more template instantiations; a FULLV compile error costs a second PSEUDO build's compile time) and register spills on variant 13. The gate requires 1.05x both raw and baseline-subtracted, which is stricter than the spec's subtracted-only rule and could reject a thin win.
