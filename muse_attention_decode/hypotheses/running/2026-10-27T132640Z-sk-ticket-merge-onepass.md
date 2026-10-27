---
problem: muse_attention_decode
slug: sk-ticket-merge-onepass
created: 2026-10-27T132640Z
base: master
area: "stream-K ticket merge (splitkv_attn_sk_kernel merge_range: full decode 1<B<=64, PSEUDO sliding verify_1, sk-window sliding decode_32)"
change: >
  Built on master 2eb03f76 (no depends_on). ONE mechanism, the load structure of the in-kernel
  ticket merge in _SK_CUDA splitkv_attn_sk_kernel. Today merge_range (called through merge_slots
  by the winner CTA of request c_b on the single-level ticket path, and twice on the two-level
  path) makes TWO dependent passes over slots s_lo..s_hi with runtime-bounded loops: pass 1 loads
  every slot m from ws_ml to find M, pass 2 reloads m, l and the float4 of ws_acc and folds with
  ex2(m - Mu). nvcc unrolls such loops about 4x at most, so the winner pays roughly 2*ceil(n/4)
  dependent L2 round trips (n = slots per request: ~17.5 at full decode_16 grid 2xSM, ~9 at full
  decode_32 and at PSEUDO sliding verify_1 grid p8, ~4 at sd sliding decode_32 grid p4) while its
  next stream-K segment, or the kernel end, waits; sk-diag priced the ticket epilogue at 5.8 us
  (B=16) and 2.8 us (B=32) over no-ticket, and DEFER (moving the draw) did not recover it. NEW
  merge_range_onepass<MB> with MB = 8: for c0 = s_lo; c0 <= s_hi; c0 += MB, issue ALL loads of the
  batch first (m and l via __ldcg, the float4 acc via __ldcg; slot c0+u > s_hi gives m = -inf, l =
  0, a = 0), then fold in registers with an online LSE exactly as sw_combine_kernel and c320
  ONEPASS do (Mn = fmaxf(M, m_u); sc = ex2(M - Mn), ex2(-inf) = 0; w = m_u == -inf ? 0 : ex2(m_u -
  Mn); L = L*sc + w*l_u; o = o*sc + w*a_u; M = Mn), returning Mout = the final raw M (-inf when
  every slot is empty) so the two-level group-slot store keeps its meaning. Add template flag bool
  ONEPASS = false to splitkv_attn_sk_kernel that selects merge_range_onepass at every merge site;
  finalize_store and everything else unchanged. Instantiate ONLY three new twins in sk_pick: 8 =
  <4, ticket, ONEPASS> (full decode 1<B<=64), 9 = <4, ticket, PSEUDO, ONEPASS> (sliding verify
  B=1), 10 = <4, ticket, PSEUDO, TOK=1, ONEPASS> (sliding decode B=17..32); variants 0-7 stay
  byte-identical (grow sk_prepare done[] and sk_smem). Print sk_attrs (regs, local bytes) for 8-10
  beside 0/4/6; if a twin spills where its incumbent does not, rebuild with MB = 4. IN-PREPARE
  GATE per routed cell, after that cell incumbent route is decided: run the twin through the
  incumbent own correctness cases (_sk_case set for 8; _ps_case at the routed grid for 9; _sd_case
  at the routed grid for 10) and its two-state replay check, then time incumbent vs twin
  interleaved through the real pair (pre-attention + kernel; cold-L2 flush, empty graph
  subtracted, two rounds of 21 reps) at B=16 and B=32 (full decode bands 2..24 and 25..64), B=1
  (PSEUDO) and B=32 (sd band 3); route the twin for a cell only if it wins >= 1.01x in BOTH rounds
  with IQR <= 1.5x the incumbent (0.5 us floor), by swapping the variant index in that cell frozen
  table (_SK_VARIANT becomes an import-frozen per-band MappingProxyType read by _full_decode_sk;
  the _PS_ROUTE / _SD_ROUTE entries take 9 / 10). Print PATH CHOSEN: sk ticket merge [cell]:
  onepass|incumbent (t_inc vs t_one us both rounds, slots per request). Full decode_1 (c320
  chain), FA3, sw and cluster paths unchanged. Same fp32 arithmetic up to summation order: KV FP8,
  E4M3 q, every key attended.
expected:
  - shapes: "*layer=sliding load=verify_1 *"
    direction: faster
    why: "PSEUDO kernel 9.9 us, ~9 slots per pseudo-row at grid p8: ~6 merge round trips -> 2 at each segment-end merge; -0.5..-1.0 us at 0.68%/us = +0.35..0.7% headline"
  - shapes: "*layer=full load=decode_16 *"
    direction: faster
    why: "~17.5 slots per request: ~10 round trips -> 3 on the winner; -1..-2 us of sk 104.1 us at 0.112%/us = +0.1..0.2%"
  - shapes: "*layer=full load=decode_32 *"
    direction: faster
    why: "~9 slots: -0.5..-1 us of 191 us at 0.074%/us (small, may read flat)"
  - shapes: "*layer=sliding load=decode_32 *"
    direction: flat
    why: "sd route at p4 has ~4 slots (2 round trips -> 1): -0.1..-0.3 us at 0.21%/us, inside noise"
  - shapes: "*load=decode_1 *"
    direction: flat
    why: c320 chain and PERHEAD cluster untouched
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
prior: "c320-combine-onepass: the same one-pass batched-load online-LSE fold (routed 1.119x on the full decode_1 combine, landed) applied to the stream-K ticket merge inside splitkv_attn_sk_kernel, which still does two dependent runtime-bounded passes; full-decode-sk-s5-deferred-ticket-fault-fix (neutral) moved WHEN the ticket draws (DEFER), not how the winner loads the slots"
review: proceed
status: running
review_at: 2026-10-27T134016Z
review_companions: null
stage: implemented
claimed_at: 2026-10-27T134016Z
base_sha: bb6c492319e7d6ef89caaf609741556209472f17
base_blob: 2eb03f76ceefa9bf81e28616ced3b33be3a70443
branch: hyp/muse_attention_decode/sk-ticket-merge-onepass
worktree: .worktrees/muse_attention_decode/sk-ticket-merge-onepass
mode: null
restack_of: null
stale_base: null
implemented_at: 2026-10-27T135306Z
impl_commit: 3cd1be97015371e42732d3caa9018a11ee7d341e
submission_sha256: 73665e96b0abe161442fe2108d35e318c50eaa59f67968888064ae559ccea178
norm_sha256: 972c35a3e910948f5d2cc91f8c5182a50e243270e769c217b3f14ae1a2bfffdb
submission_blob: faed6a62e4ecaed3859d1ed77eaabcb22951d8fd
lint_warnings: 101
---

# sk-ticket-merge-onepass

## Change

Built on master 2eb03f76 (no depends_on). ONE mechanism, the load structure of the in-kernel ticket merge in _SK_CUDA splitkv_attn_sk_kernel. Today merge_range (called through merge_slots by the winner CTA of request c_b on the single-level ticket path, and twice on the two-level path) makes TWO dependent passes over slots s_lo..s_hi with runtime-bounded loops: pass 1 loads every slot m from ws_ml to find M, pass 2 reloads m, l and the float4 of ws_acc and folds with ex2(m - Mu). nvcc unrolls such loops about 4x at most, so the winner pays roughly 2*ceil(n/4) dependent L2 round trips (n = slots per request: ~17.5 at full decode_16 grid 2xSM, ~9 at full decode_32 and at PSEUDO sliding verify_1 grid p8, ~4 at sd sliding decode_32 grid p4) while its next stream-K segment, or the kernel end, waits; sk-diag priced the ticket epilogue at 5.8 us (B=16) and 2.8 us (B=32) over no-ticket, and DEFER (moving the draw) did not recover it. NEW merge_range_onepass<MB> with MB = 8: for c0 = s_lo; c0 <= s_hi; c0 += MB, issue ALL loads of the batch first (m and l via __ldcg, the float4 acc via __ldcg; slot c0+u > s_hi gives m = -inf, l = 0, a = 0), then fold in registers with an online LSE exactly as sw_combine_kernel and c320 ONEPASS do (Mn = fmaxf(M, m_u); sc = ex2(M - Mn), ex2(-inf) = 0; w = m_u == -inf ? 0 : ex2(m_u - Mn); L = L*sc + w*l_u; o = o*sc + w*a_u; M = Mn), returning Mout = the final raw M (-inf when every slot is empty) so the two-level group-slot store keeps its meaning. Add template flag bool ONEPASS = false to splitkv_attn_sk_kernel that selects merge_range_onepass at every merge site; finalize_store and everything else unchanged. Instantiate ONLY three new twins in sk_pick: 8 = <4, ticket, ONEPASS> (full decode 1<B<=64), 9 = <4, ticket, PSEUDO, ONEPASS> (sliding verify B=1), 10 = <4, ticket, PSEUDO, TOK=1, ONEPASS> (sliding decode B=17..32); variants 0-7 stay byte-identical (grow sk_prepare done[] and sk_smem). Print sk_attrs (regs, local bytes) for 8-10 beside 0/4/6; if a twin spills where its incumbent does not, rebuild with MB = 4. IN-PREPARE GATE per routed cell, after that cell incumbent route is decided: run the twin through the incumbent own correctness cases (_sk_case set for 8; _ps_case at the routed grid for 9; _sd_case at the routed grid for 10) and its two-state replay check, then time incumbent vs twin interleaved through the real pair (pre-attention + kernel; cold-L2 flush, empty graph subtracted, two rounds of 21 reps) at B=16 and B=32 (full decode bands 2..24 and 25..64), B=1 (PSEUDO) and B=32 (sd band 3); route the twin for a cell only if it wins >= 1.01x in BOTH rounds with IQR <= 1.5x the incumbent (0.5 us floor), by swapping the variant index in that cell frozen table (_SK_VARIANT becomes an import-frozen per-band MappingProxyType read by _full_decode_sk; the _PS_ROUTE / _SD_ROUTE entries take 9 / 10). Print PATH CHOSEN: sk ticket merge [cell]: onepass|incumbent (t_inc vs t_one us both rounds, slots per request). Full decode_1 (c320 chain), FA3, sw and cluster paths unchanged. Same fp32 arithmetic up to summation order: KV FP8, E4M3 q, every key attended.

## Expected

- `*layer=sliding load=verify_1 *` faster: PSEUDO kernel 9.9 us, ~9 slots per pseudo-row at grid p8: ~6 merge round trips -> 2 at each segment-end merge; -0.5..-1.0 us at 0.68%/us = +0.35..0.7% headline
- `*layer=full load=decode_16 *` faster: ~17.5 slots per request: ~10 round trips -> 3 on the winner; -1..-2 us of sk 104.1 us at 0.112%/us = +0.1..0.2%
- `*layer=full load=decode_32 *` faster: ~9 slots: -0.5..-1 us of 191 us at 0.074%/us (small, may read flat)
- `*layer=sliding load=decode_32 *` flat: sd route at p4 has ~4 slots (2 round trips -> 1): -0.1..-0.3 us at 0.21%/us, inside noise
- `*load=decode_1 *` flat: c320 chain and PERHEAD cluster untouched

## Risk

Weighted effect ~+0.5..1.0%: below the ~1.2% one-run margin, so on its own it lands only if sliding verify_1 reaches the top of its range; otherwise it lands stacked (combo-tail-onepass-push-fastepi). Judge it on the PATH CHOSEN ratios and the splitkv_attn_sk_kernel rows (verify_1 sliding 9.9 us, decode_16 full 104.1 us, decode_32 full 191.0 us, decode_32 sliding 13.0 us). Risks: register pressure at the merge site under launch_bounds(256, 2) (the 8-slot batch holds 48 live regs of loads; MB = 4 fallback); three extra big instantiations add compile time (print the [sk] compiled-in line vs master, keep the new gate phases behind the import-clock guard, deadline 430 s). Composes with the queued sliding-verify1-pseudo-fused-preattn-rope (prologue) and sk-full-line-5stage-select (same per-band variant table: whichever lands second re-keys it).

## Adversarial review

Reviewed 2026-10-27T134016Z by an adversarial subagent (model opus).

**Verdict: proceed.** A real, untested mechanism: the ticket winner loads its slots in one batched pass instead of two dependent runtime-bounded passes. Neither prior covers it (c320 ONEPASS was a standalone combine kernel; DEFER only moved the ticket draw). Its in-prepare interleaved A/B makes one run readable without companions; judge it on the PATH CHOSEN ratios and the splitkv_attn_sk_kernel rows, not the headline (+0.5..1.0% expected, below the ~1.2% margin).

### Justification
merge_range runs pass 1 (an __ldcg of m per slot to find M) then pass 2 (m, l and float4 acc per slot), both runtime-bounded: ~2*ceil(n/4) dependent L2 round trips on the winner CTA. The batched form routed c320-combine-onepass at 1.119x (18.94 -> 16.93 us) and moved the stage-1 combine row 8.2 -> 6.0 us. It applies to full decode B=2..64 (v2 <4,ticket>, grid 2xSM), sliding verify_1 (PSEUDO, grid p8) and sliding decode B=17..32 (PSEUDO TOK=1, grid p4). %/us coefficients checked: 0.68 (sl verify_1), 0.112 (fd16), 0.074 (fd32), 0.21 (sl decode_32). The strongest case is sliding verify_1 (latency-bound tail). Weakest link: how much of the ticket gap is merge loads vs fence/atomic/finalize is unmeasured, and the 'nvcc unrolls ~4x' premise is unverified. The prior omits splitkv-combine-wide-mlp (do-not-try; same idea on a different kernel; not a duplicate, and its kernel row supports this).

### Confounders on the current kernel
- full-decode1-streamk-s3-two-level-ticket: the host of the change; slot count n set by its grid choice.
- restack-sk-pseudo-cluster-plus-v3-fused-preattn: FUSED_PRE re-gates every run (~0.99-1.01x); the twin-8 gate must run after _skf_setup and skip any band in _FUSED_ROUTE.
- restack-sliding-decode-sk-window-on-cluster-master: the sd grid was picked with the two-pass merge; gating twin 10 only at p4 underestimates it.
- sk-full-line-gather: the live sk module is SK_FULLLINE when its canary passes, else the half-line rebuild; twins compile into whichever is live.

### Interference and A/B needs
No companion; the in-run A/B (incumbent vs ONEPASS twin, interleaved cold-L2, 2 rounds x 21 reps, 1.01x both rounds, IQR guard) attributes it. Optionally print an MB=4 twin readout in the same gate to separate pass fusion from batch depth.

### Open questions
- Is merge_range unrolled/load-batched in SASS?
- Register count / local bytes at launch_bounds(256,2) with MB=8; instantiate MB=4 twins up front if the compile budget allows, or pick MB from compile-time attrs.
- Compile/prepare budget: master's import clock ends ~295 s; print [sk] compiled-in time vs master; keep the gate behind the clock guard.
- Print (do not route) twin-10 / twin-9 readouts one grid step up.
- Run the whole phase after _sw_setup (after _ps_setup, _skf_setup and _sd_setup).


## Implementation notes

Worktree `.worktrees/muse_attention_decode/sk-ticket-merge-onepass`, file `muse_attention_decode/submission.py` only (+645 / -17). Not compiled locally (this node has no nvcc).

What changed, and where:
- `_SK_CUDA` helper head: new `merge_range_onepass<MB>` after `merge_slots`. For each batch of MB slots it issues every `__ldcg` load first (m, l, float4 acc; a slot past `s_hi` reads as m = -inf, l = 0, a = 0). It then folds with c320 ONEPASS's -inf handling exactly, in the per-batch form: gm = max of the batch's m, Mn = max(M, gm), Mnu = (Mn == -inf) ? 0 : Mn, alpha = (M == -inf) ? 0 : ex2(M - Mnu) applied once per batch to L and o, and w_u = (m_u == -inf) ? 0 : ex2(m_u - Mnu). It returns Mout = the final raw M (-inf with L = 0 and o = 0 when every slot is empty), so the two-level group-slot store keeps its meaning.
- `splitkv_attn_sk_kernel`: new last template parameter `int ONEPASS = 0`. All three merge sites (single-level `merge_slots`, two-level group merge, two-level final merge) are now `if constexpr (ONEPASS > 0) merge_range_onepass<ONEPASS>(...) else <the old call>`. With ONEPASS = 0 the code variants 0-7 compile from is unchanged. Nothing else in the kernel changed, and `finalize_store` is untouched.
- Explicit per-variant table `SK_V_PSEUDO / SK_V_TOK / SK_V_TWO[SK_NV = 12]` plus `sk_has(v)`, which replace the inferences. `sk_attn` previously used `pseudo = v >= 4`, `tok = v >= 6 ? 1 : 8`, `v in {3, 5, 7}` and the `SK_MAX_VARIANT` bound; `sk_smem` used `v >= 4`; `sk_attrs` used the `SK_MAX_VARIANT` bound. For 0-7 the table gives the same values as before. `sk_prepare` `done[]` grows to 12.
- New `sk_pick` twins:
  - 8 = `<4, true, false, false, 8, 8>`: non-PSEUDO, so it has the Lay<4> layout and the non-pseudo checks, and it is built in both the PSEUDO and the fallback build.
  - 9 = `<4, true, false, true, 8, 8>`: checked as TOK 8.
  - 10 = `<4, true, false, true, 1, 8>`
  - 11 = `<4, true, false, true, 8, 4>`: the MB = 4 readout. It is never routed.
- New `sk_merge_test_kernel<MB>` (MB 8 and 4) and an exported `sk_merge_test`, used as a prepare-time canary.
- Python:
  - `_SK_VARIANT` is now `MappingProxyType({2: 0, 3: 0})`, keyed by `_fused_band`. `_full_decode_sk` reads `variant = _SK_VARIANT[_fused_band(batch)]`, and its `variant == 1` combine branch is kept.
  - `_SK_LABELS` gains 8-11.
  - `_build_sk_one` prints `[sk] compiled-in ...`: the time, the number of instantiations against master (12 vs 8 PSEUDO, 5 vs 4 fallback) and the import clock.
  - `_say` gains a `_SAY_MUTE` flag (PATH CHOSEN lines are always printed). The gate uses it only around its replay checks.
- New phase `_op_setup` (log tag `[op]`), called after `_sw_setup` and before the `[paths]` reprint. That puts it after `_ps_setup`, `_skf_setup` and `_sd_setup`, so every cell's incumbent is final. It does the following:
  1. Prints cubin / regs / local B / smem / CTAs-per-SM for v0, v4, v6 next to the twins v8-v11.
  2. Runs the merge canary at MB 8 and 4. Crafted slots cover a single slot, a single empty slot, n = 3 / 8 / 12 / 17 / 20 / 48, leading empty slots (at MB 4, a whole first batch is empty), every slot empty, and random per-head -inf. Empty slots carry finite garbage l / acc. The canary requires M bit-exact against `merge_range`, L and o within 1e-4 relative, and (-inf, 0, 0) for the every-slot-empty case.
  3. Gates four cells in this order: sliding verify B=1 (twin 9), full decode B=2..16 (twin 8, at B=16), full decode B=17..64 (twin 8, at B=32), sliding decode B=17..32 (twin 10, at B=32).
  - Each cell:
    - It is skipped if the twin has local-memory spill, a non-90 cubin or 0 CTAs/SM.
    - A full-decode band in `_FUSED_ROUTE` is skipped.
    - The PS / SD cell is gated only if the incumbent is the single-level variant (4 / 6).
    - The twin runs the incumbent's own correctness cases: the `_sk_validate` set for twin 8, and `_ps_validate` / `_sd_validate`-style sets at the routed grid plus grids 3 / 7 / 2xSM for 9 and 10. The runner prints one line per set.
    - Then the two-state replay check runs through `kernel_fn` with the twin routed temporarily (`_alt_check` / `_ps_replay_check` / `_sd_replay_check`), and the route is restored afterwards.
    - Then incumbent vs twin are timed through the real pair (pre-attention + kernel) with `_sw_rounds`: interleaved, cold-L2 read flush, empty graph subtracted, 2 rounds x 21 reps, each round with fresh graphs. The twin is routed only if incumbent / twin >= 1.01x in both rounds and the twin's IQR <= 1.5 x max(incumbent IQR, 0.5 us).
    - Print-only readouts: MB-4 twin 11 is timed in the same rounds in the verify B=1 cell. Twin 9 vs 4 and twin 10 vs 6 get one more round one grid step up.
  - Clock guard: checks start only at or before 430 s, timing at or before 460 s, and a win whose timing ends after 480 s is discarded.
  - Winners are swapped in at once at the end: `_SK_VARIANT` is re-frozen with band -> 8, `_PS_ROUTE[1] = (9, spec)`, and `_SD_ROUTE` is re-frozen with 3 -> (10, spec). If any ticket is left non-zero, every win is dropped. Tickets are re-zeroed in `finally`.
  - The phase prints one `PATH CHOSEN: sk ticket merge [cell]: onepass|incumbent (r1 t_inc vs t_one us (x), r2 ..., IQR, ~slots/request ...)` line per cell, capped at 400 chars, and the `[paths]` reprint repeats them last.

Deviations from the spec, and why:
- ONEPASS is an `int` template parameter that carries the batch depth MB (0 = two-pass) instead of a `bool`. This lets the MB-4 readout twin be instantiated up front with no runtime rebuild (review items 4 and 6).
- The full-decode cells use the `_fused_band` partition (B=2..16 gated at B=16, B=17..64 gated at B=32) instead of 2..24 / 25..64. With this partition, the review's "skip any band routed to FUSED_PRE" is exact per band, and the table key is the same predicate `kernel_fn` already uses.
- An MB-4 twin exists only for sliding verify B=1 (variant 11). That is the strongest expected cell, and keeping to one MB-4 twin holds compile cost at 4 extra large instantiations instead of 6. The full and sd cells print "MB-4 readout n/a".
- A twin that spills is not routed, and there is no MB-4 rebuild fallback (review item 6, "no runtime rebuild").
- Leading-empty and every-slot-empty cases cannot occur in the real kernel, because every stream-K segment holds at least one valid key. They are covered by the crafted-workspace merge canary, which compares against `merge_range` and runs before any twin launches on real data.
- The FUSED_PRE module's source now also contains the unused `merge_range_onepass` template, because `_SKF_HEAD` is a prefix slice of `_SK_CUDA`. The FUSED kernel's code is unchanged.

Uncertainties:
- CUDA was not compiled here.
- The added nvcc time for 4 large instantiations is unknown. Read the `[sk] compiled-in` line against master's `[sk] load_inline (muse_decode_sk_v4p_fl1) compiled in` line. A large increase could push `_sw_setup`'s 330 s build deadline.
- MB = 8 holds about 48 load registers at launch_bounds(256, 2), next to an incumbent at 121 regs, so a spill is possible. If the twin spills, its cell reads "not routable: local-memory spill". The v11 attributes line shows whether MB 4 fits.

## Rules check

Checked against `git -C .worktrees/muse_attention_decode/sk-ticket-merge-onepass diff`:

- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints in kernel_fn's path. The only kernel_fn-path change is `_full_decode_sk` reading `_SK_VARIANT[_fused_band(batch)]`, a lookup on config only. Every new sync / `.item()` is in `_op_*` at import (lint: "outside kernel_fn's call graph").
- [x] Nothing derived from `data` is cached across calls. The route tables are frozen at import from synthetic gate data. `_OP_FULL_CHECKED` holds only a prepare-time pass/fail string.
- [x] Weights: n/a: this problem has no weights. Nothing is repacked, and the KV cache is untouched.
- [x] Activations BF16 where SGLang keeps BF16, KV cache FP8, queries e4m3 (same pre-attention kernel). The merge is the same fp32 LSE arithmetic up to summation order.
- [x] Attention sees every key. Only the merge's load structure changed, and every slot s_lo..s_hi is loaded and weighted. A slot is weighted 0 only when its m is -inf, exactly as merge_range does. No skipping, top-k or sparsity.
- [x] No timing patches, harness frames / I/O / secrets, module edits, reference import, subprocesses or network. `_SAY_MUTE` only gates this file's own `_say`.
- [x] All work happens inside the timed call; no output caching or replay. The twins are used at runtime only when routed, through the same launch path.
- [x] First line `# okc: problem=muse_attention_decode` kept; no triton.autotune added.

Lint (`okc_lint.py ... --problem muse_attention_decode`): `OK with warnings`. That is 101 warnings against master's 95. The only new ones are the six prepare-time host-sync lines in the `[op]` phase: 4 x `torch.cuda.synchronize()` in `_op_merge_canary` / `_op_run_cases` / `_op_eager_ok`, and the two lines of the end-of-phase ticket sum `.item()`, each tagged "(outside kernel_fn's call graph; fine in prepare())". Master's pre-existing set is unchanged, including `warn dtype L7786 ... .view(FP8) (allowed only where SGLang does the same)`.

```
OK with warnings: muse_attention_decode/submission.py
(new vs master)
warn  host-sync    left = (int(_sk_ticket(dev).abs().sum().item()) + int(_sk_gticket(dev).abs().sum().item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    + int(_ps_ticket(dev).abs().sum().item()) + int(_ps_gticket(dev).abs().sum().item()))  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())   x4 (L9624, L9652, L9658, L9681)
```

`python3 -I -m py_compile`: OK. The `__pycache__` it left in the worktree has been removed.
