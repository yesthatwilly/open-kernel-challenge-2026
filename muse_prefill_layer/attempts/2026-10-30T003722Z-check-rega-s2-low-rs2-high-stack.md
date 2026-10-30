# muse_prefill_layer · 2026-10-30T003722Z · check · rega-s2-low-rs2-high-stack

Hypothesis: `muse_prefill_layer/hypotheses/running/2026-10-30T002745Z-rega-s2-low-rs2-high-stack.md` (branch `hyp/muse_prefill_layer/rega-s2-low-rs2-high-stack`).

- **Job:** `2c9af73f6b0f4e1b` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `6287ccb0`

## Goal
Stack two shelved, bit-exact rega arms in master's per-class dq race (prefill master = A5b+D+SC, 1.5721): (1) S2, the 2-CTA/SM low-class tile from experts-rega-low-2cta-a5b (branch hyp/muse_prefill_layer/experts-rega-low-2cta-a5b; judge 074a0e02 routed it at low: S2/A5b .931-.932, guard .968), restacked from its patch with all its gates; (2) RS2 at the HIGH class only (takemyjob hopper/rgx3 2b6d50e; research/h100-rega3-1029.md: wait<1> every group so the next stage's dequant runs under the previous stage's second wgmma group; node pair vs A5b+D+SC at high 0.959-0.963, w2 0.955; low/ft no gain; check ptxas C7510-C7514 and route only if clean), raced at high with the same <= 0.97 vs A5b every round + exact rule. ft keeps A5b+D+SC. Keep the build gate: count instantiations (S2 2 + RS2 2 for high) and probe only the new ones so the 233 s need holds; print the [rega-dq] summary with both arms.

## What changed, and where
**Mechanism:** experts rega dq module: S2 (2-CTA SC tile, restacked byte-for-byte) raced/routed at low, plus RS2 = A5b+D+SC with a rolling split commit (wgmma_wait<1> before each record's dequant, previous stage released after its second group retires) as `rega_kernel<G, false, 5, 2>` on master's 1-CTA 168-reg tile, raced/routed at high only (turn_4 rega rows); ft keeps A5b+D+SC.

- Worktree diff vs HEAD: 475+/95- (S2 restack staged by churn: 336+/88-; RS2 on top, unstaged: 187+/55-), all in `muse_prefill_layer/submission.py`.
- CUDA (REGA_DQ_EDITS): the existing `int S2` template param takes the value 2 for RS2 (no new template axis): `Geo<2> == Geo<0>` (every `S2 ?` became `S2 == 1 ?`, so S2 = 0/1 compile exactly as before; static_assert that SmemS<.,2> is master's smem). consume_item gets an `if constexpr (DQ == 5 && S2 == 2)` MODE-0 loop ported from rgx3 `RGX_SCHED 3, RGX_NG 2` (wait<1>, dq5_rec<0>, fence, 4 wgmma, commit; wait<1>, release prev stage, dq5_rec<1>, fence, 4 wgmma, commit; final wait<0> + fence_acc + release). MODE 2/5 and the epilogue are shared, so RS2 carries master's fast-silu w13, w2st arm a and the non-local zero store (and exactness vs master's dispatched launches checks that). Host: dq 9 accepted; `case 9` launches `<G,false,5,2>` (a BIG table, impossible at high since T <= 4096 gives <= 2078 blocks, falls back to A5b's existing `<G,true,5,0>`, which is also bit-exact, and adds no instantiation). Flag id 9.
- Build gate (review fix): instantiations are 12 + S2 2 + RS2 2 = 16 (`REGA_DQ_INST = 16`). The ptxas probe build (`-DREGA_DQ_PTXAS_PROBE`) now instantiates RS2's 2 only, and S2 moved inside `#if !defined(REGA_DQ_PTXAS_PROBE)` (judge-proven 074a0e02, not re-probed). est = 40 x 18 / 4 = 180 s, need = used + 259 s <= 270 s while used <= 11 s. A local dry count (exec of the file with stubbed imports): info_one main 16 / probe 2; launch tuples {B_,0,0},{B_,3,0},{B_,5,0},{false,5,1},{false,5,2}; assert need(0.1 s) <= 270 passed. The head line logs `need Ns (16+2 inst) <= 270s`.
- Probe parse: entries matched on `Li5ELi2E` (RS2), plus a new `c751x` count of any ptxas line naming C7510-C7514. `_rega_dq_probe_ok`: state ok, 2 entries, 0 spill, c751x 0.
- Race/route: RS2 is raced only at high, with A5b raced, both RS2 attrs ok (master's criteria + 0 local bytes), and A5b's bank (shared, no copy). It is raced even with an unclean probe so the timing is still read, but `_rega_dq_rs2_gate` routes it only when the probe is clean AND A5b passes its own gate AND hs/y are torch.equal to master's dispatched launches on every exactness routing AND pair MODE 0 is <= 0.97 x A5b, x master and x DQ0 in every round and L2 mode on pop and random (threshold not loosened). It then goes first in the order and still needs master's two-state replay check (fresh routings, plan high T=3012) before `routes[high] = 9`; on a replay failure it falls back to A5b. RS2's first launches at high are announced (print, sync, print).
- Prints: ROUTE high carries `RS2 <m>m/<d>d/<b>b<=0.97 eq n/N probe ok|MISS [A5b-miss] pass|miss`. TIME/PAIR lines give RS2 vs DQ0/m/A5b('b') as S2's do. The probe line reads `ptxas probe RS2 ... C7510-C7514 n ... clean|NOT clean`. The attrs line has `RS2 r/l/o`. `_rega_dq_cap` now cuts the SUMMARY head, ROUTE lines, PAIR low/high and the probe line last. A synthetic dry run of `_rega_dq_summary` + cap (all 3 classes, S2 at low, RS2 at high) came to 3281 chars uncapped and 3000 capped, with the head (build/est/need), probe, PAIR high incl. random and every ROUTE line intact. Only the key/SCALES/low RATIOS lines are trimmed.
- Deviations: (1) RS2 reuses the `S2` template int (value 2) instead of a new param, so there is no extra axis. (2) S2 is "pinned" by leaving its template text semantically unchanged and unprobed, not by a source hash: the dq module hash changes anyway with RS2. (3) The cap tiering is a summary-only change. Not done (review: separate change): no REGA_DQ_EST_4INST_S recalibration.
- Biggest risk: node -> judge transfer of RS2 at the 0.97-every-round gate (node val 0.970-0.974; p(route) ~0.4-0.5). Compile risk: ptxas serialising the in-flight register-A wgmma (C7510-C7514). The probe then shows it and RS2 does not route. Build slack is 11 s.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_prefill_layer/rega-s2-low-rs2-high-stack diff` (and `diff HEAD` for the staged S2 restack):
- [x] CUDA-graph capturable: kernel_fn's only change is `dq in (..., REGA_DQ_RS2)` on a prepare-time int; no `.item()`/`.cpu()`/sync/print, no allocation or control flow on input values (all new syncs/prints are in prepare()'s dq phase).
- [x] Nothing derived from `data` is cached across calls (RS2 reads A5b's prepare-time bank copy built from the weights only).
- [x] Weights: repacked losslessly in `prepare()` only (A5b's ENC4/packed-scale copy, inverse-checked; RS2 adds no new repack); NVFP4 stays NVFP4.
- [x] Activations BF16 as before; RS2 is bit-exact by construction (same k order and accumulator chain) and gated on torch.equal vs master; KV cache / queries untouched.
- [x] Attention untouched (n/a: change is in the MoE expert GEMMs).
- [x] No timing patches, harness access, module edits, reference import, subprocesses (load_inline in-process only, as master) or network.
- [x] All work inside the timed call; no output caching or replay (routing chosen in prepare from synthetic routings).
- [x] First line `# okc: problem=muse_prefill_layer` kept; no triton.autotune; sm_90a PTX stays under master's `__CUDA_ARCH_FEAT_SM90_ALL` guard with TORCH_CUDA_ARCH_LIST=9.0a set/restored as before; mbarrier waits bounded (master's mbar_wait).

okc_lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_prefill_layer/submission.py --problem muse_prefill_layer`):
```
OK with warnings: .worktrees/muse_prefill_layer/rega-s2-low-rs2-high-stack/muse_prefill_layer/submission.py
(warnings: host-sync / dtype lines, all "outside kernel_fn's call graph; fine in prepare()" or pre-existing uint8 buffers; no errors)
```
py_compile: ok; __pycache__ removed. Size 494706 B.

## Risks / what could fail
medium
Each part alone is ~+1% headline (below one run); stacked ~+1.5-2%. S2's judge read is direct; RS2's is node-only (A5b-family node->judge transfer held within 0.01-0.02).

## Result (after the run)
- job `2c9af73f6b0f4e1b` (check), status done, passed True, tests 5/5
- speedup 1.5720x, 95% CI [1.5484, 1.5956], score 246436.67391746154 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.5721154841318719, "ci": [1.5485337518698938, 1.5956972163938499], "job": "9f93484606cd4f9e"}; base result: muse_prefill_layer/hypotheses/done/gain/experts-rega-a5b-desc-splitcommit.result.json
- **outcome: neutral** (within-ci)
- files: `muse_prefill_layer/hypotheses/done/neutral/rega-s2-low-rs2-high-stack.result.json`, `muse_prefill_layer/hypotheses/done/neutral/rega-s2-low-rs2-high-stack.patch`; branch `hyp/muse_prefill_layer/rega-s2-low-rs2-high-stack`
