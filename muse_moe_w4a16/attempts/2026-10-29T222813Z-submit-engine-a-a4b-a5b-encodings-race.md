# muse_moe_w4a16 · 2026-10-29T222813Z · submit · engine-a-a4b-a5b-encodings-race

Hypothesis: `muse_moe_w4a16/hypotheses/done/gain/2026-10-29T215527Z-engine-a-a4b-a5b-encodings-race.md` (branch `hyp/muse_moe_w4a16/engine-a-a4b-a5b-encodings-race`).

- **Job:** `24f183ff300949f2` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `f994ecf5`

## Goal
Promote check-proven gain engine-a-a4b-a5b-encodings-race (2.0646x [2.0336,2.0955], check job 702f1004b45f4bfa) to the board.

Add two lossless dequant arms to moe's Engine A kernels (ea6 K1/K2 at T=1; ea4x / v4x-family w13 and w2 at T=8/16/32; the cold-half ea4x at T=128/256), ported from takemyjob hopper/ea-enc-1029 d6f55c5 (probe njenc; research/h100-ea-enc-1029.md): E4 = A4b code re-encoding of the 4-bit codes in a prepare-time permuted copy of the bank (nibble->bf16x2 9 ops/word vs 11), E5 = E4 + A5b packed 7-bit (b-8) scale fields (one LOP3 per step; requires every scale byte in [0x08,0x7E], scanned in prepare). Both are value-identical to master's dq2 (bit-exact). Compile them as template variants of the existing kernels in a separate background-built extension (or as template args in the existing ones if the build budget allows), and route PER CLASS (T=1, 8, 16, 32, 128-cold, 256-cold) by an in-prepare interleaved race through kernel_fn (read-flushed, like the existing class races): pick E4 or E5 only if torch.equal to master on every draw, <= 0.98 vs master in every round, and replay-exact; else master. Expected picks from the node: E4 at T<=32 and T=1, E5 at the cold half. Keep master's routing, fork, hot half (wh5ds) and PDL unchanged. Print an [ea-enc] summary (<= 1.5 KB) at the end of prepare: build s, regs/spill per arm, scale scan, per-class per-round ratios E4/m and E5/m, exactness, PATH CHOSEN. Size: moe is 993,167 B (cap 1,000,000): trim dead diagnostics first; report wc -c.

## What changed, and where
**Mechanism:** lossless A4b code (E4) / A4b+A5b scale (E5) dequant arms of Engine A's consumers (v4x twin ea4x_w13/w2 and FAST v6 K1/K2), built as a separate module pair from master's CUDA strings, raced per judged T (1 K1xK2, 8, 16, 32, 128, 256 cold half) through kernel_fn and routed only where exact and <= 0.98 every round.

Ported from takemyjob hopper/ea-enc-1029 d6f55c5 (patch_enc.py / encb.py / enc_check.py), rebased onto master 37ea90fc (wh5ds landed). Only `muse_moe_w4a16/submission.py` changed.

- **CUDA (no second copy of ea4x/ea6 text).** `_eae_src4x()` = `_ea4x_source()` and `_eae_src6()` = `"#define EA6_FAST 1\n" + _ea6_source()`, i.e. master's own strings, plus `_EAE_COMMON` (ext_a4b<I>, rtv) inserted into ea_common, `_EAE_CONSUME` (one `consume_e<DQ, XB, NT>`, covering consume XB 0/1 and consume2's two tiles; DQ 3 = A4b + dq2 paired scale decode, DQ 4 = A4b + A5b LOP3) before `eak::consume`, and `if constexpr (DQ >= 3)` branches in consume/consume2. The changes are anchored `str.replace` calls with exact counts; an anchor miss fails that build, and that T stays on master. Host replacements keep only the E instances: ea4x `<3,false>` / `<4,false>` (pdl bit 4096 = E5) and ea6 `<W, D, 2, true, 3|4>` (new template arg `EQ = 2`; flags bit 4096 = E5). Neither kernel reads bit 4096. I checked offline (AST-extracted strings): every anchor applies, the E sources are 101,345 B and 56,607 B, and consume_e passes `g++ -std=c++20 -fsyntax-only` against stubs. Master's 40 module-level strings are byte-identical, and the v4x name recomputes to `okc_campaign_engine_a_v4x_5a6b0753a8`; ea6, es1a, wh and stk are unchanged by construction (no string or name function touched).
- **Build:** `_eae_start` runs at `_prepare_main`'s first line: two non-daemon threads (`eaenc-4x`, `eaenc-6`), each `_rt_build` (sm_90a, -O3 -std=c++20 -lineinfo; the arch flags print on compile). `_eae_run` joins them with a hard deadline of EAE_JOIN_S = 220 s into prepare. A thread still alive means no race for that module (master).
- **Prepare-time copies:** a scale scan (min/max byte, count outside [0x08,0x7E]; E5 OFF everywhere if any). `_eae_recode` (encb's GPU recode) makes the A4b copy, plus the A5b copy when eligible, of `w13r`/`w2r` into `ea['w13e3'|'w2e3'|'w13e4'|'w2e4']` (~906 MB each). Copies no route uses are freed after the picks.
- **Routes (shape and prepare-time values only):** `ea['on'][T] = '<base>:e3|e4'` (new `_EAE_CORES` entries in `_ea_forward`'s table; `_ea3_forward_core` on `ext_key='exte'`, the E records, the twins' "x" buffers). At T=1 it is `'<FAST arm>:e<k1><k2>'` (`_ea6_forward_core` gains `enc=(k1,k2)`, which picks module and records per kernel). At T=256 it is `state['hm']['cold_e'][256] = 3|4` (`_hm_experts`' xarm block now calls `_eae_pair`, the same two launches as before when unset). E applies only where `_eae_base` finds master's route mirrorable: a v4x s twin, the hybrid's v4x s cold half, or a FAST K1+K2 v6 arm.
- **Race (`_eae_class`, all inside prepare):** per T, the p10/p50/p90 draws of `_ea4x_draws` (local-expert counts at T<=32; T=32 runs on the v4cxs bits-112 layout because the base's bits are used; `_ea_judge_routings` at 128/256). Per draw:
  - exactness: h and y on NaN-poisoned buffers (NaN positions must match, values torch.equal), plus kernel_fn's out eager and from the graph after the timed replays, all vs master's eager kernel_fn;
  - timing: EAE_ROUNDS 5 x EAE_REPS 15 interleaved, read-flushed `_rt_time` rounds of kernel_fn graphs (at 256 the isolated cold-half pair is timed in the same interleave);
  - after timing, tk13x/tk2x/errx and ea['err'] must all be 0.

  An arm is picked only if it is exact everywhere and its E/m <= 0.98 in every (draw, round); among those the lowest median wins. T=1 races all 8 K1 x K2 combinations (E4/m, m/E4, E4/E5, ...), so the K1-E + K2-B arms are included. The race stops starting classes after 20 s or 255 s into prepare. A tensor-cloned state copy (`_wg_copy_tree`) must give kernel_fn outputs equal to the original's at each picked T, or every pick reverts. Any exception reverts everything (`_eae_off`).
- **[ea-enc] block,** prepare's last prints (<= 400 chars/line, ~1.2-1.5 KB):
  - build s per module, join wait, phase-at;
  - regs/lmem/occ per E instance (K1e4 K1e5 K2e4 K2e5 for 4x and 6), recode s and MB, scale scan;
  - per T: exact y/n and median[min,max] of per-round E/m per arm (T=1 K1 and K2 separately and combined), plus T=256 `cold iso` ratios;
  - PATH CHOSEN per T with the fallback reason.
- **SIZE:** 1,018,791 B before stripping. Applied rewrite/build.py's AST-checked `strip_comments` (the AST is identical, so every string constant is unchanged): **948,272 B (`wc -c`)**. No diagnostics were hand-trimmed and no docstrings stripped.

Deviations, with reasons:
1. E routes only the exact judged T (1, 8, 16, 32, 128, 256). The other T of each class (2..7, 9..15, 17..31) stay on master because those are the only T the race measures.
2. Two E modules rather than one, so a failure in one does not cost the other. Both are separate, non-daemon, and start at t=0, as the review asks.
3. E5 is selected by pdl/flags bit 4096 instead of the `dq` argument, so master's host checks stay unchanged.
4. Draws are the 3 per T above; I did not add a separate 12-T canary, because routes exist only at the raced T.
5. A state-copy check was added; the review did not ask for it.
6. The race cap is checked before each class, so the last class can overrun 20 s by about one class (~1-3 s).
7. Node validation SKIPPED: the user's permission policy refuses node runs from subagents.

Biggest risks: nvcc for two more modules (~45-60 s each) at t=0 competes for CPU with es1a/ea6/ea4x/stk/wh, which may push master's own join deadlines (EA6_JOIN_S 170 s, EA4X_START_S 190 s). The CUDA was never compiled here: an nvcc error only costs the E lane, not master.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_moe_w4a16/engine-a-a4b-a5b-encodings-race diff`. The diff is 820+/825- lines, most of them comment-only line removals from strip_comments (AST identical). The functional hunks are `_prepare_main` (start, final block), `_ea_forward` (`**_EAE_CORES`), `_ea6_forward_core` (`enc`), `_hm_experts` (xarm -> `_eae_pair`) and the appended ea-enc section.
- [x] CUDA-graph capturable: kernel_fn's new paths are only the `_EAE_CORES` closures, `_ea6_forward_core`'s `enc` branch and `_eae_pair`. They do dict lookups on prepare-set keys and module launches, with no `.item()`/`.cpu()`/sync/print and no allocation or branch on input values. The race captures and replays every E route through `_rt_graph`. The two new `torch.cuda.synchronize()` calls are in `_eae_class`/`_eae_run` (prepare).
- [x] Nothing derived from `data` is cached across calls: the new state keys (`exte`, `ext6e`, `w13e*`/`w2e*`, `on[T]` strings, `hm['cold_e']`) come from the weights and prepare-time races only.
- [x] Weights repacked losslessly in prepare() only: the A4b copy is a fixed 32-bit permutation of each NVFP4 code word, with the scale bytes untouched. The A5b copy packs b-8 into 7-bit fields and is built only when every scale byte is in [0x08,0x7E], so it is lossless. Both stay NVFP4 codes and E4M3 scales, dequantized in-kernel. Every route requires h/y/out torch.equal to master.
- [x] Activations BF16 where SGLang keeps BF16: the same bf16x2 fma dequant and mma.sync fp32 accumulation; bit-exact to master is required to route.
- [x] Attention sees every key: n/a (MoE expert block, no attention).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: the builds are `load_inline` in threads (as es1a/ea6/ea4x/wh are), and there is no subprocess/multiprocessing/os.fork in the diff.
- [x] All work happens inside the timed call; no output caching or replay: every E route still runs K1 + K2 (+ combine) per call. Prepare graphs are dropped after timing.
- [x] Size: `wc -c` = 948,272 B < 1,000,000.
- [x] First line kept (`# okc: problem=muse_moe_w4a16`); `python3 -I -m py_compile` OK; `__pycache__` removed.

okc_lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py .worktrees/muse_moe_w4a16/engine-a-a4b-a5b-encodings-race/muse_moe_w4a16/submission.py --problem muse_moe_w4a16`):
```
... (master's pre-existing prepare-side host-sync / dtype warnings unchanged) ...
warn  host-sync    torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())   [new: _eae_class]
warn  host-sync    torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())   [new: _eae_run]
OK with warnings: .worktrees/muse_moe_w4a16/engine-a-a4b-a5b-encodings-race/muse_moe_w4a16/submission.py
```
(rc 0; 205 lines vs master's 203; the only new warnings are the two prepare-side syncs above.)

## Risks / what could fail
medium

## Result (after the run)
Promotion passed: 2.0201x [1.9898,2.0504], board: {"rank": 333, "team": "yesthatwilly", "speedup": 2.0201, "first": false, "ahead": 332, "to_first": 1.3324}
