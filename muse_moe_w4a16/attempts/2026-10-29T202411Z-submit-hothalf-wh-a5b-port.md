# muse_moe_w4a16 · 2026-10-29T202411Z · submit · hothalf-wh-a5b-port

Hypothesis: `muse_moe_w4a16/hypotheses/done/gain/2026-10-29T195507Z-hothalf-wh-a5b-port.md` (branch `hyp/muse_moe_w4a16/hothalf-wh-a5b-port`).

- **Job:** `<job id>` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `ddc98f98`

## Goal
Promote check-proven gain hothalf-wh-a5b-port (1.9827x [1.9529,2.0124], check job 54b70249868f468b) to the board.

Port decode's routed wh hot half (module muse_decode_wg_wh from branch hyp/muse_decode_layer/hothalf-a3-exactn-ring-wh, check 44f30364: wg_kernel<G,V> with D0 control V=0 and wh V=7 = A3 dequant on a word-permuted whole bank + exact-N tiles + R4 ring wait<2> for N>=48; background-thread build; isolated-pair route gate <= 0.97 vs master and D0 every round, exact, 16-replay) into moe's hot half (muse_moe_w4a16/submission.py wg_kernel at :867-1700, dispatched at T=128/256 by the hybrid). Add one more arm wh5ds = wh + A5b+D+SC from takemyjob hopper/hh3-1029 6a33dbf (A4b code re-encoding of the same permuted bank, 7-bit packed (b-8) scale fields, per-stage wgmma descriptor, split commit; landed on prefill as experts-rega-a5b-desc-splitcommit, scored 9f934846), raced as a candidate against wh: route wh5ds only if it also beats wh by <= 0.98 every round, else wh. Scale bytes scanned in [0x08,0x7E] (else master); banks from weights only. Keep moe's fork, routing, cold half and PDL unchanged. Print the [hh-wh] BUILD/ATTRS/SCALES/ROUTE/FORK lines (<= 1.5 KB) at the end of prepare. Size: moe is 964,183 B; the cap is 1,000,000 B (build.py strips comments above 990 KB): measure and trim dead diag text first if needed.

## What changed, and where
**Mechanism:** moe's T=256 hybrid wgmma hot half (wg_kernel w13+w2 at _hm_experts' wg-hot call site, verify_32) re-routed in prepare to the best bit-exact arm of a new module (a3 = A3 dequant on a word-permuted bank, wh = A3 + exact-N + R4 ring, wh5ds = wh + A5b codes/packed scales + D + SC), gated on the isolated pair and the routed forked kernel_fn.

What changed (worktree `muse_moe_w4a16/submission.py` only, base db626b76, blob 691018b5):
- New section before the es_core include (`# ---- hothalf-wh-a5b-port`): module `muse_moe_wg_wh_<sha10>` derived from master's `_WG_SRC_HEAD/_WG_SRC_BODY` by decode's asserted edits (moe's wg text is byte-identical to decode's base, checked). `wg_kernel<G, V>`, V bits 1 = A3, 2 = exact N, 4 = R4 ring wait<2> for N >= 48, 8 = A5b+D+SC. Arms: D0 = 0 (master's text in this TU, the control), a3 = 1, wh = 7, wh5ds = 14 (exact N + ring with D for N >= 48, the SC+D stage loop below N 48, x5 dequant step everywhere; hh3's `hh_wh5ds.cuh` shape). The ring keeps decode's 64-bit descriptors for V & 8 == 0, so wh is the same code as decode's 44f30364 module.
- Banks, from the weights' wg bank only: the scale bytes are scanned in [0x08, 0x7E] (else OFF -> master); a3/wh read an A3 word-permuted copy (decode's `_wh_perm` map), wh5ds reads an A5b copy (ENC4 code map, prefill's `REGA_DQ_ENC4`, plus the packed 7-bit b-8 scale words). Each chunk is checked lossless by the inverse. A host self-test also checks, on 2^16 random words and all 119^2 scale pairs, that A5b's four code views of ENC4(q) equal A3's ext_prop of perm(q) and that the LOP3 scale equals A3's prmt*16 + 0x77807780 (the same model passed locally). Only the routed arm's copy is kept (~0.86 GB).
- Dispatch: only `_hm_experts`' wg-hot branch (serial and W6-M fork) calls `_wh_pair(state, wb, state['whk'].get(T), ...)`. `state['whk']` is a prepare-set {256: V or None}. `_moe_wgmma` (the bs=257 test path, T=128..512 non-hybrid) is untouched.
- Build: `prepare()` is now a wrapper. It starts a non-daemon thread (nice +10; ptxas `-Werror` probe first, C75xx grepped; plain rebuild if time allows) before es1a/ea6/ea4x/EPLF start, then `_prepare_main` runs master's prepare unchanged, and `finally` joins the thread. TORCH_CUDA_ARCH_LIST is set to 9.0a, as es1a/ea6 already leave it, and is not restored (master behaviour).
- Phase order (review open question 1): `_wh_phase` runs after `_hm_setup` (W6-M race, fork gate, v4x-t256, copy check) and `_eb_setup`. T=128 is never touched (`whk` has only key 256), so verify_16 is a noise control. The phase needs T=256 routed to a wg-hot H, else OFF.
- Route at T=256: exactness first (torch.equal of hs and output vs master's dispatched launches on the 3 judge-style draws `_ea_judge_routings` p10/p50/p90 + 4 synthetic all-N routings; err word [0, magic]). Then the isolated pair (gather once; graphs of w13+w2 per arm, `_rt_time`, 3 draws x 2 rounds = 6 rounds). An arm qualifies when it is exact, D0/m in 0.97..1.03, arm/m <= 0.97 and arm/D0 <= 0.97 in every round, and, for wh5ds only, wh5ds/wh <= 0.98 in every round (isolated, per the review). Among qualifiers the best mean pair goes first and needs replay 16/16 (two scratch sets, 8 refills alternating judge draws and synthetic routings, bitwise vs master's eager pair). Then the production-fork condition: kernel_fn T=256 graphs (routed H, fork as routed; forked and serial for m/a3/wh/wh5ds on the same 3 draws x 2 rounds) must give pick/m <= 1.00 in every round, else master. Last, `_hm_copy_check` with the pick routed; a FAIL goes to master. Every step starts before 240 s and the copy check before HM_COPY_DEADLINE_S, else master.
- Prints: at the very end of prepare (after the [ea6] lines), at most ~1.5 KB: `[hh-wh] BUILD` (s, bg start, join wait, -Werror/C75xx, phase start s), `ATTRS` (regs/spill per arm, w13/w2; 168 regs / 0 local required, else OFF), `SCALES` (count in range, copies lossless, self-test), `ROUTE T=256` (per-round wh/m, wh/D0, a3/D0, wh5ds/wh; D0/m range; pair us per arm; hot experts + N-chunk histogram per draw; exact; replay), `FORK T=256` (kfn us f/s per arm; per-round pick-mode ratios vs m; serial worst; m f/s; hot pair vs kfn serial anatomy), the re-printed `[w6m-fork] PATH T=128/256` lines, and `PATH CHOSEN T=256 <arm> wg_kernel<G,V>` (profile symbol) or `master wg_kernel<G> (why)`.

Deviations / review fixes:
- Review (1): a3 competes in the route. wh5ds is not hard-wired on wh; it is one of three candidates, plus the wh-relative bar.
- Review (2): verify_16 is expected `same` (in-run noise control). T=128 stays master by construction.
- Review open question 2: wh5ds's 0.98 is on the isolated pair; the fork gate is a no-loss gate (<= 1.00) on the routed kernel_fn.
- decode's a3nx attribution arm was dropped (size; no route use).
- The spec text says the A4b codes re-encode "the same permuted bank". hh3 (`k_recode_bank(bk.q13, ...)`) and prefill (`_rega_dq_a5b_bank`) both encode master's words, so wh5ds's bank is ENC4 of master's words, a second copy.
- SIZE: the base was 964,183 B. Trimmed dead diagnostics: `_ea_kernel_rates` and `_combo_f2_select_diag` (both only reachable under `not CAMPAIGN_LEAN_DIAG`, which is True) with their two call blocks, and the unreferenced `_ea_ref_block` (~5.2 KB). **Uploaded file: 993,167 B (`wc -c`) < 1,000,000.**
- Memory (review open question 4): the routed copy is a tensor in the state, so the judge's tensor-cloning copy (modelled by `_wg_copy_tree` in the copy check) clones it like master's other banks.

Prior (frontmatter `prior: none` is wrong; corrected here): decode hothalf-a3-exactn-ring-wh (check 44f30364, neutral 1.6794, the parent being ported); moe CLAIMS.md:104 (A3 hot-half [lead]; this is its first judge check); CLAIMS.md:54 (W5 exact-N 80/96/112 tiles refuted on moe, <= x1.006). On moe's N32-dominated T=256 draw, the exact-N and ring parts are expected to add nothing; the operative content is A3 (+ A5b + D).

Both H100 lanes' pre-registered rules argued against a moe check: hh2 said "on moe ship A3 alone", and hh3's band (> 0.98 forked vs wh) meant no follow-up check. This run goes ahead on the user's request (user_requested: true) and today's steers, not on the H100 lanes' decision rules.

Read/kill rule (review): read the in-run FORK ratios first, then verify_32 (291.1 -> expect 272-276 us; above ~285 us is inside row noise), then the `wg_kernel<G,V>` profile rows, then the headline. If the forked kernel_fn T=256 ratio vs master is > 0.98, the moe fork absorbs the hot cut: mark the mechanism [refuted] for moe.

Node validation (H100 gpunode06 GPU 1; nodejudge config `hhwh-moe`, 1 cold isolated prepare of these exact bytes, sha256 8c4b56a992bd, no tests, no timed shapes; takemyjob run `runs/20261029-131634-gpunode06`, 20:16-20:19Z). passed=True, prepare 125.7 s / 300 (peak 18.0 GiB alloc), `[cs-wh] whk {256: 14}`, hm on {128: 0, 256: -16} fon {256: True}. The [hh-wh] end block is 1,333 B:
```
[hh-wh] BUILD 42s bg from 0.0s, join waited 0s at 123s; ptxas -Werror clean, C75xx none
[hh-wh] ATTRS regs/spill w13 D0 r168/l0 a3 r168/l0 wh r168/l0 wh5ds r168/l0; w2 D0 r168/l0 a3 r168/l0 wh r168/l0 wh5ds r168/l0
[hh-wh] SCALES 100663296 in 0x08..0x7E; A3 perm + A5b ENC4/pack copies lossless, self-test PASS (0.7s)
[hh-wh] ROUTE T=256 pick wh5ds: wh/m .875/.874/.861/.859/.893/.893 wh/D0 .880/.877/.858/.859/.891/.891 a3/D0 .874/.874/.882/.884/.896/.896 wh5ds/wh .979/.979/.942/.944/.960/.961 D0/m 0.995..1.003; pair us m 115.6 D0 115.7 a3 102.5 wh 101.2 wh5ds 96.9; hot:N 4:32x2,48x2 7:32x5,80x1,96x1 7:32x5,64x1,128x2; exact 7/7; replay wh5ds 16/16 exact
[hh-wh] FORK T=256 kfn us f/s m 318.6/343.9 a3 308.0/331.0 wh 304.5/329.5 wh5ds 300.4/324.8; f/m a3 .974/.973/.966/.966/.960/.961 wh .975/.975/.933/.934/.957/.959 wh5ds .965/.964/.925/.925/.937/.940; s/m worst a3 0.977 wh 0.975 wh5ds 0.973; m f/s 0.889..0.955; hot pair m 115.6 vs kfn serial 343.9 (cold+rest 228.3)
[hh-wh] PATH CHOSEN T=256 wh5ds wg_kernel<G,14> (copy PASS), T=128 master (2.4s, prepare at 125.3s)
```
Read: wh5ds routed on H100 (isolated pair x0.84 vs master; forked kernel_fn x0.925-0.965, about -18 us on the routed T=256 graph). The forked ratio is below the review's 0.98 kill line on this node. On H100, wh beats a3 isolated on 4 of 6 rounds, unlike hh2's moe draw. The phase took 2.4 s and the build was hidden behind the other builds (join wait 0 s). The judge's H200 decides the transfer.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_moe_w4a16/hothalf-wh-a5b-port diff` (12 hunks: `_wg_wgmma_src(nbs)`, the prepare wrapper, the phase call, the final prints, 3 dead-diag removals + their 2 call blocks, the 2 `_hm_experts` call-site lines, the new section).
- [x] CUDA-graph capturable: kernel_fn's only new code is `_wh_pair(...)` at `_hm_experts`' wg-hot branch. It reads a prepare-set int from `state['whk']` keyed by `num_tokens` (shape), then makes two `wh_gemm` launches that encode tensor maps per call in C++, as master's `wg_gemm` does. No `.item()`/`.cpu()`/sync/print. No allocation or branch on input values. The routed graph was captured and replayed in prepare (replay check, fork race, copy check).
- [x] Nothing derived from `data` is cached across calls: `whk`, `whk_bank`/`whk_x5`, `whk_ext`, `whk_err` are set in prepare from the weights only; no input-derived tensor is stored.
- [x] Weights repacked losslessly in `prepare()` only: the A3 copy is a bit permutation of each 32-bit NVFP4 code word (scales untouched). The A5b copy is the ENC4 bit permutation plus the scale word packed as 7-bit b-8 fields, which is lossless for bytes in [0x08, 0x7E]: scanned first, and each chunk is checked by the inverse. Still NVFP4 codes and E4M3 scales, dequantized in-kernel.
- [x] Activations BF16 where SGLang keeps BF16: every arm is bit-exact to master's hot pair (torch.equal of h and y; required to route). The dequant arithmetic is the same bf16x2 fma per element.
- [x] Attention sees every key: n/a (MoE expert block, no attention).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network. The build is `torch.utils.cpp_extension.load_inline` in a thread, as master's es1a/ea6/ea4x builds are; `os.setpriority` lowers the build thread's nice. No subprocess/multiprocessing/os.fork in the diff.
- [x] All work happens inside the timed call; no output caching or replay: the hot half still computes w13+silu and w2 per call. Prepare-time graphs are dropped after timing.
- [x] Size: `wc -c` = 993,167 B < 1,000,000 (CUDA comments included; no build.py strip needed).
- [x] First line kept: `# okc: problem=muse_moe_w4a16`. `python3 -I -m py_compile` OK, `__pycache__` removed.

okc_lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_moe_w4a16/submission.py --problem muse_moe_w4a16`):
```
... (pre-existing prepare-side host-sync / dtype warnings unchanged) ...
warn  host-sync    L15495: loc = ids.long().flatten().cpu()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L15496: cnt = torch.bincount(loc[(loc >= 0) & (loc < 32)], minlength=32).tolist()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L15551: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L15553: e = [int(t) for t in sb["whk_err"].tolist()]  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L15600: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L15646: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L15660: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L15661: e = [int(v) for v in state["whk_err"].tolist()]  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L15733: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: .worktrees/muse_moe_w4a16/hothalf-wh-a5b-port/muse_moe_w4a16/submission.py
```
(rc 0; every warning is prepare-side; the new ones are the `_wh_*` prepare functions above.)

## Risks / what could fail
medium

## Result (after the run)
(pending)
