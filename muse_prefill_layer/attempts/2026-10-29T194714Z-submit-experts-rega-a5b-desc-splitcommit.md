# muse_prefill_layer · 2026-10-29T194714Z · submit · experts-rega-a5b-desc-splitcommit

Hypothesis: `muse_prefill_layer/hypotheses/done/gain/2026-10-29T191222Z-experts-rega-a5b-desc-splitcommit.md` (branch `hyp/muse_prefill_layer/experts-rega-a5b-desc-splitcommit`).

- **Job:** `9f93484606cd4f9e` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `d95bbcb3`

## Goal
Promote check-proven gain experts-rega-a5b-desc-splitcommit (1.5792x [1.5555,1.6029], check job da80fdb9bed749a3) to the board.

Stack the H100-validated A5b+D+SC form on master's landed A3 rega dequant (muse_prefill_layer/submission.py master 1b14174f, the rega dq module from experts-rega-dq2-dequant-cost-diag), porting the code from takemyjob hopper/rgx2 771f8eb (worktree /home/user/git/.takemyjob-wt/rgx2; findings campaigns/2026-10-26-expert-bandwidth/research/h100-rega2-1029.md): (1) A4b: a new lossless prepare-time re-encoding of the 4-bit code words so nibble->bf16x2 takes 9 ops/word instead of 11; (2) A5b: scale bytes stored at prepare as packed 7-bit fields (b-8) so each k16 step's scale is one LOP3 (exactly A3's scale value; requires every scale byte in [0x08,0x7E], scanned in prepare, else keep A3); (3) D: the wgmma smem descriptor computed once per stage (halves uniform-pipe instructions); (4) SC: split commit, dequantizing record 1 while record 0's 4 wgmmas run, no extra registers. Ship it as a new arm in the existing dq module (template DQ value) next to A3 and DQ0, raced per class with the same gate as A3 (<= 0.97 vs the dispatched A3 path AND vs DQ0 every round, both L2 modes, both routings, replay pass, bit-exact torch.equal); if it misses in a class, that class keeps A3. Re-encoded banks are a prepare-time lossless repack of weights only (never from a routing); free superseded copies only when no class routes to them. Keep the summary lines ([rega-dq] RATIOS/PAIR/ROUTE/VERDICT) at the end of prepare inside the 16 KB tail; build in the existing background thread with its 233 s gate.

## What changed, and where
**Mechanism:** a new DQ 5 arm A5b+D+SC (ENC4 code re-encoding + packed 7-bit scales + per-stage wgmma descriptor + split commit) in the rega dq module's rega_kernel (w13 and w2, every N bucket, low/high/ft), raced in prepare next to A3 and DQ0 and routed per class only when <= 0.97 x A3, DQ0 and master every round; A1/A2 retired.

Worktree `muse_prefill_layer/submission.py` only, all inside the dq module (plus the one dispatch line in `_experts_rega`).

- **Build gate (review MUST 1).** A1 and A2 are retired: `REGA_DQ_NAMES {0: DQ0, 3: A3, 5: A5b}`, `REGA_DQ_ARMS (3, 5)`, `REGA_DQ_INST 12` (rega_kernel<G, BIG, DQ> for DQ 0/3/5). `REGA_DQ_EST_4INST_S` stays 40. The est now counts the 12 instantiations plus the 4 of the ptxas probe build (below): 40 x 16 / 4 = 160 s, need 233 s <= 270 s, the same gate as today. No A4b+D+SC64 fourth arm.
- **Device code (ported from takemyjob hopper/rgx2 771f8eb gen2.py, RGX_DQ 7 / RGX_DESC 1 / RGX_SCHED 1 / RGX_SC_MINN 16 = rgx2 arm 6).**
  - `dq5_k<KK>`: the A4b code views (q & M, (q << 6) & M, rotl(q, 10) & M, ((q << 4) & 0x81808180) | ((q >> 4) & 0x00400040), plain SHF, no IMAD.HI) and the one-LOP3 scale (`lop3 0xEA`, `__byte_perm(w, w, 0x2103)` for odd k, 0x78007800 held in a register as `m28 ^ 0x68007800` with `m28 = dq_rtv(KS, 1 << 28)`).
  - `dq5_rec<R>`: one record.
  - `dq_desc_lo` / `dq_desc_at` / `dq_batch_d<NB, R0, R1>`: D.
  - The MODE 0 k loop for DQ 5 is a new `else if constexpr (DQ == 5)` branch with rgx2's split commit, verbatim. DQ 0 and A3 keep master's loop and make_desc. A3's `dq_k` lost only its dead A1/A2 branches.
  - MODE 2/5 run `dq_stage_t<5>`.
- **Banks (MUST 3).**
  - `_rega_dq_a5b_bank` builds A5b's copy from `state['rega_bank']` (the weights' bank) only. Codes are ENC4-encoded through a nibble LUT (`_rega_dq_enc_luts`; it asserts that ENC4 is a bijection), and scale words are packed with `_rega_dq_pack_scales`.
  - Every 8192-record chunk is checked by its inverse with `torch.equal` (decode, unpack); a mismatch raises and that arm goes OFF.
  - The scale scan (`_rega_dq_scales`, [0x08, 0x7E]) gates both A3 and A5b, as before.
  - `state['rega_dq_bank']` is now `{arm: bank}` and keeps only the routed arms' copies, so A3's copy is freed when no class routes A3. Master's bank is never touched. The dispatch reads `state['rega_dq_bank'].get(dq)`.
- **Canary (MUST 3, 4).**
  - `rega_dq_exact` now takes words / A3-permuted words / ENC4 words / scales / packed scales. It writes master's per-k body and the A3 and A5b bodies for all 121,856 cases at both k16 parities, and compares them bitwise.
  - The same launch checks D: `dq_desc_at(dq_desc_lo(slab), r*(SLAB_HALF>>4)+2k)` against `make_desc(slab + r*SLAB_HALF + 32k)` for every 1024-aligned slab with slab + 2*SLAB_HALF <= 2^18 (1800 cases, `REGA_DQ_DESC_EXPECT`). A mismatch, or a count other than 1800, turns A5b OFF.
  - Two static_asserts pin make_desc's high word (0x40000040) and the offset field.
- **Route gate (MUST 2), `_rega_dq_route_pick`, coded explicitly.**
  - A5b passes when its hs/y equal master's dispatched launches on every self-check routing (and equal DQ0, which `_rega_dq_class` enforces), and its pair MODE 0 is <= 0.97 x master AND x DQ0 AND x A3 in every round, both L2 modes, pop and random. Without an A3 timing, A5b misses.
  - A3 keeps its existing gate.
  - The order is A5b, then A3. `_rega_dq_route` runs the two-state replay check for the first passing arm and falls back to A3's replay if A5b's fails; otherwise master.
- **ptxas (MUST 5).**
  - After the main build, the same background thread runs a probe build. It starts only before 120 s of prepare and its module is never loaded.
  - The probe uses the same source and flags plus `-DREGA_DQ_PTXAS_PROBE` (only rega_kernel<G, BIG, 5> is instantiated: 4) and `-Xptxas -v`, linked against a library that does not exist. The build fails at the link after ptxas has run, and the exception text carries ninja's output.
  - `_rega_dq_ptxas_parse` extracts the A5b entries' registers and spill bytes, C7510/C7512 counts and up to 2 warning lines. The result is printed in the phase and as one summary head line.
  - Regs/spill per instantiation (cudaFuncGetAttributes via rega_init) are printed as before; any spill turns that arm OFF.
- **Log (MUST 6).** The summary is still the last echo, capped at `REGA_DQ_SUMMARY_MAX` 3000. PAIR shows A5b vs DQ0 / vs m / vs A3 ('a'), and ROUTE shows the gate's worst rounds. With A1/A2 gone the RATIOS lines are about 40% shorter; a 3-class synthetic summary measured about 2.0 KB before the head lines.
- **Budget (MUST 7).** Same background thread, same deadlines (240 s ft step / 255 s steps / REGA_COPY_DEADLINE_S), two arms timed instead of three, and the A5b bank copy costs about as much as A3's.

**Deviations and choices.**
- The fourth arm (A4b+D+SC64) is not added: it is optional per the review, and 12 instantiations leave margin for the probe.
- ptxas warnings are captured by a link-stopped probe build instead of moe's `-Xptxas -Werror` probe. The -Werror form misses info-level C7510 notes and needs a failing build to show anything at all. It runs in the existing thread under the same 233 s gate.
- `_rega_dq_launch` now raises if an arm is launched without its bank copy (defensive).

### Node validation, H100 gpunode06 GPU 5 (2026-10-29 19:37Z)

The built submission (sha256 8c34de1978ce, byte-identical to the worktree file) ran through guarded hopper-probe. Everything ran as the operator, with no root and no Slurm. The nodejudge venv was used read-only: every cache, tmp and extension dir was redirected into the run dir, which was then removed. The package is takemyjob worktree `~/git/.takemyjob-wt/pfa5b` (branch hopper/pfa5b, uncommitted), `probes/pfa5b/` (the test drives the submission's own functions). Output: `~/git/.takemyjob-wt/pfa5b/prototypes/hopper-probe/runs/20261029-123722-gpunode06/out/pfa5b/pfa5b.txt`.

The bank is synthetic: random codes, scale bytes in [0x08, 0x47], and HBM-resident (not L2-aliased). It uses nodejudge shims for the judge-only sglang muse_spark modules (the dq path does not use them), and 'master' = DQ0 MODE 0 through master's signature.

- **Build:** the main module (12 instantiations) took 40.9 s and the ptxas probe 34.7 s, 75.6 s of wall time in total. The probe's link stop worked: ptxas output was captured, and A5b showed regs 168x4, spill 0, **C7510 0, C7512 0, no ptxas warning**. attrs (cudaFuncGetAttributes): all 12 at r168 l0 o1.
- **Canary:** 121,856 cases, A3 0 and A5b 0 mismatching; D descriptor 0 of 1800 mismatching. The A5b bank copy took 0.11 s with every inverse check passing. Negative control: a 0x07 scale byte is rejected.
- **Exactness:** hs and y of A3 and A5b are torch.equal to master and to DQ0 on random, skewed, one and pop routings, in all three classes (low T 753, high 3012, ft 16384).
- **Pair timing, A5b vs A3 per round (gate <= 0.97):**

  | class | A5b vs A3 | A5b vs master | route pick |
  |---|---|---|---|
  | low | .961-.972 (worst .972) | .865 | **A3** (A5b missed by 0.002) |
  | high | .931-.958 | .854-.890 | **A5b** |
  | ft | .924-.949 | .852-.875 | **A5b** |

  This bank is HBM-resident with random codes, the node's "low HBM" condition (rgx2 got 0.954-0.963 there). The judge's low class sat nearer the L2 figure (0.923-0.925), so low may pass on the judge; if it misses again, low stays on A3, which is the specified gate.
- **Total test time:** 100 s (classes 1.5 / 2.4 / 10.7 s).

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_prefill_layer/experts-rega-a5b-desc-splitcommit diff` (1 file).

- [x] CUDA-graph capturable. The only kernel_fn-path change is `_experts_rega`'s `dq_bank = (state.get("rega_dq_bank") or {}).get(dq)`, a dict lookup on a prepare-time int chosen by the token count (a shape). There is no `.item()`/`.cpu()`/`synchronize()`, no allocation on input values and no data-dependent control flow. All new syncs and `.tolist()` (canary, bank copies, probe) are in prepare(). The DQ 5 kernel has the same launch signature and grid as A3.
- [x] Nothing derived from `data` is cached across calls. The A5b bank copy comes from `state['rega_bank']` (the weights) only and never from a routing. The ptxas probe and canary use synthetic words.
- [x] Weights: repacked losslessly in prepare() only, and NVFP4 stays NVFP4. ENC4 is a bit permutation of each 32-bit code word: the same e2m1 bits in new positions, checked as a bijection, with an inverse `torch.equal` per chunk. The scale bytes are the same e4m3 values stored as b - 8 (7-bit fields), with an inverse `torch.equal` per chunk and the [0x08, 0x7E] range scan first. The dequant product is bit-identical to A3's and master's: the exhaustive canary checks it bitwise, then hs/y are compared with `torch.equal` on every self-check routing.
- [x] Activations BF16 where SGLang keeps BF16. The wgmma A fragments are the same bf16 values as master's, with fp32 accumulation, the same k order and the same accumulator chain; split commit only splits one commit group into two. KV cache and queries are untouched.
- [x] n/a: attention is untouched, so it still sees every key (full window / full context).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, network, or subprocess / multiprocessing / os.fork / spawn. The only build is `load_inline` (torch's own ninja), as master's builds. The probe build's deliberate link stop is a missing `-l` library in `extra_ldflags`, and its module is never loaded.
- [x] All work happens inside the timed call, with no output caching or replay. Prepare-time work is the build, canary, bank repack and race; kernel_fn only launches the routed arm.
- [x] sm_90a: the new PTX (wgmma via the existing `Wg<NB>::mma`, lop3, mov.b64) sits inside the module that is already built with TORCH_CUDA_ARCH_LIST=9.0a and REGA_GENCODE. wgmma stays under master's `__CUDA_ARCH_FEAT_SM90_ALL` guards, and mbarrier waits stay bounded (master's `mbar_wait`).
- [x] A build failure, canary miss, spill or gate miss lands on A3 (if it passed its own gate) or master, and never raises from prepare(): `_rega_dq_phase` keeps its try/finally.

okc_lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_prefill_layer/submission.py --problem muse_prefill_layer`):

```
OK with warnings: muse_prefill_layer/submission.py
(warnings only: host-sync lines all "outside kernel_fn's call graph; fine in prepare()" and 5 pre-existing uint8 torch.empty dtype notes; none new in kernel_fn)
```

py_compile OK; __pycache__ removed. Diff: 1 file, +498 / -196 (muse_prefill_layer/submission.py).

## Risks / what could fail
medium

## Result (after the run)
Promotion passed: 1.5721x [1.5485,1.5957], board: {"rank": 230, "team": "yesthatwilly", "speedup": 1.5721, "first": false, "ahead": 229, "to_first": 1.2929}
