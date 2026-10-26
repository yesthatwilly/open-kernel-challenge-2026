# muse_moe_w4a16 · STAMP · check · campaign-dq-ab-ladder-moe

Campaign `2026-10-26-expert-bandwidth`: a dequant A/B in the ladder. Can a rotation layout cut the consumer's cost?

- **Job:** `<job id>` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `COMMIT` (branch campaign/expert-bandwidth)
- **Base:** board e854db4683dc495f (1.6851); ladder-2 5eafb39e194d4d8b (C-R2 3.98, C-R4 3.45 / 2.94 TB/s).

## Goal
Ladder-2 found v4's consumer bound by its instruction stream. A rotation layout extracts every code pair with fewer
instructions in total, but how they split across pipes matters:
- prod (dq2): about 6-8 ALU + 3-5 IMAD on the FMA pipe; IMAD.SHL does the left shifts.
- dq3: 4 SHF.L.W + 5 LOP3, all ALU.
- dq4: the same rotations as IMAD.WIDE.U32 by run-time powers of two, about 6 ALU + 4 FMA.

The audit pointed out that dq3 may be slower if the ALU pipe is the limit. Production keeps dq2, and the ladder times
all three in v4's geometry (8 consumer warps, 2-record stages, D=2) at C-R2 (dequant only), C-R4 at 8 tokens and C-R4
at 16 tokens.

Questions:
1. VERDICT-dq: does dq3 or dq4 beat production by 5% or more at C-R4? If so, the production layout changes next (moe
   and decode checks).
2. dq_exact, now with the other three pairs of every word filled to expose crosstalk: production 0 mismatches? The
   rotation layout's extractions 0 / 0?
3. The hybrid gains H=-8 (`rewrite/lib/hotcold_moe.py`): wgmma-hot with a hot threshold of 8 routes, so Engine A
   keeps only one-tile cold passes. Does it win moe's T=128, where H=-16 lost (x0.970), or T=256 over H=-16?

- v4c (both problems): v4's kernels with the stream-K ranges weighted by pass cost. A two-tile pass's unit costs about
  1.17x a one-tile one (ladder-2), so equal unit counts leave the CTAs on two-tile passes as the kernel's tail. v4c
  rides the pdl argument's bits 4..7 and races v4 in every engine gate (T classes 1 / 8 / 16 / 32 and big T = 128 /
  256). Does it win where two-tile passes exist (T >= 16, big T)?

## What changed, and where
- `rewrite/lib/cuda/ea_ladder.cuh`:
  - `ext_rot_shf`, `ext_rot_mul` (PTX `mul.wide.u32`) and `ext_dq<DQ, I>`;
  - `consume` takes a run-time zero for dq4's multipliers;
  - kernel ids 19..24;
  - dq_exact gets crosstalk filler and two rotation counters.
- `rewrite/lib/ea_ladder.py`: the D- rungs and the VERDICT-dq line.
- `rewrite/bin/cuda_stub/cuda_stub.h`: the funnel-shift declarations.
- The ladder is diagnostic only: the engines and the repack are unchanged.
- `rewrite/lib/hotcold_moe.py`: the hybrid gains H=-8. Thresholds and hot block sizes go through `_hm_hot_rows` /
  `_hm_hot_block` in the align, caps, host reference and path text; no CUDA change. Gated like every H (canary vs
  master / fp32, whole-layer gate, copy check).
- Read-only audits: the rotation layout (all 2^32 words on the host: 0 mismatches) and the ladder change (no BLOCKER or MAJOR; MINORs applied).
  cuda_syntax ok; py_compile ok.
- v4c: `rewrite/lib/cuda/ea4_kernels.cuh` gets `Map::s_cost`, the weighted begin / owner (a binary search over
  prefix pass costs) and the slot rule, with `pdl & 1` as the PDL flag. The engine is registered in `ea3_core.py`
  (`pdl_or`), `ea4_core.py`, `ea2_core.py` (occupancy alias), `ea_decode.py` and `ea_moe.py`. A Python model over 400
  random pass lists checked ranges, owners, slots and contributors. Read-only audit: no BLOCKER or MAJOR (with the bits clear the production path is unchanged). Its MINORs: v4's kernel attribute lines now print again at the end of prepare, so registers and spill can be compared; a stale v5 comment is fixed; prepare grows by v4c's canaries and gates. The hybrids' cold halves take Engine A's split per class (`cold_bits`).

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: path choice by T and prepare-set flags only (the ladder runs in prepare only).
- [x] Nothing derived from `data` is cached across calls.
- [x] Weights unchanged (the ladder streams a read-only view of w13's bytes).
- [x] Activations BF16.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network.
- [x] All work inside the timed call; no output caching or replay.

## Risks / what could fail
- About +10 s of prepare for 6 more ladder kernels.
- A device fault in a new rung would fail the run. No board effect: this is a check.

## Result (after the run)
