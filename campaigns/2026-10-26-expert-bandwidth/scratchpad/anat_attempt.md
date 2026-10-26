# muse_decode_layer · STAMP · check · campaign-anatomy-decode

Campaign `2026-10-26-expert-bandwidth`: where the expert block's time goes, per T class. This is a measurement check;
the timed path changes only if the new hybrid candidate H=-8 wins its gates.

- **Job:** `<job id>` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `COMMIT` (branch campaign/expert-bandwidth)
- **Base:** board BOARD; check f59cb301895240e6 (1.5347). These bytes differ from it by Engine D being off and the
  print-only anatomy.

## Goal
The dense GEMMs turned out to be near speed of light (f59cb301), so the remaining gap at T <= 32 sits in the expert
block and the glue. d1's profile suggests about 45 us for the expert block against about 17 at the ring's rate. v16's
cold half (v4 K1) reads about 2.45 TB/s in the profile. Profile rows of PDL kernels include waits, so measure the
parts graph-timed in isolation:
- T=1/8/16/32: Engine A's block (K1 + K2 + combine) on 'judge' routing, and the same launches with no local route
  (their fixed cost);
- T=128/256: the chosen hybrid whole, its wgmma hot half and its v4 cold half, on judge-like routing (master's router
  scores of `_ead_cases` inputs). Each part prints its bytes and TB/s.

Questions:
1. At T=1, how much of the block is fixed launch cost versus streaming rate?
2. At T=128/256, what does each half stream at? Which half to attack: cold (v4 at nt1/nt2) or hot (wgmma)?
3. With Engine D off, rows stay within noise of f59cb301?
4. The hybrid gains H=-8: wgmma-hot with a hot threshold of 8 routes, so Engine A keeps only one-tile cold passes.
   Does it beat H=-16 at T=128 / 256 (v16 is our weakest load)?

- v4c (both problems): v4's kernels with the stream-K ranges weighted by pass cost. A two-tile pass's unit costs about
  1.17x a one-tile one (ladder-2), so equal unit counts leave the CTAs on two-tile passes as the kernel's tail. v4c
  rides the pdl argument's bits 4..7 and races v4 in every engine gate (T classes 1 / 8 / 16 / 32 and big T = 128 /
  256). Does it win where two-tile passes exist (T >= 16, big T)?

## What changed, and where
- `rewrite/src/muse_decode_layer.py`: CAMPAIGN_ENGINE_D off, and the anatomy prints at the end of prepare.
- `rewrite/lib/ea_decode.py`: `_ead_anatomy`.
- `rewrite/lib/hotcold_decode.py`: `_hc_anatomy`, and `_hc_experts(part=)`, which only prepare passes.
- `rewrite/lib/hotcold_decode.py`: the H=-8 candidate. Thresholds and hot block sizes go through `_hc_hot_rows` /
  `_hc_hot_block` in the align, caps, host reference and path text.
- No CUDA change.
- v4c: `rewrite/lib/cuda/ea4_kernels.cuh` gets `Map::s_cost`, the weighted begin / owner (a binary search over
  prefix pass costs) and the slot rule, with `pdl & 1` as the PDL flag. The engine is registered in `ea3_core.py`
  (`pdl_or`), `ea4_core.py`, `ea2_core.py` (occupancy alias), `ea_decode.py` and `ea_moe.py`. A Python model over 400
  random pass lists checked ranges, owners, slots and contributors. Read-only audit: no BLOCKER or MAJOR (with the bits clear the production path is unchanged). Its MINORs: v4's kernel attribute lines now print again at the end of prepare, so registers and spill can be compared; a stale v5 comment is fixed; prepare grows by v4c's canaries and gates. The hybrids' cold halves take Engine A's split per class (`cold_bits`).

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn unchanged (`part` defaults to the full path).
- [x] Nothing derived from `data` is cached across calls.
- [x] Weights unchanged.
- [x] Activations BF16; KV FP8.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network.
- [x] All work inside the timed call; no output caching or replay.

## Risks / what could fail
- About +10-15 s of prepare. The anatomy never raises; any error prints its type.

## Result (after the run)
