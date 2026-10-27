# muse_prefill_layer · 2026-10-27T1426Z · submit · experts-rega-w2-early-loads-low-class

3:1 rhythm ("use a scoring submission any time there is a gain to be secured"): scored run of the exact bytes of check
59eb665f162a4ac9 (branch hyp/muse_prefill_layer/experts-rega-w2-early-loads-low-class, impl commit 52169f17, submission
sha256 874433d1d7a6e540...).

- **Job:** `99767cf374bb4688` (`okc status <job>`)
- **Base:** board 1.4889 (ec3bd018, fast silu). Check 59eb665f scored 1.4986; its confirm 8716d98c scored 1.4994.

## Why
Both checks beat the board. Churn pooled them as unproven-vs-best (+0.68%, under the 1.07% two-run margin), so it does
not promote. But the change routed in both runs with the same in-prepare evidence:
- Arm a (early gs/sid loads in the rega w2 epilogue) at the LOW class: pop:1 / random x0.975, guard routing x0.966-0.968,
  bit-exact (torch.equal) on all routings.
- fsilu stays ON; rega_low_variant 0; high and ft stay master's w2.

## Rules check (each must be true before the run)
- [x] Weights lossless and unchanged (NVFP4 rega weights untouched).
- [x] fp32 accumulation unchanged; the output y is bit-identical to master's on every routing.
- [x] Activations bf16.
- [x] CUDA-graph capturable: routes frozen in prepare; no launch or sync change in kernel_fn.
- [x] Nothing derived from `data` is cached across calls.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network.
- [x] All work inside the timed call; no output caching or replay.

## Result (after the run)
- 2026-10-27T1433Z: **passed, speedup 1.4897** (CI 1.467-1.512; score 260046 us). The new board (was 1.4889, ec3bd018); rank 176. Arm a routed at low, fsilu ON. Landed on master.
