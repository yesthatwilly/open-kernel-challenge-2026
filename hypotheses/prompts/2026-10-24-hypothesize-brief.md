# Minting hypotheses for one problem (/hypothesize, N=5)

Repo: /home/user/git/musespark (shell is nushell; use `bash -c` for bash syntax). No GPU here.
H="python3 -I /home/user/git/musespark/.claude/skills/hypothesize/bin/hyp.py"

Follow /home/user/git/musespark/.claude/skills/hypothesize/SKILL.md (read it): run `$H context <problem>`,
read FINDINGS.md first (it may be stale: also read the per-shape `.result.json` files and their
`## Adversarial review` sections under <problem>/hypotheses/done/ — they hold today's real
learnings), then TASK.md, baseline.py and master's submission.py. Mint exactly 5 hypotheses with
`$H new` (lint-clean). Write files only: never commit, stage or edit anything else.

Rules you must respect (rules/technical-rules.llms.txt): no top-k/sparse attention, no precision
below the rules, NVFP4 weights stay NVFP4 and BF16 stay BF16 (dequantizing INSIDE a kernel is fine;
the user has ruled that materializing a full BF16 copy of NVFP4 weights per call is NOT acceptable),
no caching of anything derived from inputs, no host syncs, no work outside the timed call, no
triton.autotune or timing inside kernel_fn.

Lessons from today's runs (all problems):
- Run-to-run per-shape drift is ~1.5-2.5%; a geomean move under ~1.5% cannot be resolved. Prefer
  ideas whose per-shape effect is >= ~5% on heavily weighted shapes.
- Launch-fusion of a few tiny kernels was flat on decode-size shapes (decode cache-store fusion,
  expert-id-map, moe align-single-token, side-stream overlap all neutral).
- FA3: num_splits is only a cap under varlen dynamic split; FA3 already fills the GPU (~600 TFLOP/s
  on prefill). The page-size-1 gather limits full decode attention bandwidth (1.3-1.7 TB/s).
- Marlin MoE: per-m-block cost grows with block_size_m (thread_m_blocks): next-larger block lost
  18-28% on moe_w4a16 at bs=32/128; workspace size, fp32-reduce, atomic-add knobs all flat.
- A Triton W4A16 GEMV with in-kernel FP4 dequant was 2-2.6x SLOWER than Marlin at small M.
- The one big win: bf16_proj's router as one 3-D split-K Triton kernel (grid cdiv(M,BM) x N/BN x
  SPLIT_K, ~128 CTAs, fp32 atomic_add into zeroed output, bf16 hi+lo split) went 10.5-17.7 us ->
  5.7-9.3 us at every m (1.151x). Its code is in master's muse_bf16_proj/submission.py.
- Triton skinny GEMMs on bf16: BLOCK_N=64 streamed 67 MB at 3.73 TB/s (beat nvjet's 3.6) at m=16;
  BLOCK_N=16/32 were slower; the m=32 penalty is not the 32-row tile.

Use `--depends-on` only on a slug that exists; a hypothesis that should build on master needs none.
Reply with one line per file: path, area, expected shapes, why worth a run.
