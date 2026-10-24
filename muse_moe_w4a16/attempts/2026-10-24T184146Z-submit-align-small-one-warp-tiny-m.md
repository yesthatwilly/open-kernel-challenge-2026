# muse_moe_w4a16 · 2026-10-24T184146Z · submit · align-small-one-warp-tiny-m

Hypothesis: `muse_moe_w4a16/hypotheses/running/2026-10-24T183722Z-align-small-one-warp-tiny-m.md` (branch `hyp/muse_moe_w4a16/align-small-one-warp-tiny-m`).

- **Job:** `38532ba10e394e3c` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `1333b37`

## Goal
num_tokens <= 8 only (decode_1 bs=1, verify_1 bs=8, the bs=1/bs=4 tests); bs=16 and above keep master's launches exactly. In _align_small (host wrapper), replace the fixed num_warps=4 of the _align_small_kernel launch with num_warps = 1 if n = num_tokens*8 <= 64 else 4. Nothing in the kernel changes. The prepare() warm-up loop already calls _align_small for every tokens 1..32, so it compiles the 1-warp variants for 1..8 automatically; _align_self_check runs through the same wrapper and so checks the 1-warp variants too. Mechanism: at n <= 64 every tensor in the kernel is tiny (ids [<=64], one-hot [<=64, 32], CAP_BLOCK <= 512, MB_BLOCK <= 64), so 4 warps leave most threads idle while every tl.sum / tl.cumsum across the 4 warps goes through shared memory and a bar.sync; one warp does all of them with shuffles only, so the kernel's serial chain of ~8 reductions/scans gets shorter. One-warp register use: the [64, 32] one-hot and its cumsum are 64 int32 per thread per temporary, under the 255 limit (print n_regs/n_spills of the 1-warp bs=8 variant in prepare via the compiled kernel's .n_regs/.n_spills).

## What changed, and where
In `_align_small` (host wrapper, submission.py ~L277) the `_align_small_kernel` launch now passes `num_warps=1 if n <= 64 else 4` (n = num_tokens*8, so 1 warp at tokens <= 8). Kernel body unchanged. Triton's tl.sum/tl.cumsum are layout-generic, so the 1-warp variant is correct with no kernel change (nothing is hard-sized for 4 warps). The prepare() warm-up loop calls `_align_small` for tokens 1..32, so the num_warps=1 variants for 1..8 compile there, and `_align_self_check` (tokens 1, 8, 16, 32) runs through the same wrapper. Deviations: did not add the n_regs/n_spills print (no GPU to verify; the profile row at bs=8 decides) and did not add tokens=4 to the self-check shapes (reviewer's optional suggestion, kept diff minimal).

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: diff is one host-side int expression on a static shape; no syncs, no value-dependent control flow.
- [x] Nothing derived from `data` is cached across calls: n comes from the shape only.
- [x] Weights: n/a: no weight handling touched.
- [x] Activations dtype: n/a: only a launch parameter changed; outputs bitwise identical.
- [x] Attention: n/a: MoE align kernel only.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: diff contains none.
- [x] All work inside the timed call; no output caching or replay.
- [x] No triton.autotune; first-line marker `# okc: problem=muse_moe_w4a16` unchanged.

Lint: `OK with warnings: muse_moe_w4a16/submission.py` (only prepare()-side host-sync warnings, pre-existing and outside kernel_fn).

## Risks / what could fail
low
Low-gain, low-risk launch-latency tune; the headline cannot resolve it, the _align_small_kernel rows at bs=1/8 can (Triton rows reproduce within ~1%). If 1 warp spills at bs=8 (n_spills > 0 in the log) the expected direction there flips to slower; that is the main risk. Overlap: the running bs1 fused rewrite replaces this kernel at bs=1 and the queued midm rewrite at 2..32; the running swap-AB keeps _align_small but at a larger BSM (8*NT), which does not change the variant's size class at bs<=8 (BSM 8).

## Result (after the run)
- job `38532ba10e394e3c` (submit), status done, passed True, tests 5/5
- speedup 1.3287x, 95% CI [1.3088, 1.3486], score 10121.477200687748 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 1.3249153782392282, "ci": [1.30504164756564, 1.3447891089128166], "job": "45eacf4ca2e849d2"}; base result: muse_moe_w4a16/hypotheses/done/neutral/align-parallel-per-expert-verify.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_moe_w4a16/hypotheses/done/neutral/align-small-one-warp-tiny-m.result.json`, `muse_moe_w4a16/hypotheses/done/neutral/align-small-one-warp-tiny-m.patch`; branch `hyp/muse_moe_w4a16/align-small-one-warp-tiny-m`
