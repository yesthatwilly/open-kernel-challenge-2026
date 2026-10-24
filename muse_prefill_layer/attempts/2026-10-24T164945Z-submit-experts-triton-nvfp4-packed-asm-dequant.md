# muse_prefill_layer · 2026-10-24T164945Z · submit · experts-triton-nvfp4-packed-asm-dequant

Hypothesis: `muse_prefill_layer/hypotheses/running/2026-10-24T140058Z-experts-triton-nvfp4-packed-asm-dequant.md` (branch `hyp/muse_prefill_layer/experts-triton-nvfp4-packed-asm-dequant`).

- **Job:** `49ff8248b96547de` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `3c44487`

## Goal
Stage 5 of the Triton NVFP4 expert rewrite, built on master (stage 4, experts-hybrid-marlin-turn-triton-first-turn). Only the dequant inside _moe_nvfp4_gemm_kernel in muse_prefill_layer/submission.py changes; the shape gate (Marlin at T<=4096, Triton above), EXPERT_CONFIGS, the nibble order prepare() stores, the scale tile construction (sb_lo/sb_hi), the epilogue, _experts_* and kernel_fn stay byte-identical. Today each code byte is loaded as one int32 lane (b = tl.load(b_ptrs).to(tl.int32)) and every E2M1 element costs ~6 scalar int ops (two and/shift/or chains), an int32->uint16 truncation, a bitcast and a bf16 multiply that Triton lowers as fp32 mul plus two converts: roughly 10-12 SASS instructions per weight element, repeated per token tile. LESSONS records this engine as per-tile dequant-overhead bound (losses shrinking with T; BLOCK_T 256 gaining by halving redundant dequants). Replace the lines from b = tl.load(...) through w_hi = ... with one tl.inline_asm_elementwise call (pack=4, is_pure=True) on the raw uint8 code tile [BLOCK_W, HALF] (4 bytes = one 32-bit register) plus the two broadcast scale tiles sb_lo, sb_hi (bf16; pack 4 = two bf16x2 registers each), returning a tuple (w_lo, w_hi) of bf16 [BLOCK_W, HALF]. Inside the asm, per pair of bytes: prmt.b32 spreads bytes (0,1) and (2,3) into the low byte of each 16-bit half (t); the low-nibble bf16x2 is ((t << 6) & 0x01C001C0) | ((t << 12) & 0x80008000) and the high-nibble one ((t << 2) & 0x01C001C0) | ((t << 8) & 0x80008000) (shl.b32 + lop3.b32: the same bit placement as today, E2M1 * 2^-126 with e=0 codes on bf16 subnormals), then mul.rn.bf16x2 (sm_90; or fma.rn.bf16x2 with a -0.0 addend) by the matching scale register. About 3 instructions per element instead of 10-12, and the product stays exact as today (<= 6 significant bits). If the Triton build rejects a tuple dtype, use two asm calls (one per nibble half) on the same packed input. Keep today's Python dequant behind a module constant PACKED_DEQUANT = True (False restores today's code exactly) so a control can flip it. _check_expert_gemms is unchanged and must pass for every EXPERT_CONFIGS entry (it compares against nvfp4.dequantize of the original packing, including the +-0.5 subnormal codes; if the bf16x2 multiply flushes subnormals the check says so, and the fix is to place codes at a larger power of two and fold the difference into SCALE_BIAS / EPILOGUE_BIAS). Diagnostics in prepare() only: extend the existing report print with the number of ttg.convert_layout and ttg.local_alloc ops in compiled.asm["ttgir"] for each variant, so the log shows whether the dequantized tile still round-trips through shared memory before the register-LHS wgmma (that decides what stage 6 restructures). No BF16 weight copy anywhere (dequant stays in registers per the rules ruling), no autotune, no host sync, nothing cached from inputs.

## What changed, and where
Worktree `.worktrees/muse_prefill_layer/experts-triton-nvfp4-packed-asm-dequant`, file `muse_prefill_layer/submission.py` only (112 insertions, 17 deletions, uncommitted).

- New module constant `PACKED_DEQUANT = True`. False skips the packed variant entirely, so the Triton path runs master's scalar dequant.
- New `@triton.jit _e2m1x4_dequant_bf16(b, s_lo, s_hi)`: one `tl.inline_asm_elementwise` (pack=4, is_pure=True, dtype=(bf16, bf16), constraints `=r,=r,=r,=r,r,r,r,r,r`). Operand map: $0,$1 = w_lo bf16x2 regs, $2,$3 = w_hi, $4 = 4 packed code bytes, $5,$6 = s_lo, $7,$8 = s_hi (the same result-major register order as Triton's test_inline_asm_packed_multiple_outputs). Body: `prmt.b32 t0, $4, 0, 0x4140` / `0x4342` spreads bytes (0,1)/(2,3) into the low byte of each half. Then, per register, shl/and/shl/and/or (ptxas should fuse to shl+lop3) gives the spec's masks 0x01C001C0 / 0x80008000 at shifts 6/12 (low nibble) and 2/8 (high nibble). `fma.rn.bf16x2 d, x, s, 0x80008000` (-0.0 addend) is the multiply. I used fma rather than mul.rn.bf16x2 because fma works on sm_80+ and needs no PTX 7.8 check. The spec allows either.
- I checked the bit placement offline with a Python simulation of prmt + shifts + masks against the scalar `((b&7)<<6)|((b&8)<<12)` and `((b&0x70)<<2)|((b&0x80)<<8)` on 200k random 4-byte words, with element j = byte j: 0 mismatches. I could not test the PTX itself or Triton's packing order (no GPU).
- `_moe_nvfp4_gemm_kernel` gets a `PACKED: tl.constexpr` argument, as the review requires. `if PACKED:` loads the raw uint8 code tile and calls the asm helper. The else branch is master's lines unchanged (`.to(tl.int32)` load, the two `_e2m1_bits_to_bf16(...) * sb_*` lines). The sb_lo/sb_hi construction, the dots and the epilogue are untouched.
- `_nvfp4_moe_gemm` takes a positional `packed` argument and passes `PACKED=packed`. Its prepare()-only report line now also prints `packed=`, plus `convert_layout=N local_alloc=N` counted by regex `(?:ttg|triton_gpu)[.]convert_layout|local_alloc` over `compiled.asm["ttgir"]` (new helper `_ttgir_counts`, which prints `?` if asm is unavailable).
- `_check_expert_gemms(bank, weights, packed)`: the only change is the flag passed through. The check logic is unchanged.
- prepare(): it always runs `_check_expert_gemms(..., packed=False)` first. That is master's check, still fatal on failure, and it gives the control layout counts. If PACKED_DEQUANT is set, it then runs `_check_expert_gemms(..., packed=True)` under try/except. On any exception it prints the type and the first 3000 chars of the message (including the mismatch summary) and leaves `state["packed"] = False`. It then prints `[expert gemm] packed dequant ON|OFF`.
- Deviation, a small extra safety step beyond the review: `_check_experts_e2e` (Triton vs Marlin at T=8192) also falls back. If it fails with packed=True, prepare logs it, sets packed False and reruns the e2e check, which must then pass. A failure with packed=False raises as on master.
- `_experts_triton` passes `state["packed"]` to both GEMMs. This is the review's sanctioned deviation from "byte-identical _experts_*". It branches on prepare-time state only, never on input values. `_experts`, `_experts_marlin`, kernel_fn, EXPERT_CONFIGS, the nibble order and the Marlin path are byte-identical.
- Limit: a CUDA-level fault in the packed kernel (sticky context error) cannot be recovered by the fallback. The asm touches no memory, so this is unlikely. Compile errors and value mismatches are recovered.
- Compile cost: prepare() now compiles 12 expert-GEMM variants instead of 6 (3 configs x 2 GEMMs x packed on/off), well inside 300 s.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_prefill_layer/experts-triton-nvfp4-packed-asm-dequant diff`:

- [x] CUDA-graph capturable: the diff adds no `.item()`/`.cpu()`/tensor prints in kernel_fn's call graph. The new synchronize calls are in prepare() only (lint: outside kernel_fn's call graph). The `state["packed"]` branch is on a prepare-time Python bool, not on input values. Allocations are unchanged.
- [x] Nothing derived from `data` is cached across calls: `state["packed"]` comes from the prepare() checks on weights and synthetic data only.
- [x] Weights: there is no new weight storage and no BF16 weight copy. The packed path dequantizes the same NVFP4 codes in registers per tile. The nibble reorder is unchanged (lossless, checked).
- [x] Activations BF16 where SGLang keeps BF16: the A tiles, dot inputs and outputs are unchanged. The dequantized W tile is BF16 as before and the product is exact. KV cache/queries: n/a, attention is untouched.
- [x] Attention sees every key: n/a, the attention code is not in the diff.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: the only new import is stdlib `re`, used on Triton's own compiled TTGIR string in prepare().
- [x] All work happens inside the timed call; no output caching or replay: the expert GEMMs still run per call. prepare() only compiles and checks.
- [x] First-line marker `# okc: problem=muse_prefill_layer` kept. No triton.autotune.

okc_lint output:

```
warn  host-sync    L536: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L578: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L632: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L637: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L648: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L656: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L665: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: muse_prefill_layer/submission.py
```

`python3 -I -m py_compile`: OK (__pycache__ removed).

## Risks / what could fail
medium
Risk: ptxas may already pack part of today's chain, in which case the gain is smaller (the per-variant ttgir counts and n_regs in the log tell); inline asm with pack=4 on uint8 needs 4 consecutive bytes per thread in the code tile's layout (Triton packs them or inserts shuffles); bf16x2 subnormal handling must pass _check_expert_gemms. If the log shows the dequantized tile goes through shared memory before the dot, stage 6 should restructure the K loop rather than tune.

## Result (after the run)
- job `49ff8248b96547de` (submit), status done, passed True, tests 5/5
- speedup 1.0556x, 95% CI [1.0397, 1.0714], score 367009.08611723257 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.0591912383762452, "ci": [1.0433033698006016, 1.0750791069518888], "job": "ba9c16b9cc6b4605"}; base result: muse_prefill_layer/hypotheses/done/gain/experts-hybrid-marlin-turn-triton-first-turn.result.json
- **outcome: neutral** (within-ci)
- files: `muse_prefill_layer/hypotheses/done/neutral/experts-triton-nvfp4-packed-asm-dequant.result.json`, `muse_prefill_layer/hypotheses/done/neutral/experts-triton-nvfp4-packed-asm-dequant.patch`; branch `hyp/muse_prefill_layer/experts-triton-nvfp4-packed-asm-dequant`
