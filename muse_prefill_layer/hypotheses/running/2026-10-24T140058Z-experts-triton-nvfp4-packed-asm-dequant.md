---
problem: muse_prefill_layer
slug: experts-triton-nvfp4-packed-asm-dequant
created: 2026-10-24T140058Z
base: master
area: "w13 marlin + w2 marlin (Triton NVFP4 engine, stage 5: packed inline-PTX dequant)"
change: >
  Stage 5 of the Triton NVFP4 expert rewrite, built on master (stage 4,
  experts-hybrid-marlin-turn-triton-first-turn). Only the dequant inside _moe_nvfp4_gemm_kernel in
  muse_prefill_layer/submission.py changes; the shape gate (Marlin at T<=4096, Triton above),
  EXPERT_CONFIGS, the nibble order prepare() stores, the scale tile construction (sb_lo/sb_hi),
  the epilogue, _experts_* and kernel_fn stay byte-identical. Today each code byte is loaded as
  one int32 lane (b = tl.load(b_ptrs).to(tl.int32)) and every E2M1 element costs ~6 scalar int ops
  (two and/shift/or chains), an int32->uint16 truncation, a bitcast and a bf16 multiply that
  Triton lowers as fp32 mul plus two converts: roughly 10-12 SASS instructions per weight element,
  repeated per token tile. LESSONS records this engine as per-tile dequant-overhead bound (losses
  shrinking with T; BLOCK_T 256 gaining by halving redundant dequants). Replace the lines from b =
  tl.load(...) through w_hi = ... with one tl.inline_asm_elementwise call (pack=4, is_pure=True)
  on the raw uint8 code tile [BLOCK_W, HALF] (4 bytes = one 32-bit register) plus the two
  broadcast scale tiles sb_lo, sb_hi (bf16; pack 4 = two bf16x2 registers each), returning a tuple
  (w_lo, w_hi) of bf16 [BLOCK_W, HALF]. Inside the asm, per pair of bytes: prmt.b32 spreads bytes
  (0,1) and (2,3) into the low byte of each 16-bit half (t); the low-nibble bf16x2 is ((t << 6) &
  0x01C001C0) | ((t << 12) & 0x80008000) and the high-nibble one ((t << 2) & 0x01C001C0) | ((t <<
  8) & 0x80008000) (shl.b32 + lop3.b32: the same bit placement as today, E2M1 * 2^-126 with e=0
  codes on bf16 subnormals), then mul.rn.bf16x2 (sm_90; or fma.rn.bf16x2 with a -0.0 addend) by
  the matching scale register. About 3 instructions per element instead of 10-12, and the product
  stays exact as today (<= 6 significant bits). If the Triton build rejects a tuple dtype, use two
  asm calls (one per nibble half) on the same packed input. Keep today's Python dequant behind a
  module constant PACKED_DEQUANT = True (False restores today's code exactly) so a control can
  flip it. _check_expert_gemms is unchanged and must pass for every EXPERT_CONFIGS entry (it
  compares against nvfp4.dequantize of the original packing, including the +-0.5 subnormal codes;
  if the bf16x2 multiply flushes subnormals the check says so, and the fix is to place codes at a
  larger power of two and fold the difference into SCALE_BIAS / EPILOGUE_BIAS). Diagnostics in
  prepare() only: extend the existing report print with the number of ttg.convert_layout and
  ttg.local_alloc ops in compiled.asm["ttgir"] for each variant, so the log shows whether the
  dequantized tile still round-trips through shared memory before the register-LHS wgmma (that
  decides what stage 6 restructures). No BF16 weight copy anywhere (dequant stays in registers per
  the rules ruling), no autotune, no host sync, nothing cached from inputs.
expected:
  - shapes: "*load=first_turn*"
    direction: faster
    why: "replaced kernels: _moe_nvfp4_gemm_kernel x2 is 8149-8873 us per first_turn shape (57% of sliding first_turn, 14.3 ms); the w13+w2 floor is 1.65 TFLOP = 1.67 ms at 989 TFLOP/s, so the kernel is ~5x its floor. Cutting per-element dequant from ~10-12 to ~3 instructions should take w13+w2 to ~5-6 ms (Milestone: under 7.0 ms on every first_turn shape), 2.5-3.5 ms off a 14-22 ms shape, 15-25% on the first_turn load"
  - shapes: "*load=turn_*"
    direction: neutral
    why: T<=4096 still runs Marlin; that code path is untouched
risk: medium
depends_on: experts-hybrid-marlin-turn-triton-first-turn
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 5
replaces: Marlin w13+w2 (moe_wna16_marlin_gemm x2) at every prefill T
review: proceed
status: running
review_at: 2026-10-24T164610Z
review_companions: null
stage: dispatched
claimed_at: 2026-10-24T164610Z
base_sha: 9b3df649d0389c5bd6c8e6885d9ab4d33f1e701d
base_blob: 635cfee3ac2c3f7c66840b8c8a082cad2fa41a68
branch: hyp/muse_prefill_layer/experts-triton-nvfp4-packed-asm-dequant
worktree: .worktrees/muse_prefill_layer/experts-triton-nvfp4-packed-asm-dequant
mode: submit
implemented_at: 2026-10-24T164945Z
impl_commit: 3c444875aef3f2a85c86b412e22d10ce0a6290dc
submission_sha256: 1906d1a59c89b18216925dfaec1f22e22238b39df6c45c6c9db25c5eebd01ef0
norm_sha256: 5908ab4aba141d6199bf362659b5869fea3728447c301eafd5f0f1a02825ee69
submission_blob: 526518b54879e7929be2f03a0c8084907596fa70
lint_warnings: 7
attempt_file: 2026-10-24T164945Z-submit-experts-triton-nvfp4-packed-asm-dequant.md
prepared_at: 2026-10-24T164945Z
job: 49ff8248b96547de
dispatched_at: 2026-10-24T165008Z
---

# experts-triton-nvfp4-packed-asm-dequant

## Change

Stage 5 of the Triton NVFP4 expert rewrite, built on master (stage 4, experts-hybrid-marlin-turn-triton-first-turn). Only the dequant inside _moe_nvfp4_gemm_kernel in muse_prefill_layer/submission.py changes; the shape gate (Marlin at T<=4096, Triton above), EXPERT_CONFIGS, the nibble order prepare() stores, the scale tile construction (sb_lo/sb_hi), the epilogue, _experts_* and kernel_fn stay byte-identical. Today each code byte is loaded as one int32 lane (b = tl.load(b_ptrs).to(tl.int32)) and every E2M1 element costs ~6 scalar int ops (two and/shift/or chains), an int32->uint16 truncation, a bitcast and a bf16 multiply that Triton lowers as fp32 mul plus two converts: roughly 10-12 SASS instructions per weight element, repeated per token tile. LESSONS records this engine as per-tile dequant-overhead bound (losses shrinking with T; BLOCK_T 256 gaining by halving redundant dequants). Replace the lines from b = tl.load(...) through w_hi = ... with one tl.inline_asm_elementwise call (pack=4, is_pure=True) on the raw uint8 code tile [BLOCK_W, HALF] (4 bytes = one 32-bit register) plus the two broadcast scale tiles sb_lo, sb_hi (bf16; pack 4 = two bf16x2 registers each), returning a tuple (w_lo, w_hi) of bf16 [BLOCK_W, HALF]. Inside the asm, per pair of bytes: prmt.b32 spreads bytes (0,1) and (2,3) into the low byte of each 16-bit half (t); the low-nibble bf16x2 is ((t << 6) & 0x01C001C0) | ((t << 12) & 0x80008000) and the high-nibble one ((t << 2) & 0x01C001C0) | ((t << 8) & 0x80008000) (shl.b32 + lop3.b32: the same bit placement as today, E2M1 * 2^-126 with e=0 codes on bf16 subnormals), then mul.rn.bf16x2 (sm_90; or fma.rn.bf16x2 with a -0.0 addend) by the matching scale register. About 3 instructions per element instead of 10-12, and the product stays exact as today (<= 6 significant bits). If the Triton build rejects a tuple dtype, use two asm calls (one per nibble half) on the same packed input. Keep today's Python dequant behind a module constant PACKED_DEQUANT = True (False restores today's code exactly) so a control can flip it. _check_expert_gemms is unchanged and must pass for every EXPERT_CONFIGS entry (it compares against nvfp4.dequantize of the original packing, including the +-0.5 subnormal codes; if the bf16x2 multiply flushes subnormals the check says so, and the fix is to place codes at a larger power of two and fold the difference into SCALE_BIAS / EPILOGUE_BIAS). Diagnostics in prepare() only: extend the existing report print with the number of ttg.convert_layout and ttg.local_alloc ops in compiled.asm["ttgir"] for each variant, so the log shows whether the dequantized tile still round-trips through shared memory before the register-LHS wgmma (that decides what stage 6 restructures). No BF16 weight copy anywhere (dequant stays in registers per the rules ruling), no autotune, no host sync, nothing cached from inputs.

## Expected

- `*load=first_turn*` faster: replaced kernels: _moe_nvfp4_gemm_kernel x2 is 8149-8873 us per first_turn shape (57% of sliding first_turn, 14.3 ms); the w13+w2 floor is 1.65 TFLOP = 1.67 ms at 989 TFLOP/s, so the kernel is ~5x its floor. Cutting per-element dequant from ~10-12 to ~3 instructions should take w13+w2 to ~5-6 ms (Milestone: under 7.0 ms on every first_turn shape), 2.5-3.5 ms off a 14-22 ms shape, 15-25% on the first_turn load
- `*load=turn_*` neutral: T<=4096 still runs Marlin; that code path is untouched

## Risk

Risk: ptxas may already pack part of today's chain, in which case the gain is smaller (the per-variant ttgir counts and n_regs in the log tell); inline asm with pack=4 on uint8 needs 4 consecutive bytes per thread in the code tile's layout (Triton packs them or inserts shuffles); bf16x2 subnormal handling must pass _check_expert_gemms. If the log shows the dequantized tile goes through shared memory before the dot, stage 6 should restructure the K loop rather than tune.

## Milestones

Rewrite stage 5 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) at every prefill T`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-24T164610Z by an adversarial subagent (model opus).

**Verdict: proceed.** Stage 5 of the rewrite is sound and attacks a real cost, and it is not moot under master, which still runs the scalar Python dequant at T>4096. But the expected magnitude is about 2-3x too optimistic, and with no GPU to test the hand-written PTX before a scored submit, prepare() must fall back to today's dequant if the packed variant fails to compile or fails _check_expert_gemms.

### Justification
Mechanism: plausible. On master the T>4096 path (_moe_nvfp4_gemm_kernel, BLOCK_T 256/BLOCK_W 128/BLOCK_K 128, 8 warps, 2 stages) takes 8149-8873 us per first_turn shape (37-58% of the shape), about 5x the 1.67 ms w13+w2 math floor. It dequantizes every [128,128] weight tile once per 256-token tile with scalar int32 chains: load as int32, two and/shift/or chains, a truncate, a bitcast and a bf16 mul. The bit placement checks out. (t<<6)&0x01C001C0 | (t<<12)&0x80008000 and (t<<2)&0x01C001C0 | (t<<8)&0x80008000 reproduce today's ((b&7)<<6)|((b&8)<<12) and ((b&0x70)<<2)|((b&0x80)<<8) in each 16-bit half, with no cross-half leakage after masking, and the product stays exact (<= 6 significant bits). Because the op is elementwise and is_pure, correctness does not depend on how Triton groups the 4 packed elements, as long as output j maps to input byte j. Weakest link: magnitude. (1) An ALU count bounds the gain. Dequantized elements ~ 8.25e11 MAC / 256 ~ 3.2e9 (plus padding) x ~11 instructions ~ 3.5e10 lane-ops, against ~3.3e13 lane-ops/s on 132 SMs: about 1 ms of the 8.1-8.9 ms in total, not 3 ms. (2) Measured evidence agrees. Stage 3 halved the redundant dequants (BLOCK_T 128 to 256) and took w13+w2 only from 9.1-10.0 to 8.0-8.7 ms (~1.1-1.3 ms), so dequant at BLOCK_T 256 is about 1.1-1.3 ms. A ~70% cut saves ~0.7-0.9 ms. The realistic outcome is w13+w2 ~7.3-8.0 ms (5-11% on the kernel, ~4-6% on first_turn shapes, about +2-3% on the headline since first_turn is one of three loads), not 5-6 ms. Milestone 2 (beat the incumbent kernels) is likely, the stated 'under 7.0 ms' is unlikely, and milestone 3 (1.5x floor, ~2.5 ms) is out of reach in this stage. The other ~6 ms of the 5x gap is untouched: the sb_lo/sb_hi permute/reshape/broadcast (likely its own convert_layout), the register-LHS operand conversion, 2-stage pipelining at 256x128 and per-expert padding (~512 rows per expert padded to 256-row tiles). The ttgir convert_layout/local_alloc diagnostic that targets this is the most valuable part of the run. (3) The claim that the bf16 mul is fp32 mul plus two converts is unverified on this Triton; on sm_90 it may already emit native bf16 mul that ptxas pairs into HMUL2, which would shrink the win further. (4) Packing risk: if the code tile's layout does not give each thread 4 contiguous bytes (wgmma register-A layouts hold 2-element k pairs), Triton assembles each 32-bit register with extra byte inserts or prmts, and the asm result may need a blocked-to-dot-operand convert that today's elementwise chain avoided by hoisting. Either can erase the gain, and the per-variant n_regs/ttgir counts are what tell. Expected directions are consistent with TASK.md: first_turn faster, and turn_* structurally neutral because the stage-4 gate routes T<=4096 to Marlin and that code is byte-identical.

### Confounders on the current kernel
- experts-triton-w4a16-first-turn-blockt256 (stage 3): sets BLOCK_T 256, which already halved the per-token-tile dequant count. The gain from cheaper dequant is smallest at this tile and would be about 2x larger at BLOCK_T 128. This sizes the effect but does not confound its attribution (the change applies at the same config), so no A/B is needed. Note it when sizing stage 6 (turn_4, BLOCK_T 128), where packed dequant should matter relatively more.
- experts-hybrid-marlin-turn-triton-first-turn (stage 4): the T<=4096 Marlin gate means only the first_turn shapes run the changed kernel. turn_1 and turn_4 are byte-identical code, so any movement there is run drift (LESSONS: 1.5-2.5% per shape). _check_expert_gemms still compiles and checks the BLOCK_T 64/128 configs in prepare(), so a packed-asm bug in those configs fails the run even though they are never timed.
- sglang-pipeline-parity: everything around the experts. It does not interact with the dequant path.

### Interference and A/B needs
No companions. This is a rewrite stage judged on _moe_nvfp4_gemm_kernel's own time per first_turn shape in top_kernels. Triton kernel times reproduce within ~1%, so a 0.5-1 ms move on 8.1-8.9 ms is readable from one run without a headline A/B. A PACKED_DEQUANT=False control would be byte-identical in behavior to master, which is already measured (fb1e52d2119d4175), so it adds nothing. Reverting stage 3 or 4 would only re-measure known configs. The follow-up is stage 6 (experts-triton-packed-dequant-turn4), informed by this run's ttgir counts.

### Open questions
- Scored-submit safety, which I recommend as a condition of dispatch. With no GPU here the PTX (prmt selectors, packing order, tuple return) cannot be tested before the run, and any compile error or _check_expert_gemms mismatch aborts prepare() and burns the scored submit. Make PACKED (or PACKED_DEQUANT) a tl.constexpr kernel argument, not a module global: Triton rejects or ignores a global changed after compile. In prepare(), try the packed variant for every EXPERT_CONFIGS entry under try/except around compile plus _check_expert_gemms. On failure, print the exception and the mismatch summary to the log and set state['packed']=False so every variant runs today's dequant. _experts_triton then passes state's flag. This is a small, deliberate deviation from 'byte-identical _experts_*'. kernel_fn branches only on prepare-time state (fine under the rules), and the run then measures master's kernel instead of failing.
- Does Triton on this image hoist the dot-operand convert_layout above today's elementwise dequant chain, so the dequant runs directly in the wgmma register-A layout? If it does, and it refuses to hoist above inline_asm with pack=4, the packed version adds a blocked-to-dot-operand convert, possibly through shared memory, and could be slower despite fewer ALU ops. Print the convert_layout/local_alloc counts for both PACKED=False and PACKED=True (compile both in prepare) so the log answers this directly.
- Is the bf16 multiply already native HMUL2.BF16 on sm_90 in this Triton version? If yes, the per-element saving is ~6-8 instructions, not 7-9, and the win is near the low end of ~0.5-0.9 ms.
- Should the sb_lo/sb_hi permute/reshape/broadcast also move into the asm? Each register could take the 8 compact scales per row and select by group index. It is left byte-identical here and may be a comparable cost. The ttgir counts should show whether it is a convert_layout of its own.


## Implementation notes

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

## Rules check

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
