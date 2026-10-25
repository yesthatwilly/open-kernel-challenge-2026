---
problem: muse_decode_layer
slug: moe-chain-cuda-swapab-expert-once-fix-inproc
created: 2026-10-25T001824Z
base: master
area: w13 marlin + silu-mul + w2 marlin (T>32)
change: >
  Stage 1b FIX (numbered stage 2 only because stage is an int; the original stage plan's stage 2
  still comes after this) of failed moe-chain-cuda-swapab-expert-once-large-t (job
  1bba1bab70da4f61: built, then a sticky 715 CUDA_ERROR_ILLEGAL_INSTRUCTION in the first prepare
  self-check launch killed the worker, 0/0 tests; REWRITE_READ_2026-10-25 section 5, LESSONS
  2026-10-25). Take its patch (done/failed/moe-chain-cuda-swapab-expert-once-large-t.patch)
  re-applied on current master (stack-align-par-sidestream-sliding1: parallel align at T>32 and
  MoE dispatch on the side stream; keep the same gate tokens > 32 with its OWN bound constant,
  LESSONS stack note, and launch the three new kernels on the same side stream master uses for MoE
  dispatch). Design unchanged; only these fixes: (1) IN-PROCESS GUARDED FIRST LAUNCH (never in a
  child process: the rules statically reject submissions that start processes and okc_lint flags
  multiprocessing (LESSONS 2026-10-25, corrected)): in prepare(), after the build, confirm the
  sm_90a image actually loaded (log torch.cuda.get_device_capability() == (9, 0), the nvcc
  -gencode flags showing only compute_90a/sm_90a, and cudaFuncGetAttributes
  binaryVersion/ptxVersion for moe_route_list, moe_w13_silu_swapab and moe_w2_swapab; anything
  else turns the path OFF); then a 1-CTA micro canary of dequant8 + one mma tile on synthetic
  weights, then a tiny synthetic canary launch of moe_route_list, moe_w13_silu_swapab and
  moe_w2_swapab (separate launches, smallest T above the gate), each followed by
  torch.cuda.synchronize() and a print so a fault names its kernel; only then the whole
  T=40/128/256 self-check in-process, enabling the path only if everything passes and otherwise
  keeping master Marlin. (2) Replace createpolicy.fractional.L2::evict_first +
  cp.async.cg...L2::cache_hint with plain cp.async.cg.shared.global [d],[s],16 (the form
  dense-chain and splitkv run on the judge). (3) In dequant8 replace the inline fma.rn.bf16x2 with
  a -0 addend by __hmul2 on __nv_bfloat162 (scale fold unchanged). (4) Build flags: add -lineinfo;
  set TORCH_CUDA_ARCH_LIST=9.0a / only -gencode arch=compute_90a,code=sm_90a so a single sm_90a
  cubin is built (hygiene; no wgmma). Keep 221 KB dyn smem at 256 threads but print
  cudaFuncGetAttributes and the max-dynamic-smem attribute result before the canary. Print
  [moe-cuda-large-t] ON|OFF with self-check status, faulting kernel name if any, max row rel, and
  per-kernel CUDA-event times at T=128/256. The first measurement of this hypothesis must be an
  `okc check`, never a scored submit (a sticky fault cannot be recovered in-process).
expected:
  - shapes: "*load=verify_16 *"
    direction: faster
    why: "Marlin w13+w2+act_and_mul 254/271 us (sliding/full) at ~1.1-1.2 TB/s vs MoE floor ~101 us; expert-once swap-AB streams each local expert once per <=64-route chunk: target replaced sum <= Marlin (M1/M2), ~150 us (M3 ~1.5x floor)"
  - shapes: "*load=verify_32 *"
    direction: faster
    why: Marlin 443/427 us vs floor ~109 us; target <= ~200 us
  - shapes: "*load=decode_*"
    direction: flat
    why: "T<=32 outside gate: control"
  - shapes: "*load=verify_1 *"
    direction: flat
    why: "T=8 outside gate: control"
risk: high
depends_on: moe-chain-cuda-swapab-expert-once-large-t
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 2
replaces: moe_align + Marlin w13 + act_and_mul + Marlin w2 at tokens > 32
review: proceed
status: running
review_at: 2026-10-25T002143Z
review_companions: null
stage: dispatched
claimed_at: 2026-10-25T002143Z
base_sha: e65eda8e263ae1408bdb9e263068ca89a29ceb0d
base_blob: 81d11fec488c02b0e7fe3a48f764aca20bb6a438
branch: hyp/muse_decode_layer/moe-chain-cuda-swapab-expert-once-fix-inproc
worktree: .worktrees/muse_decode_layer/moe-chain-cuda-swapab-expert-once-fix-inproc
mode: check
restack_of: null
stale_base: true
implemented_at: 2026-10-25T003525Z
impl_commit: 1177e314e5b0c1b2d21bb03967270350659d0f15
submission_sha256: 6d41970e84202091134a0a93b2751855d8c308929f1f67bbae40ff4ec9ee1cd2
norm_sha256: abc0472c2583c0b8bd66ea59c18519191a43b73fac1b73155b7bfea42c41cd10
submission_blob: 451c3c09efdc6fc55db461ae20f27cd8e1cd4e9c
lint_warnings: 25
attempt_file: 2026-10-25T003526Z-check-moe-chain-cuda-swapab-expert-once-fix-inproc.md
prepared_at: 2026-10-25T003526Z
job: 4ebd055adab34ecd
dispatched_at: 2026-10-25T003532Z
---

# moe-chain-cuda-swapab-expert-once-fix-inproc

## Change

Stage 1b FIX (numbered stage 2 only because stage is an int; the original stage plan's stage 2 still comes after this) of failed moe-chain-cuda-swapab-expert-once-large-t (job 1bba1bab70da4f61: built, then a sticky 715 CUDA_ERROR_ILLEGAL_INSTRUCTION in the first prepare self-check launch killed the worker, 0/0 tests; REWRITE_READ_2026-10-25 section 5, LESSONS 2026-10-25). Take its patch (done/failed/moe-chain-cuda-swapab-expert-once-large-t.patch) re-applied on current master (stack-align-par-sidestream-sliding1: parallel align at T>32 and MoE dispatch on the side stream; keep the same gate tokens > 32 with its OWN bound constant, LESSONS stack note, and launch the three new kernels on the same side stream master uses for MoE dispatch). Design unchanged; only these fixes: (1) IN-PROCESS GUARDED FIRST LAUNCH (never in a child process: the rules statically reject submissions that start processes and okc_lint flags multiprocessing (LESSONS 2026-10-25, corrected)): in prepare(), after the build, confirm the sm_90a image actually loaded (log torch.cuda.get_device_capability() == (9, 0), the nvcc -gencode flags showing only compute_90a/sm_90a, and cudaFuncGetAttributes binaryVersion/ptxVersion for moe_route_list, moe_w13_silu_swapab and moe_w2_swapab; anything else turns the path OFF); then a 1-CTA micro canary of dequant8 + one mma tile on synthetic weights, then a tiny synthetic canary launch of moe_route_list, moe_w13_silu_swapab and moe_w2_swapab (separate launches, smallest T above the gate), each followed by torch.cuda.synchronize() and a print so a fault names its kernel; only then the whole T=40/128/256 self-check in-process, enabling the path only if everything passes and otherwise keeping master Marlin. (2) Replace createpolicy.fractional.L2::evict_first + cp.async.cg...L2::cache_hint with plain cp.async.cg.shared.global [d],[s],16 (the form dense-chain and splitkv run on the judge). (3) In dequant8 replace the inline fma.rn.bf16x2 with a -0 addend by __hmul2 on __nv_bfloat162 (scale fold unchanged). (4) Build flags: add -lineinfo; set TORCH_CUDA_ARCH_LIST=9.0a / only -gencode arch=compute_90a,code=sm_90a so a single sm_90a cubin is built (hygiene; no wgmma). Keep 221 KB dyn smem at 256 threads but print cudaFuncGetAttributes and the max-dynamic-smem attribute result before the canary. Print [moe-cuda-large-t] ON|OFF with self-check status, faulting kernel name if any, max row rel, and per-kernel CUDA-event times at T=128/256.
 The first measurement of this hypothesis must be an `okc check`, never a scored submit (a sticky fault cannot be recovered in-process).

## Expected

- `*load=verify_16 *` faster: Marlin w13+w2+act_and_mul 254/271 us (sliding/full) at ~1.1-1.2 TB/s vs MoE floor ~101 us; expert-once swap-AB streams each local expert once per <=64-route chunk: target replaced sum <= Marlin (M1/M2), ~150 us (M3 ~1.5x floor)
- `*load=verify_32 *` faster: Marlin 443/427 us vs floor ~109 us; target <= ~200 us
- `*load=decode_*` flat: T<=32 outside gate: control
- `*load=verify_1 *` flat: T=8 outside gate: control

## Risk

supersedes moe-chain-cuda-swapab-expert-once-fix, held for a child-process self-check

## Milestones

Rewrite stage 2 of `moe_align + Marlin w13 + act_and_mul + Marlin w2 at tokens > 32`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T002143Z by an adversarial subagent (model opus).

**Verdict: proceed.** This rewrite stage only fixes faults, and the target is real: Marlin takes 254-443 us at verify_16/32 against a MoE floor of about 101-109 us. The fixes don't identify the 715 root cause, so treat this run as a diagnostic okc check whose main job is to name the faulting kernel. That only works if the logging is unbuffered and announces each launch before it starts.

### Justification
Mechanism is plausible. FINDINGS shows Marlin at 271.6/326.1 us (sliding/full verify_16) and 443.2/449.7 us (verify_32), 47-70% of layer time, running at about 1.1-1.2 TB/s against a weight-stream floor of roughly 101-109 us. A persistent 132-CTA swap-AB mma.sync kernel that streams each touched local expert once per chunk of up to 64 routes can reasonably reach about 2 TB/s on H100. The 'decode_* / verify_1 flat' controls are consistent because the tokens > 32 gate only admits T=128/256. Rewrite milestones M1-M3 can be read from profile top_kernels (moe_w13_silu_swapab plus moe_w2_swapab plus moe_route_list against Marlin x2 plus act_and_mul plus align). Weakest link: neither of the 'proven-form' swaps clearly removes the fault. (3) is probably a SASS no-op, because cuda_bf16.hpp implements __hmul2 for bf16 on sm_90 as fma.rn.bf16x2 with a -0.0 addend, i.e. the same instruction as before. (2) createpolicy/cache_hint is standard CUTLASS PTX on sm_80+ and an unlikely trap source. ptxas accepted every form, so an encoding problem is less likely than a trap inside a .sync.aligned collective (ldmatrix/mma inside `if (f < me_nf)`), a stray trap/assert, or a bad smem configuration. The run will probably fault again. Its value is the canary ladder that attributes the fault, which the sibling moe failure never got. -lineinfo will not name the instruction in a judge log without compute-sanitizer. Two design points check out: chunk capacity (routes//64 + 33 >= worst case sum ceil(c_e/64) <= routes/64 + 32), and the bf16 rounding of hg/hu before silu, which matches Marlin's bf16 w13 output.

### Confounders on the current kernel
- stack-align-par-sidestream-sliding1: at T>32 the new moe_route_list replaces master's parallel align on the side stream. The incumbent is therefore parallel-align + Marlin, which is the right baseline. Stream placement matters, though: the new kernels must run on state['side_stream'] inside the existing wait_stream fence (submission.py ~1830-1844). Otherwise a gain or loss may come partly from lost or gained overlap rather than from the GEMMs. Read kernel times, not the headline.
- marlin-block-m-cap-16: the incumbent Marlin times (254-449 us) are measured with block_m capped at 16 at T=128/256. A different cap would move the incumbent, not this kernel. Comparing against today's master is correct and needs no A/B.
- post-expert-norm-triton-small-t: gated to small T, so at T>32 the SGLang fused_post_expert_norm_weighted_sum consumes y. It is not a confounder, as long as y's layout ([T,8,K], remote rows never read, slot_expert_ids=local_ids) is unchanged.

### Interference and A/B needs
No companions. This is a rewrite stage whose first measurement must be an okc check. Its main outcomes are 'faults with the kernel named' or 'passes, and the kernel times can be read against Marlin per shape', and a revert or tune of any landed item disambiguates neither. The decode_*/verify_1 shapes act as built-in controls because they fall outside the gate. If it passes, the next stage is the follow-up.

### Open questions
- Prints must use flush=True (or sys.stdout.flush()), and each kernel needs a 'launching <kernel>' line before its launch as well as an 'ok' line after its synchronize. A sticky 715 kills the worker, and buffered stdout would lose exactly the line that names the kernel.
- Order the canaries so each step adds one suspect form: cvt e4m3x2 alone, then dequant8 (hmul2) alone, then ldmatrix.x4 plus one mma.sync tile, then cp.async.cg at the 221 KB smem configuration, then the route list, then w13, then w2. A fault then identifies a form, not just a kernel.
- Is the __hmul2 change actually different SASS from the old fma.rn.bf16x2 with a -0 addend? If not, add a cuobjdump/SASS note to the attempt file so the next stage does not count it as an eliminated suspect.
- If the canaries pass but the full launch faults, the remaining suspects are an unconverged .sync.aligned collective (me_nf or item-end not truly CTA-uniform across persistent items) and the 221 KB smem configuration. A next stage could test 2 stages and about 120 KB smem.


## Implementation notes

Worktree: `.worktrees/muse_decode_layer/moe-chain-cuda-swapab-expert-once-fix-inproc`, edited on top of stage 1's implementation (HEAD e65eda8). Only `muse_decode_layer/submission.py` changed.

**Deviation: not re-stacked onto today's master.** The coordinator dispatched this build on stage 1's base (stale_base: true). The stage-1 patch does not apply to current master: 4 of its 5 hunks reject, because master's _experts / align / dense-chain regions differ. Because the review frames this run as a diagnostic check whose job is to name the faulting kernel, the build stays on stage 1's base. As a result:
- The new kernels run where stage 1 ran them. On this base that is the main stream; only the router is on the side stream.
- The gate keeps its own constant, `MOE_CUDA_LARGE_T_MIN = 32`, i.e. tokens > 32.
- Kernel times from this run should be read against Marlin within this same run (the timing line below), not against master's headline.
- If this run passes, the restack onto master (parallel align plus side-stream MoE dispatch) is a separate step.

**CUDA (_LT_SRC):**
- (2) `cp_async16_ef` (createpolicy.fractional.L2::evict_first plus cp.async.cg...L2::cache_hint with a 64-bit policy) is replaced by `cp_async16_cg`, i.e. plain `cp.async.cg.shared.global [d],[s],16`. The `createpolicy` asm in swapab_body is removed.
- (3) `mul_bf16x2` in dequant8 now uses `__hmul2` on `__nv_bfloat162`. Per the review, cuda_bf16.hpp may lower this to the same `fma.rn.bf16x2` with a -0 addend on sm_90. No nvcc is available on this node, so there is no cuobjdump/SASS diff. **Do not count this as an eliminated suspect** unless the dequant canary isolates it.
- New canary kernels, with `lt_init` and `lt_canary(which, a, b, c, out)` host entries:
  - `lt_arch_probe` reports `__CUDA_ARCH__` and the `__CUDA_ARCH_FEAT_SM90_ALL` flag; 0 is the error flag.
  - `lt_canary_cvt` (cvt e4m3x2 alone)
  - `lt_canary_dequant` (dequant8 alone)
  - `lt_canary_mma` (scale cvt plus dequant8 plus the swizzled slab plus ldmatrix.x4 plus 4x mma.sync, one warp, one tile)
  - `lt_canary_cpasync` (cp.async.cg, cp.async.ca 16, and cp.async.ca src-size 0 at 221184 B dynamic smem x 256 threads, 1 CTA)
- `lt_init` now also returns:
  - the device opt-in max smem
  - for each of 8 kernels: maxDynamicSharedSizeBytes, binaryVersion, ptxVersion and static smem
- The kernel design (tiling, chunking, epilogue, route list) is unchanged.

**Build:**
- `extra_cuda_cflags = -O3 -lineinfo -gencode=arch=compute_90a,code=sm_90a`. This builds a single sm_90a cubin with no PTX and no plain sm_90.
- `TORCH_CUDA_ARCH_LIST=9.0a` is set around load_inline and restored in a `finally`.
- The flags are printed.
- The module name is bumped to `muse_decode_moe_large_t_v2`.
- There is no wgmma or setmaxnreg, so there is no sm_90a-only PTX to guard. The feature macro is still read by the probe, and the gate requires it.

**Python, prepare() only:**
- `_lt_log` prints with flush=True.
- `_lt_step(cur, name, fn)` prints `launching <name>`, runs the launch, calls `torch.cuda.synchronize()`, then prints `ok <name>`. `cur['stage']` keeps the last launch that was announced.
- `_lt_experts` gained `hook=None`, which wraps each of moe_route_list, w13, w2 and the post-norm combine. kernel_fn passes None, so its behaviour and launches are unchanged.

The ladder in `_lt_setup` runs in this order. Any miss turns the path OFF and stops further launches.
1. **_lt_arch_check:**
   - log the capability, which must be (9, 0)
   - log the flags
   - log binaryVersion and ptxVersion for all 8 kernels; each must be 90/90
   - check that maxDynamicSharedSizeBytes is at least the requested smem
   - launch the probe, which must report 900 / 1
2. **Canaries, in the order the review suggested.** Each step adds one suspect form, and each is checked bitwise or against a float64 reference:
   - cvt over every non-NaN E4M3 byte
   - dequant8 over 256 random words, exact bf16
   - the mma tile, row rel <= 1e-4
   - cp.async, bytewise reversed plus zero-fill
3. **Retile.**
4. **Smallest T above the gate (T=33), synthetic.** moe_route_list, moe_w13_silu_swapab and moe_w2_swapab run as separate announced launches, followed by a finiteness check.
5. **The full T=40/128/256 self-check.** Every Marlin and CUDA launch is announced and synchronized. The self-check now stops at the first failing T.
6. **If everything passes, per-kernel CUDA-event timing at T=128/256** for route_list, w13, w2 and the combine, plus route+w13+w2 and the Marlin path total, print only.
7. **Final lines:**
   - `[moe-cuda-large-t] ON|OFF: compile, max row rel, last launch / stage <name>, reason`
   - `PATH CHOSEN = CUDA|MARLIN for tokens > 32`

**Review fixes applied:**
- All prints are flushed.
- Each launch gets a 'launching' line before it and an 'ok' line after its synchronize.
- The canary ordering follows the review's open question.
- No child process is used anywhere.
- -lineinfo is added.
- The __hmul2 change is flagged as possibly not changing the SASS.

**Not addressed in this stage:**
- Testing 2 stages at about 120 KB smem. That is for the next stage if the canaries pass and the full launch faults.
- Bounded spin waits are n/a: the code has no mbarrier and no spin loops.

The sticky-fault caveat still applies. If any launch hits a 715, the worker dies at the next CUDA call; nothing in-process can recover it. The value of this run is that the last `launching <name>` line in the log names the kernel and instruction form. **The first measurement must be `okc check`, never a scored submit.**

## Rules check

Checked against `git -C .worktrees/muse_decode_layer/moe-chain-cuda-swapab-expert-once-fix-inproc diff`:
- [x] CUDA-graph capturable. kernel_fn's path only gained a Python `hook=None` lambda wrapper. Every synchronize, .cpu() and print is in prepare() (_lt_setup and its helpers). There is no allocation or branching on input values; the gate branches on the shape and a host flag.
- [x] Nothing derived from `data` is cached. prepare() uses only synthetic inputs.
- [x] Weights: the same lossless NVFP4 retiling as stage 1, with the round trip asserted. The global scales are fp32 with an exact 2^7 fold.
- [x] Activations stay BF16. act is rounded to bf16 as Marlin's w13 output is. The KV cache and queries are untouched.
- [x] Attention: n/a, attention is untouched.
- [x] No subprocess, multiprocessing, fork or spawn (grep of the diff is empty). No timing patches, harness I/O, module edits, reference import or network. `os.environ` is touched only for TORCH_CUDA_ARCH_LIST around the build, and it is restored.
- [x] All work happens inside the timed call; there is no output caching or replay.
- [x] An unsupported device or image, a build failure or a canary miss lands on Marlin with loud OFF and PATH CHOSEN lines and does not raise. A sticky 715 is the one exception: it is unrecoverable in-process by nature.
- [x] Bounded waits: n/a, there are no mbarrier or spin waits.

okc_lint output:
```
warn  host-sync    L885: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L990: q13 = torch.empty((count, w13[0].numel()), dtype=torch.uint8, device=w13.device)  (allowed only where SGLang does the same)
warn  dtype        L991: s13 = torch.empty((count, weights["w13_sf"][0].numel()), dtype=torch.uint8, device=w13.device)  (allowed only where SGLang does the same)
warn  dtype        L992: q2 = torch.empty((count, w2[0].numel()), dtype=torch.uint8, device=w2.device)  (allowed only where SGLang does the same)
warn  dtype        L993: s2 = torch.empty((count, weights["w2_sf"][0].numel()), dtype=torch.uint8, device=w2.device)  (allowed only where SGLang does the same)
warn  host-sync    L1102: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1176: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L1191: dummy = torch.zeros((16,), dtype=torch.uint8, device=device)  (allowed only where SGLang does the same)
warn  dtype        L1202: src = torch.tensor(vals, dtype=torch.uint8)  (allowed only where SGLang does the same)
warn  dtype        L1203: ref = src.view(torch.float8_e4m3fn).float().to(torch.float16)  (allowed only where SGLang does the same)
warn  dtype        L1204: out = torch.zeros((256,), dtype=torch.float16, device=device)  (allowed only where SGLang does the same)
warn  host-sync    L1207: if not torch.equal(out.cpu().view(torch.int16), ref.view(torch.int16)):  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L1213: s = torch.randint(0x20, 0x58, (2 * n,), generator=gen, dtype=torch.int64).to(torch.uint8) \  (allowed only where SGLang does the same)
warn  host-sync    L1225: got = out.cpu().view(torch.bfloat16).float()  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L1233: s_val = s_bytes.to(torch.uint8).view(torch.float8_e4m3fn).double()  (allowed only where SGLang does the same)
warn  dtype        L1238: sc = torch.zeros((32, 8), dtype=torch.uint8)  (allowed only where SGLang does the same)
warn  host-sync    L1254: e_mma = _row_rel(out.cpu().double().view(16, 8), ref)  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L1260: src = torch.randint(0, 256, (LT_SMEM_BYTES,), generator=gen, dtype=torch.int64).to(torch.uint8)  (allowed only where SGLang does the same)
warn  dtype        L1265: out = torch.zeros((LT_SMEM_BYTES,), dtype=torch.uint8, device=device)  (allowed only where SGLang does the same)
warn  host-sync    L1268: if not torch.equal(out.cpu(), exp):  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L1300: dummy = torch.zeros((16,), dtype=torch.uint8, device=device)  (allowed only where SGLang does the same)
warn  host-sync    L1302: arch, feat = (int(v) for v in probe.cpu())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1331: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1410: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1414: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: .worktrees/muse_decode_layer/moe-chain-cuda-swapab-expert-once-fix-inproc/muse_decode_layer/submission.py
```
py_compile: OK; __pycache__ removed.

## Pre-flight audit

Audited 2026-10-24 by an opus subagent, read-only (no GPU or nvcc on this node).

**Verdict: dispatch. No fixes were applied.** okc_lint returned OK with warnings; every warning is a prepare()-only sync or a uint8/fp16 test tensor. py_compile passed.

**What was checked:**

- **Canary gates can pass.** The references were derived by hand against the m16n8k16 A, B and C fragment layouts, the ldmatrix.x4 lane mapping and the slab swizzle, and they match the kernels:
  - The cvt canary covers every non-NaN E4M3 byte. Each value is exact in fp16, so a bitwise compare is valid.
  - In the dequant canary, scale bytes 0x20-0x57 are normal E4M3. s·2^119 is exact in bf16 and below 2^127. Every product is exact, so a bitwise compare is valid.
  - The mma canary uses a row rel bound of 1e-4 on a 64-term fp32 sum.
  - The cp.async canary's expected buffer gets the right zero-fill (chunk 63 of 64) and the right reversal. NCH is 13824, which is 216·64.
  - The canary sizes match the TORCH_CHECKs.
- **221 KB smem.** cudaFuncSetAttribute(MaxDynamicSharedMemorySize, 221184) is set in lt_init (info_one) for moe_w13_silu_swapab, moe_w2_swapab and lt_canary_cpasync. That happens before any of them launches, and the attribute persists for kernel_fn's graph capture. The gate also checks maxDynamicSharedSizeBytes >= 221184 and logs the opt-in max (232448 on H100/H200). No GEMM kernel has static smem.
- **__hmul2 does not change the fault class.** On sm_90, cuda_bf16.hpp lowers bf16 __hmul2 to mul.bf16x2 (an HMUL2.BF16 instead of an HFMA2.BF16 with a -0 addend). At most that changes the SASS. Both forms are legal on sm_90, and neither is a plausible source of a 715. Results are unchanged: the products are exact and feed mma directly, so contraction cannot change them. Count it as eliminated only if the dequant canary runs and the fault shows up later.
- **TORCH_CUDA_ARCH_LIST.** cpp_extension returns no arch flags of its own once a cflag contains "arch", so 9.0a is never parsed. It is harmless, and it is restored in a finally block.
- **Rules.**
  - No child process: no subprocess, multiprocessing or os.system. load_inline is the same build mechanism master already uses.
  - Nothing derived from data is cached, because the self-check and canaries are synthetic.
  - kernel_fn only gained hook=None. It has no sync, no value-dependent allocation and no branch on values, so it is graph-capturable.
  - NVFP4 retiling is lossless and asserted. Activations stay BF16. Attention and KV are untouched.
- **Fallback.**
  - These all land on Marlin with ON/OFF and PATH CHOSEN printed: a build failure (caught in _lt_build), any Python or TORCH_CHECK or non-sticky launch error (caught in _lt_setup), and a failure at the arch, probe, canary, T=33 or self-check stage.
  - The only exception is a sticky 715, which cannot be recovered by nature. That is why this run must be an `okc check`.

**Stale base.** The check is still interpretable for its two jobs. The first job is fault attribution: the ladder is independent of the base. The second is the replaced kernels' time against Marlin: the in-run event-timing line and the profile's top_kernels compare both within the same run. The stale base does need a note, and the harvest must record it:

- This run's headline and per-shape speedups are against stage 1's base, not master (stack-align-par-sidestream-sliding1, 1.2521). Do not compare them to master's board number, and do not promote from them.
- The decode_*/verify_1 controls are flat relative to stage 1's base.
- MoE runs on the main stream here, not master's side stream, so overlap effects are absent.
- Promotion requires the restack onto master as a separate stage.

**Residual risk.** The underlying 715 was not root-caused. The likeliest outcome is that it faults again, with the last `launching <name>` line naming the kernel and form. A minor risk is that the ptxVersion==90 gate turns the path OFF if the driver reports a different value for a cubin-only sm_90a build. That would be logged loudly, and the arch probe is the stronger check.
