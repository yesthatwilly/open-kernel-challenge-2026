---
problem: muse_decode_layer
slug: dense-chain-stage2-attn-side-pdl
created: 2026-10-24T224424Z
base: master
area: qkv GEMM + o GEMM + attention-side small kernels (T<=32)
change: >
  Stage 2 of the landed small-t-launch-minimal-dense-chain-pdl rewrite (master f1f2f35, its stage
  plan item 2). Gate: tokens <= DL_MAX_TOKENS (config) and state["dl"] is not None and the new
  state["dl2_ok"] flag; else master byte for byte (T=128/256 control). Add to the existing
  _DL_CUDA TU (rename module muse_decode_dense_chain_v2) two kernels built from the
  dl_pooler_router_kernel<NF> body (same swap-AB mma.sync m16n8k16, 64 weight rows per CTA =
  exactly one head, 4-stage cp.async ring, fixed-order ticket split-K reduce, PDL launch via
  cudaLaunchKernelEx with cp.async of weight stages 0-2 BEFORE griddepcontrol.wait and everything
  that reads activations or writes global AFTER it). KERNEL Q dl_qkv_kernel<NF>: W = dense
  qkv_gate bf16 [2304, 8192] as is (36 head tiles x SPLIT 6 = 216 CTAs). The CTA that draws the
  last ticket of a tile owns the 64 fp32 outputs per token and does the epilogue by tile index:
  tiles 0-15 (q heads): per-head RMSNorm exactly as FusedQKNorm does (read
  sglang/kernels/ops/muse_spark_v12/fused_qk_norm.py for weight/eps, eps RMS_EPS), RoPE when cfg
  rope_theta (neox layout and cos_sin from state["cos_sin"] at data positions, exactly as
  FusedQKNorm applies it), cast to FP8 E4M3 and store to q_attn: unpaired [T,16,64], or when
  master paired predicate holds (window==0 and batch>1 and hkv%2==0) into the own half of the
  zero-padded [T,16,128] row (write 64 zeros to the other half); tiles 16-17 (k heads): same norm
  and RoPE, divide by k_scale (device scalar, only when cfg kv_scale != 1.0, passed as a flag),
  FP8 cast, write bytes to k_cache row cache_loc[t] head h; tiles 18-19 (v): divide by v_scale
  when scaled, FP8 cast, store at cache_loc; tiles 20-35 (gate): bf16 to gate [T,1024]. FA3 call
  unchanged (same q_attn/k_attn/v_attn/metadata logic). KERNEL O dl_o_kernel<NF>: W = dense o bf16
  [8192, 1024], 128 row tiles x SPLIT 2 (K halves of 512 = 8 heads) = 256 CTAs at 3/SM (one wave);
  prologue after griddepcontrol.wait loads FA3 out (bf16, unpaired [T,16,64] or paired [T,16,128]
  taking its own half) for its 8 heads, per-head scaleless RMSNorm eps RMS_EPS times sigmoid(gate)
  in fp32 exactly as FusedRMSNormGate, writes the bf16 x tile into the smem x ring (all K of the
  half fits: 32 x 512 x 2 B = 32 KB), then the weight GEMV; epilogue writes attn_output bf16
  [T,8192]. kernel_fn on the gate: qkv = None; Q replaces _gemm(attn_input, qkv_gate), qk_norm,
  the k/v divide, both .to(FP8), store_cache and the q cast / pad copies; O replaces the unpad
  copies, o_norm_gate and _gemm(o, dense o). attn_boundary, P, R, experts, unpooler, ffn_boundary
  unchanged. PREPARE self-check (synthetic, T in 1, 8, 9, 16, 32; sliding+rope and full; paired
  and unpaired; kv_scale 1 and 0.25): q bytes vs master path rel L2 <= 1e-2 (decode fp8 back to
  f32; do not require byte identity, LESSONS: Triton sin/cos), cache rows rel L2 <= 1e-2 at
  cache_loc and untouched elsewhere, attn_output rel L2 <= 5e-3, plus a CUDA-graph capture+replay
  of the whole gated chain; any failure sets dl2_ok False (master attention side, stage-1 FFN
  kept). Print [dense-chain-s2] ON|OFF pdl=.. max errs, and Q/O CUDA-event times pdl 0/1 for T=1
  and 32.
expected:
  - shapes: "*load=decode_1 *"
    direction: faster
    why: "incumbent 132.7/142.6 us: qkv nvjet 13.1 + o 7.0 + 2 splitKreduce ~3.4 + qk_norm/casts/store/q cast/norm-gate ~6 nodes ~10 us -> Q ~11 (37.7 MB at ~3.5 TB/s) + O ~5.5: ~12-15 us off (+9-11%)"
  - shapes: "*load=verify_1 *"
    direction: faster
    why: "same chain at T=8, 186.5/195.9 us: ~11-14 us off (+6-7%)"
  - shapes: "*layer=full load=decode_16 *"
    direction: faster
    why: "paired pad/unpad elementwise rows 6.5+4.5 us vanish plus qkv 12.7 us splitK and norm nodes: ~15-18 us off of 347.5 (+4-5%)"
  - shapes: "*layer=sliding load=decode_16 *"
    direction: faster
    why: ~10-12 us off 217 (+5%)
  - shapes: "*load=decode_32 *"
    direction: faster
    why: ~10-16 us off 247/535 (+2-6%). Headline ~+3-5% if Q/O stream >=3.4 TB/s
  - shapes: "*load=verify_16 *"
    direction: flat
    why: "T=128 outside gate: control"
  - shapes: "*load=verify_32 *"
    direction: flat
    why: "T=256 outside gate: control"
risk: high
depends_on: small-t-launch-minimal-dense-chain-pdl
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 2
replaces: "T<=32 attention side: qkv nvjet GEMM + splitKreduce, FusedQKNorm, k/v divide + FP8 casts, store_cache, q FP8 cast or paired pad copies, paired unpad copies, FusedRMSNormGate, o nvjet GEMM"
review: proceed
status: running
review_at: 2026-10-25T003616Z
review_companions: null
stage: implemented
claimed_at: 2026-10-25T003624Z
base_sha: 6a839f514cc7024729ac75ac28402e894091c0bd
base_blob: 4da94677234faaa06b42b5ced541db8961fd6f34
branch: hyp/muse_decode_layer/dense-chain-stage2-attn-side-pdl
worktree: .worktrees/muse_decode_layer/dense-chain-stage2-attn-side-pdl
mode: null
restack_of: null
stale_base: null
implemented_at: 2026-10-25T004858Z
impl_commit: 605685b75c53ea22b855f05ee2d1e4163f2c9b52
submission_sha256: 783ad43899b0d22545f0d6798b67695ef9946e8eb9c58bb884a0939ac0f3ea26
norm_sha256: c4f2d6012193c3d0e7de0a42d10bab87fe52833b3fd4f5684ed007e1c529729b
submission_blob: b4771cc5ac7c69bf030cba94a480f3ae96f5d06c
lint_warnings: 33
---

# dense-chain-stage2-attn-side-pdl

## Change

Stage 2 of the landed small-t-launch-minimal-dense-chain-pdl rewrite (master f1f2f35, its stage plan item 2). Gate: tokens <= DL_MAX_TOKENS (config) and state["dl"] is not None and the new state["dl2_ok"] flag; else master byte for byte (T=128/256 control). Add to the existing _DL_CUDA TU (rename module muse_decode_dense_chain_v2) two kernels built from the dl_pooler_router_kernel<NF> body (same swap-AB mma.sync m16n8k16, 64 weight rows per CTA = exactly one head, 4-stage cp.async ring, fixed-order ticket split-K reduce, PDL launch via cudaLaunchKernelEx with cp.async of weight stages 0-2 BEFORE griddepcontrol.wait and everything that reads activations or writes global AFTER it). KERNEL Q dl_qkv_kernel<NF>: W = dense qkv_gate bf16 [2304, 8192] as is (36 head tiles x SPLIT 6 = 216 CTAs). The CTA that draws the last ticket of a tile owns the 64 fp32 outputs per token and does the epilogue by tile index: tiles 0-15 (q heads): per-head RMSNorm exactly as FusedQKNorm does (read sglang/kernels/ops/muse_spark_v12/fused_qk_norm.py for weight/eps, eps RMS_EPS), RoPE when cfg rope_theta (neox layout and cos_sin from state["cos_sin"] at data positions, exactly as FusedQKNorm applies it), cast to FP8 E4M3 and store to q_attn: unpaired [T,16,64], or when master paired predicate holds (window==0 and batch>1 and hkv%2==0) into the own half of the zero-padded [T,16,128] row (write 64 zeros to the other half); tiles 16-17 (k heads): same norm and RoPE, divide by k_scale (device scalar, only when cfg kv_scale != 1.0, passed as a flag), FP8 cast, write bytes to k_cache row cache_loc[t] head h; tiles 18-19 (v): divide by v_scale when scaled, FP8 cast, store at cache_loc; tiles 20-35 (gate): bf16 to gate [T,1024]. FA3 call unchanged (same q_attn/k_attn/v_attn/metadata logic). KERNEL O dl_o_kernel<NF>: W = dense o bf16 [8192, 1024], 128 row tiles x SPLIT 2 (K halves of 512 = 8 heads) = 256 CTAs at 3/SM (one wave); prologue after griddepcontrol.wait loads FA3 out (bf16, unpaired [T,16,64] or paired [T,16,128] taking its own half) for its 8 heads, per-head scaleless RMSNorm eps RMS_EPS times sigmoid(gate) in fp32 exactly as FusedRMSNormGate, writes the bf16 x tile into the smem x ring (all K of the half fits: 32 x 512 x 2 B = 32 KB), then the weight GEMV; epilogue writes attn_output bf16 [T,8192]. kernel_fn on the gate: qkv = None; Q replaces _gemm(attn_input, qkv_gate), qk_norm, the k/v divide, both .to(FP8), store_cache and the q cast / pad copies; O replaces the unpad copies, o_norm_gate and _gemm(o, dense o). attn_boundary, P, R, experts, unpooler, ffn_boundary unchanged. PREPARE self-check (synthetic, T in 1, 8, 9, 16, 32; sliding+rope and full; paired and unpaired; kv_scale 1 and 0.25): q bytes vs master path rel L2 <= 1e-2 (decode fp8 back to f32; do not require byte identity, LESSONS: Triton sin/cos), cache rows rel L2 <= 1e-2 at cache_loc and untouched elsewhere, attn_output rel L2 <= 5e-3, plus a CUDA-graph capture+replay of the whole gated chain; any failure sets dl2_ok False (master attention side, stage-1 FFN kept). Print [dense-chain-s2] ON|OFF pdl=.. max errs, and Q/O CUDA-event times pdl 0/1 for T=1 and 32.

## Expected

- `*load=decode_1 *` faster: incumbent 132.7/142.6 us: qkv nvjet 13.1 + o 7.0 + 2 splitKreduce ~3.4 + qk_norm/casts/store/q cast/norm-gate ~6 nodes ~10 us -> Q ~11 (37.7 MB at ~3.5 TB/s) + O ~5.5: ~12-15 us off (+9-11%)
- `*load=verify_1 *` faster: same chain at T=8, 186.5/195.9 us: ~11-14 us off (+6-7%)
- `*layer=full load=decode_16 *` faster: paired pad/unpad elementwise rows 6.5+4.5 us vanish plus qkv 12.7 us splitK and norm nodes: ~15-18 us off of 347.5 (+4-5%)
- `*layer=sliding load=decode_16 *` faster: ~10-12 us off 217 (+5%)
- `*load=decode_32 *` faster: ~10-16 us off 247/535 (+2-6%). Headline ~+3-5% if Q/O stream >=3.4 TB/s
- `*load=verify_16 *` flat: T=128 outside gate: control
- `*load=verify_32 *` flat: T=256 outside gate: control

## Risk

High: Q epilogue must match FusedQKNorm (weights, RoPE layout) and the cache byte layout; a wrong RoPE fails tests, so the self-check gates it. LESSONS: small-kernel fusion alone has been flat, so the gain must come from the dense streams plus node count together; judge on replaced rows. Overlaps queued fused-preattention-paired-pad-restack and held prefork-qkv-o-swapab-t16-t32 (same qkv/o rows at T=16/32 with Triton); if either lands first, compare against master at claim. The running sliding-fused-batched-decode-port replaces the sliding batch>1 attention chain: if it lands first, keep its branch on sliding batch>1 and gate this to the remaining shapes.

## Milestones

Rewrite stage 2 of `T<=32 attention side: qkv nvjet GEMM + splitKreduce, FusedQKNorm, k/v divide + FP8 casts, store_cache, q FP8 cast or paired pad copies, paired unpad copies, FusedRMSNormGate, o nvjet GEMM`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T003616Z by an adversarial subagent (model opus).

**Verdict: proceed.** The design is sound and the target is physically reachable: 37.7 MB of qkv plus 16.8 MB of o weight, about 15 us at 3.6 TB/s against roughly 20-24 us of incumbent GEMM rows plus small nodes. But the spec predates the landed sliding batch-1 fused path (_sliding_forward), so at sliding decode_1 it must not touch the attention chain, and its expected gain there is overstated. Proceed with no companions. The builder must apply the sliding decode_1 carve-out, use its own token-bound constant, and add a kv_scale=1/64 self-check case.

### Justification
Q and O are swap-AB mma.sync split-K GEMVs from the dl_pooler_router_kernel body, which already ran on the judge (no faults) at 3.6 TB/s at T=1 and 2.85 TB/s at T=32. Q is 36 head tiles x SPLIT 6 = 216 CTAs, the same wave shape as P, so Q should land near 10.5-11 us at T<=8 against qkv nvjet 13.2 us plus splitKreduce. O is 16.8 MB, ~4.7 us plus a norm-gate prologue, against nvjet o ~7.0 us plus reduce ~1.7 us. GEMV-only saving ~4-6 us per T<=8 shape; node removal adds ~6-10 us; M3 (<=1.5x floor) reachable at T<=8. Weak links: (1) the expected rows predate stack-align-par-sidestream-sliding1 (master 8fa8fbf); at sliding decode_1 _sliding_forward reads raw [T,2304] qkv and returns normed-gated o, so only the qkv nvjet and the o GEMM are replaceable there: ~5-7 us off 127.5 us (+4-5%), not +9-11%. Incumbent decode_1 is now 127.5/142.9 us. (2) At T=16/32 Q inherits P's x L2 re-read (2.85 TB/s), so the decode_32 gain rests mostly on node removal, which LESSONS found flat alone, except the full decode_16/32 paired pad/unpad rows (~11 us). PDL buys ~0 here: no two of our kernels are adjacent; do not count PDL in the milestone read. verify_16/32 flat is the in-run control.

### Confounders on the current kernel
- stack-align-par-sidestream-sliding1 (master): _sliding_forward owns sliding decode_1 from qkv output to normed-gated o. On that shape the builder must either (a) run Q in raw mode (bf16 qkv [T,2304] for _sliding_forward) and O without the prologue, or (b) keep nvjet there; config-only, printed in the log.
- small-t-launch-minimal-dense-chain-pdl (parent, master): do not reuse DL_MAX_TOKENS; use DL2_MAX_TOKENS = 32 (LESSONS 2026-10-25: each path its own bound constant). A moved P row would be a cache/L2 or code-layout interaction.
- fa3-hdim128-head-pairing-batched (master): at full decode_16/32 Q writes zero-padded [T,16,128] paired q and O reads its half of FA3's paired output; report pad/unpad removal (~6.5 + 4.5 us) separately from Q/O stream rates.
- full-decode-splitkv-restack-on-pdl (in flight): if it lands before the claim, O's prologue would duplicate the splitkv combine's norm+gate at full decode_16/32; re-gate O and re-derive expected rows.

### Interference and A/B needs
No companion warranted. A rewrite judged on replaced-row time; the profile separates dl_qkv_kernel and dl_o_kernel from incumbent nvjet rows, and prepare prints Q/O CUDA-event times at pdl 0/1 for T=1 and T=32. Judge on the Q/O rows plus the count of vanished nodes, with verify_16/32 as the flat control. Marlin rows are noise.

### Open questions
- Sliding decode_1: Q raw mode + O without prologue feeding _sliding_forward (preferred), or keep nvjet?
- Does the nvjet_sm90_tst_64x16/64x32 row at decode_16/32 aggregate qkv + o + the unpooler slice?
- Will Q at T=32 hit P's 2.85 TB/s ceiling? If so stage 3 should share dl-p-wide-tile-t16-t32's BLOCK_N 128 tiling.
- Does FA3 here execute griddepcontrol.launch_dependents? Print O's pdl 0 vs 1 event times.


## Implementation notes

Edited only `muse_decode_layer/submission.py` in the worktree (not run, no GPU here). The CUDA module is renamed `muse_decode_dense_chain_v2` and gains `-lineinfo` (flags printed at build and in the stage-2 log; no wgmma/setmaxnreg, so plain `-arch=sm_90a` as stage 1 ran).

What changed:
- Two kernels added to the `_DL_CUDA` TU, copied from the P body (swap-AB `mma.sync m16n8k16`, 64 weight rows per CTA, cp.async.cg, ldmatrix.x4, PDL weight prefetch before `griddepcontrol.wait`, fixed-order ticket split-K, in-kernel counter reset).
  - Q `dl_qkv_kernel<NF>`: 36 head tiles x 6 K-splits = 216 CTAs. The last-ticket CTA per tile does the epilogue, one warp per token, lane = element pair, so a lane owns an interleaved RoPE pair. Tiles 0-15: scaleless RMSNorm (eps 1e-5), interleaved RoPE from `state["cos_sin"]` at `positions`, bf16 round, E4M3 into `q_attn`, unpaired [T,16,64] or paired [T,16,128] (own half plus 64 zero bytes, so no zeros fill). Tiles 16-17: same, then divide by k_scale when `kv_scale != 1`, bf16 round, E4M3 to the `cache_loc` row. Tiles 18-19: v, same divide, E4M3 store. Tiles 20-35: bf16 gate [T,1024]. mode 1 (raw) writes the bf16 [T,2304] projection.
  - O `dl_o_kernel<NF>`: 128 row tiles x 2 K-halves = 256 CTAs, all four 16 KB weight K-blocks of the CTA resident in smem (cp.async before the PDL wait). Prologue 1 builds the bf16 x tile in smem from FA3's output (unpaired, or paired taking the head's own half (h>>3)&1) and the gate: per-head scaleless RMSNorm x sigmoid(gate) in fp32 with one bf16 rounding. Prologue 0 (raw) copies the already normed-and-gated [T,1024] input. Epilogue writes attn_output bf16 [T,8192].
- `kernel_fn`: master's attention side moved verbatim into `_master_attention`. `kernel_fn` calls `_dl2_attention` (Q, then the unchanged FA3 call, then O) when `state["dl2_ok"]` and `state["dl"]` are set and `_dl2_applies(cfg)` (config only); otherwise `_master_attention`. Everything after the attention (boundary, P, R, experts, unpooler, ffn_boundary) is untouched.
- prepare(): `_dl2_setup` allocates the ws/counters, prints occupancy, runs the self-check (pdl as stage 1 chose, then 0) and the Q/O timing, and prints `[dense-chain-s2] ON|OFF pdl=... max errs ...` plus three `PATH CHOSEN` lines.

Adversarial-review fixes applied (the review wins over the spec):
- (a) Sliding decode_1 carve-out: `DL2_SLIDING1 = "raw"` runs Q in raw mode (bf16 qkv [T,2304]) into master's `_sliding_forward` and O without the prologue; `"nvjet"` keeps master's whole path there. Config-only, logged. Expected gain at that shape is therefore only the qkv and o GEMM rows, about 5-7 us (review).
- (b) New `DL2_MAX_TOKENS = 32`; `DL_MAX_TOKENS` is not used for gating (ws is sized with the new constant).
- (c) Self-check case batch 4, q_len 8, sliding, kv_scale 0.015625 (T=32).
- (d) Instruction forms limited to cp.async.cg, ldmatrix.x4, mma.sync m16n8k16 bf16; no createpolicy / cache_hint. FP8 encode is `__nv_cvt_float2_to_fp8x2(..., __NV_SATFINITE, __NV_E4M3)` on values already rounded through bf16 (including after the k/v divide); the satfinite note (clamps at 448 where torch's e4m3fn cast gives NaN) is logged. The self-check scales x down so `v / kv_scale` stays below 448.
- (e) Self-check prints before each launch and syncs and prints after it, separately for Q, Q-raw, O and O-raw and the graph chain; Q and O event times at pdl 0 and 1 for T=1 and T=32 are printed.
- (f) Gates are config only; no host sync in `kernel_fn`; counters (`cnt_q`, `cnt_o`) reset in-kernel and are checked back at 0 after the self-check; FA3 attends every key.

Deviations from the spec:
- RoPE is the interleaved (GPT-J) layout (pairs 2i, 2i+1), as reference.py and `_norm_rope` define it, not the neox layout the spec text says; cos/sin come from `state["cos_sin"]` (cos in columns 0-31, sin in 32-63).
- O keeps all four K-blocks of its weight slab resident (64 KB + up to 32 KB x tile, 2 CTAs/SM, 256 CTAs fit one wave) instead of the spec's 3 CTAs/SM ring; occupancy is printed.
- Q's gate output goes to a separate [T,1024] bf16 tensor that O reads, not a slice of a [T,2304] qkv.
- The graph check captures Q, O, Q-raw and O-raw with a stand-in FA3 output, not the real FA3 call (the synthetic inputs have no page table).
- FusedQKNorm / FusedRMSNormGate sources are not on this node, so their bf16 rounding points are inferred from master's tensor dtypes; the self-check compares against the live modules (q and cache rel L2 <= 1e-2, gate 1e-2, attention output <= 5e-3).

## Rules check

- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints in `kernel_fn`'s call graph (`_dl2_attention`, `_dl2_q`, `_dl2_o`, `_master_attention`); the only synchronizes are in prepare()-only helpers (lint reports them as outside kernel_fn). Allocation is by shape only; no value-dependent control flow (gates read `config`). The self-check captures and replays Q/O in a CUDA graph.
- [x] Nothing derived from `data` is cached across calls: ws/counters are prepare() buffers that hold no data (every launch leaves the counters at 0).
- [x] Weights: no new weight copies at all; Q and O read `state["dense"]["qkv_gate"]` and `["o"]` as BF16.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8 (E4M3 written through bf16); queries E4M3, no coarser.
- [x] Attention sees every key: FA3 call is unchanged (same q_attn / cache / metadata logic); the sliding decode_1 shape keeps `_sliding_forward`.
- [x] No timing patches, harness access, module edits, reference import, subprocesses or network (git diff grepped for subprocess, multiprocessing, os.fork: none).
- [x] All work happens inside the timed call; no output caching or replay.
- [x] Only instruction forms already proven on the judge (cp.async.cg, ldmatrix.x4, mma.sync m16n8k16 bf16); no createpolicy/cache_hint; no sm_90a-only PTX (no TORCH_CUDA_ARCH_LIST change needed); `-lineinfo` added.
- [x] n/a: bounded mbarrier/spin waits: the new kernels use no mbarrier or spin wait (cp.async groups, `__syncthreads`, a single atomic ticket).
- [x] Any build error, self-check miss or exception lands on master's attention side (stage-1 FFN kept) and never raises; loud `PATH CHOSEN` lines per shape class.

okc_lint output:

```
OK with warnings: muse_decode_layer/submission.py
(warnings only: host-sync lines outside kernel_fn's call graph in prepare(); dtype warnings for the self-check's synthetic uint8 -> FP8 cache init at _dl2_make_data)
```

`python3 -I -m py_compile` passed; `__pycache__` removed.

Uncertainties: never compiled or run (no nvcc here); a first-launch fault would poison the context, so the first run should be an `okc check`.

## Pre-flight audit

Audited 2026-10-24 by an opus subagent (read-only review plus a numpy emulation of the FP8 gate). **Verdict: dispatch-with-fixes.**

Verified: interleaved RoPE matches reference.py `_rope` and the `cos_sin` table (cos 0-31 | sin 32-63), the convention the judge-passed fused-preattention-triton also used. The cache row is `(cache_loc[t]*2 + kv_head)*64` into contiguous [pages,page,2,64] (valid for any page size). The k/v divide is on the bf16 value, rounded to bf16, then E4M3 RN with satfinite, as master's bf16 `k / k_scale` then `.to(FP8)`. The raw sliding decode_1 feed is bf16 [T,2304] contiguous, as nvjet produced, and O skips the prologue only there. Paired q puts head h in half (h>>3)&1 of [T,16,128] and zeroes the other half; O reads the same half. Ticket counters reset in-kernel, gates read config only, there are no host syncs in kernel_fn, FA3 is unchanged over every key, there is no child process, only cp.async.cg / ldmatrix.x4 / mma.sync are used, `-lineinfo` is on, and there is no ptxVersion gate.

Fixes (worktree submission.py; lint OK with warnings only, py_compile OK):
1. Self-check FP8 tolerance for q/k/v went from 1e-2 to 4e-2; the bf16 gate stays at 1e-2. Reason: a numpy emulation shows that one extra bf16 rounding inside FusedQKNorm before RoPE (its source is not on this node) alone gives about 1.05e-2 rel L2 from E4M3 boundary crossings. With 1e-2 the gate could fail on every case and stage 2 would never run. A wrong RoPE or layout gives O(1), and the judge's cache tolerance is 8%.
2. `_dl2_ref_q` now returns a clone of the qkv GEMM output taken before `state["qk_norm"]` runs. If FusedQKNorm normalizes in place, the Q-raw comparison would have compared against normed q/k and failed every case.

Residual risk: the build has never been compiled. A compile error now also takes stage 1 down to master, because the two stages share one TU. A device fault in the first self-check launch would poison the context, so the first run should be an okc check.
