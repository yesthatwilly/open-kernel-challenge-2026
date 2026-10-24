---
problem: muse_bf16_proj
slug: bf16-lossless-exp4-weight-stream-rewrite
created: 2026-10-24T172152Z
base: master
area: bf16 GEMM dispatch (m<=32, lossless-compressed weight stream)
change: >
  Built on master (8b9ad43); no dependency on an unharvested hypothesis. WHY: every m<=32 bf16
  shape is a pure weight stream (FINDINGS: pooler 19.0-20.5 us kernel for 67.1 MB, qkv 11.8-12.3
  us for 37.7 MB, unpooler-full 18.5-18.9 us, o 4.5-5.2 us), and three generations of Triton/CUDA
  stream tuning top out at ~3.5-3.7 TB/s (LESSONS). The only lever left that moves these rows by
  20%+ is to read fewer bytes. The rules allow it in so many words (TASK.md: 'Repack them in
  prepare(), and compress them losslessly if you like'; rules page: 'You can repack the weights
  and compress them losslessly, but not change their precision or values'; 'Dequantizing inside
  your kernels is fine'). BF16 exponents are low-entropy: for the judge's N(0, 1/sqrt(K)) weights
  the 8-bit exponent field carries 2.54 bits (measured on a 512-row numpy simulation of
  reference.generate_weights, all three K), and real LLM weights behave the same (DFloat11). CODEC
  (lossless for ANY bf16 input; built in prepare() with torch ops on the GPU): split each weight
  row into groups of 64 consecutive K elements (K = 8192/1024/4096, all multiples of 64). Per
  group store e_ref = max exponent (uint8). Per element store one byte SM = sign|mantissa7 (bit
  layout chosen so one PRMT + one LOP3 rebuilds a bf16x2 pair with exponent e_ref) and one 4-bit
  code c: c in 0..14 means exponent = e_ref - c, c = 15 means exponent field 0 with mantissa 0
  (signed zero). A group containing any element outside that (exponent < e_ref - 14, subnormal,
  Inf/NaN) is an ESCAPE group: its header gets a flag bit and its 64 raw bf16 go to a side table
  (int32 slot index per group). Simulated: 0.56-0.66% escape groups, 12.23-12.34 bits/element,
  i.e. 0.765x the bytes: pooler/unpooler-full 67.1 -> 51.4 MB, qkv 37.7 -> 28.9 MB, o 16.8 -> 12.9
  MB, unpooler slice (rows 0-1023) 8.4 -> 6.4 MB. The SM bytes, nibbles and headers are REPACKED
  into mma.sync A-fragment order (Marlin-style: per 16-row x 64-K warp tile, each lane's 32
  elements are contiguous: two uint4 of SM bytes + one uint4 of nibbles + its 2 rows' e_ref), so
  every global load is a coalesced 16-byte vector. The original bf16 weight stays in state
  untouched (fallback path, and the unpooler slice / m>32 / cuBLAS paths keep using it). Nothing
  derived from inputs is cached; no per-call decompressed copy is ever written (the LESSONS rules
  ruling for NVFP4 applies by analogy: dequant in registers only). DEQUANT in registers: base pair
  = PRMT/LOP3 of two SM bytes with e_ref; scale pair = bf16x2 2^-c looked up from a 256-entry smem
  table indexed by the two nibbles (c=15 -> +0.0, which keeps the sign of the base); w =
  __hmul2(base, scale) (bf16 HMUL2 by a power of two is exact for every in-range code). Target <=
  3 integer ops + 1 LDS + 1 HMUL2 per element pair; verify in the SASS (print instruction counts
  from cuobjdump if available, else skip). An escape group (warp ballot on the header flag, rare)
  loads its raw bf16 from the side table instead. KERNEL (one CUDA translation unit,
  torch.utils.cpp_extension.load_inline at import, -gencode arch=compute_90a,code=sm_90a, one
  template instantiated per token count NT in {8, 16, 32}): swap-AB mma.sync.m16n8k16 bf16->fp32
  with the dequantized WEIGHT as A (16 weight rows per warp) and x^T as B (tokens on the n8 side,
  NT/8 fragments; tokens >= m zero-filled), x staged per K-stage in smem. Weight stream: 4-stage
  cp.async.cg (16 B) ring per CTA of >= 12 KB compressed per stage per warp-group (budget 227
  KB/block; set cudaFuncAttributeMaxDynamicSharedMemorySize explicitly), 4 warps per CTA each
  owning 16 rows -> 64 weight rows per CTA. Work items <= 132 CTAs, chosen statically per layer:
  pooler 64 row-tiles x split-K 2 = 128; unpooler-full 128 x 1; o 128 x 1; qkv 36 x 3 with K
  chunks of 2752/2752/2688 = 108 (not 36 x 4 = 144, which would be two waves); unpooler slice 16 x
  8 = 128. Split-K reduce uses master's proven ticket semaphore (fp32 partials of rows < m only
  into a prepare()-allocated workspace, __threadfence, one atomicAdd ticket per row-tile, the last
  CTA sums in fixed index order, stores bf16, resets its counter to 0), with its OWN workspace and
  counters, never master's Triton buffers. ROUTING: kernel_fn takes the new path for layer in
  {qkv_gate, o, pooler, unpooler} at 1 <= cfg m <= 32 (both the n == 1024 slice and the full
  unpooler), when state['lossless'][layer] is True and x is contiguous bf16 [m, k]; every other
  shape, the router, and any layer whose flag is False keeps master byte for byte. SELF-CHECK in
  prepare() (any failure prints '[lossless] <layer> DISABLED: <reason>' and sets the flag False, a
  constant fixed in prepare, never input-derived): (1) codec round trip: a decode-only debug
  kernel (or a torch reference decoder) reconstructs every weight and torch.equal(decoded, W) must
  hold bit-exactly; (2) compression ratio <= 0.80 of the raw bytes, else disable (held-out weights
  with a wide exponent spread fall back automatically); (3) GEMM check on synthetic x at m in {1,
  5, 7, 8, 16, 32} per layer, run twice (counters must return to 0): rel L2 <= 1e-3 vs fp64,
  one-hot power-of-two probe rows within 1e-5 (exact weights), finite. If load_inline fails, print
  '[lossless] load_inline failed' and keep master everywhere. DIAGNOSTICS printed once per layer
  (try/except, prints only): bits/element, escape-group fraction, compressed MB, compile seconds,
  regs/spills (cuobjdump or cudaFuncGetAttributes), and warm-L2 cudaEvent kernel time at m=1/16/32
  with effective compressed TB/s and raw-equivalent TB/s. STAGES: stage 1 = this (m <= 32, all
  four bf16 layers). Stage 2 = the same codec at m = 128/256 via wgmma with the dequantized weight
  as the register A operand (wgmma allows A from registers for bf16; tokens as B from smem), where
  pooler/unpooler m=128 are co-bound and 24% fewer bytes still pays. Stage 3 = the router: its
  hi/lo bf16 pair compressed with the same codec (lo's exponents are spread wider; measure the
  escape rate first). MILESTONES readable from the profile top_kernels (kernel name
  _lossless_swapab_kernel<NT>): M1 all tests pass and the prepare log shows round trip exact +
  ratio ~0.765 on every layer; M2 pooler m=16 kernel <= 17.5 us and qkv m=16 <= 11 us (both beat
  the incumbent even if the compressed stream runs no faster than master's 3.5 TB/s raw); M3
  pooler <= 15.5 us, unpooler-full <= 15.5 us, qkv <= 9.5 us, o <= 4.3 us (compressed stream at >=
  3.6 TB/s + ~1 us fixed cost).
expected:
  - shapes: layer=pooler load=* m=1
    direction: faster
    why: incumbent _bf16_splitk_kernel 19.0 us (row 19.0), floor 14.0 raw / 10.7 compressed (51.4 MB). Target M3 <= 15.5 us
  - shapes: layer=pooler load=* m=8
    direction: faster
    why: incumbent 19.3 us kernel (row 19.1), floor 14.0 raw / 10.7 compressed. Target <= 15.5 us
  - shapes: layer=pooler load=decode_16 m=16
    direction: faster
    why: incumbent 19.5 us kernel (row 19.8), floor 14.1 raw / 10.8 compressed. M2 <= 17.5, M3 <= 15.5 us
  - shapes: layer=pooler load=decode_32 m=32
    direction: faster
    why: incumbent _bf16_swapab_kernel 20.5 us (row 20.0), floor 14.1 raw / 10.8 compressed. Target <= 16 us
  - shapes: layer=unpooler load=decode_* m=16
    direction: faster
    why: incumbent nvjet 64x16 18.5 us (row 18.6), floor 14.1 raw / 10.8 compressed. Target <= 15.5 us
  - shapes: layer=unpooler load=decode_32 m=32
    direction: faster
    why: incumbent _bf16_swapab_kernel 18.9 us (row 18.5), floor 14.1 raw / 10.8 compressed. Target <= 16 us
  - shapes: layer=unpooler load=* m=1
    direction: faster
    why: "slice n=1024: incumbent _bf16_splitk_kernel 4.5 us (row 5.9), floor 1.7 raw / 1.3 compressed (6.4 MB). Latency-bound; target <= 3.8 us"
  - shapes: layer=unpooler load=* m=8
    direction: faster
    why: "slice: incumbent 4.6 us (row 6.1), floor 1.8 / 1.3. Target <= 3.8 us"
  - shapes: layer=qkv_gate load=* m=1
    direction: faster
    why: incumbent _qkv_splitk_sem_kernel 11.8 us (row 12.9), floor 7.9 raw / 6.0 compressed (28.9 MB). M3 <= 9.5 us
  - shapes: layer=qkv_gate load=* m=8
    direction: faster
    why: incumbent 12.0 us (row 13.2), floor 7.9 / 6.0. Target <= 9.5 us
  - shapes: layer=qkv_gate load=decode_16 m=16
    direction: faster
    why: incumbent 12.3 us (row 13.4), floor 7.9 / 6.0. M2 <= 11, M3 <= 9.5 us
  - shapes: layer=qkv_gate load=decode_32 m=32
    direction: faster
    why: incumbent _bf16_swapab_kernel 12.1 us (row 13.7), floor 8.0 / 6.1. Target <= 10 us
  - shapes: layer=o load=decode_* m=*
    direction: faster
    why: incumbent 4.5-4.8 us kernel (rows 6.3-6.7), floor 3.5-3.6 raw / 2.7 compressed (12.9 MB). Small row, mostly fixed cost; target <= 4.3 us
  - shapes: layer=o load=verify_1 m=8
    direction: faster
    why: incumbent _pooler_skinny_kernel 5.2 us (row 6.3), floor 3.5 / 2.7. Target <= 4.3 us
  - shapes: layer=* load=verify_16 m=128
    direction: flat
    why: m > 32 keeps master (stage 2 scope)
  - shapes: layer=* load=verify_32 m=256
    direction: flat
    why: m > 32 keeps master (stage 2 scope)
  - shapes: layer=router *
    direction: flat
    why: router untouched (stage 3 scope)
risk: high
depends_on: null
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: "m<=32 bf16 weight streams: pooler _bf16_splitk_kernel / _bf16_swapab_kernel, qkv_gate _qkv_splitk_sem_kernel / _bf16_swapab_kernel, o _pooler_skinny_kernel / _bf16_swapab_kernel, unpooler slice _bf16_splitk_kernel, unpooler-full nvjet_sm90_tst_64x16 / _bf16_swapab_kernel (decode_1, decode_16, decode_32, verify_1)"
review: proceed
status: running
review_at: 2026-10-24T172632Z
review_companions: null
stage: claimed
claimed_at: 2026-10-24T172707Z
base_sha: 43ae369dae50af5a98bad52c50a5e7ba904c492f
base_blob: a8bb42ac483318e9fbc84bc7826a27b7a704d1fb
branch: hyp/muse_bf16_proj/bf16-lossless-exp4-weight-stream-rewrite
worktree: .worktrees/muse_bf16_proj/bf16-lossless-exp4-weight-stream-rewrite
mode: null
---

# bf16-lossless-exp4-weight-stream-rewrite

## Change

Built on master (8b9ad43); no dependency on an unharvested hypothesis. WHY: every m<=32 bf16 shape is a pure weight stream (FINDINGS: pooler 19.0-20.5 us kernel for 67.1 MB, qkv 11.8-12.3 us for 37.7 MB, unpooler-full 18.5-18.9 us, o 4.5-5.2 us), and three generations of Triton/CUDA stream tuning top out at ~3.5-3.7 TB/s (LESSONS). The only lever left that moves these rows by 20%+ is to read fewer bytes. The rules allow it in so many words (TASK.md: 'Repack them in prepare(), and compress them losslessly if you like'; rules page: 'You can repack the weights and compress them losslessly, but not change their precision or values'; 'Dequantizing inside your kernels is fine'). BF16 exponents are low-entropy: for the judge's N(0, 1/sqrt(K)) weights the 8-bit exponent field carries 2.54 bits (measured on a 512-row numpy simulation of reference.generate_weights, all three K), and real LLM weights behave the same (DFloat11). CODEC (lossless for ANY bf16 input; built in prepare() with torch ops on the GPU): split each weight row into groups of 64 consecutive K elements (K = 8192/1024/4096, all multiples of 64). Per group store e_ref = max exponent (uint8). Per element store one byte SM = sign|mantissa7 (bit layout chosen so one PRMT + one LOP3 rebuilds a bf16x2 pair with exponent e_ref) and one 4-bit code c: c in 0..14 means exponent = e_ref - c, c = 15 means exponent field 0 with mantissa 0 (signed zero). A group containing any element outside that (exponent < e_ref - 14, subnormal, Inf/NaN) is an ESCAPE group: its header gets a flag bit and its 64 raw bf16 go to a side table (int32 slot index per group). Simulated: 0.56-0.66% escape groups, 12.23-12.34 bits/element, i.e. 0.765x the bytes: pooler/unpooler-full 67.1 -> 51.4 MB, qkv 37.7 -> 28.9 MB, o 16.8 -> 12.9 MB, unpooler slice (rows 0-1023) 8.4 -> 6.4 MB. The SM bytes, nibbles and headers are REPACKED into mma.sync A-fragment order (Marlin-style: per 16-row x 64-K warp tile, each lane's 32 elements are contiguous: two uint4 of SM bytes + one uint4 of nibbles + its 2 rows' e_ref), so every global load is a coalesced 16-byte vector. The original bf16 weight stays in state untouched (fallback path, and the unpooler slice / m>32 / cuBLAS paths keep using it). Nothing derived from inputs is cached; no per-call decompressed copy is ever written (the LESSONS rules ruling for NVFP4 applies by analogy: dequant in registers only). DEQUANT in registers: base pair = PRMT/LOP3 of two SM bytes with e_ref; scale pair = bf16x2 2^-c looked up from a 256-entry smem table indexed by the two nibbles (c=15 -> +0.0, which keeps the sign of the base); w = __hmul2(base, scale) (bf16 HMUL2 by a power of two is exact for every in-range code). Target <= 3 integer ops + 1 LDS + 1 HMUL2 per element pair; verify in the SASS (print instruction counts from cuobjdump if available, else skip). An escape group (warp ballot on the header flag, rare) loads its raw bf16 from the side table instead. KERNEL (one CUDA translation unit, torch.utils.cpp_extension.load_inline at import, -gencode arch=compute_90a,code=sm_90a, one template instantiated per token count NT in {8, 16, 32}): swap-AB mma.sync.m16n8k16 bf16->fp32 with the dequantized WEIGHT as A (16 weight rows per warp) and x^T as B (tokens on the n8 side, NT/8 fragments; tokens >= m zero-filled), x staged per K-stage in smem. Weight stream: 4-stage cp.async.cg (16 B) ring per CTA of >= 12 KB compressed per stage per warp-group (budget 227 KB/block; set cudaFuncAttributeMaxDynamicSharedMemorySize explicitly), 4 warps per CTA each owning 16 rows -> 64 weight rows per CTA. Work items <= 132 CTAs, chosen statically per layer: pooler 64 row-tiles x split-K 2 = 128; unpooler-full 128 x 1; o 128 x 1; qkv 36 x 3 with K chunks of 2752/2752/2688 = 108 (not 36 x 4 = 144, which would be two waves); unpooler slice 16 x 8 = 128. Split-K reduce uses master's proven ticket semaphore (fp32 partials of rows < m only into a prepare()-allocated workspace, __threadfence, one atomicAdd ticket per row-tile, the last CTA sums in fixed index order, stores bf16, resets its counter to 0), with its OWN workspace and counters, never master's Triton buffers. ROUTING: kernel_fn takes the new path for layer in {qkv_gate, o, pooler, unpooler} at 1 <= cfg m <= 32 (both the n == 1024 slice and the full unpooler), when state['lossless'][layer] is True and x is contiguous bf16 [m, k]; every other shape, the router, and any layer whose flag is False keeps master byte for byte. SELF-CHECK in prepare() (any failure prints '[lossless] <layer> DISABLED: <reason>' and sets the flag False, a constant fixed in prepare, never input-derived): (1) codec round trip: a decode-only debug kernel (or a torch reference decoder) reconstructs every weight and torch.equal(decoded, W) must hold bit-exactly; (2) compression ratio <= 0.80 of the raw bytes, else disable (held-out weights with a wide exponent spread fall back automatically); (3) GEMM check on synthetic x at m in {1, 5, 7, 8, 16, 32} per layer, run twice (counters must return to 0): rel L2 <= 1e-3 vs fp64, one-hot power-of-two probe rows within 1e-5 (exact weights), finite. If load_inline fails, print '[lossless] load_inline failed' and keep master everywhere. DIAGNOSTICS printed once per layer (try/except, prints only): bits/element, escape-group fraction, compressed MB, compile seconds, regs/spills (cuobjdump or cudaFuncGetAttributes), and warm-L2 cudaEvent kernel time at m=1/16/32 with effective compressed TB/s and raw-equivalent TB/s. STAGES: stage 1 = this (m <= 32, all four bf16 layers). Stage 2 = the same codec at m = 128/256 via wgmma with the dequantized weight as the register A operand (wgmma allows A from registers for bf16; tokens as B from smem), where pooler/unpooler m=128 are co-bound and 24% fewer bytes still pays. Stage 3 = the router: its hi/lo bf16 pair compressed with the same codec (lo's exponents are spread wider; measure the escape rate first). MILESTONES readable from the profile top_kernels (kernel name _lossless_swapab_kernel<NT>): M1 all tests pass and the prepare log shows round trip exact + ratio ~0.765 on every layer; M2 pooler m=16 kernel <= 17.5 us and qkv m=16 <= 11 us (both beat the incumbent even if the compressed stream runs no faster than master's 3.5 TB/s raw); M3 pooler <= 15.5 us, unpooler-full <= 15.5 us, qkv <= 9.5 us, o <= 4.3 us (compressed stream at >= 3.6 TB/s + ~1 us fixed cost).

## Expected

- `layer=pooler load=* m=1` faster: incumbent _bf16_splitk_kernel 19.0 us (row 19.0), floor 14.0 raw / 10.7 compressed (51.4 MB). Target M3 <= 15.5 us
- `layer=pooler load=* m=8` faster: incumbent 19.3 us kernel (row 19.1), floor 14.0 raw / 10.7 compressed. Target <= 15.5 us
- `layer=pooler load=decode_16 m=16` faster: incumbent 19.5 us kernel (row 19.8), floor 14.1 raw / 10.8 compressed. M2 <= 17.5, M3 <= 15.5 us
- `layer=pooler load=decode_32 m=32` faster: incumbent _bf16_swapab_kernel 20.5 us (row 20.0), floor 14.1 raw / 10.8 compressed. Target <= 16 us
- `layer=unpooler load=decode_* m=16` faster: incumbent nvjet 64x16 18.5 us (row 18.6), floor 14.1 raw / 10.8 compressed. Target <= 15.5 us
- `layer=unpooler load=decode_32 m=32` faster: incumbent _bf16_swapab_kernel 18.9 us (row 18.5), floor 14.1 raw / 10.8 compressed. Target <= 16 us
- `layer=unpooler load=* m=1` faster: slice n=1024: incumbent _bf16_splitk_kernel 4.5 us (row 5.9), floor 1.7 raw / 1.3 compressed (6.4 MB). Latency-bound; target <= 3.8 us
- `layer=unpooler load=* m=8` faster: slice: incumbent 4.6 us (row 6.1), floor 1.8 / 1.3. Target <= 3.8 us
- `layer=qkv_gate load=* m=1` faster: incumbent _qkv_splitk_sem_kernel 11.8 us (row 12.9), floor 7.9 raw / 6.0 compressed (28.9 MB). M3 <= 9.5 us
- `layer=qkv_gate load=* m=8` faster: incumbent 12.0 us (row 13.2), floor 7.9 / 6.0. Target <= 9.5 us
- `layer=qkv_gate load=decode_16 m=16` faster: incumbent 12.3 us (row 13.4), floor 7.9 / 6.0. M2 <= 11, M3 <= 9.5 us
- `layer=qkv_gate load=decode_32 m=32` faster: incumbent _bf16_swapab_kernel 12.1 us (row 13.7), floor 8.0 / 6.1. Target <= 10 us
- `layer=o load=decode_* m=*` faster: incumbent 4.5-4.8 us kernel (rows 6.3-6.7), floor 3.5-3.6 raw / 2.7 compressed (12.9 MB). Small row, mostly fixed cost; target <= 4.3 us
- `layer=o load=verify_1 m=8` faster: incumbent _pooler_skinny_kernel 5.2 us (row 6.3), floor 3.5 / 2.7. Target <= 4.3 us
- `layer=* load=verify_16 m=128` flat: m > 32 keeps master (stage 2 scope)
- `layer=* load=verify_32 m=256` flat: m > 32 keeps master (stage 2 scope)
- `layer=router *` flat: router untouched (stage 3 scope)

## Risk

Headline reasoning: at the M3 targets the m<=32 rows drop ~8.7 us on decode_1 (49.8 -> ~41), ~11 us on decode_16/32 (64.5/65.5 -> ~54) and ~8.7 us on verify_1 (50.6 -> ~42), about 1.17-1.21x on each of 4 of the 6 loads: roughly +10-13% on the geomean (1.196x -> ~1.33x). If the dequant makes the stream ALU-bound at ~3.0 TB/s compressed (3.9 TB/s raw-equivalent) the gain halves to ~+5-6%. This is the one lever here that can push m<=32 rows BELOW their raw-byte floors, which is how a 1.56x board leader is reachable on loads whose raw floors cap master at ~2.0x. BIGGEST RISK: dequant ALU cost. Hopper has 64 INT32 lanes/SM/clk; at 4 TB/s compressed that is ~2.6e12 elements/s, so the budget is ~3-4 integer ops per element before the integer pipe, not HBM, binds (Triton W4A16 arithmetic dequant was ALU-bound in moe, LESSONS). Hence the HMUL2-by-power-of-two design that moves half the work to the FP pipe, and the smem nibble-pair table; read the printed regs and the raw-equivalent TB/s first. RULES: explicitly allowed (lossless compression of BF16 weights; in-kernel decompression; no lower precision anywhere; output bit-identical math to a bf16 GEMM with fp32 accumulation). The prepare() round-trip torch.equal check, the escape path for ANY bf16 value and the ratio gate make it lossless and safe on held-out weights. Because it is a new class of technique for this repo, churnsubs should surface a one-line heads-up in PENDING_USER.md (like the cuBLASLt one) without blocking the run. Overlap: shares the m<=8 shapes with bf16-small-m-cuda-gemv-rewrite (raw-byte CUDA stream) and m<=16 with bf16-m16-swapab-wgmma-extend; the mechanism differs (fewer bytes, not a faster stream). If the gemv rewrite lands first, reuse its fragment-load and ticket-reduce skeleton and swap in the codec; per-shape rows decide the combo. Compile time counts toward the 10-minute run: keep one translation unit and three NT instantiations (attention's load_inline took 41 s); prepare's encode is a few vectorized torch ops per layer (<2 s). Risk high, so check runs only.

## Milestones

Rewrite stage 1 of `m<=32 bf16 weight streams: pooler _bf16_splitk_kernel / _bf16_swapab_kernel, qkv_gate _qkv_splitk_sem_kernel / _bf16_swapab_kernel, o _pooler_skinny_kernel / _bf16_swapab_kernel, unpooler slice _bf16_splitk_kernel, unpooler-full nvjet_sm90_tst_64x16 / _bf16_swapab_kernel (decode_1, decode_16, decode_32, verify_1)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-24T172632Z by an adversarial subagent (model opus).

**Verdict: proceed.** The rules explicitly allow lossless compression of BF16 weights in prepare() with dequant inside the kernel. The codec is sound and the 0.765x byte cut puts M2 (about 2.9-3.0 TB/s compressed) within reach of streams this repo already runs. Proceed with three fixes: no cuobjdump or other subprocess call, no per-group int32 slot index (it silently costs 0.5 bit/element), and add an escape-fraction gate next to the ratio gate.

### Justification
Mechanism: every m<=32 bf16 row in FINDINGS is a single weight-streaming kernel at 3.3-3.6 TB/s raw. Pooler _bf16_splitk_kernel runs 19.0-19.5 us for 67.1 MB (floor 14.0). qkv _qkv_splitk_sem_kernel runs 11.8-12.3 us for 37.7 MB (floor 7.9). unpooler-full nvjet/swapab runs 18.5-18.9 us (floor 14.1). o runs 4.5-5.2 us (floor 3.5). Because these rows are byte-bound, cutting bytes is the only lever that goes below the raw floors. The byte estimate holds up: 8 bits sign+mantissa, 4-bit code and 8 bits/64 of header is 12.125 bits/element, and 0.6% escape groups add 16 bits each, about 0.1 bit. That gives about 12.23 bits. The escape rate also checks out analytically: for N(0,s) groups of 64 with max near 2.5-3s, P(|w| < max*2^-14) is about 1.5e-4 per element, about 1% per group. The reference's generate_weights is exactly randn/sqrt(K) cast to bf16, so the simulation matches the judge's distribution. The milestone arithmetic is consistent with TASK.md's floors and weights: the compressed floors are 10.7/10.8 us for pooler and unpooler-full, 6.0 us for qkv and 2.7 us for o. M2 (pooler m=16 <= 17.5 us, qkv <= 11 us) only needs 2.9-3.4 TB/s compressed, below what master's Triton streams already reach raw. M3 needs about 3.6 TB/s compressed plus 1 us of fixed cost, which is the plausible ceiling. The ALU budget is less tight than the hypothesis fears. At 4 TB/s compressed (about 1.53 B/element) an SM must dequant about 10 elements/clk, which is 5 pairs/clk. The design spends 3 INT ops per pair, about 15 of 64 INT32 lanes/clk, plus 5 HMUL2 and 5 LDS lane-ops/clk. So the integer pipe is not the binding resource at 1 CTA/SM. The real dequant risk is (a) bank conflicts on the 256-entry nibble-pair LUT (the peaked code distribution means many lanes hit different entries in the same bank) and (b) issue and latency when only 4 warps per SM must cover both the cp.async producer work and the dequant+mma consumer work. Other design issues: (1) 'int32 slot index per group' adds 32/64 = 0.5 bit/element (about 4%, which takes the ratio to about 0.79), and that does not match the simulated 12.23-12.34 bits. Use e_ref=255 as the escape sentinel (Inf/NaN groups escape anyway, so 255 is otherwise unused) and store the int32 side-table slot inside the escape group's own unused 96 bytes of SM/nibble slots. That costs zero extra bytes and keeps the fragment layout fixed. (2) x is restaged per K-stage from L2 by every CTA. At pooler m=32 that is 128 CTAs x 4096 x 32 x 2 B = 32 MB of L2 reads beside 51 MB of HBM, and it must fit in the 227 KB stage budget next to the 4x12 KB weight ring. That is fine, but budget the smem with x included and keep x in an ldmatrix-friendly swizzle. (3) qkv runs 108 CTAs on 132 SMs (82% of the SMs). A 1-CTA/SM stream may not reach 3.6 TB/s from 108 SMs. 2 CTAs/SM (216 items, 36 x 6 split-K) with a smaller ring is the obvious next-stage knob. (4) mma.sync on Hopper runs at roughly 60-70% of wgmma peak. At m=32 that is about 3.3 us of tensor work for the pooler, hidden under an 11-15 us stream, so it is fine through m=32 but marginal for stage 2. Weakest link: whether one CTA of 4 warps per SM, mixing the cp.async ring, LUT dequant and mma.sync, sustains >= 3.6 TB/s of compressed bytes. Master's best raw streams top out at 3.5-3.7 TB/s with no dequant at all. If the compressed stream lands near 3.0 TB/s, M2 still passes but M3 and the 'below raw floor' claim do not. The printed warm-L2 compressed and raw-equivalent TB/s per layer is the right diagnostic to read first.

### Confounders on the current kernel
- none identified

### Interference and A/B needs
No hypothesis has landed on master's submission.py, so the list is empty and nothing can be reverted to isolate an interaction. The new path is self-contained: its own load_inline kernel, its own prepare()-allocated workspace and ticket counters, and routing gated on a prepare-time constant per layer. It replaces master's m<=32 incumbents wholesale rather than stacking on them. The original bf16 weights stay untouched for m>32, the router and fallbacks, so the verify_16/verify_32 and router rows should read flat, and any movement there is noise or a cold-L2 side effect. This is a kind: rewrite judged on the replaced kernels' own times in the profile's top_kernels (_lossless_swapab_kernel<NT>) against the incumbents and floors, so no companion is needed. The one sub-change a control could isolate (the codec versus a same-skeleton raw-bf16 CUDA stream) is already being run by the in-flight bf16-small-m-cuda-gemv-rewrite (m<=8, raw bytes, the same mma.sync swap-AB plus ticket skeleton). Read the two runs' per-shape kernel times side by side at harvest: gemv-rewrite's raw TB/s against this run's compressed and raw-equivalent TB/s attributes the gain to fewer bytes versus a better stream. A companion would duplicate that run.

### Open questions
- Escape-slot addressing: drop the per-group int32 index (0.5 bit/element). Is the e_ref=255 sentinel with the slot stored in the escape group's own unused SM/nibble bytes acceptable to the implementer? Confirm the printed bits/element includes every byte read in kernel_fn (headers, side table, LUT excluded).
- LUT bank conflicts: with the code distribution peaked at c=0..3, how many replays does the 256-entry bf16x2 table take per warp? If the SASS or timings show the LDS path binding, the fallback is an integer path: subtract (c<<7) per half via a PRMT-spread of the nibble pair, plus a zero mask.
- Is 1 CTA/SM with 4 warps enough to keep >= 36 KB of compressed weight in flight per SM while also doing dequant+mma, or should the consumer be split from a dedicated producer warp (warp specialization) in stage 2?
- qkv at 108 CTAs: does the 82% SM coverage cap bandwidth? Compare with 216 items at 2 CTAs/SM in the next stage.
- Will compile time (one TU, 3 NT instantiations, plus master's Triton warmups) stay well inside the 10-minute run? Record the compile seconds that are printed.
- If bf16-small-m-cuda-gemv-rewrite (running) lands first, rebase onto its skeleton as the hypothesis suggests, and decide per shape at m<=8 which wins.

