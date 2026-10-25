# okc: problem=muse_moe_w4a16
"""Parity with SGLang's Marlin MoE path (hypothesis sglang-marlin-path-parity).

This is SGLang's NVFP4 expert path for Muse Spark Open on H200 at commit a55ab7d5
(MuseSpark1xFp4Experts._forward_marlin, after the pre-expert norm), composed exactly as
baseline.py composes it: prepare() repacks this GPU's 32 experts into Marlin's layout with
prepare_moe_nvfp4_layer_for_marlin (SGLang's own lossless repack, NVFP4 stays NVFP4), builds
the expert_map (-1 for remote experts), the Marlin workspace and the effective (1 + w) weight of
the post-expert norm; kernel_fn maps the global ids to local ids, picks block_size_m, runs
moe_align_block_size, moe_wna16_marlin_gemm for w13 (W4A16, FP32 reduce), silu_and_mul,
moe_wna16_marlin_gemm for w2 and the Triton post-expert-norm weighted sum. Activations stay
BF16 where SGLang keeps them BF16; nothing derived from the inputs is kept between calls.

The one deliberate difference from baseline.py: moe_align_block_size is called with the local
bank's expert count (32) instead of N_EXPERTS (256). The ids it receives are already mapped by
expert_map to [-1] U [0, 32), so 32 is the true expert count; SGLang's wrapper then sizes
sorted_ids and block_experts from 32 experts (M*8 + 32*(block_size_m-1) entries instead of
M*8 + 256*(block_size_m-1)) and its per-expert cumsum runs over 32 counters.

Up to 32 tokens (hypothesis triton-map-align-small-m-port), one single-program Triton kernel
(_align_small_kernel) does the expert_map gather and moe_align_block_size's work over the 32 local
experts (capacity M*8 + 32*(block_size_m - 1), as the 32-expert align allocates; remote routes get
no bucket). Above 32 tokens (hypothesis align-parallel-per-expert-verify) one launch of the
33-program _align_parallel_kernel does the same work: program e < 32 counts every expert and places
expert e's routes, padding and block_experts entries; the last program writes local_ids and the
tail past the used length.

Differs from SGLang: the expert all-reduce is not run."""

from typing import TypedDict

import types

import torch
import triton
import triton.language as tl
from sgl_kernel.scalar_type import scalar_types

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    fused_post_expert_norm_weighted_sum,
)
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin


class Weights(TypedDict):
    w13: torch.Tensor          # uint8 [256, 2I, H/2], gate rows [0, I) then up rows [I, 2I), packed E2M1
    w13_sf: torch.Tensor       # float8_e4m3fn [256, 2I, H/16]
    w13_gs: torch.Tensor       # float32 [256]
    w2: torch.Tensor           # uint8 [256, H, I/2]
    w2_sf: torch.Tensor        # float8_e4m3fn [256, H, I/16]
    w2_gs: torch.Tensor        # float32 [256]
    post_norm_w: torch.Tensor  # bfloat16 [H], the post-expert RMSNorm's weight (applied as 1 + w)


class Config(TypedDict):
    bs: int           # tokens, M
    d_hidden: int     # H = 4096, the width the MoE block runs its experts at
    d_expert: int     # I = 4096
    n_experts: int    # 256 experts in the model
    local_start: int  # this GPU computes experts [local_start, local_start + local_count)
    local_count: int  # 32: one GPU under 8-way expert parallelism
    top_k: int        # 8
    sharding: str     # "ep8"


class Inputs(TypedDict):
    hidden_states: torch.Tensor  # bfloat16 [M, H]
    topk_ids: torch.Tensor       # int32 [M, 8], global expert ids (0-255)
    topk_weights: torch.Tensor   # float32 [M, 8]
    config: Config


N_EXPERTS = 256
TOP_K = 8
# MuseSpark1xFp4Experts' candidates, smallest first.
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
POST_NORM_EPS = 1e-8
# Below this many tokens kernel_fn combines with _post_norm_combine_kernel instead of SGLang's
# fused_post_expert_norm_weighted_sum (hypothesis post-expert-norm-triton-small-m).
SMALL_M_POST_NORM = 32
# Up to this many tokens _align_small_kernel does the expert_map gather and the align in one
# launch (hypothesis triton-map-align-small-m-port); above it, _align_parallel_kernel does the same
# in one 33-program launch (hypothesis align-parallel-per-expert-verify).
ALIGN_FUSED_MAX_TOKENS = 32
# _align_parallel_kernel: routes per chunk (a [512, 32] one-hot at 8 warps is 64 int32 per thread
# per temporary), warps, and the block_experts store tile.
ALIGN_PAR_CHUNK = 512
ALIGN_PAR_WARPS = 8
ALIGN_PAR_BLK = 64


@triton.jit
def _post_norm_combine_kernel(y_ptr, ids_ptr, tw_ptr, w_ptr, out_ptr, H, eps,
                              TOP_K_C: tl.constexpr, BLOCK_H: tl.constexpr):
    """out[t] = sum over local slots s of g[t, s] * rmsnorm(y[t, s] * (1 + w)), fp32 math.

    One program per token. Remote slots (local id -1) are skipped and never loaded: their y rows
    are uninitialised memory. A token with no local slot stores zeros."""
    t = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H
    nw = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32) + 1.0
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for s in tl.static_range(TOP_K_C):
        eid = tl.load(ids_ptr + t * TOP_K_C + s)
        if eid >= 0:
            g = tl.load(tw_ptr + t * TOP_K_C + s)
            row = (t * TOP_K_C + s).to(tl.int64) * H
            v = tl.load(y_ptr + row + offs, mask=mask, other=0.0).to(tl.float32) * nw
            r = tl.rsqrt(tl.sum(v * v, axis=0) / H + eps)
            acc += g * v * r
    tl.store(out_ptr + t.to(tl.int64) * H + offs, acc.to(tl.bfloat16), mask=mask)


@triton.jit(do_not_specialize=["local_start", "local_count", "n", "cap", "n_blocks"])
def _align_small_kernel(topk_ids_ptr, local_ids_ptr, sorted_ids_ptr, block_experts_ptr, n_padded_ptr,
                        local_start, local_count, n, cap, n_blocks, BLOCK_N: tl.constexpr,
                        CAP_BLOCK: tl.constexpr, MB_BLOCK: tl.constexpr, BSM: tl.constexpr,
                        NUM_E: tl.constexpr):
    """One program: the expert_map gather plus moe_align_block_size over the bank's local experts.

    local_ids = id - local_start where 0 <= id - local_start < local_count, else -1 (the bank is
    the contiguous slice [local_start, local_start + local_count), so no wrap). sorted_ids holds the
    flat route indices grouped by local expert (ascending expert), each expert padded up to a
    multiple of BSM with the sentinel n; every unused slot is n. block_experts is each m-block's
    expert, -1 past the used blocks; n_padded the used length. Remote routes (-1) get no bucket.
    The three store sets into sorted_ids are disjoint, so no ordering is needed."""
    offs = tl.arange(0, BLOCK_N)
    mask = offs < n
    ids = tl.load(topk_ids_ptr + offs, mask=mask, other=0).to(tl.int32)
    local = ids - local_start
    local = tl.where(mask & (local >= 0) & (local < local_count), local, -1)
    tl.store(local_ids_ptr + offs, local, mask=mask)

    experts = tl.arange(0, NUM_E)
    onehot = (local[:, None] == experts[None, :]).to(tl.int32)        # [BLOCK_N, NUM_E]
    counts = tl.sum(onehot, axis=0)                                    # [NUM_E]
    padded = (counts + BSM - 1) // BSM * BSM
    seg_end = tl.cumsum(padded, axis=0)
    seg_start = seg_end - padded
    total = tl.sum(padded, axis=0)

    # Real routes: segment start plus rank among the earlier routes of the same expert.
    rank = tl.sum(onehot * (tl.cumsum(onehot, axis=0) - 1), axis=1)
    pos = tl.sum(onehot * seg_start[None, :], axis=1) + rank
    tl.store(sorted_ids_ptr + pos, offs, mask=mask & (local >= 0))
    # Each expert's padding slots.
    j = tl.arange(0, BSM)
    pad_ptrs = sorted_ids_ptr + (seg_start + counts)[:, None] + j[None, :]
    sentinel = tl.zeros([NUM_E, BSM], dtype=tl.int32) + n
    tl.store(pad_ptrs, sentinel, mask=j[None, :] < (padded - counts)[:, None])
    # Every slot past the used length.
    slots = tl.arange(0, CAP_BLOCK)
    tl.store(sorted_ids_ptr + slots, tl.zeros([CAP_BLOCK], dtype=tl.int32) + n,
             mask=(slots >= total) & (slots < cap))

    blocks = tl.arange(0, MB_BLOCK)
    block_start = blocks * BSM
    inseg = (block_start[:, None] >= seg_start[None, :]) & (block_start[:, None] < seg_end[None, :])
    block_expert = tl.sum(tl.where(inseg, experts[None, :], 0), axis=1)
    block_expert = tl.where(tl.sum(inseg.to(tl.int32), axis=1) > 0, block_expert, -1)
    tl.store(block_experts_ptr + blocks, block_expert, mask=blocks < n_blocks)
    tl.store(n_padded_ptr, total)


@triton.jit(do_not_specialize=["local_start", "local_count", "n", "cap", "n_blocks"])
def _align_parallel_kernel(topk_ids_ptr, local_ids_ptr, sorted_ids_ptr, block_experts_ptr, n_padded_ptr,
                           local_start, local_count, n, cap, n_blocks, CHUNK: tl.constexpr,
                           PAD_BLOCK: tl.constexpr, BLK: tl.constexpr, BSM: tl.constexpr,
                           NUM_E: tl.constexpr):
    """_align_small_kernel's outputs for any n, grid (local_count + 1,).

    Every program streams the n route ids in CHUNK chunks and counts all local experts, so each
    knows every segment start. Program e < local_count then places expert e's routes in ascending
    route order (segment start + routes placed by earlier chunks + rank in the chunk), its padding
    slots (sentinel n) and its block_experts entries. Program local_count writes local_ids, the
    sentinel into [total, cap), -1 into block_experts past the used blocks, and n_padded. The
    store sets are disjoint across programs, so no atomics or ordering are needed; no program
    reads what another writes (local ids are recomputed from topk_ids)."""
    pid = tl.program_id(0)
    offs0 = tl.arange(0, CHUNK)
    experts = tl.arange(0, NUM_E)

    counts = tl.zeros([NUM_E], dtype=tl.int32)
    for c in range(0, n, CHUNK):
        offs = c + offs0
        mask = offs < n
        ids = tl.load(topk_ids_ptr + offs, mask=mask, other=0).to(tl.int32)
        local = ids - local_start
        local = tl.where(mask & (local >= 0) & (local < local_count), local, -1)
        counts += tl.sum((local[:, None] == experts[None, :]).to(tl.int32), axis=0)
    padded = (counts + BSM - 1) // BSM * BSM
    seg_end = tl.cumsum(padded, axis=0)
    seg_start = seg_end - padded
    total = tl.sum(padded, axis=0)

    if pid < local_count:
        sel = experts == pid
        my_count = tl.sum(tl.where(sel, counts, 0), axis=0)
        my_padded = tl.sum(tl.where(sel, padded, 0), axis=0)
        my_start = tl.sum(tl.where(sel, seg_start, 0), axis=0)
        # This expert's routes, ascending route index.
        base = my_start
        for c in range(0, n, CHUNK):
            offs = c + offs0
            mask = offs < n
            ids = tl.load(topk_ids_ptr + offs, mask=mask, other=0).to(tl.int32)
            local = ids - local_start
            match = (mask & (local == pid)).to(tl.int32)
            rank = tl.cumsum(match, axis=0) - 1
            tl.store(sorted_ids_ptr + base + rank, offs, mask=match > 0)
            base += tl.sum(match, axis=0)
        # Its padding slots: fewer than BSM, over a power-of-2 tile (BSM 48 is not one).
        j = tl.arange(0, PAD_BLOCK)
        tl.store(sorted_ids_ptr + my_start + my_count + j, tl.zeros([PAD_BLOCK], dtype=tl.int32) + n,
                 mask=j < my_padded - my_count)
        # Its m-blocks [my_start / BSM, (my_start + my_padded) / BSM); a runtime loop, since one
        # expert can own up to cdiv(M, BSM) blocks.
        b0 = my_start // BSM
        nb = my_padded // BSM
        kb0 = tl.arange(0, BLK)
        for k in range(0, nb, BLK):
            kb = k + kb0
            tl.store(block_experts_ptr + b0 + kb, tl.zeros([BLK], dtype=tl.int32) + pid, mask=kb < nb)
    else:
        for c in range(0, n, CHUNK):
            offs = c + offs0
            mask = offs < n
            ids = tl.load(topk_ids_ptr + offs, mask=mask, other=0).to(tl.int32)
            local = ids - local_start
            local = tl.where((local >= 0) & (local < local_count), local, -1)
            tl.store(local_ids_ptr + offs, local, mask=mask)
        # Every slot past the used length.
        for s in range(0, cap, CHUNK):
            slots = s + offs0
            tl.store(sorted_ids_ptr + slots, tl.zeros([CHUNK], dtype=tl.int32) + n,
                     mask=(slots >= total) & (slots < cap))
        used_blocks = total // BSM
        kb0 = tl.arange(0, BLK)
        for k in range(0, n_blocks, BLK):
            kb = k + kb0
            tl.store(block_experts_ptr + kb, tl.zeros([BLK], dtype=tl.int32) - 1,
                     mask=(kb >= used_blocks) & (kb < n_blocks))
        tl.store(n_padded_ptr, total)


def _block_size_m(num_tokens, count):
    """MuseSpark1xFp4Experts' block_size_m choice (unchanged)."""
    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / count / block_size_m < 0.9:
            break
    return block_size_m


def _align_small(topk_ids, local_start, local_count, block_size_m):
    """local_ids [M, 8] int32 and Marlin's (sorted_ids, block_experts, n_padded) in one launch, for
    M <= ALIGN_FUSED_MAX_TOKENS. local_start/local_count are the bank's host ints from prepare().
    Sizes come from M, local_count and block_size_m only: capacity M*8 + count*(block_size_m - 1),
    as moe_align_block_size(..., count) allocates. topk_ids is cast to int32 inside the kernel."""
    topk_ids = topk_ids.contiguous()
    device = topk_ids.device
    n = topk_ids.numel()
    cap = n + local_count * (block_size_m - 1)
    n_blocks = triton.cdiv(cap, block_size_m)
    local_ids = torch.empty(topk_ids.shape, dtype=torch.int32, device=device)
    sorted_ids = torch.empty((cap,), dtype=torch.int32, device=device)
    block_experts = torch.empty((n_blocks,), dtype=torch.int32, device=device)
    n_padded = torch.empty((1,), dtype=torch.int32, device=device)
    _align_small_kernel[(1,)](
        topk_ids, local_ids, sorted_ids, block_experts, n_padded, local_start, local_count, n, cap,
        n_blocks, BLOCK_N=triton.next_power_of_2(n), CAP_BLOCK=triton.next_power_of_2(cap),
        MB_BLOCK=triton.next_power_of_2(n_blocks), BSM=block_size_m,
        NUM_E=triton.next_power_of_2(local_count), num_warps=4)
    return local_ids, (sorted_ids, block_experts, n_padded)


def _align_parallel(topk_ids, local_start, local_count, block_size_m):
    """_align_small's outputs (same sizes) for M > ALIGN_FUSED_MAX_TOKENS, in one launch of
    _align_parallel_kernel over local_count + 1 programs. n, cap and n_blocks are runtime
    arguments, so only block_size_m selects a compiled variant."""
    topk_ids = topk_ids.contiguous()
    device = topk_ids.device
    n = topk_ids.numel()
    cap = n + local_count * (block_size_m - 1)
    n_blocks = triton.cdiv(cap, block_size_m)
    local_ids = torch.empty(topk_ids.shape, dtype=torch.int32, device=device)
    sorted_ids = torch.empty((cap,), dtype=torch.int32, device=device)
    block_experts = torch.empty((n_blocks,), dtype=torch.int32, device=device)
    n_padded = torch.empty((1,), dtype=torch.int32, device=device)
    _align_parallel_kernel[(local_count + 1,)](
        topk_ids, local_ids, sorted_ids, block_experts, n_padded, local_start, local_count, n, cap,
        n_blocks, CHUNK=ALIGN_PAR_CHUNK, PAD_BLOCK=triton.next_power_of_2(block_size_m),
        BLK=ALIGN_PAR_BLK, BSM=block_size_m, NUM_E=triton.next_power_of_2(local_count),
        num_warps=ALIGN_PAR_WARPS)
    return local_ids, (sorted_ids, block_experts, n_padded)


def _align_self_check(device, count):
    """Untimed, prepare() only: _align_small (tokens <= 32) and _align_parallel (tokens > 32, with
    SGLang's block_size_m and, at >= 128 tokens, kernel_fn's two-smaller pick) against SGLang's
    moe_align_block_size(ref_local, bsm, count) on synthetic routings (fixed seeds), per local
    expert. Prints a one-line summary, any mismatch, and per case n_padded vs SGLang's own r_pad
    (info: whether SGLang's align buckets the -1 routes into m-blocks of its own); for the
    tokens > 32 cases only the pad lines that differ, and a separate summary line."""
    tag0 = "[triton-map-align-small-m-port]"
    gen = torch.Generator().manual_seed(20261024)
    bad = []
    pads = []
    cases = 0
    shapes = [(tokens, _block_size_m(tokens, count)) for tokens in (1, 8, 16, 32)]
    for tokens in (33, 57, 120, 128, 256, 257):
        bsm0 = _block_size_m(tokens, count)
        shapes.append((tokens, bsm0))
        if tokens >= 128:
            # kernel_fn's marlin-block-m-two-smaller-on-postnorm pick.
            bsm1 = BLOCK_SIZE_M_CANDIDATES[max(BLOCK_SIZE_M_CANDIDATES.index(bsm0) - 2, 0)]
            if bsm1 != bsm0:
                shapes.append((tokens, bsm1))
    for tokens, bsm in shapes:
        align = _align_small if tokens <= ALIGN_FUSED_MAX_TOKENS else _align_parallel
        for start in (0, 32, N_EXPERTS - count):
            local_set = list(range(start, start + count))
            remote_set = [e for e in range(N_EXPERTS) if e not in local_set]
            rows = {
                "random": [torch.randperm(N_EXPERTS, generator=gen)[:TOP_K].tolist() for _ in range(tokens)],
                "skewed": [[local_set[0], local_set[count - 1], local_set[5]] +
                           [remote_set[i] for i in torch.randperm(len(remote_set), generator=gen)[:5].tolist()]
                           for _ in range(tokens)],
                "all-remote": [[remote_set[i] for i in torch.randperm(len(remote_set), generator=gen)[:TOP_K].tolist()]
                               for _ in range(tokens)],
                "all-one-expert": [[local_set[3]] + remote_set[7 * t % 200: 7 * t % 200 + 7] for t in range(tokens)],
                "all-local": [local_set[8:16] for _ in range(tokens)],
            }
            for name, ids in rows.items():
                cases += 1
                tag = f"T={tokens} bsm={bsm} local_start={start} {name}"
                topk_ids = torch.tensor(ids, dtype=torch.int32, device=device)
                local_ids, (s_ids, b_exp, n_pad) = align(topk_ids, start, count, bsm)
                ref_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
                ref_map[start:start + count] = torch.arange(count, dtype=torch.int32, device=device)
                ref_local = ref_map[topk_ids].contiguous()
                if local_ids.dtype != torch.int32 or not local_ids.is_contiguous() \
                        or not torch.equal(local_ids, ref_local):
                    bad.append(f"{tag}: local_ids differ")
                    continue
                r_ids, r_exp, r_pad = moe_align_block_size(ref_local, bsm, count)
                torch.cuda.synchronize()
                numel = tokens * TOP_K
                if s_ids.numel() != r_ids.numel() or b_exp.numel() != r_exp.numel():
                    bad.append(f"{tag}: capacity {s_ids.numel()}/{b_exp.numel()} vs SGLang "
                               f"{r_ids.numel()}/{r_exp.numel()}")
                s_ids, b_exp, n_pad = s_ids.tolist(), b_exp.tolist(), int(n_pad.item())
                r_ids, r_exp, r_pad = r_ids.tolist(), r_exp.tolist(), int(r_pad.item())
                if tokens <= ALIGN_FUSED_MAX_TOKENS or n_pad != r_pad:
                    pads.append(f"{tag}: n_pad={n_pad} r_pad={r_pad}" + (" DIFF" if n_pad != r_pad else ""))

                def per_expert(sorted_ids, block_experts, used):
                    groups, blocks = {}, {}
                    for b in range(used // bsm):
                        e = block_experts[b]
                        if e < 0:
                            continue
                        blocks[e] = blocks.get(e, 0) + 1
                        groups.setdefault(e, []).extend(i for i in sorted_ids[b * bsm:(b + 1) * bsm] if i < numel)
                    return groups, blocks

                mine, mine_blocks = per_expert(s_ids, b_exp, n_pad)
                ref, ref_blocks = per_expert(r_ids, r_exp, r_pad)
                if {e: sorted(v) for e, v in mine.items()} != {e: sorted(v) for e, v in ref.items()}:
                    bad.append(f"{tag}: per-expert routes differ")
                if mine_blocks != ref_blocks or n_pad != bsm * sum(ref_blocks.values()):
                    bad.append(f"{tag}: blocks {mine_blocks} n_padded {n_pad} vs SGLang {ref_blocks} {r_pad}")
                used_blocks = n_pad // bsm
                if any(e < 0 for e in b_exp[:used_blocks]) or any(e != -1 for e in b_exp[used_blocks:]) \
                        or b_exp[:used_blocks] != sorted(b_exp[:used_blocks]):
                    bad.append(f"{tag}: block_experts layout {b_exp}")
                real = sum(len(v) for v in mine.values())
                if sum(1 for i in s_ids if i == numel) != len(s_ids) - real or any(i > numel for i in s_ids):
                    bad.append(f"{tag}: sorted_ids sentinel fill")
                if mine != ref:
                    bad.append(f"{tag}: (info) within-expert order differs from SGLang's")
    hard = [m for m in bad if "(info)" not in m]
    n_diff = sum(1 for p in pads if p.endswith("DIFF"))
    print(f"{tag0} self-check vs moe_align_block_size(ref_local, bsm, {count}): {cases} cases, "
          f"{len(hard)} mismatches, {len(bad) - len(hard)} order-only differences, "
          f"{n_diff} cases n_pad != SGLang r_pad")
    for message in bad[:20]:
        print(f"{tag0}   {message}")
    for line in pads:
        print(f"{tag0}   pad {line}")
    # The _align_parallel cases (tokens > 32), apart, so a real mismatch there is not hidden above.
    tag1 = "[align-parallel-per-expert-verify]"
    big_cases = sum(1 for tokens, _ in shapes if tokens > ALIGN_FUSED_MAX_TOKENS) * 3 * 5
    big = [m for m in hard if int(m.split()[0][2:]) > ALIGN_FUSED_MAX_TOKENS]
    big_real = [m for m in big if ": capacity " not in m]
    print(f"{tag1} self-check tokens > {ALIGN_FUSED_MAX_TOKENS}: {big_cases} cases, {len(big)} mismatches, "
          f"{len(big) - len(big_real)} capacity-only, {len(big_real)} other")
    for message in big_real[:10]:
        print(f"{tag1}   {message}")


def _marlin_layer(weights, start, count):
    """The parameters prepare_moe_nvfp4_layer_for_marlin reads, for experts [start, start+count)."""
    sl = slice(start, start + count)
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"][sl].contiguous(),
        w2_weight=weights["w2"][sl].contiguous(),
        w13_weight_scale=weights["w13_sf"][sl].contiguous(),
        w2_weight_scale=weights["w2_sf"][sl].contiguous(),
        # One global scale per expert for gate and up, as in SGLang's Marlin path.
        w13_weight_scale_2=weights["w13_gs"][sl].contiguous(),
        w2_weight_scale_2=weights["w2_gs"][sl].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    return layer


def prepare(weights: Weights) -> dict:
    """Called once, untimed. Repack this GPU's experts into Marlin's layout as SGLang does at load."""
    device = weights["w13"].device
    # One repacked bank: this GPU's 32 experts under EP8.
    banks = {}
    for start, count in ((0, 32),):
        layer = _marlin_layer(weights, start, count)
        expert_map = None
        if count != N_EXPERTS:
            expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
            expert_map[start:start + count] = torch.arange(count, dtype=torch.int32, device=device)
        banks[(start, count)] = {
            "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
            "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
            "expert_map": expert_map, "start": start, "count": count,
            "size_n": layer.intermediate_size_per_partition,
        }
        if expert_map is not None:
            # Compile _align_small_kernel for every M it serves (each BLOCK_N / CAP_BLOCK / MB_BLOCK /
            # block_size_m variant) with this bank's start/count, untimed, on dummy ids; then check
            # it against SGLang's op on synthetic routings. Nothing here sees any input.
            for tokens in range(1, ALIGN_FUSED_MAX_TOKENS + 1):
                _align_small(torch.full((tokens, TOP_K), start, dtype=torch.int32, device=device),
                             start, count, _block_size_m(tokens, count))
            # _align_parallel_kernel: one variant per block_size_m it can see above 32 tokens
            # (n is runtime), on dummy ids.
            for bsm in (16, 32, 48, 64):
                _align_parallel(torch.full((ALIGN_FUSED_MAX_TOKENS + 1, TOP_K), start, dtype=torch.int32,
                                           device=device), start, count, bsm)
            torch.cuda.synchronize()
            _align_self_check(device, count)
    torch.cuda.synchronize()
    state = {
        "banks": banks,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
        # MuseSpark1xFp4Experts binds the effective (1 + w) weight of the post-expert norm.
        "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16),
        # The original BF16 weight for _post_norm_combine_kernel, which adds the 1 in fp32.
        "post_norm_w": weights["post_norm_w"].contiguous(),
        "w2_mode": "master",
    }
    if (0, 32) in banks and banks[(0, 32)]["expert_map"] is not None:
        state["w2_mode"] = _w2_atomic_self_check(weights, state, device)
    return state


# Hypothesis w2-marlin-atomic-add-small-m: up to this many tokens the w2 Marlin call may use
# use_atomic_add=True (use_fp32_reduce stays True); prepare()'s self-check picks the mode.
W2_ATOMIC_MAX_TOKENS = 32


def _w2_atomic_self_check(weights, state, device):
    """Untimed, synthetic inputs only. Pick the w2 mode for num_tokens <= 32: "atomic" (atomic add,
    no memset), "atomic_zero" (atomic add after cache3.zero_()) or "master". cache3 aliases cache13,
    which holds w13's output when w2 runs, so a non-zeroing atomic path that needs a zeroed C fails
    the comparison here."""
    count = 32
    bank = state["banks"][(0, count)]
    mode, reason = "master", "no candidate passed"
    try:
        gen = torch.Generator(device=device)
        gen.manual_seed(1234)
        worst = {"atomic": 0.0, "atomic_zero": 0.0}
        ok = {"atomic": True, "atomic_zero": True}
        err = {}
        for tokens in (1, 4, 8, 16, 32):
            ids = torch.randint(0, N_EXPERTS, (tokens, TOP_K), generator=gen, device=device, dtype=torch.int32)
            ids[:, :3] = torch.randint(bank["start"], bank["start"] + count, (tokens, 3), generator=gen,
                                       device=device, dtype=torch.int32)
            data = {
                "hidden_states": torch.randn((tokens, 4096), generator=gen, device=device, dtype=torch.bfloat16),
                "topk_ids": ids,
                "topk_weights": torch.rand((tokens, TOP_K), generator=gen, device=device, dtype=torch.float32),
                "config": {"bs": tokens, "d_hidden": 4096, "d_expert": 4096, "n_experts": N_EXPERTS,
                           "local_start": bank["start"], "local_count": count, "top_k": TOP_K,
                           "sharding": "ep8"},
            }
            ref = _forward(data, state, "master").float()
            for cand in ("atomic", "atomic_zero"):
                if not ok[cand]:
                    continue
                try:
                    out = _forward(data, state, cand).float()
                    torch.cuda.synchronize()
                    rel = ((out - ref).norm() / ref.norm().clamp_min(1e-12)).item()
                    if not (torch.isfinite(out).all().item() and rel <= 5e-3):
                        ok[cand] = False
                        err[cand] = "T=%d rel-L2 %.3g" % (tokens, rel)
                    worst[cand] = max(worst[cand], rel)
                except Exception as e:  # binding rejected the combination
                    ok[cand] = False
                    err[cand] = "T=%d %s: %s" % (tokens, type(e).__name__, str(e)[:200])
        for cand in ("atomic", "atomic_zero"):
            if ok[cand]:
                mode, reason = cand, "worst rel-L2 vs master %.3g" % worst[cand]
                break
        else:
            reason = "; ".join("%s failed (%s)" % (k, v) for k, v in err.items())
    except Exception as e:
        mode, reason = "master", "self-check error %s: %s" % (type(e).__name__, str(e)[:200])
    print("[w2-atomic] PATH CHOSEN = %s (%s)" % ("master" if mode == "master" else "atomic", reason))
    print("[w2-atomic] atomic add enabled for w2 at T<=%d: %s; use_fp32_reduce=True; zeroing memset in path: %s"
          % (W2_ATOMIC_MAX_TOKENS, mode != "master", mode == "atomic_zero"))
    return mode


def kernel_fn(data: Inputs, state: dict) -> torch.Tensor:
    """This GPU's share of the MoE output, bfloat16 [M, H]."""
    return _forward(data, state, state.get("w2_mode", "master"))


def _forward(data: Inputs, state: dict, w2_mode: str) -> torch.Tensor:
    cfg = data["config"]
    bank = state["banks"][(cfg["local_start"], cfg["local_count"])]
    x = data["hidden_states"].contiguous()
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    expert_map = bank["expert_map"]

    topk_weights = data["topk_weights"].to(torch.float32).contiguous()
    block_size_m = _block_size_m(num_tokens, bank["count"])

    if num_tokens >= 128:
        # Hypothesis marlin-block-m-two-smaller-on-postnorm: two candidates below SGLang's pick
        # at the verify shapes (bs=128: 48 -> 16, bs=256/257: 64 -> 32). A host-side pick on the
        # shape only; num_tokens < 128 keeps SGLang's pick.
        block_size_m = BLOCK_SIZE_M_CANDIDATES[max(BLOCK_SIZE_M_CANDIDATES.index(block_size_m) - 2, 0)]
    if expert_map is not None:
        # One launch does the expert_map gather and the align, with the bank's prepare-time
        # start/count: up to 32 tokens _align_small (hypothesis triton-map-align-small-m-port), above
        # it _align_parallel (hypothesis align-parallel-per-expert-verify). Branch on shape only.
        align = _align_small if num_tokens <= ALIGN_FUSED_MAX_TOKENS else _align_parallel
        local_ids, (sorted_ids, block_experts, n_padded) = align(
            data["topk_ids"], bank["start"], bank["count"], block_size_m)
    else:
        local_ids = data["topk_ids"].to(torch.int32)
        if expert_map is not None:
            local_ids = expert_map[local_ids]
        local_ids = local_ids.contiguous()
        if num_tokens == 1 and expert_map is None:
            from sglang.kernels.ops.moe.moe_align_single_token import moe_align_single_token

            sorted_ids, block_experts, n_padded = moe_align_single_token(local_ids, block_size_m)
        else:
            # The one change from baseline.py: the ids are already local ([-1] U [0, count)), so
            # align over the bank's count (32) rather than N_EXPERTS (256). SGLang's wrapper sizes
            # sorted_ids and block_experts from this count.
            sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, bank["count"])

    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=expert_map is not None,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = state["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    w2_common = common
    if w2_mode != "master" and num_tokens <= W2_ATOMIC_MAX_TOKENS:
        w2_common = dict(common, use_atomic_add=True)
        if w2_mode == "atomic_zero":
            cache3.zero_()
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **w2_common).view(num_tokens, TOP_K, size_k)
    if num_tokens < SMALL_M_POST_NORM:
        y = y.contiguous()
        out = torch.empty((num_tokens, size_k), device=x.device, dtype=torch.bfloat16)
        _post_norm_combine_kernel[(num_tokens,)](
            y, local_ids, topk_weights, state["post_norm_w"], out, size_k, POST_NORM_EPS,
            TOP_K_C=TOP_K, BLOCK_H=triton.next_power_of_2(size_k), num_warps=8)
        return out
    return fused_post_expert_norm_weighted_sum(
        y, topk_weights, norm_weight=state["norm_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)
