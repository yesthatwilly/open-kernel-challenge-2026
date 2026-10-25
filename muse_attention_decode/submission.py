# okc: problem=muse_attention_decode
"""Gated attention decode with the pre-attention chain fused into one Triton kernel.

SGLang's path launches, before FlashAttention 3: the fused QK-norm (+RoPE), a k cast to FP8,
a v cast to FP8, the cache store and a q cast to FP8. On sliding-window layers at decode the
attention itself moves under a megabyte, so those launches are most of the time. This kernel
does all five in one launch: per (token, head) it RMS-normalizes q or k, applies interleaved
RoPE when the layer has it, and writes q as FP8 for FA3 and the new k and v rows as FP8 into
the cache (divided by the cache scales). FA3 and the output norm x sigmoid(gate) are SGLang's, except on full-layer decode at
1 < batch <= 64 (tp8, page 1), where one CUDA kernel (stream-K split-KV gather attention with
an in-kernel ticket combine and epilogue, stage 2; see below) replaces FA3 and the epilogue, with
stage 1's two kernels kept as its fallback.
"""

import os
import random
import time
from typing import TypedDict

import torch
import triton
import triton.language as tl
from sgl_kernel.flash_attn import flash_attn_with_kvcache

from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate


class Config(TypedDict):
    batch: int            # B
    q_len: int            # L: 1 for decode, 8 for a speculative verify block (causal inside it)
    num_q_heads: int      # Hq: 16 per GPU under TP8
    num_kv_heads: int     # Hkv: 2; each serves Hq / Hkv query heads
    head_dim: int         # 64
    page_size: int        # 1 in the scored cases (as SGLang serves Muse on H200), 64 in some tests
    window: int           # 2048 on sliding-window layers (a token and the 2,047 before it); 0 = all
    rope_theta: float     # 500000 on sliding-window layers; 0 = no RoPE
    qk_scale_factor: float
    sm_scale: float       # 1/8
    kv_scale: float       # the cache scales' value: 1 in the scored cases (SGLang leaves them unset)


class Inputs(TypedDict):
    qkv: torch.Tensor          # bfloat16 [B, L, (2*Hq + 2*Hkv)*64], 2304 under tp8: per token q (Hq*64),
                               # then the block's keys before norm and RoPE (Hkv*64), values
                               # (Hkv*64) and the output gate's logits (Hq*64)
    positions: torch.Tensor    # int64 [B*L], each new token's position (seq_lens[b] + i)
    cache_loc: torch.Tensor    # int64 [B*L], each new token's flat cache row (page * page_size + offset)
    cache_seqlens: torch.Tensor  # int32 [B], seq_lens + L: what each request attends over
    cu_seqlens_q: torch.Tensor   # int32 [B+1], 0, L, 2L, ...: the block's query offsets
    scheduler_metadata: torch.Tensor | None  # int32, FA3's tile schedule (get_scheduler_metadata) on
                                             # full-attention decode steps, else None
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, 64], written in place
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, 64], written in place
    k_scale: torch.Tensor     # float32 [], key = e4m3 * k_scale
    v_scale: torch.Tensor     # float32 [], value = e4m3 * v_scale
    page_table: torch.Tensor  # int32 [B, max_pages], physical page of each logical page (-1 unused)
    seq_lens: torch.Tensor    # int32 [B], cached tokens per request; the block starts at this position
    config: Config


RMS_EPS = 1e-5
FP8 = torch.float8_e4m3fn
_INV_FREQ = {}


def _inv_freq(theta: float, d: int, device) -> torch.Tensor:
    """[d/2] float32 theta^(-2i/d), built exactly as the reference builds it; cached per theta."""
    key = (theta, d, str(device))
    cache = _INV_FREQ.get(key)
    if cache is None:
        cache = _INV_FREQ[key] = 1.0 / (theta ** (torch.arange(0, d, 2, device=device).float() / d))
    return cache


@triton.jit
def _pre_attention_kernel(
    qkv_ptr, pos_ptr, loc_ptr, inv_freq_ptr, k_scale_ptr, v_scale_ptr,
    q_out_ptr, k_cache_ptr, v_cache_ptr,
    qkv_stride,
    HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, ROPE: tl.constexpr, EPS: tl.constexpr,
    PAIRED: tl.constexpr, G: tl.constexpr,
):
    """One program per (token, head). Heads 0..HQ-1 are queries; HQ..HQ+HKV-1 are the new keys,
    whose program also stores the matching value head. The q and k heads are contiguous in the
    fused row, so head h starts at h*D for both. When PAIRED, q_out is the zero-padded
    [tokens, HQ, 2*D] FA3 query: query h writes its D values into the half (h // G) % 2 (its KV
    head's half of the paired 128-byte row) and D FP8 zeros into the other half, so the program
    owns its whole padded row and every byte of it is written on every call."""
    t = tl.program_id(0)
    h = tl.program_id(1)
    HALF: tl.constexpr = D // 2
    i = tl.arange(0, HALF)
    even = 2 * i
    odd = 2 * i + 1

    row = qkv_ptr + t.to(tl.int64) * qkv_stride
    base = row + h * D
    xe = tl.load(base + even).to(tl.float32)
    xo = tl.load(base + odd).to(tl.float32)

    # Scaleless RMSNorm over the head's D entries.
    ss = tl.sum(xe * xe, axis=0) + tl.sum(xo * xo, axis=0)
    r = 1.0 / tl.sqrt(ss / D + EPS)
    xe = xe * r
    xo = xo * r

    if ROPE:
        # Interleaved (GPT-J) RoPE: pair (2i, 2i+1) rotates by pos * theta^(-2i/d).
        p = tl.load(pos_ptr + t).to(tl.float32)
        ang = p * tl.load(inv_freq_ptr + i)
        c = tl.cos(ang)
        s = tl.sin(ang)
        ye = xe * c - xo * s
        yo = xe * s + xo * c
        xe = ye
        xo = yo

    if h < HQ:
        # Query, rounded to FP8 E4M3 for FA3 (the 5.473 factor is folded into the softmax scale).
        if PAIRED:
            half = (h // G) % 2
            qrow = q_out_ptr + (t.to(tl.int64) * HQ + h) * (2 * D)
            out = qrow + half * D
            zj = tl.arange(0, D)
            tl.store(qrow + (1 - half) * D + zj, tl.zeros([D], dtype=tl.float32).to(tl.float8e4nv))
        else:
            out = q_out_ptr + (t.to(tl.int64) * HQ + h) * D
        tl.store(out + even, xe.to(tl.float8e4nv))
        tl.store(out + odd, xo.to(tl.float8e4nv))
    else:
        kh = h - HQ
        loc = tl.load(loc_ptr + t).to(tl.int64)
        k_scale = tl.load(k_scale_ptr)
        v_scale = tl.load(v_scale_ptr)
        kout = k_cache_ptr + (loc * HKV + kh) * D
        tl.store(kout + even, (xe / k_scale).to(tl.float8e4nv))
        tl.store(kout + odd, (xo / k_scale).to(tl.float8e4nv))
        # The value head: no norm, no RoPE, just the scale and the FP8 cast.
        j = tl.arange(0, D)
        v = tl.load(row + (HQ + HKV + kh) * D + j).to(tl.float32)
        vout = v_cache_ptr + (loc * HKV + kh) * D
        tl.store(vout + j, (v / v_scale).to(tl.float8e4nv))


LOG2E = 1.4426950408889634
_COUNTERS = {}


def _counters(batch: int, hkv: int, device) -> torch.Tensor:
    """int32 [B*Hkv] ticket counters for the in-kernel split combine. Zeroed once per
    (B, Hkv, device) on the first (eager) call; every launch leaves them at 0 again (the
    program that draws the last ticket resets its counter), so they hold no data."""
    key = (batch, hkv, str(device))
    buf = _COUNTERS.get(key)
    if buf is None:
        buf = _COUNTERS[key] = torch.zeros(batch * hkv, dtype=torch.int32, device=device)
    return buf


def _num_splits(batch: int, hkv: int) -> int:
    """Window splits per (request, kv head), from config only: the power of two that brings
    the grid to ~128 CTAs, capped at 16 (B=1: 16, B=16: 4, B=32: 2, B>=64: 1)."""
    s = 1
    while s < 16 and batch * hkv * s * 2 <= 128:
        s *= 2
    return s


@triton.jit
def _norm_rope(x, xs, pos, freq, d, D: tl.constexpr, ROPE: tl.constexpr, EPS: tl.constexpr):
    """Rows of x [R, D] (xs: the same rows with each pair swapped, xs[:, d] = x[:, d ^ 1]):
    scaleless RMSNorm, then interleaved RoPE at pos [R] with freq [D] = inv_freq[d // 2].
    Even d: x_e cos - x_o sin; odd d: x_o cos + x_e sin, as _pre_attention_kernel."""
    r = 1.0 / tl.sqrt(tl.sum(x * x, axis=1) / D + EPS)
    x = x * r[:, None]
    if ROPE:
        xs = xs * r[:, None]
        ang = pos[:, None] * freq[None, :]
        sign = tl.where(d % 2 == 0, -1.0, 1.0)
        x = x * tl.cos(ang) + sign[None, :] * xs * tl.sin(ang)
    return x


@triton.jit
def _gpu_fence(x):
    """fence.acq_rel.gpu executed by every thread holding an element of x (int32)."""
    return tl.inline_asm_elementwise("fence.acq_rel.gpu;\n\tmov.b32 $0, $1;", "=r,r", [x],
                                     dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _online_update(qk, valid, v, m_i, l_i, acc):
    """One key block of the online softmax (base-2 logits qk [M, N], values v [N, D] bf16).
    -inf safe: a row with no valid key so far keeps m = -inf, l = 0, acc = 0, never NaN."""
    qk = tl.where(valid, qk, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(qk, axis=1))
    m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
    alpha = tl.exp2(m_i - m_use)
    p = tl.exp2(qk - m_use[:, None])
    l_i = l_i * alpha + tl.sum(p, axis=1)
    acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
    return m_new, l_i, acc


@triton.jit
def _gate_epilogue(o, qkv_ptr, out_ptr, tok, head, rvalid, d, qkv_stride,
                   HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, EPS: tl.constexpr):
    """out[tok, head] = rmsnorm(o) * sigmoid(gate), o [M, D] fp32 (no bf16 rounding before)."""
    r = 1.0 / tl.sqrt(tl.sum(o * o, axis=1) / D + EPS)
    gptr = qkv_ptr + tok.to(tl.int64)[:, None] * qkv_stride + (HQ + 2 * HKV) * D + head[:, None] * D + d[None, :]
    gate = tl.load(gptr, mask=rvalid[:, None], other=0.0).to(tl.float32)
    y = o * r[:, None] * tl.sigmoid(gate)
    optr = out_ptr + (tok.to(tl.int64)[:, None] * HQ + head[:, None]) * D + d[None, :]
    tl.store(optr, y.to(tl.bfloat16), mask=rvalid[:, None])


@triton.jit
def _sliding_fused_kernel(
    qkv_ptr, pos_ptr, loc_ptr, seq_ptr, pt_ptr, inv_freq_ptr, k_scale_ptr, v_scale_ptr,
    k_cache_ptr, v_cache_ptr, ws_acc_ptr, ws_ml_ptr, cnt_ptr, out_ptr,
    qkv_stride, pt_stride, q_scale, score_scale,
    HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, L: tl.constexpr, M: tl.constexpr,
    LN: tl.constexpr, WINDOW: tl.constexpr, PAGE: tl.constexpr, S: tl.constexpr,
    BLOCK_N: tl.constexpr, ROPE: tl.constexpr, EPS: tl.constexpr,
):
    """Sliding-window layer, one launch: pre-attention, attention, output norm x gate.
    Grid (B, HKV, S). Program (b, g, s) serves kv head g's GROUP query heads for the block's L
    tokens as one M-row tile (row r = j * GROUP + hh; rows past GROUP * L are padding), over
    chunk s of the old keys [lo, seq - 1], lo = max(0, seq - (WINDOW - 1)). Program S - 1 also
    builds the L new keys/values, stores them (E4M3) at cache_loc and attends them from
    registers (causal i <= j). Splits combine in-kernel through a ticket counter."""
    b = tl.program_id(0)
    g = tl.program_id(1)
    s = tl.program_id(2)
    GROUP: tl.constexpr = HQ // HKV
    d = tl.arange(0, D)
    dsw = d ^ 1
    freq = tl.load(inv_freq_ptr + d // 2) if ROPE else tl.zeros([D], dtype=tl.float32)

    # Query rows: token j = r // GROUP (clamped for padding rows, which stay finite), head h.
    r = tl.arange(0, M)
    rvalid = r < GROUP * L
    j = tl.minimum(r // GROUP, L - 1)
    head = g * GROUP + r % GROUP
    tok = b * L + j
    qrow = qkv_ptr + tok.to(tl.int64)[:, None] * qkv_stride + head[:, None] * D
    xq = tl.load(qrow + d[None, :], mask=rvalid[:, None], other=0.0).to(tl.float32)
    xqs = tl.load(qrow + dsw[None, :], mask=rvalid[:, None], other=0.0).to(tl.float32)
    qpos = tl.load(pos_ptr + tok).to(tl.float32)
    q = _norm_rope(xq, xqs, qpos, freq, d, D, ROPE, EPS)
    # bf16 operands (finer than FA3's E4M3), rounded after qk_scale_factor as the reference.
    q = (q * q_scale).to(tl.bfloat16)

    k_scale = tl.load(k_scale_ptr)
    v_scale = tl.load(v_scale_ptr)
    qk_scale = score_scale * k_scale  # sm_scale * log2(e) * k_scale
    seq = tl.load(seq_ptr + b)
    p_row = seq + j  # each row's token position

    m_i = tl.full([M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([M], dtype=tl.float32)
    acc = tl.zeros([M, D], dtype=tl.float32)

    if s == S - 1:
        # The block's new keys/values for head g: never read back from the cache.
        i = tl.arange(0, LN)
        ivalid = i < L
        ntok = b * L + tl.minimum(i, L - 1)
        nrow = qkv_ptr + ntok.to(tl.int64)[:, None] * qkv_stride
        xk = tl.load(nrow + (HQ + g) * D + d[None, :], mask=ivalid[:, None], other=0.0).to(tl.float32)
        xks = tl.load(nrow + (HQ + g) * D + dsw[None, :], mask=ivalid[:, None], other=0.0).to(tl.float32)
        kpos = tl.load(pos_ptr + ntok).to(tl.float32)
        kn = _norm_rope(xk, xks, kpos, freq, d, D, ROPE, EPS)
        k8 = (kn / k_scale).to(tl.float8e4nv)
        vn = tl.load(nrow + (HQ + HKV + g) * D + d[None, :], mask=ivalid[:, None], other=0.0).to(tl.float32)
        v8 = (vn / v_scale).to(tl.float8e4nv)
        loc = tl.load(loc_ptr + ntok).to(tl.int64)
        cptr = (loc[:, None] * HKV + g) * D + d[None, :]
        tl.store(k_cache_ptr + cptr, k8.to(tl.uint8, bitcast=True), mask=ivalid[:, None])
        tl.store(v_cache_ptr + cptr, v8.to(tl.uint8, bitcast=True), mask=ivalid[:, None])
        # Attend the stored E4M3 values (dequantized via the scales), causal inside the block.
        qk = tl.dot(q, tl.trans(k8.to(tl.bfloat16))) * qk_scale
        valid = ivalid[None, :] & (i[None, :] <= j[:, None])
        m_i, l_i, acc = _online_update(qk, valid, v8.to(tl.bfloat16), m_i, l_i, acc)

    # Old keys: chunk s of [lo, seq - 1]; every key of every row's window is scored.
    lo = tl.maximum(seq - (WINDOW - 1), 0)
    count = seq - lo
    chunk = ((count + S - 1) // S + BLOCK_N - 1) // BLOCK_N * BLOCK_N
    start = lo + s * chunk
    end = tl.minimum(start + chunk, seq)
    n = tl.arange(0, BLOCK_N)
    for n0 in range(start, end, BLOCK_N):
        t = n0 + n
        tmask = t < end
        page = tl.load(pt_ptr + b.to(tl.int64) * pt_stride + t // PAGE, mask=tmask, other=0)
        row = page.to(tl.int64) * PAGE + t % PAGE
        kvoff = (row[:, None] * HKV + g) * D + d[None, :]
        kb = tl.load(k_cache_ptr + kvoff, mask=tmask[:, None], other=0).to(tl.float8e4nv, bitcast=True)
        vb = tl.load(v_cache_ptr + kvoff, mask=tmask[:, None], other=0).to(tl.float8e4nv, bitcast=True)
        qk = tl.dot(q, tl.trans(kb.to(tl.bfloat16))) * qk_scale
        # t <= p_j for every old key; the window's lower edge is per row.
        valid = tmask[None, :] & (t[None, :] >= p_row[:, None] - (WINDOW - 1))
        m_i, l_i, acc = _online_update(qk, valid, vb.to(tl.bfloat16), m_i, l_i, acc)

    if S == 1:
        # Program 0 holds the new keys, so l > 0 on every row.
        o = acc / l_i[:, None] * v_scale
        _gate_epilogue(o, qkv_ptr, out_ptr, tok, head, rvalid, d, qkv_stride, HQ, HKV, D, EPS)
    else:
        bg = b * HKV + g
        part = (bg * S + s).to(tl.int64)
        tl.store(ws_acc_ptr + (part * M + r[:, None]) * D + d[None, :], acc)
        tl.store(ws_ml_ptr + part * 2 * M + r, m_i)
        tl.store(ws_ml_ptr + part * 2 * M + M + r, l_i)
        # Release: every thread fences its partial stores at gpu scope, the CTA syncs, then one
        # acq_rel gpu-scope ticket. The program drawing the last ticket combines.
        _gpu_fence(r)
        tl.debug_barrier()
        ticket = tl.atomic_add(cnt_ptr + bg, 1, sem="acq_rel", scope="gpu")
        tl.debug_barrier()
        if ticket == S - 1:
            _gpu_fence(r)  # acquire side, every thread, before the L2 (.cg) reloads
            base = (bg * S).to(tl.int64)
            m_max = tl.full([M], float("-inf"), dtype=tl.float32)
            for s2 in tl.static_range(S):
                m_s = tl.load(ws_ml_ptr + (base + s2) * 2 * M + r, cache_modifier=".cg")
                m_max = tl.maximum(m_max, m_s)
            # Partial S - 1 holds the new keys and is finite on every row, so m_max is too;
            # empty or fully masked partials (m = -inf) get weight 0, never exp2(-inf - -inf).
            m_max = tl.where(m_max == float("-inf"), 0.0, m_max)
            l_sum = tl.zeros([M], dtype=tl.float32)
            o = tl.zeros([M, D], dtype=tl.float32)
            for s2 in tl.static_range(S):
                m_s = tl.load(ws_ml_ptr + (base + s2) * 2 * M + r, cache_modifier=".cg")
                l_s = tl.load(ws_ml_ptr + (base + s2) * 2 * M + M + r, cache_modifier=".cg")
                a_s = tl.load(ws_acc_ptr + ((base + s2) * M + r[:, None]) * D + d[None, :], cache_modifier=".cg")
                w = tl.where(m_s == float("-inf"), 0.0, tl.exp2(m_s - m_max))
                l_sum += l_s * w
                o += a_s * w[:, None]
            o = o / l_sum[:, None] * v_scale
            _gate_epilogue(o, qkv_ptr, out_ptr, tok, head, rvalid, d, qkv_stride, HQ, HKV, D, EPS)
            tl.store(cnt_ptr + bg, 0)  # ready for the next launch / graph replay


def _sliding_forward(data: Inputs) -> torch.Tensor:
    """Sliding-window layers: one launch of _sliding_fused_kernel. Branches on config only."""
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    tokens = batch * length
    qkv = data["qkv"].reshape(tokens, -1)
    device = qkv.device
    theta = cfg["rope_theta"]
    group = hq // hkv
    m = max(16, triton.next_power_of_2(group * length))
    # Decode (L = 1): 16 splits at every batch, 2 BLOCK_N tiles of the window per program.
    splits = 16 if length == 1 else _num_splits(batch, hkv)
    out = torch.empty((batch, length, hq * d), dtype=torch.bfloat16, device=device)
    if splits > 1:
        ws_acc = torch.empty((batch * hkv * splits, m, d), dtype=torch.float32, device=device)
        ws_ml = torch.empty((batch * hkv * splits, 2, m), dtype=torch.float32, device=device)
        counters = _counters(batch, hkv, device)
    else:
        ws_acc = ws_ml = counters = out  # unused when S == 1
    inv_freq = _inv_freq(theta, d, device) if theta else data["k_scale"]  # unused when ROPE is off
    page_table = data["page_table"]
    _sliding_fused_kernel[(batch, hkv, splits)](
        qkv, data["positions"], data["cache_loc"], data["seq_lens"], page_table, inv_freq,
        data["k_scale"], data["v_scale"],
        data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), ws_acc, ws_ml, counters, out,
        qkv.stride(0), page_table.stride(0), cfg["qk_scale_factor"], cfg["sm_scale"] * LOG2E,
        HQ=hq, HKV=hkv, D=d, L=length, M=m, LN=max(16, triton.next_power_of_2(length)),
        WINDOW=cfg["window"], PAGE=cfg["page_size"], S=splits, BLOCK_N=64, ROPE=bool(theta), EPS=RMS_EPS,
        num_warps=4, num_stages=2,
    )
    return out


@triton.jit
def _paired_norm_gate_kernel(
    attn_ptr, qkv_ptr, out_ptr,
    attn_stride_t, attn_stride_h, qkv_stride,
    HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, G: tl.constexpr, EPS: tl.constexpr,
):
    """Full-layer epilogue for the paired path, one program per (token, query head): reads head
    h's own half (h // G) % 2 of FA3's [tokens, HQ, 2*D] output (the other half is the other KV
    head's P@V and is never read), applies the scaleless RMSNorm (eps inside the sqrt of the mean
    of D squares, fp32) times sigmoid(gate) and stores bf16 into the [tokens, HQ*D] output."""
    t = tl.program_id(0)
    h = tl.program_id(1)
    j = tl.arange(0, D)
    half = (h // G) % 2
    src = attn_ptr + t.to(tl.int64) * attn_stride_t + h * attn_stride_h + half * D
    x = tl.load(src + j).to(tl.float32)
    r = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / D + EPS)
    gate = tl.load(qkv_ptr + t.to(tl.int64) * qkv_stride + (HQ + 2 * HKV) * D + h * D + j)
    gate = gate.to(tl.float32)
    y = x * r * tl.sigmoid(gate)
    tl.store(out_ptr + (t.to(tl.int64) * HQ + h) * D + j, y.to(tl.bfloat16))


# ---------------------------------------------------------------------------------------------
# Full-layer decode at batch > 1 (tp8 heads, page 1): split-KV gather attention in CUDA.
#
# Kernel A (splitkv_attn_kernel): a persistent grid of 2 CTAs per SM, 8 warps each. Work items
# are fixed 1024-token chunks of each request, laid out flat (request-major) from an in-kernel
# prefix sum of the chunk counts, and CTA x walks items x, x + grid, ... Per item the 1024 page
# indices and the request's FP8 query (16 heads x 64) are fetched into shared memory with
# cp.async one item ahead (double-buffered), so no K/V address waits on an index load. The item
# is streamed as 16 tiles of 64 tokens: each token is one 128-byte K row and one 128-byte V row
# (both KV heads), copied with 16-byte cp.async.cg into a 4-stage ring (16 KB per stage), whose
# 16-byte chunks are XOR-swizzled by token so the fragment reads are bank-conflict free. The
# tile sequence runs across items without draining. Warp w serves KV head w % 2 and tokens
# (w / 2) * 16 .. +16 of each tile with mma.sync m16n8k16 (f16 in, fp32 accumulate): S = Q K^T
# with the head's 8 queries as rows 0-7 (rows 8-15 zero), then O += P V. FP8 E4M3 is widened to
# f16 exactly (cvt.rn.f16x2.e4m3x2). The dot product's k order is permuted identically for Q
# and K so each lane reads 4 contiguous K bytes; P@V's output columns are permuted (column n of
# n-block nb is dim n * 8 + nb) so each lane reads 8 contiguous V bytes per token. Softmax is
# online in the exp2 domain (fp32 m, l; -inf safe). Positions >= cache_seqlens[b] are never
# dereferenced (zero-filled copies) and are masked to -inf. At item end the four warps of each
# KV head merge in shared memory and the CTA stores one fp32 partial (acc [16, 64] unnormalized,
# m [16], l [16]) at slot b * max_chunks + chunk.
# Kernel B (splitkv_combine_kernel): grid (B, 16), 4 warps per (request, head): log-sum-exp merge
# of the request's partials, times v_scale, then the master epilogue (scaleless RMSNorm over 64,
# eps inside the sqrt, times sigmoid(gate)), bf16 out.
# ---------------------------------------------------------------------------------------------

_SPLITKV_CHUNK = 1024  # must match CHUNK in the CUDA source

_SPLITKV_CPP = r"""
#include <torch/extension.h>
void splitkv_attn(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor page_table,
                  torch::Tensor cache_seqlens, torch::Tensor k_scale, torch::Tensor ws_acc, torch::Tensor ws_ml,
                  double score_scale, int64_t max_chunks, int64_t grid);
void splitkv_combine(torch::Tensor ws_acc, torch::Tensor ws_ml, torch::Tensor cache_seqlens, torch::Tensor v_scale,
                     torch::Tensor qkv, int64_t qkv_stride, torch::Tensor out, int64_t max_chunks, double eps);
"""

_SPLITKV_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {

constexpr int D = 64;              // head dim
constexpr int HQ = 16;             // query heads
constexpr int HKV = 2;             // kv heads
constexpr int G = HQ / HKV;        // queries per kv head (8)
constexpr int ROW = HKV * D;       // bytes per cache row (page 1: both kv heads, 128)
constexpr int CHUNK = 1024;        // tokens per work item
constexpr int TILE = 64;           // tokens per pipeline tile
constexpr int TILES = CHUNK / TILE;
constexpr int STAGES = 4;
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int MAXB = 64;
constexpr int STAGE_BYTES = 2 * TILE * ROW;                  // K tile + V tile = 16 KB
constexpr int RING_OFF = 0;
constexpr int IDX_OFF = RING_OFF + STAGES * STAGE_BYTES;     // int32 [2][CHUNK]
constexpr int Q_OFF = IDX_OFF + 2 * CHUNK * 4;               // e4m3 [2][HQ][D]
constexpr int ACC_OFF = Q_OFF + 2 * HQ * D;                  // f32 [WARPS][G][D]
constexpr int ML_OFF = ACC_OFF + WARPS * G * D * 4;          // f32 [WARPS][G][2]
constexpr int SEQ_OFF = ML_OFF + WARPS * G * 2 * 4;          // int32 [MAXB]
constexpr int PRE_OFF = SEQ_OFF + MAXB * 4;                  // int32 [MAXB + 1]
constexpr int SMEM_A = PRE_OFF + (MAXB + 4) * 4;

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_async4(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory"); }

// Four E4M3 bytes -> two f16x2: lo from bytes 0,1 (byte 0 in the low half), hi from bytes 2,3.
__device__ __forceinline__ void e4m3x4_to_f16x2x2(uint32_t w, uint32_t& lo, uint32_t& hi) {
  asm("{\n\t.reg .b16 a, b;\n\tmov.b32 {a, b}, %2;\n\t"
      "cvt.rn.f16x2.e4m3x2 %0, a;\n\tcvt.rn.f16x2.e4m3x2 %1, b;\n\t}\n"
      : "=r"(lo), "=r"(hi) : "r"(w));
}
__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);  // .x (low half) = lo
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
// D = A B + D, m16n8k16, A row-major f16 (4 regs), B col-major f16 (2 regs), fp32 accumulate.
__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__global__ void __launch_bounds__(THREADS, 2)
splitkv_attn_kernel(const uint8_t* __restrict__ q,          // e4m3 [B, HQ, D]
                    const uint8_t* __restrict__ kc,         // e4m3 [rows, ROW]
                    const uint8_t* __restrict__ vc,         // e4m3 [rows, ROW]
                    const int* __restrict__ page_table, int64_t pt_stride,
                    const int* __restrict__ seqlens, int B,
                    const float* __restrict__ k_scale, float score_scale,
                    float* __restrict__ ws_acc,             // f32 [B * max_chunks, HQ, D]
                    float* __restrict__ ws_ml,              // f32 [B * max_chunks, 2, HQ]
                    int max_chunks) {
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* ring = smem + RING_OFF;
  int* idx_s = reinterpret_cast<int*>(smem + IDX_OFF);
  uint8_t* q_s = smem + Q_OFF;
  float* acc_s = reinterpret_cast<float*>(smem + ACC_OFF);
  float* ml_s = reinterpret_cast<float*>(smem + ML_OFF);
  int* seq_s = reinterpret_cast<int*>(smem + SEQ_OFF);
  int* pre_s = reinterpret_cast<int*>(smem + PRE_OFF);
  const float NEG_INF = __int_as_float(0xff800000u);

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int bx = blockIdx.x, gx = gridDim.x;
  if (tid < B) seq_s[tid] = seqlens[tid];
  __syncthreads();
  if (tid == 0) {
    int run = 0;
    pre_s[0] = 0;
    for (int b = 0; b < B; ++b) {
      run += (seq_s[b] + CHUNK - 1) / CHUNK;
      pre_s[b + 1] = run;
    }
  }
  __syncthreads();
  const int total = pre_s[B];
  if (bx >= total) return;  // uniform over the CTA
  const int nk = (total - bx + gx - 1) / gx;  // items of this CTA
  const int N = nk * TILES;                   // tiles of this CTA, one flat pipeline
  const float scale = score_scale * __ldg(k_scale);  // sm_scale * qk_scale_factor * log2(e) * k_scale

  // Item k of this CTA -> (request, chunk, cache length).
  auto item_of = [&](int k, int& b, int& chunk, int& seq) {
    const int i = bx + k * gx;
    int lo = 0;
    while (lo + 1 < B && pre_s[lo + 1] <= i) ++lo;
    b = lo;
    chunk = i - pre_s[lo];
    seq = seq_s[lo];
  };
  // Page indices and query of item k into buffer k & 1 (positions past the length zero-filled).
  auto issue_meta = [&](int k) {
    int b, chunk, seq;
    item_of(k, b, chunk, seq);
    const int buf = k & 1;
    const int* prow = page_table + (int64_t)b * pt_stride;
#pragma unroll
    for (int j = 0; j < CHUNK / THREADS; ++j) {
      const int e = tid + j * THREADS;
      const int pos = chunk * CHUNK + e;
      const bool ok = pos < seq;
      cp_async4(smem_u32(idx_s + buf * CHUNK + e), ok ? prow + pos : prow, ok ? 4 : 0);
    }
    if (tid < HQ * D / 16)
      cp_async16(smem_u32(q_s + buf * HQ * D + tid * 16), q + (int64_t)b * HQ * D + tid * 16, 16);
  };
  // Tile nn of the flat sequence into ring stage nn % STAGES: thread -> token tid / 4, 16-byte
  // chunks 2 * (tid % 4) + {0, 1} of its K row and its V row, chunk c stored at c ^ (token % 8).
  int ik = -1, ib = 0, ichunk = 0, iseq = 0;
  auto issue_tile = [&](int nn) {
    const int k = nn / TILES, t = nn % TILES;
    if (k != ik) { ik = k; item_of(k, ib, ichunk, iseq); }
    uint8_t* st = ring + (nn % STAGES) * STAGE_BYTES;
    const int tt = tid >> 2;
    const int e = t * TILE + tt;
    const bool ok = ichunk * CHUNK + e < iseq;
    const int64_t row = ok ? (int64_t)idx_s[(k & 1) * CHUNK + e] : 0;
    const uint8_t* ks = kc + row * ROW;
    const uint8_t* vs = vc + row * ROW;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int c = (tid & 3) * 2 + j;
      const int dst = tt * ROW + ((c ^ (tt & 7)) << 4);
      cp_async16(smem_u32(st + dst), ks + c * 16, ok ? 16 : 0);
      cp_async16(smem_u32(st + TILE * ROW + dst), vs + c * 16, ok ? 16 : 0);
    }
  };

  // Prologue: item 0's meta (waited), then tiles 0 .. STAGES-2; item 1's meta rides tile 0's group.
  issue_meta(0);
  cp_commit();
  cp_wait<0>();
  __syncthreads();
#pragma unroll
  for (int j = 0; j < STAGES - 1; ++j) {
    if (j < N) {
      issue_tile(j);
      if (j % TILES == 0 && j / TILES + 1 < nk) issue_meta(j / TILES + 1);
    }
    cp_commit();
  }

  const int g = warp & 1;             // kv head
  const int slice = (warp >> 1) * 16; // token slice of each tile
  const int quad = lane >> 2, qi = lane & 3;
  uint32_t qa[4][2];
  float m_i = NEG_INF, l_i = 0.f;
  float acc[8][4];
  int cb = 0, cchunk = 0, cseq = 0;

  for (int n = 0; n < N; ++n) {
    cp_wait<STAGES - 2>();  // tile n (and everything issued before it) has landed
    __syncthreads();        // visible to all; every warp is done with tile n - 1's stage
    {
      const int nn = n + STAGES - 1;
      if (nn < N) {
        issue_tile(nn);
        if (nn % TILES == 0 && nn / TILES + 1 < nk) issue_meta(nn / TILES + 1);
      }
      cp_commit();  // always, so the group count stays uniform
    }
    const int k = n / TILES, t = n % TILES;
    if (t == 0) {
      item_of(k, cb, cchunk, cseq);
      // A operand for kstep s: row quad = query g * G + quad; k index 2 * qi + {0, 1} holds dims
      // s * 16 + 4 * qi + {0, 1}, k index 8 + 2 * qi + {0, 1} holds dims s * 16 + 4 * qi + {2, 3}
      // (K uses the same permutation, so the dot product is unchanged).
      const uint8_t* qrow = q_s + (k & 1) * HQ * D + (g * G + quad) * D + qi * 4;
#pragma unroll
      for (int s = 0; s < 4; ++s)
        e4m3x4_to_f16x2x2(*reinterpret_cast<const uint32_t*>(qrow + s * 16), qa[s][0], qa[s][1]);
      m_i = NEG_INF;
      l_i = 0.f;
#pragma unroll
      for (int nb = 0; nb < 8; ++nb) acc[nb][0] = acc[nb][1] = acc[nb][2] = acc[nb][3] = 0.f;
    }
    const uint8_t* Ks = ring + (n % STAGES) * STAGE_BYTES;
    const uint8_t* Vs = Ks + TILE * ROW;

    // S = Q K^T over the warp's 16 tokens: n-block nb holds tokens slice + nb * 8 + (0..7).
    float s[2][4];
#pragma unroll
    for (int nb = 0; nb < 2; ++nb) {
      s[nb][0] = s[nb][1] = s[nb][2] = s[nb][3] = 0.f;
      const int tt = slice + nb * 8 + quad;
      const uint8_t* kr = Ks + tt * ROW + qi * 4;
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        const uint32_t w = *reinterpret_cast<const uint32_t*>(kr + (((g * 4 + ks) ^ (tt & 7)) << 4));
        uint32_t b0, b1;
        e4m3x4_to_f16x2x2(w, b0, b1);
        mma16816(s[nb], qa[ks][0], 0u, qa[ks][1], 0u, b0, b1);
      }
    }
    // Row quad's scores at tokens slice + 2 * qi + {0, 1} (nb 0) and + 8 (nb 1).
    const int pos0 = cchunk * CHUNK + t * TILE + slice + qi * 2;
    float x0 = pos0 < cseq ? s[0][0] * scale : NEG_INF;
    float x1 = pos0 + 1 < cseq ? s[0][1] * scale : NEG_INF;
    float x2 = pos0 + 8 < cseq ? s[1][0] * scale : NEG_INF;
    float x3 = pos0 + 9 < cseq ? s[1][1] * scale : NEG_INF;
    float mx = fmaxf(fmaxf(x0, x1), fmaxf(x2, x3));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
    const float m_new = fmaxf(m_i, mx);
    const float m_use = m_new == NEG_INF ? 0.f : m_new;
    const float alpha = ex2(m_i - m_use);
    x0 = ex2(x0 - m_use);
    x1 = ex2(x1 - m_use);
    x2 = ex2(x2 - m_use);
    x3 = ex2(x3 - m_use);
    l_i = l_i * alpha + (x0 + x1) + (x2 + x3);  // this lane's columns; quad-reduced at item end
    m_i = m_new;
#pragma unroll
    for (int nb = 0; nb < 8; ++nb) { acc[nb][0] *= alpha; acc[nb][1] *= alpha; }
    const uint32_t pa0 = pack_f16x2(x0, x1), pa2 = pack_f16x2(x2, x3);

    // O += P V. B operand of dim-block nb: k = token slice + 2 * qi + {0, 1} (reg 0) and + 8
    // (reg 1), column quad = dim quad * 8 + nb. Each lane reads 8 contiguous bytes per token.
    const int tv = slice + qi * 2;
    const int vlc = g * 4 + (quad >> 1);
    const int voff = (quad & 1) * 8;
    const uint2 v0 = *reinterpret_cast<const uint2*>(Vs + tv * ROW + ((vlc ^ (tv & 7)) << 4) + voff);
    const uint2 v1 = *reinterpret_cast<const uint2*>(Vs + (tv + 1) * ROW + ((vlc ^ ((tv + 1) & 7)) << 4) + voff);
    const uint2 v8 = *reinterpret_cast<const uint2*>(Vs + (tv + 8) * ROW + ((vlc ^ ((tv + 8) & 7)) << 4) + voff);
    const uint2 v9 = *reinterpret_cast<const uint2*>(Vs + (tv + 9) * ROW + ((vlc ^ ((tv + 9) & 7)) << 4) + voff);
#pragma unroll
    for (int hw = 0; hw < 2; ++hw) {
      const uint32_t w0 = hw ? v0.y : v0.x, w1 = hw ? v1.y : v1.x;
      const uint32_t w8 = hw ? v8.y : v8.x, w9 = hw ? v9.y : v9.x;
      // Interleave the two tokens' bytes: byte 2j = token a dim j, byte 2j + 1 = token b dim j.
      uint32_t r0[4], r1[4];
      e4m3x4_to_f16x2x2(__byte_perm(w0, w1, 0x5140), r0[0], r0[1]);
      e4m3x4_to_f16x2x2(__byte_perm(w0, w1, 0x7362), r0[2], r0[3]);
      e4m3x4_to_f16x2x2(__byte_perm(w8, w9, 0x5140), r1[0], r1[1]);
      e4m3x4_to_f16x2x2(__byte_perm(w8, w9, 0x7362), r1[2], r1[3]);
#pragma unroll
      for (int j = 0; j < 4; ++j) mma16816(acc[hw * 4 + j], pa0, 0u, pa2, 0u, r0[j], r1[j]);
    }

    if (t == TILES - 1) {
      // Item end: merge the four warps of each kv head, store the item's partial.
      l_i += __shfl_xor_sync(0xffffffffu, l_i, 1);
      l_i += __shfl_xor_sync(0xffffffffu, l_i, 2);
      const int rw = warp * G + quad;
      if (qi == 0) { ml_s[rw * 2] = m_i; ml_s[rw * 2 + 1] = l_i; }
      float* ar = acc_s + rw * D;
#pragma unroll
      for (int nb = 0; nb < 8; ++nb) {
        ar[(qi * 2) * 8 + nb] = acc[nb][0];
        ar[(qi * 2 + 1) * 8 + nb] = acc[nb][1];
      }
      __syncthreads();
      const int h = tid >> 4, d0 = (tid & 15) * 4;
      const int hg = h / G, r = h % G;
      float mw[4];
      float M = NEG_INF;
#pragma unroll
      for (int j = 0; j < 4; ++j) { mw[j] = ml_s[((hg + 2 * j) * G + r) * 2]; M = fmaxf(M, mw[j]); }
      const float Mu = M == NEG_INF ? 0.f : M;
      float4 o = make_float4(0.f, 0.f, 0.f, 0.f);
      float L = 0.f;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int wr = (hg + 2 * j) * G + r;
        const float wt = mw[j] == NEG_INF ? 0.f : ex2(mw[j] - Mu);
        L += wt * ml_s[wr * 2 + 1];
        const float4 a = *reinterpret_cast<const float4*>(acc_s + wr * D + d0);
        o.x += wt * a.x; o.y += wt * a.y; o.z += wt * a.z; o.w += wt * a.w;
      }
      const int64_t slot = (int64_t)cb * max_chunks + cchunk;
      *reinterpret_cast<float4*>(ws_acc + (slot * HQ + h) * D + d0) = o;
      if ((tid & 15) == 0) { ws_ml[slot * 2 * HQ + h] = M; ws_ml[slot * 2 * HQ + HQ + h] = L; }
    }
  }
  cp_wait<0>();  // the trailing groups are empty; drain anyway
}

__global__ void __launch_bounds__(128)
splitkv_combine_kernel(const float* __restrict__ ws_acc, const float* __restrict__ ws_ml,
                       const int* __restrict__ seqlens, const float* __restrict__ v_scale,
                       const __nv_bfloat16* __restrict__ qkv, int64_t qkv_stride,
                       __nv_bfloat16* __restrict__ out, int max_chunks, float eps) {
  __shared__ float red_m[4];
  __shared__ float red_l[4];
  __shared__ float red_acc[4][D];
  const float NEG_INF = __int_as_float(0xff800000u);
  const int b = blockIdx.x, h = blockIdx.y;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int n = (seqlens[b] + CHUNK - 1) / CHUNK;
  const int64_t base = (int64_t)b * max_chunks;

  float M = NEG_INF;
  for (int c = tid; c < n; c += 128) M = fmaxf(M, ws_ml[(base + c) * 2 * HQ + h]);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));
  if (lane == 0) red_m[warp] = M;
  __syncthreads();
  M = fmaxf(fmaxf(red_m[0], red_m[1]), fmaxf(red_m[2], red_m[3]));
  const float Mu = M == NEG_INF ? 0.f : M;

  float a0 = 0.f, a1 = 0.f, L = 0.f;
#pragma unroll 4
  for (int c = warp; c < n; c += 4) {
    const int64_t sl = base + c;
    const float m = ws_ml[sl * 2 * HQ + h];
    const float wt = m == NEG_INF ? 0.f : ex2(m - Mu);  // empty partials weigh 0
    L += wt * ws_ml[sl * 2 * HQ + HQ + h];
    const float2 a = *reinterpret_cast<const float2*>(ws_acc + (sl * HQ + h) * D + lane * 2);
    a0 += wt * a.x;
    a1 += wt * a.y;
  }
  red_acc[warp][lane * 2] = a0;
  red_acc[warp][lane * 2 + 1] = a1;
  if (lane == 0) red_l[warp] = L;
  __syncthreads();
  if (warp == 0) {
    float o0 = 0.f, o1 = 0.f, Ls = 0.f;
#pragma unroll
    for (int w = 0; w < 4; ++w) { o0 += red_acc[w][lane * 2]; o1 += red_acc[w][lane * 2 + 1]; Ls += red_l[w]; }
    const float vs = __ldg(v_scale);
    o0 = o0 / Ls * vs;
    o1 = o1 / Ls * vs;
    float ss = o0 * o0 + o1 * o1;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    const float r = 1.f / sqrtf(ss / D + eps);
    const __nv_bfloat16* gp = qkv + (int64_t)b * qkv_stride + (HQ + 2 * HKV) * D + h * D + lane * 2;
    const float g0 = __bfloat162float(gp[0]), g1 = __bfloat162float(gp[1]);
    const float y0 = o0 * r / (1.f + expf(-g0));
    const float y1 = o1 * r / (1.f + expf(-g1));
    *reinterpret_cast<__nv_bfloat162*>(out + ((int64_t)b * HQ + h) * D + lane * 2) = __floats2bfloat162_rn(y0, y1);
  }
}

}  // namespace

void splitkv_attn(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor page_table,
                  torch::Tensor cache_seqlens, torch::Tensor k_scale, torch::Tensor ws_acc, torch::Tensor ws_ml,
                  double score_scale, int64_t max_chunks, int64_t grid) {
  const int B = static_cast<int>(cache_seqlens.size(0));
  TORCH_CHECK(B >= 1 && B <= MAXB, "splitkv_attn: batch out of range");
  TORCH_CHECK(page_table.stride(1) == 1 && q.is_contiguous() && k_cache.is_contiguous() && v_cache.is_contiguous(),
              "splitkv_attn: layout");
  static bool attr_set = false;
  if (!attr_set) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(splitkv_attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_A));
    attr_set = true;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  splitkv_attn_kernel<<<static_cast<unsigned>(grid), THREADS, SMEM_A, stream>>>(
      reinterpret_cast<const uint8_t*>(q.data_ptr()), reinterpret_cast<const uint8_t*>(k_cache.data_ptr()),
      reinterpret_cast<const uint8_t*>(v_cache.data_ptr()), page_table.data_ptr<int>(), page_table.stride(0),
      cache_seqlens.data_ptr<int>(), B, k_scale.data_ptr<float>(), static_cast<float>(score_scale),
      ws_acc.data_ptr<float>(), ws_ml.data_ptr<float>(), static_cast<int>(max_chunks));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void splitkv_combine(torch::Tensor ws_acc, torch::Tensor ws_ml, torch::Tensor cache_seqlens, torch::Tensor v_scale,
                     torch::Tensor qkv, int64_t qkv_stride, torch::Tensor out, int64_t max_chunks, double eps) {
  const int B = static_cast<int>(cache_seqlens.size(0));
  auto stream = at::cuda::getCurrentCUDAStream();
  splitkv_combine_kernel<<<dim3(B, HQ), 128, 0, stream>>>(
      ws_acc.data_ptr<float>(), ws_ml.data_ptr<float>(), cache_seqlens.data_ptr<int>(), v_scale.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()), qkv_stride,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(max_chunks), static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


def _build_splitkv():
    """Compile the CUDA module once per process, at import (never during graph capture)."""
    t0 = time.perf_counter()
    try:
        from torch.utils.cpp_extension import load_inline
        ext = load_inline(
            name="muse_full_decode_splitkv_v1",
            cpp_sources=_SPLITKV_CPP,
            cuda_sources=_SPLITKV_CUDA,
            functions=["splitkv_attn", "splitkv_combine"],
            extra_cuda_cflags=["-O3", "-arch=sm_90a"],
            verbose=False,
        )
    except Exception as e:  # keep the master path if the build fails
        print(f"[splitkv] load_inline failed after {time.perf_counter() - t0:.1f}s, master path kept: {e!r}")
        return None
    print(f"[splitkv] load_inline compiled in {time.perf_counter() - t0:.1f}s")
    return ext


_EXT = None          # the compiled module; None keeps the master FA3 path everywhere
_SPLITKV_GRID = None  # grid override, set only by the import-time self-check
_SMS = {}


def _sm_count(device) -> int:
    """Streaming multiprocessors of the device (hardware property, not data)."""
    key = device.index if device.index is not None else torch.cuda.current_device()
    n = _SMS.get(key)
    if n is None:
        n = _SMS[key] = torch.cuda.get_device_properties(key).multi_processor_count
    return n


_B1_PATH = False  # batch 1 on the two-level stream-K kernel; set only once its in-process validation passes


def _shape_gate(cfg) -> bool:
    """Config-only shape predicate: full-layer decode, 1 <= batch <= 64, tp8 heads, head dim 64, page 1."""
    return (cfg["window"] == 0 and cfg["q_len"] == 1 and 1 <= cfg["batch"] <= 64
            and cfg["num_q_heads"] == 16 and cfg["num_kv_heads"] == 2 and cfg["head_dim"] == 64
            and cfg["page_size"] == 1)


def _splitkv_gate(cfg) -> bool:
    """Config-only predicate for the CUDA split-KV path. Batch 1 additionally needs the validated
    two-level stage-3 path (otherwise it stays on master FA3); 1 < batch <= 64 is stage 2/1 as before."""
    return _EXT is not None and _shape_gate(cfg) and (cfg["batch"] > 1 or _B1_PATH)


def _full_decode_splitkv(data: Inputs, q: torch.Tensor, qkv: torch.Tensor) -> torch.Tensor:
    """Kernels A and B on the gated path; q is the unpaired FP8 query [B, 16, 64]. The workspace
    and grid come from shapes and the device only."""
    if _EXT2 is not None:
        return _full_decode_sk(data, q, qkv)
    cfg = data["config"]
    batch, hq, d = cfg["batch"], cfg["num_q_heads"], cfg["head_dim"]
    device = qkv.device
    page_table = data["page_table"]
    max_chunks = -(-page_table.shape[1] // _SPLITKV_CHUNK)  # page 1: one page per token
    ws_acc = torch.empty((batch * max_chunks, hq, d), dtype=torch.float32, device=device)
    ws_ml = torch.empty((batch * max_chunks, 2, hq), dtype=torch.float32, device=device)
    grid = _SPLITKV_GRID or 2 * _sm_count(device)
    _EXT.splitkv_attn(q, data["k_cache"], data["v_cache"], page_table, data["cache_seqlens"], data["k_scale"],
                      ws_acc, ws_ml, cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E, max_chunks, grid)
    out = torch.empty((batch, 1, hq * d), dtype=torch.bfloat16, device=device)
    _EXT.splitkv_combine(ws_acc, ws_ml, data["cache_seqlens"], data["v_scale"], qkv, qkv.stride(0), out,
                         max_chunks, RMS_EPS)
    return out


# ---------------------------------------------------------------------------------------------
# Stage 2 (module muse_full_decode_splitkv_v2, kernel splitkv_attn_sk_kernel): the same 2-CTA/SM,
# 256-thread, all-thread cp.async.cg ring and the same m16n8k16 f16 tile math as kernel A, with
# two changes.
# (1) Stream-K over 64-token tiles. Every CTA rebuilds the tile prefix TP[b] (tiles_b =
#     ceil(seq_b / 64), TP[B] = Ttot) and owns the flat contiguous tile range
#     [c * Ttot / grid, (c + 1) * Ttot / grid), which may span requests. Page indices are
#     prefetched 16 tiles at a time into a double-buffered smem block (4-byte cp.async, zero-fill
#     past each request end); the ring runs across request boundaries, and the first tile of each
#     segment also carries that request's FP8 query (1 KB) into a per-stage query slot.
# (2) Combine + norm-gate folded in: a segment that covers a whole request is finalized in place
#     (LSE merge of the 4 warps per kv head, times v_scale, scaleless RMSNorm, times
#     sigmoid(gate), bf16 out). Otherwise the CTA stores its fp32 partial at slot c + b, fences,
#     and draws a per-request ticket; the CTA drawing nseg_b - 1 (nseg_b = owner(TP[b+1]-1) -
#     owner(TP[b]) + 1) fences, merges slots c_lo..c_hi and finalizes, then resets the ticket to 0
#     so the next launch (graph replay) starts clean. No CTA ever waits on another.
# Variants (template <STAGES, TICKET>): 0 = <4, true> (the timed path), 1 = <4, false> (every
# segment to the workspace + splitkv_sk_combine_kernel; diagnostics only), 2 = <5, true>
# (diagnostics only).
# ---------------------------------------------------------------------------------------------

_SK_CPP = r"""
#include <torch/extension.h>
#include <vector>
void sk_attn(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor page_table,
             torch::Tensor cache_seqlens, torch::Tensor k_scale, torch::Tensor v_scale, torch::Tensor qkv,
             int64_t qkv_stride, torch::Tensor out, torch::Tensor ws_acc, torch::Tensor ws_ml, torch::Tensor ticket,
             torch::Tensor gticket, torch::Tensor gacc, torch::Tensor gml,
             double score_scale, double eps, int64_t grid, int64_t variant);
void sk_combine(torch::Tensor ws_acc, torch::Tensor ws_ml, torch::Tensor cache_seqlens, torch::Tensor v_scale,
                torch::Tensor qkv, int64_t qkv_stride, torch::Tensor out, int64_t attn_grid, double eps);
std::vector<int64_t> sk_attrs(int64_t variant);
"""

_SK_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <vector>

namespace {

constexpr int D = 64;              // head dim
constexpr int HQ = 16;             // query heads
constexpr int HKV = 2;             // kv heads
constexpr int G = HQ / HKV;        // queries per kv head (8)
constexpr int ROW = HKV * D;       // bytes per cache row (page 1: both kv heads, 128)
constexpr int TILE = 64;           // tokens per tile (the stream-K unit)
constexpr int MB_TILES = 16;       // tiles per page-index block
constexpr int MB_TOK = MB_TILES * TILE;
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int MAXB = 64;
constexpr int STAGE_BYTES = 2 * TILE * ROW;  // K tile + V tile = 16 KB
constexpr int QBYTES = HQ * D;               // one request's e4m3 query

template <int STAGES>
struct Lay {
  static constexpr int RING = 0;
  static constexpr int QR = RING + STAGES * STAGE_BYTES;      // e4m3 [STAGES][HQ][D]
  static constexpr int IDX = QR + STAGES * QBYTES;            // int32 [2][MB_TOK]
  static constexpr int ACC = IDX + 2 * MB_TOK * 4;            // f32 [WARPS][G][D]
  static constexpr int ML = ACC + WARPS * G * D * 4;          // f32 [WARPS][G][2]
  static constexpr int SEQ = ML + WARPS * G * 2 * 4;          // int32 [MAXB]
  static constexpr int TP = SEQ + MAXB * 4;                   // int32 [MAXB + 1] (+pad)
  static constexpr int FLAG = TP + (MAXB + 4) * 4;            // int32 [4]
  static constexpr int BYTES = FLAG + 16;
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_async4(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory"); }

__device__ __forceinline__ void e4m3x4_to_f16x2x2(uint32_t w, uint32_t& lo, uint32_t& hi) {
  asm("{\n\t.reg .b16 a, b;\n\tmov.b32 {a, b}, %2;\n\t"
      "cvt.rn.f16x2.e4m3x2 %0, a;\n\tcvt.rn.f16x2.e4m3x2 %1, b;\n\t}\n"
      : "=r"(lo), "=r"(hi) : "r"(w));
}
__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// cache_seqlens -> seq_s, and the tile prefix tp_s[0..B] (tp_s[B] = Ttot). Ends with a barrier.
__device__ __forceinline__ void load_prefix(const int* __restrict__ seqlens, int B, int* seq_s, int* tp_s) {
  const int tid = threadIdx.x;
  if (tid < B) { const int s = seqlens[tid]; seq_s[tid] = s > 0 ? s : 0; }
  __syncthreads();
  if (tid == 0) {
    int run = 0;
    tp_s[0] = 0;
    for (int b = 0; b < B; ++b) {
      run += (seq_s[b] + TILE - 1) / TILE;
      tp_s[b + 1] = run;
    }
  }
  __syncthreads();
}
// First tile of CTA c, and the CTA owning tile t (the exact inverse: s_c <= t < s_{c+1}).
__device__ __forceinline__ int range_start(int c, int T, int gx) {
  return static_cast<int>((static_cast<long long>(c) * T) / gx);
}
__device__ __forceinline__ int tile_owner(int t, int T, int gx) {
  return static_cast<int>(((static_cast<long long>(t) + 1) * gx - 1) / T);
}

// LSE merge of the partials in slots s_lo..s_hi for head h, dims d0..d0+3 (M = the raw max, so the
// result can itself be stored as a slot). L2 loads (__ldcg): the slots were written by other CTAs.
// Stage-2 callers pass s = c + b (c = c_lo..c_hi); the two-level path passes group-slot indices.
__device__ __forceinline__ void merge_range(const float* ws_acc, const float* ws_ml, int s_lo, int s_hi,
                                            int h, int d0, float4& o, float& L, float& Mout) {
  const float NEG_INF = __int_as_float(0xff800000u);
  float M = NEG_INF;
  for (int c = s_lo; c <= s_hi; ++c) M = fmaxf(M, __ldcg(ws_ml + (int64_t)c * 2 * HQ + h));
  const float Mu = M == NEG_INF ? 0.f : M;
  Mout = M;
  o = make_float4(0.f, 0.f, 0.f, 0.f);
  L = 0.f;
  for (int c = s_lo; c <= s_hi; ++c) {
    const int64_t sl = c;
    const float m = __ldcg(ws_ml + sl * 2 * HQ + h);
    const float wt = m == NEG_INF ? 0.f : ex2(m - Mu);
    L += wt * __ldcg(ws_ml + sl * 2 * HQ + HQ + h);
    const float4 a = __ldcg(reinterpret_cast<const float4*>(ws_acc + (sl * HQ + h) * D + d0));
    o.x += wt * a.x; o.y += wt * a.y; o.z += wt * a.z; o.w += wt * a.w;
  }
}
__device__ __forceinline__ void merge_slots(const float* ws_acc, const float* ws_ml, int b, int c_lo, int c_hi,
                                            int h, int d0, float4& o, float& L) {
  float M;
  merge_range(ws_acc, ws_ml, c_lo + b, c_hi + b, h, d0, o, L, M);
}

// Release/acquire ticket draw (two-level path): the CTA's partial stores are ordered before the
// increment (acq_rel at gpu scope, on top of the caller's __threadfence + __syncthreads), and the
// returned count orders the winner after every earlier arrival's stores.
__device__ __forceinline__ int ticket_arrive(int* p) {
  int old;
  asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(1) : "memory");
  return old;
}

// Stage-1 B epilogue for head h, dims d0..d0+3 (16 consecutive lanes hold one head): o / L *
// v_scale, scaleless RMSNorm over 64 (eps inside the sqrt), times sigmoid(gate), bf16 out.
// Every lane of the warp must call it (16-lane shuffles).
__device__ __forceinline__ void finalize_store(float4 o, float L, float vs, int b, int h, int d0,
                                               const __nv_bfloat16* __restrict__ qkv, int64_t qkv_stride,
                                               __nv_bfloat16* __restrict__ out, float eps) {
  o.x = o.x / L * vs; o.y = o.y / L * vs; o.z = o.z / L * vs; o.w = o.w / L * vs;
  float ss = o.x * o.x + o.y * o.y + o.z * o.z + o.w * o.w;
  ss += __shfl_xor_sync(0xffffffffu, ss, 1);
  ss += __shfl_xor_sync(0xffffffffu, ss, 2);
  ss += __shfl_xor_sync(0xffffffffu, ss, 4);
  ss += __shfl_xor_sync(0xffffffffu, ss, 8);
  const float r = 1.f / sqrtf(ss / D + eps);
  const __nv_bfloat162* gp =
      reinterpret_cast<const __nv_bfloat162*>(qkv + (int64_t)b * qkv_stride + (HQ + 2 * HKV) * D + h * D + d0);
  const float2 g01 = __bfloat1622float2(gp[0]);
  const float2 g23 = __bfloat1622float2(gp[1]);
  const float y0 = o.x * r / (1.f + expf(-g01.x));
  const float y1 = o.y * r / (1.f + expf(-g01.y));
  const float y2 = o.z * r / (1.f + expf(-g23.x));
  const float y3 = o.w * r / (1.f + expf(-g23.y));
  __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(out + ((int64_t)b * HQ + h) * D + d0);
  op[0] = __floats2bfloat162_rn(y0, y1);
  op[1] = __floats2bfloat162_rn(y2, y3);
}

template <int STAGES, bool TICKET, bool TWO = false>
__global__ void __launch_bounds__(THREADS, 2)
splitkv_attn_sk_kernel(const uint8_t* __restrict__ q,          // e4m3 [B, HQ, D]
                       const uint8_t* __restrict__ kc,         // e4m3 [rows, ROW]
                       const uint8_t* __restrict__ vc,         // e4m3 [rows, ROW]
                       const int* __restrict__ page_table, int64_t pt_stride,
                       const int* __restrict__ seqlens, int B,
                       const float* __restrict__ k_scale, const float* __restrict__ v_scale, float score_scale,
                       const __nv_bfloat16* __restrict__ qkv, int64_t qkv_stride,
                       __nv_bfloat16* __restrict__ out,        // bf16 [B, HQ * D]
                       float* __restrict__ ws_acc,             // f32 [grid + B, HQ, D]
                       float* __restrict__ ws_ml,              // f32 [grid + B, 2, HQ]
                       int* __restrict__ ticket,               // int32 [>= B], 0 between launches
                       int* __restrict__ gticket,              // int32 [>= B * 32], 0 between launches (TWO)
                       float* __restrict__ gacc,               // f32 [B * 32, HQ, D] group slots (TWO)
                       float* __restrict__ gml,                // f32 [B * 32, 2, HQ] (TWO)
                       float eps) {
  using S = Lay<STAGES>;
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* ring = smem + S::RING;
  uint8_t* qr = smem + S::QR;
  int* idx_s = reinterpret_cast<int*>(smem + S::IDX);
  float* acc_s = reinterpret_cast<float*>(smem + S::ACC);
  float* ml_s = reinterpret_cast<float*>(smem + S::ML);
  int* seq_s = reinterpret_cast<int*>(smem + S::SEQ);
  int* tp_s = reinterpret_cast<int*>(smem + S::TP);
  int* flag_s = reinterpret_cast<int*>(smem + S::FLAG);
  const float NEG_INF = __int_as_float(0xff800000u);

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int bx = blockIdx.x;
  load_prefix(seqlens, B, seq_s, tp_s);
  const int T = tp_s[B];
  if (T <= 0) return;  // uniform
  // Effective grid min(gridDim.x, T): below it every CTA owns >= 1 tile, so CTAs c_lo..c_hi of a
  // request all hold a segment (the ticket count and the merge rely on it). CTAs above it exit.
  const int gx = min(static_cast<int>(gridDim.x), T);
  if (bx >= gx) return;  // uniform
  const int t0 = range_start(bx, T, gx);
  const int N = range_start(bx + 1, T, gx) - t0;  // tiles of this CTA, one flat pipeline
  if (N <= 0) return;  // uniform
  const int nblk = (N + MB_TILES - 1) / MB_TILES;
  const float scale = score_scale * __ldg(k_scale);  // sm_scale * qk_scale_factor * log2(e) * k_scale
  const float vs = __ldg(v_scale);

  // Request holding global tile gt: the largest b with tp_s[b] <= gt (gt < Ttot).
  auto find_req = [&](int gt) {
    int lo = 0, hi = B - 1;
    while (lo < hi) {
      const int mid = (lo + hi + 1) >> 1;
      if (tp_s[mid] <= gt) lo = mid; else hi = mid - 1;
    }
    return lo;
  };
  // Page indices of block k (flat tiles 16k .. 16k+15) into buffer k & 1: thread -> tile tid / 16,
  // tokens (tid % 16) * 4 .. +4. Positions past the request end (or past the range) zero-filled.
  auto issue_meta = [&](int k) {
    const int buf = k & 1;
    const int tl = tid >> 4;
    const int e0 = (tid & 15) * 4;
    const int n = k * MB_TILES + tl;
    int b = 0, pos = 0, seq = 0;
    if (n < N) {
      const int gt = t0 + n;
      b = find_req(gt);
      pos = (gt - tp_s[b]) * TILE + e0;
      seq = seq_s[b];
    }
    const int* prow = page_table + (int64_t)b * pt_stride;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const bool ok = pos + j < seq;
      cp_async4(smem_u32(idx_s + buf * MB_TOK + tl * TILE + e0 + j), ok ? prow + pos + j : prow, ok ? 4 : 0);
    }
  };
  // Tile nn (issued in order 0, 1, 2, ...) into ring stage nn % STAGES, as kernel A: thread ->
  // token tid / 4, 16-byte chunks 2 * (tid % 4) + {0, 1} of its K row and V row, chunk c stored
  // at c ^ (token % 8). A segment's first tile also brings the request's query into its q slot.
  int p_b = find_req(t0);
  auto issue_tile = [&](int nn) {
    const int gt = t0 + nn;
    while (gt >= tp_s[p_b + 1]) ++p_b;
    const int lt = gt - tp_s[p_b];
    const int st = nn % STAGES;
    uint8_t* stp = ring + st * STAGE_BYTES;
    const int tt = tid >> 2;
    const bool ok = lt * TILE + tt < seq_s[p_b];
    const int64_t row = ok ? (int64_t)idx_s[((nn / MB_TILES) & 1) * MB_TOK + (nn % MB_TILES) * TILE + tt] : 0;
    const uint8_t* ks = kc + row * ROW;
    const uint8_t* vsrc = vc + row * ROW;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int c = (tid & 3) * 2 + j;
      const int dst = tt * ROW + ((c ^ (tt & 7)) << 4);
      cp_async16(smem_u32(stp + dst), ks + c * 16, ok ? 16 : 0);
      cp_async16(smem_u32(stp + TILE * ROW + dst), vsrc + c * 16, ok ? 16 : 0);
    }
    if ((nn == 0 || lt == 0) && tid < QBYTES / 16)
      cp_async16(smem_u32(qr + st * QBYTES + tid * 16), q + (int64_t)p_b * QBYTES + tid * 16, 16);
  };

  // Prologue: block 0's indices (waited), then tiles 0 .. STAGES-2; block 1 rides tile 0's group.
  issue_meta(0);
  cp_commit();
  cp_wait<0>();
  __syncthreads();
#pragma unroll
  for (int j = 0; j < STAGES - 1; ++j) {
    if (j < N) {
      issue_tile(j);
      if (j % MB_TILES == 0 && j / MB_TILES + 1 < nblk) issue_meta(j / MB_TILES + 1);
    }
    cp_commit();
  }

  const int g = warp & 1;             // kv head
  const int slice = (warp >> 1) * 16; // token slice of each tile
  const int quad = lane >> 2, qi = lane & 3;
  uint32_t qa[4][2];
  float m_i = NEG_INF, l_i = 0.f;
  float acc[8][4];
  int c_b = find_req(t0);
  bool seg_head = false;  // the current segment began at its request's first tile

  for (int n = 0; n < N; ++n) {
    cp_wait<STAGES - 2>();  // tile n (and everything issued before it) has landed
    __syncthreads();        // visible to all; every warp is done with tile n - 1's stage
    {
      const int nn = n + STAGES - 1;
      if (nn < N) {
        issue_tile(nn);
        if (nn % MB_TILES == 0 && nn / MB_TILES + 1 < nblk) issue_meta(nn / MB_TILES + 1);
      }
      cp_commit();  // always, so the group count stays uniform
    }
    const int gt = t0 + n;
    while (gt >= tp_s[c_b + 1]) ++c_b;
    const int lt = gt - tp_s[c_b];
    const int nt_b = tp_s[c_b + 1] - tp_s[c_b];
    const int cseq = seq_s[c_b];
    const int st = n % STAGES;
    if (n == 0 || lt == 0) {
      seg_head = lt == 0;
      const uint8_t* qrow = qr + st * QBYTES + (g * G + quad) * D + qi * 4;
#pragma unroll
      for (int s = 0; s < 4; ++s)
        e4m3x4_to_f16x2x2(*reinterpret_cast<const uint32_t*>(qrow + s * 16), qa[s][0], qa[s][1]);
      m_i = NEG_INF;
      l_i = 0.f;
#pragma unroll
      for (int nb = 0; nb < 8; ++nb) acc[nb][0] = acc[nb][1] = acc[nb][2] = acc[nb][3] = 0.f;
    }
    const uint8_t* Ks = ring + st * STAGE_BYTES;
    const uint8_t* Vs = Ks + TILE * ROW;

    float s[2][4];
#pragma unroll
    for (int nb = 0; nb < 2; ++nb) {
      s[nb][0] = s[nb][1] = s[nb][2] = s[nb][3] = 0.f;
      const int tt = slice + nb * 8 + quad;
      const uint8_t* kr = Ks + tt * ROW + qi * 4;
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        const uint32_t w = *reinterpret_cast<const uint32_t*>(kr + (((g * 4 + ks) ^ (tt & 7)) << 4));
        uint32_t b0, b1;
        e4m3x4_to_f16x2x2(w, b0, b1);
        mma16816(s[nb], qa[ks][0], 0u, qa[ks][1], 0u, b0, b1);
      }
    }
    const int pos0 = lt * TILE + slice + qi * 2;
    float x0 = pos0 < cseq ? s[0][0] * scale : NEG_INF;
    float x1 = pos0 + 1 < cseq ? s[0][1] * scale : NEG_INF;
    float x2 = pos0 + 8 < cseq ? s[1][0] * scale : NEG_INF;
    float x3 = pos0 + 9 < cseq ? s[1][1] * scale : NEG_INF;
    float mx = fmaxf(fmaxf(x0, x1), fmaxf(x2, x3));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
    const float m_new = fmaxf(m_i, mx);
    const float m_use = m_new == NEG_INF ? 0.f : m_new;
    const float alpha = ex2(m_i - m_use);
    x0 = ex2(x0 - m_use);
    x1 = ex2(x1 - m_use);
    x2 = ex2(x2 - m_use);
    x3 = ex2(x3 - m_use);
    l_i = l_i * alpha + (x0 + x1) + (x2 + x3);
    m_i = m_new;
#pragma unroll
    for (int nb = 0; nb < 8; ++nb) { acc[nb][0] *= alpha; acc[nb][1] *= alpha; }
    const uint32_t pa0 = pack_f16x2(x0, x1), pa2 = pack_f16x2(x2, x3);

    const int tv = slice + qi * 2;
    const int vlc = g * 4 + (quad >> 1);
    const int voff = (quad & 1) * 8;
    const uint2 v0 = *reinterpret_cast<const uint2*>(Vs + tv * ROW + ((vlc ^ (tv & 7)) << 4) + voff);
    const uint2 v1 = *reinterpret_cast<const uint2*>(Vs + (tv + 1) * ROW + ((vlc ^ ((tv + 1) & 7)) << 4) + voff);
    const uint2 v8 = *reinterpret_cast<const uint2*>(Vs + (tv + 8) * ROW + ((vlc ^ ((tv + 8) & 7)) << 4) + voff);
    const uint2 v9 = *reinterpret_cast<const uint2*>(Vs + (tv + 9) * ROW + ((vlc ^ ((tv + 9) & 7)) << 4) + voff);
#pragma unroll
    for (int hw = 0; hw < 2; ++hw) {
      const uint32_t w0 = hw ? v0.y : v0.x, w1 = hw ? v1.y : v1.x;
      const uint32_t w8 = hw ? v8.y : v8.x, w9 = hw ? v9.y : v9.x;
      uint32_t r0[4], r1[4];
      e4m3x4_to_f16x2x2(__byte_perm(w0, w1, 0x5140), r0[0], r0[1]);
      e4m3x4_to_f16x2x2(__byte_perm(w0, w1, 0x7362), r0[2], r0[3]);
      e4m3x4_to_f16x2x2(__byte_perm(w8, w9, 0x5140), r1[0], r1[1]);
      e4m3x4_to_f16x2x2(__byte_perm(w8, w9, 0x7362), r1[2], r1[3]);
#pragma unroll
      for (int j = 0; j < 4; ++j) mma16816(acc[hw * 4 + j], pa0, 0u, pa2, 0u, r0[j], r1[j]);
    }

    if (n == N - 1 || lt == nt_b - 1) {
      // Segment end (uniform over the CTA): merge the four warps of each kv head in smem.
      l_i += __shfl_xor_sync(0xffffffffu, l_i, 1);
      l_i += __shfl_xor_sync(0xffffffffu, l_i, 2);
      const int rw = warp * G + quad;
      if (qi == 0) { ml_s[rw * 2] = m_i; ml_s[rw * 2 + 1] = l_i; }
      float* ar = acc_s + rw * D;
#pragma unroll
      for (int nb = 0; nb < 8; ++nb) {
        ar[(qi * 2) * 8 + nb] = acc[nb][0];
        ar[(qi * 2 + 1) * 8 + nb] = acc[nb][1];
      }
      __syncthreads();
      const int h = tid >> 4, d0 = (tid & 15) * 4;
      const int hg = h / G, r = h % G;
      float mw[4];
      float M = NEG_INF;
#pragma unroll
      for (int j = 0; j < 4; ++j) { mw[j] = ml_s[((hg + 2 * j) * G + r) * 2]; M = fmaxf(M, mw[j]); }
      const float Mu = M == NEG_INF ? 0.f : M;
      float4 o = make_float4(0.f, 0.f, 0.f, 0.f);
      float L = 0.f;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int wr = (hg + 2 * j) * G + r;
        const float wt = mw[j] == NEG_INF ? 0.f : ex2(mw[j] - Mu);
        L += wt * ml_s[wr * 2 + 1];
        const float4 a = *reinterpret_cast<const float4*>(acc_s + wr * D + d0);
        o.x += wt * a.x; o.y += wt * a.y; o.z += wt * a.z; o.w += wt * a.w;
      }
      const bool whole = seg_head && lt == nt_b - 1;  // this CTA is the request's only segment
      if (TICKET && whole) {
        finalize_store(o, L, vs, c_b, h, d0, qkv, qkv_stride, out, eps);
      } else {
        const int64_t slot = (int64_t)bx + c_b;
        *reinterpret_cast<float4*>(ws_acc + (slot * HQ + h) * D + d0) = o;
        if ((tid & 15) == 0) { ws_ml[slot * 2 * HQ + h] = M; ws_ml[slot * 2 * HQ + HQ + h] = L; }
        if (TICKET && TWO) {
          // Two-level ticket. CTAs c_lo..c_hi hold request c_b's segments (nseg of them); group
          // size Gs = ceil(sqrt(nseg)), group gi = (c - c_lo) / Gs, ngrp = ceil(nseg / Gs) <= 32.
          const int c_lo = tile_owner(tp_s[c_b], T, gx);
          const int c_hi = tile_owner(tp_s[c_b + 1] - 1, T, gx);
          const int nseg = c_hi - c_lo + 1;
          int Gs = 1;
          while (Gs * Gs < nseg) ++Gs;
          const int gi = (bx - c_lo) / Gs;
          const int ngrp = (nseg + Gs - 1) / Gs;
          const int g_lo = c_lo + gi * Gs;
          const int g_hi = min(c_hi, g_lo + Gs - 1);
          __threadfence();  // release: this CTA's partial is visible device-wide before the ticket
          __syncthreads();
          if (tid == 0) flag_s[0] = ticket_arrive(gticket + c_b * 32 + gi) == g_hi - g_lo ? 1 : 0;
          __syncthreads();
          if (flag_s[0]) {  // uniform: the last arrival of group gi
            __threadfence();  // acquire: every group member's partial is visible
            float4 om;
            float Lm, Mm;
            merge_range(ws_acc, ws_ml, g_lo + c_b, g_hi + c_b, h, d0, om, Lm, Mm);
            if (tid == 0) gticket[c_b * 32 + gi] = 0;  // all members arrived: clean for the next launch
            if (ngrp == 1) {
              finalize_store(om, Lm, vs, c_b, h, d0, qkv, qkv_stride, out, eps);
            } else {
              const int64_t gs = (int64_t)c_b * 32 + gi;
              *reinterpret_cast<float4*>(gacc + (gs * HQ + h) * D + d0) = om;
              if ((tid & 15) == 0) { gml[gs * 2 * HQ + h] = Mm; gml[gs * 2 * HQ + HQ + h] = Lm; }
              __threadfence();  // release: the group slot is visible before the second-level ticket
              __syncthreads();
              if (tid == 0) flag_s[1] = ticket_arrive(ticket + c_b) == ngrp - 1 ? 1 : 0;
              __syncthreads();
              if (flag_s[1]) {  // uniform: the last group of request c_b
                __threadfence();  // acquire
                float4 of;
                float Lf, Mf;
                merge_range(gacc, gml, c_b * 32, c_b * 32 + ngrp - 1, h, d0, of, Lf, Mf);
                finalize_store(of, Lf, vs, c_b, h, d0, qkv, qkv_stride, out, eps);
                if (tid == 0) ticket[c_b] = 0;  // every group has arrived: clean for the next launch
              }
            }
          }
        } else if (TICKET) {
          const int c_lo = tile_owner(tp_s[c_b], T, gx);
          const int c_hi = tile_owner(tp_s[c_b + 1] - 1, T, gx);
          __threadfence();  // this CTA's partial is visible device-wide before the ticket
          __syncthreads();
          if (tid == 0) flag_s[0] = atomicAdd(ticket + c_b, 1) == c_hi - c_lo ? 1 : 0;
          __syncthreads();
          if (flag_s[0]) {  // uniform: the last segment of request c_b to finish
            __threadfence();
            float4 om;
            float Lm;
            merge_slots(ws_acc, ws_ml, c_b, c_lo, c_hi, h, d0, om, Lm);
            finalize_store(om, Lm, vs, c_b, h, d0, qkv, qkv_stride, out, eps);
            if (tid == 0) ticket[c_b] = 0;  // every segment has arrived: clean for the next launch
          }
        }
      }
    }
  }
  cp_wait<0>();  // the trailing groups are empty; drain anyway
}

// Diagnostics only (variant 1): one CTA per request merges its slots and finalizes.
__global__ void __launch_bounds__(THREADS)
splitkv_sk_combine_kernel(const float* __restrict__ ws_acc, const float* __restrict__ ws_ml,
                          const int* __restrict__ seqlens, int B, const float* __restrict__ v_scale,
                          const __nv_bfloat16* __restrict__ qkv, int64_t qkv_stride,
                          __nv_bfloat16* __restrict__ out, int attn_grid, float eps) {
  __shared__ int seq_s[MAXB];
  __shared__ int tp_s[MAXB + 4];
  load_prefix(seqlens, B, seq_s, tp_s);
  const int b = blockIdx.x, T = tp_s[B];
  if (tp_s[b + 1] == tp_s[b]) return;
  attn_grid = min(attn_grid, T);  // the attention kernel's effective grid
  const int c_lo = tile_owner(tp_s[b], T, attn_grid);
  const int c_hi = tile_owner(tp_s[b + 1] - 1, T, attn_grid);
  const int h = threadIdx.x >> 4, d0 = (threadIdx.x & 15) * 4;
  float4 o;
  float L;
  merge_slots(ws_acc, ws_ml, b, c_lo, c_hi, h, d0, o, L);
  finalize_store(o, L, __ldg(v_scale), b, h, d0, qkv, qkv_stride, out, eps);
}

using SkFn = void (*)(const uint8_t*, const uint8_t*, const uint8_t*, const int*, int64_t, const int*, int,
                      const float*, const float*, float, const __nv_bfloat16*, int64_t, __nv_bfloat16*,
                      float*, float*, int*, int*, float*, float*, float);

SkFn sk_pick(int64_t v) {
  if (v == 3) return splitkv_attn_sk_kernel<4, true, true>;
  if (v == 1) return splitkv_attn_sk_kernel<4, false>;
  if (v == 2) return splitkv_attn_sk_kernel<5, true>;
  return splitkv_attn_sk_kernel<4, true>;
}
int sk_smem(int64_t v) { return v == 2 ? Lay<5>::BYTES : Lay<4>::BYTES; }  // variant 3 = <4, ticket, two-level>
void sk_prepare(int64_t v) {
  static bool done[4] = {false, false, false, false};
  if (!done[v]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(sk_pick(v)),
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, sk_smem(v)));
    done[v] = true;
  }
}

}  // namespace

void sk_attn(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor page_table,
             torch::Tensor cache_seqlens, torch::Tensor k_scale, torch::Tensor v_scale, torch::Tensor qkv,
             int64_t qkv_stride, torch::Tensor out, torch::Tensor ws_acc, torch::Tensor ws_ml, torch::Tensor ticket,
             torch::Tensor gticket, torch::Tensor gacc, torch::Tensor gml,
             double score_scale, double eps, int64_t grid, int64_t variant) {
  const int B = static_cast<int>(cache_seqlens.size(0));
  TORCH_CHECK(B >= 1 && B <= MAXB, "sk_attn: batch out of range");
  TORCH_CHECK(variant >= 0 && variant <= 3, "sk_attn: variant");
  if (variant == 3) {
    TORCH_CHECK(grid <= 1024, "sk_attn: two-level ticket needs grid <= 1024 (<= 32 groups)");
    TORCH_CHECK(gticket.numel() >= B * 32 && gticket.scalar_type() == at::kInt, "sk_attn: gticket");
    TORCH_CHECK(gacc.size(0) >= B * 32 && gml.size(0) >= B * 32 && gacc.scalar_type() == at::kFloat
                && gml.scalar_type() == at::kFloat, "sk_attn: group slots");
  }
  TORCH_CHECK(grid >= 1 && ws_acc.size(0) >= grid + B && ws_ml.size(0) >= grid + B, "sk_attn: workspace");
  TORCH_CHECK(ticket.numel() >= B && ticket.scalar_type() == at::kInt, "sk_attn: ticket");
  TORCH_CHECK(cache_seqlens.scalar_type() == at::kInt && page_table.scalar_type() == at::kInt, "sk_attn: dtypes");
  TORCH_CHECK(page_table.stride(1) == 1 && q.is_contiguous() && k_cache.is_contiguous() && v_cache.is_contiguous()
              && out.is_contiguous(), "sk_attn: layout");
  sk_prepare(variant);
  auto stream = at::cuda::getCurrentCUDAStream();
  SkFn fn = sk_pick(variant);
  fn<<<static_cast<unsigned>(grid), THREADS, sk_smem(variant), stream>>>(
      reinterpret_cast<const uint8_t*>(q.data_ptr()), reinterpret_cast<const uint8_t*>(k_cache.data_ptr()),
      reinterpret_cast<const uint8_t*>(v_cache.data_ptr()), page_table.data_ptr<int>(), page_table.stride(0),
      cache_seqlens.data_ptr<int>(), B, k_scale.data_ptr<float>(), v_scale.data_ptr<float>(),
      static_cast<float>(score_scale), reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()), qkv_stride,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), ws_acc.data_ptr<float>(), ws_ml.data_ptr<float>(),
      ticket.data_ptr<int>(), gticket.data_ptr<int>(), gacc.data_ptr<float>(), gml.data_ptr<float>(),
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sk_combine(torch::Tensor ws_acc, torch::Tensor ws_ml, torch::Tensor cache_seqlens, torch::Tensor v_scale,
                torch::Tensor qkv, int64_t qkv_stride, torch::Tensor out, int64_t attn_grid, double eps) {
  const int B = static_cast<int>(cache_seqlens.size(0));
  TORCH_CHECK(B >= 1 && B <= MAXB, "sk_combine: batch out of range");
  auto stream = at::cuda::getCurrentCUDAStream();
  splitkv_sk_combine_kernel<<<B, THREADS, 0, stream>>>(
      ws_acc.data_ptr<float>(), ws_ml.data_ptr<float>(), cache_seqlens.data_ptr<int>(), B, v_scale.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()), qkv_stride,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(attn_grid), static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// {binaryVersion, ptxVersion, numRegs, localSizeBytes, maxThreadsPerBlock, dynamic smem, CTAs/SM}
std::vector<int64_t> sk_attrs(int64_t variant) {
  TORCH_CHECK(variant >= 0 && variant <= 3, "sk_attrs: variant");
  sk_prepare(variant);
  cudaFuncAttributes a;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a, reinterpret_cast<const void*>(sk_pick(variant))));
  int occ = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, sk_pick(variant), THREADS, sk_smem(variant)));
  return {static_cast<int64_t>(a.binaryVersion), static_cast<int64_t>(a.ptxVersion), static_cast<int64_t>(a.numRegs),
          static_cast<int64_t>(a.localSizeBytes), static_cast<int64_t>(a.maxThreadsPerBlock),
          static_cast<int64_t>(sk_smem(variant)), static_cast<int64_t>(occ)};
}
"""

_SK_LABELS = ("<4 stages, ticket>", "<4 stages, no ticket>", "<5 stages, ticket>", "<4 stages, two-level ticket>")
_EXT2 = None       # the stage-2 module; None keeps stage 1 (A + B) on the gated path
_SK_VARIANT = 0    # the timed path: 4 stages + ticket (fixed; the diagnostics call the module directly)
_SK_ARCH_FLAGS = None
_TICKETS = {}


def _say(msg: str) -> None:
    print(msg, flush=True)


def _sk_ticket(device) -> torch.Tensor:
    """int32 [64] per-request ticket counters, zeroed once at import (allocated from the device
    alone, never from inputs); every launch leaves them at 0 (the last arriver resets its own)."""
    key = device.index if device.index is not None else torch.cuda.current_device()
    t = _TICKETS.get(key)
    if t is None:
        t = _TICKETS[key] = torch.zeros(64, dtype=torch.int32, device=device)
    return t


_GTICKETS = {}


def _sk_gticket(device) -> torch.Tensor:
    """int32 [64 * 32] group-level ticket counters of the two-level path, zeroed once at import
    (allocated from the device alone); every launch leaves them at 0."""
    key = device.index if device.index is not None else torch.cuda.current_device()
    t = _GTICKETS.get(key)
    if t is None:
        t = _GTICKETS[key] = torch.zeros(64 * 32, dtype=torch.int32, device=device)
    return t


def _build_sk():
    """Compile the stage-2 module at import, sm_90a only (TORCH_CUDA_ARCH_LIST=9.0a, restored after)."""
    global _SK_ARCH_FLAGS
    t0 = time.perf_counter()
    prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    try:
        from torch.utils.cpp_extension import load_inline
        try:
            from torch.utils.cpp_extension import _get_cuda_arch_flags
            _SK_ARCH_FLAGS = list(_get_cuda_arch_flags())
        except Exception as e:  # private helper; the cubin check below is authoritative
            _say(f"[sk] could not read torch's nvcc arch flags: {e!r}")
        _say(f"[sk] building muse_full_decode_splitkv_v3: TORCH_CUDA_ARCH_LIST=9.0a, arch flags {_SK_ARCH_FLAGS}, "
             f"extra -O3 -lineinfo")
        ext = load_inline(
            name="muse_full_decode_splitkv_v3",
            cpp_sources=_SK_CPP,
            cuda_sources=_SK_CUDA,
            functions=["sk_attn", "sk_combine", "sk_attrs"],
            extra_cuda_cflags=["-O3", "-lineinfo"],
            verbose=False,
        )
    except Exception as e:
        _say(f"[sk] load_inline failed after {time.perf_counter() - t0:.1f}s: {e!r}")
        return None
    finally:
        if prev is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = prev
    _say(f"[sk] load_inline compiled in {time.perf_counter() - t0:.1f}s")
    return ext


def _sk_launch(data: Inputs, q: torch.Tensor, qkv: torch.Tensor, variant: int, grid: int):
    """splitkv_attn_sk_kernel on the gated shape; workspace from grid and batch only."""
    cfg = data["config"]
    batch, hq, d = cfg["batch"], cfg["num_q_heads"], cfg["head_dim"]
    device = qkv.device
    ws_acc = torch.empty((grid + batch, hq, d), dtype=torch.float32, device=device)
    ws_ml = torch.empty((grid + batch, 2, hq), dtype=torch.float32, device=device)
    out = torch.empty((batch, 1, hq * d), dtype=torch.bfloat16, device=device)
    ng = batch * 32 if variant == 3 else 1  # group slots only for the two-level variant
    gacc = torch.empty((ng, hq, d), dtype=torch.float32, device=device)
    gml = torch.empty((ng, 2, hq), dtype=torch.float32, device=device)
    _EXT2.sk_attn(q, data["k_cache"], data["v_cache"], data["page_table"], data["cache_seqlens"], data["k_scale"],
                  data["v_scale"], qkv, qkv.stride(0), out, ws_acc, ws_ml, _sk_ticket(device),
                  _sk_gticket(device), gacc, gml, cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E, RMS_EPS, grid, variant)
    return out, ws_acc, ws_ml


def _sk_combine(data: Inputs, qkv: torch.Tensor, out, ws_acc, ws_ml, grid: int) -> None:
    _EXT2.sk_combine(ws_acc, ws_ml, data["cache_seqlens"], data["v_scale"], qkv, qkv.stride(0), out, grid, RMS_EPS)


def _full_decode_sk(data: Inputs, q: torch.Tensor, qkv: torch.Tensor) -> torch.Tensor:
    """Stage 2 on the gated path: one launch after the pre-attention kernel (variant 1, the
    no-ticket diagnostic, adds the separate combine)."""
    sms = _sm_count(qkv.device)
    if data["config"]["batch"] == 1:  # stage 3: one CTA per SM + the two-level ticket (fixed by shape)
        out, _, _ = _sk_launch(data, q, qkv, 3, _SPLITKV_GRID or sms)
        return out
    grid = _SPLITKV_GRID or 2 * sms
    out, ws_acc, ws_ml = _sk_launch(data, q, qkv, _SK_VARIANT, grid)
    if _SK_VARIANT == 1:
        _sk_combine(data, qkv, out, ws_acc, ws_ml, grid)
    return out


def kernel_fn(data: Inputs) -> torch.Tensor:
    cfg = data["config"]
    # Sliding decode at every batch (config-only predicate): the whole step in one fused launch.
    # Every other shape (sliding verify, every full layer) keeps the path below.
    if cfg["window"] and cfg["q_len"] == 1:
        return _sliding_forward(data)
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    tokens = batch * length
    qkv = data["qkv"].reshape(tokens, -1)
    device = qkv.device
    theta = cfg["rope_theta"]

    k_cache, v_cache = data["k_cache"], data["v_cache"]
    window = cfg["window"]
    # Full layers: pair the two KV heads of each token into one 128-byte FP8 row (heads 2s and
    # 2s+1 are adjacent halves of it) so FA3 gathers one row per token instead of two. Each
    # query sits in its own head's half of a zero-padded 128-wide query, so the zero half adds
    # exactly 0 to every logit; FA3 maps query head h to super-head h // (2G), which keeps the
    # original grouping. The pre-attention kernel writes that padded query directly and a Triton
    # epilogue reads each head's half of FA3's output, so no pad/un-pad copies are launched. This
    # one config-only predicate selects both the query layout and the FA3/epilogue branch.
    # Companion paired-heads-launch-cleanup-ab-tune-batch-gate: pair only at batch > 1 (batch 1
    # keeps the unpaired path, where pairing lost 27-35%).
    # The CUDA split-KV path (full decode, 1 < batch <= 64, tp8, page 1) takes the unpaired query.
    splitkv = _splitkv_gate(cfg)
    paired = window == 0 and hkv % 2 == 0 and batch > 1 and not splitkv
    g = hq // hkv
    q = torch.empty((tokens, hq, 2 * d if paired else d), dtype=FP8, device=device)
    inv_freq = _inv_freq(theta, d, device) if theta else data["k_scale"]  # unused when ROPE is off
    _pre_attention_kernel[(tokens, hq + hkv)](
        qkv, data["positions"], data["cache_loc"], inv_freq, data["k_scale"], data["v_scale"],
        q, k_cache, v_cache,
        qkv.stride(0),
        HQ=hq, HKV=hkv, D=d, ROPE=bool(theta), EPS=RMS_EPS, PAIRED=paired, G=g,
        num_warps=1,
    )
    if splitkv:
        return _full_decode_splitkv(data, q, qkv)

    # The judge's scheduler_metadata was built for the unpaired geometry, so on the paired path
    # it is not passed (FA3 builds its own schedule from the paged lengths).
    if paired:
        s = hkv // 2
        pages, page_size = k_cache.shape[0], k_cache.shape[1]
        k_attn = k_cache.view(pages, page_size, s, 2 * d)
        v_attn = v_cache.view(pages, page_size, s, 2 * d)
        q_attn = q
        attn_heads = s
        metadata = None
    else:
        k_attn, v_attn, q_attn = k_cache, v_cache, q
        attn_heads = hkv
        metadata = data["scheduler_metadata"]
    descale = {}
    if cfg["kv_scale"] != 1.0:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, attn_heads),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, attn_heads))
    out = flash_attn_with_kvcache(
        q=q_attn,
        k_cache=k_attn,
        v_cache=v_attn,
        page_table=data["page_table"],
        cache_seqlens=data["cache_seqlens"],
        cu_seqlens_q=data["cu_seqlens_q"],
        max_seqlen_q=length,
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
        causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=metadata,
        **descale,
    )
    if isinstance(out, tuple):
        out = out[0]
    if paired:
        # Each query's own half of FA3's 128-wide output, normalized and gated in one launch.
        attn = out.view(tokens, hq, 2 * d)
        o = torch.empty((tokens, hq * d), dtype=torch.bfloat16, device=device)
        _paired_norm_gate_kernel[(tokens, hq)](
            attn, qkv, o,
            attn.stride(0), attn.stride(1), qkv.stride(0),
            HQ=hq, HKV=hkv, D=d, G=g, EPS=RMS_EPS,
            num_warps=1,
        )
        return o.view(batch, length, hq * d)
    gate = qkv[:, (hq + 2 * hkv) * d:]
    o = rms_norm_gate(out.reshape(tokens, hq * d).to(torch.bfloat16), gate, None, head_dim=d, eps=RMS_EPS,
                      has_norm=True)
    return o.view(batch, length, hq * d)


def _splitkv_synthetic(lens, kv_scale, gen, device):
    """Gated-shape inputs (full layer, decode, tp8, page 1) with ragged lengths, built as
    reference.generate_input builds them (scattered pages, odd page-table stride)."""
    batch, hq, hkv, d = len(lens), 16, 2, 64
    max_pages = max(lens) + 1 + 37
    total = batch * max_pages
    perm = torch.randperm(total, generator=gen, device=device).to(torch.int32)
    page_table = torch.full((batch, max_pages), -1, dtype=torch.int32, device=device)
    off = 0
    for b, n in enumerate(lens):
        page_table[b, :n + 1] = perm[off:off + n + 1]
        off += n + 1
    seq = torch.tensor(lens, dtype=torch.int64, device=device)
    k_cache = (torch.randn((total, 1, hkv, d), generator=gen, device=device) / kv_scale).to(FP8)
    v_cache = (torch.randn((total, 1, hkv, d), generator=gen, device=device) / kv_scale).to(FP8)
    qkv = (2 * torch.randn((batch, 1, (2 * hq + 2 * hkv) * d), generator=gen, device=device)).to(torch.bfloat16)
    return {
        "qkv": qkv,
        "positions": seq.clone(),
        "cache_loc": torch.gather(page_table.long(), 1, seq[:, None]).reshape(-1),
        "cache_seqlens": (seq + 1).to(torch.int32),
        "cu_seqlens_q": torch.arange(0, batch + 1, dtype=torch.int32, device=device),
        "scheduler_metadata": None,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "k_scale": torch.tensor(kv_scale, dtype=torch.float32, device=device),
        "v_scale": torch.tensor(kv_scale, dtype=torch.float32, device=device),
        "page_table": page_table,
        "seq_lens": seq.to(torch.int32),
        "config": {"batch": batch, "q_len": 1, "num_q_heads": hq, "num_kv_heads": hkv, "head_dim": d,
                   "page_size": 1, "window": 0, "rope_theta": 0.0, "qk_scale_factor": 5.4730064863875,
                   "sm_scale": d ** -0.5, "kv_scale": kv_scale},
    }


def _splitkv_reference(data, lens):
    """FP32 attention over the cache as kernel_fn left it (new rows included), FP8 queries."""
    cfg = data["config"]
    batch, hq, hkv, d = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    qkv = data["qkv"].reshape(batch, -1).float()
    q = qkv[:, :hq * d].view(batch, hq, d)
    q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + RMS_EPS)
    q = q.to(FP8).float() * cfg["qk_scale_factor"]
    kf = data["k_cache"].view(-1, hkv, d)
    vf = data["v_cache"].view(-1, hkv, d)
    outs = []
    for b, n in enumerate(lens):
        rows = data["page_table"][b, :n + 1].long()
        keys = (kf[rows].float() * data["k_scale"]).repeat_interleave(hq // hkv, dim=1)
        vals = (vf[rows].float() * data["v_scale"]).repeat_interleave(hq // hkv, dim=1)
        p = torch.softmax(torch.einsum("hd,nhd->hn", q[b], keys) * cfg["sm_scale"], dim=-1)
        outs.append(torch.einsum("hn,nhd->hd", p, vals))
    a = torch.stack(outs)
    a = a * torch.rsqrt(a.pow(2).mean(-1, keepdim=True) + RMS_EPS)
    gate = qkv[:, (hq + 2 * hkv) * d:].view(batch, hq, d)
    return (a * torch.sigmoid(gate)).reshape(batch, 1, hq * d)


def _splitkv_self_check() -> bool:
    """One-time eager check at import, on synthetic data only (never in the timed path): the
    gated path against an FP32 reference, with ragged lengths (partial chunks, fully masked
    warps), kv scale 1 and 1/4, the device grid and a 3-CTA grid (several items per CTA, so the
    cross-item pipeline and double-buffered meta are exercised)."""
    global _SPLITKV_GRID
    lens = [9000, 1025, 70, 4097, 1023]
    gen = torch.Generator(device="cuda").manual_seed(1234)
    ok = True
    try:
        for kv_scale, grid in ((1.0, None), (0.25, 3), (1.0, 3)):
            data = _splitkv_synthetic(lens, kv_scale, gen, torch.device("cuda"))
            assert _splitkv_gate(data["config"])
            _SPLITKV_GRID = grid
            out = kernel_fn(data).float()
            ref = _splitkv_reference(data, lens)
            err = float((out - ref).norm() / ref.norm())
            finite = bool(torch.isfinite(out).all())
            print(f"[splitkv] self-check kv_scale={kv_scale} grid={grid or 'device'}: rel L2 {err:.2e}, finite {finite}")
            ok = ok and finite and err < 2e-2
    except Exception as e:
        print(f"[splitkv] self-check raised: {e!r}")
        ok = False
    finally:
        _SPLITKV_GRID = None
    return ok


_SK_LENS16 = [9000, 1024, 66, 4097, 1028, 69, 6, 1, 39, 131, 2000, 5003, 63, 64, 3075, 700]


def _sk_prepare_q(data: Inputs):
    """The gated path's pre-attention launch (unpaired FP8 query, new cache rows), as kernel_fn
    issues it. Import-time self-check and diagnostics only."""
    cfg = data["config"]
    batch, hq, hkv, d = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    qkv = data["qkv"].reshape(batch, -1)
    q = torch.empty((batch, hq, d), dtype=FP8, device=qkv.device)
    _pre_attention_kernel[(batch, hq + hkv)](
        qkv, data["positions"], data["cache_loc"], data["k_scale"], data["k_scale"], data["v_scale"],
        q, data["k_cache"], data["v_cache"],
        qkv.stride(0),
        HQ=hq, HKV=hkv, D=d, ROPE=False, EPS=RMS_EPS, PAIRED=False, G=hq // hkv,
        num_warps=1,
    )
    return q, qkv


def _rel_l2(out, ref):
    out = out.float()
    return float((out - ref).norm() / ref.norm()), bool(torch.isfinite(out).all())


def _sk_case(tag, lens, kv_scale, grid, gen, variant=0) -> bool:
    """One synthetic gated-shape case, each launch announced before and synced + confirmed after
    (a fault names its kernel), against the FP32 reference."""
    dev = torch.device("cuda")
    data = _splitkv_synthetic(lens, kv_scale, gen, dev)
    assert _shape_gate(data["config"])
    g = grid or 2 * _sm_count(dev)
    _say(f"[sk] {tag}: launching _pre_attention_kernel")
    q, qkv = _sk_prepare_q(data)
    torch.cuda.synchronize()
    _say(f"[sk] {tag}: launching splitkv_attn_sk_kernel{_SK_LABELS[variant]} grid {g}")
    out, ws_acc, ws_ml = _sk_launch(data, q, qkv, variant, g)
    torch.cuda.synchronize()
    _say(f"[sk] {tag}: splitkv_attn_sk_kernel returned")
    if variant == 1:
        _say(f"[sk] {tag}: launching splitkv_sk_combine_kernel")
        _sk_combine(data, qkv, out, ws_acc, ws_ml, g)
        torch.cuda.synchronize()
        _say(f"[sk] {tag}: splitkv_sk_combine_kernel returned")
    err, finite = _rel_l2(out, _splitkv_reference(data, lens))
    ok = finite and err < 2e-2
    _say(f"[sk] {tag}: rel L2 {err:.2e}, finite {finite} -> {'ok' if ok else 'FAIL'}")
    return ok


def _sk_replay_check(gen, lens=None, kv_scale=1.0):
    """kernel_fn captured as a CUDA graph and replayed twice, the output poisoned with NaN before
    each replay: the second replay is right only if the first left every ticket (both levels) at 0."""
    dev = torch.device("cuda")
    lens = lens or _SK_LENS16
    data = _splitkv_synthetic(lens, kv_scale, gen, dev)
    _say("[sk] replay: eager warm-up of kernel_fn on a side stream")
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        kernel_fn(data)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    _say("[sk] replay: warm-up returned")
    ref = _splitkv_reference(data, lens)
    graph = torch.cuda.CUDAGraph()
    _say("[sk] replay: capturing kernel_fn")
    with torch.cuda.graph(graph):
        out = kernel_fn(data)
    torch.cuda.synchronize()
    for i in (1, 2):
        out.fill_(float("nan"))
        _say(f"[sk] replay B={len(lens)} {i}: launching graph (_pre_attention_kernel + splitkv_attn_sk_kernel)")
        graph.replay()
        torch.cuda.synchronize()
        err, finite = _rel_l2(out, ref)
        ok = finite and err < 2e-2
        _say(f"[sk] replay B={len(lens)} {i}: rel L2 {err:.2e}, finite {finite} -> {'ok' if ok else 'FAIL'}")
        if not ok:
            return f"graph replay B={len(lens)} {i}"
    del graph
    return None


def _sk_validate():
    """In-process gate for stage 2: image check, canary, self-check, replays. Returns None on
    success, else the reason the path stays on stage 1."""
    dev = torch.device("cuda")
    cap = tuple(torch.cuda.get_device_capability())
    _say(f"[sk] device capability {cap}, {torch.cuda.get_device_name()}, {_sm_count(dev)} SMs")
    attrs = []
    for v in range(3):
        a = list(_EXT2.sk_attrs(v))
        attrs.append(a)
        _say(f"[sk] splitkv_attn_sk_kernel{_SK_LABELS[v]}: binaryVersion {a[0]}, ptxVersion {a[1]}, regs {a[2]}, "
             f"local {a[3]} B, dyn smem {a[5]} B, {a[6]} CTAs/SM")
    if attrs[0][6] < 2:
        _say("[sk] WARNING: the timed variant does not reach 2 CTAs/SM (performance, not correctness)")
    if cap != (9, 0):
        return f"device capability {cap}"
    # ptxVersion is logged only: its value for a PTX-less sm_90a cubin is not documented.
    if any(a[0] != 90 for a in attrs):
        return "cubin version is not 90 (sm_90a image not loaded)"
    if _SK_ARCH_FLAGS is not None:
        bad = [f for f in _SK_ARCH_FLAGS if "arch=" in f and "90a" not in f]
        if bad:
            return f"nvcc arch flags beyond sm_90a: {bad}"
    gen = torch.Generator(device="cuda").manual_seed(4321)
    if not _sk_case("canary B=2 short", [4, 66], 1.0, None, gen):
        return "canary"
    rng = random.Random(2025)
    lens32 = [rng.randint(1, 12000) for _ in range(32)]
    lens32[5], lens32[9], lens32[17] = 6, 64, 1
    lens64 = [rng.randint(1, 70) if i % 3 == 0 else rng.randint(64, 6000) for i in range(64)]
    cases = (
        ("B=2 one request spans every CTA, grid 7", [20000, 2], 1.0, 7),
        ("B=2 one request spans every CTA, grid device", [60000, 2], 0.25, None),
        ("B=16 ragged kv 0.25 grid device", _SK_LENS16, 0.25, None),
        ("B=16 ragged grid 3", _SK_LENS16, 1.0, 3),
        ("B=32 ragged grid 3", lens32, 1.0, 3),
        ("B=32 ragged kv 0.25 grid device", lens32, 0.25, None),
        ("B=64 mixed kv 0.25 grid device", lens64, 0.25, None),
        ("B=64 mixed grid 7", lens64, 1.0, 7),
    )
    for tag, lens, kv_scale, grid in cases:
        if not _sk_case(tag, lens, kv_scale, grid, gen):
            return f"self-check {tag}"
    reason = _sk_replay_check(gen)
    if reason is not None:
        return reason
    left = int(_sk_ticket(dev).abs().sum().item())
    _say(f"[sk] tickets after the self-check: sum {left}")
    if left:
        return "tickets not reset"
    return None


def _sk_diagnostics() -> None:
    """Import-time eager per-kernel timings at synthetic B=16 and B=32 x 80000 (page table from a
    global randperm, as reference.py builds it), cold L2 per rep. Diagnostics only: they never
    choose the path. Bytes = sum(seq_b) x 256 (K + V rows, both kv heads)."""
    dev = torch.device("cuda")
    gen = torch.Generator(device="cuda").manual_seed(99)
    usable = {0}
    for v in (1, 2):
        if _EXT2.sk_attrs(v)[6] < 1:
            _say(f"[sk-diag] variant {_SK_LABELS[v]} cannot launch (0 CTAs/SM): skipped")
            continue
        if _sk_case(f"diag variant check B=16 ragged", _SK_LENS16, 1.0, None, gen, variant=v):
            usable.add(v)
        else:
            _say(f"[sk-diag] variant {_SK_LABELS[v]} failed its correctness check: not timed")
    grid = 2 * _sm_count(dev)
    flush = torch.empty(64 << 20, dtype=torch.float32, device=dev)  # 256 MB > L2

    def timed(fns, warm=2, reps=10):
        per = [[] for _ in fns]
        for it in range(warm + reps):
            flush.zero_()
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(len(fns) + 1)]
            ev[0].record()
            for i, f in enumerate(fns):
                f()
                ev[i + 1].record()
            torch.cuda.synchronize()
            if it >= warm:
                for i in range(len(fns)):
                    per[i].append(ev[i].elapsed_time(ev[i + 1]) * 1e3)
        return [sorted(t)[len(t) // 2] for t in per]

    for batch in (16, 32):
        lens = [80000] * batch
        data = _splitkv_synthetic(lens, 1.0, gen, dev)
        _say(f"[sk-diag] B={batch}: launching _pre_attention_kernel")
        q, qkv = _sk_prepare_q(data)
        torch.cuda.synchronize()
        nbytes = sum(n + 1 for n in lens) * 256
        cfg = data["config"]
        scale = cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E
        kc, vc, pt = data["k_cache"], data["v_cache"], data["page_table"]
        seql, ksc, vsc = data["cache_seqlens"], data["k_scale"], data["v_scale"]
        max_chunks = -(-pt.shape[1] // _SPLITKV_CHUNK)
        w1a = torch.empty((batch * max_chunks, 16, 64), dtype=torch.float32, device=dev)
        w1m = torch.empty((batch * max_chunks, 2, 16), dtype=torch.float32, device=dev)
        wa = torch.empty((grid + batch, 16, 64), dtype=torch.float32, device=dev)
        wm = torch.empty((grid + batch, 2, 16), dtype=torch.float32, device=dev)
        outs = {k: torch.empty((batch, 1, 1024), dtype=torch.bfloat16, device=dev) for k in ("s1", "nt", "t4", "t5")}
        tk = _sk_ticket(dev)
        gdum_a = torch.empty((1, 16, 64), dtype=torch.float32, device=dev)  # unused below variant 3
        gdum_m = torch.empty((1, 2, 16), dtype=torch.float32, device=dev)

        def bw(us):
            return nbytes / (us * 1e-6) / 1e12

        def s1a():
            _EXT.splitkv_attn(q, kc, vc, pt, seql, ksc, w1a, w1m, scale, max_chunks, grid)

        def s1b():
            _EXT.splitkv_combine(w1a, w1m, seql, vsc, qkv, qkv.stride(0), outs["s1"], max_chunks, RMS_EPS)

        def sk(variant, key):
            return lambda: _EXT2.sk_attn(q, kc, vc, pt, seql, ksc, vsc, qkv, qkv.stride(0), outs[key], wa, wm, tk,
                                         _sk_gticket(dev), gdum_a, gdum_m, scale, RMS_EPS, grid, variant)

        def skc():
            _EXT2.sk_combine(wa, wm, seql, vsc, qkv, qkv.stride(0), outs["nt"], grid, RMS_EPS)

        def rel(key):
            ref = outs["s1"].float()
            return float((outs[key].float() - ref).norm() / ref.norm())

        _say(f"[sk-diag] B={batch}: timing stage-1 A + B")
        a1, b1 = timed([s1a, s1b])
        _say(f"[sk-diag] B={batch} x 80000: {nbytes / 1e6:.0f} MB of K+V rows, grid {grid}, eager, cold L2, "
             f"median of 10")
        _say(f"[sk-diag]   stage-1 A splitkv_attn_kernel          {a1:7.1f} us  {bw(a1):.2f} TB/s")
        _say(f"[sk-diag]   stage-1 B splitkv_combine_kernel       {b1:7.1f} us  -> A+B {a1 + b1:7.1f} us")
        t4 = timed([sk(0, "t4")])[0]
        _say(f"[sk-diag]   stream-K + ticket, 4 st (timed path)   {t4:7.1f} us  {bw(t4):.2f} TB/s  "
             f"(vs stage-1 out rel {rel('t4'):.1e})")
        nt = ntc = t5 = None
        if 1 in usable:
            _say(f"[sk-diag] B={batch}: timing stream-K no ticket + combine")
            nt, ntc = timed([sk(1, "nt"), skc])
            _say(f"[sk-diag]   stream-K no ticket, 4 st           {nt:7.1f} us  {bw(nt):.2f} TB/s  "
                 f"+ combine {ntc:.1f} us -> {nt + ntc:.1f} us  (rel {rel('nt'):.1e})")
        if 2 in usable:
            _say(f"[sk-diag] B={batch}: timing stream-K + ticket, 5 stages")
            t5 = timed([sk(2, "t5")])[0]
            _say(f"[sk-diag]   stream-K + ticket, 5 st            {t5:7.1f} us  {bw(t5):.2f} TB/s  "
                 f"(rel {rel('t5'):.1e})")
        summary = f"[sk-diag] B={batch} ratios (>1 = first is slower): stage-1 A+B / v2 {(a1 + b1) / t4:.3f}x"
        if nt is not None:
            summary += (f"; split/locality: stage-1 A / stream-K no-ticket A {a1 / nt:.3f}x"
                        f"; ticket fold: (no-ticket A + combine) / ticket {(nt + ntc) / t4:.3f}x")
        if t5 is not None:
            summary += f"; stages: 4 st / 5 st {t4 / t5:.3f}x"
        _say(summary)
        del data, q, qkv, kc, vc, pt, w1a, w1m, wa, wm, outs
        torch.cuda.empty_cache()
    del flush
    torch.cuda.empty_cache()


def _sk_validate_b1() -> "str | None":
    """In-process gate for stage 3 (batch 1, two-level ticket, grid = SM count). Requires stage 2
    validated. Returns None on success, else the reason batch 1 stays on master FA3."""
    global _B1_PATH
    dev = torch.device("cuda")
    sms = _sm_count(dev)
    a = list(_EXT2.sk_attrs(3))
    _say(f"[sk3] splitkv_attn_sk_kernel{_SK_LABELS[3]}: binaryVersion {a[0]}, ptxVersion {a[1]}, regs {a[2]}, "
         f"local {a[3]} B, dyn smem {a[5]} B, {a[6]} CTAs/SM")
    if a[0] != 90:
        return "cubin version is not 90 (two-level variant)"
    if a[6] < 1:
        return "two-level variant cannot launch (0 CTAs/SM)"
    if a[6] < 2:
        _say("[sk3] WARNING: the two-level variant does not reach 2 CTAs/SM (performance, not correctness)")
    gen = torch.Generator(device="cuda").manual_seed(8642)
    cases = (
        ("B=1 80000 keys grid SM", [80000], 1.0, sms),
        ("B=1 80000 keys kv 0.25 grid SM", [80000], 0.25, sms),
        ("B=1 80000 keys grid 7", [80000], 1.0, 7),
        ("B=1 80000 keys grid 3", [80000], 0.25, 3),
        ("B=1 80000 keys grid 2xSM", [80000], 1.0, 2 * sms),
        ("B=1 4097 keys grid SM", [4096], 1.0, sms),
        ("B=1 4097 keys kv 0.25 grid 7", [4096], 0.25, 7),
        ("B=1 63 keys grid SM", [62], 1.0, sms),
        ("B=1 63 keys kv 0.25 grid 3", [62], 0.25, 3),
        ("B=1 1 key grid SM", [0], 1.0, sms),
        ("B=1 1 key kv 0.25 grid 3", [0], 0.25, 3),
        ("B=16 ragged two-level grid device", _SK_LENS16, 0.25, 2 * sms),
    )
    for tag, lens, kv_scale, grid in cases:
        if not _sk_case(tag, lens, kv_scale, grid, gen, variant=3):
            return f"self-check {tag}"
    _B1_PATH = True  # kernel_fn must take the batch-1 path for the replay checks below
    try:
        for lens, kv_scale in (([80000], 1.0), ([4096], 0.25)):
            reason = _sk_replay_check(gen, lens, kv_scale)
            if reason is not None:
                return reason
    finally:
        _B1_PATH = False
    left = int(_sk_ticket(dev).abs().sum().item()) + int(_sk_gticket(dev).abs().sum().item())
    _say(f"[sk3] tickets (both levels) after the self-check: sum {left}")
    if left:
        return "two-level tickets not reset"
    return None


def _sk_b1_diagnostics() -> None:
    """Import-time eager timings at synthetic B=1 x 80000 (cold L2 per rep, median of 10) bracketing
    the two fixed stage-3 choices: (a) stage-2 kernel as is (grid 2xSM, single-level) = the
    widen-gate-only control, (b) grid SM single-level, (c) grid SM two-level (the timed path),
    (d) grid 2xSM two-level; plus master FA3 main + combine + norm-gate on the same input and the
    two-level kernel at B=16 (diagnostic only). Prints only: the timed path is fixed by shape."""
    dev = torch.device("cuda")
    sms = _sm_count(dev)
    gen = torch.Generator(device="cuda").manual_seed(77)
    flush = torch.empty(64 << 20, dtype=torch.float32, device=dev)  # 256 MB > L2

    def timed(fns, warm=2, reps=10):
        per = [[] for _ in fns]
        for it in range(warm + reps):
            flush.zero_()
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(len(fns) + 1)]
            ev[0].record()
            for i, f in enumerate(fns):
                f()
                ev[i + 1].record()
            torch.cuda.synchronize()
            if it >= warm:
                for i in range(len(fns)):
                    per[i].append(ev[i].elapsed_time(ev[i + 1]) * 1e3)
        return [sorted(t)[len(t) // 2] for t in per]

    for batch, variants in ((1, ((0, 2 * sms, "(a) single-level grid 2xSM (stage 2 as is)"),
                                 (0, sms, "(b) single-level grid SM"),
                                 (3, sms, "(c) two-level grid SM (timed path)"),
                                 (3, 2 * sms, "(d) two-level grid 2xSM"))),
                            (16, ((3, 2 * sms, "two-level grid 2xSM at B=16 (diagnostic)"),
                                  (0, 2 * sms, "single-level grid 2xSM at B=16 (stage 2)")))):
        lens = [80000] * batch
        data = _splitkv_synthetic(lens, 1.0, gen, dev)
        _say(f"[sk3-diag] B={batch}: launching _pre_attention_kernel")
        q, qkv = _sk_prepare_q(data)
        torch.cuda.synchronize()
        nbytes = sum(n + 1 for n in lens) * 256
        cfg = data["config"]
        scale = cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E
        kc, vc, pt = data["k_cache"], data["v_cache"], data["page_table"]
        seql, ksc, vsc = data["cache_seqlens"], data["k_scale"], data["v_scale"]
        tk, gtk = _sk_ticket(dev), _sk_gticket(dev)
        ws = {}
        fns, labels, outs = [], [], []
        for v, grid, label in variants:
            wa = torch.empty((grid + batch, 16, 64), dtype=torch.float32, device=dev)
            wm = torch.empty((grid + batch, 2, 16), dtype=torch.float32, device=dev)
            ng = batch * 32 if v == 3 else 1
            ga = torch.empty((ng, 16, 64), dtype=torch.float32, device=dev)
            gm = torch.empty((ng, 2, 16), dtype=torch.float32, device=dev)
            out = torch.empty((batch, 1, 1024), dtype=torch.bfloat16, device=dev)
            ws[label] = (wa, wm, ga, gm)
            outs.append(out)
            labels.append(label)
            fns.append(lambda v=v, grid=grid, wa=wa, wm=wm, ga=ga, gm=gm, out=out: _EXT2.sk_attn(
                q, kc, vc, pt, seql, ksc, vsc, qkv, qkv.stride(0), out, wa, wm, tk, gtk, ga, gm,
                scale, RMS_EPS, grid, v))
        _say(f"[sk3-diag] B={batch} x 80000: {nbytes / 1e6:.0f} MB of K+V rows, eager, cold L2, median of 10")
        times = timed(fns)
        ref = _splitkv_reference(data, lens)
        for label, t, out in zip(labels, times, outs):
            err, finite = _rel_l2(out, ref)
            _say(f"[sk3-diag]   {label:46s} {t:7.1f} us  {nbytes / (t * 1e-6) / 1e12:.2f} TB/s  "
                 f"(rel L2 {err:.1e}, finite {finite})")
        if batch == 1:
            fastest = min(range(len(times)), key=times.__getitem__)
            _say(f"[sk3-diag] fastest of the four: {labels[fastest]}; timed path (c) is {times[2]:.1f} us, "
                 f"(a) control {times[0]:.1f} us, (c)/(a) {times[2] / times[0]:.3f}x")
            if fastest != 2:
                _say("[sk3-diag] WARNING: the static choice (c) is not the fastest in this run; "
                     "the path stays fixed by shape (the order is a prepare-time ranking only)")
            try:
                def fa3():
                    return flash_attn_with_kvcache(
                        q=q, k_cache=kc, v_cache=vc, page_table=pt, cache_seqlens=seql,
                        cu_seqlens_q=data["cu_seqlens_q"], max_seqlen_q=1,
                        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True,
                        window_size=(-1, -1), scheduler_metadata=None)

                holder = {}

                def fa3_main():
                    o = fa3()
                    holder["o"] = o[0] if isinstance(o, tuple) else o

                def fa3_norm():
                    gate = qkv[:, (16 + 4) * 64:]
                    holder["r"] = rms_norm_gate(holder["o"].reshape(1, 1024).to(torch.bfloat16), gate, None,
                                                head_dim=64, eps=RMS_EPS, has_norm=True)

                _say("[sk3-diag] B=1: timing master FA3 (main + combine) and norm-gate")
                t_fa3, t_ng = timed([fa3_main, fa3_norm])
                _say(f"[sk3-diag]   master FA3 main+combine {t_fa3:7.1f} us + norm-gate {t_ng:.1f} us = "
                     f"{t_fa3 + t_ng:.1f} us (pre-attention kernel excluded from both sides)")
            except Exception as e:
                _say(f"[sk3-diag] master FA3 timing raised (path unaffected): {e!r}")
        del data, q, qkv, kc, vc, pt, ws, outs, fns
        torch.cuda.empty_cache()
    del flush
    torch.cuda.empty_cache()


if torch.cuda.is_available():
    _EXT = _build_splitkv()
    if _EXT is not None and not _splitkv_self_check():
        print("[splitkv] SELF-CHECK FAILED: CUDA split-KV path disabled, master FA3 path kept")
        _EXT = None
    if _EXT is None:
        _say("PATH CHOSEN: full decode 1<B<=64 tp8 page 1: master FA3 (splitkv v1 unavailable; v2 not tried)")
    else:
        _sk_ticket(torch.device("cuda", torch.cuda.current_device()))  # zeroed once, here
        _sk_gticket(torch.device("cuda", torch.cuda.current_device()))  # zeroed once, here
        _EXT2 = _build_sk()
        _sk_reason = "build failed" if _EXT2 is None else None
        if _EXT2 is not None:
            try:
                _sk_reason = _sk_validate()
            except Exception as e:
                _sk_reason = f"self-check raised {e!r}"
        _b1_reason = "stage 2 unavailable"
        if _sk_reason is not None:
            _EXT2 = None
            _say(f"PATH CHOSEN: full decode splitkv v1 (fallback: {_sk_reason})")
        else:
            _say("PATH CHOSEN: full decode streamk+ticket v2")
            try:
                _b1_reason = _sk_validate_b1()
            except Exception as e:
                _b1_reason = f"batch-1 self-check raised {e!r}"
            if _b1_reason is None:
                _B1_PATH = True
            try:
                _sk_diagnostics()
                if _b1_reason is None:
                    _sk_b1_diagnostics()
            except Exception as e:
                _say(f"[sk-diag] diagnostics raised (path unaffected): {e!r}")
            finally:  # a diagnostic variant must not leave the shared tickets dirty for the timed path
                try:
                    _sk_ticket(torch.device("cuda", torch.cuda.current_device())).zero_()
                    _sk_gticket(torch.device("cuda", torch.cuda.current_device())).zero_()
                    torch.cuda.synchronize()
                except Exception as e:
                    _say(f"[sk-diag] ticket re-zero raised: {e!r}")
        if _B1_PATH:
            _say("PATH CHOSEN: full decode_1 streamk two-level")
        else:
            _say(f"PATH CHOSEN: full decode_1 FA3 (fallback: {_b1_reason})")
    if _EXT is None:
        _say("PATH CHOSEN: full decode_1 FA3 (fallback: splitkv v1 unavailable)")
    _say("PATH CHOSEN: every other shape (sliding, verify, page 64, dp): master path, unchanged")
