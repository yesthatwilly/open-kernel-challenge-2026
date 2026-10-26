"""Independent numpy emulation of section (10): encoder (line-by-line translation of _b12_encode),
kernel decode (b12_hi4 / b12_frag), PTX m16n8k16 A/B/D fragment maps (written from the PTX ISA
tables, not from the encoder), escape-block parse and the m == 1 epilogue."""
import sys
import numpy as np

ESC_BYTES, ESC_CAP = 2048, (2048 - 256) // 8


def bf16_bits(f32):
    u = f32.astype(np.float32).view(np.uint32).astype(np.uint64)
    bias = ((u >> 16) & 1) + 0x7FFF
    return ((u + bias) >> 16).astype(np.uint16)


def bits_to_f32(b16):
    return (b16.astype(np.uint32) << 16).view(np.float32)


def encode(wb, r):
    n, k = wb.shape
    ctas, ckk = n // r, 8192 // r
    nch, ns, mt = k // ckk, ckk // 16, r // 16
    bits = wb.view(np.int16).astype(np.int32) & 0xFFFF
    hi, lo = bits >> 8, bits & 0xFF
    e71, sign = hi & 0x7F, hi >> 7
    hist = np.bincount(e71.ravel(), minlength=128)
    base = int(np.lib.stride_tricks.sliding_window_view(hist, 8).sum(1).argmax())
    assert 1 <= base <= 120, base
    esc = (e71 < base) | (e71 > base + 7)
    code = np.where(esc, 0, (sign << 3) | (e71 - base))
    lo_enc = np.where(esc, 0, lo)
    dec = ((code + base + 120 * (code >> 3)) << 8) | lo_enc
    assert (dec[esc] == (base << 8)).all()
    dec = np.where(esc, bits, dec)
    assert np.array_equal(dec, bits)
    ln = np.arange(32)
    g, q = ln >> 2, ln & 3
    rows_l = np.stack([g, g, g + 8, g + 8, g, g, g + 8, g + 8], 1)
    cols_l = np.stack([2 * q, 2 * q + 1, 2 * q, 2 * q + 1, 2 * q + 8, 2 * q + 9, 2 * q + 8, 2 * q + 9], 1)

    def frag(t):
        t6 = t.reshape(ctas, mt, 16, nch, ns, 16).transpose(0, 3, 4, 1, 2, 5)
        return t6[..., rows_l, cols_l]

    lf = frag(lo_enc).astype(np.uint8).reshape(ctas, nch, ns, mt, 256)
    cf = frag(code)
    cwd = (cf[..., 0:4] | (cf[..., 4:8] << 4)).astype(np.uint8).reshape(ctas, nch, ns, mt, 128)
    wc = np.ascontiguousarray(np.concatenate([lf, cwd], -1).reshape(ctas, nch, ns * mt * 384))
    idx = np.argwhere(esc)
    n_esc = idx.shape[0]
    blocks = np.zeros((ctas, ESC_BYTES), dtype=np.uint8)
    per_cta = np.bincount(idx[:, 0] // r, minlength=ctas) if n_esc else np.zeros(ctas, np.int64)
    max_cta = int(per_cta.max())
    if max_cta > ESC_CAP:
        raise RuntimeError(f"overflow {max_cta}")
    if n_esc:
        rr, kk = idx[:, 0].astype(np.int64), idx[:, 1].astype(np.int64)
        vals = bits[idx[:, 0], idx[:, 1]]
        rloc = rr % r
        starts = np.cumsum(per_cta) - per_cta
        for c in np.nonzero(per_cta)[0].tolist():
            s0, cnt = int(starts[c]), int(per_cta[c])
            rl = rloc[s0:s0 + cnt]
            roff = np.zeros(r + 1, dtype=np.int64)
            roff[1:] = np.cumsum(np.bincount(rl, minlength=r))
            blocks[c, :2 * (r + 1)] = roff.astype(np.int16).view(np.uint8)
            ent = np.stack([(rl << 16) | kk[s0:s0 + cnt], vals[s0:s0 + cnt].astype(np.int64)], 1).astype(np.int32)
            blocks[c, 256:256 + 8 * cnt] = ent.reshape(-1).view(np.uint8)
    return wc, blocks.reshape(-1), base, n_esc, max_cta, per_cta


def u32(b):  # [..., 4] uint8 -> uint32 little-endian
    b = b.astype(np.uint32)
    return b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16) | (b[..., 3] << 24)


def byte_perm(x, y, s):
    bytes_ = [(x >> (8 * i)) & 0xFF for i in range(4)] + [(y >> (8 * i)) & 0xFF for i in range(4)]
    out = np.zeros_like(x)
    for i in range(4):
        sel = (s >> (4 * i)) & 0x7
        out |= bytes_[sel] << (8 * i)
    return out


def hi4(c4, base4):
    t = (c4 >> 3) & 0x01010101
    return (t * 0x78 + (c4 + base4)) & 0xFFFFFFFF


def kernel_decode(wc, n, k, r, base):
    """Reconstruct W_dec[n, k] (uint16 bits) from the stream using the kernel's loads and the PTX
    A-fragment table (element i of a lane: row g (+8 if i in 2,3,6,7), col 2q + (i&1) (+8 if i >= 4))."""
    ckk = 8192 // r
    nch, ns, mt = k // ckk, ckk // 16, r // 16
    ctas = n // r
    tiles = wc.reshape(ctas, nch, ns, mt, 384)
    lane = np.arange(32)
    lbytes = tiles[..., :256].reshape(ctas, nch, ns, mt, 32, 8)
    cbytes = tiles[..., 256:].reshape(ctas, nch, ns, mt, 32, 4)
    l0, l1, cw = u32(lbytes[..., 0:4]), u32(lbytes[..., 4:8]), u32(cbytes)
    base4 = np.uint32(base * 0x01010101)
    h0 = hi4(cw & 0x0F0F0F0F, base4)
    h1 = hi4((cw >> 4) & 0x0F0F0F0F, base4)
    regs = [byte_perm(l0, h0, 0x5140), byte_perm(l0, h0, 0x7362),
            byte_perm(l1, h1, 0x5140), byte_perm(l1, h1, 0x7362)]
    out = np.zeros((n, k), dtype=np.uint16)
    g, q = lane >> 2, lane & 3
    cta_i = np.arange(ctas)[:, None, None, None, None]
    j_i = np.arange(nch)[None, :, None, None, None]
    ks_i = np.arange(ns)[None, None, :, None, None]
    mt_i = np.arange(mt)[None, None, None, :, None]
    for reg in range(4):
        for half in range(2):
            i = 2 * reg + half                       # PTX element index a_i
            row = g + (8 if i in (2, 3, 6, 7) else 0)
            col = 2 * q + (i & 1) + (8 if i >= 4 else 0)
            val = (regs[reg] >> (16 * half)) & 0xFFFF
            R_ = cta_i * r + mt_i * 16 + row[None, None, None, None, :]
            C_ = j_i * ckk + ks_i * 16 + col[None, None, None, None, :]
            out[R_, C_] = val.astype(np.uint16)
    return out


def epilogue(wdec_bits, blocks, x_bits, n, k, r, base):
    """m == 1: out[n] = sum_k x[k] * w_dec[n, k] (fp64 here) + per-row escapes from the block."""
    xf = bits_to_f32(x_bits).astype(np.float64)
    acc = bits_to_f32(wdec_bits).astype(np.float64) @ xf
    wfloor = float(np.array([base << 24], dtype=np.uint32).view(np.float32)[0])
    ctas = n // r
    blk = blocks.reshape(ctas, ESC_BYTES)
    for c in range(ctas):
        roff = blk[c, :2 * (r + 1)].view(np.uint16)
        ent = blk[c, 256:].view(np.uint32)
        for t in range(r):
            for e in range(int(roff[t]), int(roff[t + 1])):
                pos, val = int(ent[2 * e]), int(ent[2 * e + 1])
                assert (pos >> 16) == t, (pos >> 16, t)
                kk = pos & 0xFFFF
                wt = float(np.array([val << 16], dtype=np.uint32).view(np.float32)[0])
                acc[c * r + t] += xf[kk] * wt - xf[kk] * wfloor
    return acc


def run(name, n, k, r, seed, full):
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((n, k), dtype=np.float32) / np.float32(np.sqrt(k))).astype(np.float32)
    wb = bf16_bits(w)
    wc, blocks, base, n_esc, max_cta, per_cta = encode(wb, r)
    print(f"{name}: N={n} K={k} R={r} base={base} escapes={n_esc} ({n_esc / (n * k):.2e}) "
          f"mean/CTA={per_cta.mean():.1f} max/CTA={max_cta} cap={ESC_CAP} wc={wc.size} (=1.5NK: {wc.size == n * k * 3 // 2})",
          flush=True)
    if not full:
        return
    dec = kernel_decode(wc, n, k, r, base)
    bits = wb
    hi = (bits.view(np.int16).astype(np.int32) >> 8) & 0x7F
    escm = (hi < base) | (hi > base + 7)
    assert np.array_equal(dec[~escm], bits[~escm]), "decode mismatch"
    assert (dec[escm] == (base << 8)).all(), "escape slot mismatch"
    x = bf16_bits(rng.standard_normal(k, dtype=np.float32))
    out = epilogue(dec, blocks, x, n, k, r, base)
    ref = bits_to_f32(bits).astype(np.float64) @ bits_to_f32(x).astype(np.float64)
    rel = np.linalg.norm(out - ref) / np.linalg.norm(ref)
    print(f"  decode bit-exact on non-escapes, escapes at w_floor; epilogue fp64 rel err {rel:.2e}", flush=True)
    assert rel < 1e-12


if __name__ == "__main__":
    full = "--full" in sys.argv
    for seed in (1, 2, 3):
        run("o", 8192, 1024, 64, seed, full)
        run("qkv_gate", 2304, 8192, 16, seed, full)
        run("pooler", 4096, 8192, 32, seed, full)
