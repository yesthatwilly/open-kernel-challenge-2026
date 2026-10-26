"""numpy model of _b12_encode + b12_unpack_kernel + b12_kernel's epilogue (m == 1)."""
import numpy as np
ESCB = 2048; CAP = (ESCB - 256) // 8

def to_bf16_bits(f32):
    u = f32.astype(np.float32).view(np.uint32).astype(np.uint64)
    r = ((u >> 16) & 1) + 0x7FFF
    return ((u + r) >> 16).astype(np.int64) & 0xFFFF

def bf16_val(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)

def encode(bits, r):
    n, k = bits.shape
    ctas, ckk = n // r, 8192 // r
    nch, ns, mt = k // ckk, ckk // 16, r // 16
    hi, lo = bits >> 8, bits & 0xFF
    e71, sign = hi & 0x7F, hi >> 7
    hist = np.bincount(e71.ravel(), minlength=128)
    win = np.array([hist[b:b + 8].sum() for b in range(121)])
    base = int(win.argmax())
    esc = (e71 < base) | (e71 > base + 7)
    code = np.where(esc, 0, (sign << 3) | (e71 - base))
    lo_enc = np.where(esc, 0, lo)
    dec = ((code + base + 120 * (code >> 3)) << 8) | lo_enc
    assert (dec[esc] == (base << 8)).all()
    dec = np.where(esc, bits, dec)
    assert (dec == bits).all()
    ln = np.arange(32); g, q = ln >> 2, ln & 3
    rows_l = np.stack([g, g, g + 8, g + 8, g, g, g + 8, g + 8], 1)
    cols_l = np.stack([2*q, 2*q+1, 2*q, 2*q+1, 2*q+8, 2*q+9, 2*q+8, 2*q+9], 1)
    def frag(t):
        t6 = t.reshape(ctas, mt, 16, nch, ns, 16).transpose(0, 3, 4, 1, 2, 5)
        return t6[..., rows_l, cols_l]
    lf = frag(lo_enc).astype(np.uint8).reshape(ctas, nch, ns, mt, 256)
    cf = frag(code)
    cwd = (cf[..., 0:4] | (cf[..., 4:8] << 4)).astype(np.uint8).reshape(ctas, nch, ns, mt, 128)
    wc = np.concatenate([lf, cwd], -1).reshape(ctas, nch, ns * mt * 384)
    idx = np.argwhere(esc)
    n_esc = len(idx)
    blocks = np.zeros((ctas, ESCB), np.uint8)
    per_cta = np.bincount(idx[:, 0] // r, minlength=ctas) if n_esc else np.zeros(ctas, np.int64)
    assert per_cta.max() <= CAP
    if n_esc:
        rr, kk = idx[:, 0], idx[:, 1]
        vals = bits[rr, kk]
        rloc = rr % r
        starts = np.cumsum(per_cta) - per_cta
        for c in np.nonzero(per_cta)[0]:
            s0, cnt = starts[c], per_cta[c]
            rl = rloc[s0:s0 + cnt]
            roff = np.zeros(r + 1, np.int64); roff[1:] = np.cumsum(np.bincount(rl, minlength=r))
            blocks[c, :2 * (r + 1)] = roff.astype(np.int16).view(np.uint8)
            ent = np.stack([(rl << 16) | kk[s0:s0 + cnt], vals[s0:s0 + cnt]], 1).astype(np.int32)
            blocks[c, 256:256 + 8 * cnt] = ent.reshape(-1).view(np.uint8)
    return wc, blocks, base, n_esc

def hi4(c4, base4):
    t = (c4 >> 3) & 0x01010101
    return (t * 0x78 + (c4 + base4)) & 0xFFFFFFFF

def byte_perm(x, y, s):
    b = [(x >> (8*i)) & 0xFF for i in range(4)] + [(y >> (8*i)) & 0xFF for i in range(4)]
    out = 0
    for i in range(4):
        out |= b[(s >> (4*i)) & 7] << (8*i)
    return out

def frag_dec(l0, l1, cw, base4):
    h0 = hi4(cw & 0x0F0F0F0F, base4); h1 = hi4((cw >> 4) & 0x0F0F0F0F, base4)
    return [byte_perm(l0, h0, 0x5140), byte_perm(l0, h0, 0x7362), byte_perm(l1, h1, 0x5140), byte_perm(l1, h1, 0x7362)]

def unpack(wc, n, k, r, base):
    ckk = 8192 // r; nch, ns, mt = k // ckk, ckk // 16, r // 16
    out = np.full((n, k), -1, np.int64)
    base4 = base * 0x01010101
    for cta in range(n // r):
        for j in range(nch):
            ch = wc[cta, j]
            for ks in range(ns):
                for m in range(mt):
                    t = ch[(ks * mt + m) * 384:(ks * mt + m + 1) * 384]
                    for lane in range(32):
                        l0 = int(t[lane*8:lane*8+4].view(np.uint32)[0]); l1 = int(t[lane*8+4:lane*8+8].view(np.uint32)[0])
                        c = int(t[256 + lane*4:256 + lane*4 + 4].view(np.uint32)[0])
                        a = frag_dec(l0, l1, c, base4)
                        g, q = lane >> 2, lane & 3
                        r0 = cta * r + m * 16 + g; k0 = j * ckk + ks * 16 + 2 * q
                        for i in range(4):
                            rr = r0 + (8 if i & 1 else 0); kk = k0 + (8 if i & 2 else 0)
                            out[rr, kk] = a[i] & 0xFFFF; out[rr, kk + 1] = a[i] >> 16
    return out

def gemv(wc, blocks, x_bits, n, k, r, base, cw_warps):
    """the kernel's arithmetic for m == 1: per warp partials over its k16 steps (fp32), epilogue in order + escapes"""
    ckk = 8192 // r; nch, ns, mt = k // ckk, ckk // 16, r // 16
    ksw = ns // cw_warps
    dec = unpack(wc, n, k, r, base)
    w = bf16_val(dec)
    xv = bf16_val(x_bits)
    out = np.zeros(n, np.float32)
    wfloor = np.uint32(base << 24).view(np.float32)
    for cta in range(n // r):
        rows = slice(cta * r, (cta + 1) * r)
        part = np.zeros((cw_warps, r), np.float32)
        for c in range(cw_warps):
            ks = [j * ckk + (c * ksw + i) * 16 for j in range(nch) for i in range(ksw)]
            cols = np.concatenate([np.arange(s, s + 16) for s in ks])
            part[c] = (w[rows][:, cols].astype(np.float64) @ xv[cols].astype(np.float64)).astype(np.float32)
        s = part[0].copy()
        for c in range(1, cw_warps): s = (s + part[c]).astype(np.float32)
        blk = blocks[cta]
        roff = blk[:2 * (r + 1)].view(np.uint16)
        ent = blk[256:].view(np.uint32)
        for row in range(r):
            acc = np.float32(s[row])
            for e in range(roff[row], roff[row + 1]):
                pos, val = int(ent[2*e]), int(ent[2*e+1])
                xe = xv[pos & 0xFFFF]; wt = np.uint32(val << 16).view(np.float32)
                assert (pos >> 16) == row
                acc = np.float32(np.float64(xe) * np.float64(wt) + np.float64(acc))
                acc = np.float32(np.float64(-xe) * np.float64(wfloor) + np.float64(acc))
            out[cta * r + row] = acc
    return out

rng = np.random.default_rng(5)
for (n, k, r) in ((64, 1024, 32), (32, 2048, 16), (128, 1024, 64)):
    w = rng.standard_normal((n, k)).astype(np.float32) / np.sqrt(k)
    w.ravel()[rng.choice(n * k, 6, replace=False)] = 0.0              # zeros: escapes
    w.ravel()[rng.choice(n * k, 4, replace=False)] *= 1e-6            # tiny: escapes
    w.ravel()[rng.choice(n * k, 2, replace=False)] = 3.0              # huge: escapes above the window
    bits = to_bf16_bits(w)
    wc, blocks, base, n_esc = encode(bits, r)
    dec = unpack(wc, n, k, r, base)
    hi = (bits >> 8) & 0x7F
    escm = (hi < base) | (hi > base + 7)
    assert (dec[~escm] == bits[~escm]).all(), "device round trip"
    assert (dec[escm] == (base << 8)).all(), "escape slots"
    x = rng.standard_normal(k).astype(np.float32)
    xb = to_bf16_bits(x)
    got = gemv(wc, blocks, xb, n, k, r, base, 8 if (k // (8192 // r)) else 4)
    ref = bf16_val(bits).astype(np.float64) @ bf16_val(xb).astype(np.float64)
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    # probe: x a single 0.5 at an escape column, if any
    print(f"n={n} k={k} r={r}: base {base} escapes {n_esc} ({escm.sum()}), wc {wc.size} B vs raw {n*k*2} B, rel L2 {rel:.2e}")
    if n_esc:
        rr, kk = np.argwhere(escm)[0]
        xp = np.zeros(k, np.float32); xp[kk] = 0.5
        gp = gemv(wc, blocks, to_bf16_bits(xp), n, k, r, base, 8)
        exp = 0.5 * bf16_val(bits)[:, kk]
        print("  probe col", kk, "max abs err", np.abs(gp - exp).max(), "row err at escape", gp[rr] - exp[rr])
print("all ok")
