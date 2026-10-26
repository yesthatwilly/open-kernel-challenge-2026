"""b12x_kernel instances 6..17: smem layout / static_asserts, and epilogue shared-memory wavefronts
(bank conflicts) for the current scratch layout [cw][R][NT] vs a transposed padded [cw][NT][R+4]."""
ESC = 2048
CAP = (ESC - 256) // 8
LIMIT = 232448
INST = {6: (32, 8192, 8, 1, 8, 8), 7: (32, 8192, 8, 1, 4, 8), 8: (16, 8192, 4, 2, 4, 8), 9: (16, 8192, 4, 2, 8, 8),
        10: (64, 1024, 8, 1, 8, 8), 11: (64, 1024, 8, 1, 4, 8), 12: (32, 8192, 8, 1, 8, 16),
        13: (32, 8192, 6, 1, 8, 32), 14: (64, 4096, 8, 1, 8, 16), 15: (64, 4096, 8, 1, 8, 32),
        16: (64, 1024, 8, 1, 8, 16), 17: (64, 1024, 8, 1, 8, 32)}


def wavefronts(addrs):
    """4-byte word addresses of one warp access -> number of shared-memory wavefronts (distinct words per bank)."""
    banks = {}
    for a in addrs:
        banks.setdefault(a % 32, set()).add(a)
    return max(len(v) for v in banks.values())


for inst, (R, K, S, MINB, CW, NT) in INST.items():
    CKK = 8192 // R
    NS, MT, NN = CKK // 16, R // 16, NT // 8
    WB = NS * MT * 384
    XS = CKK * 2 + 16
    SLOT = WB + NT * XS
    RING = 256 + ESC + CAP * NT * 4
    smem = RING + S * SLOT
    smem_py = 256 + ESC + CAP * NT * 4 + S * (12288 + NT * ((8192 // R) * 2 + 16))
    ok = (K % CKK == 0 and NS % CW == 0 and WB == 12288 and RING % 128 == 0 and SLOT % 128 == 0 and XS % 16 == 0
          and CW * R * NT * 4 <= S * SLOT and 2 * (R + 1) <= 256)
    occ_need = MINB * (smem + 1024) <= 233472
    m = NT
    threads = 32 * (CW + 1)
    # current write: scratch[(cw*R + row)*NT + tok]
    w_cur = 0
    for mt in range(MT):
        for nn in range(NN):
            for c in range(4):
                addrs = [((0 * R + mt * 16 + (l >> 2) + (c >> 1) * 8) * NT + nn * 8 + (l & 3) * 2 + (c & 1)) for l in range(32)]
                w_cur += wavefronts(addrs)
    w_cur *= CW
    # current read: per idx (tok = idx / R, row = idx % R), CW loads scratch[(c*R + row)*NT + tok]
    r_cur = 0
    for w0 in range(0, R * m, 32):
        idxs = [i for i in range(w0, min(w0 + 32, R * m))]
        for c in range(CW):
            r_cur += wavefronts([((c * R + i % R) * NT + i // R) for i in idxs])
    # proposed: scratch[(cw*NT + tok)*RP + row], RP = R + 4
    RP = R + 4
    w_new = 0
    for mt in range(MT):
        for nn in range(NN):
            for c in range(4):
                addrs = [((0 * NT + nn * 8 + (l & 3) * 2 + (c & 1)) * RP + mt * 16 + (l >> 2) + (c >> 1) * 8) for l in range(32)]
                w_new += wavefronts(addrs)
    w_new *= CW
    r_new = 0
    for w0 in range(0, R * m, 32):
        idxs = [i for i in range(w0, min(w0 + 32, R * m))]
        for c in range(CW):
            r_new += wavefronts([((c * NT + i // R) * RP + i % R) for i in idxs])
    fits_new = CW * NT * RP * 4 <= S * SLOT
    print(f"inst {inst:2d} R{R} K{K} S{S} MINB{MINB} CW{CW} NT{NT}: smem {smem} (py {smem_py}, <= {LIMIT}: "
          f"{smem <= LIMIT}), asserts {ok}, {MINB}/SM fits {occ_need}; epilogue wavefronts @m=NT: current "
          f"write {w_cur} + read {r_cur} = {w_cur + r_cur} (~{(w_cur + r_cur) / 1.9e3:.2f} us @1.9GHz); "
          f"[cw][NT][R+4]: {w_new} + {r_new} = {w_new + r_new} (fits {fits_new})")
