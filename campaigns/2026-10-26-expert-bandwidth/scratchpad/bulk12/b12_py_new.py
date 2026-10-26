def _b12_route_ok(layer: str, x: torch.Tensor, cfg, state: dict) -> bool:
    """bulk12 cell: layer in B12_CELLS with its prepare()-set flag, cfg m == 1 and the cell's n / k, x a contiguous
    16-byte-aligned bf16 [1, k]. cfg, shapes and state flags only."""
    spec = B12_CELLS.get(layer)
    if spec is None or not state.get(spec[5], False):
        return False
    n, k = spec[1], spec[2]
    return (cfg["m"] == B12_M and cfg["n"] == n and cfg["k"] == k
            and x.dim() == 2 and x.shape[0] == B12_M and x.shape[1] == k
            and x.dtype == torch.bfloat16 and x.is_contiguous() and x.data_ptr() % 16 == 0)


def _b12_forward(layer: str, x: torch.Tensor, state: dict) -> torch.Tensor:
    """ONE launch: out[1, n] = x @ W.T from the state's 12-bit weight, whole K per CTA, bf16 stored once."""
    out = torch.empty((1, B12_CELLS[layer][1]), dtype=torch.bfloat16, device=x.device)
    _B12_EXT.b12_run(state["b12_wc_" + layer], state["b12_esc_" + layer], x, out, state["b12_err"],
                     state["b12_inst_" + layer], state["b12_base_" + layer])
    return out


def _b12x_route_ok(layer: str, x: torch.Tensor, cfg, state: dict) -> bool:
    """bulk12 m == 8 cell: layer in B12X_CELLS with its prepare()-set flag, cfg m == 8 and the cell's n / k, x a
    contiguous 16-byte-aligned bf16 [8, k]. cfg, shapes and state flags only."""
    spec = B12X_CELLS.get(layer)
    if spec is None or not state.get(spec[5], False):
        return False
    n, k = spec[1], spec[2]
    return (cfg["m"] == B12X_M and cfg["n"] == n and cfg["k"] == k
            and x.dim() == 2 and x.shape[0] == B12X_M and x.shape[1] == k
            and x.dtype == torch.bfloat16 and x.is_contiguous() and x.data_ptr() % 16 == 0)


def _b12x_forward(layer: str, x: torch.Tensor, state: dict) -> torch.Tensor:
    """ONE launch: out[m, n] = x @ W.T (m == 8) from the state's 12-bit weight, x streamed with the chunks."""
    out = torch.empty((x.shape[0], B12X_CELLS[layer][1]), dtype=torch.bfloat16, device=x.device)
    _B12_EXT.b12x_run(state["b12_wc_" + layer], state["b12_esc_" + layer], x, out, state["b12_err"],
                      state["b12x_inst_" + layer], state["b12_base_" + layer])
    return out


def _b12_encode_into(state: dict, layer: str, w: torch.Tensor, rows: int, inst: int, verbose: bool) -> None:
    """Encode w into the state's buffers for this layer (shared by its m == 1 and m == 8 cells) and check the round
    trip on the device through the kernel's own decode (b12_unpack, with inst's R / K): every non-escape element
    bit-exact, every escape slot at w_floor's bits. Raises on any mismatch."""
    wc, esc, base, n_esc, max_cta = _b12_encode(w, rows)
    dec = torch.empty(tuple(w.shape), dtype=torch.int16, device=w.device)
    _B12_EXT.b12_unpack(wc, dec, inst, base)
    torch.cuda.synchronize()
    bits = w.contiguous().view(torch.int16)
    hi = (bits.to(torch.int32) >> 8) & 0x7F
    escm = (hi < base) | (hi > base + 7)
    if not torch.equal(dec[~escm], bits[~escm]):
        raise RuntimeError("device round trip (b12_unpack) differs from the weight")
    if not bool((dec[escm].to(torch.int32) == (base << 8)).all()):
        raise RuntimeError("device escape slots do not decode to w_floor")
    state["b12_wc_" + layer], state["b12_esc_" + layer], state["b12_base_" + layer] = wc, esc, base
    if verbose:
        el = w.numel()
        print(f"[b12] {layer}: base {base} (e71 window {base}..{base + 7}, binades {2 * base - 127}.."
              f"{2 * base + 15 - 127}), escapes {n_esc} ({n_esc / el:.2e} of {el}), max {max_cta} per CTA (cap "
              f"{B12_ESC_CAP}); stream {wc.numel() / 1e6:.2f} MB + escape blocks {esc.numel() / 1e6:.3f} MB vs "
              f"raw {el * 2 / 1e6:.2f} MB (x{(wc.numel() + esc.numel()) / (el * 2):.4f}); host + device round "
              f"trips bit-exact", flush=True)


def _b12_cell(state: dict, layer: str, spec, m: int, inst_key: str, fwd, dev):
    """One bulk12 cell (m == 1: b12_kernel, m == 8: b12x_kernel) on the layer's encoded weight: per instance the
    occupancy query, canary + correctness (_cell_check) and the graph gate against kernel_fn with the cell's flag off
    (_cell_gate); the fastest passing instance routes at <= CELL_MAX_RATIO after the kernel-name assertion. Leaves
    state[inst_key] at the chosen instance. Returns (routed, (cand_us, incumbent_us, inst, label) or None, why)."""
    src, n, k, rows, variants, flag = spec
    kname = "b12_kernel" if m == 1 else "b12x_kernel"
    best, why = None, ""
    w = state[layer]
    st_on = dict(state)
    st_on[flag] = True
    for inst, stages, minb, cw in variants:
        label = f"{kname}<{rows},{k},{stages},{minb},{cw}>"
        info = [int(v) for v in _B12_EXT.b12_setup(inst)]
        print(f"[b12] {src} {layer} {label}: regs={info[0]} local(spill)={info[1]} B static smem={info[2]} B dyn "
              f"smem={info[3]} B, occupancy {info[4]} CTA/SM (needs {minb}), grid {n // rows} x {32 * (cw + 1)} "
              f"threads", flush=True)
        if info[4] < minb:
            why = f"{label}: occupancy {info[4]} CTA/SM < {minb}"
            continue
        state[inst_key] = inst
        st_on[inst_key] = inst
        bad = _cell_check(f"{src} {label}", lambda xx: fwd(layer, xx, st_on), w, m, n, k, err=state["b12_err"])
        if bad:
            why = f"{label}: {bad}"
            continue
        tg = time.perf_counter()
        cand_us, inc_us = _cell_gate(f"{src} {label}", layer, m, n, k, flag, state)
        code = int(state["b12_err"].item())
        print(f"[prepare-time] gate {src} {label}: {time.perf_counter() - tg:.2f}s; cand {cand_us:.2f} us (raw-equivalent "
              f"{n * k * 2 / (cand_us * 1e-6) / 1e12:.2f} TB/s, stream {n * k * 1.5 / (cand_us * 1e-6) / 1e12:.2f} "
              f"TB/s) vs incumbent {inc_us:.2f} us", flush=True)
        if code:   # a fault in any instance keeps the incumbent for the whole cell
            return False, None, f"{label}: error word {code} after the gate"
        if best is None or cand_us < best[0]:
            best = (cand_us, inc_us, inst, label)
    if best is None:
        return False, None, why
    cand_us, inc_us, inst, label = best
    state[inst_key] = inst
    st_on[inst_key] = inst
    if not cand_us <= CELL_MAX_RATIO * inc_us:
        return False, best, f"gate: best {label} cand/incumbent {cand_us / inc_us:.3f} > {CELL_MAX_RATIO}"
    data = {"x": torch.randn((m, k), device=dev, dtype=torch.float32).to(torch.bfloat16),
            "config": {"m": m, "n": n, "k": k, "layer": layer}}
    try:
        names = _cell_kernels(lambda: kernel_fn(data, st_on))
        print(f"[b12] {src} {layer} m={m}: profiled kernels {names}", flush=True)
        if names and not (len(names) == 1 and kname in names[0]):
            return False, best, f"profiled kernels {names}: expected exactly one {kname}"
    except Exception as e:  # own extension, direct launch: keep the route
        print(f"[b12] {src} {layer}: profiler unavailable ({type(e).__name__}: {e}); route kept (kernel_fn calls "
              f"{kname} directly)", flush=True)
    return True, best, ""


def _b12_prepare(state: dict, flags=None) -> None:
    """A12 / A2_12 / A3_12 (m == 1) and A12x8 / A2_12x8 / A3_12x8 (m == 8). Sets each cell's flag (False unless every
    step passes) and, per layer with a routed cell, the state's own encoded weight; one per-state error word. flags
    None (the judge's state): encode + host / device round trips once per layer, then each cell through _b12_cell;
    one PATH CHOSEN line per cell. flags a dict (the A/B check's second state): inherit each flag and instance,
    encode only for layers with a cell on."""
    for spec in list(B12_CELLS.values()) + list(B12X_CELLS.values()):
        state[spec[5]] = False
    if _B12_EXT is None:
        if flags is None:
            for cells, m in ((B12_CELLS, B12_M), (B12X_CELLS, B12X_M)):
                for layer, spec in cells.items():
                    print(f"PATH CHOSEN {spec[0]} {layer} m={m}: incumbent (cand n/a us vs incumbent n/a us; "
                          f"no b12 module)", flush=True)
        return
    dev = state["router"].device
    state["b12_err"] = torch.zeros(1, dtype=torch.int32, device=dev)
    if flags is not None:
        i1, i8 = flags.get("_b12_inst", {}), flags.get("_b12x_inst", {})
        for layer, spec in B12_CELLS.items():
            spec8 = B12X_CELLS[layer]
            on1 = bool(flags.get(spec[5])) and i1.get(layer) is not None
            on8 = bool(flags.get(spec8[5])) and i8.get(layer) is not None
            if not (on1 or on8):
                continue
            try:   # a failure here turns off only this layer's cells in this state, never the whole A/B
                _b12_encode_into(state, layer, state[layer], spec[3], i1[layer] if on1 else i8[layer], verbose=False)
                if on1:
                    state["b12_inst_" + layer] = i1[layer]
                    state[spec[5]] = True
                if on8:
                    state["b12x_inst_" + layer] = i8[layer]
                    state[spec8[5]] = True
            except Exception as e:
                print(f"[b12] second state: {layer} encode raised {type(e).__name__}: {e}; cells off in this state",
                      flush=True)
        return
    for layer, spec in B12_CELLS.items():
        spec8 = B12X_CELLS[layer]
        t0 = time.perf_counter()
        res = {}
        try:
            w = state[layer]
            n, k, rows = spec[1], spec[2], spec[3]
            if not (w.dim() == 2 and tuple(w.shape) == (n, k) and w.dtype == torch.bfloat16 and w.is_contiguous()):
                why = f"weight {tuple(w.shape)} {w.dtype} unexpected"
                res = {B12_M: (False, None, why), B12X_M: (False, None, why)}
            else:
                _b12_encode_into(state, layer, w, rows, spec[4][0][0], verbose=True)
                for m, sp, key, fwd in ((B12_M, spec, "b12_inst_", _b12_forward),
                                        (B12X_M, spec8, "b12x_inst_", _b12x_forward)):
                    try:
                        res[m] = _b12_cell(state, layer, sp, m, key + layer, fwd, dev)
                    except Exception as e:  # never raises: any surprise keeps the incumbent for this cell
                        res[m] = (False, None, f"raised {type(e).__name__}: {str(e)[-300:]}")
                    state[sp[5]] = bool(res[m][0])
                    state["b12_err"].zero_()
        except Exception as e:  # never raises: any surprise keeps the incumbent paths for this layer
            why = f"raised {type(e).__name__}: {str(e)[-300:]}"
            res = {B12_M: (False, None, why), B12X_M: (False, None, why)}
            state[spec[5]] = state[spec8[5]] = False
        if not state[spec[5]]:
            state.pop("b12_inst_" + layer, None)
        if not state[spec8[5]]:
            state.pop("b12x_inst_" + layer, None)
        if not (state[spec[5]] or state[spec8[5]]):
            for key in ("b12_wc_", "b12_esc_", "b12_base_"):
                state.pop(key + layer, None)
        state["b12_err"].zero_()
        print(f"[prepare-time] cells {spec[0]} / {spec8[0]} {layer} total: {time.perf_counter() - t0:.2f}s", flush=True)
        for m, sp in ((B12_M, spec), (B12X_M, spec8)):
            routed, best, why = res.get(m, (False, None, "not run"))
            c_txt = "n/a" if best is None else f"{best[0]:.2f}"
            i_txt = "n/a" if best is None else f"{best[1]:.2f}"
            print(f"PATH CHOSEN {sp[0]} {layer} m={m}: {'candidate ' + best[3] if routed else 'incumbent'} "
                  f"(cand {c_txt} us vs incumbent {i_txt} us)" + ("" if routed else f"; {why}"), flush=True)
    torch.cuda.synchronize()


NEW_CELL_FLAGS = NEW_CELL_FLAGS + B12_FLAGS + B12X_FLAGS   # section (10)'s cells join the A/B check and FINAL FLAGS
