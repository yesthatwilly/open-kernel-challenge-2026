def _b12_say(state: dict, msg: str) -> None:
    """Print a bulk12 prepare line now and keep it for prepare()'s end-of-prepare reprint (the judge keeps a 16 KB
    log tail, which the A/B check alone fills)."""
    print(msg, flush=True)
    state.setdefault("b12_summary", []).append(msg)


def _b12_route_ok(layer: str, x: torch.Tensor, cfg, state: dict) -> bool:
    """bulk12 cell for (layer, cfg m) with its prepare()-set flag, cfg n / k the layer's, x a contiguous 16-byte-
    aligned bf16 [m, k]. cfg, shapes and state flags only."""
    cell = B12_BY_KEY.get((layer, cfg["m"]))
    if cell is None or not state.get(cell[4], False):
        return False
    n, k, _ = B12_LAYERS[layer]
    return (cfg["n"] == n and cfg["k"] == k and x.dim() == 2 and x.shape[0] == cfg["m"] and x.shape[1] == k
            and x.dtype == torch.bfloat16 and x.is_contiguous() and x.data_ptr() % 16 == 0)


def _b12_forward(layer: str, x: torch.Tensor, state: dict) -> torch.Tensor:
    """ONE launch: out[m, n] = x @ W.T from the state's 12-bit weight (m == 1: b12_kernel; else b12x_kernel)."""
    m = x.shape[0]
    out = torch.empty((m, B12_LAYERS[layer][0]), dtype=torch.bfloat16, device=x.device)
    run = _B12_EXT.b12_run if m == 1 else _B12_EXT.b12x_run
    run(state["b12_wc_" + layer], state["b12_esc_" + layer], x, out, state["b12_err"],
        state[f"b12_inst_{layer}_{m}"], state["b12_base_" + layer])
    return out


def _b12_encode_into(state: dict, layer: str, w: torch.Tensor, rows: int, inst: int, verbose: bool) -> None:
    """Encode w into the state's buffers for this layer (shared by its cells) and check the round trip on the device
    through the kernel's own decode (b12_unpack, with inst's R / K): every non-escape element bit-exact, every escape
    slot at w_floor's bits. Raises on any mismatch."""
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
        _b12_say(state, f"[b12] {layer}: base {base} (e71 window {base}..{base + 7}, binades {2 * base - 127}.."
                        f"{2 * base + 15 - 127}), escapes {n_esc} ({n_esc / el:.2e} of {el}), max {max_cta} per CTA "
                        f"(cap {B12_ESC_CAP}); stream {wc.numel() / 1e6:.2f} MB + escape blocks {esc.numel() / 1e6:.3f} "
                        f"MB vs raw {el * 2 / 1e6:.2f} MB (x{(wc.numel() + esc.numel()) / (el * 2):.4f}); host + "
                        f"device round trips bit-exact")


def _b12_cell(state: dict, cell, dev):
    """One bulk12 cell on its layer's encoded weight: per instance the occupancy query, canary + correctness
    (_cell_check) and the graph gate against kernel_fn with the cell's flag off (_cell_gate); the fastest passing
    instance routes at <= CELL_MAX_RATIO after the kernel-name assertion. Leaves the cell's instance key at the chosen
    instance. Returns (routed, (cand_us, incumbent_us, inst, label) or None, why)."""
    src, layer, m, variants, flag = cell
    n, k, rows = B12_LAYERS[layer]
    inst_key = f"b12_inst_{layer}_{m}"
    kname = "b12_kernel" if m == 1 else "b12x_kernel"
    best, why = None, ""
    w = state[layer]
    st_on = dict(state)
    st_on[flag] = True
    for inst, stages, minb, cw in variants:
        label = f"{kname}<{rows},{k},{stages},{minb},{cw}" + ("" if m == 1 else f",{m}") + ">"
        info = [int(v) for v in _B12_EXT.b12_setup(inst)]
        _b12_say(state, f"[b12] {src} {layer} m={m} {label}: regs={info[0]} local(spill)={info[1]} B static smem="
                        f"{info[2]} B dyn smem={info[3]} B, occupancy {info[4]} CTA/SM (needs {minb}), grid "
                        f"{n // rows} x {32 * (cw + 1)} threads")
        if info[4] < minb:
            why = f"{label}: occupancy {info[4]} CTA/SM < {minb}"
            continue
        state[inst_key] = inst
        st_on[inst_key] = inst
        bad = _cell_check(f"{src} {label}", lambda xx: _b12_forward(layer, xx, st_on), w, m, n, k,
                          err=state["b12_err"])
        if bad:
            why = f"{label}: {bad}"
            continue
        tg = time.perf_counter()
        cand_us, inc_us = _cell_gate(f"{src} {label}", layer, m, n, k, flag, state)
        code = int(state["b12_err"].item())
        _b12_say(state, f"[b12] gate {src} m={m} {label}: {time.perf_counter() - tg:.2f}s; cand {cand_us:.2f} us "
                        f"(raw-equivalent {n * k * 2 / (cand_us * 1e-6) / 1e12:.2f} TB/s, stream "
                        f"{n * k * 1.5 / (cand_us * 1e-6) / 1e12:.2f} TB/s) vs incumbent {inc_us:.2f} us (net of the "
                        f"empty-graph baseline)")
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
    """Every B12_CELLS cell. Sets each cell's flag (False unless every step passes) and, per layer with a routed
    cell, the state's own encoded weight; one per-state error word. flags None (the judge's state): per layer encode
    + host / device round trips once, then each of its cells through _b12_cell; one PATH CHOSEN line per cell. flags
    a dict (the A/B check's second state): inherit each flag and instance, encode only for layers with a cell on."""
    for cell in B12_CELLS:
        state[cell[4]] = False
    if _B12_EXT is None:
        if flags is None:
            for cell in B12_CELLS:
                print(f"PATH CHOSEN {cell[0]} {cell[1]} m={cell[2]}: incumbent (cand n/a us vs incumbent n/a us; "
                      f"no b12 module)", flush=True)
        return
    dev = state["router"].device
    state["b12_err"] = torch.zeros(1, dtype=torch.int32, device=dev)
    if flags is not None:
        insts = flags.get("_b12_inst", {})
        for layer, (n, k, rows) in B12_LAYERS.items():
            on = [c for c in B12_CELLS if c[1] == layer and flags.get(c[4]) and insts.get(c[0]) is not None]
            if not on:
                continue
            try:   # a failure here turns off only this layer's cells in this state, never the whole A/B
                _b12_encode_into(state, layer, state[layer], rows, insts[on[0][0]], verbose=False)
                for c in on:
                    state[f"b12_inst_{layer}_{c[2]}"] = insts[c[0]]
                    state[c[4]] = True
            except Exception as e:
                print(f"[b12] second state: {layer} encode raised {type(e).__name__}: {e}; cells off in this state",
                      flush=True)
        return
    for layer, (n, k, rows) in B12_LAYERS.items():
        cells = [c for c in B12_CELLS if c[1] == layer]
        t0 = time.perf_counter()
        res = {}
        try:
            w = state[layer]
            if not (w.dim() == 2 and tuple(w.shape) == (n, k) and w.dtype == torch.bfloat16 and w.is_contiguous()):
                why = f"weight {tuple(w.shape)} {w.dtype} unexpected"
                res = {c[0]: (False, None, why) for c in cells}
            else:
                _b12_encode_into(state, layer, w, rows, cells[0][3][0][0], verbose=True)
                for c in cells:
                    try:
                        res[c[0]] = _b12_cell(state, c, dev)
                    except Exception as e:  # never raises: any surprise keeps the incumbent for this cell
                        res[c[0]] = (False, None, f"raised {type(e).__name__}: {str(e)[-300:]}")
                    state[c[4]] = bool(res[c[0]][0])
                    state["b12_err"].zero_()
        except Exception as e:  # never raises: any surprise keeps the incumbent paths for this layer
            why = f"raised {type(e).__name__}: {str(e)[-300:]}"
            res = {c[0]: (False, None, why) for c in cells}
            for c in cells:
                state[c[4]] = False
        for c in cells:
            if not state[c[4]]:
                state.pop(f"b12_inst_{layer}_{c[2]}", None)
        if not any(state[c[4]] for c in cells):
            for key in ("b12_wc_", "b12_esc_", "b12_base_"):
                state.pop(key + layer, None)
        state["b12_err"].zero_()
        print(f"[prepare-time] b12 cells {layer} total: {time.perf_counter() - t0:.2f}s", flush=True)
        for c in cells:
            routed, best, why = res.get(c[0], (False, None, "not run"))
            c_txt = "n/a" if best is None else f"{best[0]:.2f}"
            i_txt = "n/a" if best is None else f"{best[1]:.2f}"
            _b12_say(state, f"PATH CHOSEN {c[0]} {layer} m={c[2]}: {'candidate ' + best[3] if routed else 'incumbent'} "
                            f"(cand {c_txt} us vs incumbent {i_txt} us)" + ("" if routed else f"; {why}"))
    torch.cuda.synchronize()


NEW_CELL_FLAGS = NEW_CELL_FLAGS + B12_FLAGS   # section (10)'s cells join the A/B check and the FINAL FLAGS line
