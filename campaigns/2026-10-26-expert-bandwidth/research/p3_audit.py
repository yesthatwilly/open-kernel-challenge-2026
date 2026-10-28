#!/usr/bin/env python3
"""P3 audit: Engine A measurement table from the nodejudge P3 ncu raw page (H100 SXM, gpunode06, 2026-10-28 03:39Z).

Usage:  python3 -I p3_audit.py <ncu.csv> <outdir> [<party.json>]
        party.json defaults to <dir of ncu.csv>/../pinned.json (rep_features of the profiled arm).
Writes: <outdir>/tables.md, <outdir>/tables.json, <outdir>/columns.txt.  Stdlib only.

What the CSV can and cannot say (checked against the file, see tables.md "Derivations"):
  * The export was `ncu --import x.ncu-rep --csv --page raw` WITHOUT --print-metric-instances, so every instanced
    sass__*_per_opcode* metric holds only its total: no opcode or pipeline breakdown exists. No .ncu-rep was kept.
  * The kernel was replayed in 19 passes (profiler__replayer_passes); counts backed out of pct_of_peak metrics
    multiply a pct from one pass by gpc__cycles_elapsed from (possibly) another, ~1-3 % noise.
  * `inst_executed` is the SASS-patching (instrumented) count; the hardware count is backed out of
    sm__inst_executed pct (peak 4 warp-inst/clk/SM) and cross-checked against warps_active / (warps per inst).
Geometry (read from snap/board-498afbd8e73d.py, namespaces eak and eak4):
  v4 (eak4): stage = 2 x 18432 B = 36864 B = 128 rows x 512 k = 65536 weights; a pass = one local expert with
     <= 16 routes (CAP); every pass streams the whole expert: rows/128 x 8 stages; x rows: 1024 B per route per stage.
     Passes with 9..16 routes run two n8 tiles (consume2: ldmatrix.x4 + 2 mma per k16 step), else one.
  v1 (eak): stage = 18432 B = 64 rows x 512 k; one pass per active expert; T token rows (1024 B) per stage.
  Consumer, per warp per 512-k record (32 k16 steps): per kq (8 steps) 2 x LDS.128 (4 wavefronts each) + 2 x LDS.64
     (8 distinct 8-B addresses, broadcast: 1 wavefront each), and per step one ldmatrix (.x2 at q=1: 2 wavefronts,
     .x4 at q=2: 4 wavefronts, XROW = 1040 B pad = conflict-free).  Ideal shared wavefronts/step = 1.25 + 2 q.
"""
import csv
import json
import math
import os
import re
import statistics
import sys

NSM_DEFAULT = 132
K = 4096
BPW = 0.5625  # NVFP4 bytes per weight (codes + E4M3 scale per 16)
REC_W = 65536  # weights per 36864-B record
XBAR_PEAK_SECT = 4  # l1tex__m_xbar2l1tex_read_sectors peak per SM per clk (assumed; validated on clean launches)
LSU_PEAK = 1.0  # sm__inst_executed_pipe_lsu / l1tex__data_pipe_lsu_wavefronts peak per SM per clk (assumed)
HM_HOT_ROWS = 16  # board: an expert with more local routes than this is HOT (verify 32 hybrid)

# p3.json arm "pinned" (configs/p3.json): which Engine A variant each shape ran
VARIANT = {"decode 1": "v1", "decode 16": "v4", "decode 32": "v4c", "verify 1": "v4",
           "verify 16": "v4c", "verify 32": "v4 (cold half, hybrid H=-16)"}
HYBRID = {"verify 32"}

UNIT = {"": 1.0, "%": 1.0, "Tbyte/s": 1e12, "Gbyte/s": 1e9, "Mbyte/s": 1e6, "Ghz": 1e9, "Mhz": 1e6, "us": 1e-6,
        "ns": 1e-9, "ms": 1e-3, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "byte": 1.0, "Kbyte/block": 1e3,
        "cycle": 1.0, "inst": 1.0, "sector": 1.0, "warp": 1.0, "block": 1.0, "register/thread": 1.0,
        "thread": 1.0, "pass": 1.0, "SM": 1.0}

STALLS = ["barrier", "branch_resolving", "dispatch_stall", "drain", "gmma", "imc_miss", "lg_throttle",
          "long_scoreboard", "math_pipe_throttle", "membar", "mio_throttle", "misc", "no_instruction",
          "not_selected", "selected", "short_scoreboard", "sleeping", "tex_throttle", "wait"]
CYC_PIPES = ["alu", "fma", "fmaheavy", "fp64", "shared", "tensor", "tensor_type_hmma_hgmma_qgmma_imma_igmma_bmma_bgmma",
             "tensor_op_dmma", "tma"]
INST_PIPES = ["lsu", "adu", "cbu_pred_on_any", "xu", "uniform", "tex", "ipa"]

ABSENT_NEEDED = {
    "per-PC loop attribution": [
        "no .ncu-rep kept (nodejudge keep_rep false) and no SourceCounters section: no per-PC inst_executed, "
        "no smsp__pcsamp_warps_issue_stalled_* per PC (only smsp__warpidsamp_* kernel totals)",
        "sass__inst_executed_per_opcode / _pipeline / _category / _with_modifier_all: instanced, exported as "
        "totals only (CSV export lacked --print-metric-instances values): no opcode or pipeline breakdown",
        "smsp__inst_executed.sum and smsp__inst_issued.sum raw counts (only sm__inst_executed pct; no issued at all)",
    ],
    "bank conflicts per x-load PC": [
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared(_op_ld/_op_st).sum",
        "l1tex__data_pipe_lsu_wavefronts_mem_shared(_op_ld/_op_st).sum (only all-LSU wavefronts as pct of peak)",
        "smsp__sass_inst_executed_op_shared_ld/st.sum, smsp__sass_inst_executed_op_ldsm.sum",
        "per-PC source metrics memory_l1_wavefronts_shared / memory_l1_wavefronts_shared_ideal (SourceCounters)",
    ],
    "per-pipe cycles-active utilization": [
        "sm__pipe_*_cycles_active.avg.pct_of_peak_sustained_ACTIVE (only _elapsed); smsp__cycles_active",
        "sm__inst_executed_pipe_alu / fma / fmaheavy / fmalite / fp16 / tensor_op_hmma (inst counts or pct; only "
        "lsu, adu, cbu, xu, uniform, tex, ipa exist, and only as pct of peak)",
        "sm__pipe_fmalite_cycles_active, sm__pipe_fp16_cycles_active, sm__pipe_tensor_op_hmma_cycles_active "
        "(only the hmma+hgmma+...+bgmma union and the shared pipe)",
    ],
    "bytes (raw counts)": [
        "dram__bytes.sum / dram__bytes_read.sum (only .per_second)",
        "l1tex__m_xbar2l1tex_read_bytes/sectors.sum (only pct of peak); lts__t_sectors_srcunit_tex_op_read.sum",
        "no TMA / bulk-copy byte counter (l1tex__m_xbar2l1tex_read_sectors_mem_dshared is DSMEM, zero here)",
    ],
}


def num(s):
    s = (s or "").strip().replace(",", "")
    if s in ("", "n/a", "N/A"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1], rows[2:]


def med(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return statistics.median(v) if v else None


def main():
    csvp, outd = sys.argv[1], sys.argv[2]
    partyp = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(csvp))),
                                                               "pinned.json")
    os.makedirs(outd, exist_ok=True)
    hdr, units, data = load(csvp)
    ix = {c: i for i, c in enumerate(hdr)}
    with open(os.path.join(outd, "columns.txt"), "w") as f:
        for i, c in enumerate(hdr):
            f.write(f"{i}\t{c}\t{units[i]}\n")

    party = json.load(open(partyp))
    feats, bs = {}, {}
    for rec in party["result"]["records"]:
        if rec.get("event") == "benchmark":
            feats[rec["case"]["load"]] = rec["rep_features"]["rows"]
            bs[rec["case"]["load"]] = rec["case"]["bs"]

    def g(x, c, scale=True):
        if c not in ix:
            return None
        v = num(x[ix[c]])
        if v is None:
            return None
        return v * UNIT.get(units[ix[c]], 1.0) if scale else v

    def pct(x, c):
        return g(x, c, scale=False)

    launches = []
    byid0 = {int(x[0]): x for x in data}
    for x in data:
        kname = x[ix["Kernel Name"]]
        m = re.search(r"\b(eak4?)::(ea4?_w(13|2))_kernel<([^>]*)>", kname)
        if not m:
            continue
        nv = re.search(r"nj\|(\w+)\|([^|]+)\|(\d+)", x[4])
        party_name, shape, rep = nv.group(1), nv.group(2), int(nv.group(3))
        ns, w13 = m.group(1), m.group(3) == "13"
        v1 = ns == "eak"
        kern = ("ea_" if v1 else "ea4_") + ("w13" if w13 else "w2") + (" (v1)" if v1 else " (v4)")
        rows = 8192 if w13 else 4096
        We = rows * K * BPW
        lr, dl, mx = feats[shape][rep]
        T = bs[shape]
        nsm = g(x, "launch__sm_count") or NSM_DEFAULT
        el = g(x, "gpc__cycles_elapsed.avg")
        act = g(x, "sm__cycles_active.avg")
        clk = g(x, "gpc__cycles_elapsed.avg.per_second")
        dur = g(x, "gpu__time_duration.sum")
        dram = g(x, "dram__bytes.sum.per_second") * dur
        xbar = pct(x, "l1tex__m_xbar2l1tex_read_sectors.sum.pct_of_peak_sustained_elapsed") / 100 * XBAR_PEAK_SECT * 32 * nsm * el
        dshared = pct(x, "l1tex__m_xbar2l1tex_read_sectors_mem_dshared.sum.pct_of_peak_sustained_elapsed")
        ltsmiss = g(x, "lts__t_sectors_lookup_miss.sum") * 32
        hybrid = shape in HYBRID
        L = {"id": int(x[0]), "kernel": kern, "kname": re.sub(r"\(.*", "", kname).replace("void ", ""), "shape": shape,
             "rep": rep, "party": party_name, "variant": VARIANT.get(shape, "?") if not v1 else "v1", "T": T,
             "lr": lr, "dl": dl, "mx": mx, "rows": rows, "We": We, "nsm": nsm, "el": el, "act": act, "clk_ghz": clk / 1e9,
             "dur_us": dur * 1e6, "dram_B": dram, "xbar_B": xbar, "dshared_pct": dshared, "ltsmiss_B": ltsmiss,
             "hybrid": hybrid, "v1": v1}
        # ---- passes and x bytes from geometry ----
        if v1:
            L["P_lb"] = L["P_ub"] = dl
            L["x_B"] = dl * T * We / 18.0  # T rows x 1024 B per 18432-B stage
            L["clean"] = True
            L["q_lo"] = L["q_hi"] = 1.0
        elif not hybrid:
            n2max = mx // 16 + (1 if mx % 16 > 8 else 0)
            L["P_lb"] = dl + max(0, math.ceil(mx / 16) - 1)
            L["P_ub"] = max(L["P_lb"], (lr + 15 * dl) // 16)
            L["x_B"] = lr * We / 36.0  # 1024 B per route per 36864-B stage
            L["clean"] = mx <= 16
            L["R"] = lr
            L["n2_lb"] = n2max
            L["q_exact"] = mx <= 8
        else:
            L["clean"] = False
        launches.append(L)

    # ---- calibration from clean launches: xbar residual and DRAM offset per kernel ----
    calib = {}
    for kern in sorted({l["kernel"] for l in launches}):
        cl = [l for l in launches if l["kernel"] == kern and l["clean"]]
        res = [(l["xbar_B"] - l["x_B"]) / l["We"] - l["P_lb"] for l in cl]
        rat = [l["xbar_B"] / (l["P_lb"] * l["We"] + l["x_B"]) for l in cl]
        doff = [l["dram_B"] / l["We"] - l["P_lb"] for l in cl]
        lts = [l["ltsmiss_B"] / (l["P_lb"] * l["We"]) for l in cl]
        calib[kern] = {"n_clean": len(cl), "xbar_resid_passes_med": med(res), "xbar_resid_min": min(res) if res else None,
                       "xbar_resid_max": max(res) if res else None, "xbar_over_geometry_med": med(rat),
                       "xbar_over_geometry_min": min(rat) if rat else None, "xbar_over_geometry_max": max(rat) if rat else None,
                       "dram_offset_passes_med": med(doff), "dram_offset_min": min(doff) if doff else None,
                       "dram_offset_max": max(doff) if doff else None, "ltsmiss_over_weights_med": med(lts)}

    # ---- passes. Primary for non-clean launches: ALU-pipe busy cycles (dequant work is per weight), fitted per kernel
    # on the clean launches as alu_busy(SM-avg cycles) = slope * P + icpt. xbar and DRAM are kept as cross-checks
    # (xbar over-reads at verify 16, DRAM misses L2-hit re-reads). w13 and w2 of one rep share routing: pair mean.
    for l in launches:
        x = byid0[l["id"]]
        l["alu_busy"] = pct(x, "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed") / 100 * l["el"]
    for kern, c in calib.items():
        cl = [(l["P_lb"], l["alu_busy"]) for l in launches if l["kernel"] == kern and l["clean"]]
        n = len(cl)
        mp, mt = sum(p for p, _ in cl) / n, sum(t for _, t in cl) / n
        slope = sum((p - mp) * (t - mt) for p, t in cl) / sum((p - mp) ** 2 for p, _ in cl)
        icpt = mt - slope * mp
        sd = (sum((t - icpt - slope * p) ** 2 for p, t in cl) / max(1, n - 2)) ** 0.5 / slope
        c.update(alu_slope_per_pass=slope, alu_icpt=icpt, alu_resid_sd_passes=sd)
    for l in launches:
        c = calib[l["kernel"]]
        l["P_dram"] = l["dram_B"] / l["We"] - c["dram_offset_passes_med"]
        l["P_alu"] = (l["alu_busy"] - c["alu_icpt"]) / c["alu_slope_per_pass"]
        if l["v1"] or not l["hybrid"]:
            l["P_xbar"] = (l["xbar_B"] - l["x_B"]) / l["We"]
            l["P_xbar_corr"] = l["P_xbar"] - c["xbar_resid_passes_med"]
    pair = {}
    for l in launches:
        pair.setdefault((l["shape"], l["rep"], l["v1"]), []).append(l)
    for key, ls in pair.items():
        est = statistics.mean(l["P_alu"] for l in ls)
        fr = est - math.floor(est)
        for l in ls:
            l["P_pair_alu"] = est
        if ls[0]["v1"] or not ls[0]["hybrid"]:
            for l in ls:
                if l["clean"]:
                    l["P_used"] = l["P_lb"]
                    l["P_src"] = "exact (max_expert_routes <= 16: one pass per distinct local expert)"
                else:
                    l["P_used"] = min(max(round(est), l["P_lb"]), l["P_ub"])
                    l["P_src"] = "ALU-busy fit (pair mean), rounded, clipped to routing bounds"
                l["P_ambiguous"] = (not l["clean"]) and 0.35 < fr < 0.65
                l["P_xbar_disagrees"] = round(l["P_xbar_corr"]) != l["P_used"]
        else:
            # cold half only: P = cold experts (<= 16 routes each); routes R unknown, from the xbar residual.
            P = round(est)
            for l in ls:
                c = calib[l["kernel"]]
                l["P_used"] = P
                l["P_ambiguous"] = 0.35 < fr < 0.65
                l["P_dram_disagrees"] = round(statistics.mean(m["P_dram"] for m in ls)) != P
                l["R"] = 36.0 * (l["xbar_B"] / l["We"] - P - c["xbar_resid_passes_med"])
                l["x_B"] = l["R"] * l["We"] / 36.0
                l["P_xbar"] = None
                l["P_lb"] = l["P_ub"] = None
                l["P_src"] = "ALU-busy fit (pair mean, cold experts); R from xbar residual"
                l["R_ok"] = P <= l["R"] <= 16 * P
                l["q_exact"] = False
    for l in launches:
        l["xbar_over_geometry"] = l["xbar_B"] / (l["P_used"] * l["We"] + l["x_B"])
    for l in launches:
        P, nsm, el, act = l["P_used"], l["nsm"], l["el"], l["act"]
        weights = P * l["rows"] * K
        steps = weights / 256.0
        recs_sm = weights / REC_W / nsm
        steps_sm = steps / nsm
        l.update(weights=weights, weight_B=weights * BPW, steps=steps, recs_per_sm=recs_sm, steps_per_sm=steps_sm)
        l["x_frac_of_xbar"] = l["x_B"] / l["xbar_B"]
        l["cyc_rec_active"] = act / recs_sm
        l["cyc_rec_elapsed"] = el / recs_sm
        l["eff_TBps_weights"] = l["weight_B"] / (l["dur_us"] * 1e-6) / 1e12
    # second pass needs the raw rows again
    byid = {int(x[0]): x for x in data}
    for l in launches:
        x = byid[l["id"]]
        P, nsm, el, act, steps = l["P_used"], l["nsm"], l["el"], l["act"], l["steps"]

        def gp(c):
            return pct(x, c)

        inst_sass = g(x, "inst_executed")
        inst_hw = gp("sm__inst_executed.sum.pct_of_peak_sustained_elapsed") / 100 * 4 * nsm * el
        wa_sum = g(x, "sm__warps_active.avg.per_cycle_active") * g(x, "sm__cycles_active.sum")
        inst_hw2 = wa_sum / g(x, "smsp__average_warps_active_per_inst_executed.ratio")
        l.update(inst_sass=inst_sass, inst_hw=inst_hw, inst_hw2=inst_hw2, I_step_hw=inst_hw / steps,
                 I_step_hw2=inst_hw2 / steps, I_step_sass=inst_sass / steps, sass_over_hw=inst_sass / inst_hw,
                 hw2_over_hw=inst_hw2 / inst_hw, threads_per_inst=g(x, "smsp__thread_inst_executed_per_inst_executed.ratio"),
                 pipe_sum_over_inst=g(x, "sass__inst_executed_per_opcode_pipeline") / inst_sass)
        # ---- issue ----
        l["issue_active_pct_elapsed"] = gp("sm__issue_active.avg.pct_of_peak_sustained_elapsed")
        l["issue_active_pct_active"] = gp("smsp__issue_active.avg.pct_of_peak_sustained_active")
        l["issue_per_cycle_active_smsp"] = g(x, "smsp__issue_active.avg.per_cycle_active")
        l["warps_active_per_sched"] = g(x, "smsp__warps_active.avg.per_cycle_active")
        l["warps_eligible_per_sched"] = g(x, "smsp__warps_eligible.avg.per_cycle_active")
        l["achieved_occ_pct"] = gp("sm__warps_active.avg.pct_of_peak_sustained_active")
        l["theoretical_occ_pct"] = gp("sm__maximum_warps_per_active_cycle_pct")
        l["sm_active_frac"] = act / el
        l["smsp_slots_per_step_elapsed"] = 4 * el / l["steps_per_sm"]
        l["smsp_slots_per_step_active"] = 4 * act / l["steps_per_sm"]
        l["no_issue_pct_active"] = gp("smsp__issue_inst0.avg.pct_of_peak_sustained_active")
        # ---- stalls ----
        st = {s: g(x, f"smsp__average_warps_issue_stalled_{s}_per_issue_active.ratio") for s in STALLS}
        l["stalls"] = st
        l["stall_sum"] = sum(v for v in st.values() if v is not None)
        l["warps_active_per_inst"] = g(x, "smsp__average_warps_active_per_inst_executed.ratio")
        samp = {}
        for c in hdr:
            mm = re.match(r"smsp__warpidsamp_warps_issue_stalled_(\w+)_not_issued$", c)
            if mm:
                samp[mm.group(1)] = g(x, c)
        l["warpidsamp"] = samp
        # ---- pipes ----
        l["pipe_cyc_pct"] = {p: gp(f"sm__pipe_{p}_cycles_active.avg.pct_of_peak_sustained_elapsed") for p in CYC_PIPES}
        l["pipe_busy_cyc_per_rec"] = {p: (v / 100 * el / l["recs_per_sm"] if v is not None else None)
                                      for p, v in l["pipe_cyc_pct"].items()}
        l["pipe_inst_pct"] = {p: gp(f"sm__inst_executed_pipe_{p}.avg.pct_of_peak_sustained_elapsed") for p in INST_PIPES}
        l["inst_pct"] = gp("sm__inst_executed.avg.pct_of_peak_sustained_elapsed")
        l["lsu_inst_per_step"] = l["pipe_inst_pct"]["lsu"] / 100 * LSU_PEAK * nsm * el / steps
        l["tensor_cyc_per_rec"] = l["pipe_busy_cyc_per_rec"]["tensor"]
        # ---- shared memory / MIO ----
        l["lsu_wavefront_pct"] = gp("l1tex__data_pipe_lsu_wavefronts.avg.pct_of_peak_sustained_elapsed")
        l["lsu_wavefronts_per_step"] = l["lsu_wavefront_pct"] / 100 * LSU_PEAK * nsm * el / steps
        l["bank_reads_pct"] = gp("l1tex__data_bank_reads.avg.pct_of_peak_sustained_elapsed")
        l["bank_writes_pct"] = gp("l1tex__data_bank_writes.avg.pct_of_peak_sustained_elapsed")
        l["mio_issue_pct"] = gp("sm__mio_inst_issued.avg.pct_of_peak_sustained_elapsed")
        l["mio_pq_read_pct"] = gp("sm__mio_pq_read_cycles_active.avg.pct_of_peak_sustained_elapsed")
        l["mio_pq_write_pct"] = gp("sm__mio_pq_write_cycles_active.avg.pct_of_peak_sustained_elapsed")
        l["l1tex_throughput_pct_active"] = gp("l1tex__throughput.avg.pct_of_peak_sustained_active")
        l["dram_pct"] = gp("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed")
        # ---- resources ----
        l["regs"] = g(x, "launch__registers_per_thread")
        l["regs_alloc"] = g(x, "launch__registers_per_thread_allocated")
        l["block"] = int(g(x, "launch__block_size"))
        l["grid"] = int(g(x, "launch__grid_size"))
        l["smem_static_B"] = g(x, "launch__shared_mem_per_block_static")
        l["smem_dynamic_B"] = g(x, "launch__shared_mem_per_block_dynamic")
        l["smem_driver_B"] = g(x, "launch__shared_mem_per_block_driver")
        l["smem_alloc_B"] = g(x, "launch__shared_mem_per_block_allocated")
        l["occ_limits"] = {k: g(x, f"launch__occupancy_limit_{k}") for k in
                           ("registers", "shared_mem", "warps", "blocks", "barriers")}
        l["waves"] = g(x, "launch__waves_per_multiprocessor")
        l["spill_insts"] = g(x, "sass__inst_executed_register_spilling")
        l["spill_local_B"] = g(x, "sass__inst_executed_register_spilling_mem_local")
        l["spill_shared_B"] = g(x, "sass__inst_executed_register_spilling_mem_shared")
        l["local_spill_requests"] = g(x, "derived__local_spilling_requests")
        l["replay_passes"] = g(x, "profiler__replayer_passes")

    # ---- HMMA proxy: tensor-pipe busy cycles per record, calibrated on launches that are q = 1 by routing ----
    q1 = [l for l in launches if (not l["v1"]) and l.get("q_exact")]
    c1 = med([l["tensor_cyc_per_rec"] for l in q1])
    c1_v1 = med([l["tensor_cyc_per_rec"] for l in launches if l["v1"]])
    for l in launches:
        l["hmma_per_step_proxy"] = l["tensor_cyc_per_rec"] / c1
        q = 1.0 if (l["v1"] or l.get("q_exact")) else l["hmma_per_step_proxy"]
        l["q_used_for_ideal"] = q
        l["q_exact_by_routing"] = bool(l["v1"] or l.get("q_exact"))
        l["smem_wf_ideal_per_step"] = 1.25 + 2 * q
        l["smem_wf_actual_over_ideal"] = l["lsu_wavefronts_per_step"] / l["smem_wf_ideal_per_step"]
        l["smem_wf_excess_per_step"] = l["lsu_wavefronts_per_step"] - l["smem_wf_ideal_per_step"]
        if not l["v1"] and not l["hybrid"]:
            n2lo = l["n2_lb"]
            n2hi = min(l["P_used"], l["lr"] // 9)
            l["q_lo"] = 1 + n2lo / l["P_used"]
            l["q_hi"] = 1 + n2hi / l["P_used"]
        elif l["hybrid"]:
            l["q_lo"], l["q_hi"] = 1.0, 2.0
    integ = [abs(l["hmma_per_step_proxy"] * l["P_used"] - round(l["hmma_per_step_proxy"] * l["P_used"])) < 0.01
             and l["q_lo"] - 1e-3 <= l["hmma_per_step_proxy"] <= l["q_hi"] + 1e-3 for l in launches]
    hmma_cal = {"n_launches_integer_N_and_within_routing_q_bounds": sum(integ), "n_launches": len(launches),
                "tensor_busy_cyc_per_rec_q1_v4_median": c1, "n_q1_v4": len(q1),
                "tensor_busy_cyc_per_rec_q1_v4_min": min(l["tensor_cyc_per_rec"] for l in q1),
                "tensor_busy_cyc_per_rec_q1_v4_max": max(l["tensor_cyc_per_rec"] for l in q1),
                "v1_q1_tensor_busy_cyc_per_rec_median": c1_v1, "v1_over_v4_q1": c1_v1 / c1,
                "implied_tensor_cyc_per_hmma_per_smsp": c1 / (256 / 4)}

    # ---- aggregate per kernel x shape ----
    order = ["decode 1", "verify 1", "decode 16", "decode 32", "verify 16", "verify 32"]
    groups = {}
    for l in launches:
        groups.setdefault((l["kernel"], l["shape"]), []).append(l)
    scalar_keys = [k for k, v in launches[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    scalar_keys += [k for k in ("R", "P_xbar", "P_xbar_corr", "P_dram", "P_alu", "q_lo", "q_hi") if k not in scalar_keys]
    agg = []
    for (kern, shape), ls in sorted(groups.items(), key=lambda kv: (kv[0][0], order.index(kv[0][1]))):
        a = {"kernel": kern, "shape": shape, "variant": ls[0]["variant"], "T": ls[0]["T"], "n": len(ls),
             "launch_ids": [l["id"] for l in ls], "reps": [l["rep"] for l in ls]}
        for k in scalar_keys:
            v = [l.get(k) for l in ls if l.get(k) is not None]
            if v:
                a[k] = {"med": med(v), "min": min(v), "max": max(v)}
        for dk in ("stalls", "pipe_cyc_pct", "pipe_busy_cyc_per_rec", "pipe_inst_pct", "occ_limits", "warpidsamp"):
            a[dk] = {s: med([l[dk][s] for l in ls if l[dk].get(s) is not None]) for s in ls[0][dk]}
        a["P_src"] = sorted({l["P_src"] for l in ls})
        a["P_ambiguous_reps"] = [l["rep"] for l in ls if l.get("P_ambiguous")]
        a["hmma_total_over_steps_per_pass"] = [round(l["hmma_per_step_proxy"] * l["P_used"], 3) for l in ls]
        a["q_exact_by_routing"] = [l["q_exact_by_routing"] for l in ls]
        if any(l["hybrid"] for l in ls):
            a["R_ok"] = [l["R_ok"] for l in ls]
        agg.append(a)

    out = {"source_csv": os.path.abspath(csvp), "party_json": os.path.abspath(partyp), "n_launches_total": len(data),
           "n_engine_a_launches": len(launches), "constants": {"XBAR_PEAK_SECT_per_clk_per_SM": XBAR_PEAK_SECT,
           "LSU_PEAK_per_clk_per_SM": LSU_PEAK, "BPW": BPW, "REC_W": REC_W}, "calibration": calib,
           "hmma_proxy_calibration": hmma_cal, "absent_metric_families": ABSENT_NEEDED, "aggregates": agg,
           "launches": [{k: v for k, v in l.items() if not k.startswith("_")} for l in launches]}
    with open(os.path.join(outd, "tables.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    write_md(os.path.join(outd, "tables.md"), out)


def f_(v, nd=2):
    if v is None:
        return "-"
    if isinstance(v, dict):
        v = v.get("med")
        if v is None:
            return "-"
    if abs(v) >= 1e6:
        return f"{v:.4g}"
    if abs(v) >= 100:
        return f"{v:.0f}"
    return f"{v:.{nd}f}"


def mmm(a, k, nd=2):
    v = a.get(k)
    if not v:
        return "-"
    return f"{f_(v['med'], nd)} [{f_(v['min'], nd)}, {f_(v['max'], nd)}]"


def write_md(path, o):
    A = o["aggregates"]
    L = []
    w = L.append
    w("# P3 audit: Engine A on H100 (gpunode06, ncu --clock-control none, 2026-10-28 03:39Z)\n")
    w(f"Source: `{o['source_csv']}`; routing features: `{o['party_json']}`. {o['n_engine_a_launches']} Engine A launches "
      f"of {o['n_launches_total']}. Cells: median [min, max] over the 6 reps of a shape. All counts are KERNEL-LEVEL "
      "(producer warp, build_map setup, stream-K fix-up, epilogue and idle CTAs included).\n")
    w("Shape mapping: NVTX push/pop range column (`nj|<party>|<shape>|<rep>`) on every launch; rep indexes "
      "rep_features.rows of that shape in the party JSON ([local_routes, distinct_local, max_expert_routes]).\n")
    w("## 1. Bytes, passes, steps, cycles per record\n")
    w("| kernel | shape | var | P used | P_alu (pair fit) | P_xbar bias-corr | P_dram | P bounds | xbar/geometry | weight MB | xbar MB | x MB (% of xbar) | DRAM MB | "
      "clk GHz | SM-active cyc/rec | elapsed cyc/rec | active frac | weight TB/s (prof) |")
    w("|" + "---|" * 18)
    for a in A:
        pb = f"{f_(a.get('P_lb'),0)}..{f_(a.get('P_ub'),0)}" if a.get("P_lb") else "cold: n/a"
        w(f"| {a['kernel']} | {a['shape']} | {a['variant']} | {mmm(a,'P_used',0)} | {mmm(a,'P_alu',2)} | {mmm(a,'P_xbar_corr',2)} | {mmm(a,'P_dram',2)} | {pb} | {mmm(a,'xbar_over_geometry',3)} | "
          f"{f_(a['weight_B']['med']/1e6,1)} | {f_(a['xbar_B']['med']/1e6,1)} | {f_(a['x_B']['med']/1e6,1)} "
          f"({f_(100*a['x_frac_of_xbar']['med'],1)}%) | {f_(a['dram_B']['med']/1e6,1)} | {mmm(a,'clk_ghz',3)} | "
          f"{mmm(a,'cyc_rec_active',0)} | {mmm(a,'cyc_rec_elapsed',0)} | {f_(a['sm_active_frac'],3)} | "
          f"{f_(a['eff_TBps_weights'],2)} |")
    amb = [f"{a['kernel']} {a['shape']} reps {a['P_ambiguous_reps']}" for a in A if a["P_ambiguous_reps"]]
    w("\nP = weight passes (one pass streams a whole expert once). Clean launches (max_expert_routes <= 16, not hybrid): "
      "P = distinct_local exactly. Others: ALU-busy fit (see Derivations). Ambiguous (pair estimate within 0.35..0.65 "
      "of an integer): " + ("; ".join(amb) if amb else "none") + ". xbar/geometry = L2->SM bytes / (P x expert bytes + "
      "x bytes): the excess over 1 is traffic the geometry does not name (stream-K partials, tickets, topk ids).")
    w("\nrecord = 65,536 weights (one 36,864-B v4 stage); cycles per record per SM = cycles.avg / (weights / 65,536 / 132), "
      "SM-active = sm__cycles_active.avg, elapsed = gpc__cycles_elapsed.avg; clock = gpc__cycles_elapsed.avg.per_second. "
      "Profiled TB/s is under ncu replay (cache control on, clocks unlocked): not comparable to the judge.\n")
    w("## 2. Instructions per 256-weight step\n")
    w("| kernel | shape | steps (M) | I_step HW | I_step HW (warps cross-check) | I_step SASS-instrumented | SASS/HW | "
      "LSU inst/step | inst pct | HMMA/step proxy | q by routing [lo, hi] |")
    w("|" + "---|" * 11)
    for a in A:
        w(f"| {a['kernel']} | {a['shape']} | {f_(a['steps']['med']/1e6,3)} | {mmm(a,'I_step_hw')} | {mmm(a,'I_step_hw2')} | "
          f"{mmm(a,'I_step_sass')} | {f_(a['sass_over_hw'],3)} | {mmm(a,'lsu_inst_per_step')} | {f_(a['inst_pct'],1)} | "
          f"{mmm(a,'hmma_per_step_proxy')} | {f_(a.get('q_lo'),2)}..{f_(a.get('q_hi'),2)} |")
    w("\nPer-opcode / per-pipeline instruction counts and the top-25 opcodes: ABSENT (sass__*_per_opcode* exported as "
      "totals only). HMMA/step is a proxy: tensor-pipe busy cycles per record divided by its median over the v4 launches "
      "that are q = 1 by routing (max_expert_routes <= 8).\n")
    w("Pipe utilisation (pct of peak, elapsed; cycle metrics are busy fractions, not instruction counts; do not add a "
      "union pipe to its parts):\n")
    pk = ["alu", "fma", "fmaheavy", "shared", "tensor", "tma"]
    w("| kernel | shape | " + " | ".join(f"{p} cyc%" for p in pk) + " | " + " | ".join(f"{p} busy cyc/rec" for p in ("alu", "fmaheavy", "tensor")) +
      " | inst lsu% | inst adu% | inst cbu% | inst xu% | inst uniform% |")
    w("|" + "---|" * (2 + len(pk) + 3 + 5))
    for a in A:
        w(f"| {a['kernel']} | {a['shape']} | " + " | ".join(f_(a['pipe_cyc_pct'][p], 1) for p in pk) + " | " +
          " | ".join(f_(a['pipe_busy_cyc_per_rec'][p], 0) for p in ("alu", "fmaheavy", "tensor")) + " | " +
          " | ".join(f_(a['pipe_inst_pct'][p], 2) for p in ("lsu", "adu", "cbu_pred_on_any", "xu", "uniform")) + " |")
    w("\n## 3-4. Issue\n")
    w("| kernel | shape | issue-active % (elapsed) | issue-active % (SMSP active) | warps active / sched | warps eligible / sched | "
      "no-issue % (issue_inst0) | achieved occ % | theor. occ % | SMSP slots/step (elapsed) | SMSP slots/step (active) |")
    w("|" + "---|" * 11)
    for a in A:
        w(f"| {a['kernel']} | {a['shape']} | {f_(a['issue_active_pct_elapsed'],1)} | {f_(a['issue_active_pct_active'],1)} | "
          f"{mmm(a,'warps_active_per_sched')} | {mmm(a,'warps_eligible_per_sched')} | {f_(a.get('no_issue_pct_active'),1)} | "
          f"{f_(a['achieved_occ_pct'],1)} | {f_(a['theoretical_occ_pct'],1)} | {mmm(a,'smsp_slots_per_step_elapsed',1)} | "
          f"{mmm(a,'smsp_slots_per_step_active',1)} |")
    w("\nIssued instructions (smsp__inst_issued) are ABSENT, so issued vs executed cannot be compared.\n")
    w("## 5. Stall reasons (warps per issue-active cycle, median)\n")
    for a in A:
        st = sorted(((v, s) for s, v in a["stalls"].items() if v is not None), reverse=True)
        w(f"- **{a['kernel']} / {a['shape']}** (sum {f_(sum(v for v, _ in st),2)}; warps active per inst "
          f"{f_(a['warps_active_per_inst'],2)}): " + ", ".join(f"{s} {v:.3f}" for v, s in st if v > 0.0005))
    w("\n## 6. Shared memory and MIO\n")
    w("| kernel | shape | LSU wavefronts/step (all LSU) | ideal/step (source, 1.25+2q) | actual/ideal | excess/step | "
      "LSU wf % | bank reads % | bank writes % | MIO issue % | MIO pq read % | L1 throughput % (active) |")
    w("|" + "---|" * 12)
    for a in A:
        w(f"| {a['kernel']} | {a['shape']} | {mmm(a,'lsu_wavefronts_per_step')} | {mmm(a,'smem_wf_ideal_per_step')} | "
          f"{mmm(a,'smem_wf_actual_over_ideal')} | {mmm(a,'smem_wf_excess_per_step')} | {f_(a['lsu_wavefront_pct'],1)} | "
          f"{f_(a['bank_reads_pct'],1)} | {f_(a['bank_writes_pct'],1)} | {f_(a['mio_issue_pct'],1)} | "
          f"{f_(a['mio_pq_read_pct'],1)} | {f_(a['l1tex_throughput_pct_active'],1)} |")
    w("\nWavefronts are l1tex__data_pipe_lsu_wavefronts (shared + global + local through the LSU data pipe; the "
      "shared-only family is ABSENT), backed out with an assumed peak of 1 per clk per SM. The ideal is from source, "
      "not from an opcode mix (absent). q for the ideal: exact 1 where routing proves it, else the HMMA proxy.\n")
    w("## 7. Resources\n")
    w("| kernel | shape | regs (alloc) | block | grid | smem static/dyn/driver KB | occ limit regs/smem/warps (blocks) | "
      "waves | spill insts | local spill B | local spill req |")
    w("|" + "---|" * 11)
    for a in A:
        ol = a["occ_limits"]
        w(f"| {a['kernel']} | {a['shape']} | {f_(a['regs'],0)} ({f_(a['regs_alloc'],0)}) | {f_(a['block'],0)} | "
          f"{f_(a['grid'],0)} | {f_(a['smem_static_B']['med']/1e3,3)}/{f_(a['smem_dynamic_B']['med']/1e3,3)}/"
          f"{f_(a['smem_driver_B']['med']/1e3,3)} | {f_(ol['registers'],0)}/{f_(ol['shared_mem'],0)}/{f_(ol['warps'],0)} | "
          f"{f_(a['waves'],0)} | {f_(a['spill_insts'],0)} | {f_(a['spill_local_B'],0)} | {f_(a['local_spill_requests'],0)} |")
    w("\n## Derivations and checks\n")
    w("- `l1tex__m_xbar2l1tex_read_sectors_mem_dshared` is distributed-shared-memory (cluster DSMEM) traffic: 0 in every "
      "Engine A launch (no clusters). It does NOT measure bulk copies into shared memory.")
    w(f"- L2->SM bytes = l1tex__m_xbar2l1tex_read_sectors pct x {XBAR_PEAK_SECT} sectors/clk/SM x 32 B x 132 x "
      "gpc__cycles_elapsed.avg (peak assumed). On clean launches (passes and routes known exactly) xbar / geometry:")
    for k, c in o["calibration"].items():
        w(f"  - {k}: n={c['n_clean']}, xbar/geometry med {c['xbar_over_geometry_med']:.4f} "
          f"[{c['xbar_over_geometry_min']:.4f}, {c['xbar_over_geometry_max']:.4f}], residual {c['xbar_resid_passes_med']:+.3f} "
          f"passes; DRAM/We - P med {c['dram_offset_passes_med']:+.3f} [{c['dram_offset_min']:+.3f}, {c['dram_offset_max']:+.3f}]; "
          f"lts lookup-miss bytes / weight bytes {c['ltsmiss_over_weights_med']:.3f}")
    w("- Pass estimator for non-clean launches: SM-average ALU-pipe busy cycles = slope x P + icpt, fitted on clean "
      "launches: " + "; ".join(f"{k}: slope {c['alu_slope_per_pass']:.0f}/pass, icpt {c['alu_icpt']:.0f}, resid sd "
      f"{c['alu_resid_sd_passes']:.3f} passes" for k, c in o["calibration"].items()) + ". The w13 slope is identical for "
      "v1 and v4 (same dequant code per weight). xbar over-reads at verify 16 by ~0.5 pass, so it is a cross-check only.")
    h = o["hmma_proxy_calibration"]
    w(f"- HMMA proxy: v4 q=1 launches (n={h['n_q1_v4']}) give {h['tensor_busy_cyc_per_rec_q1_v4_median']:.1f} tensor-busy "
      f"cycles per record [{h['tensor_busy_cyc_per_rec_q1_v4_min']:.1f}, {h['tensor_busy_cyc_per_rec_q1_v4_max']:.1f}]; the "
      f"v1 kernel (q=1, different code) gives {h['v1_q1_tensor_busy_cyc_per_rec_median']:.1f} (ratio {h['v1_over_v4_q1']:.3f}); "
      f"implied {h['implied_tensor_cyc_per_hmma_per_smsp']:.2f} busy cycles per m16n8k16 per SMSP. The tensor-busy count is "
      "exact (identical to 4+ digits at q=1 across 24 v4 and 12 v1 launches of different sizes), so HMMA = proxy x steps "
      "is an exact count: per launch HMMA / (steps per pass) is an integer N = P + (two-tile passes), listed in "
      "tables.json as hmma_total_over_steps_per_pass. Normalization: HMMA per step = 1.000 wherever routing proves q = 1; "
      f"{h['n_launches_integer_N_and_within_routing_q_bounds']}/{h['n_launches']} launches give an integer N inside the "
      "routing bounds on q (a wrong P would generally break one of the two).")
    w("- I_step HW = sm__inst_executed pct x 4 warp-inst/clk/SM x 132 x elapsed / steps; cross-check = "
      "sm__warps_active.avg.per_cycle_active x sm__cycles_active.sum / smsp__average_warps_active_per_inst_executed. "
      "I_step SASS = `inst_executed` (SASS-patching count, its own replay pass, = sass__inst_executed_per_opcode total). "
      "smsp__thread_inst_executed_per_inst_executed ~31 confirms warp-level counts.")
    w("- sass__inst_executed_per_opcode_pipeline total exceeds inst_executed (an instruction can count in more than one "
      "pipe), so pipeline totals are not additive.")
    w("\n## Absent metric families\n")
    for k, v in o["absent_metric_families"].items():
        w(f"- **{k}**: " + "; ".join(v))
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
