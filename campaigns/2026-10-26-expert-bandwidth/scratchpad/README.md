# Scratchpad copies (session 3734a126, copied 2026-10-26T23:51Z)

These are copies of files that lived only in that session's /tmp scratchpad, which a new session cannot find. The
originals ran from /tmp, so some scripts hardcode paths: check a script's paths before running it. Read DISTILL.md §1
first.

## Reset-plan drafts (fill `STAMP` with `date -u +%Y-%m-%dT%H%MZ`; keep the `**Job:** \`<job id>\`` placeholder)

| file | use |
|---|---|
| `submit_attempt.md` | decode scored run of check f59cb301's bytes (`submission.edoff-27ef83df.py`). Goes to the worktree's `muse_decode_layer/attempts/<STAMP>-submit-campaign-wghyb-pw0-decode.md`. |
| `dqab_attempt.md` | moe check at eacf0722: ladder dq A/B (dq2 / dq3 / dq4), dq_exact with crosstalk, hybrid H=-8, v4c. Set `COMMIT` to eacf0722. |
| `anat_attempt.md` | decode check at eacf0722: expert-block anatomies, H=-8, v4c, Engine D off. Set `COMMIT` to eacf0722 and `BOARD` to the decode board after the scored run. |

## LOG tooling

| file | use |
|---|---|
| `now.md` | an old (20:57Z) template of the LOG "## Now" block. Do not reuse it as is. |
| `log_now.py` | `python3 -I log_now.py NOW_FILE [ENTRY_FILE]` replaces LOG.md's Now block and appends an entry. **Trap:** it overwrites Now without reading it. Twice it destroyed a user steer that had been put there. Read the current Now block first and carry every user bullet forward. |

## Run comparison and queue tools

| file | use |
|---|---|
| `rowcmp2.py` | `python3 -I rowcmp2.py OLD.json[.gz] NEW.json[.gz]`: decode / moe rows, each against its own run's SGLang time. Use it to separate code gains from SGLang drift. |
| `rowcmp.py` | an older form of rowcmp2 (ours only) |
| `bfcmp.py` | the bf16 per-shape compare (load, layer, m) of two run JSONs |
| `hypsum.py` | `python3 -I hypsum.py [problem ...]`: a compact summary of queued hypotheses (the sprint's and the reviewed ones) |
| `sim_routing.py` | a CPU model of the decode judge's routing (reference.py's structure, our own seeds) against the X0a surrogate basis. It prints the busiest range's routes, distinct experts, max and hot rows per load. |

## bulk12 (bf16 lossless 12-bit weights): models and the section source

| file | use |
|---|---|
| `bulk12/b12_section.py` | section (10) as first written, for reference. The live code is section (10) of the campaign branch's `muse_bf16_proj/submission.py`. |
| `bulk12/b12_consts.py`, `b12_funcs.py`, `b12_py_new.py` | the Python side of the m >= 8 refactor: B12_LAYERS / B12_CELLS, `_b12_say`, `_b12_cell`, routes |
| `bulk12/b12_emul.py` | an independent numpy emulation of the encoder, the kernel decode (hi4 / frag), the PTX m16n8k16 fragment maps (written from the ISA tables), the escape block and the m == 1 epilogue. Reuse it to verify the decode port. |
| `bulk12/b12sim.py` | a numpy model of `_b12_encode`, `b12_unpack_kernel` and the m == 1 epilogue |
| `bulk12/esc_sim.py` | the encoder's base choice and per-CTA escape counts per layer |
| `bulk12/ex_model.py`, `smem_model.py` | b12x epilogue bank-conflict and smem-layout models (the audit's MAJOR fix) |
| `bulk12/b12r_audit_report.txt` | the b12r audit, no BLOCKER or MAJOR. Open MINOR: `inst >= 18` is brittle, so use an explicit B12R id set. |

## Engine A emulators and the dq exhaustive check

| file | use |
|---|---|
| `emu/ea_layout_emu.py` | a CPU emulation of Engine A's record layout against consume()'s device decode. Exact per element. |
| `emu/ea2_map_emu.py` | emulates eak2::build_map: positions, passes, the stream-K partition and ticket targets |
| `dqcheck/gen.py` + `main.cpp` (+ generated `gen.inc`) | a host C++ harness built from the real ea_common.cuh / ea_ladder.cuh and `_ea_luts`. It checks the dq3 rotation extraction over all 2^32 code words (0 mismatches). `gen.py` hardcodes the worktree path. Build with `g++ -O2 main.cpp`. |

## Reviews (Opus, adversarial; also recorded in the hypotheses' notes by churn review-record)

| file | hypothesis | verdict |
|---|---|---|
| `reviews/rev_sliding_s3.json` | attention `sliding-decode-sk-s3-fused-preattn-rope` | proceed, with two blocking builder fixes |
| `reviews/rev_combo.json` | bf16 `combo-verify-cells-pooler-khalf-plus-qkv-rowowner-m8-m16` | proceed narrowed; I held it on the rules question (bf16 partials) |
| `reviews/rev_mc4.json` | prefill `experts-rega-mc4-cluster-high-ft` | hold: its premise is falsified |
