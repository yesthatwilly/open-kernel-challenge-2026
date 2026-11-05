# musespark: a systematic run at the Muse Spark Open Kernel Challenge

Team `yesthatwilly`. Kernel track, 2026-10-22 to 2026-11-04 23:59 PT. This repository is the complete
record of that run: every kernel file sent to the judge, the attempt note written before each send, the
per-problem ledgers of what was tried and what it measured, the campaign plans and logs, the H100/H200
proxy harnesses, and the third-party notices for the public code the final entries are built on.

The method is the point. See [METHODOLOGY.md](METHODOLOGY.md) for how the work was run and how the
philosophy changed along the way.

## The challenge

The [Muse Spark Open Kernel Challenge](https://openkernelchallenge.dev/) (Paradigm, with Meta and
Andromeda) asks for faster serving of Meta's Muse Spark Open, an open mixture-of-experts model with 256
experts and 4-bit (NVFP4) expert weights, on NVIDIA H200s. Two tracks:

- **Kernel track** (this repo): five problems, each a single Python file with a `prepare()` for weight
  repacking and a timed call. The judge checks every call against a reference on fresh inputs, then
  times it against SGLang's own kernel in the same run. A problem's score is the geometric mean over its
  timed shapes of SGLang time divided by ours. The board keeps each team's best scored run. Limits were
  10 scored runs and 20 checks per problem per day (doubled for the final day), 15 minutes per run,
  1,000,000 bytes per file.
- **Serving track** (Nov 5 to Nov 19): a patch to SGLang scored by throughput on a whole 8×H200 node.
  Prizes are paid there. The top 50 on the overall kernel board get serving-queue priority.

The five kernel problems and what they time:

| Problem | What it is | Timed shapes |
|---|---|---|
| `muse_decode_layer` | one whole decoder layer (attention, dense projections, router, routed experts) | decode and verify (q_len 1 and 8) × batch 1/16/32 × sliding/full attention |
| `muse_attention_decode` | gated attention decode over an FP8 paged KV cache | same 12 shapes |
| `muse_moe_w4a16` | the routed experts: NVFP4 weights, BF16 activations, 256 experts | token counts 1 to 256 |
| `muse_bf16_proj` | the dense projections and the FP32 router | decode and verify at batch 1/16/32 |
| `muse_prefill_layer` | the prefill layer | long prompts |

Hard rules that shaped everything: CUDA-graph capturable calls (no host syncs, no data-dependent
control flow), fresh inputs every call, lossless weight repacking only, BF16 activations where SGLang
keeps BF16, FP8 KV cache, attention over every key, and from 2026-11-03 an explicit ban on compressed or
encoded source. Violations are removed on review; the final ranks come from organizer reruns on held-out
shapes. The local rules mirror is in [rules/](rules/).

## Results

**Status: provisional, as of 2026-11-05 04:00 UTC, four hours before the close.** The organizers rerun
the top entries on held-out shapes after the close; the finalized ranks will replace this table when they
are published.

| Problem | Our best | Rank | Board leader | Start of the run |
|---|---:|---:|---:|---:|
| muse_decode_layer | 2.284× | 139 | 2.576× | 1.316× (10-26, rank 448) |
| muse_attention_decode | 2.703× | 101 | 3.339× | 1.634× (10-27) |
| muse_moe_w4a16 | 2.414× | 151 | 2.788× | 1.402× (10-26, rank 490) |
| muse_bf16_proj | 1.584× | 140 | 1.728× | 1.257× (10-26) |
| muse_prefill_layer | 1.947× | 49 | 2.147× | 1.490× (10-27) |
| **Overall** | | **139 of ~2,000** | | 480 of 1,163 (10-26) |

Speedups are over SGLang's own kernel on the judge's H200. "Start of the run" is the board entry when
the systematic process began on 2026-10-26/27.

The entries that hold those rows are mostly public, Apache-2.0 code, adopted with attribution and
improved with audited changes of our own (see "Lineage" below and [third_party/](third_party/)). The
honest summary of the last four days is that the public files from keell0renz and alygg77/vjeux are the
fastest known kernels for these problems, our contributions on top of them are riders worth 0.1% to 2%
per row, and a complete, documented process is what this repo adds.

## Lineage of the final files

| Problem | Base | Our landed changes on top |
|---|---|---|
| decode | keell0renz KZ-r3 (@890809c) | ZL guard in the WGS stage-range planner (a hang fix for launches with no local route) |
| attention | keell0renz attention + the KZA knob set | our RXA build for batch-1 q_len-1 sliding decode, where it is faster; HSK8 head split and the relaxed q8 softmax on the earlier base |
| moe | alygg77 lineage + keell0renz MFIN/WPRO | FRX (bulk copy of x into dead ring smem, route tables before the join), NRB4, MD16F fence, B4X placement, SC6 scales only for 16<M≤32, concurrent compile of prepare's variants |
| bf16 | keell0renz KZ-r3, plain source (its zlib.crc32 replaced by hashlib) | landed on the earlier base before adoption: P1F static cuBLASLt plan, u1pin unpooler route, ring-slot WAR hardening, every compressed payload replaced by readable source |
| prefill | keell0renz KZ-r3 | PFC2L work-queue reorder (bit-identical output) on the earlier base |

Before 2026-10-31 every file was our own. Those files are kept at git tag `ours-final-1031` (decode
1.699×, attention 2.005×, moe 2.100×, bf16 1.343×, prefill 1.572×).

## What is where

| Path | Contents |
|---|---|
| `muse_<problem>/submission.py` | the current master file for that problem, byte-identical to its scored run |
| `muse_<problem>/attempts/` | one dated note per judge send, written before the send (468 in all); a hook refuses a send without one |
| `muse_<problem>/hypotheses/CLAIMS.md` | the append-only ledger of mechanisms: landed, lead, shelf, refuted, with evidence (792 lines across the five problems) |
| `muse_<problem>/hypotheses/{queue,running,done}/` | the hypothesis lifecycle run by `/hypothesize` and `/churnsubs` (416 completed records) |
| `campaigns/` | multi-day programs larger than one hypothesis: plan, findings, log, runs, retrospective; [INDEX.md](campaigns/INDEX.md) lists them |
| `rules/` | local mirror of the competition rules and the dated updates |
| `third_party/` | upstream LICENSE, NOTICE and exact commit for every public repository used |
| `.claude/skills/` | the project skills that ran the loop: `/hypothesize`, `/churnsubs`, `/3to1`, with `churn.py` and `okc_lint.py` |
| `.worktrees/` (ignored) | one git worktree per campaign or hypothesis branch; 59 at the close, all committed on their branches |

Repository statistics at the close: 4,131 commits on master over 13 days, 59 branches, 468 attempt
notes, 792 claim lines.

## Acknowledgements

Public kernels and measurements from keell0renz, alygg77, vjeux, trxphxle, chenwainuo, ZhimaoL and
Nathen Fernandes, under Apache-2.0 and MIT, made the final entries what they are; each is named in the
header of every file that carries their code and in [third_party/README.md](third_party/README.md).
SGLang and NVIDIA CUTLASS underlie all of it.
