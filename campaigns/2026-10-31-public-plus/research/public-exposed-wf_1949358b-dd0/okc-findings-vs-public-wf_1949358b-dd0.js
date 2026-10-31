export const meta = {
  name: 'okc-findings-vs-public',
  description: 'Read-only: inventory every OKC finding per problem, classify it against the public (alygg77/vjeux) kernels, adversarially verify graft-onto-public candidates, and diagnose why the gap persists',
  phases: [
    { title: 'Inventory', detail: 'one agent per problem: our CLAIMS/FINDINGS vs the public file and docs' },
    { title: 'Verify', detail: 'one skeptic per problem: refute each graft-onto-public candidate against the public source' },
    { title: 'Why', detail: 'cross-cutting diagnosis of why the gap stayed open after a day of rungs' },
  ],
}

const REPO = '/home/user/git/musespark'
const PUB = '/home/user/.git-pub-placeholder'
const RO = `STRICTLY READ-ONLY. Do not edit, write, create, move or delete any file. Do not run git checkout/switch/reset/stash/worktree add/commit, okc, hopper-probe, hopper-queue, ssh, or anything that touches gpunode06 or the network. Other sessions have agents running in worktrees under ${REPO}/.worktrees/ right now; only read them. Read branch-only files with \`git -C ${REPO} show <branch>:<path>\` or by reading the worktree path directly. Cite every claim as path:line (or branch:path:line).`

const SOURCES = `Sources (read what you need; grep large files instead of reading them whole):
OURS
- ${REPO}/<P>/hypotheses/CLAIMS.md on master (curated, one line per mechanism; newest line per mechanism wins). ALSO the newer lines on branches: \`git -C ${REPO} show campaign/arch-mirror:<P>/hypotheses/CLAIMS.md\` and \`git -C ${REPO} show campaign/union-confirm:<P>/hypotheses/CLAIMS.md\` (diff them against master's to find branch-only lines).
- ${REPO}/<P>/hypotheses/FINDINGS.md (has "where the best spends its time" profile + Do-not-try), DIAGNOSIS.md, INDEX.md, REWRITE_READ_2026-10-25.md, BEST.json; done/<outcome>/*.md for detail when a CLAIMS line is terse.
- ${REPO}/hypotheses/LESSONS.md (cross-problem lessons).
- Public-ladder campaign (files on disk in ${REPO}/.worktrees/arch-mirror/campaigns/2026-10-30-public-ladder/): PLAN.md, LADDER.md, FINDINGS.md (its §1 = transfer rules), REPLICATION.md, C-CAMPAIGNS.md, LOG.md, research/*-1030.md (rung notes; dattn-1030.md = side-by-side per-kernel profile of their attention vs ours; moe-gap-1030.md; attnfull-1030.md), research/gap-ledger-20261030.txt.
- Arch-mirror campaign: ${REPO}/.worktrees/arch-mirror/campaigns/2026-10-31-arch-mirror/{PLAN.md,LOG.md,LANES.tsv,ARCH-attn-full.md,ARCH-attn-sliding.md}.
- Union-confirm: \`git -C ${REPO} show campaign/union-confirm:campaigns/2026-10-31-union-confirm/LOG.md\`.
- Rank ladder: ${REPO}/campaigns/2026-10-30-rank-ladder/{PLAN.md,LOG.md,LANES.tsv} and research/ if present.
- H100 ledger: ${REPO}/campaigns/2026-10-26-expert-bandwidth/research/h100-findings.md and h100-*.md (moe/decode expert block mostly).
- Our current submission: ${REPO}/<P>/submission.py (master); the union-confirm branch may have a newer attention file.
PUBLIC ("the proven repo", alygg77@6cae49a, built on vjeux's Apache-2.0 base; bf16 also carries Nathen Fernandes MIT routes)
- /home/user/git/okc-muse-spark-kernels/kernels/<P>.py (the judge-proven file), RESULTS.md (per-row judge times), docs/H200_MEASURED_FACTS.md, docs/NEGATIVE_RESULTS.md (their closed levers), README.md, NOTICE, timing_kit/.`

const PROBLEMS = [
  { p: 'muse_attention_decode', short: 'attention', ours: '1.903 (master 3d4f06b2, scored 47b125ee)', pub: '2.6530',
    note: 'vjeux public file submitted unchanged by alygg77. Rows: 12 (sliding/full x decode1/16/32 x verify1/16/32). We already ran a side-by-side per-kernel profile (dattn-1030.md, CLAIMS commit 6c1e259e: sliding transfers 1:1 H100->H200, full rows issue-bound on H200). Landings so far: AT2/AT3b/AT8 family, AT8F, S1 single-launch sliding verify B=1.' },
  { p: 'muse_decode_layer', short: 'decode', ours: '1.699 board (check de3dfe5e 1.7165 beat it; scored cb182920 crashed in setup on the dense-swapab prepare race)', pub: '2.2058',
    note: 'Decode composes attention + bf16 projections + MoE expert block. Ours had a router-precision compliance fix (ROUTER_FP32_MAX_TOKENS); public facts say their fix cost ~3.8% on v1 rows. Byte budget: ~3.3 KB free of 1,000,000 B. D-decode (their file on nodejudge) was blocked by shims.' },
  { p: 'muse_moe_w4a16', short: 'moe', ours: '2.100', pub: '2.4015',
    note: 'Expert-bandwidth campaign (ea4 engine, swap-AB, fork, hot/cold, EPF, UFS, MO3) is the biggest body of our work. D-moe harvest: gap in ea4x_w13 (moe CLAIMS ~131). Node routing draws break transfer (node 0.913 vs judge 0.963). File is ~982 KB of the 1,000,000 B limit.' },
  { p: 'muse_bf16_proj', short: 'bf16', ours: '1.343 (BF4e landed)', pub: '1.5502',
    note: '30 rows (5 layers x 6 loads). Ladder rungs BF1 sk2 fragment regstream, BF4 router, BF4e router cluster push reduce, BF5 pooler; VT of public sk2 launch values refuted on H100; BF2/BF2R refuted; 12-bit stream decode-issue-bound. Public file contains Nathen Fernandes MIT routes (m16 pooler, m8 O).' },
  { p: 'muse_prefill_layer', short: 'prefill', ours: '1.572', pub: '1.8167',
    note: 'vjeux public file submitted unchanged by alygg77. 8 rows; t1s/fts/t4s heaviest. Our prefill has rega A3/A5b landings, PF3 slab sharing shelf; arch-mirror cut prefill (no oracle).' },
]

const INV_SCHEMA = {
  type: 'object',
  properties: {
    problem: { type: 'string' },
    claims_read: { type: 'integer', description: 'number of distinct mechanisms (newest line per mechanism) inventoried' },
    inventory: {
      type: 'array',
      description: 'EVERY distinct mechanism from CLAIMS (all tags), newest line per mechanism, branch-only lines included',
      items: {
        type: 'object',
        properties: {
          slug: { type: 'string' },
          tag: { type: 'string', description: 'landed|lead|shelf|unresolved|node-only|refuted|superseded|diag' },
          mechanism: { type: 'string' },
          rows: { type: 'string' },
          our_measured: { type: 'string', description: 'the number(s) we measured, with where (judge/H100) ' },
          public_relation: { type: 'string', enum: ['public_does_same', 'public_does_differently', 'public_lacks', 'public_negative_agrees', 'public_negative_contradicts', 'public_positive_contradicts_our_refutation', 'not_comparable'] },
          public_evidence: { type: 'string', description: 'public file path:line or doc path:line supporting the relation' },
          cite: { type: 'string', description: 'our CLAIMS path:line' },
        },
        required: ['slug', 'tag', 'mechanism', 'public_relation', 'cite'],
      },
    },
    public_unmet_gains: {
      type: 'array',
      description: 'mechanisms/architecture in the public file that produce gains we have NOT matched, ranked by row-weight x gap',
      items: {
        type: 'object',
        properties: {
          mechanism: { type: 'string' },
          rows: { type: 'string' },
          their_us: { type: 'string' }, our_us: { type: 'string' },
          public_location: { type: 'string', description: 'kernels/<P>.py line ranges' },
          our_attempts: { type: 'string', description: 'CLAIMS lines where we tried it or something like it, and outcome' },
          why_not_transferred: { type: 'string' },
          est_headline_share: { type: 'string', description: 'share of the headline gap this explains, with arithmetic' },
        },
        required: ['mechanism', 'rows', 'public_location', 'why_not_transferred'],
      },
    },
    graft_candidates: {
      type: 'array',
      description: 'findings of OURS that could plausibly make the PUBLIC file faster if applied on top of it (things public lacks or does worse, that we measured to help). Include aggressive/low-confidence ones, flagged as such.',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          our_finding: { type: 'string' }, cite: { type: 'string' },
          public_target: { type: 'string', description: 'which function/kernel/route in the public file it would change, path:line' },
          rows: { type: 'string' },
          predicted_effect: { type: 'string', description: 'per-row us and headline estimate, with arithmetic' },
          confidence: { type: 'string', enum: ['high', 'medium', 'low', 'speculative'] },
          transfer_risk: { type: 'string', description: 'why it might not apply on their architecture' },
          h100_prescreenable: { type: 'boolean' },
          compliance_notes: { type: 'string' },
          byte_cost: { type: 'string' },
          effort: { type: 'string', description: 'rough build effort (lines, CUDA vs python knob)' },
        },
        required: ['id', 'our_finding', 'cite', 'public_target', 'rows', 'predicted_effect', 'confidence', 'transfer_risk'],
      },
    },
    public_compliance_flags: { type: 'array', items: { type: 'string' }, description: 'anything in the public file that may violate rules/technical-rules.llms.txt or the TASK.md rules (precision, caching, fallback, host sync), with path:line; empty if none found' },
    public_file_facts: { type: 'string', description: 'byte size, import/prepare/compile cost hints, nvcc use, tuned tables (e.g. _TUNED), attribution header lines, anything needed to run it as an okc check' },
    why_gap_persists: { type: 'string', description: 'evidence-based diagnosis for this problem: why one-mechanism rungs on our base have not closed the gap' },
  },
  required: ['problem', 'claims_read', 'inventory', 'public_unmet_gains', 'graft_candidates', 'public_compliance_flags', 'public_file_facts', 'why_gap_persists'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    problem: { type: 'string' },
    verdicts: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          verdict: { type: 'string', enum: ['survives', 'already_in_public', 'not_transferable', 'unsupported_by_our_evidence', 'non_compliant', 'too_small_to_see'] },
          evidence: { type: 'string', description: 'public path:line and our path:line that decide it' },
          corrected_prediction: { type: 'string' },
        },
        required: ['id', 'verdict', 'evidence'],
      },
    },
    missed_candidates: { type: 'array', items: { type: 'string' }, description: 'graft candidates the inventory agent missed, with cites' },
    unmet_gain_corrections: { type: 'array', items: { type: 'string' }, description: 'errors in the public_unmet_gains list (wrong line refs, wrong numbers, mechanism mis-described)' },
    inventory_misclassifications: { type: 'array', items: { type: 'string' }, description: 'inventory entries whose public_relation is wrong, with cites' },
  },
  required: ['problem', 'verdicts', 'missed_candidates', 'unmet_gain_corrections', 'inventory_misclassifications'],
}

function invPrompt(x) {
  return `You are inventorying one problem of the Muse Spark Open Kernel Challenge repo at ${REPO} for the user, and comparing it with the public, judge-proven kernels. ${RO}

Problem: ${x.p} (${x.short}). Our board: ${x.ours}. Public judge-proven: ${x.pub}x. ${x.note}

${SOURCES.split('<P>').join(x.p)}

Tasks:
1. INVENTORY every distinct mechanism our repo has a finding about for this problem (CLAIMS.md is the curated list; include branch-only lines from campaign/arch-mirror and campaign/union-confirm; newest line per mechanism wins). For each, classify its relation to the public file by READING the public kernel at the relevant place (grep for the kernel/route names, launch configs, tile sizes, merge scheme, PDL, persistent grid, cluster/DSMEM, swap-AB, split-K, stream-K, precision handling, etc.). Classes: public_does_same (same mechanism present), public_does_differently, public_lacks (public does not do it), public_negative_agrees (their NEGATIVE_RESULTS also found it useless/harmful), public_negative_contradicts (their negative results say it fails but we measured it helping, or vice versa), public_positive_contradicts_our_refutation (we refuted it but public uses it successfully), not_comparable.
2. PUBLIC UNMET GAINS: from RESULTS.md per-row judge times vs our latest judge rows (BEST.result.json / the scored job records under ${x.p}/hypotheses/archive or attempt files / gap ledger), identify which public mechanisms/architecture carry the gap, with their public_location line ranges and whether/how we tried them (cite CLAIMS). Use our side-by-side profiles (dattn-1030.md, moe-gap-1030.md, arch-mirror ARCH files, CLAIMS lines mentioning "public"/"theirs"/"D-") wherever they exist — measured beats inferred.
3. GRAFT CANDIDATES: the user now wants an aggressive, check-only (non-scoring) experiment track whose base is the PUBLIC file, to test whether ANYTHING our repo has learned can improve the already-proven public record. List every finding of ours that could plausibly make the public file faster on some rows (things the public file lacks or does worse, and that we measured to help on our kernels, or H100/node facts that point at a public inefficiency, e.g. a tuned value, a launch-count reduction, PDL, a merge scheme, a route gate at a shape where we beat their per-row time). Compare per-row: on any row where OUR judge time beats THEIR judge time, name the mechanism responsible — that is the strongest graft class. Be aggressive: include low-confidence/speculative ones but label them. Predict effects with arithmetic on their row times and judge weights.
4. public_compliance_flags: read rules at ${REPO}/rules/technical-rules.llms.txt and ${x.p}/TASK.md Rules; flag anything in the public file that could be non-compliant (we must not send a non-compliant file even as a check).
5. public_file_facts: byte size (wc -c), whether it compiles CUDA at import (nvcc/load_inline time), tuned tables, attribution header, what an okc check of it would need (e.g. license header, nothing else).
6. why_gap_persists: an evidence-based diagnosis for this problem of why a day of one-mechanism rungs on our base did not close the gap (architecture vs mechanism, transfer failures, values that only work inside their structure, H100->H200 transfer, judge noise, compliance costs, byte budget, etc.). Cite numbers.

Be exhaustive on the inventory (every mechanism), precise on line citations, and honest where you could not verify something.`
}

function verifyPrompt(x, inv) {
  return `You are an adversarial skeptic checking another agent's comparison of our OKC repo against the public judge-proven kernel for ${x.p}. ${RO}

${SOURCES.split('<P>').join(x.p)}

The other agent produced this (JSON):
${JSON.stringify({ graft_candidates: inv.graft_candidates, public_unmet_gains: inv.public_unmet_gains, public_compliance_flags: inv.public_compliance_flags, inventory_sample_public_lacks: (inv.inventory || []).filter(e => e.public_relation === 'public_lacks' || e.public_relation === 'public_positive_contradicts_our_refutation' || e.public_relation === 'public_negative_contradicts') }, null, 1)}

For EACH graft candidate, try hard to REFUTE it by reading the public kernel /home/user/git/okc-muse-spark-kernels/kernels/${x.p}.py at the cited place and around it, and our CLAIMS evidence:
- already_in_public: the public file already does this (or something strictly better) on those rows;
- not_transferable: our measured effect depended on our architecture (e.g. it fixed a problem public does not have, or their structure makes it inapplicable);
- unsupported_by_our_evidence: the cited CLAIMS line does not show what is claimed (wrong tag, single noisy run, node-only that flipped on judge);
- non_compliant: would break a rule in ${REPO}/rules/technical-rules.llms.txt or TASK.md;
- too_small_to_see: below ~1/3 of judge row noise (rows drift 1.5-2.5% per shape between runs; one unchanged file swung ~6% on a row);
- survives: you could not refute it. Give a corrected_prediction if the numbers were off.
Default to a refuting verdict when the evidence is ambiguous. Also list graft candidates the agent MISSED (especially rows where our judge time beats theirs), errors in public_unmet_gains (line refs, numbers, mechanism descriptions), and inventory entries whose public_relation is wrong.`
}

const WHY_PROMPT = `You are diagnosing, for the user, why a full day (2026-10-30) of rigorous one-mechanism-per-rung testing toward a KNOWN public answer (alygg77@6cae49a: attn 2.653, decode 2.206, moe 2.402, bf16 1.550, prefill 1.817) only moved our boards to attn 1.903, decode ~1.70, moe 2.100, bf16 1.343, prefill 1.572. ${RO}

Read: ${REPO}/hypotheses/LESSONS.md; ${REPO}/hypotheses/DECISIONS.md (recent part); ${REPO}/hypotheses/COSTS.md (summary); ${REPO}/campaigns/2026-10-30-rank-ladder/{PLAN.md,LOG.md,LANES.tsv}; ${REPO}/.worktrees/arch-mirror/campaigns/2026-10-30-public-ladder/{PLAN.md,FINDINGS.md,LADDER.md,REPLICATION.md,C-CAMPAIGNS.md,LOG.md}; ${REPO}/.worktrees/arch-mirror/campaigns/2026-10-31-arch-mirror/{PLAN.md,LOG.md}; \`git -C ${REPO} show campaign/union-confirm:campaigns/2026-10-31-union-confirm/LOG.md\`; \`git -C ${REPO} log --since=2026-10-29T12:00 --pretty='%h %ad %s' --date=iso-strict master campaign/arch-mirror campaign/union-confirm campaign/public-ladder\`; the per-problem CLAIMS.md tails (last ~40 lines each, master + branches); public docs /home/user/git/okc-muse-spark-kernels/{README.md,RESULTS.md,docs/NEGATIVE_RESULTS.md,docs/H200_MEASURED_FACTS.md}.

Produce a quantitative diagnosis:
1. The ladder's arithmetic: rungs attempted, hit rate, gain per landing, checks/scored used, H100 hours, and the implied days-to-parity; what fraction of each problem's gap is architectural (whole-kernel structure, launch graph, fusion) vs single knobs.
2. Every recorded instance where moving a public VALUE or a single public DECISION onto our code failed (and why, per the records), vs instances where it worked.
3. Structural reasons: e.g. public gains are emergent from co-designed decisions (bundles), our base carries different structure (so knobs do not map), H100 vs H200 transfer of memory-bound rows, judge noise vs 1.2% margins, compliance fixes that cost us time the public may not pay (check whether the public file is compliant on the same points), byte budget, prepare/import clocks, process overhead (attempt files, audits, stops/starts by the user, agent caps).
4. Which of these an aggressive, check-only track that starts FROM the public file (adding our findings on top, never scored) would remove, and which it would not.
5. A short ranked list of the top 5 levers for the next day, with expected value, whichever base they use.
Cite path:line for every number.`

const WHY_SCHEMA = {
  type: 'object',
  properties: {
    ladder_arithmetic: { type: 'string' },
    value_transfer_failures: { type: 'array', items: { type: 'string' } },
    value_transfer_successes: { type: 'array', items: { type: 'string' } },
    structural_reasons: { type: 'array', items: { type: 'object', properties: { reason: { type: 'string' }, evidence: { type: 'string' }, weight: { type: 'string' } }, required: ['reason', 'evidence'] } },
    what_public_base_track_fixes: { type: 'string' },
    what_it_does_not_fix: { type: 'string' },
    top_levers: { type: 'array', items: { type: 'object', properties: { lever: { type: 'string' }, base: { type: 'string' }, ev: { type: 'string' }, evidence: { type: 'string' } }, required: ['lever', 'ev'] } },
    one_paragraph_answer: { type: 'string', description: 'plain-language answer to: why is having the right answer still so far away after a full day of testing' },
  },
  required: ['ladder_arithmetic', 'value_transfer_failures', 'structural_reasons', 'what_public_base_track_fixes', 'what_it_does_not_fix', 'top_levers', 'one_paragraph_answer'],
}

const whyP = agent(WHY_PROMPT, { label: 'why-gap', phase: 'Why', schema: WHY_SCHEMA, agentType: 'Plan' })

const per = await pipeline(
  PROBLEMS,
  x => agent(invPrompt(x), { label: `inv:${x.short}`, phase: 'Inventory', schema: INV_SCHEMA, agentType: 'Plan' }),
  (inv, x) => inv ? agent(verifyPrompt(x, inv), { label: `verify:${x.short}`, phase: 'Verify', schema: VERDICT_SCHEMA, agentType: 'Plan' }).then(v => ({ problem: x.p, inv, verify: v })) : { problem: x.p, inv: null, verify: null },
)

const why = await whyP
for (const r of per) {
  if (!r || !r.inv) { log(`inventory missing for ${r ? r.problem : '?'}`); continue }
  const surv = r.verify ? r.verify.verdicts.filter(v => v.verdict === 'survives').length : 'n/a'
  log(`${r.problem}: ${r.inv.claims_read} mechanisms, ${r.inv.graft_candidates.length} graft candidates, ${surv} survive`)
}
return { per, why }
