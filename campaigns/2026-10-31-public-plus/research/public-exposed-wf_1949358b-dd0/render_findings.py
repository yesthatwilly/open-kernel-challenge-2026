import json, ast, sys, collections
J, out = sys.argv[1], sys.argv[2]
lab, res = {}, {}
for l in open(J):
    d = json.loads(l)
    if d.get('type') == 'started': lab[d['agentId']] = d['label']
for l in open(J):
    d = json.loads(l)
    if d.get('type') != 'result': continue
    r = d['result']
    if not isinstance(r, dict):
        try: r = json.loads(r)
        except Exception: r = ast.literal_eval(r)
    res[lab[d['agentId']]] = r
def c(s, n=None):
    s = '' if s is None else str(s)
    s = s.replace('|', '/').replace('\n', ' ')
    return s if n is None else (s if len(s) <= n else s[:n - 1] + '…')
P = [('attention', 'muse_attention_decode'), ('decode', 'muse_decode_layer'), ('moe', 'muse_moe_w4a16'),
     ('bf16', 'muse_bf16_proj'), ('prefill', 'muse_prefill_layer')]
o = []
o.append('# Public-plus FINDINGS: our findings against the proven public files (alygg77@6cae49a)\n')
o.append('Generated from the read-only workflow `wf_1949358b-dd0` (journal in `research/public-exposed-wf_1949358b-dd0/`).')
o.append('That run had 5 inventory agents, 5 adversarial verifiers and 1 "why" agent, and finished 2026-10-31 00:14Z.')
o.append('PLAN.md sections 3-4 hold the summary.\n')
o.append('**Contamination note:** this file describes the public kernels at line level. arch-mirror clean-room writers must count reading it as a public read.\n')
o.append('Verifier caveats:')
o.append('- Each problem\'s verifier listed misclassified inventory rows (about 11% overall). They are in the per-problem "verifier corrections" lists.')
o.append('- Line refs are into `~/git/okc-muse-spark-kernels/kernels/<problem>.py` (PUB) unless a path says otherwise.\n')
o.append('## Results so far\n\nNo checks yet (see RESULTS.tsv).\n')
w = res.get('why-gap', {})
o.append('## Why the gap persists (why-gap agent)\n')
o.append(c(w.get('one_paragraph_answer')) + '\n')
o.append('### Ladder arithmetic\n\n' + str(w.get('ladder_arithmetic', '')) + '\n')
o.append('### Structural reasons\n\n| reason | evidence | weight |\n|---|---|---|')
for s in w.get('structural_reasons', []):
    o.append(f"| {c(s.get('reason'))} | {c(s.get('evidence'))} | {c(s.get('weight'))} |")
o.append('\n### Value/decision transfer failures\n')
o += ['- ' + c(x) for x in w.get('value_transfer_failures', [])]
o.append('\n### What worked\n')
o += ['- ' + c(x) for x in w.get('value_transfer_successes', [])]
o.append('\n### What a public-base track fixes / does not fix\n\n' + str(w.get('what_public_base_track_fixes', '')) + '\n\n' + str(w.get('what_it_does_not_fix', '')) + '\n')
o.append('### Top levers (any base)\n\n| lever | base | EV | evidence |\n|---|---|---|---|')
for t in w.get('top_levers', []):
    o.append(f"| {c(t.get('lever'))} | {c(t.get('base'))} | {c(t.get('ev'))} | {c(t.get('evidence'))} |")
for short, p in P:
    inv, ver = res.get('inv:' + short, {}), res.get('verify:' + short, {})
    o.append(f'\n## {p}\n')
    rel = collections.Counter(e.get('public_relation') for e in inv.get('inventory', []))
    o.append(f"{inv.get('claims_read')} mechanisms. Relations: " + ', '.join(f'{k} {v}' for k, v in rel.most_common()) + '\n')
    o.append('### Public file facts\n\n' + str(inv.get('public_file_facts', '')) + '\n')
    o.append('### Public compliance flags\n')
    o += ['- ' + c(x) for x in inv.get('public_compliance_flags', [])]
    o.append('\n### Graft-onto-public candidates and verifier verdicts\n\n| id | verdict | our finding | public target | predicted (inventory) | corrected (verifier) | verifier evidence |\n|---|---|---|---|---|---|---|')
    vd = {v['id']: v for v in ver.get('verdicts', [])}
    for g in inv.get('graft_candidates', []):
        v = vd.get(g['id'], {})
        o.append(f"| {c(g['id'])} | **{c(v.get('verdict', 'n/a'))}** | {c(g.get('our_finding'))} ({c(g.get('cite'))}) | {c(g.get('public_target'))} | {c(g.get('predicted_effect'))} | {c(v.get('corrected_prediction'))} | {c(v.get('evidence'))} |")
    o.append('\n### Candidates the verifier added\n')
    o += ['- ' + c(x) for x in ver.get('missed_candidates', [])]
    o.append('\n### Public mechanisms carrying the unmet gains\n\n| mechanism | rows | ours (us) | theirs (us) | share | where (PUB) | our attempts | why not transferred |\n|---|---|---|---|---|---|---|---|')
    for u in inv.get('public_unmet_gains', []):
        o.append(f"| {c(u.get('mechanism'))} | {c(u.get('rows'))} | {c(u.get('our_us'))} | {c(u.get('their_us'))} | {c(u.get('est_headline_share'))} | {c(u.get('public_location'))} | {c(u.get('our_attempts'))} | {c(u.get('why_not_transferred'))} |")
    o.append('\n### Verifier corrections to the unmet-gains list\n')
    o += ['- ' + c(x) for x in ver.get('unmet_gain_corrections', [])]
    o.append('\n### Why the gap persists here (inventory agent)\n\n' + str(inv.get('why_gap_persists', '')) + '\n')
    o.append('### Full inventory (newest CLAIMS line per mechanism)\n\n| slug | tag | relation | mechanism | rows | our measured | public evidence | cite |\n|---|---|---|---|---|---|---|---|')
    for e in inv.get('inventory', []):
        o.append(f"| {c(e.get('slug'))} | {c(e.get('tag'))} | {c(e.get('public_relation'))} | {c(e.get('mechanism'))} | {c(e.get('rows'))} | {c(e.get('our_measured'))} | {c(e.get('public_evidence'))} | {c(e.get('cite'))} |")
    o.append('\n### Verifier: inventory misclassifications\n')
    o += ['- ' + c(x) for x in ver.get('inventory_misclassifications', [])]
open(out, 'w').write('\n'.join(o) + '\n')
print(len(o), 'lines')
