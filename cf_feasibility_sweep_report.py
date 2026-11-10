"""Render the complete prespecified penalty sweep from audited artifacts."""
import json
from cf_feasibility_comparison_report import HEADERS, format_cell


def render_sweep(root):
    prefix='trained_models/cf_feasibility_penalty_sweep'
    out=root/prefix
    data=json.loads((out/'summary.json').read_text())
    audit=json.loads((out/'audit.json').read_text())
    assert audit['passed'] and audit['runs']==36 and audit['new_trained_runs']==24
    assert data['alphas']==[.1,.3,1.,3.,10.]
    def m(setting,alpha,cohort='all'):
        return data['summary'][f'{setting}/{alpha:g}/{cohort}']
    def table(cohort,table_id):
        rows=[]
        for setting,weight in [('c1',192),('c2',87)]:
            for alpha in [0,*data['alphas']]:
                values=m(setting,alpha,cohort)
                label='Original baseline · α = 0' if not alpha else f'Penalties · α = {alpha:g}<br><span class="small">Added weight: {alpha*weight:g}</span>'
                css='comparison-original' if not alpha else 'comparison-loss_only'
                rows.append(f'<tr class="{css}"><th scope="row">{setting.upper()}</th><td class="model-name">{label}</td>'+''.join(format_cell(values[k],k) for k,_ in HEADERS)+'</tr>')
        headers=''.join(f'<th scope="col">{title}</th>' for _,title in HEADERS)
        return f'<div class="table-wrap comparison-scroll" tabindex="0" aria-label="Scrollable full coefficient sweep"><table class="comparison-metrics" id="{table_id}"><thead><tr><th scope="col">Exp.</th><th scope="col">Model / multiplier</th>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    seedrows=[]
    for row in data['rows']:
        if row['cohort']!='all': continue
        keys=('validity','raw_validity','proximity','viol_any','viol_age_decrease','out_disallowed_any')
        cells=''.join(format_cell(dict(mean=row[k],sd=None),k) for k in keys)
        seedrows.append(f'<tr><th>{row["setting"].upper()}</th><td>{row["alpha"]:g}</td><td>{row["seed"]}</td>{cells}</tr>')
    variation=[]
    for setting in ('c1','c2'):
        for alpha in [0,*data['alphas']]:
            values=m(setting,alpha)
            cells=''.join(f'<td>{values[k]["mean"]:.4f} ± {values[k]["sd"]:.4f}</td>' for k in ('validity','proximity','viol_any','viol_age_decrease'))
            variation.append(f'<tr><th>{setting.upper()}</th><td>{alpha:g}</td>{cells}</tr>')
    directions=''.join(f'<h3>{title} · {int(m("c1",0,cohort)["n_test"]["mean"]):,} observations</h3>'+table(cohort,'sweep-'+cohort)
        for cohort,title in [('low_to_high','Predicted low → high income'),('high_to_low','Predicted high → low income')])
    paragraphs=[]
    for setting in ('c1','c2'):
        original,weak,strong=m(setting,0),m(setting,.1),m(setting,10)
        paragraphs.append(f'<p><strong>{setting.upper()}.</strong> At α = 0.1, decoded validity is {100*weak["validity"]["mean"]:.2f}% and proximity is {weak["proximity"]["mean"]:.3f}; at α = 10, they are {100*strong["validity"]["mean"]:.2f}% and {strong["proximity"]["mean"]:.3f}. Across those two weights, the rate of changing at least one immutable feature changes from {100*weak["viol_any"]["mean"]:.2f}% to {100*strong["viol_any"]["mean"]:.2f}%. The original baseline has {100*original["validity"]["mean"]:.2f}% validity and {100*original["viol_any"]["mean"]:.2f}% immutable-feature violations.</p>')
    assert all(m(s,a)['validity']['mean'] < m(s,.1)['validity']['mean'] for s in ('c1','c2') for a in (.3,1,3,10))
    race_rates=[m(s,a)['viol_race']['mean'] for s in ('c1','c2') for a in data['alphas']]
    assert max(race_rates)-min(race_rates)<1e-12
    findings=f'''<div class="callout"><p><strong>What changing the coefficient achieved.</strong> α = 0.1 has the highest observed mean decoded validity among the five extensions in both settings, but still changes immutable features frequently. Validity is not monotonic in the coefficient: α = 0.3 performs below α = 1 in both settings. These are descriptive comparisons of this grid, not selection of a tuned final model.</p>
<p>Race violations remain {100*race_rates[0]:.2f}% at every tested extension weight. At α = 10, the fraction changing any immutable feature is still {100*m('c1',10)['viol_any']['mean']:.2f}% in C1 and {100*m('c2',10)['viol_any']['mean']:.2f}% in C2; age decreases remain {100*m('c1',10)['viol_age_decrease']['mean']:.2f}% and {100*m('c2',10)['viol_age_decrease']['mean']:.2f}%, respectively. Stronger penalties therefore do not eliminate violations in this experiment. The earlier projected variants enforce the encoded six-rule policy exactly; their reference results remain below.</p></div>'''
    section=f'''<div class="section-heading" id="coefficient-sweep-results"><span class="section-no">NEW</span><h2>Five penalty strengths for C1 and C2</h2></div>
<p class="intro">The complete sweep uses α = 0.1, 0.3, 1, 3 and 10, with three matched Generator seeds per setting. All <strong>24 additional fits are complete</strong>. The six existing α = 1 runs and six original baselines (α = 0) are reused read-only: 36 evaluated runs in this comparison. No policy projection is used.</p>
<p>Only the coefficient of the added feature penalties changes: <strong>α × 192 for C1 and α × 87 for C2</strong>. The complete original objectives, including their age penalties and C2’s education term, remain unchanged. <a href="#sweep-full-objectives">Full formulas and coefficient map</a>.</p>
<h3>Full coefficient sweep — test split (n = 8,141)</h3>
<p class="small">Means across seeds 42, 43 and 44; ten generated candidates per test observation and seed, counting both prediction directions. Scroll horizontally for all metrics. Hover over a value for its mean and seed standard deviation; the expandable table below also gives standard deviations explicitly. <a href="#loss-diagnostic-definitions">L₁–L₄ and column definitions</a>.</p>
{table('all','sweep-consolidated-results')}
{''.join(paragraphs)}
{findings}
<p>Accuracy remains {100*m('c1',0)['acc']['mean']:.2f}% in every setting because the classifier is frozen. The table measures the decoded feature vectors that could be delivered as counterfactuals; validity on raw categorical scores is reported separately below. Increasing a soft penalty does not provide the hard guarantee supplied by projection. Proximity includes every generated candidate, so a shorter distance must be read alongside validity and violations.</p>
<p>This is an exploratory sensitivity comparison on the existing test split, which had already been examined. Every prespecified coefficient and seed is reported; no coefficient, checkpoint or candidate is selected using test performance.</p>
<details class="reproduce"><summary>Seed variation · mean ± sample standard deviation</summary><div class="table-wrap"><table id="sweep-seed-variation"><thead><tr><th>Setting</th><th>α</th><th>Validity</th><th>Proximity</th><th>Any immutable violation</th><th>Age decrease</th></tr></thead><tbody>{''.join(variation)}</tbody></table></div><p class="small">Validity and violation rates use proportions in this table. Standard deviations describe variation across three Generator seeds, not confidence intervals across independent datasets.</p></details>
<details class="reproduce"><summary>Full sweep by prediction direction</summary>{directions}</details>
<details class="reproduce"><summary>All 36 runs · decoded and raw-score validity</summary><div class="table-wrap"><table id="sweep-all-seeds"><thead><tr><th>Setting</th><th>α</th><th>Seed</th><th>Decoded validity</th><th>Raw-score validity</th><th>Proximity</th><th>Any immutable violation</th><th>Age ↓</th><th>Other/Unk.</th></tr></thead><tbody>{''.join(seedrows)}</tbody></table></div><p class="small">Raw scores can assign fractional values to several categories within one feature. They are diagnostic outputs, not categorical feature vectors. All main results use the same categorical argmax decoding and numerical bounds, without policy projection.</p></details>
<h3>Sweep verification and artifacts</h3>
<p>All {audit['decoded_candidates_audited']:,} decoded candidates were audited. Every raw draw was replayed exactly from its checkpoint and evaluation noise. Independent checks confirmed categorical decoding, predictions, distances, violation rates and L₁–L₃ diagnostics. The original and α = 1 checkpoint and candidate hashes are unchanged. All runs retain the same data split, classifier, architecture, seed-specific initialization, training budget and sampling protocol. The recorded training losses were also checked against the specified added coefficients.</p>
<p><a href="{prefix}/results.csv">Full sweep CSV</a> · <a href="{prefix}/summary.json">Summary, per-seed metrics and paired differences</a> · <a href="{prefix}/audit.json">36-run audit</a> · <a href="{prefix}/runtime_environment.json">Runtime environment</a> · <a href="cf_feasibility_penalty_sweep.py">Sweep implementation</a> · <a href="cf_feasibility_sweep_audit.py">Audit implementation</a> · <a href="test_cf_feasibility_sweep.py">Objective-preservation tests</a></p>
<pre class="loss-equation">.venv-cf-feasibility/bin/python cf_feasibility_penalty_sweep.py train
.venv-cf-feasibility/bin/python cf_feasibility_penalty_sweep.py aggregate
.venv-cf-feasibility/bin/python cf_feasibility_sweep_audit.py
.venv-cf-feasibility/bin/python cf_feasibility_report.py</pre>'''
    ranges={s:f'{100*min(m(s,a)["validity"]["mean"] for a in data["alphas"]):.1f}–{100*max(m(s,a)["validity"]["mean"] for a in data["alphas"]):.1f}' for s in ('c1','c2')}
    return dict(section=section,ranges=ranges)
