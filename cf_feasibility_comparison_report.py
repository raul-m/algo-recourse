"""Shared table layout matching the actionability_recourse.html diagnostics."""
import json

HEADERS = [('acc','Acc.'),('validity','Validity'),
           ('L1','L₁ <span class="loss-column-label">Prediction error</span>'),
           ('L2','L₂ <span class="loss-column-label">CF target error</span>'),
           ('L3','L₃ <span class="loss-column-label">Squared feature change</span>'),
           ('L4','L₄ <span class="loss-column-label">Predictor recourse</span>'),
           ('proximity','Prox. (ℓ₁)'),('prox_norm','Prox. / 29'),('viol_race','Viol. race'),
           ('viol_gender','Viol. gender'),('viol_marital_status','Viol. marital'),
           ('viol_any','Viol. any'),('viol_age_decrease','Age ↓'),('out_disallowed_any','Other/Unk.')]
NAMES={'original':'Original baseline','loss_only':'Extended · penalties, no projection','projected':'Earlier extension · projection'}


def load_comparison(root):
    data=json.loads((root/'trained_models/cf_feasibility_penalties/counternet_style_metrics.json').read_text())
    assert data['checks']['passed'] and data['checks']['runs']==18
    return data


def format_cell(value,key):
    if value is None:
        return '<td title="Not applicable: the classifier is frozen; no predictor recourse loss">—</td>'
    percent=key.startswith('viol_') or key=='out_disallowed_any'
    factor=100 if percent else 1
    decimals=1 if percent else 3 if key=='proximity' else 4
    suffix='%' if percent else ''
    cls=' class="metric-group"' if key in ('proximity','viol_race') else ''
    displayed=f'{factor*value["mean"]:.{decimals}f}{suffix}'
    if percent and 0 < 100*value['mean'] < .05:
        displayed='&lt;0.1%'
    deviation='' if value.get('sd') is None else f'; seed SD: {factor*value["sd"]:.9f}{suffix}'
    return f'<td{cls} title="Mean: {factor*value["mean"]:.9f}{suffix}{deviation}">{displayed}</td>'


def render_table(data, selections, cohort='all', table_id='', caption=None):
    rows=[]
    for setting,variant in selections:
        values=data['summary'][f'{setting}/{variant}/{cohort}']
        rows.append(f'<tr class="comparison-{variant}"><th scope="row">{setting.upper()}</th><td class="model-name">{NAMES[variant]}</td>'+''.join(format_cell(values[k],k) for k,_ in HEADERS)+'</tr>')
    caption=f'<caption>{caption}</caption>' if caption else ''
    headers=''.join(f'<th scope="col"'+(' class="metric-group"' if k in ('proximity','viol_race') else '')+f'>{label}</th>' for k,label in HEADERS)
    return f'<div class="table-wrap comparison-scroll" tabindex="0" aria-label="Scrollable consolidated results"><table class="comparison-metrics" id="{table_id}">{caption}<thead><tr><th scope="col">Exp.</th><th scope="col">Model</th>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


DEFINITIONS='''<p class="small">Accuracy and validity use proportions; proximity is the mean ℓ₁ distance over all 29 encoded columns, and “prox. / 29” divides that distance by 29. Violation columns use percentages. “Viol. any” is the fraction changing at least one of race, gender or marital status. “Age ↓” measures age decreases; “Other/Unk.” is the fraction selecting Other/Unknown in workclass or occupation. These last two columns are separate from “viol. any”; overlapping violations are counted once within each union. Race = Other is not forbidden.</p>
<div id="loss-diagnostic-definitions"><h3>What L₁, L₂ and L₃ mean in this table</h3>
<p>These columns are <strong>test diagnostics calculated using the CounterNet report’s formulas</strong>. They are evaluated on the frozen classifier and decoded counterfactuals. They are not the native loss terms used to train Model-approx, and the column subscripts do not denote norm orders.</p>
<div class="table-wrap"><table class="diagnostic-definitions"><thead><tr><th>Column</th><th>Formula and averaging</th><th>What it tells us</th></tr></thead><tbody>
<tr><th scope="row">L₁ · Prediction error</th><td>½ mean<sub>observations</sub>[(p(x) − y)²]</td><td>How far the predicted class-1 probabilities on factual inputs are from the observed labels. Lower is better. Accuracy counts correct labels; L₁ also reflects the predicted probability. It is identical across these variants because the classifier is frozen.</td></tr>
<tr><th scope="row">L₂ · CF target error</th><td>½ mean<sub>observations, draws</sub>[(p(c) − t)²]</td><td>How far the counterfactual’s predicted probability is from its desired class t. Lower means closer to the target. Validity counts whether the class actually flips; L₂ also measures how confidently the probability approaches that target, so it is not simply 1 − validity.</td></tr>
<tr><th scope="row">L₃ · Squared feature change</th><td>½ mean<sub>observations, draws, 29 coordinates</sub>[(c<sub>j</sub> − x<sub>j</sub>)²]</td><td>The average squared change in encoded features. Larger individual coordinate changes receive more weight than small changes. Lower indicates less squared displacement, but does not establish that the candidate is valid or respects the constraints.</td></tr>
<tr><th scope="row">L₄ · Predictor recourse</th><td>Not applicable (—)</td><td>CounterNet uses this column for its predictor-side recourse regularizer. These Model-approx experiments train a post-hoc Generator while keeping the classifier fixed, so there is no corresponding trained L₄ term.</td></tr>
</tbody></table></div>
<p class="small">x is a factual test input, y its observed label (0 or 1), c a decoded generated candidate, and p(v) the frozen classifier’s class-1 softmax probability at input v. The target t = 1 − predicted class of x requests a prediction flip in either direction. The half factor matches the CounterNet implementation. Each displayed diagnostic is averaged over the three Generator seeds after its per-seed calculation.</p>
<p><strong>L₁ is not the ℓ₁ proximity norm.</strong> The “prox. (ℓ₁)” column is mean<sub>observations, draws</sub>[Σ<sub>j=1…29</sub>|c<sub>j</sub> − x<sub>j</sub>|]. It sums absolute changes, whereas L₃ averages squared changes with a half factor. Model-approx’s actual reconstruction, KL, validity-hinge and constraint terms are defined in the <a href="#extended-objective-formula">original and extended objectives below</a>.</p></div>'''

CSS='''
.comparison-scroll{max-width:100%;overflow-x:auto}.comparison-metrics{min-width:1470px;font-size:12px}.comparison-metrics th,.comparison-metrics td{padding:12px 10px;white-space:nowrap;font-variant-numeric:tabular-nums}.comparison-metrics th:first-child{width:45px}.comparison-metrics .model-name{white-space:normal;min-width:210px;max-width:245px}.comparison-metrics thead th{font-size:10px}.comparison-metrics .metric-group{border-left:2px solid var(--line)}.comparison-metrics .comparison-loss_only>*{background:#eef4f5}.comparison-metrics .comparison-projected>*{background:#f4f2ed}.comparison-metrics caption{text-align:left;font-weight:650;padding:14px 12px;font-size:14px}.constraint-formula{font-family:ui-monospace,Menlo,monospace;font-size:13px;line-height:2;overflow-wrap:anywhere;background:#f5f6f4;border:1px solid var(--line);border-radius:6px;padding:17px 20px;margin:16px 0}.constraint-formula p:last-child{margin-bottom:0}
.loss-column-label{display:block;white-space:normal;max-width:92px;min-width:60px;font-size:9px;font-weight:500;line-height:1.45;text-transform:none;letter-spacing:0;margin-top:5px}.diagnostic-definitions{min-width:720px}.diagnostic-definitions th:first-child{width:20%}.diagnostic-definitions td:nth-child(2){width:31%;font-family:ui-monospace,Menlo,monospace;font-size:12px}#loss-diagnostic-definitions{margin:26px 0 30px}
@media(min-width:1300px){#loss-only-results>.comparison-scroll{width:calc(100vw - 80px);max-width:1640px;position:relative;left:50%;transform:translateX(-50%)}}
@media print{#loss-only-results>.comparison-scroll{width:100%;max-width:100%;left:auto;transform:none}.comparison-metrics{min-width:0;font-size:7px}.comparison-metrics th,.comparison-metrics td{padding:4px;white-space:normal}.comparison-metrics thead th{font-size:7px}.comparison-metrics .model-name{min-width:0}}
'''
