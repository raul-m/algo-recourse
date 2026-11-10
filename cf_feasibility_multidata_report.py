"""Render audited Credit/OULAD adaptations with explicit constraint conversions."""
from html import escape
import json

from cf_feasibility_comparison_report import format_cell, HEADERS


def render_multidata(root):
    prefix='trained_models/cf_feasibility_multidata'
    out=root/prefix
    data=json.loads((out/'summary.json').read_text())
    audit=json.loads((out/'audit.json').read_text())
    availability=json.loads((out/'availability.json').read_text())
    assert audit['passed'] and audit['runs']==72 and len(data['runs'])==72
    assert all(v['encoded_policy_checked'] for v in availability.values())
    def m(dataset,setting,alpha,cohort='all'):
        return data['summary'][f'{dataset}/{setting}/{alpha:g}/{cohort}']
    adult=json.loads((root/'trained_models/cf_feasibility_penalty_sweep/summary.json').read_text())
    overview=[]
    for dataset,name in [('adult','Adult'),('credit','Credit'),('oulad','OULAD')]:
        for setting in ('c1','c2'):
            values=[adult['summary'][f'{setting}/{a:g}/all']['validity'] if dataset=='adult' else m(dataset,setting,a)['validity'] for a in data['alphas']]
            cells=''.join(f'<td title="Seed SD: {100*v["sd"]:.4f} percentage points">{100*v["mean"]:.2f}%</td>' for v in values)
            overview.append(f'<tr><th>{name} · {setting.upper()}{"*" if dataset!="adult" else ""}</th>{cells}</tr>')
    def table(dataset,cohort='all'):
        fixed=data['manifests'][dataset]['policy']['fixed']
        headers=HEADERS[:8]+[(f'viol_{name}',f'Viol. {escape(name)}') for name in fixed]+HEADERS[-3:]
        headers=[(key,f'Prox. / {data["manifests"][dataset]["dimensions"]}' if key=='prox_norm' else 'Forbidden output' if key=='out_disallowed_any' else label) for key,label in headers]
        rows=[]
        for setting in ('c1','c2'):
            for alpha in data['alphas']:
                metrics=m(dataset,setting,alpha,cohort)
                if not metrics['n_test']['mean']: continue
                if not alpha:
                    title='<strong>Adapted baseline</strong><br><span class="small">No added CounterNet penalties · α = 0</span>'
                else:
                    title=(f'<strong>Extended + CounterNet feasibility penalties</strong>'
                           f'<br><span class="small">α = {alpha:g} · Added weight: {alpha*(192 if setting=="c1" else 87):g}'
                           f'<br>Baseline loss retained · no projection</span>')
                rows.append(f'<tr class="comparison-{"loss_only" if alpha else "original"}"><th scope="row">{setting.upper()}*</th><td class="model-name">{title}</td>'+''.join(format_cell(metrics[key],key) for key,_ in headers)+'</tr>')
        thead=''.join(f'<th scope="col">{label}</th>' for _,label in headers)
        return f'<div class="table-wrap comparison-scroll" tabindex="0" aria-label="Scrollable {dataset} coefficient sweep"><table class="comparison-metrics" id="multidata-{dataset}-{cohort}"><thead><tr><th scope="col">Setting</th><th scope="col">Model / multiplier</th>{thead}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    pieces=[]
    for dataset,name in [('credit','Credit'),('oulad','OULAD')]:
        meta=data['manifests'][dataset]
        paragraphs=[]
        for setting in ('c1','c2'):
            before=m(dataset,setting,0)
            variants=[m(dataset,setting,a) for a in data['alphas'] if a]
            lo=min(v['validity']['mean'] for v in variants);hi=max(v['validity']['mean'] for v in variants)
            vlo=min(v['viol_any']['mean'] for v in variants); vhi=max(v['viol_any']['mean'] for v in variants)
            below=sum(v['validity']['mean']<before['validity']['mean'] for v in variants)
            conclusion=('All five tested coefficients have lower mean validity than this baseline.' if below==5 else
                'The Adult pattern of a validity decrease at every tested coefficient does not hold here.' if below<5 else '')
            paragraphs.append(f'<p><strong>{setting.upper()}*.</strong> Baseline validity is {100*before["validity"]["mean"]:.2f}%; extension means range from {100*lo:.2f}% to {100*hi:.2f}%. {conclusion} However, the extensions still change at least one immutable feature in {100*vlo:.2f}%–{100*vhi:.2f}% of generated candidates. Read the violation columns alongside validity: these are soft penalties, and none of these runs uses projection.</p>')
        directions=[]
        for cohort in ('0_to_1','1_to_0'):
            n=int(m(dataset,'c1',0,cohort)['n_test']['mean'])
            if not n: continue
            a,b=map(int,cohort.split('_to_'))
            label=f'{meta["label_semantics"][a]} → {meta["label_semantics"][b]}'
            directions.append(f'<h3>{escape(label)} · {n:,} test rows</h3>'+table(dataset,cohort))
        seedrows=[]
        for row in data['rows']:
            if row['dataset']!=dataset or row['cohort']!='all': continue
            keys=('validity','raw_validity','proximity','viol_any','viol_age_decrease','out_disallowed_any')
            seedrows.append(f'<tr><th>{row["setting"].upper()}*</th><td>{row["alpha"]:g}</td><td>{row["seed"]}</td>'+''.join(format_cell(dict(mean=row[k],sd=None),k) for k in keys)+'</tr>')
        variations=[]
        for setting in ('c1','c2'):
            for alpha in data['alphas']:
                values=m(dataset,setting,alpha)
                cells=''.join(f'<td>{values[k]["mean"]:.4f} ± {values[k]["sd"]:.4f}</td>' for k in ('validity','proximity','viol_any','viol_age_decrease'))
                variations.append(f'<tr><th>{setting.upper()}*</th><td>{alpha:g}</td>{cells}</tr>')
        pieces.append(f'''<h3 id="multidata-{dataset}">{name} · {meta['train_rows']:,} training / {meta['test_rows']:,} test rows · {meta['dimensions']} encoded inputs</h3>
<p>Shared classifier accuracy: <strong>{100*data['classifiers'][dataset]['test']['accuracy']:.2f}%</strong>. Every row below uses this same frozen classifier and exact saved CounterNet split and encoding.</p>
{table(dataset)}{''.join(paragraphs)}
<details class="reproduce"><summary>{name} · means ± seed standard deviations</summary><div class="table-wrap"><table id="multidata-{dataset}-variation"><thead><tr><th>Setting</th><th>α</th><th>Validity</th><th>Proximity</th><th>Any immutable violation</th><th>Age decrease</th></tr></thead><tbody>{''.join(variations)}</tbody></table></div><p class="small">Rates are proportions in this table. These are sample standard deviations across three seeds, not uncertainty across independent datasets.</p></details>
<details class="reproduce"><summary>{name} · results by prediction direction</summary>{''.join(directions)}</details>
<details class="reproduce"><summary>{name} · all 36 seeds/settings and raw-output validity</summary><div class="table-wrap"><table id="multidata-{dataset}-seeds"><thead><tr><th>Setting</th><th>α</th><th>Seed</th><th>Decoded validity</th><th>Raw-score validity</th><th>Proximity</th><th>Any immutable violation</th><th>Age ↓</th><th>Forbidden output</th></tr></thead><tbody>{''.join(seedrows)}</tbody></table></div></details>''')
    scale_rows=[]
    for dataset in ('credit','oulad'):
        for setting in ('c1','c2'):
            runs=[r for r in data['runs'] if (r['dataset'],r['setting'],r['alpha'])==(dataset,setting,0)]
            cfg=data['protocols'][dataset]['configs'][setting]
            avg=lambda k:sum(r['final_training_terms'][k] for r in runs)/len(runs)
            scale_rows.append(f'<tr><th>{dataset.title()} · {setting.upper()}*</th><td>{avg("reconstruction"):,.3f}</td><td>{cfg["validity_weight"]*avg("validity_hinge"):,.3f}</td><td>{cfg["age_weight"]*avg("age_penalty"):,.3f}</td><td>{cfg["age_weight"]*avg("education_term"):,.3f}</td></tr>')
    section=f'''<section id="other-datasets"><div class="section-heading"><span class="section-no">NEW</span><h2>Credit and OULAD · C1/C2-inspired constraint sweeps</h2></div>
<p class="intro">Both datasets now have an age-only baseline and an age–education baseline, each extended with the CounterNet feasibility rules expressed as soft penalties. We evaluated α = 0.1, 0.3, 1, 3 and 10 against α = 0, with three matched Generator seeds: <strong>72 completed fits</strong>. No policy projection is used.</p>
<p><strong>The asterisk matters:</strong> C1* and C2* are our adaptations of the released Model-approx losses to these datasets. They are not published Credit/OULAD baselines. Input/output widths change with the saved representations; age and education terms are converted as explained below. Adult’s C1/C2 loss coefficients and the raw-range reconstruction convention are transferred without dataset-specific tuning.</p>
<div class="table-wrap"><table id="multidata-constraints"><thead><tr><th>Dataset</th><th>C1* baseline</th><th>C2* baseline</th><th>Added CounterNet constraints in both extensions</th></tr></thead><tbody>
<tr><th>Credit</th><td>Penalize decreases in numerical AGE.</td><td>Keep the age penalty and add an education-improvement term for comparable education codes.</td><td>Keep SEX and MARRIAGE fixed; discourage EDUCATION codes 0, 5 and 6. AGE is already present and is not counted twice. EDUCATION = 4 (“others”) remains allowed.</td></tr>
<tr><th>OULAD</th><td>Penalize scores assigned to age bands below the factual band.</td><td>Keep that age-band penalty and add the five-level education-score term.</td><td>Keep gender, region and disability fixed; discourage imd_band = “0”, the saved missing-value placeholder. The existing age-band penalty remains unchanged.</td></tr>
</tbody></table></div>
<p class="small">The same encoded rows, label values, category order, numerical transformations and 75/25 row indices are loaded from the saved CounterNet assets. No rows or unknown categories are removed. Original observations with a forbidden target category remain in the data. The frozen Model-approx classifier is trained separately on these shared inputs; it is not the CounterNet Predictor.</p>
<h3>Consolidated results</h3>
<div class="table-wrap"><table id="multidata-validity-overview"><caption>Decoded validity across all three datasets · three-seed means</caption><thead><tr><th>Dataset / setting</th><th>Baseline<br>α = 0</th><th>+ CounterNet penalties<br>α = 0.1</th><th>+ CounterNet penalties<br>α = 0.3</th><th>+ CounterNet penalties<br>α = 1</th><th>+ CounterNet penalties<br>α = 3</th><th>+ CounterNet penalties<br>α = 10</th></tr></thead><tbody>{''.join(overview)}</tbody></table></div>
<p><strong>How to identify the extended configurations:</strong> every row labelled <strong>“Extended + CounterNet feasibility penalties”</strong> (α &gt; 0, blue background) retains the full C1* or C2* baseline loss and adds the dataset’s CounterNet constraints as soft penalties. Credit adds fixed SEX and MARRIAGE plus penalties on EDUCATION codes 0, 5 and 6; OULAD adds fixed gender, region and disability plus a penalty on imd_band = “0”. The age restriction is already included in each adapted baseline. These extensions use no policy projection.</p>
<p class="small">This overview reuses the completed Adult results. Compare each row with its own baseline. C1*/C2* are the dataset adaptations described here; full distances and constraint violations follow below. A higher validity alone does not establish better actionable recourse.</p>
<p class="small">Means over seeds 42, 43 and 44; ten draws for every test row, all counted in both prediction directions. Scroll horizontally for all columns; hover for precise means and seed SDs. L₁ is prediction error, L₂ is CF target error, and L₃ is squared feature change, using the <a href="#loss-diagnostic-definitions">same test-diagnostic formulas as above</a>, with d = 33 or 127 instead of 29. L₄ is not applicable because the classifier is frozen. Proximity sums absolute differences across d encoded coordinates; “prox. / d” divides by d. The native training loss is different from these reporting diagnostics.</p>
<p class="small">“Viol. any” is the union of changes to the listed immutable features only. Age decreases and forbidden categories are separate columns. “Forbidden output” means EDUCATION ∈ &#123;0,5,6&#125; for Credit and imd_band = “0” for OULAD; it does not ban Credit’s education code 4. Small nonzero violations are shown as &lt;0.1%. Credit’s favourable direction is 1 → 0 (default → no default); OULAD’s is 0 → 1 (Fail/Withdrawn → Pass/Distinction). Both directions count toward validity.</p>
{''.join(pieces)}
<h3>Exact conversion of the soft constraints</h3>
<p>x denotes the factual encoded vector, r the raw sigmoid output, and mean averages over training observations and Monte Carlo draws. Every term below is added to the loss; none copies, masks or projects the output.</p>
<div class="constraint-formula" id="multidata-formulas">
<p>A<sub>Credit</sub> = mean[max(0, x<sub>AGE</sub> − r<sub>AGE</sub>)]</p>
<p>A<sub>OULAD</sub> = mean[Σ<sub>j: rank(j) &lt; rank(x<sub>age_band</sub>)</sub> max(0, r<sub>age_band,j</sub>)]</p>
<p>P<sub>Credit</sub> = mean[‖r<sub>SEX</sub> − x<sub>SEX</sub>‖<sub>1</sub> + ‖r<sub>MARRIAGE</sub> − x<sub>MARRIAGE</sub>‖<sub>1</sub> + Σ<sub>k∈&#123;0,5,6&#125;</sub> max(0, r<sub>EDUCATION,k</sub>)]</p>
<p>P<sub>OULAD</sub> = mean[‖r<sub>gender</sub> − x<sub>gender</sub>‖<sub>1</sub> + ‖r<sub>region</sub> − x<sub>region</sub>‖<sub>1</sub> + ‖r<sub>disability</sub> − x<sub>disability</sub>‖<sub>1</sub> + max(0, r<sub>imd_band,0</sub>)]</p>
<p><strong>L<sub>C1*,extended</sub>(α) = L<sub>base</sub> + 29 H<sub>0.764</sub> + 192 A<sub>dataset</sub> + (α × 192) P<sub>dataset</sub></strong></p>
<p><strong>L<sub>C2*,extended</sub>(α) = L<sub>base</sub> + 76 H<sub>0.344</sub> + 87 A<sub>dataset</sub> − (87 × 0.05) D<sub>dataset</sub> + (α × 87) P<sub>dataset</sub></strong></p>
<p>The corresponding adapted baseline is the same formula with α = 0. Only the last term changes in each coefficient sweep.</p>
</div>
<p>For OULAD, the age order is 0–35 &lt; 35–55 &lt; 55+. A candidate’s lower-band scores are penalized individually. For example, with factual age band 55+, scores [0.2, 0.3, 0.5] contribute 0.5 to A; with factual band 0–35, no band is lower and the contribution is zero. Evaluation compares the selected bands’ ranks. It does not treat the one-hot coordinates as separate numerical ages.</p>
<h3>What C2* can express in each dataset</h3>
<p><strong>Credit.</strong> The documented order is high school (code 3) &lt; university (2) &lt; graduate school (1), assigned scores 0, 1 and 2. Code 4 means “others”; the saved data also contain 0, 5 and 6, which are not documented as education levels in the UCI description. We do not invent ranks for these four categories. <a href="https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients">UCI variable definitions</a>.</p>
<div class="constraint-formula" id="multidata-credit-education"><p>K = &#123;3, 2, 1&#125;, with S(3) = 0, S(2) = 1, S(1) = 2</p><p>d<sub>Credit</sub>(x,r) = Σ<sub>k∈K</sub> r<sub>EDUCATION,k</sub> (S(k) − S(education(x))), if education(x) ∈ K; otherwise 0</p><p>D<sub>Credit</sub> = mean[d<sub>Credit</sub>(x,r)]</p></div>
<p>This term rewards an increase or penalizes a decrease when the factual category is ranked and the generated scores concern ranked categories. For a one-hot generated ranked category, it equals the difference in education scores. An unranked factual category contributes zero, and an unranked generated category contributes no education comparison. This is explicitly a partial conversion of C2, not a claim that all education codes can be ordered. All such rows stay in the training and test sets. The additional CounterNet penalties separately discourage output codes 0, 5 and 6, while leaving code 4 allowed.</p>
<p><strong>OULAD.</strong> We use the order No Formal quals (0), Lower Than A Level (1), A Level or Equivalent (2), HE Qualification (3), Post Graduate Qualification (4). D<sub>OULAD</sub> = mean[Σ<sub>k</sub> r<sub>highest_education,k</sub>S(k) − S(education(x))], retaining the released linear education-score incentive. The dataset records the highest education level on entry and an age band. It cannot establish whether a person became strictly older while remaining in the same band, so we do not claim that this adaptation tests or enforces the full original age–education rule in years. <a href="https://www.nature.com/articles/sdata2017171">OULAD data description</a>.</p>
<p>As in the released Adult implementation, the separate age penalty and education incentive are soft terms; they do not guarantee the verbal C2 condition that an education increase must be accompanied by a strict age increase. These Model-approx conversions are feature-level constraints, not identified causal interventions.</p>
<h3>Architecture, training and loss scaling</h3>
<p>The classifier remains d → 10 → 2 with no intervening activation. The two VAE encoders use (d + 1) → 20 → 16 → 14 → 12 → 10; the decoder uses 11 → 12 → 14 → 16 → 20 → d. Hidden BatchNorm, dropout 0.1 and ReLU, the latent variance parameterization, and the sigmoid output are retained. Only input/output widths depend on the dataset. Each Generator trains for 50 epochs, batch size 2,048, 50 draws per training input, Adam learning rate 0.01 and weight decay 0.01. Final checkpoints are evaluated without early stopping or candidate selection.</p>
<p>L<sub>base</sub> retains reconstruction, categorical block-sum penalties and KL. Numerical reconstruction differences are multiplied by their original feature ranges; categorical differences use the released slice that omits the final encoded coordinate (MARRIAGE = 3 in Credit; disability = Y in OULAD). Reported proximity and added immutable penalties include every relevant coordinate. Vectorization reduces computation without changing these mathematical terms.</p>
<p><strong>The original Adult coefficients are not calibrated for these datasets.</strong> Credit has numerical ranges above one million; OULAD has click-count ranges above ten thousand. These can give reconstruction a much larger contribution than validity and constraint terms even though network inputs are min–max encoded. The final-epoch baseline training components below make this visible. They average the logged minibatch means over three seeds and are training diagnostics, not test losses.</p>
<div class="table-wrap"><table id="multidata-loss-scales"><thead><tr><th>Adapted baseline</th><th>Reconstruction</th><th>Weighted validity hinge</th><th>Weighted age penalty</th><th>Weighted education term</th></tr></thead><tbody>{''.join(scale_rows)}</tbody></table></div>
<p>The controlled comparison isolates the added penalties within each specified adaptation. It does not establish the best achievable Model-approx performance on Credit or OULAD, nor whether the Adult finding must generalize after dataset-specific validation and tuning. No coefficient or checkpoint was selected using test performance.</p>
<details class="reproduce"><summary>Separate diagnostic: does the encoded policy permit a prediction flip?</summary>
<p>Because the frozen classifier is affine, we can independently maximize the desired class margin over each permitted feature block. This finds a policy-compliant prediction flip for {availability['credit']['n_with_encoded_policy_flip']:,}/{availability['credit']['n_test']:,} Credit test inputs and {availability['oulad']['n_with_encoded_policy_flip']:,}/{availability['oulad']['n_test']:,} OULAD test inputs. Thus a low Generator validity does not, by itself, show that the encoded CounterNet policy makes a flip impossible.</p>
<p>These constructed inputs are separate availability witnesses, not Generator outputs, repaired samples or replacements in the reported metrics. The check has no distance or affordability budget, makes no causal claim and does not impose the full C2 age–education condition. <a href="{prefix}/availability.json">Diagnostic results</a> · <a href="cf_feasibility_multidata_availability.py">Calculation</a>.</p></details>
<h3>Audit and reproducibility</h3>
<p>All 72 runs and {audit['decoded_candidates_audited']:,} decoded candidates passed the audit. Every raw sample was replayed exactly; ordinary categorical decoding, predictions, L₁–L₃, proximity and each violation rate were independently recomputed. Source-file hashes and split/encoding hashes are unchanged. All coefficients and complete 50-epoch histories were checked, with identical Generator initialization per seed across both settings and every coefficient. Preliminary runs made before vectorization are archived separately and excluded from all reported results.</p>
<p><a href="{prefix}/results.csv">All metrics CSV</a> · <a href="{prefix}/summary.json">Means, SDs, paired differences and source runs</a> · <a href="{prefix}/audit.json">72-run audit</a> · <a href="{prefix}/credit/protocol.json">Credit protocol</a> · <a href="{prefix}/oulad/protocol.json">OULAD protocol</a> · <a href="{prefix}/credit/data_manifest.json">Credit data hashes</a> · <a href="{prefix}/oulad/data_manifest.json">OULAD data hashes</a> · <a href="{prefix}/vectorization_check.json">Loss/gradient equivalence checks</a> · <a href="cf_feasibility_multidata.py">Training implementation</a> · <a href="cf_feasibility_multidata_results.py">Aggregation and audit</a> · <a href="test_cf_feasibility_multidata.py">Tests</a></p>
<pre class="loss-equation">.venv/bin/python cf_feasibility_multidata_prepare.py
.venv-cf-feasibility/bin/python -m unittest test_cf_feasibility_multidata -v
.venv-cf-feasibility/bin/python cf_feasibility_multidata.py all --dataset credit
.venv-cf-feasibility/bin/python cf_feasibility_multidata.py all --dataset oulad
.venv-cf-feasibility/bin/python cf_feasibility_multidata_results.py all
.venv-cf-feasibility/bin/python cf_feasibility_multidata_availability.py
.venv-cf-feasibility/bin/python cf_feasibility_report.py</pre>
</section>'''
    return section
