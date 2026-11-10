"""Loss-only C1/C2 results, kept distinct from the earlier projection study."""
import json
from cf_feasibility_comparison_report import load_comparison, render_table, format_cell, DEFINITIONS
from cf_feasibility_sweep_report import render_sweep


def render_penalties(root):
    sweep = render_sweep(root)
    prefix = 'trained_models/cf_feasibility_penalties'
    out = root/prefix
    data = json.loads((out/'summary.json').read_text())
    audit = json.loads((out/'audit.json').read_text())
    assert audit['passed'] and audit['new_trained_runs']==6 and audit['reused_baselines']==6
    assert len(data['runs'])==12
    def m(setting,variant,key,cohort='all'):
        return data['summary'][f'{setting}/{variant}/{cohort}']['metrics'][key]
    comparison=load_comparison(root)
    selections=[(s,v) for s in ('c1','c2') for v in ('original','loss_only','projected')]
    consolidated=render_table(comparison,selections,table_id='consolidated-results')
    narratives=[]
    for setting in ('c1','c2'):
        old=comparison['summary'][f'{setting}/original/all']
        new=comparison['summary'][f'{setting}/loss_only/all']
        narratives.append(f'<p><strong>{setting.upper()}.</strong> Validity changes from {old["validity"]["mean"]:.4f} to {new["validity"]["mean"]:.4f}, while L1 proximity changes from {old["proximity"]["mean"]:.3f} to {new["proximity"]["mean"]:.3f}. The fraction changing at least one immutable feature changes from {100*old["viol_any"]["mean"]:.1f}% to {100*new["viol_any"]["mean"]:.1f}%; age decreases change from {100*old["viol_age_decrease"]["mean"]:.1f}% to {100*new["viol_age_decrease"]["mean"]:.1f}%.</p>')
    soft_means={s:sum(r['native_soft']['all']['valid'] for r in data['runs'] if r['setting']==s and r['variant']=='loss_only')/3 for s in ('c1','c2')}
    narratives.append(f'<p><strong>Soft scores versus categorical inputs.</strong> Before categorical decoding, the new models have {100*soft_means["c1"]:.4f}% validity for C1 and {100*soft_means["c2"]:.4f}% for C2. Many prediction flips disappear when each categorical block is converted to one actual category. As in the CounterNet report, the main table measures the delivered, decoded feature vectors. This common decoding applies no feasibility rules.</p>')
    directions=[]
    for cohort,title in [('low_to_high','Predicted low → high income'),('high_to_low','Predicted high → low income')]:
        n=int(comparison['summary'][f'c1/original/{cohort}']['n_test']['mean'])
        directions.append(f'<h3>{title} · {n:,} test observations</h3>'+render_table(comparison,
            [(s,v) for s in ('c1','c2') for v in ('original','loss_only')],cohort,table_id='penalty-'+cohort))
    seed_rows=[]
    for row in comparison['rows']:
        if row['cohort']!='all' or row['variant']=='projected': continue
        cells=''.join(format_cell(dict(mean=row[k],sd=None),k) for k in ('acc','validity','proximity','prox_norm','viol_any','viol_age_decrease','out_disallowed_any'))
        seed_rows.append(f'<tr><th>{row["setting"].upper()} · {row["seed"]} · {row["variant"]}</th>{cells}</tr>')
    soft=[]
    for setting in ('c1','c2'):
        for variant,title in [('original','Original baseline'),('loss_only','Added penalties')]:
            values=[r['native_soft']['all']['valid'] for r in data['runs'] if r['setting']==setting and r['variant']==variant]
            soft.append(f'<tr><th>{setting.upper()} · {title}</th><td>{100*sum(values)/len(values):.4f}%</td><td>{100*m(setting,variant,"valid")["mean"]:.4f}%</td></tr>')
    c2_failures=[]
    for key,label in [('c2_education_decrease','Education rank decreases'),('c2_higher_education_without_older_age','Education rank increases without strictly older age'),('c2_same_education_with_younger_age','Equal education rank with younger age')]:
        cells=''.join(format_cell(m('c2',v,key),'viol_c2') for v in ('original','loss_only'))
        c2_failures.append(f'<tr><th>{label}</th>{cells}</tr>')
    section=f'''<section id="loss-only-results">{sweep['section']}{DEFINITIONS}
<h2>Reference comparison · α = 1 and projection</h2>
<p>The table below retains the initial six loss-only fits at α = 1 and compares them with the original and projected variants. The α = 1 results are also included in the full sweep above; they are not additional runs. The following penalty explanations apply to every coefficient, with the full sweep objectives given below.</p>
<h3>Reference results — test split (n = 8,141)</h3><p class="small">Scroll the table horizontally to see all loss, proximity and violation columns.</p>{consolidated}
<p class="small">Column definitions and order follow the consolidated table in <a href="actionability_recourse.html">actionability_recourse.html</a>. Each cell is the mean across seeds 42, 43 and 44; a nonzero violation rate below the displayed precision is shown as &lt;0.1%; hover over a cell for its unrounded mean and seed standard deviation. Every run counts all ten draws for each test observation, in both prediction directions. CounterNet reports one delivered candidate per observation; Model-approx is stochastic, so these results average ten. Blue rows are the α = 1 loss-only extensions; grey rows are the earlier projected extensions, included for reference. Accuracy is identical because all variants share one frozen classifier.</p>
<div class="callout">{''.join(narratives)}<p>Read validity and proximity alongside each violation column: the added losses encourage the intended rules but do not guarantee them. A smaller distance is not an improvement in recourse if candidates fail to change the prediction or change features that should remain fixed.</p></div>
<h3>Which constraints were added, and how?</h3>
<div class="table-wrap"><table class="constraint-table"><thead><tr><th>Feature / rule</th><th>Original C1 / C2</th><th>New loss-only C1 / C2</th></tr></thead><tbody>
<tr><th>Age may not decrease</th><td>Both already use the raw age-decrease hinge.</td><td>Retain that hinge once, unchanged. No age floor.</td></tr>
<tr><th>Race, gender and marital status stay fixed</th><td>No dedicated immutability loss; the reconstruction term merely encourages similarity.</td><td>For every coordinate in these three one-hot blocks, add penalties for both increases and decreases: |r<sub>j</sub> − x<sub>j</sub>|.</td></tr>
<tr><th>No Other/Unknown workclass or occupation</th><td>No category-specific prohibition.</td><td>Add a one-sided hinge on each forbidden output score, max(0, r<sub>j</sub>). This encourages its score toward zero without masking it.</td></tr>
<tr><th>Education</th><td>C1: unrestricted. C2: original age hinge and linear education-score term.</td><td>Retain the same C1/C2 treatment. No new education penalty or repair step.</td></tr>
<tr><th>Hours per week / other categories</th><td>Sigmoid numerical bounds and common categorical decoding.</td><td>Unchanged. “Other” race remains allowed and should match the factual race.</td></tr>
</tbody></table></div>
<h3>Exact additional penalties</h3>
<p id="extension-confirmation"><strong>These are additions to the original objectives.</strong> The complete C1 or C2 loss is computed first, and the extra-feature penalties are then added. The original age term is retained once, C2 retains its education term, and the reconstruction, KL and validity terms remain intact. The saved training-code fingerprints confirm this implementation for the completed runs.</p>
<p>Here x is the factual encoded input, and r is the Generator’s raw output before categorical decoding. The categorical coordinates of x are one-hot. The coordinates of r are individual sigmoid scores between 0 and 1; they are not yet one-hot categories and need not sum to one within a feature block.</p>
<p>For factual input x and raw decoder output r, the three immutability penalties are <strong>two-sided hinge losses</strong>, equivalently an L1 distance over each one-hot feature block. The two category bans are <strong>one-sided hinge losses</strong> on the forbidden output scores:</p>
<div class="constraint-formula" id="added-constraint-formula">
<p>P<sub>race</sub>(x,r) = Σ<sub>j∈race</sub> [max(0, r<sub>j</sub> − x<sub>j</sub>) + max(0, x<sub>j</sub> − r<sub>j</sub>)] = ‖r<sub>race</sub> − x<sub>race</sub>‖<sub>1</sub></p>
<p>P<sub>gender</sub>(x,r) = ‖r<sub>gender</sub> − x<sub>gender</sub>‖<sub>1</sub></p>
<p>P<sub>marital</sub>(x,r) = ‖r<sub>marital_status</sub> − x<sub>marital_status</sub>‖<sub>1</sub></p>
<p>P<sub>workclass</sub>(r) = max(0, r<sub>workclass, Other/Unknown</sub>)</p>
<p>P<sub>occupation</sub>(r) = max(0, r<sub>occupation, Other/Unknown</sub>)</p>
<p>P<sub>extra</sub> = (1 / (N M)) Σ<sub>i=1…N</sub> Σ<sub>s=1…M</sub> [P<sub>race</sub>(x<sub>i</sub>,r<sub>is</sub>) + P<sub>gender</sub>(x<sub>i</sub>,r<sub>is</sub>) + P<sub>marital</sub>(x<sub>i</sub>,r<sub>is</sub>) + P<sub>workclass</sub>(r<sub>is</sub>) + P<sub>occupation</sub>(r<sub>is</sub>)]</p>
</div>
<div id="penalty-explanations"><h3>How each penalty changes the training incentive</h3>
<p><strong>Race.</strong> P<sub>race</sub> compares both coordinates of the generated race block with the original block. For the original category, whose input coordinate is 1, it pushes the generated score upward toward 1. For the other category, whose input coordinate is 0, it pushes the score downward toward 0. Both directions of departure from the factual block are penalized.</p>
<p><strong>Gender.</strong> P<sub>gender</sub> applies the same mechanism to the two gender coordinates: preserve the original category by increasing its score and decreasing the alternative’s score.</p>
<p><strong>Marital status.</strong> P<sub>marital</sub> applies the same mechanism to all five marital-status coordinates. It encourages the factual category’s score toward 1 and the four alternatives toward 0. Its coordinate differences are summed, not averaged over the five categories.</p>
<div class="callout"><p><strong>A numerical example.</strong> If the factual block is x = [1, 0] and the generated scores are r = [0.7, 0.3], the penalty is |0.7 − 1| + |0.3 − 0| = <strong>0.6</strong>. An exact match costs zero. A complete switch to the other one-hot category, r = [0, 1], costs two. The equality |r<sub>j</sub> − x<sub>j</sub>| = max(0, r<sub>j</sub> − x<sub>j</sub>) + max(0, x<sub>j</sub> − r<sub>j</sub>) explains why the formula is a pair of opposing hinges.</p></div>
<p><strong>Workclass.</strong> P<sub>workclass</sub> penalizes only the Other/Unknown output score. A score of 0.3 contributes 0.3 to the unweighted penalty, so minimization pushes it toward zero. The penalty does not choose which allowed workclass should replace it, and does not require the original workclass to stay fixed.</p>
<p><strong>Occupation.</strong> P<sub>occupation</sub> works identically for the Other/Unknown occupation score. It discourages that output category without preventing changes between allowed occupations. Race = Other is a different, legitimate category; it is preserved by P<sub>race</sub>, not banned by either of these terms.</p>
<p><strong>Combined effect.</strong> Each candidate contributes the sum of these five penalties. P<sub>extra</sub> averages that sum across the minibatch and its sampled candidates. The loss gradient encourages the Generator to satisfy these preferences; there is no output-copying, age floor or forbidden-category mask. Other loss terms can compete with these incentives, so violations can remain after training.</p></div>
<p>N is the minibatch size (up to 2,048), M = 50 is the number of training draws per observation, and r<sub>is</sub> is draw s for observation i. The immutable blocks have 2 race, 2 gender and 5 marital-status coordinates. We sum all nine coordinate differences and both forbidden scores, then average over observations and draws. There is no division by the number of features or coordinates. Because the decoder uses a sigmoid, the two forbidden scores are nonnegative and their hinge equals the score itself.</p>
<h3>Where these penalties enter C1 and C2 · initial α = 1 setting</h3>
<div class="constraint-formula" id="extended-objective-formula">
<p>A = mean[max(0, x<sub>age</sub> − r<sub>age</sub>)] &nbsp; (original age-decrease penalty)</p>
<p>D = mean[S(r) − S(x)] &nbsp; (original C2 education-score change)</p>
<p>L<sub>C1, original</sub> = L<sub>base</sub> + 29 H<sub>0.764</sub> + 192 A</p>
<p><strong>L<sub>C1, extended</sub> = L<sub>base</sub> + 29 H<sub>0.764</sub> + 192 (A + P<sub>extra</sub>) = L<sub>C1, original</sub> + 192 P<sub>extra</sub></strong></p>
<p>L<sub>C2, original</sub> = L<sub>base</sub> + 76 H<sub>0.344</sub> + 87 (A − 0.05 D)</p>
<p><strong>L<sub>C2, extended</sub> = L<sub>base</sub> + 76 H<sub>0.344</sub> + 87 (A − 0.05 D + P<sub>extra</sub>) = L<sub>C2, original</sub> + 87 P<sub>extra</sub></strong></p>
</div>
<p>L<sub>base</sub> is the original reconstruction, categorical sum-to-one and VAE KL loss. H<sub>m</sub> is the original validity hinge with margin m. Both use raw output r in these loss-only experiments. The age term A is retained once, unchanged. C2 retains its released <em>linear</em> education term −0.05 D; it is not replaced by a new hinge. S(v) is the weighted sum of the education coordinates: HS-grad/School have weight 0; Assoc/Bachelors/Some-college 1; Masters/Prof-school 2; Doctorate 3.</p>
<p><strong>C1 therefore retains its age-decrease penalty and adds 192 P<sub>extra</sub>.</strong> C2 retains its age and education terms and adds <strong>87 P<sub>extra</sub></strong>. This extends the released soft objectives. In particular, the released C2 linear education term encourages a higher education score but does not strictly encode the complete verbal rule that an education increase must be accompanied by an age increase.</p>
<details class="reproduce" id="native-loss-definitions"><summary>Full definitions of the unchanged native terms</summary>
<div class="constraint-formula">
<p>L<sub>base</sub> = mean[R(x,r) + Q(r)] + K</p>
<p>R(x,r) = 73 |r<sub>age</sub> − x<sub>age</sub>| + 98 |r<sub>hours</sub> − x<sub>hours</sub>| + Σ<sub>j=2…27</sub> |r<sub>j</sub> − x<sub>j</sub>|</p>
<p>Q(r) = Σ<sub>categorical blocks b</sub> |1 − Σ<sub>j∈b</sub> r<sub>j</sub>|</p>
<p>K = ½ mean[μ² + v − log(v) − 1]</p>
<p>H<sub>m</sub> = Σ<sub>t∈&#123;0,1&#125;</sub> mean<sub>i: target(i)=t, s</sub> max(0, m − [σ(ℓ<sub>t</sub>(r<sub>is</sub>)) − σ(ℓ<sub>1−t</sub>(r<sub>is</sub>))])</p>
</div>
<p class="small">Coordinates are indexed from zero: age is 0, hours per week is 1, and categorical coordinates occupy 2–28. The native reconstruction slice omits coordinate 28 (the final gender coordinate); this released-code behaviour is preserved, whereas P<sub>extra</sub> and reported proximity include both gender coordinates. μ and v are the encoder mean and variance; K averages over observations and latent coordinates. ℓ denotes the frozen classifier’s two logits, σ is the sigmoid, and target(i) is the opposite predicted class of x<sub>i</sub>. H averages within each target class and then sums the two class means, skipping an empty class. This native sigmoid-logit hinge is different from the softmax squared-error L₂ diagnostic in the consolidated table.</p></details>
<p>We reuse each setting’s published feasibility coefficient: <strong>192 for C1 and 87 for C2</strong>. The extra-feature penalties are our extension of the authors’ soft-constraint approach. Adding summed constraints increases their possible contribution; the initial runs used α = 1 without dimension normalization. The completed sensitivity sweep additionally scales P<sub>extra</sub> by α while preserving every native coefficient. These terms update only the Generator.</p>
<div id="coefficient-choices"><h3>Where the coefficients come from</h3>
<p>The C1 feasibility coefficient is <strong>192, not 198</strong>. The authors report a 100-iteration random search using validation data; Table 1 gives the selected settings, not the only values considered. Their search aimed to improve feasibility subject to validity and proximity requirements. The selected Model-approx Adult settings are listed below. <a href="https://arxiv.org/html/1912.03277v3">Paper, Appendix C.1.4 and Table 1</a>.</p>
<div class="table-wrap"><table id="native-coefficients"><thead><tr><th>Setting</th><th>Validity margin</th><th>Validity weight</th><th>Feasibility weight</th><th>Education factor</th></tr></thead><tbody><tr><th>C1 · Adult-Age</th><td>0.764</td><td>29</td><td>192</td><td>Not used</td></tr><tr><th>C2 · Adult-Age-Ed</th><td>0.344</td><td>76</td><td>87</td><td>0.05 in the released code</td></tr></tbody></table></div>
<p>The education factor 0.05 is hard-coded inside the released C2 loss; the script exposes the overall feasibility weight, validity weight and margin as configurable arguments. The inspected paper’s tuning description does not document a separate search over 0.05. Its effective coefficient in the current C2 objective is −87 × 0.05 = −4.35 on D. <a href="trained_models/cf_feasibility_c2/upstream/unary-ed-const-generative-cf.py">Inspected C2 source snapshot</a>.</p>
<p>Reusing 192 and 87 for P<sub>extra</sub> was our initial extension choice. Those weights were selected by the authors for their original constraints and experimental protocol; we therefore evaluated five strengths for the added nine immutable coordinates and two forbidden scores. The full results are reported above.</p></div>
<details class="reproduce" id="coefficient-sweep-proposal" open><summary>Completed five-setting sensitivity study · full objectives and coefficients</summary>
<p>We varied only the strength of the added penalties, while preserving each original C1/C2 objective. The multiplier α enters as follows:</p>
<div class="constraint-formula" id="sweep-full-objectives">
<p>L<sub>base</sub> = mean[R(x,r) + Q(r)] + K</p>
<p>L<sub>C1,original</sub> = L<sub>base</sub> + 29 H<sub>0.764</sub> + 192 A</p>
<p><strong>L<sub>C1,extended</sub>(α) = L<sub>base</sub> + 29 H<sub>0.764</sub> + 192 A + (α × 192) P<sub>extra</sub></strong></p>
<p>L<sub>C2,original</sub> = L<sub>base</sub> + 76 H<sub>0.344</sub> + 87 A − (87 × 0.05) D</p>
<p><strong>L<sub>C2,extended</sub>(α) = L<sub>base</sub> + 76 H<sub>0.344</sub> + 87 A − (87 × 0.05) D + (α × 87) P<sub>extra</sub></strong></p>
</div>
<p>R is the native reconstruction distance, Q penalizes categorical block sums differing from one, and K is the VAE KL term. H<sub>m</sub> is the native validity hinge with margin m; A = mean[max(0, x<sub>age</sub> − r<sub>age</sub>)] penalizes age decreases; D = mean[S(r) − S(x)] is the education-score change; P<sub>extra</sub> is the sum of the added immutable-feature and forbidden-category penalties. <a href="#native-loss-definitions">Exact definitions of R, Q, K and H</a> are given above. These native terms are distinct from the <a href="#loss-diagnostic-definitions">L₁–L₃ test diagnostics</a> in the results table.</p>
<div class="table-wrap"><table id="sweep-coefficient-map"><thead><tr><th>Term</th><th>C1 coefficient / setting</th><th>C2 coefficient / setting</th><th>Changed by α?</th></tr></thead><tbody>
<tr><th>Native reconstruction + categorical + KL terms</th><td>1 × L<sub>base</sub></td><td>1 × L<sub>base</sub></td><td>No</td></tr>
<tr><th>Validity hinge H</th><td>Weight 29; margin 0.764</td><td>Weight 76; margin 0.344</td><td>No</td></tr>
<tr><th>Age-decrease penalty A</th><td>192</td><td>87</td><td>No</td></tr>
<tr><th>Education-score change D</th><td>Not present</td><td>−87 × 0.05 = −4.35</td><td>No</td></tr>
<tr><th>Added penalties P<sub>extra</sub></th><td>α × 192</td><td>α × 87</td><td>Yes</td></tr>
</tbody></table></div>
<p><strong>α multiplies only P<sub>extra</sub>.</strong> It does not scale the whole original loss, the original age term, or C2’s education term. The validity margins 0.764 and 0.344 determine the hinge thresholds; they are not multipliers of the loss. The 0.05 factor scales education relative to age inside C2’s native feasibility contribution.</p>
<div class="table-wrap"><table id="completed-sweep"><thead><tr><th>Multiplier α</th><th>Added C1 penalty weight</th><th>Added C2 penalty weight</th><th>Status</th></tr></thead><tbody><tr><th>0.1</th><td>19.2</td><td>8.7</td><td>Completed · three seeds per setting</td></tr><tr><th>0.3</th><td>57.6</td><td>26.1</td><td>Completed · three seeds per setting</td></tr><tr><th>1</th><td>192</td><td>87</td><td>Reused · three seeds per setting</td></tr><tr><th>3</th><td>576</td><td>261</td><td>Completed · three seeds per setting</td></tr><tr><th>10</th><td>1,920</td><td>870</td><td>Completed · three seeds per setting</td></tr></tbody></table></div>
<p>α = 0 is the existing original baseline, retained as an additional control. The native age weights remain 192 and 87, and C2’s education factor remains 0.05; only P<sub>extra</sub> is scaled. With three matched seeds per setting, this gives five extension settings for C1 and five for C2. We completed <strong>24 new Generator fits</strong> under the same training protocol and reused the six α = 1 runs read-only.</p>
<p>The <a href="#coefficient-sweep-results">complete results above</a> show how decoded validity, distance and each constraint violation respond to the added penalty strength. No coefficient is assumed to be best in advance, and a larger penalty need not improve every metric or do so monotonically. Coefficient changes do not remove the distinction between soft training outputs and decoded categorical inputs, or provide a projection-style guarantee.</p>
<p>We report all prespecified coefficients and matched seeds. This is an exploratory sensitivity comparison on the existing, already-inspected test split; a future selection of a coefficient would require a separate validation procedure. Native feasibility weights and C2’s 0.05 education factor were held fixed throughout.</p></details>
<p>The Generator’s raw output goes directly to the original classifier and reconstruction losses. At evaluation, ordinary argmax converts each categorical block into one category, and numerical values keep the same [0,1] bounds. There is no copying of fixed features, no age floor, no forbidden-category mask and no renormalization over allowed categories. Thus the architecture follows the <a href="#architecture">original raw-output branch</a>, with the extra terms added to the Generator’s training loss; the classifier is never updated.</p>
<p class="small">The consolidated table directly reports the relevant feature violations. Immutable changes and age decreases use tolerance 10<sup>−6</sup> in encoded units. The formulas above are evaluated on raw scores during training; violation rates are evaluated on decoded candidates during testing.</p>
<details class="reproduce"><summary>C2-specific age–education violations</summary><div class="table-wrap"><table><thead><tr><th>Violation</th><th>Original C2</th><th>Loss-only extended C2</th></tr></thead><tbody>{''.join(c2_failures)}</tbody></table></div><p class="small">These are separate diagnostic counts: education rank decreases; education increases without strictly older age; or education rank stays equal while age decreases. Strict age comparisons follow the source’s unrounded evaluation rule. No combined success or compliance score is used in the consolidated table.</p></details>
<details class="reproduce"><summary>Results by prediction direction</summary>{''.join(directions)}</details>
<details class="reproduce"><summary>All matched seeds and native-output diagnostic</summary><div class="table-wrap"><table class="results-table"><thead><tr><th>Setting · seed · variant</th><th>Acc.</th><th>Validity</th><th>Prox. (L1)</th><th>Prox. / 29</th><th>Viol. any immutable</th><th>Age ↓</th><th>Other/Unk.</th></tr></thead><tbody>{''.join(seed_rows)}</tbody></table></div><h3>Soft scores versus decoded feature vectors</h3><div class="table-wrap"><table><thead><tr><th>Variant</th><th>Validity on raw soft scores</th><th>Validity after categorical decoding</th></tr></thead><tbody>{''.join(soft)}</tbody></table></div><p class="small">The raw scores are not categorical feature vectors. These diagnostic values do not replace the decoded-candidate results above. Checkpoints and per-epoch losses, including each added penalty, are saved with each run.</p></details>
<h3>Initial α = 1 comparison and independent checks</h3>
<p>The initial α = 1 runs reuse the exact original data, 24,420/8,141 split, frozen classifier, model architecture, training budget, seed-specific Generator initialization, batch ordering and evaluation noise. Each runs for 50 epochs; no early stopping or candidate selection is used. All {audit['decoded_candidates_audited']:,} decoded candidates across the six original and six new runs were audited. Every raw sample was replayed from its saved checkpoint and noise, every decoded sample matched ordinary categorical argmax, and predictions, distances and constraint rates were independently recomputed. Runtime guards reject any call to policy projection in the new training and evaluation loops.</p>
<p><a href="{prefix}/counternet_style_metrics.csv">Consolidated metrics CSV</a> · <a href="{prefix}/counternet_style_metrics.json">Consolidated metrics JSON and definitions</a> · <a href="cf_feasibility_comparison_metrics.py">Metric computation</a> · <a href="{prefix}/summary.json">Original evaluation archive</a> · <a href="{prefix}/results.csv">Per-seed and per-direction CSV</a> · <a href="{prefix}/audit.json">Independent audit</a> · <a href="{prefix}/c1/protocol.json">C1 penalty protocol</a> · <a href="{prefix}/c2/protocol.json">C2 penalty protocol</a> · <a href="cf_feasibility_penalty_experiment.py">Loss-only implementation</a> · <a href="test_cf_feasibility_penalties.py">Constraint and no-projection tests</a> · <a href="cf_feasibility_penalty_audit.py">Audit implementation</a></p>
<pre class="loss-equation">.venv-cf-feasibility/bin/python -m unittest test_cf_feasibility_penalties -v
.venv-cf-feasibility/bin/python cf_feasibility_penalty_experiment.py all
.venv-cf-feasibility/bin/python cf_feasibility_penalty_audit.py
.venv-cf-feasibility/bin/python cf_feasibility_comparison_metrics.py
.venv-cf-feasibility/bin/python cf_feasibility_report.py</pre>
<p class="architecture-note"><strong>Earlier projection experiment.</strong> The remaining sections retain the previously completed original-versus-projected comparisons. In those sections, “Extended” means the projected extension. Their results are separate from the new loss-only models above; the original baselines are shared.</p></section>'''
    return dict(section=section, ranges=sweep['ranges'],
        c1_valid=f'{100*m("c1","loss_only","valid")["mean"]:.2f}',
        c2_valid=f'{100*m("c2","loss_only","valid")["mean"]:.2f}',
        c1_delta=f'{100*data["paired"]["c1/all"]["valid"]["mean"]:+.2f} pp',
        c2_delta=f'{100*data["paired"]["c2/all"]["valid"]["mean"]:+.2f} pp')
