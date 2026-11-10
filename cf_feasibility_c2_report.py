"""C2 results section for the combined offline C1/C2 report."""
import json
import statistics
from cf_feasibility_comparison_report import load_comparison, render_table
from pathlib import Path


def render_c2(root, c1):
    out=root/'trained_models/cf_feasibility_c2'
    data=json.loads((out/'summary.json').read_text())
    audit=json.loads((out/'audit.json').read_text())
    assert audit['passed'] and audit['c2_metrics_passed']
    assert data['paired']['all']['seeds']==[42,43,44] and len(data['runs'])==6
    assert data['classifier']['state_sha256']==c1['classifier']['state_sha256']
    prefix=out.relative_to(root).as_posix()
    def m(variant,key,cohort='all',mode='decoded'):
        return data['summary'][f'{variant}/{mode}/{cohort}']['metrics'][key]
    def fmt(value,percent=True):
        scale=100 if percent else 1;suffix='%' if percent else ''
        return f'{value["mean"]*scale:.2f}{suffix}<span class="sd">± {value["sd"]*scale:.2f}{suffix}</span>'
    def cells(variant,keys,cohort='all',mode='decoded'):
        return ''.join(f'<td>{fmt(m(variant,key,cohort,mode),key!="proximity")}</td>' for key in keys)
    v0=m('original','valid')['mean'];v1=m('extended','valid')['mean']
    narrative=(f'C2 validity changes from {100*v0:.2f}% to {100*v1:.2f}% '
        f'({100*(v1-v0):+.2f} percentage points). Mean L1 distance changes from '
        f'{m("original","proximity")["mean"]:.3f} to {m("extended","proximity")["mean"]:.3f}.')
    comparison=load_comparison(root)
    tables=[]
    labels=[('all','All test observations · bidirectional'),('low_to_high','Predicted low income → high income'),('high_to_low','Predicted high income → low income')]
    for cohort,title in labels:
        n=int(m('original','n_people',cohort)['mean'])
        table=render_table(comparison,[("c2","original"),("c2","projected")],cohort,table_id="c2-projection-"+cohort)
        tables.append(f'<article class="cohort-table"><h3>{title}</h3><p class="small">{n:,} people × 10 draws per person, per seed. Same metric definitions as the consolidated table.</p>{table}</article>')
    failures=''.join(f'<tr><th>{label}</th>{cells("original",[key])}{cells("extended",[key])}</tr>' for key,label in [
        ('c2_education_decrease','Education rank decreases'),('c2_higher_education_without_older_age','Education rank increases but age does not strictly increase'),('c2_same_education_with_younger_age','Education rank stays the same but age decreases')])
    policy_rows=''.join(f'<tr><th>{label}</th>{cells("original",[key])}{cells("extended",[key])}</tr>' for key,label in [
        ('viol_race','Race changes'),('viol_gender','Gender changes'),('viol_marital_status','Marital status changes'),
        ('viol_age_decrease','Age decreases'),('viol_workclass_unknown','Other/Unknown workclass'),('viol_occupation_unknown','Other/Unknown occupation')])
    diagnostic=''.join(f'<tr><th>{label}</th>{cells(v,["valid","proximity"],mode=mode)}</tr>' for v,mode,label in [
        ('original','inference_projection','Original C2 · projection only at inference'),('extended','decoded','Extended C2 · trained with projection')])
    seed_rows=[];seconds=0
    for run in sorted(data['runs'],key=lambda r:(r['seed'],r['variant'])):
        timing=json.loads((out/f'{run["variant"]}_seed{run["seed"]}'/'training.json').read_text());seconds+=timing['seconds']
        metrics=run['cohorts']['decoded']['all']
        seed_rows.append(f'<tr><th>{run["seed"]} · {run["variant"]}</th>'+''.join(f'<td>{metrics[k]*(1 if k=="proximity" else 100):.4f}{"" if k=="proximity" else "%"}</td>' for k in ['valid','proximity'])+f'<td>{timing["seconds"]:.1f}s</td></tr>')
    soft_rows=[]
    for variant in ['original','extended']:
        values=[r['native_soft']['all']['valid'] for r in data['runs'] if r['variant']==variant]
        soft_rows.append(f'<tr><th>{variant.title()} C2</th><td>{statistics.mean(values)*100:.2f}% ± {statistics.stdev(values)*100:.2f}%</td>{cells(variant,["valid"])}</tr>')
    histories={v:[json.loads((out/f'{v}_seed{s}'/'history.json').read_text()) for s in [42,43,44]] for v in ['original','extended']}
    traces={v:{key:[statistics.mean(h[e][key] for h in hs) for e in range(50)] for key in ['total','validity_hinge','age_penalty','education_term']} for v,hs in histories.items()}
    html=f'''<section id="c2-results"><div class="section-heading"><span class="section-no">02</span><h2>C2 · Earlier projection experiment</h2></div>
<p class="intro">Three matched seed pairs use the same 24,420 training rows, 8,141 test rows, 29 encoded features and frozen classifier as C1. Each C2 model is trained for 50 epochs. Both arms use the published Adult-Age-Ed settings: margin 0.344, validity weight 76 and feasibility weight 87. The original uses the released C2 objective; the extension adds the same six-rule projection as in C1.</p>
<p><strong>Measured finding.</strong> {narrative}</p>
<p>The projection preserves race, gender and marital status, prevents age decreases and excludes Other/Unknown workclass and occupation. It does not add a hard education restriction or require strictly older age when education increases. The individual age–education violations are shown separately below.</p>
{''.join(tables)}
<p class="small">Means ± sample SD across the three seeds. All ten candidates per person count equally. C2 uses the repository’s education ranks and strict age comparisons on unrounded numerical values. Unlike the repository’s class-1-only feasibility evaluator, our rates cover all generated candidates in each indicated prediction-direction cohort.</p>
<h3>Which part of C2 fails?</h3><div class="table-wrap"><table><thead><tr><th>C2 violation · all test candidates</th><th>Original C2</th><th>Extended C2</th></tr></thead><tbody>{failures}</tbody></table></div><p class="small">These three mutually exclusive cases identify which part of the source’s age–education rule is violated.</p>
<details class="reproduce"><summary>Inspect the six-rule projection policy</summary><div class="table-wrap"><table><thead><tr><th>Violation</th><th>Original C2</th><th>Extended C2</th></tr></thead><tbody>{policy_rows}</tbody></table></div></details>
<h3>Adding projection only at inference</h3><p>This diagnostic reuses the original C2 Generator’s samples and applies the projection after training. It introduces no additional trained model; both rows satisfy the six-rule policy.</p><div class="table-wrap"><table class="results-table"><thead><tr><th>C2 Generator / output treatment</th><th>Validity</th><th>L1 distance</th></tr></thead><tbody>{diagnostic}</tbody></table></div>
<h3>Released C2 code versus the paper</h3><p>The source uses an age hinge plus a <strong>linear education-score term</strong>, rather than the education hinge described in Appendix C.1.6. We preserve that released objective in both arms. If S is the weighted sum of the raw education coordinates and r is the raw decoder output, the extra C2 loss is:</p>
<div class="loss-equation">87 × &#123;mean[max(0, x<sub>age</sub> − r<sub>age</sub>)] − 0.05 mean[S(r) − S(x)]&#125;</div>
<p>The score uses these repository ranks: HS-grad/School = 0; Assoc/Bachelors/Some-college = 1; Masters/Prof-school = 2; Doctorate = 3. It acts on raw sigmoid scores, before projection or categorical decoding, and can be negative because increasing the score is rewarded. A negative total training loss is therefore possible. Both native C2 terms remain on raw output in the extension.</p>
<p>The appendix instead lists Prof-school at rank 3. The main results follow the code’s rank 2. The archived evaluation JSON also retains the sensitivity checks using the paper’s ranking on the same saved candidates. <a href="https://github.com/divyat09/cf-feasibility/blob/{data['protocol']['upstream_commit']}/generativecf/unary-ed-const-generative-cf.py">Pinned C2 training source</a> · <a href="https://github.com/divyat09/cf-feasibility/blob/{data['protocol']['upstream_commit']}/generativecf/scripts/evaluation_functions.py">Pinned evaluation source</a> · <a href="https://arxiv.org/html/1912.03277v3">Paper, Appendix C.1.6 and Table 1</a>.</p>
<details class="reproduce"><summary>Every C2 run and native soft-output validity</summary><div class="table-wrap"><table class="results-table"><thead><tr><th>Seed · C2 variant</th><th>Validity</th><th>L1 distance</th><th>Training time</th></tr></thead><tbody>{''.join(seed_rows)}</tbody></table></div><div class="table-wrap"><table><thead><tr><th>C2 variant</th><th>Native soft validity</th><th>Decoded validity</th></tr></thead><tbody>{''.join(soft_rows)}</tbody></table></div></details>
<p>Generator training took {seconds/60:.1f} minutes across all six C2 runs. Numerical parity checks verified the C2 loss and gradients against the released training function. An independent audit recomputed the decoded-candidate results, including C2 under both education rankings, and verified identical C1/C2 data, classifier and seed-matched initial weights.</p>
<p class="small">C2 artifacts: <a href="{prefix}/summary.json">Complete results</a> · <a href="{prefix}/results.csv">CSV</a> · <a href="{prefix}/protocol.json">Protocol and settings</a> · <a href="{prefix}/audit.json">Audit</a> · <a href="{prefix}/upstream_sources.json">Source fingerprints</a> · <a href="cf_feasibility_c2_experiment.py">C2 implementation</a> · <a href="test_cf_feasibility_c2.py">Parity and constraint checks</a> · <a href="cf_feasibility_c2_audit.py">Independent audit code</a>.</p>
<div class="callout report-warning"><p><strong>Comparing C1 with C2.</strong> Each original/extended pair uses identical settings within its constraint experiment. C1 and C2 use their respective published loss weights; their difference therefore includes both the education term and changed loss weights. It is not an isolated estimate of the education term’s effect. The results remain conditional on this classifier, split and fixed training budget.</p></div>
</section>'''
    compact_metrics = {variant: dict(valid=m(variant,'valid')['mean'],
        proximity=m(variant,'proximity')['mean'], accuracy=data['classifier']['test']['accuracy'])
        for variant in ['original','extended']}
    return dict(section=html,narrative=narrative,valid=f'{100*v1:.2f}',delta=f'{100*(v1-v0):+.2f} pp',
                policy=f'{100*m("extended","feasible")["mean"]:.2f}',traces=traces,prefix=prefix,
                compact_metrics=compact_metrics)
