"""Render the completed paired CF experiment into an offline HTML report."""
from __future__ import annotations
import argparse
import hashlib
from html import escape
import json
from pathlib import Path
from datetime import datetime, timezone
from cf_feasibility_c2_report import render_c2
from cf_feasibility_penalty_report import render_penalties
from cf_feasibility_multidata_report import render_multidata
from cf_feasibility_comparison_report import load_comparison, render_table, CSS as COMPARISON_CSS

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "trained_models/cf_feasibility_baseline"


def create_report(out=OUT, destination=ROOT / "cf-feasibility-baseline.html"):
    data = json.loads((out / "summary.json").read_text())
    independent_audit = json.loads((out / "audit.json").read_text())
    if not independent_audit["passed"] or independent_audit["trained_runs"] != 6:
        raise ValueError("The independent audit must pass for all six trained runs.")
    cfg = data["protocol"]["config"]
    seeds = data["paired"]["all"]["seeds"]
    if seeds != [42, 43, 44] or len(data["runs"]) != 6:
        raise ValueError("The planned three complete pairs are required before publishing this report.")
    prefix = out.relative_to(ROOT).as_posix()
    summary = data["summary"]
    c2 = render_c2(ROOT, data)
    penalties = render_penalties(ROOT)
    multidata = render_multidata(ROOT)
    comparison = load_comparison(ROOT)

    def metric(variant, name, cohort="all", mode="decoded"):
        return summary[f"{variant}/{mode}/{cohort}"]["metrics"][name]

    def fmt(m, percent=True, spread=True):
        if m is None:
            return "—"
        scale = 100 if percent else 1
        suffix = "%" if percent else ""
        value = f"{scale*m['mean']:.2f}{suffix}"
        if spread and m["sd"] is not None:
            value += f'<span class="sd">± {scale*m["sd"]:.2f}{suffix}</span>'
        return value

    def signed(x, percent=True):
        return f"{x * (100 if percent else 1):+.2f}" + (" pp" if percent else "")

    deltas = data["paired"]["all"]["metrics"]
    original, extended = metric("original", "valid"), metric("extended", "valid")
    prox_o, prox_e = metric("original", "proximity"), metric("extended", "proximity")
    clf = data["classifier"]["test"]
    compact_tables = []
    compact_c1 = {variant: dict(valid=metric(variant, 'valid')['mean'],
        proximity=metric(variant, 'proximity')['mean'], accuracy=clf['accuracy'])
        for variant in ['original', 'extended']}
    for setting, native_label, values in [
        ('C1', 'Soft age penalty', compact_c1),
        ('C2', 'Age + education terms', c2['compact_metrics'])]:
        rows = []
        for key, label, decimals in [('valid', 'Validity', 3), ('proximity', 'Proximity', 2), ('accuracy', 'Accuracy', 3)]:
            cells = ''.join(f'<td title="Unrounded mean: {values[v][key]:.9f}">{values[v][key]:.{decimals}f}</td>'
                            for v in ['original', 'extended'])
            rows.append(f'<tr><th scope="row">{label}</th>{cells}</tr>')
        compact_tables.append(f'<div class="table-wrap"><table class="compact-metrics" id="compact-{setting.lower()}">'
            f'<caption>Adult · Model-approx {setting}</caption><thead><tr><th scope="col">Measure</th>'
            f'<th scope="col">Original<span class="compact-subtitle">{native_label}</span></th>'
            f'<th scope="col">Feasibility set<span class="compact-subtitle">Projected extension</span></th>'
            f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
    narrative = (
        f'The original C1 model uses a soft penalty to discourage age decreases. '
        f'The earlier projection extension reduces bidirectional validity from '
        f'{original["mean"]*100:.2f}% to {extended["mean"]*100:.2f}% '
        f'({signed(deltas["valid"]["mean"])}), and mean L1 distance from '
        f'{prox_o["mean"]:.3f} to {prox_e["mean"]:.3f}. '
        f'All reported feature-violation rates are zero for the projected extension.'
    )
    main_tables = []
    labels = {"all": "All test observations · bidirectional", "low_to_high": "Predicted low income → high income", "high_to_low": "Predicted high income → low income"}
    for cohort, label in labels.items():
        n = int(metric("original", "n_people", cohort)["mean"])
        table=render_table(comparison,[("c1","original"),("c1","projected")],cohort,table_id="c1-projection-"+cohort)
        main_tables.append(f'<article class="cohort-table" id="results-{cohort}"><h3>{label}</h3><p class="small">{n:,} people × {cfg["eval_samples"]} draws per person, per seed. Same metric definitions as the consolidated table.</p>{table}</article>')

    violation_names = {"viol_race": "Race changes", "viol_gender": "Gender changes", "viol_marital_status": "Marital-status changes", "viol_age_decrease": "Age decreases", "viol_workclass_unknown": "Workclass becomes Other/Unknown", "viol_occupation_unknown": "Occupation becomes Other/Unknown"}
    violation_rows = []
    for key, label in violation_names.items():
        o, e = metric("original", key), metric("extended", key)
        violation_rows.append(f'<tr><th scope="row">{label}</th><td><div class="bar-cell"><div class="bar-track"><i style="width:{100*o["mean"]:.4f}%"></i></div><span>{fmt(o)}</span></div></td><td>{fmt(e)}</td></tr>')
    per_seed_rows = []
    training_times = []
    for seed in seeds:
        init_hashes = []
        for variant in ("original", "extended"):
            run = next(r for r in data["runs"] if r["seed"] == seed and r["variant"] == variant)
            result = run["cohorts"]["decoded"]["all"]
            training = json.loads((out / f"{variant}_seed{seed}/training.json").read_text())
            init_hashes.append(training["initial_sha256"])
            training_times.append(training["seconds"])
            per_seed_rows.append('<tr>'+f'<th>{seed} · {variant}</th>'+''.join(f'<td>{result[k]*(1 if k=="proximity" else 100):.4f}{"" if k=="proximity" else "%"}</td>' for k in ["valid", "proximity"])+f'<td>{training["seconds"]:.1f}s</td></tr>')
        assert len(set(init_hashes)) == 1, "Paired initializations do not match"

    diagnostic_rows = []
    for variant, mode, label in [("original", "decoded", "Original · common decoding"), ("original", "inference_projection", "Original · projection added only at inference"), ("extended", "decoded", "Extended · trained with projection")]:
        diagnostic_rows.append(f'<tr><th>{label}</th>'+''.join(f'<td>{fmt(metric(variant,k,mode=mode),percent=k!="proximity")}</td>' for k in ["valid", "proximity"])+'</tr>')
    soft_rows = []
    import statistics
    for variant in ("original", "extended"):
        values = [r["native_soft"]["all"]["valid"] for r in data["runs"] if r["variant"] == variant]
        soft_rows.append(f'<tr><th>{variant.title()}</th><td>{statistics.mean(values)*100:.2f}% ± {statistics.stdev(values)*100:.2f}%</td><td>{fmt(metric(variant,"valid"))}</td></tr>')

    histories = {}
    for variant in ("original", "extended"):
        histories[variant] = [json.loads((out / f"{variant}_seed{s}/history.json").read_text()) for s in seeds]
    traces = {}
    for variant, hs in histories.items():
        traces[variant] = {key: [statistics.mean(h[e][key] for h in hs) for e in range(cfg["epochs"])]
                           for key in ["total", "validity_hinge", "age_penalty"]}
    code_files = ["cf_feasibility_prepare.py", "cf_feasibility_experiment.py", "cf_feasibility_audit.py", "cf_feasibility_report.py", "test_cf_feasibility.py", "cf_feasibility_c2_experiment.py", "cf_feasibility_c2_audit.py", "cf_feasibility_c2_report.py", "test_cf_feasibility_c2.py", "cf_feasibility_penalty_experiment.py", "cf_feasibility_penalty_audit.py", "cf_feasibility_penalty_report.py", "test_cf_feasibility_penalties.py", "cf_feasibility_comparison_metrics.py", "cf_feasibility_comparison_report.py", "test_cf_feasibility_comparison_metrics.py", "cf_feasibility_penalty_sweep.py", "cf_feasibility_sweep_audit.py", "cf_feasibility_sweep_report.py", "test_cf_feasibility_sweep.py", "cf_feasibility_multidata_prepare.py", "cf_feasibility_multidata.py", "cf_feasibility_multidata_results.py", "cf_feasibility_multidata_report.py", "cf_feasibility_multidata_availability.py", "test_cf_feasibility_multidata.py"]
    audit = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in code_files}
    (out / "implementation_sha256.json").write_text(json.dumps(audit, indent=2))
    (ROOT / c2["prefix"] / "implementation_sha256.json").write_text(json.dumps(audit, indent=2))
    (ROOT / "trained_models/cf_feasibility_penalties/implementation_sha256.json").write_text(json.dumps(audit, indent=2))
    original_plus = metric("original", "valid", mode="inference_projection")["mean"]
    training_vs_posthoc = extended["mean"] - original_plus
    diagnostic_text = (
        f'Adding projection only after original-model training gives {original_plus*100:.2f}% validity. '
        f'Training with projection gives {extended["mean"]*100:.2f}% '
        f'({signed(training_vs_posthoc)} relative to that diagnostic). Both satisfy the encoded constraints. '
        f'Mean L1 distance is {metric("original", "proximity", mode="inference_projection")["mean"]:.2f} '
        f'for inference-only projection and {prox_e["mean"]:.2f} for training with projection. '
        f'These runs therefore show little additional validity benefit from including projection during '
        f'Generator training, and its mean distance is larger. The original and inference-projected '
        f'candidates use the same random draws; the diagnostic has no additional trained weights.'
    )
    attainable = independent_audit["decision_attainability"]["all"]
    attainability_text = (
        f'An independent check constructed a compliant, prediction-flipping feature vector for '
        f'{attainable["attainable"]:,} of {attainable["n_people"]:,} test observations '
        f'({attainable["rate"]*100:.2f}%). The repository classifier has a linear decision boundary, '
        f'so each permitted feature block can be chosen to maximize its contribution toward the target class. '
        f'The resulting vectors were checked against the classifier and all encoded restrictions.'
    )

    style = (ROOT / "dataset_highlights.html").read_text().split('<style>',1)[1].split('</style>',1)[0]
    extra_style = '''
pre.loss-equation{white-space:pre-wrap;max-width:100%}
.compact-overview{padding:38px 0 26px;border-bottom:1px solid var(--line)}.compact-overview .two-col{gap:24px}.compact-overview .table-wrap{margin:10px 0 16px}.compact-metrics caption{font-size:15px;font-weight:650;color:var(--ink);padding:16px}.compact-metrics td,.compact-metrics thead th:not(:first-child){text-align:right;font-variant-numeric:tabular-nums}.compact-metrics th:first-child{width:27%}.compact-subtitle{display:block;font-size:10px;font-weight:400;text-transform:none;letter-spacing:0;margin-top:5px;color:var(--muted)}
.results-table.c2-table{min-width:980px}.results-table.c2-table th:first-child{width:23%}.c2-table th,.c2-table td{padding:10px 12px}.curve-selectors{display:flex;gap:14px;flex-wrap:wrap}.curve-selectors>div,.curve-selectors select{max-width:100%}
.constraint-table{min-width:760px}.constraint-table th:first-child{width:23%}.constraint-table td{width:38.5%}.constraint-key{display:inline-block;font-weight:650;margin-bottom:3px}.architecture-diagram{margin:24px 0;background:var(--paper);border:1px solid var(--line);border-radius:8px;padding:24px}.architecture-diagram figcaption{font-size:13px;color:var(--muted);margin-top:22px}.arch-stage{font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 14px}.architecture-flow{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:26px;list-style:none;padding:0;margin:0 0 26px}.architecture-flow>li{position:relative;border:1px solid var(--line);border-radius:6px;background:#f8f7f3;padding:16px}.architecture-flow>li:not(:last-child)::after{content:'→';position:absolute;right:-22px;top:42%;font-size:21px;color:var(--muted)}.architecture-flow h3{font-size:16px;margin:0 0 12px}.architecture-flow p{font-size:12px;line-height:1.6;margin-bottom:8px}.architecture-flow p:last-child{margin-bottom:0}.arch-dims{display:block;font-family:ui-monospace,Menlo,monospace;font-size:11px;line-height:1.8;margin:9px 0;overflow-wrap:anywhere}.arch-branches{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:0 0 24px}.arch-branch{border:1px solid var(--line);border-top:3px solid #a65d34;border-radius:5px;padding:18px}.arch-branch.projected{border-top-color:#326f87}.arch-branch h3{font-size:17px;margin:0 0 10px}.arch-branch p{font-size:13px;margin-bottom:8px}.arch-branch p:last-child{margin-bottom:0}.arch-objective{background:#edf2f4;border:1px solid #d7e1e6;border-radius:6px;padding:19px}.arch-objective h3{font-size:17px;margin:0 0 10px}.arch-objective p{font-size:13px;margin-bottom:9px}.arch-objective p:last-child{margin-bottom:0}.loss-equation{font-family:ui-monospace,Menlo,monospace;font-size:13px;line-height:2;overflow-wrap:anywhere;background:#f5f6f4;border:1px solid var(--line);border-radius:6px;padding:17px 20px;margin:16px 0}.loss-terms{margin:0;padding-left:21px}.loss-terms li{margin:8px 0}.architecture-note{font-size:13px}
@media(max-width:850px){.architecture-flow{grid-template-columns:1fr 1fr}.architecture-flow>li:nth-child(2)::after{display:none}}
@media(max-width:550px){.architecture-diagram{padding:17px}.architecture-flow{grid-template-columns:1fr;gap:28px}.architecture-flow>li:not(:last-child)::after{content:'↓';display:block;right:50%;top:auto;bottom:-29px}.arch-branches{grid-template-columns:1fr}.loss-equation{padding:14px;font-size:12px}}
@media print{.architecture-diagram{break-inside:avoid}.architecture-flow{grid-template-columns:repeat(4,minmax(0,1fr))}.architecture-flow>li:nth-child(2)::after{display:block}.arch-branches{grid-template-columns:1fr 1fr}}
.hero{padding-bottom:32px}.hero h1 span{font-size:.48em}.kpi-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:26px 0}.kpi{background:var(--paper);border:1px solid var(--line);border-radius:7px;padding:18px}.kpi .value{font-size:29px;letter-spacing:-.035em;line-height:1.2;margin:8px 0}.kpi .label{font-size:11px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.055em}.kpi p{font-size:12px;color:var(--muted);margin:8px 0 0}.sd{display:block;font-size:11px;color:var(--muted);white-space:nowrap}.results-table{min-width:720px}.results-table td{font-variant-numeric:tabular-nums;white-space:nowrap}.results-table th:first-child{width:34%}.delta>*{background:var(--soft);font-size:12px}.cohort-table{margin:28px 0 36px}.cohort-table h3{margin-bottom:8px}.variant-dot{display:inline-block;height:9px;width:9px;border-radius:50%;margin-right:8px}.original{background:#a65d34}.extended{background:#326f87}.bar-cell{display:grid;grid-template-columns:1fr 90px;gap:15px;align-items:center;min-width:220px}.bar-track{height:12px;border-radius:2px;background:var(--soft);overflow:hidden}.bar-track i{height:100%;display:block;background:#a65d34}.status{font-size:11px;font-weight:650;color:#35674c;border:1px solid #a8beac;border-radius:4px;padding:4px 8px;background:#f3f8f3}.metric-note{max-width:90ch}.comparison-scope{font-size:14px}.artifact-list li{margin:9px 0}.curve-controls{display:flex;justify-content:space-between;gap:16px;align-items:center;margin:16px 0}.curve{width:100%;height:auto;display:block;background:var(--paper);border:1px solid var(--line);border-radius:7px}.curve-legend{display:flex;gap:20px;align-items:center;font-size:12px}.check-list{list-style:none;padding:0}.check-list li{padding:8px 0;border-bottom:1px solid var(--line)}.check-list li::before{content:'✓';color:#35674c;margin-right:10px;font-weight:bold}.hash{font-size:10px}.report-warning{background:#f5eee2;border-left-color:#a65d34}.traces-data{display:none}summary{cursor:pointer}.scroll-note{font-size:11px;color:var(--muted)}
@media(max-width:850px){.kpi-grid{grid-template-columns:1fr 1fr}.kpi .value{font-size:26px}}
@media(max-width:550px){.top .status{display:none}.kpi-grid{gap:9px}.kpi{padding:14px}.kpi .value{font-size:24px}.curve-controls{align-items:flex-start;flex-direction:column}.bar-cell{grid-template-columns:100px 75px;gap:10px}.report-subtitle{font-size:15px}.hero{padding-bottom:24px}}
@media print{.kpi-grid{grid-template-columns:repeat(4,1fr)}.results-table{min-width:0}.curve-controls select{display:none}.top .status{display:none}.results-table th:first-child{width:auto}.cohort-table{break-inside:avoid}}
'''
    html = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="Adult C1/C2 Model-approx results: original baselines versus added feasibility penalties without projection, with earlier projection experiments retained."><title>Model-approx CF · Adult feasibility baseline</title><style>@@STYLE@@</style></head><body>
<header class="top"><div class="wrap"><div class="brand">Counterfactual feasibility · Adult experiments</div><span class="status">Completed · C1 + C2 · 3 seeds each</span><button class="quiet-btn" type="button" id="print-page">Print / PDF</button></div></header>
<div class="wrap hero"><div class="kicker">Original objectives · Added soft constraints · Projection reference</div><h1>Model-approx CF<span>Can additional feasibility rules be learned through penalties?</span></h1><p class="lead">The Adult five-strength sweep is complete for C1 and C2. Each setting adds the CounterNet feasibility rules to the Generator’s objective, without projection, and is compared with its original baseline using the same frozen classifier and Adult split. The earlier projection experiments remain below for reference.</p><p class="lead"><strong>Credit and OULAD are now included:</strong> 72 additional C1/C2-inspired Generator fits reuse each dataset’s exact CounterNet split and encoding. <a href="#other-datasets">See their complete coefficient sweeps and constraint conversions →</a></p><div class="tag-line"><span class="tag">Adult · 24,420 training · 8,141 test rows</span><span class="tag">8 features → 29 encoded inputs</span><span class="tag">Seeds 42, 43, 44 · 36 sweep/control runs · 6 projection reference runs</span><span class="tag">10 draws per test row</span></div>
<div class="kpi-grid"><article class="kpi"><div class="label">Adult classifier accuracy</div><div class="value">@@ACCURACY@@%</div><p>Same frozen classifier for every variant.</p></article><article class="kpi"><div class="label">Adult C1 · validity range</div><div class="value">@@SWEEPC1@@%</div><p>Range of three-seed means across the five added weights.</p></article><article class="kpi"><div class="label">Adult C2 · validity range</div><div class="value">@@SWEEPC2@@%</div><p>Range of three-seed means across the five added weights.</p></article><article class="kpi"><div class="label">Adult sweep · added fits</div><div class="value">24</div><p>Additional fits complete; six α = 1 runs and six original baselines reused.</p></article></div>
<p class="comparison-scope"><strong>Adult scope.</strong> The additional rules preserve race, gender and marital status, disallow generated Other/Unknown workclass and occupation, and retain the existing age-decrease penalty. C2 also retains its original education term. Soft penalties encourage compliance; the results measure whether candidates actually obey the rules.</p></div>
<nav aria-label="Page sections"><div class="wrap"><a href="#other-datasets">Credit &amp; OULAD</a><a href="#loss-only-results">Adult penalty sweep</a><a href="#compact-results">Projection summary</a><a href="#results">01 Projected C1</a><a href="#c2-results">02 Projected C2</a><a href="#constraints">03 Projection constraints</a><a href="#diagnostics">04 Projection diagnostics</a><a href="#architecture">05 Architecture</a><a href="#design">06 Experiment design</a><a href="#training">07 Projection training</a><a href="#reproduce">08 Reproduce</a></div></nav>
<main class="wrap">
@@PENALTYSECTION@@
@@MULTIDATASECTION@@
<div class="compact-overview" id="compact-results"><h2>Earlier projection experiments · Compact results</h2><p class="intro">Original Model-approx versus its extension with the feasibility set enforced through projection, in the same measure-by-column layout as the <a href="Counterfactual_Training_Baselines.html">CT report</a>. Each cell contains one result: all four variants share the same frozen classifier, so accuracy is identical.</p><div class="two-col">@@COMPACTTABLES@@</div><p class="small">Validity and accuracy are proportions on a 0–1 scale; proximity is mean L1 distance in the encoded feature space. Validity and proximity average all test candidates in both prediction directions across three Generator seeds. Values are rounded: original C2 validity is 0.999619 before rounding to 1.000. “Original” uses soft constraint terms; age is not frozen. The consolidated table above reports the same core metrics and feature violations for all six variants.</p></div>
<section id="results"><div class="section-heading"><span class="section-no">01</span><h2>C1 · Earlier projection experiment</h2></div><p class="intro">Results are mean ± sample standard deviation across three Generator training seeds. Every one of the ten sampled candidates per person counts equally: there is no best-candidate selection. The classifier and test observations are shared across all runs.</p><p><strong>C1 finding.</strong> @@NARRATIVE@@</p>
@@TABLES@@
<div class="callout"><p><strong>Read the metrics separately.</strong> Validity measures whether the prediction changes, proximity measures the encoded change, and each violation column identifies which feature restriction was broken.</p><p>The validity drop measures the combined effect of enforcing nondecreasing age and adding fixed-feature and allowed-category restrictions under the tested training settings. This comparison does not isolate the effect of projection alone while keeping the constraint set unchanged.</p><p>A smaller mean distance alone does not demonstrate better recourse if more candidates fail to change the prediction. The main distance includes all candidates; conditional distances for valid candidates are also available in the result CSV.</p></div>
<p class="small">The low-to-high group contains @@NLOW@@ test observations predicted as low income; the high-to-low group contains @@NHIGH@@ predicted as high income. These are predicted-label cohorts, not the recorded-income groups. Percentage-point differences compare seed means; ± values describe Generator-seed variation, not confidence intervals or classifier uncertainty.</p>
</section>
@@C2SECTION@@
<section id="constraints"><div class="section-heading"><span class="section-no">03</span><h2>Constraints in the projection experiments</h2></div>
<div id="paper-adult-constraints"><h3>The paper’s Adult constraints: C1 and C2</h3><p>The authors define two Adult constraint settings and report their feasibility separately. <a href="https://arxiv.org/html/1912.03277v3#S4.SS1">Paper §4.1</a>.</p>
<div class="table-wrap"><table class="constraint-table"><thead><tr><th>Paper constraint</th><th>Definition</th><th>Status in our completed experiments</th></tr></thead><tbody>
<tr><th scope="row">C1 · Age</th><td>Counterfactual age must be at least the original age.</td><td>Original: soft age-decrease penalty. Extended: retains that penalty and guarantees nondecreasing age through projection.</td></tr>
<tr><th scope="row">C2 · Education and age</th><td>Education must not decrease. If education increases, age must strictly increase. If education stays the same, age must not decrease.</td><td>Now trained and evaluated in a separate matched pair. Both C2 arms retain the released age hinge and education-score term. The extension adds the six-rule projection, which does not guarantee full C2 compliance.</td></tr>
</tbody></table></div>
<p class="architecture-note"><strong>Definition versus implementation.</strong> Appendix C.1.6 describes hinge penalties for age and ranked education. The released C2 code instead combines an age hinge with a linear education-score term; our C2 runs preserve that code. Neither formulation guarantees the strict age increase required when education increases. The C2 results document this difference and the source’s education ranking.</p>
<div class="callout report-warning"><p><strong>What the results cover.</strong> Each constraint setting has an original and an extended trained variant. The C1 results use the age-only objective; C2 uses the released age–education objective. Feature-level violations and C2-specific age–education violations are reported separately.</p></div></div>
<h3>Rules used in our two C1-based experiments</h3><p class="intro">The original C1 model’s only feature-specific feasibility rule was a soft loss penalty discouraging age decreases. The extension retains that penalty and adds projection onto a stricter feasibility set. The table covers all eight retained Adult features.</p>
<div class="table-wrap"><table class="constraint-table"><thead><tr><th>Feature / rule</th><th>Original Model-approx CF · C1</th><th>Extended Model-approx CF · projection + feasibility set</th></tr></thead><tbody>
<tr><th scope="row">Age must not decrease</th><td><span class="constraint-key">Soft penalty only.</span><br>Penalizes <code>max(0, age − raw CF age)</code>, with weight 192. No age projection: decreases remain possible.</td><td><span class="constraint-key">Guaranteed by projection.</span><br>Uses <code>CF age = max(age, proposed age)</code>. Retains the same penalty on the raw proposal during training.</td></tr>
<tr><th scope="row">Race stays fixed</th><td>No equality constraint; the proposed category may change.</td><td>Copies the original race one-hot block exactly.</td></tr>
<tr><th scope="row">Gender stays fixed</th><td>No equality constraint; the proposed category may change.</td><td>Copies the original gender one-hot block exactly.</td></tr>
<tr><th scope="row">Marital status stays fixed</th><td>No equality constraint; the proposed category may change.</td><td>Copies the original marital-status one-hot block exactly.</td></tr>
<tr><th scope="row">Workclass</th><td>All four encoded categories are available, including <code>Other/Unknown</code>.</td><td>Excludes <code>Other/Unknown</code>. Allows Government, Private and Self-Employed.</td></tr>
<tr><th scope="row">Occupation</th><td>All six encoded categories are available, including <code>Other/Unknown</code>.</td><td>Excludes <code>Other/Unknown</code>. Allows Blue-Collar, Professional, Sales, Service and White-Collar.</td></tr>
<tr><th scope="row">Education</th><td>All eight encoded categories are available; no ordering or age–education relationship is enforced.</td><td>Same: all eight categories are available. Education is not fixed or restricted to increases.</td></tr>
<tr><th scope="row">Hours per week</th><td>May increase or decrease within the encoded domain, corresponding to 1–99 hours.</td><td>Same range and freedom to increase or decrease.</td></tr>
</tbody></table></div>
<p class="architecture-note"><strong>Shared representation rules.</strong> Both decoders emit sigmoid values in [0, 1]. This bounds numerical age to the stored 17–90-year range and hours per week to 1–99. The age penalty is computed in these normalized units. Both objectives retain the source penalty encouraging each categorical block to sum to one. At evaluation, both use categorical argmax to obtain one-hot vectors and clip numerical values to [0, 1]. These common encoding operations do not themselves preserve age or immutable features. The extension additionally copies fixed blocks, floors age and excludes the forbidden categories. During training, its mutable categorical blocks contain normalized allowed scores; at evaluation, they contain one allowed category.</p>
<h3>Observed violations of the extended policy</h3><p>Both variants are scored against the six policy restrictions in the first six rows above. Original violations of the added restrictions are a reference measurement, not a failure to implement its own stated objective.</p>
<div class="table-wrap"><table><thead><tr><th>Violation, across all test candidates</th><th>Original C1 · mean ± SD</th><th>Extended · mean ± SD</th></tr></thead><tbody>@@VIOLATIONS@@</tbody></table></div><p class="small">Bars use a common 0–100% scale. Rates overlap: one candidate may violate several restrictions. The consolidated table’s “viol. any” is the union of immutable-feature changes only; age decreases and forbidden categories are separate columns.</p>
<div class="two-col"><div><h3>Three fixed feature blocks</h3><p><code>race</code>, <code>gender</code> and <code>marital_status</code> are copied from the factual input. This preserves nine encoded coordinates. The features remain available to the classifier; projection restricts the suggested changes.</p></div><div><h3>Age and allowed categories</h3><p>Age may stay the same or increase. Generated workclass and occupation cannot be <code>Other/Unknown</code>. Inputs with these categories remain in the dataset. The policy does not ban <code>race=Other</code>.</p></div></div>
<p>The same projection rules apply in the C2 extension, with its additional native education-score training term. No hard education restriction or age–education coupling is added to that projection. The guarantees concern the six-rule policy; they do not establish causal effectiveness, affordability or whether a person can realise every remaining change. Numerical features are not rounded to integers.</p>
</section>
<section id="diagnostics"><div class="section-heading"><span class="section-no">04</span><h2>C1 diagnostics · Separate projection from learning</h2></div><p class="intro">The C1 experiment trains two model variants. The extra row below is a diagnostic: apply the extension’s hard projection to the original C1 Generator’s existing test samples, without retraining. The corresponding C2 diagnostic appears with the C2 results.</p>
<div class="table-wrap"><table class="results-table"><thead><tr><th>Generator / output treatment</th><th>Validity ↑</th><th>L1 proximity ↓</th></tr></thead><tbody>@@DIAGNOSTIC@@</tbody></table></div><p>@@DIAGNOSTICTEXT@@</p>
<h3>Soft network outputs versus decoded feature vectors</h3><p>The source VAE emits independent sigmoid values in categorical blocks. These can have fractional entries and need not sum to one. Main results above convert each categorical block into one observed category before asking the classifier whether the decision changes. The table below shows why the representation used for evaluation must be explicit.</p>
<div class="table-wrap"><table><thead><tr><th>Variant</th><th>Validity on native soft outputs</th><th>Validity after common hard decoding</th></tr></thead><tbody>@@SOFT@@</tbody></table></div><p class="small">For the extension, “native soft” already includes its differentiable projection. For the original, it is the raw sigmoid output. The original model receives no age floor, fixed-feature copying or unknown-category exclusions during common decoding.</p>
<h3>Are feasible alternatives actually available?</h3><p>@@ATTAINABILITY@@</p><p>Thus, the extension’s lower validity cannot be explained by the complete absence of feasible alternatives under this classifier and policy. It reflects the Generator’s performance under the fixed training settings and loss. This check places <strong>no limit on distance</strong> and may require extreme changes: it does not show that a nearby, affordable or causally effective alternative exists. The constructed vectors are diagnostic only and are not substituted for Generator outputs in any result.</p><p class="small"><a href="@@PREFIX@@/audit.json">Independent audit and availability counts</a> · <a href="@@PREFIX@@/linear_feasibility_witnesses.npz">Constructed feasible vectors</a>.</p>
</section>
<section id="architecture"><div class="section-heading"><span class="section-no">05</span><h2>How the cf-feasibility architecture becomes Model-approx CF</h2></div>
<p class="intro">The paper uses its base conditional variational autoencoder (VAE) as the counterfactual Generator for Model-approx CF. “Model-approx” identifies the use of domain-knowledge constraints in the loss, rather than a different neural-network topology. For our Adult C1 experiment, this is the unary rule that age must not decrease. <a href="https://arxiv.org/html/1912.03277v3#S2.SS4">Paper §2.4</a> · <a href="https://arxiv.org/html/1912.03277v3#S4">Shared generator in the evaluation, §4</a>.</p>
<figure class="architecture-diagram" aria-labelledby="architecture-caption">
<p class="arch-stage">1 · Shared conditional VAE — trainable in both variants</p>
<ol class="architecture-flow">
<li><h3>Input + desired class</h3><p>The factual feature vector <strong>x</strong> has 29 encoded coordinates.</p><p>The target <strong>y′ = 1 − h(x)</strong> is the opposite predicted class.</p><span class="arch-dims">[x, y′] → 30 inputs</span></li>
<li><h3>Two encoder networks</h3><p>Separate networks produce a 10-dimensional mean <strong>μ</strong> and variance <strong>v</strong>.</p><span class="arch-dims">Each: 30 → 20 → 16<br>→ 14 → 12 → 10</span><p>Variance head: <strong>v = 0.5 + sigmoid(·)</strong>.</p></li>
<li><h3>Sample a latent vector</h3><p>Draw standard normal noise <strong>ε</strong> and compute:</p><span class="arch-dims">z = μ + √v ⊙ ε<br>10 latent coordinates</span><p>Append the target class again: <strong>[z, y′]</strong> has 11 inputs.</p></li>
<li><h3>Decoder</h3><p>Maps the sampled vector and target class to a proposed feature vector <strong>r</strong>.</p><span class="arch-dims">11 → 12 → 14 → 16<br>→ 20 → 29 → sigmoid</span><p>Its categorical coordinates are soft scores at this stage.</p></li>
</ol>
<p class="arch-stage">2 · Which candidate is passed to the classifier and reconstruction loss?</p>
<div class="arch-branches">
<div class="arch-branch"><h3><span class="variant-dot original"></span>Original and loss-only Model-approx · C1 or C2</h3><p><strong>c = r</strong> — use the raw decoder proposal during training.</p><p>C1 adds the age penalty; C2 adds age and education terms. The new loss-only models follow this same path, adding penalties for fixed-feature changes and forbidden categories. There is no age floor, fixed-feature copying or forbidden-category mask in this path.</p></div>
<div class="arch-branch projected"><h3><span class="variant-dot extended"></span>Projected extension</h3><p><strong>c = P<sub>F(x)</sub>(r)</strong> — apply the projection using the original input x.</p><p>Copy race, gender and marital status; prevent age decreases; mask forbidden workclass and occupation categories and normalize the remaining scores.</p></div>
</div>
<p class="arch-stage">3 · Training signals — update the VAE, keep the classifier fixed</p>
<div class="arch-objective"><h3>Frozen classifier: 29 → 10 → 2 output scores</h3><p>The separately trained classifier <strong>f(c)</strong> supplies the validity loss. Its weights stay fixed, but gradients through its input guide the Generator toward the desired prediction. The implemented classifier has two linear layers with no intervening activation, matching the repository source.</p><p>Reconstruction/proximity compares <strong>c</strong> with <strong>x</strong>; the categorical loss acts on <strong>c</strong>; KL regularization acts on the encoder distribution. In <strong>both</strong> variants, the age penalty uses the <strong>raw proposal r</strong>, before any extension projection.</p></div>
<figcaption id="architecture-caption">Architecture redrawn from the pinned source and our implementation. The projection branch is our addition. Here h(x) = argmax f(x), ⊙ denotes element-wise multiplication, and F(x) is the allowed feature set defined above. Hidden layers in the VAE use Linear → BatchNorm → Dropout(0.1) → ReLU. The two encoder branches have separate parameters. The encoder variance output is used as a variance despite the upstream variable name <code>logvar</code>.</figcaption>
</figure>
<h3>The same architecture, different constraint objectives</h3><p>The source script <code>unary-const-generative-cf.py</code> trains <code>CF_VAE</code> with the age-decrease penalty (C1). The script <code>unary-ed-const-generative-cf.py</code> adds the education-score term (C2). The networks are the same, and both native terms are computed on raw decoder output in the C2 extension. Neither setting fits a full causal graph. The paper’s separate Example-Based CF feedback stage is not used here. <a href="https://github.com/divyat09/cf-feasibility/blob/@@COMMIT@@/generativecf/unary-const-generative-cf.py">C1 source</a> · <a href="https://github.com/divyat09/cf-feasibility/blob/@@COMMIT@@/generativecf/unary-ed-const-generative-cf.py">C2 source</a>.</p><p class="small">C1 objective below; the C2 objective and weights are specified with the <a href="#c2-results">C2 results</a>.</p>
<div class="loss-equation" role="math" aria-label="Generator objective: proximity plus categorical consistency plus KL plus 29 times validity loss plus 192 times age-decrease penalty">L<sub>G</sub> = L<sub>prox</sub>(x, c) + L<sub>cat</sub>(c) + L<sub>KL</sub>(μ, v)<br>+ 29 L<sub>valid</sub>(f(c), y′) + 192 mean[max(0, x<sub>age</sub> − r<sub>age</sub>)]</div>
<p class="architecture-note">L<sub>prox</sub> is the source reconstruction distance; L<sub>cat</sub> penalizes categorical block sums that differ from one; L<sub>KL</sub> regularizes the latent distribution toward the standard normal prior; and L<sub>valid</sub> is the source target-class hinge loss with margin 0.764. The source’s reconstruction weights, slice and class-wise reductions are retained, as documented below. The expectation over generated candidates is estimated using 50 latent samples per training example. The extended model changes c through projection while keeping these loss terms and weights.</p>
<div class="two-col"><div><h3>During training</h3><p>The encoder and decoder learn together while the already trained classifier stays frozen. Gradients pass through the soft projection in the extension where it is differentiable. There is no Predictor actionability regularizer or joint classifier–Generator update here.</p></div><div><h3>When producing test counterfactuals</h3><p>Encode x and y′, sample z, decode r, then apply the appropriate decoding and policy rules. All networks use evaluation mode. Ten latent draws give ten candidates per person without solving a fresh optimization problem for that person.</p></div></div>
<p class="small">Architecture evidence: <a href="https://github.com/divyat09/cf-feasibility/blob/@@COMMIT@@/generativecf/scripts/vae_model.py">Source CF_VAE</a> · <a href="https://github.com/divyat09/cf-feasibility/blob/@@COMMIT@@/generativecf/scripts/blackboxmodel.py">Source classifier</a> · <a href="cf_feasibility_experiment.py">Implemented ModelApproxCF, AdultPolicy and objective</a>. Layer dimensions and operations above describe the code used for these runs; the paper/source differences are recorded in the next section.</p>
</section>
<section id="design"><div class="section-heading"><span class="section-no">06</span><h2>What was held constant, and what changed</h2></div><p>The table describes the original/projected C1 pair. The C2 pair uses the identical design and projection, with its native education-score term and published C2 loss weights retained in both arms.</p>
<div class="table-wrap"><table class="comparison"><thead><tr><th>Component</th><th>Original Model-approx CF</th><th>Extended Model-approx CF</th></tr></thead><tbody>
<tr><th>Model family</th><td>Authors’ conditional VAE, Adult C1 variant</td><td>Same architecture and latent size</td></tr>
<tr><th>Classifier</th><td>One separately trained, frozen 29 → 10 → 2 classifier</td><td>Same exact classifier weights</td></tr>
<tr><th>CF input and target</th><td>Every training row; target = opposite predicted class</td><td>Same rows and targets</td></tr>
<tr><th>Base objective</th><td>Reconstruction/proximity + KL + validity hinge + C1 age penalty</td><td>Same terms and weights</td></tr>
<tr><th>Constraint enforcement</th><td>Soft age penalty only; no constraint projection during training and no extra immutable-feature or forbidden-category rules</td><td>Projection onto the extended feasibility set before CF validity and reconstruction terms</td></tr>
<tr><th>Native C1 age penalty</th><td>Evaluated on raw decoder output; discourages age decreases but does not prevent them</td><td>Also evaluated on raw output; the additional age floor guarantees nondecreasing age</td></tr>
<tr><th>Test representation</th><td>Common categorical argmax and numerical clipping only; no age floor, fixed-feature copying or forbidden-category exclusion</td><td>Same decoding, with extended policy enforced</td></tr>
<tr><th>Randomness</th><td>Three initializations and ten test draws per person</td><td>Matched initialization, batch-order seed and evaluation noise per pair</td></tr>
</tbody></table></div>
<h3>Changes from the published Adult protocol</h3><p>The authors deliberately select older low-income and younger high-income observations, and focus on low-to-high counterfactuals. Here, both variants use all 32,561 processed Adult rows with the saved CounterNet split. Their age/income filtering and low-income-only Generator restriction are removed in both arms. The 8 retained features and 29-column input structure are already compatible with their implementation. <a href="https://arxiv.org/html/1912.03277v3">Paper, §4.1–4.2</a> · <a href="https://github.com/divyat09/cf-feasibility/blob/@@COMMIT@@/generativecf/scripts/datagen.py">Original data selection</a>.</p>
<p>This is a comparison of the original method and its extension <strong>under a shared CounterNet Adult data protocol</strong>. It is not a reproduction of the paper’s published scores, and it does not compare these Generator models against CounterNet’s learned Predictor.</p>
<h3>Projection suitable for the VAE’s outputs</h3><p>Unlike CounterNet’s categorical logits, this decoder ends in sigmoid scores. During extension training, banned categories are zeroed and the remaining positive scores in each mutable block are normalized to sum to one. Fixed blocks are copied, and age is bounded below by its factual value. Test generation selects one allowed category by argmax. The allowed feature set matches the Adult policy used in the CounterNet experiments, while the soft categorical operation is adapted to this decoder.</p>
<details class="reproduce"><summary>Source fidelity, compatibility fixes and implementation caveats</summary><ul class="list">
<li><strong>Architecture parity checked.</strong> With the same initialization and random draws, the implemented VAE reproduces the upstream model’s parameters and outputs. The original objective matches the upstream loss numerically.</li>
<li><strong>Repository classifier preserved.</strong> Its two linear layers have no intervening activation, so its decision boundary is linear. This differs from the ReLU architecture described in the paper. We also retain the repository’s classifier learning rate of 0.01, rather than the paper’s stated 0.001. <a href="https://github.com/divyat09/cf-feasibility/blob/@@COMMIT@@/generativecf/scripts/blackboxmodel.py">Classifier source</a>.</li>
<li><strong>Native reconstruction slice preserved.</strong> The upstream loss uses <code>s:-1</code> after the target column has already been removed. It therefore omits the last encoded gender coordinate from its categorical reconstruction term. Both variants preserve this behaviour for implementation fidelity; reported L1 proximity includes all 29 coordinates.</li>
<li><strong>Native loss scaling preserved.</strong> Numerical reconstruction differences are weighted by their raw feature ranges. The categorical simplex penalty and the sum of per-target-class hinge means are retained. This training objective is different from the reported min–max/one-hot L1 metric.</li>
<li><strong>Only necessary runtime fixes.</strong> CUDA-specific tensor creation and unused dependencies are removed. Empty target-class slices are skipped instead of taking an undefined mean. All such changes apply equally to both variants.</li>
<li><strong>Saved preprocessing reused.</strong> The experiment uses the already fitted ReLax transformations exactly. Their original fitting provenance is not established by these assets; we do not claim that all preprocessing was fitted only on training rows.</li>
</ul></details>
</section>
<section id="training"><div class="section-heading"><span class="section-no">07</span><h2>Training settings and variation across seeds</h2></div><p class="intro">Settings were fixed before Generator evaluation within each experiment. There was no hyperparameter search, test-driven adjustment or best-test-checkpoint selection. All reported models are their final-epoch checkpoints. C2 reused the same classifier and architecture, with its own published loss weights.</p>
<div class="two-col"><div><h3>Shared classifier</h3><ul class="list"><li>Architecture: 29 → 10 → 2, repository linear layers.</li><li>100 epochs; Adam; learning rate 0.01; batches approximately 32.</li><li>Seed 42; all 24,420 training rows.</li><li>Test accuracy: @@ACCURACY@@%. Predicted low: @@NLOW@@; predicted high: @@NHIGH@@.</li></ul></div><div><h3>Both Generator variants</h3><ul class="list"><li>Conditional VAE; latent dimension 10; encoder widths 20, 16, 14, 12; decoder widths 12, 14, 16, 20.</li><li>BatchNorm, ReLU and dropout 0.1; sigmoid decoder and variance head.</li><li>50 epochs; batch size 2,048; 50 Monte Carlo samples per training example per update.</li><li>Adam: learning rate 0.01, weight decay 0.01.</li><li>Published Adult-Age settings: margin 0.764; validity weight 29; age-penalty weight 192.</li></ul></div></div>
<p>The architecture, optimizer, training budget and seeds are shared by all four variants. C2 uses margin 0.344, validity weight 76 and feasibility weight 87, with the source’s 0.05 education coefficient. The C1 and C2 pairs each use their own published settings in both arms.</p><p class="small">Configuration and runtime versions: <a href="@@PREFIX@@/protocol.json">C1 protocol</a> · <a href="@@C2PREFIX@@/protocol.json">C2 protocol</a>. C1 Generator training took @@MINUTES@@ minutes, excluding evaluation and preparation; C2 timings are reported with its results.</p>
<div class="curve-controls"><div class="curve-selectors"><div><label for="setting-select">Experiment</label><br><select id="setting-select"><option value="c1">C1 · Age</option><option value="c2">C2 · Age and education</option></select></div><div><label for="loss-select">Mean training trace across seeds</label><br><select id="loss-select"><option value="total">Total objective</option><option value="validity_hinge">Validity hinge before weighting</option><option value="age_penalty">Raw age penalty before weighting</option><option value="education_term" disabled>C2 raw education term before weight 87</option></select></div></div><div class="curve-legend"><span><i class="variant-dot original"></i>Original</span><span><i class="variant-dot extended"></i>Extended</span></div></div><svg id="training-curve" class="curve" viewBox="0 0 940 300" role="img" aria-label="Training objective by epoch"></svg><p class="small">Training losses are averages of batch means. The objective is evaluated on different candidate constructions across arms; a lower loss alone is not a performance ranking. The C2 education term can be negative by design.</p>
<details class="reproduce"><summary>Show every completed C1 run</summary><div class="table-wrap"><table class="results-table"><thead><tr><th>Seed · variant</th><th>Validity</th><th>L1 proximity</th><th>Training time</th></tr></thead><tbody>@@SEEDROWS@@</tbody></table></div></details>
<div class="callout report-warning"><p><strong>Scope of the conclusion.</strong> Results are conditional on one shared classifier, one saved train/test split and one fixed hyperparameter setting per constraint experiment, with three Generator initializations. The original method was not designed to preserve the additional immutable features. This experiment estimates the effect of adding that policy under the stated protocol; it does not establish optimal performance for either variant or performance under a known causal model.</p></div>
</section>
<section id="reproduce"><div class="section-heading"><span class="section-no">08</span><h2>Reproduce and inspect the experiment</h2></div><div class="two-col"><div><h3>Checks completed</h3><ul class="check-list"><li>Exact original train/test index membership and encoded inputs.</li><li>Upstream VAE initialization, forward-output and native-loss parity.</li><li>Projection rules, idempotence and gradients through mutable features.</li><li>Finite loss for a batch containing only one target class.</li><li>Identical classifier checksum before and after each Generator run.</li><li>Matched initial Generator checksum within each seed pair.</li><li>Every projected candidate satisfies the six-rule policy.</li><li>C2 source loss and gradients match numerically.</li><li>C2 conditional rule checked with code and paper ranks.</li></ul></div><div><h3>C1 code and result files</h3><ul class="artifact-list"><li><a href="cf_feasibility_prepare.py">Data export</a> · <a href="cf_feasibility_experiment.py">Models, training and evaluation</a> · <a href="test_cf_feasibility.py">Scientific checks</a></li><li><a href="cf_feasibility_report.py">Report generator</a> · <a href="requirements-cf-feasibility.txt">Environment requirements</a></li><li><a href="@@PREFIX@@/results.csv">Per-run / per-cohort results CSV</a> · <a href="@@PREFIX@@/summary.json">Complete results JSON</a></li><li><a href="@@PREFIX@@/data_manifest.json">Data and split fingerprints</a> · <a href="@@PREFIX@@/implementation_sha256.json">Implementation fingerprints</a></li><li><a href="@@PREFIX@@/upstream_sources.json">Upstream sources and revision</a> · <a href="@@PREFIX@@/environment.txt">Installed versions</a></li><li><a href="@@PREFIX@@/classifier_results.json">Classifier results and training history</a> · <a href="CF_FEASIBILITY_README.md">Reproduction instructions and checkpoint locations</a></li></ul></div></div>
<details class="reproduce"><summary>Commands used</summary><pre><code>.venv/bin/python cf_feasibility_prepare.py
.venv-cf-feasibility/bin/python -m unittest test_cf_feasibility test_cf_feasibility_c2 -v
.venv-cf-feasibility/bin/python cf_feasibility_experiment.py all
.venv-cf-feasibility/bin/python cf_feasibility_audit.py
.venv-cf-feasibility/bin/python cf_feasibility_c2_experiment.py all
.venv-cf-feasibility/bin/python cf_feasibility_c2_audit.py
.venv-cf-feasibility/bin/python cf_feasibility_report.py</code></pre></details>
<p class="small">Reference method: Mahajan, Tan &amp; Sharma, <a href="https://arxiv.org/abs/1912.03277">Preserving Causal Constraints in Counterfactual Explanations for Machine Learning Classifiers</a>. Source repository: <a href="https://github.com/divyat09/cf-feasibility/tree/@@COMMIT@@">divyat09/cf-feasibility at @@SHORTCOMMIT@@</a>. Adult source: <a href="https://archive.ics.uci.edu/dataset/2/adult">UCI Adult</a>. The inputs in this experiment are the project’s processed ReLax assets, described in <a href="dataset_highlights.html">Dataset highlights</a>.</p>
</section></main><footer class="wrap foot">Generated from completed checkpoints and saved evaluations · @@DATE@@ · All interactive content works offline.</footer>
<script type="application/json" id="curve-data">@@TRACES@@</script><script>
'use strict';
const allCurves=JSON.parse(document.getElementById('curve-data').textContent);
function renderCurve(){const setting=document.getElementById('setting-select').value;const curves=allCurves[setting];const selector=document.getElementById('loss-select');selector.querySelector('[value="education_term"]').disabled=setting!=='c2';if(!curves.original[selector.value])selector.value='total';const key=selector.value;const svg=document.getElementById('training-curve');const values=[...curves.original[key],...curves.extended[key]];const low=Math.min(0,Math.min(...values)*1.08);const high=Math.max(0,Math.max(...values)*1.08);const span=high-low||1;const left=72,right=912,top=24,bottom=254;const y=v=>bottom-(v-low)/span*(bottom-top);const x=i=>left+i/(curves.original[key].length-1)*(right-left);let markup='';for(let i=0;i<=4;i++){let v=low+span*i/4;markup+=`<line x1="${left}" x2="${right}" y1="${y(v)}" y2="${y(v)}" stroke="#dedbd3"/><text x="60" y="${y(v)+4}" text-anchor="end" font-size="12" fill="#625f59">${v.toFixed(Math.max(Math.abs(low),Math.abs(high))<1?3:1)}</text>`;}for(const epoch of [1,10,20,30,40,50])markup+=`<text x="${x(epoch-1)}" y="276" text-anchor="middle" font-size="12" fill="#625f59">${epoch}</text>`;markup+='<text x="492" y="296" text-anchor="middle" font-size="12" fill="#625f59">Epoch</text>';for(const [variant,color] of [['original','#a65d34'],['extended','#326f87']]){const points=curves[variant][key].map((v,i)=>`${x(i)},${y(v)}`).join(' ');markup+=`<polyline points="${points}" fill="none" stroke="${color}" stroke-width="2.5" stroke-linejoin="round"/>`;}svg.innerHTML=markup;svg.setAttribute('aria-label',setting.toUpperCase()+' '+selector.selectedOptions[0].textContent+' by epoch, averaged over three training seeds');}
document.getElementById('loss-select').addEventListener('change',renderCurve);document.getElementById('setting-select').addEventListener('change',renderCurve);renderCurve();
document.getElementById('print-page').addEventListener('click',()=>window.print());let opened=[];window.addEventListener('beforeprint',()=>{opened=[...document.querySelectorAll('details:not([open])')];opened.forEach(el=>el.open=true);});window.addEventListener('afterprint',()=>{opened.forEach(el=>el.open=false);opened=[];});
</script></body></html>'''
    replacements = dict(STYLE=style+extra_style+COMPARISON_CSS+'\n@media(min-width:1300px){#other-datasets>.comparison-scroll{width:calc(100vw - 80px);max-width:1640px;position:relative;left:50%;transform:translateX(-50%)}}\n@media print{#other-datasets>.comparison-scroll{width:100%;max-width:100%;left:auto;transform:none}}', ACCURACY=f'{clf["accuracy"]*100:.2f}',
        EXTVALID=f'{extended["mean"]*100:.2f}', VALIDDELTA=signed(deltas["valid"]["mean"]),
        EXTPROX=f'{prox_e["mean"]:.2f}', PROXDELTA=signed(deltas["proximity"]["mean"], False),
        NARRATIVE=narrative, TABLES=''.join(main_tables), VIOLATIONS=''.join(violation_rows),
        NLOW=f'{clf["predicted_low"]:,}', NHIGH=f'{clf["predicted_high"]:,}',
        DIAGNOSTIC=''.join(diagnostic_rows), DIAGNOSTICTEXT=diagnostic_text, SOFT=''.join(soft_rows),
        ATTAINABILITY=attainability_text,
        COMMIT=data["protocol"]["upstream_commit"], SHORTCOMMIT=data["protocol"]["upstream_commit"][:12],
        PREFIX=prefix, MINUTES=f'{sum(training_times)/60:.1f}', SEEDROWS=''.join(per_seed_rows),
        C2SECTION=c2['section'], C2NARRATIVE=c2['narrative'], C2VALID=c2['valid'], C2DELTA=c2['delta'], C2POLICY=c2['policy'], C2PREFIX=c2['prefix'],
        COMPACTTABLES=''.join(compact_tables),
        SWEEPC1=penalties['ranges']['c1'], SWEEPC2=penalties['ranges']['c2'],
        MULTIDATASECTION=multidata, PENALTYSECTION=penalties['section'], PENALTYC1=penalties['c1_valid'], PENALTYC2=penalties['c2_valid'],
        PENALTYDELTA1=penalties['c1_delta'], PENALTYDELTA2=penalties['c2_delta'],
        DATE=escape(datetime.now(timezone.utc).date().isoformat()), TRACES=json.dumps(dict(c1=traces,c2=c2['traces'])).replace('<','\\u003c'))
    for key, value in replacements.items():
        html=html.replace('@@'+key+'@@',value)
    assert '@@' not in html
    destination.write_text(html)
    print(f'Wrote {destination.name} ({len(html):,} characters)')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--html',type=Path,default=ROOT/'cf-feasibility-baseline.html')
    args=parser.parse_args()
    create_report(args.out,args.html)
