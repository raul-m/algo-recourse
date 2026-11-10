# CounterNet-style evaluation of trained CT experiments.
#
# For every held-out test row (not a sample), the target is the opposite of the
# model's own predicted class, as in CounterNet. Counterfactuals are generated with
# CT's own ECCCo generator and convergence rule, under the experiment's mutability
# and inferred domain. Scored with CounterNet's definitions:
#   validity   share of rows whose predicted class flips
#   proximity  mean L1 between factual and counterfactual over ALL rows
# plus compliance with the frozen columns, one-hot well-formedness, and validity
# after snapping each one-hot block to its argmax (the form a deployable profile
# must take). Nothing in CT is modified; this only reads saved experiments.
#
# Usage: julia --project=paper/experiments -t 8 counternet_style_eval.jl <grid_config.toml> <out_dir>

using CTExperiments
using CTExperiments: get_ce_data, get_cats, get_mutability, apply_inferred_domain!
using CounterfactualExplanations
using CounterfactualExplanations: counterfactual
using CTExperiments.JSON
using CTExperiments.Flux
using Statistics
using TaijaParallel
using DelimitedFiles

grid_file, out_dir = ARGS[1], ARGS[2]
mkpath(out_dir)

const LAMBDA_COST = parse(Float64, get(ENV, "LAMBDA_COST", "0.001"))    # CT training default
const LAMBDA_ENERGY = parse(Float64, get(ENV, "LAMBDA_ENERGY", "5.0"))  # CT training default
const MAXITER = parse(Int, get(ENV, "MAXITER", "50"))                    # CT evaluation default
const THRESHOLD = parse(Float64, get(ENV, "THRESHOLD", "0.95"))          # CT evaluation threshold

exper_list = load_list(ExperimentGrid(grid_file))
summary = Dict{String,Any}()

for exper in exper_list
    name = exper.meta_params.experiment_name
    objective = exper.training_params.objective
    d = exper.data
    apply_inferred_domain!(d)
    data = get_ce_data(d; test_set=true)
    if get(ENV, "OBSERVED_DOMAIN", "0") == "1"
        # Keep every feature within its observed training range instead of CT's own
        # rule (mean +/- 2 sd, widened to the observed range), which allows values
        # below anything in the data, e.g. negative counts.
        Xtr, _ = CTExperiments.get_data(d)
        data = deepcopy(data)
        data.domain = [(minimum(r), maximum(r)) for r in eachrow(Xtr)]
    end
    if haskey(ENV, "DOMAIN_NSTD")
        # CT's own domain rule with a chosen width: the paper states 3 standard
        # deviations (App. B.3); the released code defaults to 2.
        Xtr, _ = CTExperiments.get_data(d)
        data = deepcopy(data)
        data.domain = CTExperiments.CounterfactualTraining.infer_domain_constraints(
            Xtr; nstd=parse(Float64, ENV["DOMAIN_NSTD"]))
    end
    if get(ENV, "RELAX_CATS", "0") == "1"
        # Search over one-hot columns as continuous (as CT's training search does);
        # blocks are snapped to their argmax only when scoring the final profile.
        data = deepcopy(data)
        data.features_categorical = nothing
        data.features_continuous = collect(1:size(data.X, 1))
        # One-hot entries are bounded to [0, 1]. CT's domain rule (mean +/- k sd) is
        # meant for real continuous features; applied to a one-hot column it would
        # allow impossible values such as -0.5 during the search.
        if !isnothing(data.domain)
            dom = collect(data.domain)
            for blk in get_cats(d), j in blk
                dom[j] = (zero(dom[j][1]), one(dom[j][2]))
            end
            data.domain = dom
        end
    end
    _, _, M = load_results(exper)

    X = data.X
    y = Flux.onecold(data.y, data.y_levels)
    if haskey(ENV, "N_LIMIT")          # testing only: evaluate the first N_LIMIT rows
        keep = 1:min(parse(Int, ENV["N_LIMIT"]), size(X, 2))
        X, y = X[:, keep], y[keep]
    end
    n = size(X, 2)
    probs = CounterfactualExplanations.predict_proba(M, data, X)
    yhat = Flux.onecold(probs, data.y_levels)
    acc = mean(yhat .== y)
    targets = [t == data.y_levels[1] ? data.y_levels[2] : data.y_levels[1] for t in yhat]

    gen_params = CTExperiments.GeneratorParams(;
        lambda_cost=LAMBDA_COST, lambda_energy=LAMBDA_ENERGY, lr=0.25, opt="sgd",
        maxiter=MAXITER, decision_threshold=THRESHOLD,
    )
    gen = CTExperiments.get_generator(gen_params)
    conv = CTExperiments.get_convergence("threshold", MAXITER, THRESHOLD)

    @info "$name ($objective): generating $n counterfactuals on $(Threads.nthreads()) threads ..."
    t0 = time()
    xs = [X[:, i:i] for i in 1:n]
    ces = parallelize(
        ThreadsParallelizer(), generate_counterfactual, xs, targets, data, fill(M, n),
        fill(gen, n); convergence=conv, initialization=:identity, verbose=false,
    )
    elapsed = time() - t0
    CF = reduce(hcat, [vec(counterfactual(ce)) for ce in ces])

    # CounterNet-style validity and proximity
    cf_probs = CounterfactualExplanations.predict_proba(M, data, CF)
    yhat_cf = Flux.onecold(cf_probs, data.y_levels)
    flip = yhat_cf .!= yhat
    tgt_idx = [findfirst(==(t), data.y_levels) for t in targets]
    strict = [cf_probs[tgt_idx[i], i] >= THRESHOLD for i in 1:n]
    l1 = vec(sum(abs.(CF .- X); dims=1))

    # Compliance: frozen columns unchanged; one-hot blocks well formed
    mut = get_mutability(d)
    frozen = isnothing(mut) ? Int[] : findall(==(:none), mut)
    frozen_ok = [all(abs.(CF[frozen, i] .- X[frozen, i]) .< 1e-6) for i in 1:n]
    # One-hot groups, keeping only columns that are strictly 0/1 in the data. TaijaData's
    # Adult places the continuous education_num inside the education group; snapping
    # must not overwrite it.
    isbinary(j) = all(v -> v == 0 || v == 1, X[j, :])
    cats = [filter(isbinary, blk) for blk in get_cats(d)]
    cats = [blk for blk in cats if length(blk) > 1]
    onehot_ok = [all(blk -> all(v -> v < 1e-6 || abs(v - 1) < 1e-6, CF[blk, i]) &&
                            abs(sum(CF[blk, i]) - 1) < 1e-6, cats) for i in 1:n]

    # Snap each one-hot block to its argmax and re-check validity
    CFs = copy(CF)
    for i in 1:n, blk in cats
        k = argmax(CFs[blk, i]); CFs[blk, i] .= 0; CFs[blk[k], i] = 1
    end
    yhat_snap = Flux.onecold(CounterfactualExplanations.predict_proba(M, data, CFs), data.y_levels)
    flip_snap = yhat_snap .!= yhat
    l1_snap = vec(sum(abs.(CFs .- X); dims=1))

    catcols = reduce(vcat, cats; init=Int[])
    contcols = setdiff(1:size(X, 1), catcols)
    changed = abs.(CF .- X) .> 1e-6
    changed_cols = [j for j in 1:size(X, 1) if any(changed[j, :])]
    s = Dict(
        "rows_any_categorical_change" => count(i -> any(changed[catcols, i]), 1:n),
        "rows_category_switched_after_snap" =>
            count(i -> any(abs.(CFs[reduce(vcat, cats; init=Int[]), i] .- X[reduce(vcat, cats; init=Int[]), i]) .> 1e-6), 1:n),
        "validity_after_snap_0to1" => mean(flip_snap[yhat .== data.y_levels[1]]),
        "validity_after_snap_1to0" => mean(flip_snap[yhat .== data.y_levels[2]]),
        "proximity_after_snap_valid" => mean(l1_snap[flip_snap]),
        "rows_any_continuous_change" => count(i -> any(changed[contcols, i]), 1:n),
        "columns_ever_changed" => changed_cols,
        "objective" => objective, "n_test" => n, "accuracy" => acc,
        "balanced_accuracy" => mean([mean(yhat[y .== l] .== l) for l in data.y_levels]),
        "validity" => mean(flip), "validity_strict" => mean(strict),
        "proximity_all" => mean(l1), "proximity_valid" => mean(l1[flip]),
        "frozen_compliance" => mean(frozen_ok), "onehot_wellformed" => mean(onehot_ok),
        "valid_and_compliant" => mean(flip .& frozen_ok .& onehot_ok),
        "validity_after_snap" => mean(flip_snap), "proximity_after_snap" => mean(l1_snap),
        "valid_after_snap_and_frozen_ok" => mean(flip_snap .& frozen_ok),
        "direction_0to1_validity" => mean(flip[yhat .== data.y_levels[1]]),
        "direction_1to0_validity" => mean(flip[yhat .== data.y_levels[2]]),
        "n_pred_0" => count(yhat .== data.y_levels[1]), "n_pred_1" => count(yhat .== data.y_levels[2]),
        "seconds" => elapsed,
        "settings" => Dict("lambda_cost" => LAMBDA_COST, "lambda_energy" => LAMBDA_ENERGY,
                           "maxiter" => MAXITER, "threshold" => THRESHOLD),
    )
    summary[name] = s
    writedlm(joinpath(out_dir, "$(name)_counterfactuals.csv"), permutedims(CF), ',')
    writedlm(joinpath(out_dir, "$(name)_factual_pred_target.csv"), hcat(yhat, targets, yhat_cf, yhat_snap), ',')
    @info "$name done in $(round(elapsed/60, digits=1)) min" s
end

open(joinpath(out_dir, "summary.json"), "w") do io
    JSON.print(io, summary, 2)
end
println("CN_EVAL_DONE $(grid_file)")
