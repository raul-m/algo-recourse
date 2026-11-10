using CSV
using JSON

# Adult and Credit exactly as the CounterNet (jax-relax) experiments see them:
# same rows, same train/test indices, min-max scaling and one-hot encoding.
# Produced by jax-relax/ct_export_counternet_splits.py. The test split is used
# as is; CT's own validation rows are carved out of the CounterNet training rows.

const CN_DATA_DIR = get(
    ENV, "CN_DATA_DIR", "/Users/raulm/Documents/Master/Thesis/CounterNet-Baseline/ct_data"
)

abstract type CounterNetTabular <: TabularData end

cn_dir(d::CounterNetTabular) = joinpath(CN_DATA_DIR, cn_source(d))

cn_meta(d::CounterNetTabular) = JSON.parsefile(joinpath(cn_dir(d), "meta.json"))

function cn_read(d::CounterNetTabular, split::String)
    X = CSV.read(joinpath(cn_dir(d), "X_$split.csv"), CSV.Tables.matrix; header=false)
    y = vec(CSV.read(joinpath(cn_dir(d), "y_$split.csv"), CSV.Tables.matrix; header=false))
    return permutedims(Float32.(X)), Int.(y)    # features x samples, as in TaijaData
end

# `n` is ignored: the CounterNet split is fixed and always returned in full.
function load_data(d::CounterNetTabular, n::Int)
    X, y = cn_read(d, "train")
    Xtest, ytest = cn_read(d, "test")
    return X, y, Xtest, ytest
end

get_cats(d::CounterNetTabular) = [Int.(g) for g in cn_meta(d)["categorical_groups"]]

"""
    CNAdult

UCI Adult as used in the CounterNet experiments: 29 columns, 24,420 train / 8,141 test.
"""
Base.@kwdef mutable struct CNAdult <: CounterNetTabular
    n_train::Int = 23420
    batchsize::Int = 1000
    n_validation::Int = 1000
    train_test_ratio::Float32 = 0.75
    train_test_seed::Int = get_global_seed()
    mutability::Union{String,Vector{Int},Vector{String}} = "none"
    domain::Union{String,Vector{<:Any}} = "none"
    datadir::String = get_global_dev_dir()
end

cn_source(::CNAdult) = "adult"
dname(::CNAdult) = "cn_adult"
get_name(d::CNAdult; pretty::Bool=false) = pretty ? "Adult (CN)" : dname(d)
nmax(::CNAdult) = 32561

"""
    CNCredit

Credit Default as used in the CounterNet experiments: 33 columns, 22,500 train / 7,500 test.
"""
Base.@kwdef mutable struct CNCredit <: CounterNetTabular
    n_train::Int = 21500
    batchsize::Int = 1000
    n_validation::Int = 1000
    train_test_ratio::Float32 = 0.75
    train_test_seed::Int = get_global_seed()
    mutability::Union{String,Vector{Int},Vector{String}} = "none"
    domain::Union{String,Vector{<:Any}} = "none"
    datadir::String = get_global_dev_dir()
end

cn_source(::CNCredit) = "credit"
dname(::CNCredit) = "cn_credit"
get_name(d::CNCredit; pretty::Bool=false) = pretty ? "Cred (CN)" : dname(d)
nmax(::CNCredit) = 30000

"""
    CNCreditBalanced

`CNCredit` with class-balanced training rows (majority class undersampled to 4,952,
as TaijaData balances CT's own Credit data). Test rows are CounterNet's, unchanged.
"""
Base.@kwdef mutable struct CNCreditBalanced <: CounterNetTabular
    n_train::Int = 8904
    batchsize::Int = 1000
    n_validation::Int = 1000
    train_test_ratio::Float32 = 0.75
    train_test_seed::Int = get_global_seed()
    mutability::Union{String,Vector{Int},Vector{String}} = "none"
    domain::Union{String,Vector{<:Any}} = "none"
    datadir::String = get_global_dev_dir()
end

cn_source(::CNCreditBalanced) = "credit_balanced"
dname(::CNCreditBalanced) = "cn_credit_balanced"
get_name(d::CNCreditBalanced; pretty::Bool=false) = pretty ? "Cred (CN, balanced train)" : dname(d)
nmax(::CNCreditBalanced) = 30000

"""
    CNOULAD

OULAD as used in the CounterNet experiments: 127 columns, 24,444 train / 8,149 test.
CT has no published OULAD settings; the configs reuse its Adult settings.
"""
Base.@kwdef mutable struct CNOULAD <: CounterNetTabular
    n_train::Int = 23444
    batchsize::Int = 1000
    n_validation::Int = 1000
    train_test_ratio::Float32 = 0.75
    train_test_seed::Int = get_global_seed()
    mutability::Union{String,Vector{Int},Vector{String}} = "none"
    domain::Union{String,Vector{<:Any}} = "none"
    datadir::String = get_global_dev_dir()
end

cn_source(::CNOULAD) = "oulad"
dname(::CNOULAD) = "cn_oulad"
get_name(d::CNOULAD; pretty::Bool=false) = pretty ? "OULAD (CN)" : dname(d)
nmax(::CNOULAD) = 32593
