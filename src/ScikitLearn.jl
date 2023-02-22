
module ScikitLearn

using MacroTools: @capture
using Tables
using PythonCall

const numpy = PythonCall.pynew()
const sklearn = PythonCall.pynew()
const sk_base = PythonCall.pynew()
const sk_datasets = PythonCall.pynew()
const sk_dataset_methods = [pyconvert(String, i) for i in pybuiltins.dir(pyimport("sklearn.datasets"))]

function __init__()
    PythonCall.pycopy!(numpy, pyimport("numpy"))
    PythonCall.pycopy!(sklearn, pyimport("sklearn"))
    PythonCall.pycopy!(sk_base, pyimport("sklearn.base"))
    PythonCall.pycopy!(sk_datasets, pyimport("sklearn.datasets"))
end

const translated_modules = Dict{Symbol, Vector{Symbol}}(
    :model_selection => [
                        :train_test_split,
                        :check_cv,
                        :cross_validate,
                        :cross_val_predict,
                        :cross_val_score,
                        :learning_curve,
                        :permutation_test_score,
                        :validation_curve,
    ], 
    :datasets =>       [
        :make_biclusters
    ],
)

include("sk_import.jl")
include("base.jl")
include("model_selection.jl")
include("datasets.jl")

module Utils
include("ndgrid.jl")
end


export 
# python imports
sklearn,
sk_base,

# base
is_classifier,
is_regressor,
clone,
set_params!, 
get_params,

# sklearn api
fit!,
predict,
predict_proba,
predict_log_proba,
fit_predict!,
transform,
inverse_transform,
fit_transform!,
score,
score_samples,
sample,
get_feature_names,

# utils
@sk_import

end
