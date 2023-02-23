
################################################################################
# Julia => Python
################################################################################
api_map = Dict(:decision_function => :decision_function,
               :fit_predict! => :fit_predict,
               :fit_transform! => :fit_transform,
               :get_feature_names => :get_feature_names,
               :get_params => :get_params,
               :predict => :predict,
               :predict_proba => :predict_proba,
               :predict_log_proba => :predict_log_proba,
               :partial_fit! => :partial_fit,
               :score_samples => :score_samples,
               :sample => :sample,
               :score => :score,
               :transform => :transform,
               :inverse_transform => :inverse_transform,
               :set_params! => :set_params)
               
for (jl_fun, py_fun) in api_map
    @eval $jl_fun(py_estimator::Py, args...; kwargs...) =
        tweak_rval(py_estimator.$(py_fun)(args...; kwargs...))
end

fit!(py_estimator::Py, args...; kwargs...) = py_estimator.fit(args...; kwargs...)

################################################################################
# Functions
################################################################################
clone(py_estimator::Py; safe::Bool=true) = sklearn.base.clone(py_estimator, safe=safe)
is_classifier(py_estimator::Py) = pyconvert(Bool, sklearn.base.is_classifier(py_estimator))
is_regressor(py_estimator::Py) = pyconvert(Bool, sklearn.base.is_regressor(py_estimator))
config_context(; kwargs...) = sklearn.config_context(; kwargs...)
get_config() = pyconvert(Dict{Symbol, Any}, sklearn.get_config())
set_config(; kwargs...) = sklearn.set_config(; kwargs...)
show_versions() = sklearn.show_versions()
get_params(estimator::Py; deep::Bool=true) = pyconvert(Dict{Symbol, Any}, estimator.get_params(deep=deep))
set_params!(estimator::Py, params::Dict{Symbol, Any}) = estimator.set_params(; params...)
set_params!(estimator::Py, params::Dict{String, Any}) = set_params!(estimator, Dict(Symbol(name)=>value for (name, value) in params))
set_params!(estimator::Py; kwargs...) = set_params!(estimator, Dict{Symbol, Any}(kwargs))

# not in the python package
is_pairwise(estimator) = false # global default - override for specific models
is_pairwise(py_estimator::Py) =
    hasproperty(py_estimator, :_pairwise) ? pyconvert(Bool, py_estimator._pairwise) : false
get_classes(py_estimator::Py) = pyconvert(Vector, py_estimator.classes_)
get_components(py_estimator::Py) = pyconvert(Array, py_estimator.components_)
is_transformer(estimator::Type) = !isempty(methods(transform, (estimator, Any)))
is_transformer(estimator::Any) = is_transformer(typeof(estimator))
named_steps(pip::Py) = pyconvert(Tuple, pip.steps) |> Dict
