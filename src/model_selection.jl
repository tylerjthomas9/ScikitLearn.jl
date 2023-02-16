"""
Prepare input array for scikit-learn
model selection functions
"""
_prepare_input(input::Any) = input
_prepare_input(input::AbstractArray) = numpy.array(input)

################################################################################
# Splitter Functions
################################################################################

function train_test_split(args...; kwargs...)
    py_arrays = _prepare_input.(args)
    res = sklearn.model_selection.train_test_split(py_arrays...; kwargs...)
    return Tuple(pyconvert(Array, i) for i in res)
end

check_cv(cv::Int, y=nothing; classifier::Bool=false) = sklearn.model_selection.check_cv(cv, y; classifier)

################################################################################
# Model validation
################################################################################

function cross_validate(estimator::Py, X, y; kwargs...)
    py_xy = _prepare_input.((X,y))
    res = sklearn.model_selection.cross_validate(estimator, py_xy...; kwargs...)
    return pyconvert(Dict{String, Array}, res)
end

function cross_val_predict(estimator::Py, X, y; kwargs...)
    py_xy = _prepare_input.((X,y))
    preds = sklearn.model_selection.cross_val_predict(estimator, py_xy...; kwargs...)
    return tweak_rval(preds)
end

function cross_val_score(estimator::Py, X, y; kwargs...)
    py_xy = _prepare_input.((X,y))
    scores = sklearn.model_selection.cross_val_score(estimator, py_xy...; kwargs...)
    return tweak_rval(scores)
end

function learning_curve(estimator::Py, X, y; kwargs...)
    py_xy = _prepare_input.((X,y))
    scores = sklearn.model_selection.learning_curve(estimator, py_xy...; kwargs...)
    return tuple(tweak_rval.(scores)...)
end


function permutation_test_score(estimator::Py, X, y; kwargs...)
    py_xy = _prepare_input.((X,y))
    scores = sklearn.model_selection.permutation_test_score(estimator, py_xy...; kwargs...)
    return tuple(tweak_rval.(scores)...)
end

function validation_curve(estimator::Py, X, y; param_name=nothing, param_range=nothing, kwargs...)
    py_xy = _prepare_input.((X,y))
    scores = sklearn.model_selection.validation_curve(estimator, py_xy...; 
                                                      param_name, param_range, kwargs...)
    return tuple(tweak_rval.(scores)...)
end
