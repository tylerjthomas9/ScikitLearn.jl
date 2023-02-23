
function tweak_datasets(x::Py)
    if pyisinstance(x, PythonCall.pybuiltins.tuple) || pyisinstance(x, PythonCall.pybuiltins.map)
        return tuple([tweak_rval(i) for i in x]...)
    elseif pyisinstance(x, PythonCall.pybuiltins.dict)
        return Dict([Symbol(key) => ScikitLearn.tweak_rval(x[key]) for key in x.keys()]...)
    else
        return tweak_rval(x)
    end
end

# ################################################################################
# # Loaders
# ################################################################################
const data_loaders = [Symbol(i) for i in sk_dataset_methods if occursin("load_", i)]

for py_fun in data_loaders
    @eval $py_fun(args...; kwargs...) =
    tweak_datasets(sk_datasets.$(py_fun)(args...; kwargs...))
end


# ################################################################################
# # Samples generator
# ################################################################################
const sample_generators = [Symbol(i) for i in sk_dataset_methods if occursin("make_", i)]

for py_fun in sample_generators
    @eval $py_fun(args...; kwargs...) =
    tweak_datasets(sk_datasets.$(py_fun)(args...; kwargs...))
end
