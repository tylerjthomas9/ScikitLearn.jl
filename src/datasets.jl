
function tweak_datasets(x::Py)
    if pyisinstance(x, PythonCall.pybuiltins.tuple) || pyisinstance(x, PythonCall.pybuiltins.map)
        return tuple([tweak_rval(i) for i in x]...)
    else
        return tweak_rval(x)
    end
end

# ################################################################################
# # Loaders
# ################################################################################


# ################################################################################
# # Samples generator
# ################################################################################
const samples_generators = [Symbol(i) for i in sk_dataset_methods if occursin("make_", i)]

for py_fun in samples_generators
    @eval $py_fun(args...; kwargs...) =
    tweak_datasets(sk_datasets.$(py_fun)(args...; kwargs...))
end
