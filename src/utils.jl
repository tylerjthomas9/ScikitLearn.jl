
"""
Extract all symbols from an expression
"""
_symbols_in(e::Expr) = union(_symbols_in(e.head), map(_symbols_in, e.args)...)
_symbols_in(e::Symbol) = Set([e])
_symbols_in(::Any) = Set()

"""
@sk_import imports models from the Python version of scikit-learn. For instance, the
Julia equivalent of
`from sklearn.linear_model import LinearRegression, LogicisticRegression` is:
    @sk_import linear_model: (LinearRegression, LogisticRegression)
    model = fit!(LinearRegression(), X, y)
"""
macro sk_import(expr)
    @assert @capture(expr, mod_:what_) "`@sk_import` syntax error. Try something like: @sk_import linear_model: (LinearRegression, LogisticRegression)"
    if haskey(translated_modules, mod)
        @warn("Module $mod has been ported to Julia - try `import ScikitLearn: $(translated_modules[mod])` instead")
    end
    if :sklearn in _symbols_in(expr)
        error("Bad @sk_import: please remove `sklearn.` (it is implicit)")
    end
    if isa(what, Symbol)    
        members = [what]
    else
        @assert @capture(what, ((members__),)) "Bad @sk_import statement"
    end
    mod_string = "sklearn.$mod"
    :(begin
        mod_obj = pyimport($mod_string)
        $([:(const $(esc(w)) = mod_obj.$(w))
           for w in members]...)
    end)
end

# convert return values back to Julia
tweak_rval(x) = x
function tweak_rval(x::Py)
    if pyisinstance(x, numpy.ndarray)
        return pyconvert(Array, x)
    else
        return pyconvert(Any, x)
    end
end