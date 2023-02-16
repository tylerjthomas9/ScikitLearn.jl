
@testset "Pipeline Init" begin
    # Test the various init parameters of the pipeline.
    @test_throws(PyException, Pipeline())

    # Smoke test with only an estimator
    clf = SVC()
    pipe = Pipeline([("svc", clf)])

    # Check that params are set
    set_params!(pipe; svc__C=0.1)
    @test pyconvert(Float64, clf.C)==0.1

    # Test with two objects
    clf = SVC(gamma="auto")
    filter1 = SelectKBest(f_classif)
    pipe = Pipeline([("anova", filter1), ("svc", clf)])

    # Check that params are set
    set_params!(pipe, svc__C=0.1)
    @test pyconvert(Float64, clf.C)==0.1

    # Check that params are not set when naming them wrong
    @test_throws(PyException, set_params!(pipe; anova__C=0.1))

    # Test clone
    pipe2 = clone(pipe)
    @test ScikitLearn.named_steps(pipe)["svc"] !== ScikitLearn.named_steps(pipe2)["svc"]

    # Check that apart from python objects, the parameters are the same
    params = get_params(pipe)
    params2 = get_params(pipe2)
    delete!(params, :svc)
    delete!(params, :anova)
    delete!(params, :steps)
    delete!(params, :anova__score_func)
    delete!(params2, :svc)
    delete!(params2, :anova)
    delete!(params2, :steps)
    delete!(params2, :anova__score_func)
    @test params == params2
end


@testset "Pipeline Anova" begin
    # Test the various methods of the pipeline (anova).
    iris = load_iris()
    X = iris["data"]
    y = iris["target"]
    # Test with Anova + LogisticRegression
    clf = LogisticRegression(solver="lbfgs", multi_class="auto")
    filter1 = SelectKBest(f_classif, k=2)
    pipe = Pipeline([("anova", filter1), ("logistic", clf)])
    fit!(pipe, X, y)
    @test isa(predict(pipe, X), Vector)
    @test typeof(predict_proba(pipe, X)) == Matrix{Float64}
    @test typeof(predict_log_proba(pipe, X)) == Matrix{Float64}
    @test typeof(score(pipe, X, y)) == Float64

    X = rand(100, 5)
    y = rand([0,1], 100)
    fit!(pipe, X, y)
    @test isa(predict(pipe, X), Vector)
    @test typeof(predict_proba(pipe, X)) == Matrix{Float64}
    @test typeof(predict_log_proba(pipe, X)) == Matrix{Float64}
    @test typeof(score(pipe, X, y)) == Float64
end

@testset "Pipeline Fit/Transform" begin
    # Test the various methods of the pipeline (anova).
    iris = load_iris()
    X = iris["data"]
    y = iris["target"]
    # Test with Anova + LogisticRegression
    scaler = StandardScaler()
    pipe = Pipeline([("scaler", scaler)])

    # test fit_transform:
    X_trans = fit_transform!(pipe, X, y)
    X_trans2 = transform(fit!(pipe, X, y), X)
    @test isapprox(X_trans, X_trans2)
end


@testset "Feature Union" begin

    # basic sanity check for feature union
    iris = load_iris()
    X = rand(100, 5)
    y = rand(100)
    svd = TruncatedSVD(n_components=2, random_state=0)
    select = SelectKBest(k=1)
    fs = FeatureUnion([("svd", svd), ("select", select)])
    fit!(fs, X, y)
    X_transformed = transform(fs, X)
    @test size(X_transformed) == (size(X, 1), 3)

    # check if it does the expected thing
    @test isapprox(X_transformed[:, 1:end-1], fit_transform!(svd, X))
    @test(X_transformed[:, end] ==
            fit_transform!(select, X, y)[:])

    # test setting parameters
    set_params!(fs; select__k=2)
    @test size(fit_transform!(fs, X, y)) == (size(X, 1), 4)

    # feature union weights
    fs = FeatureUnion([("svd", svd), ("select", select)], 
                transformer_weights=Dict("svd"=>10))
    X_fit_transformed = fit_transform!(fs, X, y)
end

