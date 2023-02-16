
@testset "is classifier/regressor/pairwise/transformer" begin
    svc = SVC(max_iter=2)
    svr = SVR(max_iter=2)
    class_pipe = Pipeline([("scaler", StandardScaler()), ("svc", SVC())])
    reg_pipe = Pipeline([("scaler", StandardScaler()), ("svr", SVR())])
    
    @test is_classifier(svc)
    @test is_classifier(class_pipe)
    @test !is_classifier(svr)
    @test !is_classifier(reg_pipe)

    @test !is_regressor(svc)
    @test !is_regressor(class_pipe)
    @test is_regressor(svr)
    @test is_regressor(reg_pipe)
end

@testset "get_classes/get_components" begin
    iris = load_iris()
    X = iris["data"]
    y = iris["target"]
    svc = SVC(max_iter=2)
    fit!(svc, X, y)
end

@testset "get_params/set_params!" begin
    svc = SVC(C=1.0)
    params = get_params(svc)
    @test typeof(params) == Dict{Symbol, Any}
    @test params[:C] == 1.0

    set_params!(svc, C=0.5)
    @test get_params(svc)[:C] == 0.5

    params[:C] = 0.75
    set_params!(svc, params)
    @test get_params(svc)[:C] == 0.75

    #TODO: Why does this pass in the repl, but not with running tests
    @test_throws PyException set_params!(svc, D=1.0)

end

@testset "clone" begin
    svc = SVC(C=1.0)
    new_svc = clone(svc)
    @test(svc !== new_svc)
    @test(get_params(new_svc) == get_params(new_svc))
end

@testset "get_config/set_config" begin
    cfg = ScikitLearn.get_config()
    @test typeof(cfg) == Dict{Symbol, Any}

    ScikitLearn.set_config(working_memory=1025)
    @test ScikitLearn.get_config()[:working_memory] == 1025
end


@testset "fit/predict" begin
    logistic_reg = LogisticRegression(max_iter=2)
    linear_reg = LinearRegression()

    # python input
    iris = load_iris()
    X = iris["data"]
    y = iris["target"]
    fit!(logistic_reg, X, y)
    predict(logistic_reg, X)
    predict_proba(logistic_reg, X)
    predict_log_proba(logistic_reg, X)
    fit!(linear_reg, X, y)
    predict(linear_reg, X)

    # julia array
    X = rand(100, 5)
    y_classification = rand(0:1, 100)
    y_regression = rand(100)
    fit!(logistic_reg, X, y_classification)
    predict(logistic_reg, X)
    predict_proba(logistic_reg, X)
    predict_log_proba(logistic_reg, X)
    fit!(linear_reg, X, y_regression)
    predict(linear_reg, X)

    # fit_predict
    kmeans = KMeans(n_clusters=2, max_iter=2, n_init=2)
    fit_predict!(kmeans, X)
end


@testset "fit/transform" begin
    scaler = StandardScaler()
    X = rand(100, 5)

    fit!(scaler, X)
    transform(scaler, X)
    X_trans = fit_transform!(scaler, X)
    X_invtrans = inverse_transform(scaler, X_trans)
    @test isapprox(X_invtrans, X, atol=1e-3)
end

@testset "score" begin
    logistic_reg = LogisticRegression(max_iter=2)
    linear_reg = LinearRegression()

    # python input
    X = rand(100, 5)
    y_classification = rand(0:1, 100)
    y_regression = rand(100)
    fit!(logistic_reg, X, y_classification)
    fit!(linear_reg, X, y_regression)

end


