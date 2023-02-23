
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
    fit!(kmeans, X)
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
