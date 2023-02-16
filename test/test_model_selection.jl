
@testset "train_test_split" begin
    return_type = Tuple{Matrix{Float64}, Matrix{Float64}, Vector{Float64}, Vector{Float64}}

    # Python input
    X = ScikitLearn.numpy.random.rand(100, 5)
    y = ScikitLearn.numpy.random.rand(100)
    res = train_test_split(X, y, test_size=0.33, random_state=42)
    @test typeof(res) == return_type

    # Julia input
    X = rand(100,5)
    y = rand(100)
    res = train_test_split(X, y, test_size=0.33, random_state=42)
    @test typeof(res) == return_type

    # mixture
    X = ScikitLearn.numpy.random.rand(100, 5)
    y = rand(100)
    res = train_test_split(X, y, test_size=0.33, random_state=42)
    @test typeof(res) == return_type
    X = rand(100, 5)
    y = ScikitLearn.numpy.random.rand(100)
    res = train_test_split(X, y, test_size=0.33, random_state=42)
    @test typeof(res) == return_type
end

@testset "check_cv" begin
    check_cv(5)
    check_cv(5, rand(100))
    check_cv(5, rand(100); classifier=true)
end

@testset "cross_validate" begin
    logistic_reg = LogisticRegression(max_iter=2)
    X = rand(100, 5)
    y = rand(0:1, 100)
    cv_results = cross_validate(logistic_reg, X, y, cv=3)
    @test typeof(cv_results) == Dict{String, Array}
end

@testset "cross_val_predict" begin
    logistic_reg = LogisticRegression(max_iter=2)
    X = rand(100, 5)
    y = rand(0:1, 100)
    preds = cross_val_predict(logistic_reg, X, y, cv=3)
    @test typeof(preds) == Vector{Int64}
    @test size(preds) == size(y)

    preds = cross_val_predict(logistic_reg, X, y, cv=3, method="predict_proba")
    @test typeof(preds) == Matrix{Float64}
    @test size(preds) == (size(y, 1), 2)
end

@testset "learning_curve" begin
    logistic_reg = LogisticRegression(max_iter=2)
    X = rand(100, 5)
    y = rand(0:1, 100)
    res = learning_curve(logistic_reg, X, y, cv=3)
    @test length(res) == 3
    @test isa(res[1], Vector)
    @test typeof(res[2]) == Matrix{Float64}
    @test typeof(res[3]) == Matrix{Float64}

    res = learning_curve(logistic_reg, X, y, cv=3, return_times=true)
    @test length(res) == 5
    @test typeof(res[4]) == Matrix{Float64}
    @test typeof(res[5]) == Matrix{Float64}
end

@testset "permutation_test_score" begin
    logistic_reg = LogisticRegression(max_iter=2)
    X = rand(100, 5)
    y = rand(0:1, 100)
    res = permutation_test_score(logistic_reg, X, y, cv=3)
    @test length(res) == 3
    @test typeof(res[1]) == Float64
    @test typeof(res[2]) == Vector{Float64}
    @test typeof(res[3]) == Float64
end

@testset "validation_curve" begin
    logistic_reg = LogisticRegression(max_iter=2)
    X = rand(100, 5)
    y = rand(0:1, 100)
    param_name = "C"
    param_range = [0.1, 0.2, 0.5, 0.75, 1.0]
    res = validation_curve(logistic_reg, X, y, cv=3; param_name, param_range)
    @test length(res) == 2
    @test typeof(res[1]) == Matrix{Float64}
    @test typeof(res[2]) == Matrix{Float64}
end
