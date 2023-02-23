
@testset "Samples Generators" begin
    x, rows, cols = ScikitLearn.make_biclusters((100, 2), 5)
    @test typeof(x) == Matrix{Float64}
    @test typeof(rows) == Matrix{Bool}
    @test typeof(cols) == Matrix{Bool}
    @test size(x) == (100,2)
    @test size(rows) == (5,100)
    @test size(cols) == (5,2)

    x, y = ScikitLearn.make_blobs(100, 2; centers=3)
    @test typeof(x) == Matrix{Float64}
    # @test typeof(y) == Vector{Int64} skip=true
    @test size(x) == (100,2)
    @test size(y) == (100,)
    x, y, centers = ScikitLearn.make_blobs(100, 2; centers=3, return_centers=true)
    @test typeof(centers) == Matrix{Float64}
    @test size(centers) == (3,2)

    x, rows, cols = ScikitLearn.make_checkerboard((100, 2), 5)
    @test typeof(x) == Matrix{Float64}
    @test typeof(rows) == Matrix{Bool}
    @test typeof(cols) == Matrix{Bool}
    @test size(x) == (100,2)
    @test size(rows) == (5^2,100)
    @test size(cols) == (5^2,2)

    x, y = ScikitLearn.make_circles(100)
    @test typeof(x) == Matrix{Float64}
    @test typeof(y) == Vector{Int64}
    @test size(x) == (100,2)
    @test size(y) == (100,)

    x, y = ScikitLearn.make_classification(100, 20; n_classes=2)
    @test typeof(x) == Matrix{Float64}
    # @test typeof(y) == Vector{Int64} skip=true
    @test size(x) == (100,20)
    @test size(y) == (100,)
    @test size(unique(y)) == (2,)

    x, y = ScikitLearn.make_friedman1(100, 20)
    @test typeof(x) == Matrix{Float64}
    @test typeof(y) == Vector{Float64}
    @test size(x) == (100,20)
    @test size(y) == (100,)

    x, y = ScikitLearn.make_friedman1(100, 20)
    @test typeof(x) == Matrix{Float64}
    @test typeof(y) == Vector{Float64}
    @test size(x) == (100,20)
    @test size(y) == (100,)

    x, y = ScikitLearn.make_friedman2(100)
    @test typeof(x) == Matrix{Float64}
    @test typeof(y) == Vector{Float64}
    @test size(x) == (100,4)
    @test size(y) == (100,)

    x, y = ScikitLearn.make_friedman3(100)
    @test typeof(x) == Matrix{Float64}
    @test typeof(y) == Vector{Float64}
    @test size(x) == (100,4)
    @test size(y) == (100,)

    x, y = ScikitLearn.make_gaussian_quantiles(; n_samples=100, n_features=2)
    @test typeof(x) == Matrix{Float64}
    # @test typeof(y) == Vector{Int64} skip=true
    @test size(x) == (100,2)
    @test size(y) == (100,)

    x, y = ScikitLearn.make_hastie_10_2(100)
    @test typeof(x) == Matrix{Float64}
    @test typeof(y) == Vector{Float64}
    @test size(x) == (100,10)
    @test size(y) == (100,)

    x = ScikitLearn.make_low_rank_matrix(100, 5)
    @test typeof(x) == Matrix{Float64}
    @test size(x) == (100,5)

    x, y = ScikitLearn.make_moons(100)
    @test typeof(x) == Matrix{Float64}
    @test typeof(y) == Vector{Int64}
    @test size(x) == (100,2)
    @test size(y) == (100,)

    x, y = ScikitLearn.make_multilabel_classification(100, 20)
    @test typeof(x) == Matrix{Float64}
    # @test typeof(y) == Vector{Int64} skip=true
    @test size(x) == (100,20)
    @test size(y) == (100,5)
    x, y, p_c, p_w_c = ScikitLearn.make_multilabel_classification(100, 20; return_distributions=true, n_classes=5)
    @test typeof(p_c) == Vector{Float64}
    @test typeof(p_w_c) == Matrix{Float64}
    @test size(p_c) == (5,)
    @test size(p_w_c) == (20,5)

    x, y, coef = ScikitLearn.make_regression(100, 20; coef=true)
    @test typeof(x) == Matrix{Float64}
    @test typeof(y) == Vector{Float64}
    @test typeof(coef) == Vector{Float64}
    @test size(x) == (100,20)
    @test size(y) == (100,)
    @test size(coef) == (20,)

    x, t = ScikitLearn.make_s_curve(100)
    @test typeof(x) == Matrix{Float64}
    @test typeof(t) == Vector{Float64}
    @test size(x) == (100, 3)
    @test size(t) == (100,)

    x, dict, code = ScikitLearn.make_sparse_coded_signal(100; n_components=5, n_features=10, n_nonzero_coefs=5)
    @test typeof(x) == Matrix{Float64}
    @test typeof(t) == Vector{Float64}
    @test size(x) == (10, 100)
    @test size(t) == (100,)

    x = ScikitLearn.make_sparse_spd_matrix()
    @test typeof(x) == Matrix{Float64}
    @test size(x) == (1, 1)
    x = ScikitLearn.make_sparse_spd_matrix(2)
    @test typeof(x) == Matrix{Float64}
    @test size(x) == (2, 2)

    x, y = ScikitLearn.make_sparse_uncorrelated(100, 5)
    @test typeof(x) == Matrix{Float64}
    @test typeof(y) == Vector{Float64}
    @test size(x) == (100, 5)
    @test size(y) == (100,)


    x = ScikitLearn.make_spd_matrix(1)
    @test typeof(x) == Matrix{Float64}
    @test size(x) == (1, 1)
    x = ScikitLearn.make_spd_matrix(2)
    @test typeof(x) == Matrix{Float64}
    @test size(x) == (2, 2)

    x, t = ScikitLearn.make_swiss_roll(100)
    @test typeof(x) == Matrix{Float64}
    @test typeof(t) == Vector{Float64}
    @test size(x) == (100, 3)
    @test size(t) == (100,)
end
