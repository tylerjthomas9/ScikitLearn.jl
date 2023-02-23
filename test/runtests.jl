using Test
using PythonCall
using ScikitLearn

# Filter out warnings for convergence during testing
warnings = PythonCall.pynew()
PythonCall.pycopy!(warnings, pyimport("warnings"))
warnings.simplefilter(; action="ignore", 
        category=sklearn.exceptions.ConvergenceWarning)
warnings.simplefilter(; action="ignore", 
        category=PythonCall.pybuiltins.RuntimeWarning)

# python imports
@sk_import cluster: KMeans
@sk_import datasets: load_iris
@sk_import decomposition: (PCA, TruncatedSVD)
@sk_import feature_selection: (SelectKBest, f_classif)
@sk_import linear_model: (LinearRegression, LogisticRegression)
@sk_import pipeline: (FeatureUnion, Pipeline)
@sk_import preprocessing: StandardScaler
@sk_import svm: (SVC, SVR)

@testset "test_base" begin
    include("test_base.jl")
end

@testset "test_sklearn_api" begin
    include("test_sklearn_api.jl")
end

@testset "test_pipeline" begin
    include("test_pipeline.jl")
end

@testset "test_model_selection" begin
    include("test_model_selection.jl")
end

@testset "test_datasets" begin
    include("test_datasets.jl")
end
