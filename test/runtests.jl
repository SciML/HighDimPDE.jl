using Test: include
using HighDimPDE, Test

@testset "HighDimPDE" begin
    include("reflect.jl")
    include("MLP.jl")
    include("DeepSplitting")
end