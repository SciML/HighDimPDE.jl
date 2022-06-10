using Test: include
using HighDimPDE, Test

@testset "HighDimPDE" begin
    include("reflect.jl")
    include("MLP.jl")
    include("DeepSplitting.jl")
    include("DeepBSDE.jl")
    include("DeepBSDE_Han.jl")
    include("MCSample.jl")
end