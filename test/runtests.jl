using SafeTestsets, Test
@testset "HighDimPDE" begin
    @time @safetestset "Quality Assurance" include("qa.jl")
    @time @safetestset "reflect" include("reflect.jl")
    @time @safetestset "reflect" include("reflect.jl")
    @time @safetestset "MLP" include("MLP.jl")
    @time @safetestset "Deep Splitting" include("DeepSplitting.jl")
    @time @safetestset "MC Sample" include("MCSample.jl")
    @time @safetestset "NNStopping" include("NNStopping.jl")
end
