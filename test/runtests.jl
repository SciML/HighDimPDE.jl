using SafeTestsets, Test, Pkg

const GROUP = get(ENV, "GROUP", "All")

if GROUP == "QA"
    Pkg.activate(joinpath(@__DIR__, "qa"))
    Pkg.develop(PackageSpec(path = joinpath(@__DIR__, "..")))
    Pkg.instantiate()
    @time @safetestset "Quality Assurance" include(joinpath("qa", "qa.jl"))
else
    @testset "HighDimPDE" begin
        @time @safetestset "ProblemConstructors" include("ProblemConstructor.jl")
        @time @safetestset "reflect" include("reflect.jl")
        @time @safetestset "MLP" include("MLP.jl")
        @time @safetestset "Deep Splitting" include("DeepSplitting.jl")
        @time @safetestset "Deep Splitting" include("DeepBSDE.jl")
        @time @safetestset "MC Sample" include("MCSample.jl")
        @time @safetestset "NNStopping" include("NNStopping.jl")
        @time @safetestset "NNKolmogorov" include("NNKolmogorov.jl")
        @time @safetestset "NNParamKolmogorov" include("NNParamKolmogorov.jl")
    end
end
