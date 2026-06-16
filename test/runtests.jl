using SciMLTesting
using SafeTestsets

# GPU's GROUP semantics are capability-based, not folder-partitioned: the GPU lane
# runs the *same* Core test files on a CUDA-equipped runner, where the
# `CUDA.functional()`-gated testsets inside reflect.jl/MCSample.jl self-activate.
# It therefore cannot be a separate folder of files, so this uses explicit-args
# run_tests rather than folder discovery.
function core_body()
    @safetestset "ProblemConstructors" include("ProblemConstructor.jl")
    @safetestset "reflect" include("reflect.jl")
    @safetestset "MLP" include("MLP.jl")
    @safetestset "Deep Splitting" include("DeepSplitting.jl")
    @safetestset "DeepBSDE" include("DeepBSDE.jl")
    @safetestset "MC Sample" include("MCSample.jl")
    @safetestset "NNStopping" include("NNStopping.jl")
    @safetestset "NNKolmogorov" include("NNKolmogorov.jl")
    return @safetestset "NNParamKolmogorov" include("NNParamKolmogorov.jl")
end

run_tests(;
    core = core_body,
    groups = Dict(
        # GPU is the self-hosted CUDA runner lane: the same Core suite, where the
        # CUDA-gated testsets run because CUDA.functional() is true there.
        "GPU" => core_body,
    ),
    qa = (; env = joinpath(@__DIR__, "qa"), body = joinpath(@__DIR__, "qa", "qa.jl")),
    # Curated "All": run only Core. GPU (self-hosted CUDA lane) and QA stay
    # selectable by name but out of the aggregate.
    all = ["Core"],
)
