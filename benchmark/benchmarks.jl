using HighDimPDE, BenchmarkTools
using Flux, SciMLBase, Random, LinearAlgebra
using StableRNGs

const SUITE = BenchmarkGroup()

rng = StableRNG(1234)
d = 5
hls = d + 10

μ(x, p, t) = 0.0f0
σ(x, p, t) = 1.0f-1
g(x) = sum(x .^ 2, dims = 1)
f(y, v_y, ∇v_y, p, t) = 0.0f0 .* v_y
x0 = fill(2.0f0, d)
tspan = (0.0f0, 0.5f0)

prob = ParabolicPDEProblem(μ, σ, x0, tspan; g, f)

# =============================================================================
# Monte Carlo sampling primitives
# =============================================================================

SUITE["sampling"] = BenchmarkGroup()

X = rand(rng, Float32, 10, 4000)

SUITE["sampling"]["uniform"] = @benchmarkable UniformSampling(0.0f0, 1.0f0)($X)
SUITE["sampling"]["normal"] = @benchmarkable NormalSampling(1.0f0, false)($X)

# =============================================================================
# Algorithm + problem construction
# =============================================================================

SUITE["construction"] = BenchmarkGroup()

SUITE["construction"]["pde_problem"] = @benchmarkable ParabolicPDEProblem(
    $μ, $σ, $x0, $tspan; g = $g, f = $f
)
SUITE["construction"]["deep_splitting"] = @benchmarkable DeepSplitting(
    Flux.Chain(Dense($d, $hls, relu), Dense($hls, $hls, relu), Dense($hls, 1));
    opt = Flux.Optimise.Adam(0.01)
)

# =============================================================================
# Short training solves — representative of the solve pipeline, bounded by
# maxiters rather than convergence
# =============================================================================

SUITE["solve"] = BenchmarkGroup()

function make_nn()
    return Flux.Chain(Dense(d, hls, relu), Dense(hls, hls, relu), Dense(hls, 1))
end

SUITE["solve"]["deep_splitting"] = @benchmarkable solve(
    $prob, DeepSplitting(make_nn(); opt = Flux.Optimise.Adam(0.01)), 0.5f0;
    verbose = false, maxiters = 20, batch_size = 200
) seconds = 300

μ_bsde(X, p, t) = X * 0.0f0
σ_f(X, p, t) = Matrix(Diagonal(ones(eltype(X), d)))
f_bsde(X, u, σᵀ∇u, p, t) = eltype(X)(0.0)
g_bsde(X) = sum(X .^ 2)

# DeepBSDE's solve path does a Zygote pullback over a serial SDE ensemble —
# too heavy for a regression suite, so it is benchmarked at construction only.
SUITE["construction"]["deep_bsde"] = @benchmarkable DeepBSDE(
    Flux.Chain(Dense($d, $hls, relu), Dense($hls, $hls, relu), Dense($hls, 1)),
    Flux.Chain(
        Dense($d + 1, $hls, relu), Dense($hls, $hls, relu),
        Dense($hls, $hls, relu), Dense($hls, $d)
    );
    opt = Flux.Optimise.Adam(0.05)
)
SUITE["construction"]["pde_problem_bsde"] = @benchmarkable ParabolicPDEProblem(
    $μ_bsde, $σ_f, $(Float32[1.0, 1.0, 1.0, 1.0, 1.0]), $tspan;
    g = $g_bsde, f = $f_bsde
)
