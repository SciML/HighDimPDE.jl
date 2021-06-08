cd(@__DIR__)
# using Pkg; Pkg.activate(".")
using Flux, Zygote, LinearAlgebra, Statistics
# println("NNPDE_deepsplitting_tests")
using Test
# println("Starting Soon!")
using Revise
include("pde_solve_deepsplitting_4.jl")
# using NeuralPDE

using Random
Random.seed!(100)

d = 1 # number of dimensions
# one-dimensional heat equation
X0 = fill(0.0f0,d)  # initial points
tspan = (0.0f0,1f0)
dt = 0.005f0   # time step
batch_size = 16000
train_steps = 16000
σ_sampling = 0.7f0
K = 1f0

hls = d + 50 #hidden layer size

nn = Flux.Chain(Dense(d,hls,tanh),
                Dense(hls,hls,tanh),
                Dense(hls,1))

opt = Flux.Optimiser(ExpDecay(0.1,
                0.1,
                4000,
                1e-4),
                ADAM())

g(X) = exp.(-0.5f0 * sum(X.^2,dims=1))   # terminal condition
a(X) = 1f0 .- 0.1f0 .* sum(X.^2,dims=1)
_0 = CUDA.zeros(Float32,1,batch_size)
f(y,z,v_y,v_z,dv_y,dv_z,p,t) = max.(v_y , _0) .* (a(y) - 0.1f0 * max.(v_z , _0) .* (Float32(π^(d/2) .* σ_sampling^d) .* exp.( sum(z.^2,dims=1)) / σ_sampling^2)) # function from solved equation
μ(X,p,t) = 0f0
σ(X,p,t) = 0.1f0 * sqrt(2f0)
mc_sample(x) = CUDA.randn(d, batch_size) * σ_sampling

## One should look at InitialPDEProblem, this would surely be more appropriate
prob = PIDEProblem(g, f, μ, σ, X0, tspan)

alg = NNPDEDS(nn,K=1,opt = opt )

u1 = solve(prob, alg, mc_sample,
            dt=dt,
            verbose = true,
            abstol=1e-3,
            maxiters = train_steps,
            batch_size=batch_size,
            use_cuda = true)

println("u1 = ", u1)
