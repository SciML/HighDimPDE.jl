cd(@__DIR__)
# using Pkg; Pkg.activate(".")
using Flux, Zygote, LinearAlgebra, Statistics
# println("NNPDE_deepsplitting_tests")
using Test
# println("Starting Soon!")
include("../src/pde_solve_deepsplitting_4.jl")

using Random
Random.seed!(100)

d = 10 # number of dimensions
# one-dimensional heat equation
X0 = fill(0.0f0,d)  # initial points
tspan = (0.0f0,1f0)
dt = 0.1f0   # time step
batch_size = 8192
train_steps = 8000
σ_sampling = 0.1f0
K = 20

hls = d + 50 #hidden layer size

nn = Flux.Chain(Dense(d,hls,tanh),
                Dense(hls,hls,tanh),
                Dense(hls,1))

opt = Flux.Optimiser(ExpDecay(0.1,
                0.1,
                2000,
                1e-4),
                ADAM())

g(X) = exp.(-0.5f0 * sum(X.^2,dims=1))   # terminal condition
f(y,z,v_y,v_z,∇v_y,∇v_z,p,t) = v_y .* (1f0 .- v_z .* Float32(π^(d/2) * σ_sampling^d) ) # function from solved equation
μ_f(X,p,t) = 0.0f0
σ_f(X,p,t) = 0.1f0
mc_sample(x) = x + CUDA.randn(d,batch_size) * σ_sampling / sqrt(2f0)

## One should look at InitialPDEProble, this would surely be more appropriate
prob = PIDEProblem(g, f, μ_f, σ_f, X0, tspan)

alg = NNPDEDS(nn, K=K, opt = opt )

sol = solve(prob, alg, mc_sample,
            dt=dt,
            verbose = true,
            abstol=1e-5,
            maxiters = train_steps,
            batch_size=batch_size,
            use_cuda = true)
println("u1 = ", sol.u[end])

using Plots
Plots.plot(sol)
