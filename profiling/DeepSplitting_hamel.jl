using HighDimPDE
using Random
using Test
using Flux
using Revise
using PyPlot

# using the DeepSplitting alg
batch_size = 5000
train_steps = 2000
K = 10

tspan = (0.0,5f-1)
dt = 1f-1 # time step
μ(X,p,t) = 0f0 # advection coefficients
σ(X,p,t) = 1f-1 # diffusion coefficients


u_domain = [-1f0,1f0]
d = 2

hls = d + 50 #hidden layer size

nn_batch = Flux.Chain(Dense(d,hls,tanh),
        BatchNorm(hls, affine=true),
        Dense(hls,hls,tanh),
        BatchNorm(hls, affine=true),
        Dense(hls,1)) # Neural network used by the scheme, with batch normalisation

opt = Flux.Optimiser(ExpDecay(0.1,
                0.1,
                500,
                1e-6),
                ADAM() )#optimiser
alg = DeepSplitting(nn_batch, K=K, opt = opt,mc_sample = UniformSampling(u_domain[1],u_domain[2]) )


X0 = fill(0f0,d)  # initial point
g(X) = Float32(2f0^(d/2))* exp.(-2f0 * Float32(π)  * sum( X.^2, dims=1))   # initial condition
m(x) = - 5f-1 * sum(x.^2, dims=1)
f(y, z, v_y, v_z, ∇v_y, ∇v_z, t) = max.(0f0, v_y) .* ( m(y) - max.(0f0, v_z) .* m(z) * Float32((2f0 * π)^(d/2) * σ_sampling^d) .* exp.(5f-1 * sum(z.^2, dims = 1) / σ_sampling^2)) # nonlocal nonlinear part of the

# defining the problem
prob = PIDEProblem(g, f, μ, σ, X0, tspan, 
                    u_domain = u_domain
                    )
# solving
@time xgrid, sol = solve(prob, 
                alg, 
                dt=dt, 
                verbose = true, 
                abstol=5f-5,
                maxiters = train_steps,
                batch_size=batch_size,
                use_cuda = true
                )

plt.figure()
for i in 1:length(sol)
        plt.scatter(reduce(vcat,xgrid), reduce(vcat,sol[i].(xgrid)))
end
gcf()