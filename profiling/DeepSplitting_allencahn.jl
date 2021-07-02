using HighDimPDE
using Random
using Test
using Flux
using Revise
using Plots

# using the DeepSplitting alg
batch_size = 2000
train_steps = 1000
K = 1

tspan = (0.0,0.5)
dt = 0.1 # time step
μ(X,p,t) = 0.0 # advection coefficients
σ(X,p,t) = 0.1 # diffusion coefficients


u_domain = [-1f0,1f0]
d = 1

hls = d + 50 #hidden layer size

nn_batch = Flux.Chain(Dense(d,hls,tanh),
        BatchNorm(hls, affine=true),
        Dense(hls,hls,tanh),
        BatchNorm(hls, affine=true),
        Dense(hls,1)) # Neural network used by the scheme, with batch normalisation

nn = Flux.Chain(Dense(d,hls,tanh),
        Dense(hls,hls,tanh),
        Dense(hls,1)) # Neural network used by the scheme

opt = Flux.Optimiser(ExpDecay(0.1,
                0.1,
                200,
                1e-5),
                ADAM() )#optimiser
alg = DeepSplitting(nn_batch, K=K, opt = opt,mc_sample = UniformSampling(u_domain[1],u_domain[2]) )


X0 = fill(0f0,d)  # initial point
g(X) = exp.(-0.25f0 * sum(X.^2,dims=1))   # initial condition
a(u) = u - u^3
f(y,z,v_y,v_z,∇v_y,∇v_z, t) = a.(v_y) .- a.(v_z) #.* Float32(π^(d/2)) * σ_sampling^d .* exp.(sum(z.^2, dims = 1) / σ_sampling^2) # nonlocal nonlinear part of the
# f(y,z,v_y,v_z,∇v_y,∇v_z, t) = zeros(Float32,size(v_y))

# defining the problem
prob = PIDEProblem(g, f, μ, σ, X0, tspan, 
                    u_domain = u_domain
                    )
# solving
@time xgrid, sol = solve(prob, 
                alg, 
                dt=dt, 
                verbose = true, 
                abstol=1e-6,
                maxiters = train_steps,
                batch_size=batch_size,
                )

Plots.plot()
for i in 1:length(sol)
        Plots.scatter!(reduce(vcat,xgrid), reduce(vcat,sol[i].(xgrid)))
end
Plots.plot!()