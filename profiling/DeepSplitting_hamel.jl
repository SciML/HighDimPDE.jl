using HighDimPDE
using Random
using Test
using Flux
using Revise
using PyPlot

# using the DeepSplitting alg
batch_size = 1000
train_steps = 10000
K = 100

tspan = (0.0,5f-1)
dt = 1f-1 # time step
μ(X,p,t) = 0f0 # advection coefficients
σ(X,p,t) = 1f-1 # diffusion coefficients

d = 2
u_domain = repeat([-1f0,1f0]',d,1)

hls = d + 50 #hidden layer size

nn_batch = Flux.Chain(Dense(d,hls,tanh),
        BatchNorm(hls, affine=true),
        Dense(hls,hls,tanh),
        BatchNorm(hls, affine=true),
        Dense(hls,1)) # Neural network used by the scheme, with batch normalisation

opt = Flux.Optimiser(ExpDecay(0.1,
                1.0,
                1000,
                1e-6),
                ADAM() )#optimiser
alg = DeepSplitting(nn_batch, K=K, opt = opt,mc_sample = UniformSampling(u_domain[1],u_domain[2]) )


g(X) = Float32(2f0^(d/2))* exp.(-2f0 * Float32(π)  * sum( X.^2, dims=1))   # initial condition
m(x) = - 5f-1 * sum(x.^2, dims=1)
vol = prod(u_domain[:,2] - u_domain[:,1])
f(y, z, v_y, v_z, ∇v_y, ∇v_z, t) = max.(0f0, v_y) .* ( m(y) .- max.(0f0, v_z) .* m(z) / vol) # nonlocal nonlinear part of the

# defining the problem
prob = PIDEProblem(g, f, μ, σ, tspan, 
                    u_domain = u_domain
                    )
# solving
@time xgrid,sol = solve(prob, 
                alg, 
                dt, 
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

dx = 0.05
x = u_domain[1,1]:dx:u_domain[1,2]
plt.contourf(x,x,g.(repeat(x,2)))