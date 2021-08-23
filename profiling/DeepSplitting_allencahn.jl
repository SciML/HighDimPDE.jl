using HighDimPDE
using Random
using Test
using Flux
using Revise
using PyPlot

tspan = (0.0,1f0)
dt = 1f-1 # time step
μ(X,p,t) = 0f0 # advection coefficients
σ(X,p,t) = 1f-1 # diffusion coefficients

d = 5
u_domain = repeat([-5f-1,5f-1]',d,1)

##############################
####### Neural Network #######
##############################
batch_size = 1000
train_steps = 10000
K = 100

hls = d + 50 #hidden layer size

nn_batch = Flux.Chain(Dense(d,hls,tanh),
        # BatchNorm(hls, affine=true),
        Dense(hls,hls,tanh),
        # BatchNorm(hls, affine=true),
        Dense(hls,1)) # Neural network used by the scheme, with batch normalisation

opt = Flux.Optimiser(ExpDecay(0.1,
                0.2,
                2000,
                1e-6),
                ADAM() )#optimiser
alg = DeepSplitting(nn_batch, K=K, opt = opt, mc_sample = UniformSampling(u_domain[1], u_domain[2]) )

##########################
###### PDE Problem #######
##########################
g(x) = exp.(-0.25f0 * sum(x.^2, dims = 1))   # initial condition
a(u) = u - u^3

# for uniform sampling of nl term
vol = prod(u_domain[:,2] - u_domain[:,1])
f(y, z, v_y, v_z, ∇v_y, ∇v_z, t) = a.(v_y) .- a.(v_z) * vol
# for random sampling of nl term
# f(y, z, v_y, v_z, ∇v_y, ∇v_z, t) = a.(v_y) .- a.(v_z) .* Float32(π^(d/2)) * σ_sampling^d .* exp.(sum(z.^2, dims = 1) / σ_sampling^2)

# defining the problem
prob = PIDEProblem(g, f, μ, σ, tspan, 
                    u_domain = u_domain
                    )
# solving
@time xgrid, sol = HighDimPDE.solve(prob, 
                alg, 
                dt, 
                verbose = true, 
                abstol=5f-7,
                maxiters = train_steps,
                batch_size=batch_size,
                use_cuda = true
                )

# plotting
if false
        if d == 1
                plt.figure()
                for i in 1:length(sol)
                        plt.scatter(reduce(vcat,xgrid), reduce(vcat,sol[i].(xgrid)), s=0.1)
                end
        elseif d == 2
                plt.figure()
                xy = reduce(hcat,xgrid)
                plt.scatter(xy[1,:], xy[2,:], c = reduce(vcat,sol[1].(xgrid)))
        else
                plt.figure()
                ts = 0.: dt : tspan[2]
                ys = [sol[i](zeros(d,1))[] for i in 1:length(sol)]
                plt.plot(collect(ts),ys)
        end
        gcf()
end