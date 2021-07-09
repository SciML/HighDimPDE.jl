cd(@__DIR__)
using HighDimPDE
using Random
using Test
using Flux
using Revise
using PyPlot
using UnPack
plotting = true

tspan = (0.0,1f-1)
dt = 1f-1 # time step
μ(X,p,t) = 0f0 # advection coefficients
σ(X,p,t) = 1f-1 # diffusion coefficients
d = 5
ss0 = 1f0 #std g0

u_domain = repeat([-2f0,2f0]', d, 1)

##############################
####### Neural Network #######
##############################
batch_size = 1000
train_steps = 10000
K = 500

hls = d + 50 #hidden layer size

nn_batch = Flux.Chain(Dense(d,hls,tanh),
        # BatchNorm(hls, affine=true),
        Dense(hls,hls,tanh),
        # BatchNorm(hls, affine=true),
        Dense(hls,1)) # Neural network used by the scheme, with batch normalisation

opt = Flux.Optimiser(ExpDecay(0.1,
                10.0,
                1000,
                1e-6),
                ADAM() )#optimiser
alg = DeepSplitting(nn_batch, K=K, opt = opt,mc_sample = UniformSampling(u_domain[1],u_domain[2]) )

##########################
###### PDE Problem #######
##########################
g(x) = Float32((2*π)^(-d/2)) * ss0^(- Float32(d) * 5f-1) * exp.(-5f-1 *sum(x .^2f0 / ss0, dims = 1)) # initial condition
m(x) = - 5f-1 * sum(x.^2, dims=1)
vol = prod(u_domain[:,2] - u_domain[:,1])
f(y, z, v_y, v_z, ∇v_y, ∇v_z, t) = max.(0f0, v_y) .* (m(y) .- vol * max.(0f0, v_z) .* m(z) ) # nonlocal nonlinear part of the

# defining the problem
prob = PIDEProblem(g, f, μ, σ, tspan, 
                    u_domain = u_domain
                    )
# solving
@time xgrid,sol = solve(prob, 
                alg, 
                dt, 
                verbose = true, 
                abstol=5f-8,
                maxiters = train_steps,
                batch_size=batch_size,
                use_cuda = true
                )

###############################
######### Plotting ############
###############################
if plotting
        clf()
        fig, ax = plt.subplots(1,2, sharey = true)

        xgrid1 = collect((-3f0:5f-2:3f0))
        xgrid = [vcat(x, fill(0f0,d-1)) for x in xgrid1] 

        for i in 1:length(sol)
                ax[1].scatter(xgrid1, reduce(vcat,sol[i].(xgrid)), s = .2, label="t = $(dt * (i-1))")
        end
        gcf()

        function _SS(x, t, p)
                d = length(x)
                MM = σ(x, p, t) * ones(d)
                SSt = MM .* ((MM .* sinh.(MM *t) .+ ss0 .* cosh.( MM * t)) ./ (MM .* cosh.(MM * t ) .+ ss0 .* sinh.(MM * t)))
                return SSt
        end

        function uanal(x, t, p)
                d = length(x)
                return (2*π)^(-d/2) * prod(_SS(x, t, p) .^(-1/2)) * exp(-0.5 *sum(x .^2 ./ _SS(x, t, p)) )
        end

        for t in collect(0.:0.1:0.5)
                ys = uanal.(xgrid, t, Ref(Dict()))
                ax[2].plot(xgrid1, reduce(hcat,ys)[:], label = "t = $t")
        end

        ax[1].set_title("DeepSplitting")
        ax[2].set_title("Analytical solution")

        for _a in ax
                _a.legend()
        end
        gcf()
        savefig("hamel_$(d)d.pdf")

        #####
        # other DimensionMismatch
        #####
        if false
                dx = 0.05
                x = u_domain[1,1]:dx:u_domain[1,2]
                plt.contourf(x,x,g.(repeat(x,2)))
        end
end