################################
# Solving the Replicator Equator equation, 
# appearing in Corrolary 2.2 of
# https://www.biorxiv.org/content/10.1101/623330v1
# with a functor model
################################


cd(@__DIR__)
using HighDimPDE
using Random
using Test
using Flux
using Revise
using PyPlot
using UnPack
using Functors
plotting = true

tspan = (0f0,5f-1)
dt = 1f-2 # time step
μ(X,p,t) = 0f0 # advection coefficients
σ(X,p,t) = 1f-1 # diffusion coefficients
d = 5
ss0 = 5f-2 #std g0
U = 1f0
x0_sample = repeat([-U,U]', d, 1)

##############################
####### Neural Network #######
##############################
batch_size = 1000
train_steps = 10000
K = 100

 
struct mymodel
        sigma_pred
        a_pred
end
@functor mymodel
(m::mymodel)(x) = m.a_pred ./ prod(m.sigma_pred) .* exp.(-5f-1 * sum((x ./ m.sigma_pred).^2,dims=1) ) 

sigma_pred = repeat([ss0],d)
a_pred = [1f0]
fn = mymodel(sigma_pred, a_pred)

opt = Flux.Optimiser(ExpDecay(0.1,
                1,
                1000,
                1e-6),
                ADAM() )#optimiser
alg = DeepSplitting(fn, K=K, opt = opt, mc_sample = UniformSampling(x0_sample[:,1], x0_sample[:,2]) )

##########################
###### PDE Problem #######
##########################
g(x) = Float32((2*π)^(-d/2)) * ss0^(- Float32(d) * 5f-1) * exp.(-5f-1 *sum(x .^2f0 / ss0, dims = 1)) # initial condition
m(x) = - 5f-1 * sum(x.^2, dims=1)
vol = prod(x0_sample[:,2] - x0_sample[:,1])
f(y, z, v_y, v_z, ∇v_y, ∇v_z, t) = max.(0f0, v_y) .* (m(y) .- vol * max.(0f0, v_z) .* m(z) ) # nonlocal nonlinear part of the

# defining the problem
prob = PIDEProblem(g, f, μ, σ, tspan, 
                    x0_sample = x0_sample
                    )
# solving
@time xgrid,ts,sol = solve(prob, 
                alg, 
                dt, 
                verbose = true, 
                abstol=1f-6,
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

        xgrid1 = collect((-U:5f-3:U))
        xgrid = [vcat(x, fill(0f0,d-1)) for x in xgrid1] 

        # Analytic sol
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

        for t in collect(tspan[1]: dt : tspan[2])
                ys = uanal.(xgrid, t, Ref(Dict()))
                ax[2].plot(xgrid1, reduce(hcat,ys)[:], label = "t = $t")
        end
        ax[2].set_title("Analytical solution")
        gcf()

        #Deepsplitting sol
        for i in 1:length(sol)
                ax[1].scatter(xgrid1, reduce(vcat,sol[i].(xgrid)), s = .2, label="t = $(dt * (i-1))")
        end
        gcf()

        ax[1].set_title("DeepSplitting")

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
                x = x0_sample[1,1]:dx:x0_sample[1,2]
                plt.contourf(x,x,g.(repeat(x,2)))
        end
end
gcf()