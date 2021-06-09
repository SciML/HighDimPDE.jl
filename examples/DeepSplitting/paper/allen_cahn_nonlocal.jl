cd(@__DIR__)
name_sim = split(splitpath(@__FILE__)[end],".")[1]
using Pkg; Pkg.activate("../.");Pkg.instantiate()
using HighDimPDE
using Flux
using Random
using CUDA
Random.seed!(100)
# for post processes
using DataFrames
using Latexify
using LaTeXStrings
using CSV

function allen_cahn_nonlocal(;d,tspan,dt,batch_size,train_steps,σ_sampling,K)
        X0 = fill(0.5f0,d)  # initial point

        hls = d + 50 #hidden layer size

        nn = Flux.Chain(Dense(d,hls,tanh),
                        Dense(hls,hls,tanh),
                        Dense(hls,1)) # Neural network used by the scheme

        opt = Flux.Optimiser(ExpDecay(0.1,
                        0.1,
                        2000,
                        1e-4),
                        ADAM() )#optimiser

        g(X) = exp.(-0.25f0 * sum(X.^2,dims=1))   # initial condition
        a(u) = u - u^3
        f(y,z,v_y,v_z,∇v_y,∇v_z,p,t) = a.(v_y) .- a.(v_z) .* Float32(π^(d/2) * σ_sampling^d) # nonlocal nonlinear part of the
        μ_f(X,p,t) = 0.0f0 # advection coefficients
        σ_f(X,p,t) = sqrt(2f0) # diffusion coefficients
        mc_sample(x) = x + CUDA.randn(d,batch_size) * σ_sampling / sqrt(2f0) #montecarlo samples

        # defining the problem
        prob    = PIDEProblem(g, f, μ_f, σ_f, X0, tspan,
                                u_domain = [0f0,1f0]
                                )

        # using the Deep Splitting algorithm
        alg = DeepSplitting(nn, K=K, opt = opt )

        # solving
        sol = solve(prob, alg, mc_sample,
                    dt=dt,
                    verbose = true,
                    abstol=1e-5,
                    maxiters = train_steps,
                    batch_size=batch_size,
                    use_cuda = true)
        println("u1 = ", sol.u[end])

        sol
end

## Basic example
if false
        sol = allen_cahn_nonlocal(
                d = 3, # number of dimensions
                # one-dimensional heat equation
                tspan = (0.0f0,1f0),
                dt = 0.1f0,   # time step
                batch_size = 8192,
                train_steps = 8000,
                σ_sampling = 1f0,
                K = 5,
                )
        Plots.plot(sol)
end

########################
### For publication ####
########################
if true
        df = DataFrame(); [df[c] = Float64[] for c in [L"d",L"T",L"N","Mean","Std. dev."]]
        dfu = DataFrame(); [dfu[c] = Float64[] for c in ["d","T","N","u"]]
        for d in [1,2,5,10],T in [1/5,1/2,1]
                N = 10
                u = []
                dt = T / N
                for i in 1:5
                        sol = allen_cahn_nonlocal(
                                d = d, # dimension of the domain (D = \R^d)
                                # one-dimensional heat equation
                                tspan = (0.0f0,T),
                                dt = dt,   # time step
                                batch_size = 8192,
                                train_steps = 8000,
                                σ_sampling = 5f-1,
                                K = 1,
                                )
                        push!(u,sol.u[end])
                        push!(dfu,(d,T,N,sol.u[end]))
                end
        push!(df,(d,T,N,mean(u),std(u)))
        end
        tab = latexify(df,env=:tabular,fmt="%.7f") #|> String
        io = open("$(name_sim).tex", "w")
        write(io,tab);
        close(io)
        CSV.write("$(name_sim).csv", dfu)
end
