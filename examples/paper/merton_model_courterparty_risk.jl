cd(@__DIR__)
name_sim = split(splitpath(@__FILE__)[end],".")[1]
# using Pkg; Pkg.activate(".")
using Flux, Zygote, LinearAlgebra, Statistics
# println("Starting Soon!")
include("../src/pde_solve_deepsplitting_4.jl")
using Random
Random.seed!(100)

# for post processes
using DataFrames
using Latexify
using LaTeXStrings
using CSV
using Test


function merton(;d,tspan,dt,batch_size,train_steps,σ_sampling,K)
        X0 = fill(0.0f0,d)  # initial points
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
        f(y,z,v_y,v_z,∇v_y,∇v_z,p,t) = - min(v_y,0f0) - v_z .* Float32(π^(d/2) * σ_sampling^d) # function from solved equation
        μ_f(X,p,t) = 0.0f0
        σ_f(X,p,t) = sqrt(2f0)
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
end

# using Plots
# Plots.plot(sol)

if false
        sol = merton(
                d = 1, # number of dimensions
                # one-dimensional heat equation
                tspan = (0.0f0,1f0),
                dt = 0.1f0 ,  # time step
                batch_size = 8192,
                train_steps = 8000,
                σ_sampling = 0.1f0,
                K = 20,
                )
end

########################
###### For publication #######
########################
if true
        df = DataFrame(); [df[c] = Float64[] for c in [L"d",L"T",L"N","Mean","Std. dev."]]
        dfu = DataFrame(); [dfu[c] = Float64[] for c in ["d","T","N","u"]]
        for d in [1,2,5,10],T in [1/5,1/2,1]
                N = 10
                u = []
                dt = Float32(T / N)
                for i in 1:5
                        sol = merton(
                                d = d, # number of dimensions
                                # one-dimensional heat equation
                                tspan = (0.0f0,T),
                                dt = dt,   # time step
                                batch_size = 8192,
                                train_steps = 8000,
                                σ_sampling = 5f-1,
                                K = 5,
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
