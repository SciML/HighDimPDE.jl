struct NNStopping{M, O} <: HighDimPDEAlgorithm
    m::M
    opt::O

    function NNStopping(m, opt = Flux.Optimise.Adam(0.1))
        new{typeof(m), typeof(opt)}(m, opt)
    end
end

function DiffEqBase.solve(prob::PIDEProblem,
    pdealg::NNStopping,
    sdealg;
    verbose = false,
    maxiters = 300,
    trajectories = 100,
    dt = eltype(prob.tspan)(0),
    ensemblealg = EnsembleThreads(),
    kwargs...)

    tspan = prob.tspan
    sigma = prob.g
    μ = prob.f
    g = prob.kwargs.data.g
    u0 = prob.u0
    ts = tspan[1]:dt:tspan[2]
    T = tspan[2]

    m  = alg.m
    opt    = alg.opt


    prob = SDEProblem(μ,sigma,u0,tspan)
    ensembleprob = EnsembleProblem(prob)
    sim = solve(ensembleprob, sdealg, ensemblealg, dt=dt,trajectories=trajectories,adaptive=false)
    payoff = []
    times = []
    iter = 0
    # for u in sim.u
    # un = []
    # function Un(n , X )
    #     if size(un)[1] >= n
    #         return un[n]
    #     else
    #         if(n == 1)
    #               ans =  first(m(X[1])[1])
    #               un = [ans]
    #               return ans
    #           else
    #               ans =  max(first(m(X[n])[n]) , n + 1 - size(ts)[1])*(1 - sum(Un(i , X ) for i in 1:n-1))
    #               un = vcat( un , ans)
    #               return ans
    #           end
    #       end
    # end

    function Un!(un_cache, n, X)
        !isnan(un_cache[n]) && return un_cache[n]
        if n == 1
            un_cache[n] = first(m(X[1])[1])
        else
            un_cache[n] = max(first(m(X[n])[n]) , n + 1 - length(ts))*(1 - sum(un_cache[1:n-1]))
        end
        return un_cache[n]
    end


    function loss()
        reward = 0.00
        map(sim.u) do X 
            un_cache = fill(NaN, length(ts))
            reward += mapreduce((x, t) -> Un!(un_cache, i, X)*g(t,X), +, X, ts)
        end

        return 10000 - reward
    end

    dataset = Iterators.repeated(() , maxiters)

    callback = function ()
        l = loss()
        println("Current loss is: $l")
    end

    Flux.train!(loss, Flux.params(m), dataset, opt; cb = callback)

    Usum = 0
    ti = 0
    Xt = sim.u[1].u
    un_cache = fill(NaN, N)
    for i in 1:N
          Un = Un!(un_cache, i , Xt)
          Usum = Usum + Un
          if Usum >= 1 - Un
            ti = i
            break
          end
    end
    for u in sim.u
        X = u.u
        price = g(ts[ti] , X[ti])
        payoff = vcat(payoff , price)
        times = vcat(times, ti)
        iter = iter + 1
        # println("SUM : $sump")
        # println("TIME : $ti")
    end
    sum(payoff)/size(payoff)[1]

    


end