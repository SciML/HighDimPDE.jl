"""
    MLP
Multi level Picard algorithm for solving non local non linear PDES.
    
    Arguments:
    * `L`: number of Picard iterations (Level),
    * `M`: number of Monte Carlo integrations (at each level `l`, `M^(l)` 
    integrations),
    * `K`: number of Monte Carlo integrations for the non local term    
    * `mc_sample::MCSampling` : sampling method for Monte Carlo integrations of the non local term.
    Can be `UniformSampling(a,b)`, `NormalSampling(σ_sampling)`, or `NoSampling` (by default).
    """
struct MLP{T, U} <: HighDimPDEAlgorithm where {T <: Int, U}
    M::T # nb of MC integrations
    L::T # nb of levels
    K::T # nb MC integration non local term
    mc_sample::MCSampling{U}
end
MLP(;M=4,L=4,K=10,mc_sample=NoSampling()) = MLP(M,L,K,mc_sample)
    
    
# function DiffEqBase.__solve(
function solve(
        prob::PIDEProblem,
        alg::MLP;
        multithreading=true,
        verbose=false,
        neumann = nothing
        )

    # unbin stuff
    x = prob.x
    K = alg.K
    M = alg.M
    L = alg.L
    mc_sample = alg.mc_sample
    g, f = prob.g, prob.f

    isnothing(x) || !isnothing(prob.u_domain) ? error("MLP scheme needs a grid 'x', and cannot be solved on a domain") : nothing
    
    if multithreading
        usol = _ml_picard_mlt(M, L, K, x, prob.tspan[1], prob.tspan[2], mc_sample, g, f, verbose, prob, neumann)
    else
        usol = _ml_picard(M, L, K, x, prob.tspan[1], prob.tspan[2], mc_sample, g, f, verbose, prob, neumann)
    end 
    return x, prob.tspan, [g(x),usol]
    # sol = DiffEqBase.build_solution(prob,alg,ts,usol)
    # save_everystep ? iters : u0(X0)[1]

end

function _ml_picard(
        M::F, # monte carlo integration
        L::F, # level
        K::F, # non local term monte carlo
        x::Vector{xType}, # initial point
        s::tType, # time
        t::tType, # time
        mc_sample::MCSampling{xType}, 
        g::Function, 
        f::Function,
        verbose::Bool,
        prob::PIDEProblem,
        neumann::Union{Nothing,Array{xType}}
        ) where {F <: Int, xType <: AbstractFloat, tType <: AbstractFloat}
    r = zeros(tType)
    a = zero(xType)
    a2 =  zero(xType)
    b = zero(xType)
    x2 = similar(x)

    for l in 0:(min(L- 1, 1))
        verbose && println("loop l")
        b = zero(xType)
        num = M^(L - l) # ? why 0.5 in sebastian code?
        for k in 1:num
            verbose && println("loop k")
            r = s + (t - s) * rand()
            _mlt_sde_loop!(x2, x, s, r, prob, neumann)
            b2 = _ml_picard(M, l, K, x2, r, t, mc_sample, g, f, verbose, prob, neumann)
            b3 = zero(xType)
                # non local integration
            for h in 1:K
                verbose && println("loop h")
                x3 = mc_sample(x2)
                b3 += f(x2, x3, b2, _ml_picard(M, l, K, x3, r, t, mc_sample, g, f, verbose, prob, neumann), 0., 0., t) #TODO:hardcode, not sure about t
            end
            b += b3 / K
        end
        a += (t - s) * b / num
    end
            
    for l in 2:(L-1)
        b = zero(xType)
        num = M^(L - l)
        for k in 1:num
            r = s + (t - s) * rand()
            _mlt_sde_loop!(x2, x, s, r, prob, neumann)
            b2 = _ml_picard(M, l, K, x2, r, t, mc_sample, g, f, verbose, prob, neumann)
            b4 = _ml_picard(M, l - 1, K, x2, r, t, mc_sample, g, f, verbose, prob, neumann)
            b3 = zero(xType)
                # non local integration
            for h in 1:K
                x3 = mc_sample(x2)
                x32 = x3
                x34 = x3
                b3 += f(x2, x32, b2, _ml_picard(M, l, K, x32, r, t, mc_sample, g, f, verbose, prob, neumann), 0., 0., t) - f(x2, x34, b4, _ml_picard(M, l - 1, K, x34, r, t, mc_sample, g, f, verbose, prob, neumann),0., 0., t) #TODO:hardcode, not sure about t
            end
            b += b3 / K
        end
        a += (t - s) * b / num
    end

    num = M^(L)
    for k in 1:num
        verbose && println("loop k3")
        _mlt_sde_loop!(x2, x, s, t, prob, neumann)
        a2 += g(x2)
    end
    a2 /= num

    return a + a2
end

_ml_picard(M::Int, L::Int, K::Int, x::Nothing, s::Real, t::Real, mc_sample, g, f, verbose::Bool, prob, neumann) = nothing

function _ml_picard_mlt(
    M::F, # monte carlo integration
    L::F, # level
    K::F, # non local term monte carlo
    x::Vector{xType}, # initial point
    s::tType, # time
    t::tType, # time
    mc_sample::MCSampling{xType}, 
    g::Function, 
    f::Function,
    verbose::Bool,
    prob::PIDEProblem,
    neumann::Union{Nothing,Array{xType}}
    ) where {F <: Int, xType <: AbstractFloat, tType <: AbstractFloat}
    
    d = length(x)
    a2 = zero(xType)
    
    # distributing tasks
    NUM_THREADS = Threads.nthreads()
    tasks = [Threads.@spawn(_ml_picard_call(M, L, K, x, s, t, mc_sample, g, f, verbose, NUM_THREADS, thread_id, prob, neumann)) for thread_id in 1:NUM_THREADS]
    
    # ? not sure whether we should have Threads.nthreads-1 to allow next lines to run on the last threads available
    # this pretty much depends on the time required to perform following for-loop

    # first level
    num = M^(L)
    x2 = similar(x)
    for k in 1:num
        verbose && println("loop k3")
        _mlt_sde_loop!(x2, x, s, t, prob, neumann)
        a2 += g(x2)
    end
    a2 /= num
    
    # fetching tasks
    a = sum([fetch(t) for t in tasks])

    return a + a2
end


function _ml_picard_call(
    M::F, # monte carlo integration
    L::F, # level
    K::F, # non local term monte carlo
    x::Vector{xType}, # initial point
    s::tType, # time
    t::tType, # time
    mc_sample::MCSampling{xType}, 
    g::Function, 
    f::Function,
    verbose::Bool,
    NUM_THREADS::Int64,
    thread_id::Int64,
    prob::PIDEProblem,
    neumann::Union{Nothing,Array{xType}}
    ) where {F <: Int, xType <: AbstractFloat, tType <: AbstractFloat}

    a = zero(xType)
    x2 = similar(x)

    for l in 0:(min(L - 1, 1))
        b = zero(xType)
        num = M^(L - l)
        loop_num = _get_loop_num(M, num, thread_id, NUM_THREADS)
        for k in 1:loop_num
            verbose && println("loop k")
            r = s + (t - s) * rand()
            _mlt_sde_loop!(x2, x, s, r, prob, neumann)
            b2 = _ml_picard(M, l, K, x2, r, t, mc_sample, g, f, verbose, prob, neumann)
            b3 = zero(xType)
                # non local integration
            for h in 1:K
                verbose && println("loop h")
                x3 = mc_sample(x2)
                b3 += f(x2, x3, b2, _ml_picard(M, l, K, x3, r, t, mc_sample, g, f, verbose, prob, neumann), 0., 0., t) #TODO:hardcode, not sure about t
            end
        b += b3 / K
        end
        a += (t - s) * b / num
    end
            
    for l in 2:(L-1)
        b = zero(xType)
        num = (M^(L - l))
        loop_num = _get_loop_num(M, num, thread_id, NUM_THREADS)
        for k in 1:loop_num
            r = s + (t - s) * rand()
            _mlt_sde_loop!(x2, x, s, r, prob, neumann)
            b2 = _ml_picard(M, l, K, x2, r, t, mc_sample, g, f, verbose, prob, neumann)
            b4 = _ml_picard(M, l - 1, K, x2, r, t, mc_sample, g, f, verbose, prob, neumann)
            b3 = zero(xType)
                # non local integration
            for h in 1:K
                x3 = mc_sample(x2)
                x32 = x3
                x34 = x3
                b3 += f(x2, x32, b2, _ml_picard(M, l, K, x32, r, t, mc_sample, g, f, verbose, prob, neumann), 0., 0., t) - f(x2, x34, b4, _ml_picard(M, l - 1, K, x34, r, t, mc_sample, g, f, verbose, prob, neumann),0., 0., t) #TODO:hardcode, not sure about t
            end
        b += b3 / K
        end
        a += (t - s) * b / num
    end

    return a

end

function _get_loop_num(M, num, thread_id, NUM_THREADS) #decides how many iteration given thread id and num
    if num < NUM_THREADS
        # each thread only goes once through the loop
        loop_num = thread_id > num ? 0 : 1
    else
        remainder =  M % num
        if (remainder > 0) && (thread_id <= remainder) 
            # each thread goes  num / NUM_THREADS + the remainder 
            loop_num = num / NUM_THREADS + 1
        else
            loop_num = num / NUM_THREADS
        end
    end
end

function _mlt_sde_loop!(x2::Vector{xType}, 
                        x::Vector{xType}, 
                        s::tType, 
                        t::tType, 
                        prob::PIDEProblem, 
                        neumann::Union{Nothing,Array{xType}}) where {xType <: AbstractFloat, tType <: AbstractFloat}
    dt = t - s
    # @show x2
    #randn! allows to save one allocation
    x2 .= x - (prob.μ(x, prob.p, t) .* dt .+ prob.σ(x, prob.p, t) .* sqrt(dt) .* randn!(x2))
    if !isnothing(neumann)
        x2 .= _reflect(x, x2, neumann[:,1], neumann[:,2])
    end
    return x2
end