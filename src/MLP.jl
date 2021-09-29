"""
    MLP( M=4, L=4, K=10, mc_sample = NoSampling())

Multi level Picard algorithm for solving non local non linear PDES.
    
Arguments:
* `L`: number of Picard iterations (Level),
* `M`: number of Monte Carlo integrations (at each level `l`, `M^(L-l)`integrations),
* `K`: number of Monte Carlo integrations for the non local term    
* `mc_sample::MCSampling` : sampling method for Monte Carlo integrations of the non local term. Can be `UniformSampling(a,b)`, `NormalSampling(σ_sampling)`, or `NoSampling` (by default).
"""
struct MLP{T, MCS} <: HighDimPDEAlgorithm where {T <: Int, U}
    M::T # nb of MC integrations
    L::T # nb of levels
    K::T # nb MC integration non local term
    mc_sample!::MCS
end
MLP(; M=4, L=4, K=10, mc_sample = NoSampling()) = MLP(M ,L, K, mc_sample) #Note: mc_sample mutates its first argument but for the user interface we hide this technicality
    
    
# function DiffEqBase.__solve(
function solve(
        prob::PIDEProblem,
        alg::MLP;
        multithreading=true,
        verbose=false,
        )

    # unbin stuff
    x = prob.x
    neumann_bc = prob.neumann_bc
    K = alg.K
    M = alg.M
    L = alg.L
    mc_sample! = alg.mc_sample!
    g, f = prob.g, prob.f

    # errors
    !isnothing(prob.u_domain) ? error("`MLP` algorithm cannot be solved on a domain, i.e with argument `u_domain`.") : nothing
    isnothing(x) ? error("`MLP` algorithm needs a grid 'x'") : nothing
    
    if multithreading
        usol = _ml_picard_mlt(M, L, K, x, prob.tspan[1], prob.tspan[2], mc_sample!, g, f, verbose, prob, neumann_bc)
    else
        usol = _ml_picard(M, L, K, x, prob.tspan[1], prob.tspan[2], mc_sample!, g, f, verbose, prob, neumann_bc)
    end 
    return x, prob.tspan, [g(x),usol]
    # sol = DiffEqBase.build_solution(prob,alg,ts,usol)
    # save_everystep ? iters : u0(X0)[1]

end

function _ml_picard(
        M::F, # monte carlo integration
        L::F, # level
        K::F, # non local term monte carlo
        x::xType, # initial point
        s::tType, # time
        t::tType, # time
        mc_sample!::MCSampling, 
        g::Function, 
        f::Function,
        verbose::Bool,
        prob::PIDEProblem,
        ) where {F <: Int, tType <: AbstractFloat}
    
    x2 = similar(x)
    x3 = similar(x)
    x32 = similar(x)
    x34 = similar(x)
    p = prob.p

    for l in 0:(min(L- 1, 1))
        verbose && println("loop l")
        b = zero(xType)
        num = M^(L - l) # ? why 0.5 in sebastian code?
        for k in 1:num
            verbose && println("loop k")
            r = s + (t - s) * rand(tType)
            _mlt_sde_loop!(x2, x, s, r, prob, neumann_bc)
            b2 = _ml_picard(M, l, K, x2, r, t, mc_sample!, g, f, verbose, prob, neumann_bc)
            b3 = zero(xType)
            # non local integration
            for h in 1:K
                verbose && println("loop h")
                mc_sample!(x3, x2)
                b3 += f(x2, x3, b2, _ml_picard(M, l, K, x3, r, t, mc_sample!, g, f, verbose, prob, neumann_bc), 0., 0., p, t) #TODO:hardcode, not sure about t
            end
            b += b3 / K
        end
        a += (t - s) * b / num
    end
            
    for l in 2:(L-1)
        b = zero(xType)
        num = M^(L - l)
        for k in 1:num
            r = s + (t - s) * rand(tType)
            _mlt_sde_loop!(x2, x, s, r, prob, neumann_bc)
            b2 = _ml_picard(M, l, K, x2, r, t, mc_sample!, g, f, verbose, prob, neumann_bc)
            b4 = _ml_picard(M, l - 1, K, x2, r, t, mc_sample!, g, f, verbose, prob, neumann_bc)
            b3 = zero(xType)
                # non local integration
            for h in 1:K
                mc_sample!(x3, x2)
                x32 .= x3
                x34 .= x3
                b3 += f(x2, x32, b2, _ml_picard(M, l, K, x32, r, t, mc_sample!, g, f, verbose, prob, neumann_bc), 0., 0., p, t) - f(x2, x34, b4, _ml_picard(M, l - 1, K, x34, r, t, mc_sample!, g, f, verbose, prob, neumann_bc),0., 0., p, t) #TODO:hardcode, not sure about t
            end
            b += b3 / K
        end
        a += (t - s) * b / num
    end

    num = M^(L)
    for k in 1:num
        verbose && println("loop k3")
        _mlt_sde_loop!(x2, x, s, t, prob, neumann_bc)
        a2 += g(x2)
    end
    a2 /= num

    return a + a2
end

_ml_picard(M::Int, L::Int, K::Int, x::Nothing, s::Real, t::Real, mc_sample!, g, f, verbose::Bool, prob, neumann_bc) = nothing

function _ml_picard_mlt(
    M::F, # monte carlo integration
    L::F, # level
    K::F, # non local term monte carlo
    x::xType, # initial point
    s::tType, # time
    t::tType, # time
    mc_sample!::MCSampling, 
    g::Function, 
    f::Function,
    verbose::Bool,
    prob::PIDEProblem,
    neumann_bc::Union{Nothing,Array{xType}}
    ) where {F <: Int, tType <: AbstractFloat}
    
    a2 = zero(xType)
    
    # distributing tasks
    NUM_THREADS = Threads.nthreads()
    tasks = [Threads.@spawn(_ml_picard_call(M, L, K, x, s, t, mc_sample!, g, f, verbose, NUM_THREADS, thread_id, prob, neumann_bc)) for thread_id in 1:NUM_THREADS]
    
    # ? not sure whether we should have Threads.nthreads-1 to allow next lines to run on the last threads available
    # this pretty much depends on the time required to perform following for-loop

    # first level
    num = M^(L)
    x2 = similar(x)
    for k in 1:num
        verbose && println("loop k3")
        _mlt_sde_loop!(x2, x, s, t, prob, neumann_bc)
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
    mc_sample!::MCSampling{xType}, 
    g::Function, 
    f::Function,
    verbose::Bool,
    NUM_THREADS::Int64,
    thread_id::Int64,
    prob::PIDEProblem,
    neumann_bc::Union{Nothing,Array{xType}}
    ) where {F <: Int, xType <: AbstractFloat, tType <: AbstractFloat}

    x2 = similar(x)
    x3 = similar(x)
    x32 = similar(x)
    x34 = similar(x)
    p = prob.p

    for l in 0:(min(L - 1, 1))
        b = zero(xType)
        num = M^(L - l)
        loop_num = _get_loop_num(M, num, thread_id, NUM_THREADS)
        for k in 1:loop_num
            verbose && println("loop k")
            r = s + (t - s) * rand(tType)
            _mlt_sde_loop!(x2, x, s, r, prob, neumann_bc)
            b2 = _ml_picard(M, l, K, x2, r, t, mc_sample!, g, f, verbose, prob, neumann_bc)
            b3 = zero(xType)
            for h in 1:K # non local integration
                verbose && println("loop h")
                mc_sample!(x3, x2)
                b3 += f(x2, x3, b2, _ml_picard(M, l, K, x3, r, t, mc_sample!, g, f, verbose, prob, neumann_bc), 0., 0., p, t) #TODO:hardcode, not sure about t
            end
        b += b3 / K
        end
        a += (t - s) * b / num
    end
            
    for l in 2:(L-1)
        b = zero(xType)
        num = M^(L - l)
        loop_num = _get_loop_num(M, num, thread_id, NUM_THREADS)
        for k in 1:loop_num
            r = s + (t - s) * rand(tType)
            _mlt_sde_loop!(x2, x, s, r, prob, neumann_bc)
            b2 = _ml_picard(M, l, K, x2, r, t, mc_sample!, g, f, verbose, prob, neumann_bc)
            b4 = _ml_picard(M, l - 1, K, x2, r, t, mc_sample!, g, f, verbose, prob, neumann_bc)
            b3 = zero(xType)
                # non local integration
            for h in 1:K
                mc_sample!(x3, x2)
                x32 .= x3
                x34 .= x3
                b3 += f(x2, x32, b2, _ml_picard(M, l, K, x32, r, t, mc_sample!, g, f, verbose, prob, neumann_bc), 0., 0., p, t) - f(x2, x34, b4, _ml_picard(M, l - 1, K, x34, r, t, mc_sample!, g, f, verbose, prob, neumann_bc),0., 0., p, t) #TODO:hardcode, not sure about t
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
        if (remainder > 0) && (thread_id <= remainder) # remainder > 0 iff num == M or num == 1
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
                        neumann_bc::Union{Nothing,Array{xType}}) where {xType <: AbstractFloat, tType <: AbstractFloat}
    # @show x2
    #randn! allows to save one allocation
    dt = t - s
    randn!(x2)
    x2 .= x - (prob.μ(x, prob.p, t) .* dt .+ prob.σ(x, prob.p, t) .* sqrt(dt) .* x2)
    if !isnothing(neumann_bc)
        x2 .= _reflect(x, x2, neumann_bc[1], neumann_bc[2])
    end
end