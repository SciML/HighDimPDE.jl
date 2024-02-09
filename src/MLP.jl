"""
$(SIGNATURES)

Multi level Picard algorithm.

# Arguments
* `L`: number of Picard iterations (Level),
* `M`: number of Monte Carlo integrations (at each level `l`, `M^(L-l)`integrations),
* `K`: number of Monte Carlo integrations for the non-local term
* `mc_sample::MCSampling` : sampling method for Monte Carlo integrations of the non local term.
Can be `UniformSampling(a,b)`, `NormalSampling(σ_sampling)`, or `NoSampling` (by default).
"""
struct MLP{T, MCS} <: HighDimPDEAlgorithm where {T <: Int, MCS <: MCSampling}
    M::T # nb of MC integrations
    L::T # nb of levels
    K::T # nb MC integration non local term
    mc_sample!::MCS
end

#Note: mc_sample mutates its first argument but for the user interface we hide this technicality
MLP(; M = 4, L = 4, K = 10, mc_sample = NoSampling()) = MLP(M, L, K, mc_sample)

"""
$(TYPEDSIGNATURES)

Returns a `PIDESolution` object.

# Arguments
* `multithreading` : if `true`, distributes the job over all the threads available.
* `verbose`: print information over the iterations.
"""
function DiffEqBase.solve(prob::Union{PIDEProblem, ParabolicPDEProblem},
        alg::MLP;
        multithreading = true,
        verbose = false,)

    # unbin stuff
    x = prob.x
    neumann_bc = prob.neumann_bc
    K = alg.K
    M = alg.M
    L = alg.L
    mc_sample! = alg.mc_sample!
    g= prob.g 
    f = if isa(prob, ParabolicPDEProblem)
        (y, z, v_y, v_z, ∇v_y, ∇v_z, p, t) -> prob.f(y, v_y, ∇v_y, p, t )
    else
        prob.f
    end

    # errors
    prob.x0_sample isa NoSampling ? nothing :
    error("`MLP` algorithm can only be used with `x0_sample=NoSampling()`.")

    if multithreading
        usol = _ml_picard_mlt(M,
            L,
            K,
            x,
            prob.tspan[1],
            prob.tspan[2],
            mc_sample!,
            g,
            f,
            verbose,
            prob,
            neumann_bc)
    else
        usol = _ml_picard(M,
            L,
            K,
            x,
            prob.tspan[1],
            prob.tspan[2],
            mc_sample!,
            g,
            f,
            verbose,
            prob,
            neumann_bc)
    end
    return PIDESolution(x, [prob.tspan...], nothing, [g(x), usol], nothing)
end

function _ml_picard(M, # monte carlo integration
        L, # level
        K, # non local term monte carlo
        x::xType, # initial point
        s, # time
        t::tType, # time
        mc_sample!,
        g,
        f,
        verbose,
        prob,
        neumann_bc) where {xType, tType}
    elxType = eltype(xType)
    if L == 0
        return zero(elxType)
    end

    x2 = similar(x)
    _integrate(mc_sample!) ? x3 = similar(x) : x3 = nothing
    p = prob.p

    a = zero(elxType)
    for l in 0:(min(L - 1, 1))
        verbose && println("loop l")
        b = zero(elxType)
        num = M^(L - l)
        for k in 1:num
            verbose && println("loop k")
            r = s + (t - s) * rand(tType)
            _mlt_sde_loop!(x2, x, s, r, prob, neumann_bc)
            b2 = _ml_picard(M, l, K, x2, r, t, mc_sample!, g, f, verbose, prob, neumann_bc)
            b3 = zero(elxType)
            # non local integration
            for h in 1:K
                verbose && println("loop h")
                mc_sample!(x3, x2)
                b3 += f(x2, x3, b2,
                    _ml_picard(M, l, K, x3, r, t, mc_sample!, g, f, verbose,
                        prob, neumann_bc), nothing, nothing, p, t)
            end
            b += b3 / K
        end
        a += (t - s) * b / num
    end

    for l in 2:(L - 1)
        b = zero(elxType)
        num = M^(L - l)
        for k in 1:num
            r = s + (t - s) * rand(tType)
            _mlt_sde_loop!(x2, x, s, r, prob, neumann_bc)
            b2 = _ml_picard(M, l, K, x2, r, t, mc_sample!, g, f, verbose, prob, neumann_bc)
            b4 = _ml_picard(M,
                l - 1,
                K,
                x2,
                r,
                t,
                mc_sample!,
                g,
                f,
                verbose,
                prob,
                neumann_bc)
            b3 = zero(elxType)
            # non local integration
            for h in 1:K
                mc_sample!(x3, x2)
                b3 += f(x2, x3, b2,
                    _ml_picard(M, l, K, x3, r, t, mc_sample!, g, f, verbose,
                        prob, neumann_bc), nothing, nothing, p, t) -
                      f(x2, x3, b4,
                    _ml_picard(M, l - 1, K, x3, r,
                        t, mc_sample!, g, f, verbose, prob, neumann_bc), nothing, nothing, p, t)
            end
            b += b3 / K
        end
        a += (t - s) * b / num
    end

    num = M^(L)
    a2 = zero(elxType)
    for k in 1:num
        verbose && println("loop k3")
        _mlt_sde_loop!(x2, x, s, t, prob, neumann_bc)
        a2 += g(x2)
    end
    a2 /= num

    return a + a2
end

function _ml_picard(M::Int, L::Int, K::Int, x::Nothing, s::Real, t::Real, mc_sample!, g, f,
        verbose::Bool,
        prob, neumann_bc)
    nothing
end

function _ml_picard_mlt(M, # monte carlo integration
        L, # level
        K, # non local term monte carlo
        x::xType, # initial point
        s, # time
        t, # time
        mc_sample!,
        g,
        f,
        verbose,
        prob,
        neumann_bc) where {xType}
    elxType = eltype(xType)

    # distributing tasks
    NUM_THREADS = Threads.nthreads()
    tasks = [Threads.@spawn(_ml_picard_call(M, L, K, x, s, t, mc_sample!, g, f, verbose,
        NUM_THREADS, thread_id, prob, neumann_bc)) for thread_id in 1:NUM_THREADS]

    # first level
    num = M^(L)
    x2 = similar(x)
    a2 = zero(elxType)
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

function _ml_picard_call(M, # monte carlo integration
        L, # level
        K, # non local term monte carlo
        x::xType, # initial point
        s, # time
        t::tType, # time
        mc_sample!,
        g,
        f,
        verbose,
        NUM_THREADS,
        thread_id,
        prob,
        neumann_bc) where {xType, tType}
    x2 = similar(x)
    _integrate(mc_sample!) ? x3 = similar(x) : x3 = nothing
    p = prob.p
    elxType = eltype(xType)

    a = zero(elxType)
    for l in 0:(min(L - 1, 1))
        b = zero(elxType)
        num = M^(L - l)
        loop_num = _get_loop_num(M, num, thread_id, NUM_THREADS)
        for k in 1:loop_num
            verbose && println("loop k")
            r = s + (t - s) * rand(tType)
            _mlt_sde_loop!(x2, x, s, r, prob, neumann_bc)
            b2 = _ml_picard(M, l, K, x2, r, t, mc_sample!, g, f, verbose, prob, neumann_bc)
            b3 = zero(elxType)
            for h in 1:K # non local integration
                verbose && println("loop h")
                mc_sample!(x3, x2)
                b3 += f(x2, x3, b2,
                    _ml_picard(M, l, K, x3, r, t, mc_sample!, g, f, verbose,
                        prob, neumann_bc), nothing, nothing, p, t)
            end
            b += b3 / K
        end
        a += (t - s) * b / num
    end

    for l in 2:(L - 1)
        b = zero(elxType)
        num = M^(L - l)
        loop_num = _get_loop_num(M, num, thread_id, NUM_THREADS)
        for k in 1:loop_num
            r = s + (t - s) * rand(tType)
            _mlt_sde_loop!(x2, x, s, r, prob, neumann_bc)
            b2 = _ml_picard(M, l, K, x2, r, t, mc_sample!, g, f, verbose, prob, neumann_bc)
            b4 = _ml_picard(M,
                l - 1,
                K,
                x2,
                r,
                t,
                mc_sample!,
                g,
                f,
                verbose,
                prob,
                neumann_bc)
            b3 = zero(elxType)
            # non local integration
            for h in 1:K
                mc_sample!(x3, x2)
                b3 += f(x2,
                    x3,
                    b2,
                    _ml_picard(M,
                        l,
                        K,
                        x3,
                        r,
                        t,
                        mc_sample!,
                        g,
                        f,
                        verbose,
                        prob,
                        neumann_bc),
                    nothing,
                    nothing,
                    p,
                    t) - f(x2,
                    x3,
                    b4,
                    _ml_picard(M,
                        l - 1,
                        K,
                        x3,
                        r,
                        t,
                        mc_sample!,
                        g,
                        f,
                        verbose,
                        prob,
                        neumann_bc),
                    nothing,
                    nothing,
                    p,
                    t) #TODO:hardcode, not sure about t
            end
            b += b3 / K
        end
        a += (t - s) * b / num
    end

    return a
end

#decides how many iteration given thread id and num
function _get_loop_num(M, num, thread_id, NUM_THREADS)
    if num < NUM_THREADS
        # each thread only goes once through the loop
        loop_num = thread_id > num ? 0 : 1
    else
        remainder = M % num
        if (remainder > 0) && (thread_id <= remainder) # remainder > 0 iff num == M or num == 1
            # each thread goes  num / NUM_THREADS + the remainder
            loop_num = num / NUM_THREADS + 1
        else
            loop_num = num / NUM_THREADS
        end
    end
end

function _mlt_sde_loop!(x2,
        x,
        s,
        t,
        prob,
        neumann_bc)
    #randn! allows to save one allocation
    dt = t - s
    randn!(x2)
    x2 .= x + (prob.μ(x, prob.p, t) .* dt .+ prob.σ(x, prob.p, t) .* sqrt(dt) .* x2)
    if !isnothing(neumann_bc)
        x2 .= _reflect(x, x2, neumann_bc...)
    end
end
