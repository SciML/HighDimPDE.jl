"""
Multi level Picard algorithm for solving non local non linear PDES.
    
    Arguments:
    * `L`: number of Picard iterations (Level),
    * `M`: number of Monte Carlo integrations (at each level `l`, `M^(l)` 
    integrations),
    * `K`: number of Monte Carlo integrations for the non local term    
    * `mc_sample::MCSampling` : sampling method for Monte Carlo integrations of the non local term.
    Can be `UniformSampling(a,b)`, `NormalSampling(σ_sampling)`, or `NoSampling` (by default).
    """
struct MLP <: HighDimPDEAlgorithm
    M::Int # nb of MC integrations
    L::Int # nb of levels
    K::Int # nb MC integration non local term
    mc_sample::MCSampling
end
MLP(;M=4,L=4,K=10,mc_sample=NoSampling()) = MLP(M,L,K,mc_sample)
    
    
function DiffEqBase.__solve(
        prob::PIDEProblem,
        alg::MLP;
        multithreading=true,
        verbose=false,
        )

    # unbin stuff
    u_domain = prob.u_domain
    x = prob.X0
    K = alg.K
    M = alg.M
    L = alg.L
    mc_sample = alg.mc_sample
    g, f, μ, σ, p = prob.g, prob.f, prob.μ, prob.σ, prob.p

    function sde_loop(y0, s, t)
        dt = t - s
        # @show y1
        y1 = y0 .- ( μ(y0, p, t) .* dt .+ σ(y0, p, t) .* sqrt(dt) .* randn(size(y0)))
        if !isnothing(u_domain)
            y1 = _reflect(y0, y1, u_domain[1], u_domain[2])
        end
        return y1
    end
    
    if multithreading
        return _ml_picard_mlt(M, L, K, x, prob.tspan[1], prob.tspan[2], sde_loop, mc_sample, g, f, verbose)
    else
        return _ml_picard(M, L, K, x, prob.tspan[1], prob.tspan[2], sde_loop, mc_sample, g, f, verbose)
    end 
    # sol = DiffEqBase.build_solution(prob,alg,ts,usol)
    # save_everystep ? iters : u0(X0)[1]

end

function _ml_picard(
        M::Int, # monte carlo integration
        L::Int, # level
        K::Int, # non local term monte carlo
        x, # initial point
        s::Real, # time
        t::Real, # time
        sde_loop,
        mc_sample, 
        g, 
        f,
        verbose::Bool
        )
    r = 0.
    a = 0.
    a2 = 0.
    b = 0. 
    for l in 0:(min(L, 2) - 1)
        verbose && println("loop l")
        b = 0.
        num = M^(L - l) # ? why 0.5 in sebastian code?
        for k in 0:num
            verbose && println("loop k")
            r = s + (t - s) * rand()
            x2 = sde_loop(x, s, r)
            b2 = _ml_picard(M, l, K, x2, r, t, sde_loop, mc_sample, g, f, verbose)
            b3 = 0.
                # non local integration
            for h in 0:(K-1)
                verbose && println("loop h")
                x3 = mc_sample(x)
                b3 += f(x2, x3, b2, _ml_picard(M, l, K, x3, r, t, sde_loop, mc_sample, g, f, verbose), 0., 0., t)[] #TODO:hardcode, not sure about t
            end
            b += b3 / K
        end
        a += (t - s) * (b / num)
    end
            
    for l in 2:(L-1)
        b = 0.
        num = M^(L - l)
        for k in 1:num
            r = s + (t - s) * rand()
            x2 = sde_loop(x, s, r)
            b2 = _ml_picard(M, l, K, x2, r, t, sde_loop, mc_sample, g, f, verbose)
            b4 = _ml_picard(M, l - 1, K, x2, r, t, sde_loop, mc_sample, g, f, verbose)
            b3 = 0.
                # non local integration
            for h in 0:(K-1)
                x3 = mc_sample(x)
                x32 = x3
                x34 = x3
                b3 += f(x2, x32, b2, _ml_picard(M, l, K, x32, r, t, sde_loop, mc_sample, g, f, verbose), 0., 0., t)[] - 
                    f(x2, x34, b4, _ml_picard(M, l - 1, K, x34, r, t, sde_loop, mc_sample, g, f, verbose),0., 0., t)[] #TODO:hardcode, not sure about t
            end
            b += b3 / K
        end
        a += (t - s) * (b / num)
    end

    num = M^(L)
    for k in 0:(num-1)
        verbose && println("loop k3")
        x2 = sde_loop(x, s, t)
        a2 += g(x2)[]
    end
    a2 /= num

    return a + a2
end

_ml_picard(M::Int, L::Int, K::Int, x::Nothing, s::Real, t::Real, sde_loop, mc_sample, g, f, verbose::Bool) = nothing

function _ml_picard_mlt(
    M::Int, # monte carlo integration
    L::Int, # level
    K::Int, # non local term monte carlo
    x, # initial point
    s::Real, # time
    t::Real, # time
    sde_loop, 
    mc_sample,
    g, 
    f,
    verbose::Bool
    )
    a = 0.
    a2 = 0.
    for l in 0:(min(L, 2) - 1)
        verbose && println("loop l")
        b = Threads.Atomic{Float64}(0.) 
        num = M^(L - l) # ? why 0.5 in sebastian code?
        @Threads.threads for k in 0:num
            verbose && println("loop k")
            r = s + (t - s) * rand()
            x2 = sde_loop(x, s, r)
            b2 = _ml_picard(M, l, K, x2, r, t, sde_loop, mc_sample, g, f, verbose)
            b3 = 0.
                # non local integration
            for h in 0:(K-1)
                verbose && println("loop h")
                x3 = mc_sample(x)
                b3 += f(x2, x3, b2, _ml_picard(M, l, K, x3, r, t, sde_loop, mc_sample, g, f, verbose), 0., 0., t)[] #TODO:hardcode, not sure about t
            end
        Threads.atomic_add!(b, b3 / K)
        end
        a += (t - s) * (b[] / num)
    end
            
    for l in 2:(L-1)
        b = Threads.Atomic{Float64}(0.) 
        num = M^(L - l)
        @Threads.threads for k in 1:num
            r = s + (t - s) * rand()
            x2 = sde_loop(x, s, r)
            b2 = _ml_picard(M, l, K, x2, r, t, sde_loop, mc_sample, g, f, verbose)
            b4 = _ml_picard(M, l - 1, K, x2, r, t, sde_loop, mc_sample, g, f, verbose)
            b3 = 0.
                # non local integration
            for h in 0:(K-1)
                x3 = mc_sample(x)
                x32 = x3
                x34 = x3
                b3 += f(x2, x32, b2, _ml_picard(M, l, K, x32, r, t, sde_loop, mc_sample, g, f, verbose), 0., 0., t)[] - f(x2, x34, b4, _ml_picard(M, l - 1, K, x34, r, t, sde_loop, mc_sample, g, f, verbose),0., 0., t)[] #TODO:hardcode, not sure about t
            end
        Threads.atomic_add!(b, b3 / K)
        end
        a += (t - s) * (b[] / num)
    end

    num = M^(L)
    for k in 0:(num-1)
        verbose && println("loop k3")
        x2 = sde_loop(x, s, t)
        a2 += g(x2)[]
    end
    a2 /= num

    return a + a2
end