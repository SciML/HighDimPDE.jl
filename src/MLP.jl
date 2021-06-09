include("utils.jl")
include("reflect.jl")

"""
Multi level Picard algorithm for solving non local non linear PDES.
    
    Arguments:
    * `chain`: a Flux.jl chain with a d-dimensional input and a 1-dimensional output,
    * `strategy`: determines which training strategy will be used,
    * `init_params`: the initial parameter of the neural network,
    * `phi`: a trial solution,
    * `derivative`: method that calculates the derivative.
    
    """
struct MLP <: HighDimPDEAlgorithm
    M::Int # nb of MC integrations
    L::Int # nb of levels
    K::Int # nb MC integration non local term
end
MLP(;M=1) = MLP(M)
    
    
function DiffEqBase.__solve(
        prob::PIDEProblem,
        alg::MLP,
        mc_sample;
        dt,
        abstol=1f-6,
        verbose=false,
        )
        
        # unbin stuff
    u_domain = prob.u_domain
    X0 = prob.X0 |> _device
    ts = prob.tspan[1]:dt:prob.tspan[2]
    N = length(ts) - 1
    d  = length(X0)
    K = alg.K
    M = alg.M
    L = alg.L
    opt = alg.opt
    g, f, μ, σ, p = prob.g, prob.f, prob.μ, prob.σ, prob.p
        
    function ml_picard(
            M, # monte carlo integration
            L, # level
            # K, non local term monte carlo
            x, # initial point
            r,
            t,
            )
            
        a = 0.
        a2 = 0.
        b = 0. 
        for l in 1:min(L, 2)
            b = 0.
            num = M^(L - l) # ? why 0.5 in sebastian code?
            for k in 1:num
                r = s + (t - s) * rand()
                x2 = similar(x)
                sde_loop!(x, x2, s, r)
                b2 = ml_picard(M, l, x2, r, t)
                b3 = 0.
                    # non local integration
                for h in 1:K
                    x3 = randn(size(x))
                    b3 += f(x2, x3, b2, ml_picard(M, l, x3, r, t))
                end
                b += b3 / K
            end
            a += (t - s) * (b / num)
        end
                
        for l in 2:L
            b = 0.
            num = M^(L - l)
            for k in 1:num
                r = s + (t - s) * rand()
                x2 = similar(x)
                sde_loop!(x, x2, s, r)
                b2 = ml_picard(M, l, x2, r, t)
                b4 = ml_picard(M, l - 1, x2, r, t)
                b3 = 0.
                    # non local integration
                for h in 1:K
                    x3 = randn(size(x))
                    x32 = 
                        x34 = 
                        b3 += f(x2, x32, b2, ml_picard(M, l, x32, r, t)) - f(x2, x34, b4, ml_picard(M, l - 1, x34, r, t))
                end
                b += b3 / K
            end
            a += (t - s) * (b / num)
                
            num = M^(L)
            for k in 1:num
                x2 = similar(x)
                sde_loop!(x, x2, s, t)
                g(x2)
                a2 += tmp
            end
            return a + a2
        end
    end
                
    function sde_loop!(y0, y1, s, t)
        rgen!(y1)
        dt = t - s
        y1 .= y0 .- ( μ(y0, p, t) .* dt .+ σ(y0, p, t) .* sqrt(dt) .* y1)
        if !isnothing(u_domain)
            y1 .= _reflect_GPU2(y0, y1, u_domain[1], u_domain[2], d)
        end
    end
                
                # sol = DiffEqBase.build_solution(prob,alg,ts,usol)
                # save_everystep ? iters : u0(X0)[1]
end    