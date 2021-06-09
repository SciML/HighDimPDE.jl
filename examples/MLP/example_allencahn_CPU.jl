cd(@__DIR__)
using Revise
using HighDimPDE
using Random
# Random.seed!(100)

## Basic example
d = 3 # number of dimensions
# one-dimensional heat equation
tspan = (0.0f0,1f0)
dt = 0.1f0  # time step
batch_size = 500
train_steps = 500
σ_sampling = 1f0
K = 5

X0 = fill(0.5f0,d)  # initial point

g(X) = exp(-0.25f0 * sum(X.^2))   # initial condition
a(u) = u - u^3
f(y,z,v_y,v_z,∇v_y,∇v_z,p,t) = a.(v_y) .- a.(v_z) .* Float32(π^(d/2) * σ_sampling^d) # nonlocal nonlinear part of the
μ(X,p,t) = 0.0f0 # advection coefficients
σ(X,p,t) = sqrt(2f0) # diffusion coefficients
# mc_sample(x) = x + randn(d,batch_size) * σ_sampling / sqrt(2f0) #montecarlo samples

# defining the problem
prob = PIDEProblem(g, f, μ, σ, X0, tspan, 
                    # u_domain=[-1f0,1f0]
                     )

# using the Deep Splitting algorithm
alg = MLP(M=4, K=5, L = 3 )


# solving
sol = solve(prob, alg,
            dt=dt,
            verbose = false)
println("u1 = ", sol)

# sol


