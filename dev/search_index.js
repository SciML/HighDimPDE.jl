var documenterSearchIndex = {"docs":
[{"location":"getting_started.html#Getting-started","page":"Getting started","title":"Getting started","text":"","category":"section"},{"location":"getting_started.html#General-workflow","page":"Getting started","title":"General workflow","text":"","category":"section"},{"location":"getting_started.html","page":"Getting started","title":"Getting started","text":"The general workflow for using HighDimPDE.jl is as follows:","category":"page"},{"location":"getting_started.html","page":"Getting started","title":"Getting started","text":"Define a Partial Integro Differential Equation problem","category":"page"},{"location":"getting_started.html","page":"Getting started","title":"Getting started","text":"Modules = [HighDimPDE]\nPages   = [\"HighDimPDE.jl\"]","category":"page"},{"location":"getting_started.html#HighDimPDE.HighDimPDE","page":"Getting started","title":"HighDimPDE.HighDimPDE","text":"(Image: ) (Image: ) (Image: Build Status)\n\nHighDimPDE.jl\n\nHighDimPDE.jl is a Julia package to solve Highly Dimensional non-local, non-linear PDEs of the form\n\n$\n\n\\begin{aligned}     (\\partialt u)(t,x) &= \\int{\\Omega} f\\big(t,x,{\\bf x}, u(t,x),u(t,{\\bf x}), ( \\nablax u )(t,x ),( \\nablax u )(t,{\\bf x} ) \\big) d{\\bf x} \\\n    & \\quad + \\big\\langle \\mu(t,x), ( \\nablax u )( t,x ) \\big\\rangle + \\tfrac{1}{2} \\text{Trace} \\big(\\sigma(t,x) [ \\sigma(t,x) ]^* ( \\text{Hess}x u)(t, x ) \\big). \\end{aligned} $\n\nwhere u colon 0T times Omega to mathbbR Omega subseteq mathbbR^d is subject to initial and boundary conditions, and where d is large.\n\nDocumentation\n\n(Image: ) (Image: )\n\nInstallation\n\nOpen Julia and type the following\n\nusing Pkg;\nPkg.add(\"HighDimPDE.jl\")\n\nThis will download the latest version from the git repo and download all dependencies.\n\nGetting started\n\nSee documentation and test folders.\n\nReference\n\nBoussange, V., Becker, S., Jentzen, A., Kuckuck, B., Pellissier, L., Deep learning approximations for non-local nonlinear PDEs with Neumann boundary conditions. arXiv (2022)\n\n<!– - Becker, S., Braunwarth, R., Hutzenthaler, M., Jentzen, A., von Wurstemberger, P., Numerical simulations for full history recursive multilevel Picard approximations for systems of high-dimensional partial differential equations. arXiv (2020)\n\nBeck, C., Becker, S., Cheridito, P., Jentzen, A., Neufeld, A., Deep splitting method for parabolic PDEs. arXiv (2019)\nHan, J., Jentzen, A., E, W., Solving high-dimensional partial differential equations using deep learning. arXiv (2018) –>\n\n<!– ## Acknowledgements HighDimPDE.jl is inspired from Sebastian Becker's scripts in Python, TensorFlow and C++. Pr. Arnulf Jentzen largely contributed to the theoretical developments of the solver algorithms implemented. –>\n\n\n\n\n\n","category":"module"},{"location":"getting_started.html#HighDimPDE.PIDEProblem","page":"Getting started","title":"HighDimPDE.PIDEProblem","text":"PIDEProblem(g, f, μ, σ, x, tspan, p = nothing, x0_sample=nothing, neumann_bc=nothing)\n\nDefines a Partial Integro Differential Problem, of the form \n\nbeginaligned\n    fracdudt = tfrac12 textTr(sigma sigma^T) Delta u(x t) + mu nabla u(x t)  \n    quad + int f(x y u(x t) u(y t) ( nabla_x u )(x t) ( nabla_x u )(y t) p t) dy\nendaligned\n\nwith u(x0) = g(x).\n\nArguments\n\ng : initial condition, of the form g(x, p, t).\nf : nonlinear function, of the form f(x, y, u(x, t), u(y, t), ∇u(x, t), ∇u(x, t), p, t).\nμ : drift function, of the form μ(x, p, t).\nσ : diffusion function σ(x, p, t).\nx: point where u(x,t) is approximated. Is required even in the case where x0_sample is provided.\ntspan: timespan of the problem.\np: the parameter vector.\nx0_sample : sampling method for x0. Can be UniformSampling(a,b), NormalSampling(σ_sampling, shifted), or NoSampling (by default). If NoSampling, only solution at the single point x is evaluated.\nneumann_bc: if provided, neumann boundary conditions on the hypercube neumann_bc[1] × neumann_bc[2]. \n\n\n\n\n\n","category":"type"},{"location":"getting_started.html","page":"Getting started","title":"Getting started","text":"Select a solver algorithm\nSolve the problem","category":"page"},{"location":"getting_started.html#Examples","page":"Getting started","title":"Examples","text":"","category":"section"},{"location":"getting_started.html","page":"Getting started","title":"Getting started","text":"Let's illustrate that with some examples.","category":"page"},{"location":"getting_started.html#MLP","page":"Getting started","title":"MLP","text":"","category":"section"},{"location":"getting_started.html#Local-PDE","page":"Getting started","title":"Local PDE","text":"","category":"section"},{"location":"getting_started.html","page":"Getting started","title":"Getting started","text":"Let's solve the Fisher KPP PDE in dimension 10 with MLP.","category":"page"},{"location":"getting_started.html","page":"Getting started","title":"Getting started","text":"partial_t u = u (1 - u) + frac12sigma^2Delta_xu tag1","category":"page"},{"location":"getting_started.html","page":"Getting started","title":"Getting started","text":"using HighDimPDE\n\n## Definition of the problem\nd = 10 # dimension of the problem\ntspan = (0.0,0.5) # time horizon\nx0 = fill(0.,d)  # initial point\ng(x) = exp(- sum(x.^2) ) # initial condition\nμ(x, p, t) = 0.0 # advection coefficients\nσ(x, p, t) = 0.1 # diffusion coefficients\nf(x, y, v_x, v_y, ∇v_x, ∇v_y, p, t) = max(0.0, v_x) * (1 -  max(0.0, v_x)) # nonlocal nonlinear part of the\nprob = PIDEProblem(g, f, μ, σ, x0, tspan) # defining the problem\n\n## Definition of the algorithm\nalg = MLP() # defining the algorithm. We use the Multi Level Picard algorithm\n\n## Solving with multiple threads \nsol = solve(prob, alg, multithreading=true)","category":"page"},{"location":"getting_started.html#Non-local-PDE-with-Neumann-boundary-conditions","page":"Getting started","title":"Non local PDE with Neumann boundary conditions","text":"","category":"section"},{"location":"getting_started.html","page":"Getting started","title":"Getting started","text":"Let's include in the previous equation non local competition, i.e.","category":"page"},{"location":"getting_started.html","page":"Getting started","title":"Getting started","text":"partial_t u = u (1 - int_Omega u(ty)dy) + frac12sigma^2Delta_xu tag2","category":"page"},{"location":"getting_started.html","page":"Getting started","title":"Getting started","text":"where Omega = -12 12^d, and let's assume Neumann Boundary condition on Omega.","category":"page"},{"location":"getting_started.html","page":"Getting started","title":"Getting started","text":"using HighDimPDE\n\n## Definition of the problem\nd = 10 # dimension of the problem\ntspan = (0.0,0.5) # time horizon\nx0 = fill(0.,d)  # initial point\ng(x) = exp( -sum(x.^2) ) # initial condition\nμ(x, p, t) = 0.0 # advection coefficients\nσ(x, p, t) = 0.1 # diffusion coefficients\nmc_sample = UniformSampling(fill(-5f-1, d), fill(5f-1, d))\nf(x, y, v_x, v_y, ∇v_x, ∇v_y, t) = max(0.0, v_x) * (1 -  max(0.0, v_y)) \nprob = PIDEProblem(g, f, μ, \n                    σ, x0, tspan) # defining x0_sample is sufficient to implement Neumann boundary conditions\n\n## Definition of the algorithm\nalg = MLP(mc_sample = mc_sample ) \n\nsol = solve(prob, alg, multithreading=true)","category":"page"},{"location":"getting_started.html#DeepSplitting","page":"Getting started","title":"DeepSplitting","text":"","category":"section"},{"location":"getting_started.html","page":"Getting started","title":"Getting started","text":"Let's solve the previous equation with DeepSplitting.","category":"page"},{"location":"getting_started.html","page":"Getting started","title":"Getting started","text":"using HighDimPDE\n\n## Definition of the problem\nd = 10 # dimension of the problem\ntspan = (0.0, 0.5) # time horizon\nx0 = fill(0f0,d)  # initial point\ng(x) = exp.(- sum(x.^2, dims=1) ) # initial condition\nμ(x, p, t) = 0.0f0 # advection coefficients\nσ(x, p, t) = 0.1f0 # diffusion coefficients\nx0_sample = UniformSampling(fill(-5f-1, d), fill(5f-1, d))\nf(x, y, v_x, v_y, ∇v_x, ∇v_y, p, t) = v_x .* (1f0 .- v_y)\nprob = PIDEProblem(g, f, μ, \n                    σ, x0, tspan, \n                    x0_sample = x0_sample)\n\n## Definition of the neural network to use\nusing Flux # needed to define the neural network\n\nhls = d + 50 #hidden layer size\n\nnn = Flux.Chain(Dense(d, hls, tanh),\n        Dense(hls, hls, tanh),\n        Dense(hls, 1)) # neural network used by the scheme\n\nopt = ADAM(1e-2)\n\n## Definition of the algorithm\nalg = DeepSplitting(nn,\n                    opt = opt,\n                    mc_sample = x0_sample)\n\nsol = solve(prob, \n            alg, \n            0.1, \n            verbose = true, \n            abstol = 2e-3,\n            maxiters = 1000,\n            batch_size = 1000)","category":"page"},{"location":"getting_started.html#Solving-on-the-GPU","page":"Getting started","title":"Solving on the GPU","text":"","category":"section"},{"location":"getting_started.html","page":"Getting started","title":"Getting started","text":"DeepSplitting can run on the GPU for (much) improved performance. To do so, just set use_cuda = true.","category":"page"},{"location":"getting_started.html","page":"Getting started","title":"Getting started","text":"sol = solve(prob, \n            alg, \n            0.1, \n            verbose = true, \n            abstol = 2e-3,\n            maxiters = 1000,\n            batch_size = 1000,\n            use_cuda=true)","category":"page"},{"location":"Feynman_Kac.html#feynmankac","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"","category":"section"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"The Feynman Kac formula is generally stated for terminal condition problems (see e.g. Wikipedia), where","category":"page"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"partial_t u(tx) + mu(x) nabla_x u(tx) + frac12 sigma^2(x) Delta_x u(tx) + f(x u(tx))  = 0 tag1","category":"page"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"with terminal condition u(T x) = g(x), and u colon R^d to R. ","category":"page"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"In this case the FK formula states that for all t in (0T) it holds that","category":"page"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"u(t x) = int_t^T mathbbE left f(X^x_s-t u(s X^x_s-t))ds right + mathbbE left u(0 X^x_T-t) right tag2","category":"page"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"where ","category":"page"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"X_t^x = int_0^t mu(X_s^x)ds + int_0^tsigma(X_s^x)dB_s + x","category":"page"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"and B_t is a Brownian motion.","category":"page"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"(Image: Brownian motion - Wikipedia)","category":"page"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"Intuitively, this formula is motivated by the fact that the density of Brownian particles (motion) satisfes the diffusion equation.","category":"page"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"The equivalence between the average trajectory of particles and PDEs given by the Feynman-Kac formula allows to overcome the curse of dimensionality that standard numerical methods suffer from, because the expectations can be approximated Monte Carlo integrations, which approximation error decreases as 1sqrtN and is therefore not dependent on the dimensions. On the other hand, the computational complexity of traditional deterministic techniques grows exponentially in the number of dimensions. ","category":"page"},{"location":"Feynman_Kac.html#Forward-non-linear-Feynman-Kac","page":"Feynman Kac formula","title":"Forward non-linear Feynman-Kac","text":"","category":"section"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"How to transform previous equation to an initial value problem?","category":"page"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"Define v(tau x) = u(T-tau x). Observe that v(0x) = u(Tx). Further observe that by the chain rule","category":"page"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"beginaligned\npartial_tau v(tau x) = partial_tau u(T-taux)\n                        = (partial_tau (T-tau)) partial_t u(T-taux)\n                        = -partial_t u(T-tau x)\nendaligned","category":"page"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"From Eq. (1) we get that ","category":"page"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"- partial_t u(T - taux) = mu(x) nabla_x u(T - taux) + frac12 sigma^2(x) Delta_x u(T - taux) + f(x u(T - taux))","category":"page"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"Replacing  u(T-tau x) by v(tau x) we get that v satisfies","category":"page"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"partial_tau v(tau x) = mu(x) nabla_x v(taux) + frac12 sigma^2(x) Delta_x v(taux) + f(x v(taux)) ","category":"page"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"and from Eq. (2) we obtain","category":"page"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"v(tau x) = int_T-tau^T mathbbE left f(X^x_s- T + tau v(s X^x_s-T + tau))ds right + mathbbE left v(0 X^x_tau) right","category":"page"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"By using the substitution rule with tau to tau -T (shifting by T) and tau to - tau (inversing), and finally inversing the integral bound we get that ","category":"page"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"beginaligned\nv(tau x) = int_-tau^0 mathbbE left f(X^x_s + tau v(s + T X^x_s + tau))ds right + mathbbE left v(0 X^x_tau) right\n            = - int_tau^0 mathbbE left f(X^x_tau - s v(T-s X^x_tau - s))ds right + mathbbE left v(0 X^x_tau) right\n            = int_0^tau mathbbE left f(X^x_tau - s v(T-s X^x_tau - s))ds right + mathbbE left v(0 X^x_tau) right\nendaligned","category":"page"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"This leads to the ","category":"page"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"info: Non-linear Feynman Kac for initial value problems\nConsider the PDEpartial_t u(tx) = mu(t x) nabla_x u(tx) + frac12 sigma^2(t x) Delta_x u(tx) + f(x u(tx))with initial conditions u(0 x) = g(x), where u colon R^d to R.  Thenu(t x) = int_0^t mathbbE left f(X^x_t - s u(T-s X^x_t - s))ds right + mathbbE left u(0 X^x_t) right tag3with X_t^x = int_0^t mu(X_s^x)ds + int_0^tsigma(X_s^x)dB_s + x","category":"page"},{"location":"DeepSplitting.html#deepsplitting","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"","category":"section"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"Modules = [HighDimPDE]\nPages   = [\"DeepSplitting.jl\"]","category":"page"},{"location":"DeepSplitting.html#HighDimPDE.DeepSplitting","page":"The DeepSplitting algorithm","title":"HighDimPDE.DeepSplitting","text":"DeepSplitting(nn, K=1, opt = ADAM(0.01), λs = nothing, mc_sample =  NoSampling())\n\nDeep splitting algorithm.\n\nArguments\n\nnn: a Flux.Chain, or more generally a functor.\nK: the number of Monte Carlo integrations.\nopt: optimiser to be use. By default, Flux.ADAM(0.01).\nλs: the learning rates, used sequentially. Defaults to a single value taken from opt.\nmc_sample::MCSampling : sampling method for Monte Carlo integrations of the non local term. Can be UniformSampling(a,b), NormalSampling(σ_sampling, shifted), or NoSampling (by default).\n\nExample\n\nhls = d + 50 # hidden layer size\nd = 10 # size of the sample\n\n# Neural network used by the scheme\nnn = Flux.Chain(Dense(d, hls, tanh),\n                Dense(hls,hls,tanh),\n                Dense(hls, 1, x->x^2))\n\nalg = DeepSplitting(nn, K=10, opt = ADAM(), λs = [5e-3,1e-3],\n                    mc_sample = UniformSampling(zeros(d), ones(d)) )\n\n\n\n\n\n","category":"type"},{"location":"DeepSplitting.html#HighDimPDE.solve-Tuple{PIDEProblem, DeepSplitting, Any}","page":"The DeepSplitting algorithm","title":"HighDimPDE.solve","text":"solve(prob::PIDEProblem,\n    alg::DeepSplitting,\n    dt;\n    batch_size = 1,\n    abstol = 1f-6,\n    verbose = false,\n    maxiters = 300,\n    use_cuda = false,\n    cuda_device = nothing,\n    verbose_rate = 100)\n\nReturns a PIDESolution object.\n\nArguments\n\nmaxiters: number of iterations per time step. Can be a tuple, where maxiters[1] is used for the training of the neural network used in the first time step (which can be long) and maxiters[2] is used for the rest of the time steps.\nbatch_size : the batch size.\nabstol : threshold for the objective function under which the training is stopped.\nverbose : print training information.\nverbose_rate : rate for printing training information (every verbose_rate iterations).\nuse_cuda : set to true to use CUDA.\ncuda_device : integer, to set the CUDA device used in the training, if use_cuda == true.\n\n\n\n\n\n","category":"method"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"The DeepSplitting algorithm reformulates the PDE as a stochastic learning problem.","category":"page"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"The algorithm relies on two main ideas:","category":"page"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"the approximation of the solution u by a parametric function bf u^theta. This function is generally chosen as a (Feedforward) Neural Network, as it is a universal approximator.\nthe training of bf u^theta by simulated stochastic trajectories of particles, through the link between linear PDEs and the expected trajectory of associated Stochastic Differential Equations (SDEs), explicitly stated by the Feynman Kac formula.","category":"page"},{"location":"DeepSplitting.html#The-general-idea","page":"The DeepSplitting algorithm","title":"The general idea 💡","text":"","category":"section"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"Consider the PDE","category":"page"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"partial_t u(tx) = mu(t x) nabla_x u(tx) + frac12 sigma^2(t x) Delta_x u(tx) + f(x u(tx)) tag1","category":"page"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"with initial conditions u(0 x) = g(x), where u colon R^d to R. ","category":"page"},{"location":"DeepSplitting.html#Local-Feynman-Kac-formula","page":"The DeepSplitting algorithm","title":"Local Feynman Kac formula","text":"","category":"section"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"DeepSplitting solves the PDE iteratively over small time intervals by using an approximate Feynman-Kac representation locally.","category":"page"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"More specifically, considering a small time step dt = t_n+1 - t_n one has that","category":"page"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"u(t_n+1 X_T - t_n+1) approx mathbbE left f(t X_T - t_n u(t_nX_T - t_n))(t_n+1 - t_n) + u(t_n X_T - t_n)  X_T - t_n+1right tag3","category":"page"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"One can therefore use Monte Carlo integrations to approximate the expectations","category":"page"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"u(t_n+1 X_T - t_n+1) approx frac1textbatch_sizesum_j=1^textbatch_size left u(t_n X_T - t_n^(j)) + (t_n+1 - t_n)sum_k=1^K big f(t_n X_T - t_n^(j) u(t_nX_T - t_n^(j))) big right","category":"page"},{"location":"DeepSplitting.html#Reformulation-as-a-learning-problem","page":"The DeepSplitting algorithm","title":"Reformulation as a learning problem","text":"","category":"section"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"The DeepSplitting algorithm approximates u(t_n+1 x) by a parametric function bf u^theta_n(x). It is advised to let this function be a neural network bf u_theta equiv NN_theta as they are universal approximators.","category":"page"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"For each time step t_n, the DeepSplitting algorithm ","category":"page"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"Generates the particle trajectories X^x (j) satisfying Eq. (2) over the whole interval 0T.\nSeeks bf u_n+1^theta  by minimising the loss function","category":"page"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"L(theta) = bf u^theta_n+1(X_T - t_n) - left f(t X_T - t_n-1 bf u_n-1(X_T - t_n-1))(t_n - t_n-1) + bf u_n-1(X_T - t_n-1) right ","category":"page"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"This way the PDE approximation problem is decomposed into a sequence of separate learning problems. In HighDimPDE.jl the right parameter combination theta is found by iteratively minimizing L using stochastic gradient descent.","category":"page"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"tip: Tip\nTo solve with DeepSplitting, one needs to provide to solvedt\nbatch_size\nmaxiters: the number of iterations for minimising the loss function\nabstol: the absolute tolerance for the loss function\nuse_cuda: if you have a Nvidia GPU, recommended.","category":"page"},{"location":"DeepSplitting.html#Solving-point-wise-or-on-a-hypercube","page":"The DeepSplitting algorithm","title":"Solving point-wise or on a hypercube","text":"","category":"section"},{"location":"DeepSplitting.html#Pointwise","page":"The DeepSplitting algorithm","title":"Pointwise","text":"","category":"section"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"DeepSplitting allows to obtain u(tx) on a single point  x in Omega with the keyword x.","category":"page"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"prob = PIDEProblem(g, f, μ, σ, x, tspan)","category":"page"},{"location":"DeepSplitting.html#Hypercube","page":"The DeepSplitting algorithm","title":"Hypercube","text":"","category":"section"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"Yet more generally, one wants to solve Eq. (1) on a d-dimensional cube ab^d. This is offered by HighDimPDE.jl with the keyworkd x0_sample.","category":"page"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"prob = PIDEProblem(g, f, μ, σ, x, tspan, x0_sample = x0_sample)","category":"page"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"Internally, this is handled by assigning a random variable as the initial point of the particles, i.e.","category":"page"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"X_t^xi = int_0^t mu(X_s^x)ds + int_0^tsigma(X_s^x)dB_s + xi","category":"page"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"where xi a random variable uniformly distributed over ab^d. This way, the neural network is trained on the whole interval ab^d instead of a single point.","category":"page"},{"location":"DeepSplitting.html#Nonlocal-PDEs","page":"The DeepSplitting algorithm","title":"Nonlocal PDEs","text":"","category":"section"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"DeepSplitting can solve for non-local reaction diffusion equations of the type","category":"page"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"partial_t u = mu(x) nabla_x u + frac12 sigma^2(x) Delta u + int_Omegaf(xy u(tx) u(ty))dy","category":"page"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"The non-localness is handled by a Monte Carlo integration.","category":"page"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"u(t_n+1 X_T - t_n+1) approx sum_j=1^textbatch_size left u(t_n X_T - t_n^(j)) + frac(t_n+1 - t_n)Ksum_k=1^K big f(t X_T - t_n^(j) Y_X_T - t_n^(j)^(k) u(t_nX_T - t_n^(j)) u(t_nY_X_T - t_n^(j)^(k))) big right","category":"page"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"tip: Tip\nIn practice, if you have a non-local model you need to provide the sampling method and the number K of MC integration through the keywords mc_sample and K. alg = DeepSplitting(nn, opt = opt, mc_sample = mc_sample, K = 1)mc_sample can be whether UniformSampling(a, b) or NormalSampling(σ_sampling, shifted).","category":"page"},{"location":"DeepSplitting.html#References","page":"The DeepSplitting algorithm","title":"References","text":"","category":"section"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"Boussange, V., Becker, S., Jentzen, A., Kuckuck, B., Pellissier, L., Deep learning approximations for non-local nonlinear PDEs with Neumann boundary conditions. arXiv (2022)\nBeck, C., Becker, S., Cheridito, P., Jentzen, A., Neufeld, A., Deep splitting method for parabolic PDEs. arXiv (2019)\nHan, J., Jentzen, A., E, W., Solving high-dimensional partial differential equations using deep learning. arXiv (2018)","category":"page"},{"location":"MLP.html#mlp","page":"The MLP algorithm","title":"The MLP algorithm","text":"","category":"section"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"Modules = [HighDimPDE]\nPages   = [\"MLP.jl\"]","category":"page"},{"location":"MLP.html#HighDimPDE.MLP","page":"The MLP algorithm","title":"HighDimPDE.MLP","text":"MLP(; M=4, L=4, K=10, mc_sample = NoSampling())\n\nMulti level Picard algorithm.\n\nArguments\n\nL: number of Picard iterations (Level),\nM: number of Monte Carlo integrations (at each level l, M^(L-l)integrations),\nK: number of Monte Carlo integrations for the non local term    \nmc_sample::MCSampling : sampling method for Monte Carlo integrations of the non local term. \n\nCan be UniformSampling(a,b), NormalSampling(σ_sampling), or NoSampling (by default).\n\n\n\n\n\n","category":"type"},{"location":"MLP.html#HighDimPDE.solve-Tuple{PIDEProblem, MLP}","page":"The MLP algorithm","title":"HighDimPDE.solve","text":"solve(prob::PIDEProblem,\n    alg::MLP;\n    multithreading=true,\n    verbose=false)\n\nReturns a PIDESolution object.\n\nArguments\n\nmultithreading : if true, distributes the job over all the threads available.\nverbose: print information over the iterations.\n\n\n\n\n\n","category":"method"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"The MLP, for Multi-Level Picard iterations, reformulates the PDE problem as a fixed point equation through the Feynman Kac formula. ","category":"page"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"It relies on Picard iterations to find the fixed point, \nreducing the complexity of the numerical approximation of the time integral through a multilvel Monte Carlo approach.","category":"page"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"The MLP algorithm overcomes the curse of dimensionality, with a computational complexity that grows polynomially in the number of dimension (see M. Hutzenthaler et al. 2020).","category":"page"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"warning: `MLP` can only approximate the solution on a single point\nMLP only works for PIDEProblem with x0_sample = NoSampling(). If you want to solve over an entire domain, you definitely want to check the DeepSplitting algorithm.","category":"page"},{"location":"MLP.html#The-general-idea","page":"The MLP algorithm","title":"The general idea 💡","text":"","category":"section"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"Consider the PDE","category":"page"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"partial_t u(tx) = mu(t x) nabla_x u(tx) + frac12 sigma^2(t x) Delta_x u(tx) + f(x u(tx)) tag1","category":"page"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"with initial conditions u(0 x) = g(x), where u colon R^d to R. ","category":"page"},{"location":"MLP.html#Picard-Iterations","page":"The MLP algorithm","title":"Picard Iterations","text":"","category":"section"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"The MLP algorithm observes that the Feynman Kac formula can be viewed as a fixed point equation, i.e. u = phi(u). Introducing a sequence (u_k) defined as u_0 = g and ","category":"page"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"u_l+1 = phi(u_l)","category":"page"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"the Banach fixed-point theorem ensures that the sequence converges to the true solution u. Such a technique is known as Picard iterations.","category":"page"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"The time integral term is evaluated by a Monte-Carlo integration","category":"page"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"u_L  = frac1Msum_i^M left f(X^x(i)_t - s_(l i) u_L-1(T-s_i X^x( i)_t - s_(l i))) + u(0 X^x(i)_t - s_(l i)) right","category":"page"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"But the MLP uses an extra trick to lower the computational cost of the iteration. ","category":"page"},{"location":"MLP.html#Telescope-sum","page":"The MLP algorithm","title":"Telescope sum","text":"","category":"section"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"The MLP algorithm uses a telescope sum ","category":"page"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"beginaligned\nu_L = phi(u_L-1) = phi(u_L-1) - phi(u_L-2) + phi(u_L-2) - phi(u_L-3) + dots \n= sum_l=1^L-1 phi(u_l-1) - phi(u_l-2)\nendaligned","category":"page"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"As l grows, the term phi(u_l-1) - phi(u_l-2) becomes smaller - and demands more calculations. The MLP algorithm uses this fact by evaluating the integral term at level l with M^L-l samples.","category":"page"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"tip: Tip\nL corresponds to the level of the approximation, i.e. u approx u_L\nM characterises the number of samples for the monte carlo approximation of the time integral","category":"page"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"Overall, MLP can be summarised by the following formula","category":"page"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"beginaligned\nu_L = sum_l=1^L-1 frac1M^L-lsum_i^M^L-l left f(X^x(l i)_t - s_(l i) u(T-s_(l i) X^x(l i)_t - s_(l i))) + mathbf1_N(l) f(X^x(l i)_t - s_(l i) u(T-s_(l i) X^x(l i)_t - s_(l i)))right\n\nqquad + frac1M^Lsum_i^M^L u(0 X^x(l i)_t)\nendaligned","category":"page"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"Note that the superscripts (l i) indicate the independence of the random variables X across levels.","category":"page"},{"location":"MLP.html#Nonlocal-PDEs","page":"The MLP algorithm","title":"Nonlocal PDEs","text":"","category":"section"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"MLP can solve for non-local reaction diffusion equations of the type","category":"page"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"partial_t u = mu(t x) nabla_x u(t x) + frac12 sigma^2(t x) Delta u(t x) + int_Omegaf(x y u(tx) u(ty))dy","category":"page"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"The non-localness is handled by a Monte Carlo integration.","category":"page"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"beginaligned\nu_L = sum_l=1^L-1 frac1M^L-lsum_i=1^M^L-l frac1Ksum_j=1^K  bigg f(X^x(l i)_t - s_(l i) Z^(lj) u(T-s_(l i) X^x(l i)_t - s_(l i)) u(T-s_li Z^(lj))) + \nqquad \nmathbf1_N(l) f(X^x(l i)_t - s_(l i) u(T-s_(l i) X^x(l i)_t - s_(l i)))bigg + frac1M^Lsum_i^M^L u(0 X^x(l i)_t)\nendaligned","category":"page"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"tip: Tip\nIn practice, if you have a non-local model you need to provide the sampling method and the number K of MC integration through the keywords mc_sample and K. K characterises the number of samples for the Monte Carlo approximation of the last term.\nmc_sample characterises the distribution of the Z variables","category":"page"},{"location":"MLP.html#References","page":"The MLP algorithm","title":"References","text":"","category":"section"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"Boussange, V., Becker, S., Jentzen, A., Kuckuck, B., Pellissier, L., Deep learning approximations for non-local nonlinear PDEs with Neumann boundary conditions. arXiv (2022)\nBecker, S., Braunwarth, R., Hutzenthaler, M., Jentzen, A., von Wurstemberger, P., Numerical simulations for full history recursive multilevel Picard approximations for systems of high-dimensional partial differential equations. arXiv (2020)","category":"page"},{"location":"index.html#HighDimPDE.jl","page":"Home","title":"HighDimPDE.jl","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"HighDimPDE.jl is a Julia package to solve Highly Dimensional non-linear, non-local PDEs of the form","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"beginaligned\n    (partial_t u)(tx) = int_Omega fbig(txbf x u(tx)u(tbf x) ( nabla_x u )(tx )( nabla_x u )(tbf x ) big)  dbf x \n     quad + biglangle mu(tx) ( nabla_x u )( tx ) bigrangle + tfrac12 textTrace big(sigma(tx)  sigma(tx) ^* ( textHess_x u)(t x ) big)\nendaligned","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"where u colon 0T times Omega to R, Omega subseteq R^d is subject to initial and boundary conditions, and where d is large.","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"HighDimPDE.jl implements solver algorithms that break down the curse of dimensionality, including","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"the Deep Splitting scheme\nthe Multi-Level Picard iterations scheme.","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"To make the most out of HighDimPDE.jl, we advise to first have a look at the ","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"documentation on the Feynman Kac formula,","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"as all solver algorithms heavily rely on it.","category":"page"},{"location":"index.html#Algorithm-overview","page":"Home","title":"Algorithm overview","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"Features DeepSplitting MLP\nTime discretization free ❌ ✅\nMesh-free ✅ ✅\nSingle point x in R^d approximation ✅ ✅\nd-dimensional cube ab^d approximation ✅ ❌\nGPU ✅ ❌\nGradient non-linearities ✔️ ❌","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"✔️ : will be supported in the future","category":"page"}]
}
