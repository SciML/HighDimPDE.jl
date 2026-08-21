@setup_workload begin
    @compile_workload begin
        μ(x, p, t) = zero(x)
        σ(x, p, t) = LinearAlgebra.I
        g(x) = sum(abs2, x)
        f(x, u, du, p, t) = zero(u)
        f_pide(x, y, ux, uy, dux, duy, p, t) = zero(ux)

        ParabolicPDEProblem(μ, σ, zeros(2), (0.0, 1.0); g, f)
        PIDEProblem(μ, σ, zeros(2), (0.0, 1.0), g, f_pide)
    end
end
