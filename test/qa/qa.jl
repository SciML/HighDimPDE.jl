using HighDimPDE, Aqua
@testset "Aqua" begin
    Aqua.find_persistent_tasks_deps(HighDimPDE)
    Aqua.test_ambiguities(HighDimPDE, recursive = false)
    Aqua.test_deps_compat(HighDimPDE)
    Aqua.test_piracies(HighDimPDE)
    Aqua.test_project_extras(HighDimPDE)
    Aqua.test_stale_deps(HighDimPDE)
    Aqua.test_unbound_args(HighDimPDE)
    Aqua.test_undefined_exports(HighDimPDE)
end
