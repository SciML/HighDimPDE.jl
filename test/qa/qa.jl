using SciMLTesting, HighDimPDE, Test

run_qa(
    HighDimPDE;
    explicit_imports = true,
    # `@reexport using DiffEqBase` re-exports du_cache/u_cache/user_cache (owned by
    # SciMLBase, only re-exported by DiffEqBase). On Julia <=1.11 these resolve fine,
    # but Julia 1.12's binding-partition change makes the double-reexported names
    # report `isdefined == false`, so Aqua flags them as undefined exports. They are
    # absent from src/; this is a 1.12-only artifact of the SciMLBase->DiffEqBase
    # reexport chain, fixable only upstream. Pre-existing on main. Tracked in
    # https://github.com/SciML/HighDimPDE.jl/issues/157; remove this entry once the
    # upstream chain is fixed and the names resolve on 1.12.
    aqua_broken = (:undefined_exports,),
    # HighDimPDE leans on heavy `using` of CUDA/Flux/Zygote/DiffEqBase(reexport)/etc.;
    # making 37 implicit names explicit is a risky mass refactor, tracked in
    # https://github.com/SciML/HighDimPDE.jl/issues/155
    ei_broken = (:no_implicit_imports,),
    ei_kwargs = (;
        # Flux internals accessed qualified that Flux has not declared public.
        all_qualified_accesses_are_public = (;
            ignore = (
                :AbstractOptimiser,  # Flux.Optimise
                :Optimise,           # Flux
                :params,             # Flux
            ),
        ),
    )
)
