using SciMLTesting, HighDimPDE, Test

run_qa(
    HighDimPDE;
    explicit_imports = true,
    # HighDimPDE leans on heavy `using` of CUDA/Flux/Zygote/DiffEqBase(reexport)/etc.;
    # making 37 implicit names explicit is a risky mass refactor, tracked in
    # https://github.com/SciML/HighDimPDE.jl/issues/155
    ei_broken = (:no_implicit_imports,),
    ei_kwargs = (;
        # Non-public names from other packages, accessed qualified. They become
        # public as those base libs add `public` declarations / exports.
        all_qualified_accesses_are_public = (;
            ignore = (
                :AbstractODEAlgorithm,  # SciMLBase
                :AbstractODEProblem,    # SciMLBase
                :AbstractSciMLProblem,  # SciMLBase
                :AbstractOptimiser,     # Flux.Optimise
                :Optimise,              # Flux
                :params,                # Flux
            ),
        ),
    )
)
