using SciMLTesting, HighDimPDE, Test
using NNlib: fast_act

run_qa(
    HighDimPDE;
    explicit_imports = true,
    # `NNlib.fast_act(::typeof(tanh), ::CuArray) = tanh` (in src/HighDimPDE.jl) is
    # intentional type piracy: a CUDA-side opt-out of NNlib's `tanh_fast` substitution
    # that works around an `InvalidIRError` when broadcasting `ComposedFunction{tanh_fast,
    # +}` on the GPU. This is exactly the per-array-type override NNlib's `fast_act` API
    # was built for. See FluxML/Flux.jl#2633 for the analogous Metal report.
    aqua_kwargs = (; piracies = (; treat_as_own = [fast_act])),
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
                # `NNlib.fast_act` is NNlib's per-array-type activation opt-out hook;
                # HighDimPDE adds `fast_act(::typeof(tanh), ::CuArray) = tanh` to work
                # around a GPU broadcast `InvalidIRError` (see FluxML/Flux.jl#2633).
                # NNlib exposes it as a public override point but has not declared it
                # `public`, so the public-API check flags the qualified access.
                :fast_act,           # NNlib
            ),
        ),
    )
)
