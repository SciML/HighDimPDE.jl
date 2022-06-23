using Documenter, HighDimPDE

include("pages.jl")

makedocs(sitename="HighDimPDE.jl",
        format = Documenter.HTML(prettyurls = false),
        authors = "Victor Boussange",
        pages = pages,
        strict=[
        :doctest, 
        :linkcheck, 
        :parse_error,
        :example_block,
        # Other available options are
        # :autodocs_block, :cross_references, :docs_block, :eval_block, :example_block, :footnote, :meta_block, :missing_docs, :setup_block
        ],
        )

deploydocs(repo = "github.com/SciML/HighDimPDE.jl", devbranch="main")
