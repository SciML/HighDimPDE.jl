using Documenter, HighDimPDE

include("pages.jl")

makedocs(
    sitename = "HighDimPDE.jl",
    authors = "#",
    pages = pages,
    clean = true,
    doctest = true,
    checkdocs = :exports,
    linkcheck = true,
    format = Documenter.HTML(
        assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/HighDimPDE/stable/"
    )
)

deploydocs(repo = "github.com/SciML/HighDimPDE.jl", devbranch = "main")
