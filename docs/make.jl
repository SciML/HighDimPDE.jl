using Documenter, HighDimPDE

include("pages.jl")

makedocs(sitename="HighDimPDE.jl",
        format = Documenter.HTML(prettyurls = false),
        authors = "Victor Boussange",
        pages = pages)

deploydocs(repo = "github.com/SciML/HighDimPDE.jl", devbranch="main")
