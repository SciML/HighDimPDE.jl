using Documenter, HighDimPDE

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

include("pages.jl")

makedocs(sitename="HighDimPDE.jl",
        authors = "Victor Boussange",
        pages = pages,
        clean = true, doctest = false, linkcheck = true,
        format = Documenter.HTML(assets = ["assets/favicon.ico"],
                                 canonical = "https://docs.sciml.ai/HighDimPDE/stable/"),)

deploydocs(repo = "github.com/SciML/HighDimPDE.jl", devbranch="main")
