using Documenter, HighDimPDE
# push!(LOAD_PATH,"/Users/victorboussange/ETHZ/projects/EvoId/") # not sure this is necessary
pathsrc = joinpath(@__DIR__,"src")
makedocs(sitename="HighDimPDE.jl",
        format = Documenter.HTML(prettyurls = false),
        authors = "Victor Boussange",
        pages = [
            "Home" => "index.md",
            "Getting started" => "getting_started.md",
            "Algorithms" => ["MLP.md", 
                        "DeepSplitting.md", 
                        ],
            "Feynman Kac formula" => "Feynman_Kac.md",
        ],)

deploydocs(repo = "github.com/vboussange/HighDimPDE.jl", devbranch="main")
