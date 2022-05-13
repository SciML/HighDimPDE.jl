using Documenter, HighDimPDE

makedocs(sitename="HighDimPDE.jl",
        format = Documenter.HTML(prettyurls = false),
        authors = "Victor Boussange",
        pages = [
            "Home" => "index.md",
            "Getting started" => "getting_started.md",
            "Solver Algorithms" => ["MLP.md", 
                        "DeepSplitting.md", 
                        ],
            "Feynman Kac formula" => "Feynman_Kac.md",
        ],)

deploydocs(repo = "github.com/SciML/HighDimPDE.jl", devbranch="main")
