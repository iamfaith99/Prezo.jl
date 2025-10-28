using Documenter
using Prezo

makedocs(
    sitename = "Prezo.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://YOUR_USERNAME.github.io/Prezo.jl",
        assets = String[],
    ),
    modules = [Prezo],
    pages = [
        "Home" => "index.md",
        "User Guide" => [
            "Getting Started" => "guide/getting_started.md",
            "Market Data" => "guide/market_data.md",
            "Options" => "guide/options.md",
            "Pricing Engines" => "guide/pricing_engines.md",
        ],
        "API Reference" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/YOUR_USERNAME/Prezo.jl.git",
    devbranch = "main",
)
