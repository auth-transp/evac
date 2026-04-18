# Overlay Scenario 4 goals (DESTS_ALL) on Concentration Maps/Scenario 5.jpg
# Labels: global index 1:16 (indices used in DEST_INDICES_BY_SIDE point into DESTS_ALL).
# Scatter color by side A-D per DEST_INDICES_BY_SIDE.
#
# Run from repo root:
#   julia "SCENARIOS/SCENARIO 4/plot_dests_on_scenario5.jl"
#
# Uses NADEEN/Project.toml (same deps as Scenario 4.jl: CairoMakie, FileIO, Images, …).

import Pkg
const _REPO_FOR_ENV = dirname(dirname(@__DIR__))
Pkg.activate(joinpath(_REPO_FOR_ENV, "NADEEN"))

using CairoMakie
using FileIO: load
using ImageMagick
using Images
using Dates

# Same as Scenario 4.jl
const DESTS_ALL = Tuple{Float64, Float64}[
    (500., 854.), (120., 248.), (691., 750.), (351., 924.),
    (853., 659.), (795., 375.), (703., 187.), (864., 503.),
    (652., 73.), (325., 150.), (223., 198.), (473., 78.),
    (88., 493.), (145., 630.), (209., 787.), (39., 379.),
]

const DEST_INDICES_BY_SIDE = Vector{Vector{Int}}([
    [1, 3, 4, 5],
    [6, 7, 8, 9],
    [10, 11, 12, 2],
    [13, 14, 15, 16],
])

const SIDE_LABELS = ['A', 'B', 'C', 'D']
const SIDE_COLORS = [:crimson, :dodgerblue, :forestgreen, :darkorange]

function side_and_local_for_global(g::Int)
    for (s, idxs) in enumerate(DEST_INDICES_BY_SIDE)
        k = findfirst(==(g), idxs)
        if k !== nothing
            return s, SIDE_LABELS[s], SIDE_COLORS[s], Int(k)
        end
    end
    error("Global index $g not found in DEST_INDICES_BY_SIDE")
end

scenario5_path = joinpath(_REPO_FOR_ENV, "Concentration Maps", "Scenario 5.jpg")
out_dir = joinpath(@__DIR__, "Simulation Results")

img_loaded = load(scenario5_path)
img_rgb = Matrix(RGB.(img_loaded))
h_img, w_img = size(img_rgb)

xs = Float64[d[1] for d in DESTS_ALL]
ys = Float64[d[2] for d in DESTS_ALL]
colors = Symbol[]
labels_simple = String[]
for g in 1:16
    _, letter, col, loc = side_and_local_for_global(g)
    push!(colors, col)
    push!(labels_simple, "$(g) ($(letter)$(loc))")
end

fig = Figure(; size = (1100, 1000))
ax = Axis(
    fig[1, 1];
    title = "DESTS_ALL (1-16) on Scenario 5.jpg — color by side A-D",
    aspect = DataAspect(),
    xlabel = "x (same frame as Scenario 4)",
    ylabel = "y",
)

image!(ax, 1:w_img, 1:h_img, img_rgb)

scatter!(
    ax, xs, ys;
    color = colors,
    markersize = 14,
    strokecolor = :black,
    strokewidth = 1.5,
)

text!(
    ax, xs, ys;
    text = labels_simple,
    fontsize = 9,
    align = (:left, :bottom),
    offset = (6, 6),
    color = :white,
    strokewidth = 1.2,
    strokecolor = :black,
)

xlims!(ax, 0, w_img + 1)
ylims!(ax, 0, h_img + 1)

elem = [MarkerElement(color = SIDE_COLORS[i], marker = :circle, markersize = 12) for i in 1:4]
Legend(
    fig[1, 2],
    elem,
    ["Side $(SIDE_LABELS[i])" for i in 1:4];
    tellheight = false,
    tellwidth = false,
)

mkpath(out_dir)
ts = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
out_png = joinpath(out_dir, "Scenario5_DESTS_ALL_numbered_$ts.png")
save(out_png, fig)
println("Saved: $out_png")
println("Image size: $(w_img) x $(h_img) px")
for g in 1:16
    x, y = DESTS_ALL[g]
    if x < 1 || x > w_img || y < 1 || y > h_img
        @warn "Point $g = ($x, $y) outside image bounds 1:$w_img x 1:$h_img"
    end
end
