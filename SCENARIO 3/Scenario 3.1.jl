begin   # Libraries
    using Agents
    using Agents.Pathfinding
    using Random
    using ColorTypes
    using ImageMagick
    using FileIO: load
    using InteractiveDynamics
    using Images
    using DataFrames
    using Statistics
    using CairoMakie
    using DelimitedFiles
    using Observables
    using Makie
    using CSV
    using DataStructures   # BinaryMinHeap
end

# =============================
#  D* Lite Pathfinding Module
# =============================
module Pathfindinger

using DataStructures
using LinearAlgebra

# continuous (x,y) -> grid (row, col)  |  grid -> world center
@inline function to_cell(pos::Tuple{<:Real,<:Real}, spacing::Real)
    x = Float64(pos[1])
    y = Float64(pos[2])
    s = Float64(spacing)
    row = Int(clamp(round(y/s), 1, typemax(Int)))  # y -> row
    col = Int(clamp(round(x/s), 1, typemax(Int)))  # x -> col
    return (row, col)
end

@inline function to_world(cell::NTuple{2,Int}, spacing::Float64)
    row, col = cell
    x = (col * spacing)
    y = (row * spacing)
    return (x, y)
end

# 8-γειτονιά (Moore)
function moore_neighbors(cell::NTuple{2,Int}, dims::NTuple{2,Int}; periodic::Bool=false)
    (nrows, ncols) = dims
    (r, c) = cell
    out = NTuple{2,Int}[]
    for dr in -1:1, dc in -1:1
        (dr==0 && dc==0) && continue
        rr = r + dr; cc = c + dc
        if periodic
            rr = (rr < 1) ? nrows : ((rr > nrows) ? 1 : rr)
            cc = (cc < 1) ? ncols : ((cc > ncols) ? 1 : cc)
            push!(out, (rr, cc))
        else
            if 1 ≤ rr ≤ nrows && 1 ≤ cc ≤ ncols
                push!(out, (rr, cc))
            end
        end
    end
    return out
end
@inline diag_mult(dr::Int, dc::Int) = (abs(dr) + abs(dc) == 2) ? sqrt(2) : 1.0
@inline function h_octile(a::NTuple{2,Int}, b::NTuple{2,Int})
    dr = abs(a[1]-b[1]); dc = abs(a[2]-b[2])
    return max(dr,dc) + (sqrt(2)-1) * min(dr,dc)
end

# ---- D* Lite state ----
mutable struct DLitePathfinder
    dims::NTuple{2,Int}
    spacing::Float64
    periodic::Bool
    walkmap::BitMatrix             # true=walkable
    costmap::Array{Float64,2}      # terrain cost (0..1)
    w_move::Float64
    g::Array{Float64,2}
    rhs::Array{Float64,2}
    U::BinaryMinHeap{Tuple{Tuple{Float64,Float64},Tuple{Int,Int}}} # ((k1,k2),(r,c))
    k_m::Float64
    s_start::Tuple{Int,Int}
    s_goal::Tuple{Int,Int}
    s_last::Tuple{Int,Int}
    allow_diagonal::Bool
end

@inline function key(d::DLitePathfinder, s::Tuple{Int,Int})
    v = min(d.g[s...], d.rhs[s...])
    k1 = v + h_octile(d.s_start, s) + d.k_m
    k2 = v
    return (k1, k2)
end

function DLitePathfinder(
    dims::NTuple{2,Int};
    spacing::Float64=1.0,
    periodic::Bool=false,
    walkmap::BitMatrix=trues(dims...),
    costmap::Array{Float64,2},
    start_cell::Tuple{Int,Int},
    goal_cell::Tuple{Int,Int},
    allow_diagonal::Bool=true,
    w_move::Float64=1.0
)
    nrows, ncols = dims
    g   = fill(Inf, nrows, ncols)
    rhs = fill(Inf, nrows, ncols)
    U   = BinaryMinHeap{Tuple{Tuple{Float64,Float64},Tuple{Int,Int}}}()

    d = DLitePathfinder(dims, spacing, periodic, walkmap, costmap, w_move,
                        g, rhs, U, 0.0, start_cell, goal_cell, start_cell, allow_diagonal)
    initialize!(d)
    return d
end

function clear_heap!(U)
    while !isempty(U); pop!(U); end
end

function initialize!(d::DLitePathfinder)
    fill!(d.g, Inf); fill!(d.rhs, Inf); clear_heap!(d.U)
    d.k_m = 0.0; d.s_last = d.s_start
    d.rhs[d.s_goal...] = 0.0
    push!(d.U, (key(d, d.s_goal), d.s_goal))
    compute_shortest_path!(d)
end

@inline function edge_cost(d::DLitePathfinder, s::NTuple{2,Int}, sp::NTuple{2,Int})
    @inbounds begin
        d.walkmap[sp...] || return Inf
        base = d.w_move * diag_mult(sp[1]-s[1], sp[2]-s[2])
        pen  = d.costmap[sp...]
        return base + pen
    end
end

function successors(d::DLitePathfinder, s::NTuple{2,Int})
    Ns = moore_neighbors(s, d.dims; periodic=d.periodic)
    return d.allow_diagonal ? Ns : [p for p in Ns if abs(p[1]-s[1]) + abs(p[2]-s[2]) == 1]
end

function update_vertex!(d::DLitePathfinder, s::NTuple{2,Int})
    if s != d.s_goal
        minrhs = Inf
        for sp in successors(d, s)
            c = edge_cost(d, s, sp)
            val = d.g[sp...] + c
            minrhs = ifelse(val < minrhs, val, minrhs)
        end
        d.rhs[s...] = minrhs
    end
    if d.g[s...] != d.rhs[s...]
        push!(d.U, (key(d, s), s))
    end
end

# BinaryMinHeap: πάρε το min με first(U)
function topkey(U)
    isempty(U) && return ( (Inf, Inf), (0,0) )
    return first(U)
end

function compute_shortest_path!(d::DLitePathfinder)
    while true
        isempty(d.U) && break

        (k, u) = first(d.U)           # ((k1,k2),(r,c))
        ku1, ku2 = key(d, u)

        if (k[1] >= ku1 && k[2] >= ku2) && (d.rhs[d.s_start...] == d.g[d.s_start...])
            break
        end

        pop!(d.U)
        if k[1] < ku1 || k[2] < ku2
            push!(d.U, (key(d, u), u))                         # key outdated
        elseif d.g[u...] > d.rhs[u...]
            d.g[u...] = d.rhs[u...]                            # overconsistent
            for s in successors(d, u); update_vertex!(d, s); end
        else
            d.g[u...] = Inf                                    # underconsistent
            update_vertex!(d, u)
            for s in successors(d, u); update_vertex!(d, s); end
        end
    end
    return nothing
end

function next_cell(d::DLitePathfinder)
    s = d.s_start
    best = s; bestv = Inf
    for sp in successors(d, s)
        c = edge_cost(d, s, sp)
        val = d.g[sp...] + c
        if val < bestv; bestv = val; best = sp; end
    end
    return best
end

function advance_start!(d::DLitePathfinder, s_new::NTuple{2,Int})
    if s_new != d.s_start
        d.k_m += h_octile(d.s_last, s_new)
        d.s_start = s_new
        d.s_last  = s_new
        compute_shortest_path!(d)
    end
end

function notify_cost_change!(d::DLitePathfinder, sp::NTuple{2,Int})
    update_vertex!(d, sp)
    for s in successors(d, sp); update_vertex!(d, s); end
end

export DLitePathfinder, to_cell, to_world, notify_cost_change!, advance_start!, next_cell

end # module Pathfindinger

# =============================
#  Agent structure
# =============================
@agent struct AgentEscapes(ContinuousAgent{2, Float64})
    age::Float64
    mass::Float64
    toxicload::Float64
    pathX::Vector{Float64}
    pathY::Vector{Float64}
    TL1::Vector{Float64}
    TL2::Vector{Float64}
    TL3::Vector{Float64}
end

# =============================
#  Load maps (heightmap, CM)
# =============================
begin
    heightmap_data = load("Maps/Qatargas Map.jpg")
    heightmap_data = permutedims(channelview(heightmap_data), [2,3,1])[:,:,1]
    global heightmap = floor.(Int, convert.(Float64, heightmap_data) * 255)

    pmap = load("Concentration Map/1.bmp")
    pmap = permutedims(channelview(pmap), [2,3,1])[:,:,1]
    global penalty_map = floor.(Int, convert.(Float64, pmap) * 500)  # for Ct/toxic model
end



begin
# =============================
#  Terrain + Concentration as SUM of penalties (no thresholds)
# =============================
# Normalizations to [0,1]
    hn = (heightmap .- minimum(heightmap)) ./ (maximum(heightmap) - minimum(heightmap) + eps())  # light=1 (εύκολο terrain)
    cn = (penalty_map .- minimum(penalty_map)) ./ (maximum(penalty_map) - minimum(penalty_map) + eps())  # white=1 (υψηλό H2S)

    # Penalties:
    # - Terrain penalty: πιο σκοτεινό => μεγαλύτερο κόστος → 1 - hn
    # - Gas penalty: πιο άσπρο (H2S↑) => μεγαλύτερο κόστος → cn
    const W_TERRAIN = 1.0   # βάρη = 1 για “καθαρό άθροισμα”, μπορείς να τα αλλάξεις αν θες
    const W_GAS     = 1.0

    costmap = W_TERRAIN .* (1 .- hn) .+ W_GAS .* cn   # ΤΕΛΙΚΟ ΚΟΣΤΟΣ ΓΙΑ D* Lite

    # Walkable παντού (καμία σκληρή απαγόρευση)
    walkmap = trues(size(heightmap))
end

# =============================
#  Model parameters
# =============================
begin
    const dt = 1.0
    seed = 123
    n_agents = 100
    age_range = (22,60)
    speed_range = (4.0,7.0)
    mass_range = (50,80)

    ag_range_y = (size(heightmap,1) ÷ 2 - 50):(size(heightmap,1) ÷ 2 + 50)
    ag_range_x = (size(heightmap,2) ÷ 2 - 50):(size(heightmap,2) ÷ 2 + 50)

    dims = size(costmap)  # (rows, cols)
    spacing = 1.0

    dests = [(600., 900.), (100., 200.)]     # (x,y)  -- goal y clamped within rows
    rng = MersenneTwister(seed)
    # ContinuousSpace uses (x_max, y_max) = (cols, rows)
    space = ContinuousSpace( (size(costmap,2), size(costmap,1)); periodic=false, spacing=spacing )
end

# =============================
#  D* Lite instantiation
# =============================
begin
    using .Pathfindinger

    goal_xy   = dests[1]
    goal_cell = Pathfindinger.to_cell( (goal_xy[1], goal_xy[2]), spacing )
    goal_cell = (clamp(goal_cell[1], 1, dims[1]), clamp(goal_cell[2], 1, dims[2]))   # clamp in-bounds
    start_cell = goal_cell

    # Εγγύηση: goal walkable πριν χτίσουμε dstar
    walkmap[goal_cell...] = true
end


dstar = Pathfindinger.DLitePathfinder(
    dims;
    spacing=spacing,
    periodic=false,
    walkmap=walkmap,          # from CM threshold
    costmap=costmap,          # from terrain only
    start_cell=start_cell,
    goal_cell=goal_cell,
    allow_diagonal=true,
    w_move=1.0
)

properties = (
    dstar = dstar,
    heightmap = heightmap,
    penalty_map = penalty_map,
    dt = dt,
    speed_range = speed_range,
    goal = dests,
)

# =============================
#  Toxicity model
# =============================
function setupToxic()
    Atime = [0.0, 0.17, 0.83, 1.67, 4.17, 8.33]
    Arho = zeros(3, 6)
    Arho[1, 2:6] = [4.85, 4.23, 4.17, 4.06, 3.82]
    Arho[2, 2:6] = [180.79, 157.56, 155.43, 151.37, 142.48]
    Arho[3, 2:6] = [485.62, 423.22, 417.49, 406.59, 382.71]
    MW = 34
    Arho *= MW/24.04
    Arho = Arho'
    Atime = Atime*60
    taumin = 200.
    taumax = 86400.
    Brho = zeros(7,3)
    Balpha = zeros(7,3)
    rhomax = zeros(1,3)
    rhomin = zeros(1,3)
    Btime = zeros(7, 3)

    for k=1:3
        for b=2:6
            Brho[b, k] = Arho[b, k]
            Btime[b, k] = Atime[b]
        end
        Btime[1, k] = taumin
        Btime[7, k] = taumax
    end

    for k=1:3
        for b=3:6
            if Brho[b-1, k]==Brho[b, k]
                Balpha[b, k] = 0.0
            else
                Balpha[b, k] = log(Atime[b]/Atime[b-1])/log(Brho[b-1, k]/Brho[b, k])
            end
        end
        Balpha[2, k] = Balpha[3, k]
        Balpha[1, k] = Balpha[2, k]
        Balpha[7, k] = Balpha[6, k]
    end

    for k=1:3
        if Balpha[3, k]==0
            rhomax[k] = Brho[2, k]; Brho[1, k] = rhomax[k]
        else
            rhomax[k] = Brho[2, k]*(Btime[2, k]/taumin)^(1/Balpha[2, k]); Brho[1, k] = rhomax[k]
        end
        if Brho[5, k]==Brho[6, k]
            rhomin[k] = Brho[6, k]; Brho[7, k] = rhomin[k]
        else
            rhomin[k] = Brho[6, k]*(Btime[6, k]/taumax)^(1/Balpha[6, k]); Brho[7, k] = rhomin[k]
        end
    end

    for k=1:3, b=2:7
        if Balpha[b, k] == 0; Btime[b, k] = Btime[b-1, k]; end
    end
    for k=1:3, b=3:5
        if Balpha[b-1, k]==0 && Balpha[b, k]>0
            Balpha[b, k]=log(Btime[b, k]/Btime[b-1, k])/log(Brho[b-1, k]/Brho[b, k])
        end
    end
    return Balpha, Btime, Brho'
end

Balpha, Btime, Brho = setupToxic()

function update_toxic_load(Ct, TLcurrent, dt)
    TL = TLcurrent; TL_rate = 0.0
    for k = 1:3
        Cmin = Brho[k, 7]; Cmax = Brho[k, 1]
        if Ct > Cmax
            TL_rate = 1 / Btime[1]
        elseif Ct < Cmin
            TL_rate = 0.0
        else
            for i in 2:size(Btime, 1)
                if i <= size(Brho, 2)
                    if Brho[k, i-1] < Ct && Ct < Brho[k, i]
                        TL_rate = (1/Btime[i, k])*((Ct/Brho[k, i])^(Balpha[i, k]))
                    end
                end
            end
        end
        TL[k] = TL[k] .+ TL_rate * dt
    end
    TL[1] = TL[1] > 1.0 ? 1.0 : TL[1]
    TL[2] = TL[2] > 1.0 ? 1.0 : TL[2]
    TL[3] = TL[3] > 1.0 ? 1.0 : TL[3]
    return TL
end

# =============================
#  Colors
# =============================
function personcolor(person::AgentEscapes)
    if person.toxicload >= 3.      ; return :red
    elseif person.toxicload <= 1.0 ; return :green
    else                            ; return :orange
    end
end

# =============================
#  D* Lite <-> Agents.jl
# =============================
function plan_best_route!(agent, dstar::Pathfindinger.DLitePathfinder)
    s_start = Pathfindinger.to_cell( (agent.pos[1], agent.pos[2]), dstar.spacing )
    s_start = (clamp(s_start[1], 1, dstar.dims[1]), clamp(s_start[2], 1, dstar.dims[2]))
    # safety: start κελί πάντα walkable
    dstar.walkmap[s_start...] = true
    Pathfindinger.advance_start!(dstar, s_start)

    # αν δεν υπάρχει διαδρομή, αναγκαστική τοπική επιδιόρθωση
    if !isfinite(dstar.rhs[dstar.s_start...]) && !isfinite(dstar.g[dstar.s_start...])
        Pathfindinger.notify_cost_change!(dstar, s_start)
        Pathfindinger.compute_shortest_path!(dstar)
    end
    return nothing
end

function move_along_route!(agent, model, dstar::Pathfindinger.DLitePathfinder, speed::Float64, dt::Float64)
    s_here = Pathfindinger.to_cell( (agent.pos[1], agent.pos[2]), dstar.spacing )
    s_here == dstar.s_goal && return

    # Αν δεν είναι έτοιμο το path, κάνε 1 repair και ΠΑΛΙ κίνηση (fallback)
    if !isfinite(dstar.rhs[dstar.s_start...]) && !isfinite(dstar.g[dstar.s_start...])
        Pathfindinger.compute_shortest_path!(dstar)
    end

    s_next = Pathfindinger.next_cell(dstar)
    if s_next == dstar.s_start
        # Fallback: κίνηση προς τον στόχο με καθαρά ευρετική (greedy)
        neighs = Pathfindinger.successors(dstar, dstar.s_start)
        if !isempty(neighs)
            s_best = neighs[1]
            v_best = Pathfindinger.h_octile(s_best, dstar.s_goal)
            for n in neighs
                v = Pathfindinger.h_octile(n, dstar.s_goal)
                if v < v_best; v_best = v; s_best = n; end
            end
            s_next = s_best
        end
    end

    target_xy = Pathfindinger.to_world(s_next, dstar.spacing)
    x, y = agent.pos; tx, ty = target_xy
    dx = tx - x; dy = ty - y; dist = hypot(dx, dy)
    step = speed * dt
    agent.pos = (dist ≤ 1e-6 || step >= dist) ? (tx, ty) : (x + step*dx/dist, y + step*dy/dist)

    push!(agent.pathX, agent.pos[1]); push!(agent.pathY, agent.pos[2])

    s_new = Pathfindinger.to_cell( (agent.pos[1], agent.pos[2]), dstar.spacing )
    s_new = (clamp(s_new[1], 1, dstar.dims[1]), clamp(s_new[2], 1, dstar.dims[2]))
    Pathfindinger.advance_start!(dstar, s_new)
end

# =============================
#  Steps
# =============================
function agent_step!(person, model)
    # CM lookup (row=iy from y, col=ix from x)
    iy = clamp(round(Int, person.pos[2]), 1, size(model.penalty_map,1))
    ix = clamp(round(Int, person.pos[1]), 1, size(model.penalty_map,2))
    Ct = model.penalty_map[iy, ix]

    TLcurrent = [person.TL1[end], person.TL2[end], person.TL3[end]]
    TL = update_toxic_load(Ct, TLcurrent, model.dt)

    person.toxicload = sum(TL)
    push!(person.TL1, TL[1]); push!(person.TL2, TL[2]); push!(person.TL3, TL[3])

    # speed model (monotonic decreasing vs TL)
    speed = 1.35
    if 0 < person.toxicload <= 1
        speed = 1.35 * exp(-0.393 * person.toxicload)
    elseif 1 < person.toxicload < 3
        t = (person.toxicload - 1) / 2
        speed = 1.35 * (0.5 * (1 - t))
    elseif person.toxicload >= 3
        speed = 0.0
    end

    plan_best_route!(person, model.dstar)
    move_along_route!(person, model, model.dstar, speed, model.dt)
end

function model_step!(model)
    for (a1, a2) in interacting_pairs(model, 1.0, :nearest)
        elastic_collision!(a1, a2, :mass)
    end
end

# =============================
#  Build model & agents
# =============================
model = ABM(
  AgentEscapes,
  space;
  rng          = rng,
  properties   = properties,
  agent_step!  = agent_step!,
  model_step!  = model_step!
)

for _ in 1:n_agents
    age = rand(abmrng(model))*(age_range[2]-age_range[1]) + age_range[1]
    mass = rand(abmrng(model)) * (mass_range[2]-mass_range[1]) + mass_range[1]
    vel = Tuple(rand(abmrng(model), 2) .* (speed_range[2]-speed_range[1]) .+ speed_range[1])
    pos = (rand(abmrng(model), ag_range_x), rand(abmrng(model), ag_range_y))
    person = add_agent!(pos, AgentEscapes, model, vel, age, mass, 1., [pos[1]], [pos[2]], [0.0], [0.0], [0.0])
    # safety: start κελί walkable
    # --- Βεβαιώσου ότι και το κελί εκκίνησης είναι walkable ---
    s0 = Pathfindinger.to_cell((pos[1], pos[2]), model.dstar.spacing)
    s0 = (clamp(s0[1], 1, model.dstar.dims[1]), clamp(s0[2], 1, model.dstar.dims[2]))
    model.dstar.walkmap[s0...] = true
    plan_best_route!(person, model.dstar)
end

# =============================
#  CM cycling helper (update walkmap)
# =============================
function update_costmap_soft!(
    dstar::Pathfindinger.DLitePathfinder,
    new_penalty_map;
    w_terrain::Real = W_TERRAIN,
    w_gas::Real     = W_GAS,
    terrain_hn::AbstractMatrix = hn  # σταθερό terrain
)
    cn_new = (new_penalty_map .- minimum(new_penalty_map)) ./ (maximum(new_penalty_map) - minimum(new_penalty_map) + eps())
    new_costmap = w_terrain .* (1 .- terrain_hn) .+ w_gas .* cn_new

    changed = findall(abs.(new_costmap .- dstar.costmap) .> 1e-9)
    dstar.costmap .= new_costmap

    CI = CartesianIndices(dstar.dims)
    for idx in changed
        r, c = Tuple(CI[idx])
        Pathfindinger.notify_cost_change!(dstar, (r,c))
    end
    Pathfindinger.compute_shortest_path!(dstar)
end

# =============================
#  Animation + CSV export (with CM cycling)
# =============================
begin
    const T = 400

    fig = Figure(; size = (800,800))
    ax  = Makie.Axis(fig[1,1]; title="Evacuation with D* Lite (terrain cost) & CM avoidance", aspect=DataAspect())

    heatmap!(ax, heightmap; colormap=:grays, alpha=0.3)
    goals = model.goal
    scatter!(ax, getindex.(goals,1), getindex.(goals,2); color=(:red,50), marker=:circle)

    frame_obs = Observable(0)
    counter_lbl = Label(fig, @lift("Time elapsed = $((($frame_obs-1)*dt)) s"), fontsize=16, padding=(6,10,6,10), halign=:left)
    fig[1,1, TopLeft()] = counter_lbl

    xs0, ys0, colors0 = Float64[], Float64[], Symbol[]
    for a in allagents(model)
        push!(xs0, a.pos[1]); push!(ys0, a.pos[2]); push!(colors0, personcolor(a))
    end
    posobs = Observable(Point2f.(xs0, ys0))
    colobs = Observable(colors0)

    lines_plots = [ lines!(ax, [a.pos[1]], [a.pos[2]]; color=personcolor(a), linewidth=2)
                    for a in allagents(model) ]
    scatter!(ax, posobs; color=colobs, markersize=10)

    df = DataFrame(step=Int[], agent_id=Int[], x=Float64[], y=Float64[], toxicload=Float64[])

    video_file = "SCENARIO 3/Simulation Results/SCENARIO_3_DLite_$(seed).mp4"
    record(fig, video_file, 1:T; framerate=30) do frame
        frame_obs[] = frame

        # CM cycling every 120 frames: loads Penalty Map/1.bmp .. /10.bmp
        if frame % 120 == 1
            idx = 1 + ((frame-1) ÷ 120) % 10
            pmapN = load("Concentration Map/$(idx).bmp") |> x -> permutedims(channelview(x), [2,3,1])[:,:,1]
            new_penalty = floor.(Int, convert.(Float64, pmapN) * 500)
            global penalty_map = new_penalty
            update_costmap_soft!(model.dstar, new_penalty)
            # επιπλέον: υποχρεωτική επισκευή για το τρέχον start
            Pathfindinger.compute_shortest_path!(model.dstar)
        end

        step!(model, agent_step!, model_step!, 1)

        for (i,a) in enumerate(allagents(model))
            lines_plots[i][1][] = Point2f.(a.pathX, a.pathY)
        end
        xs = [a.pos[1] for a in allagents(model)]
        ys = [a.pos[2] for a in allagents(model)]
        posobs[] = Point2f.(xs, ys)
        colobs[] = [personcolor(a) for a in allagents(model)]

        for a in allagents(model)
            push!(df, (frame, a.id, a.pos[1], a.pos[2], a.toxicload))
        end
    end

    println("Saved animation: $video_file")
    csv_file = "SCENARIO 3/Simulation Results/SCENARIO_3_DLite_$(seed).csv"
    CSV.write(csv_file, df)
    println("Saved CSV: $csv_file")
end