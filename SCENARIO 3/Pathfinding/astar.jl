"""
    Path{D,T}
Alias of `MutableLinkedList{NTuple{D,T}}`. Used to represent the path to be
taken by an agent in a `D` dimensional space.
"""
const Path{D,T} = MutableLinkedList{NTuple{D,T}}

struct AStar{D,P,M,T,C<:CostMetric{D}} <: GridPathfinder{D,P,M}
    agent_paths::Dict{Int,Path{D,T}}
    dims::NTuple{D,T}
    neighborhood::Vector{CartesianIndex{D}}
    admissibility::Float64
    walkmap::BitArray{D}
    cost_metric::C

    function AStar{D,P,M,T,C}(
        agent_paths::Dict,
        dims::NTuple{D,T},
        neighborhood::Vector{CartesianIndex{D}},
        admissibility::Float64,
        walkmap::BitArray{D},
        cost_metric::C,
    ) where {D,P,M,C,T}
        @assert all(dims .> 0) "Invalid pathfinder dimensions: $(dims)"
        T <: Integer && @assert size(walkmap) == dims "Walkmap must be same dimensions as grid"
        @assert admissibility >= 0 "Invalid value for admissibility: $admissibility ≱ 0"
        if cost_metric isa PenaltyMap{D}
            @assert size(cost_metric.pmap) == size(walkmap) "Penaltymap dimensions must be same as walkable map"
        elseif cost_metric isa DirectDistance{D}
            if M
                @assert length(cost_metric.direction_costs) >= D "DirectDistance direction_costs must have as many values as dimensions"
            else
                @assert length(cost_metric.direction_costs) >= 1 "DirectDistance direction_costs must have non-zero length"
            end
        end
        new(agent_paths, dims, neighborhood, admissibility, walkmap, cost_metric)
    end
end

"""
    Pathfinding.AStar(space; kwargs...)
Enables pathfinding for agents in the provided `space` (which can be a [`GridSpace`](@ref) or
[`ContinuousSpace`](@ref)) using the A* algorithm. This struct must be passed into any
pathfinding functions.

For [`ContinuousSpace`](@ref), a walkmap or instance of [`PenaltyMap`](@ref) must be provided
to specify the level of discretisation of the space.

## Keywords
- `diagonal_movement = true` specifies if movement can be to diagonal neighbors of a
  tile, or only orthogonal neighbors. Only available for [`GridSpace`](@ref)
- `admissibility = 0.0` allows the algorithm to approximate paths to speed up pathfinding.
  A value of `admissibility` allows paths with at most `(1+admissibility)` times the optimal
  length.
- `walkmap = trues(spacesize(space))` specifies the (un)walkable positions of the
  space. If specified, it should be a `BitArray` of the same size as the corresponding
  `GridSpace`. By default, agents can walk anywhere in the space.
- `cost_metric = DirectDistance{D}()` is an instance of a cost metric and specifies the
  metric used to approximate the distance between any two points.

Utilization of all features of `AStar` occurs in the
[3D Mixed-Agent Ecosystem with Pathfinding](@id rabbit_fox_hawk)
example.
"""
function AStar(
    dims::NTuple{D,T};
    periodic::Union{Bool,NTuple{D,Bool}} = false,
    diagonal_movement::Bool = true,
    admissibility::Float64 = 0.0,
    walkmap::BitArray{D} = trues(dims),
    cost_metric::CostMetric{D} = DirectDistance{D}(),
) where {D,T}
    neighborhood = diagonal_movement ? moore_neighborhood(D) : vonneumann_neighborhood(D)
    return AStar{D,periodic,diagonal_movement,T,typeof(cost_metric)}(
        Dict{Int,Path{D,T}}(),
        dims,
        neighborhood,
        admissibility,
        walkmap,
        cost_metric,
    )
end

AStar(
    space::GridSpace{D,periodic};
    diagonal_movement::Bool = true,
    admissibility::Float64 = 0.0,
    walkmap::BitArray{D} = trues(spacesize(space)),
    cost_metric::CostMetric{D} = DirectDistance{D}(),
) where {D,periodic} =
    AStar(size(space); periodic, diagonal_movement, admissibility, walkmap, cost_metric)

function AStar(
    space::ContinuousSpace{D,periodic};
    walkmap::Union{BitArray{D},Nothing} = nothing,
    admissibility::Float64 = 0.0,
    cost_metric::CostMetric{D} = DirectDistance{D}(),
) where {D,periodic}
    @assert walkmap isa BitArray{D} || cost_metric isa PenaltyMap "Pathfinding in ContinuousSpace requires either walkmap to be specified or cost_metric to be a PenaltyMap"
    isnothing(walkmap) && (walkmap = BitArray(trues(size(cost_metric.pmap))))
    AStar(Tuple(Agents.spacesize(space));
        periodic, diagonal_movement = true,
        admissibility, walkmap, cost_metric
    )
end


moore_neighborhood(D) = [
    CartesianIndex(a)
    for a in Iterators.product([-1:1 for φ in 1:D]...) if a != Tuple(zeros(Int, D))
]

function vonneumann_neighborhood(D)
    hypercube = CartesianIndices((repeat([-1:1], D)...,))
    [β for β ∈ hypercube if LinearAlgebra.norm(β.I) == 1]
end

function Base.show(io::IO, pathfinder::AStar{D,P,M}) where {D,P,M}
    periodic = get_periodic_type(pathfinder)
    moore = M ? "diagonal, " : "orthogonal, "
    s =
        "A* in $(D) dimensions, $(periodic)$(moore)ϵ=$(pathfinder.admissibility), " *
        "metric=$(pathfinder.cost_metric)"
    print(io, s)
end
get_periodic_type(::AStar{D,false,M}) where {D,M} = ""
get_periodic_type(::AStar{D,true,M}) where {D,M} = "periodic, "
get_periodic_type(::AStar{D,P,M}) where {D,P,M} = "mixed periodicity, "

struct GridCell
    f::Int
    g::Int
    h::Int
end

GridCell(g::Int, h::Int, admissibility::Float64) =
    GridCell(round(Int, g + (1 + admissibility) * h), g, h)

GridCell() = GridCell(typemax(Int), typemax(Int), typemax(Int))

ordering(cell) = cell.f

"""
    find_path(pathfinder::AStar{D}, from::NTuple{D,Int}, to::NTuple{D,Int})
Calculate the shortest path from `from` to `to` using the A* algorithm.
If a path does not exist between the given positions, an empty linked list is returned.
"""
function find_path(pathfinder::AStar{D}, from::Dims{D}, to::Dims{D}) where {D}
    if !all(1 .<= from .<= size(pathfinder.walkmap)) ||
        !all(1 .<= to .<= size(pathfinder.walkmap)) ||
        !pathfinder.walkmap[from...] ||
        !pathfinder.walkmap[to...]
        return # nothing
    end
    parent = Dict{Dims{D},Dims{D}}()

    open_list = PriorityQueue{Dims{D},GridCell}(Base.By(ordering))
    closed_list = Set{Dims{D}}()

    enqueue!(
        open_list,
        from,
        GridCell(0, delta_cost(pathfinder, from, to), pathfinder.admissibility)
    )

    while !isempty(open_list)
        cur, cell = dequeue_pair!(open_list)
        cur == to && break
        push!(closed_list, cur)

        nbors = get_neighbors(cur, pathfinder)
        for nbor in Iterators.filter(n -> inbounds(n, pathfinder, closed_list), nbors)
            nbor_cell = haskey(open_list, nbor) ? open_list[nbor] : GridCell()
            new_g_cost = cell.g + delta_cost(pathfinder, cur, nbor)

            if new_g_cost < nbor_cell.g
                parent[nbor] = cur
                open_list[nbor] = GridCell(
                    new_g_cost,
                    delta_cost(pathfinder, nbor, to),
                    pathfinder.admissibility,
                )
            end
        end
    end

    agent_path = Path{D,Int64}()
    cur = to
    while true
        haskey(parent, cur) || break
        pushfirst!(agent_path, cur)
        cur = parent[cur]
    end
    cur == from || return # nothing
    return agent_path
end

@inline get_neighbors(cur, pathfinder::AStar{D,true}) where {D} =
    (mod1.(cur .+ β.I, size(pathfinder.walkmap)) for β in pathfinder.neighborhood)
@inline get_neighbors(cur, pathfinder::AStar{D,false}) where {D} =
    (cur .+ β.I for β in pathfinder.neighborhood)
@inline function get_neighbors(cur, pathfinder::AStar{D,P}) where {D,P}
    s = size(pathfinder.walkmap)
    (
        ntuple(i -> P[i] ? mod1(cur[i] + β[i], s[i]) : cur[i] + β[i], D)
        for β in pathfinder.neighborhood
    )
end
@inline inbounds(n, pathfinder, closed) =
    all(1 .<= n .<= size(pathfinder.walkmap)) && pathfinder.walkmap[n...] && n ∉ closed

Base.isempty(id::Int, pathfinder::AStar) =
    !haskey(pathfinder.agent_paths, id) || isempty(pathfinder.agent_paths[id])

"""
    is_stationary(agent, astar::AStar)
Same, but for pathfinding with A*.
"""
Agents.is_stationary(
    agent::AbstractAgent,
    pathfinder::AStar,
) = isempty(agent.id, pathfinder)

"""
    Pathfinding.penaltymap(pathfinder)
Return the penalty map of a [`Pathfinding.AStar`](@ref) if the
[`Pathfinding.PenaltyMap`](@ref) metric is in use, `nothing` otherwise.

It is possible to mutate the map directly, for example
`Pathfinding.penaltymap(pathfinder)[15, 40] = 115`
or `Pathfinding.penaltymap(pathfinder) .= rand(50, 50)`. If this is mutated,
a new path needs to be planned using [`plan_route!`](@ref).
"""
function penaltymap(pathfinder::AStar)
    if pathfinder.cost_metric isa PenaltyMap
        return pathfinder.cost_metric.pmap
    else
        return nothing
    end
end

"""
    Pathfinding.remove_agent!(agent, model, pathfinder)

The same as `remove_agent!(agent, model)`, but also removes the agent's path data
from `pathfinder`.
"""
function Agents.remove_agent!(agent::AbstractAgent, model::ABM, pathfinder::AStar)
    delete!(pathfinder.agent_paths, agent.id)
    Agents.remove_agent_from_model!(agent, model)
    Agents.remove_agent_from_space!(agent, model)
end


# ---------------------------
# D* Lite implementation
# ---------------------------
# (paste this at the end of astar.jl)

# Χρησιμοποιεί υπάρχουσες εξαρτήσεις του Pathfinding module:
# DataStructures, Agents, Agents.Pathfinding, LinearAlgebra

# ---- Key για το priority queue ----
struct DKey
    k1::Float64
    k2::Float64
end
Base.isless(a::DKey,b::DKey) = a.k1 < b.k1 || (isapprox(a.k1,b.k1) && a.k2 < b.k2)
Base.:(==)(a::DKey,b::DKey) = isapprox(a.k1,b.k1) && isapprox(a.k2,b.k2)

# ---- Min-heap wrapper (PriorityQueue) ----
# PriorityQueue uses highest priority as max; we rely σε Base.isless(DKey)
const DPriorityQ = PriorityQueue{Tuple{Int,Int}, DKey}

# ---- Per-agent D* state ----
mutable struct AgentDS
    rows::Int
    cols::Int
    s_start::Tuple{Int,Int}
    s_goal::Tuple{Int,Int}
    k_m::Float64
    g::Array{Float64,2}
    rhs::Array{Float64,2}
    openq::DPriorityQ
    heuristic::Function
    sensed_walkmap::BitArray{2}
    new_edges_and_old_costs::Union{Nothing,Any}
    AgentDS(rows,cols,s_start,s_goal; heuristic=(u,v)->hypot(u[1]-v[1], u[2]-v[2])) = begin
        g = fill(Inf, rows, cols)
        rhs = fill(Inf, rows, cols)
        openq = DPriorityQ()
        sensed = trues(rows, cols)
        AgentDS(rows,cols,s_start,s_goal,0.0,g,rhs,openq,heuristic,sensed,nothing)
    end
end

# ---- Top-level DStarLite pathfinder τύπος (όμοιος με AStar) ----
mutable struct DStarLite{D,P,M,T,C} <: GridPathfinder{D,P,M}
    agent_paths::Dict{Int,Path{D,T}}
    dims::NTuple{D,T}
    neighborhood::Vector{CartesianIndex{D}}
    admissibility::Float64
    walkmap::BitArray{D}
    cost_metric::C
    function DStarLite{D,P,M,T,C}(agent_paths, dims, neighborhood, admissibility, walkmap, cost_metric) where {D,P,M,T,C}
        new(agent_paths, dims, neighborhood, admissibility, walkmap, cost_metric)
    end
end

# constructor analog to AStar
function DStarLite(dims::NTuple{2,Int}; walkmap::BitArray{2}=trues(dims), cost_metric=nothing)
    neighborhood = Agents.Pathfinding.moore_neighborhood(2)
    return DStarLite{2,typeof(false),true,Int,typeof(cost_metric)}(
        Dict{Int,Path{2,Int}}(),
        dims,
        neighborhood,
        0.0,
        walkmap,
        cost_metric,
    )
end

# penaltymap accessor (parallels AStar.penaltymap)
function penaltymap(pathfinder::DStarLite)
    if pathfinder.cost_metric === nothing
        return nothing
    elseif hasproperty(pathfinder.cost_metric, :pmap)
        return pathfinder.cost_metric.pmap
    elseif hasproperty(pathfinder.cost_metric, :penalty_map)
        return pathfinder.cost_metric.penalty_map
    else
        return nothing
    end
end

# ---------- D* core helpers ----------
_calc_key(ds::AgentDS, s::Tuple{Int,Int}) = begin
    i,j = s
    gval = ds.g[i,j]
    rhsval = ds.rhs[i,j]
    k1 = min(gval, rhsval) + ds.heuristic(ds.s_start, s) + ds.k_m
    k2 = min(gval, rhsval)
    return DKey(k1, k2)
end

function _succ(ds::AgentDS, s::Tuple{Int,Int}; avoid_obstacles=true)
    i,j = s
    rows,cols = ds.rows, ds.cols
    out = Tuple{Int,Int}[]
    for di in -1:1, dj in -1:1
        if di==0 && dj==0; continue; end
        ii = i+di; jj = j+dj
        if 1 <= ii <= rows && 1 <= jj <= cols
            if !avoid_obstacles || ds.sensed_walkmap[ii,jj]
                push!(out, (ii,jj))
            end
        end
    end
    return out
end

function _c(ds::AgentDS, u::Tuple{Int,Int}, v::Tuple{Int,Int})
    # If either endpoint not walkable in sensed map treat as Inf
    if !ds.sensed_walkmap[u...] || !ds.sensed_walkmap[v...]
        return Inf
    end
    return hypot(u[1]-v[1], u[2]-v[2])
end

# push to open queue with DKey
_push_open!(ds::AgentDS, node::Tuple{Int,Int}) = begin
    k = _calc_key(ds, node)
    # PriorityQueue: priority value; if key exists, replace
    ds.openq[node] = k
end

# pop min (we need the smallest k according to isless)
function _pop_open!(ds::AgentDS)
    # PriorityQueue.dequeue gives element with highest priority by default;
    # we inverted ordering by relying on isless of DKey (lowest will be "largest" ???)
    # To avoid confusion, we will extract all keys and select the minimal according to isless.
    # (This is suboptimal but simple and robust.)
    if isempty(ds.openq)
        return nothing
    end
    # find minimal key/node
    minnode = nothing
    minkey = nothing
    for (n,k) in ds.openq
        if minnode === nothing || isless(k, minkey)
            minnode = n
            minkey = k
        end
    end
    # remove minnode
    delete!(ds.openq, minnode)
    return minnode, minkey
end

# update_vertex!
function _update_vertex!(ds::AgentDS, u::Tuple{Int,Int})
    i,j = u
    if ds.g[i,j] != ds.rhs[i,j]
        _push_open!(ds, u)
    else
        # if equal: ensure not in pq (lazy removal not implemented)
        if haskey(ds.openq, u)
            delete!(ds.openq, u)
        end
    end
end

# compute_shortest_path! (simplified/robust)
function compute_shortest_path!(ds::AgentDS; max_iters=200000)
    # initialize: ensure goal has rhs 0 and is in open
    gi,gj = ds.s_goal
    ds.rhs[gi,gj] = 0.0
    _push_open!(ds, ds.s_goal)

    iters = 0
    while !isempty(ds.openq)
        iters += 1
        if iters > max_iters
            @warn "D* compute_shortest_path!: max iters reached"
            break
        end

        topnode, topk = _pop_open!(ds)
        if topnode === nothing
            break
        end
        k_old = topk
        k_new = _calc_key(ds, topnode)

        # if key changed, reinsert
        if isless(k_old, k_new)
            ds.openq[topnode] = k_new
            continue
        end

        ui,uj = topnode
        if ds.g[ui,uj] > ds.rhs[ui,uj]
            ds.g[ui,uj] = ds.rhs[ui,uj]
            for s in _succ(ds, topnode, avoid_obstacles=true)
                if s != ds.s_goal
                    si,sj = s
                    ds.rhs[si,sj] = min(ds.rhs[si,sj], _c(ds, s, topnode) + ds.g[ui,uj])
                end
                _update_vertex!(ds, s)
            end
        else
            g_old = ds.g[ui,uj]
            ds.g[ui,uj] = Inf
            tofix = _succ(ds, topnode, avoid_obstacles=true)
            push!(tofix, topnode)
            for s in tofix
                si,sj = s
                # recompute rhs for s if necessary
                if ds.rhs[si,sj] == _c(ds, s, topnode) + g_old
                    if s != ds.s_goal
                        min_s = Inf
                        for s2 in _succ(ds, s, avoid_obstacles=true)
                            min_s = min(min_s, _c(ds, s, s2) + ds.g[s2...])
                        end
                        ds.rhs[si,sj] = min_s
                    end
                end
                _update_vertex!(ds, s)
            end
        end

        # termination condition:
        if isempty(ds.openq)
            break
        end
        # if top key >= start key and start.rhs == start.g -> done
        if !isempty(ds.openq)
            # compute start key and compare
            startk = _calc_key(ds, ds.s_start)
            # get minimal key in queue:
            mink = nothing
            for (_,k) in ds.openq
                mink = mink === nothing ? k : (isless(k,mink) ? k : mink)
            end
            if !(mink !== nothing && isless(mink, startk)) && ds.rhs[ds.s_start...] == ds.g[ds.s_start...]
                break
            end
        end
    end

    return
end

# build a path (list of grid nodes) from s_start to s_goal (greedy follow)
function build_path_from_ds(ds::AgentDS)
    path = Path{2,Int64}()
    cur = ds.s_start
    push!(path, cur)
    while cur != ds.s_goal
        succs = _succ(ds, cur, avoid_obstacles=true)
        best = nothing
        bestval = Inf
        for s in succs
            val = _c(ds, cur, s) + ds.g[s...]
            if val < bestval
                bestval = val
                best = s
            end
        end
        if best === nothing || best == cur
            break
        end
        push!(path, best)
        cur = best
        if length(path) > ds.rows*ds.cols
            @warn "build_path_from_ds: stuck, aborting"
            break
        end
    end
    return path
end

# get next step only
function get_next_step(ds::AgentDS)
    cur = ds.s_start
    succs = _succ(ds, cur, avoid_obstacles=true)
    best = nothing; bestval = Inf
    for s in succs
        val = _c(ds, cur, s) + ds.g[s...]
        if val < bestval
            bestval = val; best = s
        end
    end
    return best
end

# handle_map_changes!: accept vector of (cell, nothing) or (cell, edges_old)
function handle_map_changes!(ds::AgentDS, vc)
    if vc === nothing
        return
    end
    ds.new_edges_and_old_costs = vc
    # For now we assume caller updated ds.sensed_walkmap externally.
    # We will just recompute k_m and run compute_shortest_path! (naive)
    # (A proper incremental implementation would use the edges_and_old info.)
    # Update k_m for agent movement: not tracked here; left simple.
    compute_shortest_path!(ds)
end

# ---------- Interface with Agents.jl ----------
# plan_best_route! for ContinuousSpace (use to_discrete_position helper)
function Agents.plan_best_route!(
    agent::AbstractAgent,
    dests,
    pathfinder::DStarLite{D,P,M,Int};
    condition::Symbol = :shortest,
) where {D,P,M}
    @assert condition ∈ (:shortest, :longest)
    compare = condition == :shortest ? (a, b) -> a < b : (a, b) -> a > b
    best_path = Path{D,Int64}()
    best_target = nothing
    for target in dests
        # convert to discrete grid coords using existing helper
        discrete_from = Tuple(to_discrete_position(agent.pos, pathfinder))
        discrete_to   = Tuple(to_discrete_position(target, pathfinder))
        # quick reachability test: both walkable?
        if !all(1 .<= discrete_from .<= size(pathfinder.walkmap)) || !all(1 .<= discrete_to .<= size(pathfinder.walkmap))
            continue
        end
        # create temporary AgentDS to compute path for this target (cheap check)
        ds_temp = AgentDS(size(pathfinder.walkmap,1), size(pathfinder.walkmap,2), discrete_from, discrete_to)
        ds_temp.sensed_walkmap .= pathfinder.walkmap
        compute_shortest_path!(ds_temp)
        p = build_path_from_ds(ds_temp)
        isnothing(p) && continue
        if isempty(best_path) || compare(length(p), length(best_path))
            best_path = p
            best_target = target
        end
    end

    isnothing(best_target) && return
    pathfinder.agent_paths[agent.id] = best_path

    # attach per-agent DS instance for future incremental updates
    ds = AgentDS(size(pathfinder.walkmap,1), size(pathfinder.walkmap,2),
                 Tuple(to_discrete_position(agent.pos, pathfinder)),
                 Tuple(to_discrete_position(best_target, pathfinder)))
    ds.sensed_walkmap .= pathfinder.walkmap
    compute_shortest_path!(ds)

    try
        # store on agent if field exists
        agent.dstar = ds
    catch
        # ignore if agent has no :dstar field
    end

    return best_target
end

# plan_route! (single dest)
function Agents.plan_route!(
    agent::AbstractAgent,
    dest,
    pathfinder::DStarLite{D,P,M,Float64},
) where {D,P,M}
    # convert to discrete grid coords
    discrete_from = Tuple(to_discrete_position(agent.pos, pathfinder))
    discrete_to   = Tuple(to_discrete_position(dest, pathfinder))

    ds = AgentDS(size(pathfinder.walkmap,1), size(pathfinder.walkmap,2), discrete_from, discrete_to)
    ds.sensed_walkmap .= pathfinder.walkmap
    compute_shortest_path!(ds)
    pathfinder.agent_paths[agent.id] = build_path_from_ds(ds)

    try
        agent.dstar = ds
    catch
    end
    return
end

# move_along_route! for ContinuousSpace with DStarLite
function Agents.move_along_route!(
    agent::AbstractAgent,
    model::ABM{<:ContinuousSpace{D}},
    pathfinder::DStarLite{D},
    speed::Float64,
    dt::Real = 1.0,
) where {D}
    # If agent has a precomputed continuous path stored in pathfinder.agent_paths -> use it
    isempty(agent.id, pathfinder) && begin
        # if no path but agent has dstar with ds.s_start/s_goal, step greedily
        if hasfield(agent, :dstar) && agent.dstar !== nothing
            ds = agent.dstar
            # convert continuous position to grid start
            ds.s_start = Tuple(to_discrete_position(agent.pos, pathfinder))
            next_cell = get_next_step(ds)
            if next_cell === nothing
                return
            end
            # move continuously toward center of next_cell
            target_cts = to_continuous_position(next_cell, pathfinder)
            dir = (target_cts .- agent.pos)
            dist = norm(dir)
            if dist == 0.0
                ds.s_start = next_cell
                return
            end
            maxstep = speed * dt
            if dist <= maxstep
                move_agent!(agent, target_cts, model)
                ds.s_start = next_cell
            else
                move_agent!(agent, agent.pos .+ dir ./ dist .* maxstep, model)
            end
        end
        return
    end

    # Otherwise if continuous path exists (from plan_route!) use same logic as AStar move_along_route!
    from = agent.pos
    next_pos = agent.pos
    T = typeof(agent.pos)
    while true
        next_waypoint = T(first(pathfinder.agent_paths[agent.id]))
        dir = get_direction(from, next_waypoint, model)
        dist_to_target = norm(dir)
        if dist_to_target ≈ 0.
            from = next_waypoint
            popfirst!(pathfinder.agent_paths[agent.id])
            if isempty(agent.id, pathfinder)
                next_pos = next_waypoint
                break
            end
            continue
        end
        dir = dir ./ dist_to_target
        next_pos = from .+ dir .* (speed * dt)
        next_pos = Agents.normalize_position(T(next_pos), model)
        dist_to_next = euclidean_distance(T(from), T(next_pos), model)
        if dist_to_next > dist_to_target
            from = next_waypoint
            dt -= dist_to_target / speed
            popfirst!(pathfinder.agent_paths[agent.id])
            if isempty(agent.id, pathfinder)
                next_pos = next_waypoint
                break
            end
        else
            break
        end
    end
    move_agent!(agent, T(next_pos), model)
end

# remove agent! overload for DStarLite
function Agents.remove_agent!(agent::AbstractAgent, model::ABM, pathfinder::DStarLite)
    delete!(pathfinder.agent_paths, agent.id)
    try
        if hasfield(agent, :dstar)
            agent.dstar = nothing
        end
    catch
    end
    Agents.remove_agent_from_model!(agent, model)
    Agents.remove_agent_from_space!(agent, model)
end

# end D* block
# ---------------------------