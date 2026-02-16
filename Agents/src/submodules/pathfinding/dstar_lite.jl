export DStarLite, find_path, DStarLitePlanner, init_planner, update_after_cm_change!, extract_path
# D* Lite pathfinder for Agents.jl GridSpace / ContinuousSpace

struct DStarLite{D,P,M,T,C<:CostMetric{D}} <: GridPathfinder{D,P,M}
    agent_paths::Dict{Int,Path{D,T}}
    dims::NTuple{D,T}
    neighborhood::Vector{CartesianIndex{D}}
    admissibility::Float64      # κρατιέται για συμβατότητα, δεν χρησιμοποιείται ιδιαίτερα εδώ
    walkmap::BitArray{D}
    cost_metric::C

    function DStarLite{D,P,M,T,C}(
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
        elseif cost_metric isa AbsolutePenaltyMap{D}
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
    DStarLite(dims::NTuple{D,T}; kwargs...)

Δημιουργεί pathfinder τύπου D* Lite για grid `dims`.
"""
function DStarLite(
    dims::NTuple{D,T};
    periodic::Union{Bool,NTuple{D,Bool}} = false,
    diagonal_movement::Bool = true,
    admissibility::Float64 = 0.0,
    walkmap::BitArray{D} = trues(dims),
    cost_metric::CostMetric{D} = DirectDistance{D}(),
) where {D,T}
    neighborhood = diagonal_movement ? moore_neighborhood(D) : vonneumann_neighborhood(D)
    return DStarLite{D,periodic,diagonal_movement,T,typeof(cost_metric)}(
        Dict{Int,Path{D,T}}(),
        dims,
        neighborhood,
        admissibility,
        walkmap,
        cost_metric,
    )
end

DStarLite(
    space::GridSpace{D,periodic};
    diagonal_movement::Bool = true,
    admissibility::Float64 = 0.0,
    walkmap::BitArray{D} = trues(spacesize(space)),
    cost_metric::CostMetric{D} = DirectDistance{D}(),
) where {D,periodic} = DStarLite(size(space); periodic, diagonal_movement, admissibility, walkmap, cost_metric)

function DStarLite(
    space::ContinuousSpace{D,periodic};
    walkmap::Union{BitArray{D},Nothing} = nothing,
    admissibility::Float64 = 0.0,
    cost_metric::CostMetric{D} = DirectDistance{D}(),
) where {D,periodic}
    @assert walkmap isa BitArray{D} || cost_metric isa PenaltyMap || cost_metric isa AbsolutePenaltyMap "Pathfinding in ContinuousSpace requires either walkmap to be specified or cost_metric to be a PenaltyMap or AbsolutePenaltyMap"
    isnothing(walkmap) && (walkmap = BitArray(trues(size(cost_metric.pmap))))
    DStarLite(
        Tuple(Agents.spacesize(space));
        periodic,
        diagonal_movement = true,
        admissibility,
        walkmap,
        cost_metric,
    )
end

function Base.show(io::IO, pathfinder::DStarLite{D,P,M}) where {D,P,M}
    periodic = get_periodic_type(pathfinder)
    moore = M ? "diagonal, " : "orthogonal, "
    s = "D* Lite in $(D) dimensions, $(periodic)$(moore)ϵ=$(pathfinder.admissibility), " *
        "metric=$(pathfinder.cost_metric)"
    print(io, s)
end

get_periodic_type(::DStarLite{D,false,M}) where {D,M} = ""
get_periodic_type(::DStarLite{D,true,M}) where {D,M} = "periodic, "
get_periodic_type(::DStarLite{D,P,M}) where {D,P,M} = "mixed periodicity, "

# ---------------- D* Lite core: find_path ----------------

"""
    find_path(pathfinder::DStarLite{D}, from::Dims{D}, to::Dims{D})

Υπολογίζει διαδρομή `from → to` με τον αλγόριθμο D* Lite.
Επιστρέφει `Path{D,Int}` ή `nothing` αν δεν υπάρχει διαδρομή.
"""
function find_path(pathfinder::DStarLite{D}, from::Dims{D}, to::Dims{D}) where {D}
    # bounds / walkable όπως στο AStar
    if !all(1 .<= from .<= size(pathfinder.walkmap)) ||
       !all(1 .<= to .<= size(pathfinder.walkmap)) ||
       !pathfinder.walkmap[from...] || !pathfinder.walkmap[to...]
        return
    end

    # const INF = Inf

    g   = Dict{Dims{D},Float64}()
    rhs = Dict{Dims{D},Float64}()

    s_start = from
    s_goal  = to
    km = 0.0

    U = PriorityQueue{Dims{D},Tuple{Float64,Float64}}()

    getg(s)   = get(g,   s, Inf)
    getrhs(s) = get(rhs, s, Inf)
    setg!(s, v)   = (g[s]   = v)
    setrhs!(s, v) = (rhs[s] = v)

    h(s) = delta_cost(pathfinder, s, s_start)  # heuristic μέχρι το start

    function calc_key(s)
        gs  = getg(s)
        rs  = getrhs(s)
        m   = min(gs, rs)
        return (m + h(s) + km, m)
    end

    key_lt(a::Tuple{T,T}, b::Tuple{T,T}) where {T<:Real} =
        (a[1] < b[1]) || (a[1] == b[1] && a[2] < b[2])

    function update_vertex!(u)
        # rhs(u) = min_{s' in Succ(u)} (c(u,s') + g(s'))   για u ≠ goal
        if u != s_goal
            min_rhs = Inf
            for sp in walkable_neighbors(u, pathfinder)  # Succ(u)
                val = delta_cost(pathfinder, u, sp) + getg(sp)
                if val < min_rhs
                    min_rhs = val
                end
            end
            setrhs!(u, min_rhs)
        end

        # Remove u from queue if present, then reinsert if inconsistent
        if haskey(U, u)
            delete!(U, u)
        end
        if getg(u) != getrhs(u)
            U[u] = calc_key(u)
        end
        return nothing
    end

    # initialization
    setrhs!(s_goal, 0.0)
    U[s_goal] = calc_key(s_goal)

    function top_key()
        if isempty(U)
            return (Inf, Inf), nothing
        else
            pair = peek(U)              # Pair{Dims{D}, Tuple{Float64,Float64}}
            u    = pair.first           # το key = θέση
            k    = pair.second          # η προτεραιότητα (k1,k2)
            return k, u
        end
    end

    function compute_shortest_path!()
        while true
            k_start = calc_key(s_start)
            (k_top, u) = top_key()
            u === nothing && break

            # while (U.TopKey < CalculateKey(s_start)) OR (rhs(s_start) != g(s_start))
            if !(key_lt(k_top, k_start) || getrhs(s_start) != getg(s_start))
                break
            end

            pair  = peek(U)
            u     = pair.first
            k_old = pair.second
            dequeue!(U)

            k_new = calc_key(u)

            if key_lt(k_old, k_new)
                # U.Insert(u, CalculateKey(u))
                U[u] = k_new

            elseif getg(u) > getrhs(u)
                # g(u) = rhs(u)
                setg!(u, getrhs(u))

                # for all s in Pred(u): UpdateVertex(s)
                for s in walkable_neighbors(u, pathfinder)   # Pred(u)
                    update_vertex!(s)
                end

            else
                # g(u) = ∞
                setg!(u, Inf)

                # UpdateVertex(u); for all s in Pred(u): UpdateVertex(s)
                update_vertex!(u)
                for s in walkable_neighbors(u, pathfinder)   # Pred(u)
                    update_vertex!(s)
                end
            end
        end
    end

    compute_shortest_path!()

    if getg(s_start) == Inf
        return
    end

    # Ανακατασκευή διαδρομής: greedy, όπως στο κλασικό D* Lite
    agent_path = Path{D,Int64}()
    cur = s_start

    while cur != s_goal
        best = nothing
        best_cost = Inf
        for n in walkable_neighbors(cur, pathfinder)
            cost = delta_cost(pathfinder, cur, n) + getg(n)
            if cost < best_cost
                best_cost = cost
                best = n
            end
        end

        best === nothing && return   # δεν βρέθηκε συνέχεια

        push!(agent_path, best)
        cur = best
    end

    return agent_path
end

# ---------------- Grid helpers (ίδια API με AStar) ----------------

"""
    Agents.plan_route!(agent, dest, pathfinder::DStarLite{D})

Υπολογίζει και αποθηκεύει την διαδρομή `agent.pos → dest` με D* Lite.
"""
function Agents.plan_route!(
    agent::AbstractAgent,
    dest,
    pathfinder::DStarLite{D},
) where {D}
    path = find_path(pathfinder, agent.pos, dest)
    isnothing(path) && return
    pathfinder.agent_paths[Agents.getid(agent)] = path
end

"""
    Agents.plan_best_route!(agent, dests, pathfinder::DStarLite{D}; condition=:shortest)

Όπως για AStar, αλλά χρησιμοποιεί D* Lite για κάθε υποψήφιο προορισμό.
"""
function Agents.plan_best_route!(
    agent::AbstractAgent,
    dests,
    pathfinder::DStarLite{D,P,M,Int64};
    condition::Symbol = :shortest,
) where {D,P,M}
    @assert condition ∈ (:shortest, :longest)
    compare = condition == :shortest ? (a, b) -> a < b : (a, b) -> a > b

    best_path   = Path{D,Int64}()
    best_target = nothing

    for target in dests
        path = find_path(pathfinder, agent.pos, target)
        isnothing(path) && continue

        if isempty(best_path) || compare(length(path), length(best_path))
            best_path   = path
            best_target = target
        end
    end

    isnothing(best_target) && return
    pathfinder.agent_paths[Agents.getid(agent)] = best_path
    return best_target
end

"""
    Agents.move_along_route!(agent, model::ABM{<:GridSpace{D}}, pathfinder::DStarLite{D})

Κινεί τον agent ένα βήμα κατά μήκος της αποθηκευμένης διαδρομής.
"""
function Agents.move_along_route!(
    agent::AbstractAgent,
    model::ABM{<:GridSpace{D}},
    pathfinder::DStarLite{D},
) where {D}
    isempty(Agents.getid(agent), pathfinder) && return
    move_agent!(agent, first(pathfinder.agent_paths[Agents.getid(agent)]), model)
    popfirst!(pathfinder.agent_paths[Agents.getid(agent)])
end

"""
    Pathfinding.nearby_walkable(position, model, pathfinder::DStarLite{D}, r)

Ίδιο με την εκδοχή για AStar.
"""
nearby_walkable(
    position,
    model::ABM{<:GridSpace{D}},
    pathfinder::DStarLite{D},
    r = 1,
) where {D} = Iterators.filter(
    x -> pathfinder.walkmap[x...] == 1,
    nearby_positions(position, model, r),
)

"""
    Pathfinding.random_walkable(model, pathfinder::DStarLite{D})

Τυχαία walkable θέση, ίδιο API με AStar.
"""
function random_walkable(model::ABM{<:GridSpace{D}}, pathfinder::DStarLite{D}) where {D}
    return Tuple(
        rand(
            abmrng(model),
            filter(x -> pathfinder.walkmap[x], CartesianIndices(abmspace(model).stored_ids)),
        ),
    )
end

"""
    Pathfinding.penaltymap(pathfinder::DStarLite)

Ό,τι και για AStar.
"""
function penaltymap(pathfinder::DStarLite)
    if pathfinder.cost_metric isa PenaltyMap
        return pathfinder.cost_metric.pmap
    elseif pathfinder.cost_metric isa AbsolutePenaltyMap
        return pathfinder.cost_metric.pmap
    else
        return nothing
    end
end

"""
    Agents.remove_agent!(agent, model, pathfinder::DStarLite)

Σβήνει και τα path δεδομένα του agent.
"""
function Agents.remove_agent!(agent::AbstractAgent, model::ABM, pathfinder::DStarLite)
    delete!(pathfinder.agent_paths, Agents.getid(agent))
    Agents.remove_agent_from_container!(agent, model)
    Agents.remove_agent_from_space!(agent, model)
end

Base.isempty(id::Int, pathfinder::DStarLite) =
    !haskey(pathfinder.agent_paths, id) || isempty(pathfinder.agent_paths[id])

Agents.is_stationary(agent::AbstractAgent, pathfinder::DStarLite) =
    isempty(Agents.getid(agent), pathfinder)




# ---------------- D* Lite planner for dynamic cost maps ----------------
# This section implements an incremental planner that reuses D* Lite state
# across cost-map changes. The original version below is heuristic–free
# (keys depend only on g/rhs + km). We extend it further down with an
# optional heuristic term (Manhattan / Euclidean / delta_cost–based),
# exposed via an overloaded `init_planner` method.
#
# Sources (D* Lite & heuristics):
# - Sven Koenig, Maxim Likhachev, “D* Lite”, AAAI 2002; JAIR 22 (2004), 467–508.
# - Peter E. Hart, Nils J. Nilsson, Bertram Raphael, “A Formal Basis for the
#   Heuristic Determination of Minimum Cost Paths”, IEEE TSSC, 1968.
# - Stuart Russell, Peter Norvig, “Artificial Intelligence: A Modern
#   Approach”, 3rd ed., 2009, ch. 3 (uninformed & informed search).
mutable struct DStarLitePlanner{D}
    pf    :: DStarLite{D}
    g     :: Matrix{Float64}
    rhs   :: Matrix{Float64}
    U     :: PriorityQueue{Tuple{Int,Int}, Tuple{Float64,Float64}}
    km    :: Float64
    goal  :: Tuple{Int,Int}
end


@inline function calc_key(pl::DStarLitePlanner, s::Tuple{Int,Int})
    m = min(pl.g[s...], pl.rhs[s...])
    return (m + pl.km, m)
end


function update_vertex!(pl::DStarLitePlanner, u::Tuple{Int,Int})
    if u != pl.goal
        min_rhs = Inf
        for v in walkable_neighbors(u, pl.pf)
            c = delta_cost(pl.pf, u, v) + pl.g[v...]
            min_rhs = min(min_rhs, c)
        end
        pl.rhs[u...] = min_rhs
    end

    haskey(pl.U, u) && delete!(pl.U, u)

    if pl.g[u...] != pl.rhs[u...]
        pl.U[u] = calc_key(pl, u)
    end
end


function compute_shortest_path!(pl::DStarLitePlanner)
    while !isempty(pl.U)
        (u, k_old) = peek(pl.U)
        k_new = calc_key(pl, u)

        if k_old < k_new
            pl.U[u] = k_new
        elseif pl.g[u...] > pl.rhs[u...]
            pl.g[u...] = pl.rhs[u...]
            dequeue!(pl.U)
            for v in walkable_neighbors(u, pl.pf)
                update_vertex!(pl, v)
            end
        else
            pl.g[u...] = Inf
            dequeue!(pl.U)
            update_vertex!(pl, u)
            for v in walkable_neighbors(u, pl.pf)
                update_vertex!(pl, v)
            end
        end
    end
end



function init_planner(pf::DStarLite{2}, goal::Tuple{Int,Int})
    sz = size(pf.walkmap)

    g   = fill(Inf, sz)
    rhs = fill(Inf, sz)

    rhs[goal...] = 0.0

    U = PriorityQueue{Tuple{Int,Int}, Tuple{Float64,Float64}}()
    planner = DStarLitePlanner(pf, g, rhs, U, 0.0, goal)

    U[goal] = calc_key(planner, goal)
    compute_shortest_path!(planner)

    return planner
end


function update_after_cm_change!(
    planner::DStarLitePlanner,
    changed_cells::Vector{Tuple{Int,Int}}
)
    for u in changed_cells
        update_vertex!(planner, u)
        for v in walkable_neighbors(u, planner.pf)
            update_vertex!(planner, v)
        end
    end

    compute_shortest_path!(planner)
end


########################
# Heuristic-aware planner
########################

# A drop-in variant of `DStarLitePlanner` that adds a configurable heuristic
# term to the key calculation. This enables:
#   - :delta_cost  → reuse Agents.jl's `delta_cost` as heuristic w.r.t. goal
#   - :manhattan   → |dx| + |dy|
#   - :euclidean   → sqrt(dx^2 + dy^2)
#
# The original (non-heuristic) planner is left untouched so existing code
# keeps its current behavior. This variant is constructed via an overloaded
# `init_planner(pf, goal, heuristic::Symbol)` method.

mutable struct DStarLitePlannerHeuristic{D}
    pf    :: DStarLite{D}
    g     :: Matrix{Float64}
    rhs   :: Matrix{Float64}
    U     :: PriorityQueue{Tuple{Int,Int}, Tuple{Float64,Float64}}
    km    :: Float64
    goal  :: Tuple{Int,Int}
    heuristic_kind :: Symbol   # :delta_cost, :manhattan, :euclidean, :none
end

@inline function heuristic(pl::DStarLitePlannerHeuristic, s::Tuple{Int,Int})
    hk = pl.heuristic_kind
    if hk === :delta_cost
        # Use the same cost metric as A*/D* Lite core, evaluated w.r.t. goal.
        # This mirrors the literature practice of basing h on the edge-cost
        # metric (cf. Hart–Nilsson–Raphael 1968; Koenig–Likhachev 2002).
        return delta_cost(pl.pf, s, pl.goal)
    elseif hk === :manhattan
        dx = abs(s[1] - pl.goal[1])
        dy = abs(s[2] - pl.goal[2])
        return dx + dy
    elseif hk === :euclidean
        dx = s[1] - pl.goal[1]
        dy = s[2] - pl.goal[2]
        return sqrt(dx * dx + dy * dy)
    else
        # :none or any unsupported symbol → behave like the original planner
        return 0.0
    end
end

@inline function calc_key(pl::DStarLitePlannerHeuristic, s::Tuple{Int,Int})
    m = min(pl.g[s...], pl.rhs[s...])
    return (m + heuristic(pl, s) + pl.km, m)
end

function update_vertex!(pl::DStarLitePlannerHeuristic, u::Tuple{Int,Int})
    if u != pl.goal
        min_rhs = Inf
        for v in walkable_neighbors(u, pl.pf)
            c = delta_cost(pl.pf, u, v) + pl.g[v...]
            min_rhs = min(min_rhs, c)
        end
        pl.rhs[u...] = min_rhs
    end

    haskey(pl.U, u) && delete!(pl.U, u)

    if pl.g[u...] != pl.rhs[u...]
        pl.U[u] = calc_key(pl, u)
    end
end

function compute_shortest_path!(pl::DStarLitePlannerHeuristic)
    while !isempty(pl.U)
        (u, k_old) = peek(pl.U)
        k_new = calc_key(pl, u)

        if k_old < k_new
            pl.U[u] = k_new
        elseif pl.g[u...] > pl.rhs[u...]
            pl.g[u...] = pl.rhs[u...]
            dequeue!(pl.U)
            for v in walkable_neighbors(u, pl.pf)
                update_vertex!(pl, v)
            end
        else
            pl.g[u...] = Inf
            dequeue!(pl.U)
            update_vertex!(pl, u)
            for v in walkable_neighbors(u, pl.pf)
                update_vertex!(pl, v)
            end
        end
    end
end

"""
    init_planner(pf::DStarLite{2}, goal; heuristic = :delta_cost)

Heuristic-aware planner constructor. Creates a `DStarLitePlannerHeuristic`
using:

- `heuristic = :delta_cost`  → use `delta_cost(pf, s, goal)` as heuristic
- `heuristic = :manhattan`   → Manhattan distance to goal
- `heuristic = :euclidean`   → Euclidean distance to goal
- `heuristic = :none`        → equivalent to the original planner

This is a drop-in alternative to the existing `init_planner(pf, goal)` and
does not modify any existing behavior.
"""
function init_planner(
    pf::DStarLite{2},
    goal::Tuple{Int,Int},
    heuristic::Symbol;
) 
    sz = size(pf.walkmap)

    g   = fill(Inf, sz)
    rhs = fill(Inf, sz)
    rhs[goal...] = 0.0

    U = PriorityQueue{Tuple{Int,Int}, Tuple{Float64,Float64}}()
    planner = DStarLitePlannerHeuristic(pf, g, rhs, U, 0.0, goal, heuristic)

    U[goal] = calc_key(planner, goal)
    compute_shortest_path!(planner)

    return planner
end

function update_after_cm_change!(
    planner::DStarLitePlannerHeuristic,
    changed_cells::Vector{Tuple{Int,Int}},
)
    for u in changed_cells
        update_vertex!(planner, u)
        for v in walkable_neighbors(u, planner.pf)
            update_vertex!(planner, v)
        end
    end

    compute_shortest_path!(planner)
end

function extract_path(
    planner::DStarLitePlannerHeuristic,
    start::Tuple{Int,Int};
    max_steps = 10_000,
)
    path = Tuple{Int,Int}[]
    cur  = start

    push!(path, cur)

    for _ in 1:max_steps
        cur == planner.goal && break

        best = nothing
        best_val = Inf

        for n in walkable_neighbors(cur, planner.pf)
            val = delta_cost(planner.pf, cur, n) + planner.g[n...]
            if val < best_val
                best_val = val
                best = n
            end
        end

        best === nothing && break
        cur = best
        push!(path, cur)
    end

    return path
end

function extract_path(
    planner::DStarLitePlanner,
    start::Tuple{Int,Int};
    max_steps = 10_000
)
    path = Tuple{Int,Int}[]
    cur  = start

    push!(path, cur)

    for _ in 1:max_steps
        cur == planner.goal && break

        best = nothing
        best_val = Inf

        for n in walkable_neighbors(cur, planner.pf)
            val = delta_cost(planner.pf, cur, n) + planner.g[n...]
            if val < best_val
                best_val = val
                best = n
            end
        end

        best === nothing && break
        cur = best
        push!(path, cur)
    end

    return path
end