# Incremental D* Lite — optimized version (Koenig & Likhachev, AAAI 2002,
# Figure 4, see LITERATURE/DStarLite.pdf). Comments `{NN''}` refer to its lines.
#
# Two modes share one struct:
# - Focused (`s_start` set): Figure 4 as written.
# - Full-field (`s_start === nothing`): the same procedures with h ≡ 0 and
#   CalculateKey(s_start) ≡ [∞; ∞], so {10''} runs until U is empty and `g` is
#   the cost-to-go of every cell (one planner per exit shared by all agents,
#   Scenario 3). This is the "D* Lite without heuristic search" of Figs. 6–8.

const DStarStart = Union{Nothing,Tuple{Int,Int}}
const DStarKey = Tuple{Float64,Float64}

@inline function _key_lt(a::DStarKey, b::DStarKey)
    return (a[1] < b[1]) || (a[1] == b[1] && a[2] < b[2])
end

# ---------------- Indexed binary min-heap over grid cells ----------------
# Replaces DataStructures.PriorityQueue: `pos[i,j]` is the cell's slot in the
# heap (0 = absent), so membership / decrease-key need no hashing, and a key
# change is an in-place sift instead of delete + reinsert.

struct CellHeap
    cells::Vector{Tuple{Int,Int}}
    keys::Vector{DStarKey}
    pos::Matrix{Int32}
end

CellHeap(sz::Tuple{Int,Int}) = CellHeap(Tuple{Int,Int}[], DStarKey[], zeros(Int32, sz))

Base.isempty(h::CellHeap) = isempty(h.cells)
Base.length(h::CellHeap) = length(h.cells)
Base.haskey(h::CellHeap, u::Tuple{Int,Int}) = @inbounds h.pos[u[1], u[2]] != 0

"Smallest (cell, key) in the heap; the heap must not be empty."
@inline heap_top(h::CellHeap) = @inbounds (h.cells[1], h.keys[1])

@inline function _heap_swap!(h::CellHeap, i::Int, j::Int)
    @inbounds begin
        ci, cj = h.cells[i], h.cells[j]
        h.cells[i], h.cells[j] = cj, ci
        h.keys[i], h.keys[j] = h.keys[j], h.keys[i]
        h.pos[cj[1], cj[2]] = i
        h.pos[ci[1], ci[2]] = j
    end
    return nothing
end

@inline function _sift_up!(h::CellHeap, i::Int)
    @inbounds while i > 1
        p = i >> 1
        _key_lt(h.keys[i], h.keys[p]) || break
        _heap_swap!(h, i, p)
        i = p
    end
    return i
end

@inline function _sift_down!(h::CellHeap, i::Int)
    n = length(h.cells)
    @inbounds while true
        l = 2i
        l > n && break
        c = (l < n && _key_lt(h.keys[l+1], h.keys[l])) ? l + 1 : l
        _key_lt(h.keys[c], h.keys[i]) || break
        _heap_swap!(h, i, c)
        i = c
    end
    return i
end

"Insert `u` with key `k`, or change its key in place if already present."
function Base.setindex!(h::CellHeap, k::DStarKey, u::Tuple{Int,Int})
    i = Int(@inbounds h.pos[u[1], u[2]])
    if i == 0
        push!(h.cells, u)
        push!(h.keys, k)
        i = length(h.cells)
        @inbounds h.pos[u[1], u[2]] = i
        _sift_up!(h, i)
    else
        @inbounds old = h.keys[i]
        @inbounds h.keys[i] = k
        _key_lt(k, old) ? _sift_up!(h, i) : _sift_down!(h, i)
    end
    return h
end

function _heap_remove_at!(h::CellHeap, i::Int)
    n = length(h.cells)
    i != n && _heap_swap!(h, i, n)
    u = pop!(h.cells)
    pop!(h.keys)
    @inbounds h.pos[u[1], u[2]] = 0
    if i < n
        _sift_up!(h, i) == i && _sift_down!(h, i)
    end
    return u
end

"Remove `u` if present (no-op otherwise)."
function Base.delete!(h::CellHeap, u::Tuple{Int,Int})
    i = Int(@inbounds h.pos[u[1], u[2]])
    i != 0 && _heap_remove_at!(h, i)
    return h
end

"Remove and return the smallest cell."
heap_pop!(h::CellHeap) = _heap_remove_at!(h, 1)

# ---------------- Allocation-free walkable neighbors ----------------
# Same cells, in the same order, as `walkable_neighbors(u, pf)` (so tie-breaks
# and extracted paths are unchanged), but yielded lazily instead of collected
# into a fresh Vector on every call.

struct WalkableNeighbors{PF<:DStarLite}
    pf::PF
    u::Tuple{Int,Int}
end

Base.IteratorSize(::Type{<:WalkableNeighbors}) = Base.SizeUnknown()
Base.eltype(::Type{<:WalkableNeighbors}) = Tuple{Int,Int}

_periodicity(::DStarLite{D,P}) where {D,P} = P

@inline function Base.iterate(it::WalkableNeighbors, k::Int = 1)
    pf = it.pf
    nb = pf.neighborhood
    wm = pf.walkmap
    s1, s2 = size(wm)
    P = _periodicity(pf)
    p1, p2 = P isa Bool ? (P, P) : (P[1], P[2])
    u1, u2 = it.u
    @inbounds while k <= length(nb)
        β = nb[k]
        k += 1
        n1 = p1 ? mod1(u1 + β[1], s1) : u1 + β[1]
        n2 = p2 ? mod1(u2 + β[2], s2) : u2 + β[2]
        if 1 <= n1 <= s1 && 1 <= n2 <= s2 && wm[n1, n2]
            return ((n1, n2), k)
        end
    end
    return nothing
end

@inline _neighbors(pl, u::Tuple{Int,Int}) = WalkableNeighbors(pl.pf, u)

# ---------------- Planner ----------------

# `PF` is the concrete pathfinder type so `delta_cost`/`walkable_neighbors`
# dispatch statically in the hot loop (an abstract `DStarLite{D}` field would not).
mutable struct DStarLitePlanner{PF<:DStarLite}
    pf::PF
    g::Matrix{Float64}
    rhs::Matrix{Float64}
    U::CellHeap
    km::Float64
    goal::Tuple{Int,Int}
    s_start::DStarStart
    s_last::DStarStart
    heuristic_kind::Symbol
    # Penalty map the current edge costs were last read from: gives c_old in
    # {41''} when the caller has already overwritten the live penalty map.
    known_pmap::Union{Nothing,Matrix{Int}}
end

@inline function heuristic_cost(
    pl::DStarLitePlanner,
    from::Tuple{Int,Int},
    to::Tuple{Int,Int},
)
    hk = pl.heuristic_kind
    if hk === :delta_cost
        return Float64(delta_cost(pl.pf, _heuristic_metric(pl.pf.cost_metric), from, to))
    elseif hk === :manhattan
        d = position_delta(pl.pf, from, to)
        return Float64(d[1] + d[2])
    elseif hk === :euclidean
        d = position_delta(pl.pf, from, to)
        return sqrt(Float64(d[1]^2 + d[2]^2))
    else
        return 0.0
    end
end

@inline function heuristic(pl::DStarLitePlanner, s::Tuple{Int,Int})
    pl.s_start === nothing && return 0.0
    return heuristic_cost(pl, pl.s_start, s)
end

@inline function calc_key(pl::DStarLitePlanner, s::Tuple{Int,Int})
    m = min(pl.g[s...], pl.rhs[s...])
    return (m + heuristic(pl, s) + pl.km, m)
end

# min_{s' ∈ Succ(u)} (c(u,s') + g(s')), as in {27''} / {46''}.
@inline function _min_succ(pl::DStarLitePlanner, u::Tuple{Int,Int})
    m = Inf
    for v in _neighbors(pl, u)
        m = min(m, delta_cost(pl.pf, u, v) + pl.g[v...])
    end
    return m
end

# UpdateVertex {07''-09''}: only syncs u's queue membership/key with g/rhs; the
# optimized version sets rhs at the call sites. The heap applies U.Update in place.
@inline function update_vertex!(pl::DStarLitePlanner, u::Tuple{Int,Int})
    if pl.g[u...] != pl.rhs[u...]
        pl.U[u] = calc_key(pl, u)                 # {07''} / {08''}
    else
        delete!(pl.U, u)                          # {09''} (no-op if u ∉ U)
    end
    return nothing
end

# One iteration of the ComputeShortestPath loop body {11''-28''}.
function _process_top!(pl::DStarLitePlanner)
    u, k_old = heap_top(pl.U)                     # {11''} {12''}
    k_new = calc_key(pl, u)                       # {13''}
    if _key_lt(k_old, k_new)                      # {14''}
        pl.U[u] = k_new                           # {15''}
    elseif pl.g[u...] > pl.rhs[u...]              # {16''}
        pl.g[u...] = pl.rhs[u...]                 # {17''}
        delete!(pl.U, u)                          # {18''}
        for s in _neighbors(pl, u)                # {19''} Pred(u)
            if s != pl.goal                       # {20''}
                pl.rhs[s...] = min(pl.rhs[s...], delta_cost(pl.pf, s, u) + pl.g[u...])
            end
            update_vertex!(pl, s)                 # {21''}
        end
    else
        g_old = pl.g[u...]                        # {23''}
        pl.g[u...] = Inf                          # {24''}
        for s in _neighbors(pl, u)                # {25''} Pred(u)
            if pl.rhs[s...] == delta_cost(pl.pf, s, u) + g_old      # {26''}
                s != pl.goal && (pl.rhs[s...] = _min_succ(pl, s))   # {27''}
            end
            update_vertex!(pl, s)                 # {28''}
        end
        # {25''} also includes s = u. There is no edge (u,u), so c(u,u) = ∞ and
        # {26''} cannot hold: rhs(u) does not depend on g(u), only U changes.
        update_vertex!(pl, u)                     # {28''}
    end
    return nothing
end

# ComputeShortestPath {10''-28''}. U.TopKey() of an empty queue is [∞; ∞], so
# the loop also ends when U is empty (always the exit in full-field mode).
function compute_shortest_path!(pl::DStarLitePlanner)
    while !isempty(pl.U)
        s = pl.s_start
        if s !== nothing &&
           !(_key_lt(heap_top(pl.U)[2], calc_key(pl, s)) || pl.rhs[s...] > pl.g[s...])  # {10''}
            break
        end
        _process_top!(pl)
    end
    return pl
end

"""
    update_start!(planner, new_start)

"Move to s_start" {34''-35''} of a focused planner: the agent steps to
`new_start`, normally `argmin_{s'} c(s_start,s') + g(s')`. As in the paper,
neither `km` nor `s_last` changes here; they are updated only when edge cost
changes are detected ({37''-39''}, see [`update_after_cm_change!`](@ref)), and
no replanning is needed while costs stay the same (Theorem 8).
"""
function update_start!(pl::DStarLitePlanner, new_start::Tuple{Int,Int})
    pl.s_start === nothing &&
        throw(ArgumentError("update_start! needs a focused planner (init_planner with a start)"))
    pl.s_start = new_start
    return pl
end

# The paper requires h(s,s') ≤ c*(s,s') and h(s,s'') ≤ h(s,s') + h(s',s'')
# ("Heap Reordering", p. 480). All heuristic kinds are norms, so the triangle
# inequality holds, and since every edge cost is at least its base-metric cost,
# h ≤ c* follows from checking a single step in each neighborhood direction.
function _check_heuristic(pl::DStarLitePlanner)
    pl.heuristic_kind === :none && return nothing
    pl.heuristic_kind in (:delta_cost, :manhattan, :euclidean) ||
        throw(ArgumentError("unknown heuristic :$(pl.heuristic_kind)"))
    base = _heuristic_metric(pl.pf.cost_metric)
    u = pl.goal
    for β in pl.pf.neighborhood
        v = (u[1] + β[1], u[2] + β[2])
        h, c = heuristic_cost(pl, u, v), delta_cost(pl.pf, base, u, v)
        h <= c || throw(ArgumentError(
            "heuristic :$(pl.heuristic_kind) is not admissible for " *
            "$(pl.pf.cost_metric): h = $h > c = $c for the move $(β.I); " *
            "D* Lite requires h(s,s') ≤ c*(s,s')"))
    end
    return nothing
end

_known_pmap(m::Union{PenaltyMap{2},AbsolutePenaltyMap{2}}) = copy(m.pmap)
_known_pmap(::CostMetric) = nothing

# Cost of edge (u,v) under the penalty values in `pmap` (same formula as delta_cost).
_edge_cost(pf, m::PenaltyMap, pmap, u, v) =
    delta_cost(pf, m.base_metric, u, v) + abs(pmap[u...] - pmap[v...])
_edge_cost(pf, m::AbsolutePenaltyMap, pmap, u, v) =
    delta_cost(pf, m.base_metric, u, v) + abs(pmap[u...]) + abs(pmap[v...])

# c(u,v) as the planner last knew it (c_old in {41''}).
@inline _known_cost(pl::DStarLitePlanner, u, v) =
    pl.known_pmap === nothing ? delta_cost(pl.pf, u, v) :
    _edge_cost(pl.pf, pl.pf.cost_metric, pl.known_pmap, u, v)

function _make_planner(
    pf::DStarLite{2},
    goal::Tuple{Int,Int};
    start::DStarStart = nothing,
    heuristic::Symbol = :none,
)
    sz = size(pf.walkmap)
    hk = start === nothing ? :none : heuristic
    U = CellHeap(sz)                                                   # {02''}
    pl = DStarLitePlanner(pf, fill(Inf, sz), fill(Inf, sz), U, 0.0,    # {03''} {04''}
        goal, start, start, hk, _known_pmap(pf.cost_metric))           # {29''} s_last = s_start
    _check_heuristic(pl)
    pl.rhs[goal...] = 0.0                                              # {05''}
    h_goal = start === nothing ? 0.0 : heuristic_cost(pl, start, goal)
    U[goal] = (h_goal, 0.0)                                            # {06''}
    compute_shortest_path!(pl)                                         # {31''}
    return pl
end

"""
    init_planner(pf, goal)
    init_planner(pf, goal, start; heuristic = :delta_cost)
    init_planner(pf, goal, heuristic::Symbol)
    init_planner(pf, goal; start = nothing, heuristic = :none)

Build a D* Lite planner aimed at `goal`.

This runs Initialize() and the first ComputeShortestPath() ({29''-31''}).

With no `start`, the queue is emptied so `g` is a full cost-to-go field
(Scenario 3 / many agents sharing an exit). With `start`, search is focused
and `heuristic` is `h(start, s)`:

- `:delta_cost`: base-metric distance (penalty maps only add non-negative
  terms, so they are left out; with `MaxDistance` this is the heuristic of
  the paper's experiments)
- `:manhattan` / `:euclidean`: grid distance from `start` to `s`
- `:none`: `h = 0`

An `ArgumentError` is thrown if the heuristic is not admissible for the cost
metric, e.g. `:manhattan` with diagonal moves under `MaxDistance`.
"""
function init_planner(
    pf::DStarLite{2},
    goal::Tuple{Int,Int};
    start::DStarStart = nothing,
    heuristic::Symbol = :none,
)
    return _make_planner(pf, goal; start, heuristic)
end

function init_planner(
    pf::DStarLite{2},
    goal::Tuple{Int,Int},
    start::Tuple{Int,Int};
    heuristic::Symbol = :delta_cost,
)
    return _make_planner(pf, goal; start, heuristic)
end

function init_planner(pf::DStarLite{2}, goal::Tuple{Int,Int}, heuristic::Symbol)
    return _make_planner(pf, goal; start = nothing, heuristic)
end

"""
    update_after_cm_change!(planner, changed_cells)
    update_after_cm_change!(planner, changed_cells, new_start)

Handle edge-cost changes, lines {37''-48''} of the paper. The caller has already
written the new penalties into the pathfinder's penalty map; `changed_cells`
are the cells whose penalty changed. If the agent has moved since the last
call ({35''}), pass its cell as `new_start`.

1. `km += h(s_last, s_start)`, `s_last = s_start` {38''-39''} (focused only)
2. for every directed edge (u,v) touching a changed cell whose cost changed
   {40''}: take `c_old` from the planner's copy of the previous penalties
   {41''}, adopt the new cost {42''}, and repair `rhs(u)` exactly as in
   {43''-47''}
3. ComputeShortestPath() {48''}

Deviation from the paper: all edges adopt their new cost before the loop,
because a cell's penalty moves all its incident edges at once. {43''-47''} stay
correct, since every changed edge is still visited once with its own `c_old`.
"""
function update_after_cm_change!(
    pl::DStarLitePlanner,
    changed_cells::Vector{Tuple{Int,Int}},
    new_start::DStarStart = nothing,
)
    if pl.s_start !== nothing
        new_start !== nothing && (pl.s_start = new_start)          # {35''}
        pl.km += heuristic_cost(pl, pl.s_last, pl.s_start)         # {38''}
        pl.s_last = pl.s_start                                     # {39''}
    end

    # {40''-41''}: directed edges incident to changed cells, with c_old
    wm = pl.pf.walkmap
    seen = Set{NTuple{2,Tuple{Int,Int}}}()
    edges = Tuple{Tuple{Int,Int},Tuple{Int,Int},Float64}[]
    for x in changed_cells
        wm[x...] || continue                  # a blocked cell has no edges
        for n in _neighbors(pl, x), e in ((x, n), (n, x))
            e in seen && continue
            push!(seen, e)
            push!(edges, (e[1], e[2], _known_cost(pl, e[1], e[2])))
        end
    end

    # {42''}: adopt the new costs
    if pl.known_pmap !== nothing
        live = pl.pf.cost_metric.pmap
        for x in changed_cells
            pl.known_pmap[x...] = live[x...]
        end
    end

    for (u, v, c_old) in edges
        c_new = delta_cost(pl.pf, u, v)
        c_old == c_new && continue            # {40''} only changed edges
        if c_old > c_new                                                   # {43''}
            u != pl.goal && (pl.rhs[u...] = min(pl.rhs[u...], c_new + pl.g[v...]))  # {44''}
        elseif pl.rhs[u...] == c_old + pl.g[v...]                          # {45''}
            u != pl.goal && (pl.rhs[u...] = _min_succ(pl, u))              # {46''}
        end
        update_vertex!(pl, u)                                              # {47''}
    end

    compute_shortest_path!(pl)                                             # {48''}
    return pl
end

"""
    extract_path(planner, start; max_steps = 10_000)

Walk from `start` to `planner.goal`, always moving to a successor `s'` that
minimizes `c(s,s') + g(s')` ({34''}). Theorem 8 guarantees this is a shortest
path from the `s_start` that ComputeShortestPath() last ran for, and from any
cell on that path.

For a focused planner, a different `start` means the agent was relocated to a
cell that might not be on that path. The planner then runs
{38''-39''} and {48''} with no changed edges, i.e. what the paper's main loop
does on its next scan. For moves along the path, call
[`update_start!`](@ref) and then `extract_path(planner, planner.s_start)`.
"""
function extract_path(
    planner::DStarLitePlanner,
    start::Tuple{Int,Int};
    max_steps = 10_000,
)
    if planner.s_start !== nothing && start != planner.s_start
        planner.s_start = start
        update_after_cm_change!(planner, Tuple{Int,Int}[])
    end

    path = Tuple{Int,Int}[start]
    cur = start
    for _ in 1:max_steps
        cur == planner.goal && break
        best = nothing
        best_val = Inf
        for n in _neighbors(planner, cur)
            val = delta_cost(planner.pf, cur, n) + planner.g[n...]
            if val < best_val
                best_val = val
                best = n
            end
        end
        best === nothing && break             # {33''}: no known path
        cur = best
        push!(path, cur)
    end
    return path
end
