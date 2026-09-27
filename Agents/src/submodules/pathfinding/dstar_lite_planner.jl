# Incremental D* Lite (Koenig & Likhachev, AAAI 2002 / JAIR 2004).
#
# Two modes share one struct:
# - Full-field (`s_start === nothing`): expand until `U` is empty. Reverse LPA*
#   / Dijkstra from the goal so any cell can call `extract_path` (Scenario 3).
# - Focused (`s_start` set): stop when `U.TopKey ≥ CalculateKey(s_start)` and
#   `rhs(s_start) == g(s_start)`. Heuristic is `h(s_start, s)`; `km` tracks
#   start motion.

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
end

@inline function heuristic_cost(
    pl::DStarLitePlanner,
    from::Tuple{Int,Int},
    to::Tuple{Int,Int},
)
    hk = pl.heuristic_kind
    if hk === :delta_cost
        return delta_cost(pl.pf, from, to)
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

function update_vertex!(pl::DStarLitePlanner, u::Tuple{Int,Int})
    if u != pl.goal
        min_rhs = Inf
        for v in _neighbors(pl, u)
            min_rhs = min(min_rhs, delta_cost(pl.pf, u, v) + pl.g[v...])
        end
        pl.rhs[u...] = min_rhs
    end
    sync_queue!(pl, u)
    return nothing
end

# Sync u's membership/key in U to its current g/rhs without touching rhs itself
# (used where rhs has already been set by the caller). A key change is applied
# in place by the heap rather than as delete + reinsert.
@inline function sync_queue!(pl::DStarLitePlanner, u::Tuple{Int,Int})
    if pl.g[u...] != pl.rhs[u...]
        pl.U[u] = calc_key(pl, u)
    else
        delete!(pl.U, u)
    end
    return nothing
end

# Process the single top-of-queue vertex (one iteration of ComputeShortestPath's
# inner body). Shared by the start-focused loop and the full-drain fallback.
function _process_top!(pl::DStarLitePlanner)
    u, k_old = heap_top(pl.U)
    k_new = calc_key(pl, u)

    if _key_lt(k_old, k_new)
        pl.U[u] = k_new
    elseif pl.g[u...] > pl.rhs[u...]
        # u becomes locally consistent (g lowered to rhs): only relax the
        # predecessors whose rhs can actually improve through u's new g,
        # instead of recomputing every neighbor's rhs from scratch.
        pl.g[u...] = pl.rhs[u...]
        heap_pop!(pl.U)
        for s in _neighbors(pl, u)
            if s != pl.goal
                cand = delta_cost(pl.pf, s, u) + pl.g[u...]
                cand < pl.rhs[s...] && (pl.rhs[s...] = cand)
            end
            sync_queue!(pl, s)
        end
    else
        # u becomes (or stays) overconsistent/raised to Inf: u itself
        # always needs a full rhs recompute (it may still be reachable
        # via other neighbors), but predecessors only need one if their
        # rhs was actually derived from u's old g; the rest are untouched.
        g_old = pl.g[u...]
        pl.g[u...] = Inf
        heap_pop!(pl.U)
        update_vertex!(pl, u)
        for s in _neighbors(pl, u)
            if pl.rhs[s...] == delta_cost(pl.pf, s, u) + g_old
                update_vertex!(pl, s)
            else
                sync_queue!(pl, s)
            end
        end
    end
    return nothing
end

function compute_shortest_path!(pl::DStarLitePlanner)
    while !isempty(pl.U)
        if pl.s_start !== nothing
            k_start = calc_key(pl, pl.s_start)
            k_top = heap_top(pl.U)[2]
            if !(_key_lt(k_top, k_start) || pl.rhs[pl.s_start...] != pl.g[pl.s_start...])
                break
            end
        end
        _process_top!(pl)
    end
    return pl
end

"""
    drain_queue!(planner)

Fully process every vertex remaining in `planner.U`, ignoring the
start-focused early exit — the same guarantee full-field mode already gets.

D* Lite's focused termination only certifies `g(s_start)` itself; nodes that
were on a since-invalidated path (e.g. after a cost *increase*) can be left
locally "consistent" by stale cached values that were never revisited,
because the cascade never reached them before the focused search stopped.
`extract_path` uses this as a correctness fallback when it detects that
walking the cached gradient has stalled.
"""
function drain_queue!(pl::DStarLitePlanner)
    while !isempty(pl.U)
        _process_top!(pl)
    end
    return pl
end

"""
    update_start!(planner, new_start)

Move the D* Lite search origin. Updates `km += h(s_last, new_start)` as in
Koenig & Likhachev. Call `extract_path` or `compute_shortest_path!` afterwards.
"""
function update_start!(pl::DStarLitePlanner, new_start::Tuple{Int,Int})
    if pl.s_last !== nothing && new_start != pl.s_last
        pl.km += heuristic_cost(pl, pl.s_last, new_start)
    end
    pl.s_start = new_start
    pl.s_last = new_start
    return pl
end

function _make_planner(
    pf::DStarLite{2},
    goal::Tuple{Int,Int};
    start::DStarStart = nothing,
    heuristic::Symbol = :none,
)
    sz = size(pf.walkmap)
    g = fill(Inf, sz)
    rhs = fill(Inf, sz)
    rhs[goal...] = 0.0
    U = CellHeap(sz)
    hk = start === nothing ? :none : heuristic
    planner = DStarLitePlanner(pf, g, rhs, U, 0.0, goal, start, start, hk)
    U[goal] = calc_key(planner, goal)
    compute_shortest_path!(planner)
    return planner
end

"""
    init_planner(pf, goal)
    init_planner(pf, goal, start; heuristic = :delta_cost)
    init_planner(pf, goal, heuristic::Symbol)
    init_planner(pf, goal; start = nothing, heuristic = :none)

Build a D* Lite planner aimed at `goal`.

With no `start`, the queue is emptied so `g` is a full cost-to-go field
(Scenario 3 / many agents sharing an exit). With `start`, search is focused
and `heuristic` is `h(start, s)`:

- `:delta_cost` — `delta_cost(pf, start, s)`
- `:manhattan` / `:euclidean` — grid distance from `start` to `s`
- `:none` — `h = 0`
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

Re-plan after cells in `changed_cells` had their cost change. If the caller
already knows the agent's current position, pass it as `new_start` so the
start move (`update_start!`) happens *before* replanning runs — otherwise a
later `extract_path` call with a different start will trigger a second,
redundant replanning pass.

This always fully settles the queue ([`drain_queue!`](@ref)) rather than
stopping once `s_start` alone is consistent. A cost *increase* has to be
cascaded outward through every cell it can affect before a subsequent
`extract_path` can trust the cached `g` field along a multi-step path —
stopping early (the cheaper start-focused criterion used for pure start
movement) can leave cells "consistent" by bookkeeping but stale in fact,
because the increase's cascade never reached them.
"""
function update_after_cm_change!(
    planner::DStarLitePlanner,
    changed_cells::Vector{Tuple{Int,Int}},
    new_start::DStarStart = nothing,
)
    if planner.s_start !== nothing && new_start !== nothing
        update_start!(planner, new_start)
    end
    for u in changed_cells
        update_vertex!(planner, u)
        for v in _neighbors(planner, u)
            update_vertex!(planner, v)
        end
    end
    drain_queue!(planner)
    return planner
end

"""
    extract_path(planner, start; max_steps = 10_000)

Greedily walk the cached `g` field from `start` to `planner.goal`.

Focused D* Lite only guarantees `g(start)` itself once `compute_shortest_path!`
returns — cells further along a path that was invalidated by a recent cost
*increase* can still hold stale cached values if the update cascade never
reached them. If the greedy walk detects that (by trying to revisit an
already-visited cell), it falls back to [`drain_queue!`](@ref) once to bring
every touched cell to full consistency, then resumes.
"""
function extract_path(
    planner::DStarLitePlanner,
    start::Tuple{Int,Int};
    max_steps = 10_000,
)
    if planner.s_start !== nothing && start != planner.s_start
        update_start!(planner, start)
        compute_shortest_path!(planner)
    end

    path = Tuple{Int,Int}[]
    cur = start
    push!(path, cur)
    visited = Set{Tuple{Int,Int}}((cur,))
    drained = false

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

        if best !== nothing && best in visited && !drained
            drain_queue!(planner)
            drained = true
            continue
        end

        best === nothing && break

        push!(visited, best)
        cur = best
        push!(path, cur)
    end

    return path
end
