# Incremental D* Lite (Koenig & Likhachev, AAAI 2002 / JAIR 2004).
#
# Two modes share one struct:
# - Full-field (`s_start === nothing`): expand until `U` is empty. Reverse LPA*
#   / Dijkstra from the goal so any cell can call `extract_path` (Scenario 3).
# - Focused (`s_start` set): stop when `U.TopKey ≥ CalculateKey(s_start)` and
#   `rhs(s_start) == g(s_start)`. Heuristic is `h(s_start, s)`; `km` tracks
#   start motion.

const DStarStart = Union{Nothing,Tuple{Int,Int}}

# `PF` is the concrete pathfinder type so `delta_cost`/`walkable_neighbors`
# dispatch statically in the hot loop (an abstract `DStarLite{D}` field would not).
mutable struct DStarLitePlanner{PF<:DStarLite}
    pf::PF
    g::Matrix{Float64}
    rhs::Matrix{Float64}
    U::PriorityQueue{Tuple{Int,Int},Tuple{Float64,Float64}}
    km::Float64
    goal::Tuple{Int,Int}
    s_start::DStarStart
    s_last::DStarStart
    heuristic_kind::Symbol
end

@inline function _key_lt(a::Tuple{Float64,Float64}, b::Tuple{Float64,Float64})
    return (a[1] < b[1]) || (a[1] == b[1] && a[2] < b[2])
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
        for v in walkable_neighbors(u, pl.pf)
            min_rhs = min(min_rhs, delta_cost(pl.pf, u, v) + pl.g[v...])
        end
        pl.rhs[u...] = min_rhs
    end
    sync_queue!(pl, u)
    return nothing
end

# Sync u's membership/key in U to its current g/rhs without touching rhs itself
# (used where rhs has already been set by the caller).
@inline function sync_queue!(pl::DStarLitePlanner, u::Tuple{Int,Int})
    haskey(pl.U, u) && delete!(pl.U, u)
    if pl.g[u...] != pl.rhs[u...]
        pl.U[u] = calc_key(pl, u)
    end
    return nothing
end

# Process the single top-of-queue vertex (one iteration of ComputeShortestPath's
# inner body). Shared by the start-focused loop and the full-drain fallback.
function _process_top!(pl::DStarLitePlanner)
    pair = peek(pl.U)
    u = pair.first
    k_old = pair.second
    k_new = calc_key(pl, u)

    if _key_lt(k_old, k_new)
        pl.U[u] = k_new
    elseif pl.g[u...] > pl.rhs[u...]
        # u becomes locally consistent (g lowered to rhs): only relax the
        # predecessors whose rhs can actually improve through u's new g,
        # instead of recomputing every neighbor's rhs from scratch.
        pl.g[u...] = pl.rhs[u...]
        dequeue!(pl.U)
        for s in walkable_neighbors(u, pl.pf)
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
        dequeue!(pl.U)
        update_vertex!(pl, u)
        for s in walkable_neighbors(u, pl.pf)
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
            k_top = peek(pl.U).second
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
    U = PriorityQueue{Tuple{Int,Int},Tuple{Float64,Float64}}()
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
        for v in walkable_neighbors(u, planner.pf)
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
        for n in walkable_neighbors(cur, planner.pf)
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
