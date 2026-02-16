# ID* Lite: improved D* Lite algorithm
# When the cost map changes, avoid full recalculation when none of the changed
# cells were part of the computed expansion (g < Inf). See:
#   "ID* Lite: improved D* Lite algorithm", ACM SAC 2011, DOI: 10.1145/1982185.1982483

export IDStarLite,
    IDStarLitePlanner,
    IDStarLitePlannerHeuristic

# IDStarLite wraps DStarLite and uses planners that can skip recalculation when
# changed cells do not affect the current solution (all changed cells have g == Inf).
struct IDStarLite{D,P,M,T,C<:CostMetric{D}}
    pf::DStarLite{D,P,M,T,C}
end

# Forward property access so IDStarLite can be used like DStarLite (walkmap, cost_metric, etc.)
function Base.getproperty(id::IDStarLite, name::Symbol)
    return getproperty(getfield(id, :pf), name)
end

function Base.size(id::IDStarLite)
    return size(getfield(id, :pf))
end

"""
    IDStarLite(dims::NTuple{D,T}; kwargs...)
    IDStarLite(space::ContinuousSpace; kwargs...)

Creates an ID* Lite pathfinder with the same API as D* Lite. Use with
`init_planner`, `update_after_cm_change!`, and `extract_path`. Recalculation
after cost-map changes is skipped when no changed cell was in the computed
expansion (g < Inf).
"""
IDStarLite(
    dims::NTuple{D,T};
    kwargs...,
) where {D,T} = IDStarLite(DStarLite(dims; kwargs...))

IDStarLite(
    space::GridSpace{D,periodic};
    kwargs...,
) where {D,periodic} = IDStarLite(DStarLite(space; kwargs...))

IDStarLite(
    space::ContinuousSpace{D,periodic};
    kwargs...,
) where {D,periodic} = IDStarLite(DStarLite(space; kwargs...))

function Base.show(io::IO, id::IDStarLite)
    print(io, "ID* Lite wrapping ")
    show(io, getfield(id, :pf))
end

# --------------- ID* Lite planner (non-heuristic): wraps DStarLitePlanner ---------------

mutable struct IDStarLitePlanner{D}
    inner::DStarLitePlanner{D}
end

"""
    init_planner(pf::IDStarLite{2}, goal::Tuple{Int,Int})

Creates an ID* Lite planner for the given goal. Use `update_after_cm_change!`
and `extract_path` as with D* Lite; recalculation is skipped when all
changed cells have g == Inf.
"""
function init_planner(pf::IDStarLite{2}, goal::Tuple{Int,Int})
    return IDStarLitePlanner(init_planner(getfield(pf, :pf), goal))
end

"""
    update_after_cm_change!(planner::IDStarLitePlanner, changed_cells)

Updates the planner after cost-map changes. If every cell in `changed_cells`
has g == Inf (was never expanded), recalculation is skipped (ID* Lite
improvement). Otherwise delegates to D* Lite.
"""
function update_after_cm_change!(
    planner::IDStarLitePlanner,
    changed_cells::Vector{Tuple{Int,Int}},
)
    inner = planner.inner
    # ID* Lite: skip recalculation when no changed cell was part of the expansion
    all(c -> inner.g[c...] == Inf, changed_cells) && return
    update_after_cm_change!(inner, changed_cells)
end

"""
    extract_path(planner::IDStarLitePlanner, start::Tuple{Int,Int}; max_steps = 10_000)

Extracts path from `start` to the planner goal; forwards to the inner D* Lite planner.
"""
function extract_path(
    planner::IDStarLitePlanner,
    start::Tuple{Int,Int};
    max_steps = 10_000,
)
    return extract_path(planner.inner, start; max_steps = max_steps)
end

# --------------- ID* Lite planner (heuristic): wraps DStarLitePlannerHeuristic ---------------

mutable struct IDStarLitePlannerHeuristic{D}
    inner::DStarLitePlannerHeuristic{D}
end

"""
    init_planner(pf::IDStarLite{2}, goal::Tuple{Int,Int}, heuristic::Symbol)

Creates an ID* Lite planner with the given heuristic (:delta_cost, :manhattan,
:euclidean, :none). Recalculation is skipped when all changed cells have g == Inf.
"""
function init_planner(
    pf::IDStarLite{2},
    goal::Tuple{Int,Int},
    heuristic::Symbol,
)
    return IDStarLitePlannerHeuristic(init_planner(getfield(pf, :pf), goal, heuristic))
end

function update_after_cm_change!(
    planner::IDStarLitePlannerHeuristic,
    changed_cells::Vector{Tuple{Int,Int}},
)
    inner = planner.inner
    all(c -> inner.g[c...] == Inf, changed_cells) && return
    update_after_cm_change!(inner, changed_cells)
end

function extract_path(
    planner::IDStarLitePlannerHeuristic,
    start::Tuple{Int,Int};
    max_steps = 10_000,
)
    return extract_path(planner.inner, start; max_steps = max_steps)
end
