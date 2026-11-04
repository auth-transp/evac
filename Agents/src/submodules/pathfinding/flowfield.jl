# Flow Field pathfinding for efficient crowd movement to a common destination.
# Based on: https://howtorts.github.io/2014/01/04/basic-flow-fields.html
# Uses Dijkstra flood fill from the destination, then assigns each cell a unit vector
# toward the neighbor with lowest distance-to-destination.

mutable struct FlowField{D,P,M,T} <: GridPathfinder{D,P,M}
    destination::Union{Nothing,Dims{D}}
    distance_grid::Array{Float64,D}
    flow_field::Array{NTuple{D,Float64},D}
    dims::NTuple{D,T}
    neighborhood::Vector{CartesianIndex{D}}
    walkmap::BitArray{D}
    los_grid::BitArray{D}
    line_of_sight::Bool
    penalty_map::Array{Float64,D}
    absolute_penalty::Bool

    function FlowField{D,P,M,T}(
        destination::Union{Nothing,Dims{D}},
        distance_grid::Array{Float64,D},
        flow_field::Array{NTuple{D,Float64},D},
        dims::NTuple{D,T},
        neighborhood::Vector{CartesianIndex{D}},
        walkmap::BitArray{D},
        los_grid::BitArray{D},
        line_of_sight::Bool,
        penalty_map::Array{Float64,D},
        absolute_penalty::Bool,
    ) where {D,P,M,T}
        @assert all(dims .> 0) "Invalid pathfinder dimensions: $(dims)"
        (T <: Integer) && @assert size(walkmap) == dims "Walkmap must be same dimensions as grid"
        @assert size(distance_grid) == size(walkmap) "Distance grid must match walkmap size"
        @assert size(flow_field) == size(walkmap) "Flow field must match walkmap size"
        @assert size(los_grid) == size(walkmap) "Line of sight grid must match walkmap size"
        @assert size(penalty_map) == size(walkmap) "Penalty map must match walkmap size"
        new(destination, distance_grid, flow_field, dims, neighborhood, walkmap, los_grid, line_of_sight, penalty_map, absolute_penalty)
    end
end

"""
    Pathfinding.FlowField(space; kwargs...)

Flow field pathfinding for moving many agents to a common destination efficiently.
One destination is set per pathfinder; call [`plan_route!`](@ref) to set the destination
and regenerate the field. Agents then follow the flow using [`move_along_route!`](@ref).

Based on [Basic Flow Fields](https://howtorts.github.io/2014/01/04/basic-flow-fields.html):
Dijkstra flood fill from the destination gives distance-to-goal per cell; each cell's
flow vector points toward the neighbor with smallest distance. A diagonal step costs `√2`
(Euclidean length) and is allowed only if the cells it passes between are walkable too
(no corner cutting, see [Flow Fields Improvements](https://howtorts.github.io/2014/01/05/flow-field-improvements.html)).
With a `penalty_map`, a step between two cells also costs the absolute difference of their
penalties (the edge cost of [`PenaltyMap`](@ref) in A*), or, with `absolute_penalty = true`,
the sum of their absolute penalties (the edge cost of [`AbsolutePenaltyMap`](@ref)): cells with
zero penalty (e.g. roads) are then free and agents prefer them.

## Keywords
- `diagonal_movement = true`: allow diagonal neighbors (Moore) or only orthogonal (von Neumann).
- `walkmap = trues(spacesize(space))`: BitArray of walkable (true) vs obstacle (false) cells.
- `penalty_map = nothing`: array of penalties (e.g. a heightmap) with the size of the walkmap;
  `nothing` means flat terrain.
- `absolute_penalty = false`: cost of a step is `|p(u) - p(v)|` (`PenaltyMap`) if `false`, `|p(u)| + |p(v)|`
  (`AbsolutePenaltyMap`) if `true`.
- `line_of_sight = true`: in 2D non-periodic grids, mark the cells that see the destination
  ([Flow Fields: Line of Sight](https://howtorts.github.io/2014/01/30/Flow-Fields-LOS.html)).
  In a continuous space, [`flow_at_continuous`](@ref) then points straight at the destination
  from those cells instead of following the (grid-shaped) flow. With a `penalty_map`, line of sight is
  only marked between cells of equal penalty, and not at all if `absolute_penalty = true`
  (a straight line is not necessarily the cheapest route then).
"""
function FlowField(
    dims::NTuple{D,T};
    periodic::Union{Bool,NTuple{D,Bool}} = false,
    diagonal_movement::Bool = true,
    walkmap::BitArray{D} = trues(dims),
    line_of_sight::Bool = true,
    penalty_map::Union{Nothing,AbstractArray{<:Real,D}} = nothing,
    absolute_penalty::Bool = false,
) where {D,T}
    neighborhood = diagonal_movement ? moore_neighborhood(D) : vonneumann_neighborhood(D)
    sz = size(walkmap)
    distance_grid = fill(Inf, sz)
    flow_field = [ntuple(_ -> 0.0, D) for _ in 1:prod(sz)]
    flow_field = reshape(flow_field, sz)
    return FlowField{D,periodic,diagonal_movement,T}(
        nothing,
        distance_grid,
        flow_field,
        dims,
        neighborhood,
        walkmap,
        falses(sz),
        line_of_sight,
        penalty_map === nothing ? zeros(Float64, sz) : Array{Float64,D}(penalty_map),
        absolute_penalty,
    )
end

FlowField(
    space::GridSpace{D,periodic};
    diagonal_movement::Bool = true,
    walkmap::BitArray{D} = trues(spacesize(space)),
    line_of_sight::Bool = true,
    penalty_map::Union{Nothing,AbstractArray{<:Real,D}} = nothing,
    absolute_penalty::Bool = false,
) where {D,periodic} =
    FlowField(size(space); periodic, diagonal_movement, walkmap, line_of_sight, penalty_map, absolute_penalty)

function FlowField(
    space::ContinuousSpace{D,periodic};
    walkmap::Union{BitArray{D},Nothing} = nothing,
    line_of_sight::Bool = true,
    penalty_map::Union{Nothing,AbstractArray{<:Real,D}} = nothing,
    absolute_penalty::Bool = false,
) where {D,periodic}
    walkmap === nothing && (walkmap = BitArray(trues(Agents.spacesize(space))))
    FlowField(Tuple(Agents.spacesize(space)); periodic, diagonal_movement = true, walkmap, line_of_sight, penalty_map, absolute_penalty)
end

function Base.show(io::IO, pf::FlowField{D,P,M}) where {D,P,M}
    periodic = _flow_periodic_type(pf)
    moore = M ? "diagonal" : "orthogonal"
    dest = pf.destination === nothing ? "no destination" : "destination=$(pf.destination)"
    print(io, "FlowField in $(D)D, $(periodic)$(moore), $(dest)")
end

_flow_periodic_type(::FlowField{D,false,M}) where {D,M} = ""
_flow_periodic_type(::FlowField{D,true,M}) where {D,M} = "periodic, "
_flow_periodic_type(::FlowField{D,P,M}) where {D,P,M} = "mixed periodicity, "

# Shift `cur` by `off`, wrapping the periodic dimensions.
@inline function _flow_shift(cur::Dims{D}, off, s, ::Val{P}) where {D,P}
    return ntuple(D) do i
        per = P isa Bool ? P : P[i]
        per ? mod1(cur[i] + off[i], s[i]) : cur[i] + off[i]
    end
end

# Walkable neighbor of `cur` in direction `β`, or `nothing`. A diagonal step is allowed only if the
# cells it passes between are walkable too (no corner cutting, see "Flow Fields Improvements").
function _flow_neighbor(cur::Dims{D}, β::CartesianIndex{D}, pathfinder::FlowField{D,P}) where {D,P}
    walkmap = pathfinder.walkmap
    s = size(walkmap)
    off = β.I
    n = _flow_shift(cur, off, s, Val(P))
    all(1 .<= n .<= s) || return nothing
    walkmap[n...] || return nothing
    count(!iszero, off) > 1 || return n
    for sub in 1:(2^D-1)
        part = ntuple(i -> (sub >> (i - 1)) & 1 == 1 ? off[i] : 0, D)
        (part == off || all(iszero, part)) && continue
        c = _flow_shift(cur, part, s, Val(P))
        (all(1 .<= c .<= s) && walkmap[c...]) || return nothing
    end
    return n
end

# Euclidean length of a step in direction `β`: 1 orthogonal, √2 diagonal.
@inline _flow_step_cost(β::CartesianIndex) = sqrt(count(!iszero, β.I))

# Cost of the step from `cur` to its neighbor `nbor` (direction `β`): step length plus the penalty term
# (|p(cur) - p(nbor)| or, if `absolute_penalty`, |p(cur)| + |p(nbor)|). Symmetric, as the flood runs from the destination.
@inline function _flow_edge_cost(pathfinder::FlowField, cur, nbor, β)
    pc = pathfinder.penalty_map[cur...]
    pn = pathfinder.penalty_map[nbor...]
    return _flow_step_cost(β) + (pathfinder.absolute_penalty ? abs(pc) + abs(pn) : abs(pc - pn))
end

# Dijkstra flood fill from destination: fill distance_grid with distance-to-destination.
function _dijkstra_flood!(pathfinder::FlowField{D,P,M}) where {D,P,M}
    dest = pathfinder.destination
    dest === nothing && return
    dgrid = pathfinder.distance_grid
    walkmap = pathfinder.walkmap
    if !pathfinder.walkmap[dest...]
        return
    end
    dgrid .= Inf
    dgrid[dest...] = 0.0
    pq = DataStructures.PriorityQueue{Dims{D},Float64}()
    DataStructures.enqueue!(pq, dest, 0.0)
    while !isempty(pq)
        cur, dist = DataStructures.dequeue_pair!(pq)
        for β in pathfinder.neighborhood
            nbor = _flow_neighbor(cur, β, pathfinder)
            nbor === nothing && continue
            edge_cost = _flow_edge_cost(pathfinder, cur, nbor, β)
            new_dist = dist + edge_cost
            if new_dist < dgrid[nbor...]
                dgrid[nbor...] = new_dist
                pq[nbor] = new_dist  # insert or lower the key (a cell can be improved while queued)
            end
        end
    end
end

# For each cell, set flow vector toward the walkable neighbor with smallest distance.
function _generate_flow_vectors!(pathfinder::FlowField{D,P,M}) where {D,P,M}
    dest = pathfinder.destination
    dest === nothing && return
    dgrid = pathfinder.distance_grid
    ffield = pathfinder.flow_field
    walkmap = pathfinder.walkmap
    sz = size(walkmap)
    for I in CartesianIndices(sz)
        walkmap[I] || continue
        pos = Tuple(I)
        d_cur = dgrid[I]
        isinf(d_cur) && continue
        # At destination, flow is zero (stay).
        if d_cur == 0.0
            ffield[I] = ntuple(_ -> 0.0, D)
            continue
        end
        best_nbor = nothing
        best_dist = Inf  # lowest (distance of neighbor + step cost): the neighbor on a shortest path
        for β in pathfinder.neighborhood
            nbor = _flow_neighbor(pos, β, pathfinder)
            nbor === nothing && continue
            d_n = dgrid[nbor...] + _flow_edge_cost(pathfinder, pos, nbor, β)
            if d_n < best_dist
                best_dist = d_n
                best_nbor = nbor
            end
        end
        if best_nbor !== nothing
            diff = ntuple(i -> Float64(best_nbor[i] - pos[i]), D)
            n = LinearAlgebra.norm(diff)
            ffield[I] = n > 0 ? ntuple(i -> diff[i] / n, D) : ntuple(_ -> 0.0, D)
        else
            ffield[I] = ntuple(_ -> 0.0, D)
        end
    end
end

"""
    Pathfinding.generate_flow_field!(pathfinder::FlowField)
Regenerate the flow field from the current destination. Called automatically by
[`plan_route!`](@ref) when the destination is set.
"""
function generate_flow_field!(pathfinder::FlowField{D,P,M}) where {D,P,M}
    _dijkstra_flood!(pathfinder)
    _generate_flow_vectors!(pathfinder)
    _generate_los!(pathfinder)
end

# Line of sight to the destination (2D, non-periodic), after "Flow Fields: Line of Sight" (howtorts):
# a cell sees the destination if the cell next to it along its dominant axis does and so does the diagonal
# one toward the destination; on an exact diagonal both adjacent straight cells must also be walkable.
# Cells are visited by 4-neighbor flood from the destination, so the cells it depends on are already known.
function _generate_los!(pathfinder::FlowField{D,P}) where {D,P}
    los = pathfinder.los_grid
    fill!(los, false)
    dest = pathfinder.destination
    (D == 2 && pathfinder.line_of_sight && !pathfinder.absolute_penalty && dest !== nothing) || return
    (P isa Bool ? P : any(P)) && return
    walkmap = pathfinder.walkmap
    walkmap[dest...] || return
    visited = falses(size(walkmap))
    visited[dest...] = true
    los[dest...] = true
    queue = [dest]
    head = 1
    while head <= length(queue)
        c = queue[head]
        head += 1
        if c != dest
            dx, dy = dest[1] - c[1], dest[2] - c[2]
            sx, sy = sign(dx), sign(dy)
            has = false
            abs(dx) >= abs(dy) && sx != 0 && los[c[1]+sx, c[2]] && (has = true)
            abs(dy) >= abs(dx) && sy != 0 && los[c[1], c[2]+sy] && (has = true)
            if dx != 0 && dy != 0
                if !los[c[1]+sx, c[2]+sy]
                    has = false
                elseif abs(dx) == abs(dy) && !(walkmap[c[1]+sx, c[2]] && walkmap[c[1], c[2]+sy])
                    has = false
                end
            end
            los[c...] = has && pathfinder.penalty_map[c...] == pathfinder.penalty_map[dest...]  # straight is optimal only across equal penalties
        end
        for (ax, ay) in ((1, 0), (-1, 0), (0, 1), (0, -1))
            n = (c[1] + ax, c[2] + ay)
            all(1 .<= n .<= size(walkmap)) || continue
            (walkmap[n...] && !visited[n...]) || continue
            visited[n...] = true
            push!(queue, n)
        end
    end
end

# Continuous position of the centre of the destination cell.
_flow_goal_position(pathfinder::FlowField{D}) where {D} =
    ntuple(i -> (pathfinder.destination[i] - 0.5) * pathfinder.dims[i] / size(pathfinder.walkmap)[i], D)

# Centre of the destination if `pos` lies in a cell that sees it (and line of sight is on), else `nothing`.
function _flow_los_goal(pathfinder::FlowField{D}, pos) where {D}
    (pathfinder.line_of_sight && D == 2 && pathfinder.destination !== nothing) || return nothing
    sz = size(pathfinder.walkmap)
    cell = ntuple(i -> floor(Int, pos[i] / pathfinder.dims[i] * sz[i]) + 1, D)
    all(1 .<= cell .<= sz) || return nothing
    pathfinder.los_grid[cell...] || return nothing
    return _flow_goal_position(pathfinder)
end

"""
    Pathfinding.flow_at(pathfinder::FlowField{D}, pos::Dims{D}) -> NTuple{D,Float64}
Return the flow direction (unit vector) at grid cell `pos`. Returns zero vector
for obstacles or if no destination is set.
"""
function flow_at(pathfinder::FlowField{D}, pos::Dims{D}) where {D}
    pathfinder.destination === nothing && return ntuple(_ -> 0.0, D)
    all(1 .<= pos .<= size(pathfinder.walkmap)) || return ntuple(_ -> 0.0, D)
    pathfinder.walkmap[pos...] || return ntuple(_ -> 0.0, D)
    return pathfinder.flow_field[pos...]
end

"""
    Pathfinding.distance_at(pathfinder::FlowField{D}, pos::Dims{D}) -> Float64
Return the distance-to-destination at grid cell `pos`. Returns `Inf` for obstacles
or if no destination is set.
"""
function distance_at(pathfinder::FlowField{D}, pos::Dims{D}) where {D}
    pathfinder.destination === nothing && return Inf
    all(1 .<= pos .<= size(pathfinder.walkmap)) && return pathfinder.distance_grid[pos...]
    return Inf
end

"""
    Pathfinding.path_length_at(pathfinder::FlowField{D}, pos::Dims{D}) -> Int
Number of cells (steps) of the route that follows the flow field from grid cell `pos` to the
destination. Returns `typemax(Int)` if there is no destination or the destination cannot be reached.
This is the length that `plan_best_route!` compares when choosing among several destinations with A*
(the number of cells of the route, not its cost).
"""
function path_length_at(pathfinder::FlowField{D}, pos::Dims{D}) where {D}
    dest = pathfinder.destination
    dest === nothing && return typemax(Int)
    isfinite(distance_at(pathfinder, pos)) || return typemax(Int)
    cur = pos
    n = 0
    while cur != dest
        v = flow_at(pathfinder, cur)
        all(iszero, v) && return typemax(Int)
        cur = ntuple(i -> cur[i] + round(Int, v[i]), D)
        n += 1
        n > length(pathfinder.walkmap) && return typemax(Int)
    end
    return n
end

"""
    Pathfinding.flow_at_continuous(pathfinder::FlowField{D}, pos)
Return the flow direction (unit vector) at continuous position `pos` using
bilinear interpolation over the four surrounding grid cells. If `pos` lies in a cell
with line of sight to the destination (see `line_of_sight` in [`FlowField`](@ref)), the
direction straight to the centre of the destination cell is returned instead.
Returns zero vector if no destination or outside bounds.
"""
function flow_at_continuous(pathfinder::FlowField{D}, pos) where {D}
    pathfinder.destination === nothing && return ntuple(_ -> 0.0, D)
    sz = size(pathfinder.walkmap)
    goal = _flow_los_goal(pathfinder, pos)
    if goal !== nothing  # the destination is visible: head straight for it
        v = ntuple(i -> goal[i] - pos[i], D)
        n = LinearAlgebra.norm(v)
        return n > 1e-12 ? ntuple(i -> v[i] / n, D) : ntuple(_ -> 0.0, D)
    end
    # Grid-space coords where cell `i` is centred on the integer `i` (cell `i` spans [i-0.5, i+0.5)),
    # so that the flow vector of a cell is sampled at its centre and agents converge to the centre of the destination cell
    g = (pos ./ pathfinder.dims .* sz) .+ 0.5
    floor_cell = Tuple(floor.(Int, g))
    frac = ntuple(i -> g[i] - floor_cell[i], D)
    if D == 2
        i0 = clamp(floor_cell[1], 1, sz[1])
        j0 = clamp(floor_cell[2], 1, sz[2])
        i1 = min(floor_cell[1] + 1, sz[1])
        j1 = min(floor_cell[2] + 1, sz[2])
        f00 = pathfinder.walkmap[i0, j0] ? pathfinder.flow_field[i0, j0] : ntuple(_ -> 0.0, D)
        f10 = pathfinder.walkmap[i1, j0] ? pathfinder.flow_field[i1, j0] : ntuple(_ -> 0.0, D)
        f01 = pathfinder.walkmap[i0, j1] ? pathfinder.flow_field[i0, j1] : ntuple(_ -> 0.0, D)
        f11 = pathfinder.walkmap[i1, j1] ? pathfinder.flow_field[i1, j1] : ntuple(_ -> 0.0, D)
        xw = frac[1]
        top = ntuple(i -> f00[i] * (1 - xw) + f10[i] * xw, D)
        bottom = ntuple(i -> f01[i] * (1 - xw) + f11[i] * xw, D)
        yw = frac[2]
        dir = ntuple(i -> top[i] * (1 - yw) + bottom[i] * yw, D)
        n = LinearAlgebra.norm(dir)
        return n > 0 ? ntuple(i -> dir[i] / n, D) : ntuple(_ -> 0.0, D)
    else
        # 1D or D>=3: use flow at nearest cell
        disc = ntuple(i -> clamp(floor_cell[i], 1, sz[i]), D)
        return pathfinder.walkmap[disc...] ? pathfinder.flow_field[disc...] : ntuple(_ -> 0.0, D)
    end
end

Base.isempty(id::Int, pathfinder::FlowField) = true  # no per-agent path storage

"""
    is_stationary(agent, pathfinder::FlowField)
True if the pathfinder has no destination, or the agent is at the destination cell.
"""
function Agents.is_stationary(
    agent::AbstractAgent,
    pathfinder::FlowField{D},
) where {D}
    pathfinder.destination === nothing && return true
    pos = agent.pos
    # Grid: pos is integer indices; Continuous: pos is float coords
    disc = if all(p -> isinteger(p), pos)
        Tuple(floor.(Int, pos))
    else
        Tuple(to_discrete_position(pos, pathfinder))
    end
    dist = distance_at(pathfinder, disc)
    return dist == 0.0 || isinf(dist)
end
