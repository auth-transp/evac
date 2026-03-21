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

    function FlowField{D,P,M,T}(
        destination::Union{Nothing,Dims{D}},
        distance_grid::Array{Float64,D},
        flow_field::Array{NTuple{D,Float64},D},
        dims::NTuple{D,T},
        neighborhood::Vector{CartesianIndex{D}},
        walkmap::BitArray{D},
    ) where {D,P,M,T}
        @assert all(dims .> 0) "Invalid pathfinder dimensions: $(dims)"
        (T <: Integer) && @assert size(walkmap) == dims "Walkmap must be same dimensions as grid"
        @assert size(distance_grid) == size(walkmap) "Distance grid must match walkmap size"
        @assert size(flow_field) == size(walkmap) "Flow field must match walkmap size"
        new(destination, distance_grid, flow_field, dims, neighborhood, walkmap)
    end
end

"""
    Pathfinding.FlowField(space; kwargs...)

Flow field pathfinding for moving many agents to a common destination efficiently.
One destination is set per pathfinder; call [`plan_route!`](@ref) to set the destination
and regenerate the field. Agents then follow the flow using [`move_along_route!`](@ref).

Based on [Basic Flow Fields](https://howtorts.github.io/2014/01/04/basic-flow-fields.html):
Dijkstra flood fill from the destination gives distance-to-goal per cell; each cell's
flow vector points toward the neighbor with smallest distance.

## Keywords
- `diagonal_movement = true`: allow diagonal neighbors (Moore) or only orthogonal (von Neumann).
- `walkmap = trues(spacesize(space))`: BitArray of walkable (true) vs obstacle (false) cells.
"""
function FlowField(
    dims::NTuple{D,T};
    periodic::Union{Bool,NTuple{D,Bool}} = false,
    diagonal_movement::Bool = true,
    walkmap::BitArray{D} = trues(dims),
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
    )
end

FlowField(
    space::GridSpace{D,periodic};
    diagonal_movement::Bool = true,
    walkmap::BitArray{D} = trues(spacesize(space)),
) where {D,periodic} =
    FlowField(size(space); periodic, diagonal_movement, walkmap)

function FlowField(
    space::ContinuousSpace{D,periodic};
    walkmap::Union{BitArray{D},Nothing} = nothing,
) where {D,periodic}
    walkmap === nothing && (walkmap = BitArray(trues(Agents.spacesize(space))))
    FlowField(Tuple(Agents.spacesize(space)); periodic, diagonal_movement = true, walkmap)
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
        for nbor in get_neighbors(cur, pathfinder)
            all(1 .<= nbor .<= size(walkmap)) || continue
            pathfinder.walkmap[nbor...] || continue
            edge_cost = 1.0  # uniform step cost
            new_dist = dist + edge_cost
            if new_dist < dgrid[nbor...]
                dgrid[nbor...] = new_dist
                DataStructures.enqueue!(pq, nbor, new_dist)
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
        best_dist = d_cur
        for nbor in get_neighbors(pos, pathfinder)
            all(1 .<= nbor .<= sz) || continue
            walkmap[nbor...] || continue
            d_n = dgrid[nbor...]
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
    Pathfinding.flow_at_continuous(pathfinder::FlowField{D}, pos)
Return the flow direction (unit vector) at continuous position `pos` using
bilinear interpolation over the four surrounding grid cells. Returns zero vector
if no destination or outside bounds.
"""
function flow_at_continuous(pathfinder::FlowField{D}, pos) where {D}
    pathfinder.destination === nothing && return ntuple(_ -> 0.0, D)
    sz = size(pathfinder.walkmap)
    # Grid-space fractional coords: 1-based cell index + fraction in [0,1)
    g = (pos ./ pathfinder.dims .* sz) .+ 1
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
