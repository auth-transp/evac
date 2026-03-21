# GridSpace API for FlowField: plan_route!, move_along_route!, nearby_walkable, random_walkable.

"""
    plan_route!(agent, dest, pathfinder::FlowField{D})
Set the flow field destination to `dest` (grid position `Dims{D}`) and regenerate the field.
All agents using this pathfinder will move toward this destination when calling
[`move_along_route!`](@ref). If `dest` is not walkable, the destination is not set.
"""
function Agents.plan_route!(
    agent::AbstractAgent,
    dest::Dims{D},
    pathfinder::FlowField{D},
) where {D}
    if !all(1 .<= dest .<= size(pathfinder.walkmap)) || !pathfinder.walkmap[dest...]
        return
    end
    pathfinder.destination = dest
    generate_flow_field!(pathfinder)
end

"""
    move_along_route!(agent, model::ABM{<:GridSpace{D}}, pathfinder::FlowField{D})
Move the agent one grid step along the flow field toward the current destination.
If there is no destination or the agent is at the destination or in an obstacle, no move.
"""
function Agents.move_along_route!(
    agent::AbstractAgent,
    model::ABM{<:GridSpace{D}},
    pathfinder::FlowField{D},
) where {D}
    pathfinder.destination === nothing && return
    cur = Tuple(agent.pos)
    pathfinder.walkmap[cur...] || return
    d_cur = pathfinder.distance_grid[cur...]
    isinf(d_cur) && return
    d_cur == 0.0 && return  # at destination
    # Choose walkable neighbor with smallest distance
    best = cur
    best_d = d_cur
    for nbor in get_neighbors(cur, pathfinder)
        all(1 .<= nbor .<= size(pathfinder.walkmap)) || continue
        pathfinder.walkmap[nbor...] || continue
        d_n = pathfinder.distance_grid[nbor...]
        if d_n < best_d
            best_d = d_n
            best = nbor
        end
    end
    best == cur && return
    move_agent!(agent, best, model)
end

"""
    Pathfinding.nearby_walkable(position, model::ABM{<:GridSpace{D}}, pathfinder::FlowField{D}, r = 1)
Return an iterator over nearby positions within radius `r` that are walkable.
"""
nearby_walkable(
    position,
    model::ABM{<:GridSpace{D}},
    pathfinder::FlowField{D},
    r = 1,
) where {D} =
    Iterators.filter(x -> pathfinder.walkmap[x...], nearby_positions(position, model, r))

"""
    Pathfinding.random_walkable(model, pathfinder::FlowField{D})
Return a random walkable grid position.
"""
function random_walkable(model::ABM{<:GridSpace{D}}, pathfinder::FlowField{D}) where {D}
    return Tuple(rand(
        abmrng(model),
        filter(x -> pathfinder.walkmap[x], CartesianIndices(pathfinder.walkmap)),
    ))
end
