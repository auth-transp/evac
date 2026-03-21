# ContinuousSpace API for FlowField: plan_route!, move_along_route! with bilinear flow, random_walkable.

"""
    plan_route!(agent, dest, pathfinder::FlowField{D})
Set the flow field destination to `dest` (continuous coordinates). The destination
is discretized to a grid cell; the field is regenerated. All agents using this
pathfinder move toward this destination when calling [`move_along_route!`](@ref).
"""
function Agents.plan_route!(
    agent::AbstractAgent,
    dest,
    pathfinder::FlowField{D,P,M,<:AbstractFloat},
) where {D,P,M}
    disc = Tuple(to_discrete_position(dest, pathfinder))
    if !all(1 .<= disc .<= size(pathfinder.walkmap)) || !pathfinder.walkmap[disc...]
        return
    end
    pathfinder.destination = disc
    generate_flow_field!(pathfinder)
end

"""
    move_along_route!(agent, model::ABM{<:ContinuousSpace{D}}, pathfinder::FlowField{D}, speed, dt = 1.0)
Move the agent along the flow field for one step. Uses bilinear interpolation of the
flow at the agent's position (see [Basic Flow Fields](https://howtorts.github.io/2014/01/04/basic-flow-fields.html)),
then applies desired velocity = flow * speed and moves by speed * dt in that direction.
"""
function Agents.move_along_route!(
    agent::AbstractAgent,
    model::ABM{<:ContinuousSpace{D}},
    pathfinder::FlowField{D},
    speed::Float64,
    dt::Real = 1.0,
) where {D}
    pathfinder.destination === nothing && return
    dir = flow_at_continuous(pathfinder, agent.pos)
    if all(≈(0), dir)
        return
    end
    T = typeof(agent.pos)
    from = agent.pos
    desired = T(dir) .* speed
    next_pos = from .+ desired .* dt
    next_pos = Agents.normalize_position(T(next_pos), model)
    move_agent!(agent, next_pos, model)
end

function random_walkable(
    model::ABM{<:ContinuousSpace{D}},
    pathfinder::FlowField{D,P,M,<:AbstractFloat},
) where {D,P,M}
    discrete_pos = Tuple(rand(
        abmrng(model),
        filter(x -> pathfinder.walkmap[x], CartesianIndices(pathfinder.walkmap)),
    ))
    half_cell = pathfinder.dims ./ size(pathfinder.walkmap) ./ 2
    return to_continuous_position(discrete_pos, pathfinder) .+
        Tuple(rand(abmrng(model), D) .- 0.5) .* half_cell
end

"""
    Pathfinding.random_walkable(pos, model::ABM{<:ContinuousSpace{D}}, pathfinder::FlowField{D}, r = 1.0)
Return a random walkable position within radius `r` of `pos`. Returns `pos` if none exist.
"""
function random_walkable(
    pos,
    model::ABM{<:ContinuousSpace{D}},
    pathfinder::FlowField{D},
    r = 1.0,
) where {D}
    sz = size(pathfinder.walkmap)
    discrete_r = ntuple(i -> max(0, floor(Int, r / pathfinder.dims[i] * sz[i]) - 1), D)
    discrete_pos = Tuple(to_discrete_position(pos, pathfinder))
    options = collect(walkable_cells_in_radius_flow(discrete_pos, discrete_r, pathfinder))
    isempty(options) && return pos
    discrete_rand = rand(abmrng(model), options)
    half_cell = pathfinder.dims ./ size(pathfinder.walkmap) ./ 2
    T = typeof(pos)
    cts_rand = to_continuous_position(discrete_rand, pathfinder) .+
        (T(rand(abmrng(model)) for _ in 1:D) .- 0.5) .* half_cell
    dist = euclidean_distance(pos, cts_rand, model)
    if dist > r
        cts_rand = Agents.normalize_position(
            T(pos .+ get_direction(pos, cts_rand, model) ./ dist .* r),
            model,
        )
    end
    return cts_rand
end

walkable_cells_in_radius_flow(pos, r, pathfinder::FlowField{D,false}) where {D} =
    Iterators.filter(
        x ->
            all(1 .<= x .<= size(pathfinder.walkmap)) &&
            pathfinder.walkmap[x...] &&
            sum(((x .- pos) ./ max.(r, 1)) .^ 2) <= 1,
        Iterators.product([(pos[i]-r[i]):(pos[i]+r[i]) for i in 1:D]...),
    )

walkable_cells_in_radius_flow(pos, r, pathfinder::FlowField{D,true}) where {D} =
    Iterators.map(
        x -> mod1.(x, size(pathfinder.walkmap)),
        Iterators.filter(
            x ->
                pathfinder.walkmap[mod1.(x, size(pathfinder.walkmap))...] &&
                sum(((x .- pos) ./ max.(r, 1)) .^ 2) <= 1,
            Iterators.product([(pos[i]-r[i]):(pos[i]+r[i]) for i in 1:D]...),
        ),
    )
