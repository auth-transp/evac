# Shared helpers for continuous-space pathfinding (A* and D* Lite).
# Converts between continuous and discrete grid positions.

"""
    to_discrete_position(pos, pathfinder::GridPathfinder)
Convert continuous position `pos` to discrete grid cell indices.
"""
to_discrete_position(pos, pathfinder::GridPathfinder) =
    floor.(Int, pos ./ pathfinder.dims .* size(pathfinder.walkmap)) .+ 1

"""
    to_continuous_position(pos, pathfinder::GridPathfinder)
Convert discrete grid cell indices to continuous position (cell center).
"""
to_continuous_position(pos, pathfinder::GridPathfinder) =
    pos ./ size(pathfinder.walkmap) .* pathfinder.dims .-
    pathfinder.dims ./ size(pathfinder.walkmap) ./ 2.
