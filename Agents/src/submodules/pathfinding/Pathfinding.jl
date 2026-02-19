export Pathfinding

"""
    Pathfinding
Submodule containing functionality for path-finding based on the A* algorithm.
Currently available for [`GridSpace`](@ref) and [`ContinuousSpace`](@ref).
Discretization of [`ContinuousSpace`](@ref) is taken care of internally.

You can enable path-finding and set its options by creating an instance of a
[`Pathfinding.AStar`](@ref) struct. This must be passed to the relevant pathfinding functions
during the simulation. Call [`plan_route!`](@ref) to set the destination for an agent. This
triggers the algorithm to calculate a path from the agent's current position to the one
specified. You can alternatively use [`plan_best_route!`](@ref) to choose the best target
from a list. Once a target has been set, you can move an agent one step along its
precalculated path using the [`move_along_route!`](@ref) function.

Refer to the [Maze Solver](https://juliadynamics.github.io/AgentsExampleZoo.jl/dev/examples/maze/),
[Mountain Runners](https://juliadynamics.github.io/AgentsExampleZoo.jl/dev/examples/runners/)
and [Rabbit, Fox, Hawk](https://juliadynamics.github.io/Agents.jl/stable/examples/rabbit_fox_hawk/)
examples using path-finding and see the available functions below as well.
"""

module Pathfinding


using ..Agents
using DataStructures
using LinearAlgebra

abstract type GridPathfinder{D,P,M} end

include("metrics.jl")
include("pathfinding_utils.jl")
include("grid_common.jl")
include("astar.jl")
include("astar_grid.jl")
include("continuous_common.jl")
include("astar_continuous.jl")
include("dstar_lite.jl")
include("dstar_lite_grid.jl")
include("dstar_lite_continuous.jl")
include("id_star_lite.jl")
include("gaussian_plume.jl")

export CostMetric,
    DirectDistance,
    MaxDistance,
    PenaltyMap,
    AbsolutePenaltyMap,
    AStar,
    delta_cost,
    penaltymap,
    nearby_walkable,
    random_walkable,
    DStarLite,
    GridPathfinder,
    DStarLitePlanner,
    init_planner,
    update_after_cm_change!,
    extract_path,
    TerrainHazardMetric,
    IDStarLite,
    IDStarLitePlanner,
    IDStarLitePlannerHeuristic,
    gaussian_plume_map

# Deprecations
@deprecate set_target! plan_route!
@deprecate set_best_target! plan_best_route!
@deprecate kill_agent! remove_agent!

end # module
