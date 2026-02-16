# # Mountain Runners
# ```@raw html
# <video width="auto" controls autoplay loop>
# <source src="../runners.mp4" type="video/mp4">
# </video>
# ```
#
# Let's consider a race to the top of a mountain. Runners have been scattered about
# a map in some low lying areas and need to find the best path up to the peak.
#
# We'll use [`Pathfinding.AStar`](@ref) and a [`Pathfinding.AbsolutePenaltyMap`](@ref) to simulate this.

# ## Setup
include("../../Agents/src/Agents.jl")
    using .Agents
    import .Agents: ABM, Pathfinding
    import .Agents.Pathfinding: AStar, AbsolutePenaltyMap, MaxDistance, penaltymap
using Random
using FileIO # To load images you also need ImageMagick available to your project

@agent struct Runner(GridAgent{2}) end

# Our agent, as you can see, is very simple. Just an `id` and `pos`ition provided by
# [`@agent`](@ref). The rest of the dynamics of this example will be provided by the model.

function initialize(map_url; goal = (128, 409), seed = 88)
    ## Load an image file and convert it do a simple representation of height
    heightmap = floor.(Int, convert.(Float64, load(download(map_url))) * 255)
    ## The space of the model can be obtained directly from the image.
    ## Our example file is (400, 500).
    space = GridSpace(size(heightmap); periodic = false)
    ## The pathfinder. We use the `MaxDistance` metric together with an `AbsolutePenaltyMap`
    ## so that runners seek the easiest path to run, not just the most direct.
    pathfinder = AStar(space; cost_metric = AbsolutePenaltyMap(heightmap, MaxDistance{2}()))
    model = StandardABM(
        Runner,
        space;
        agent_step!,
        rng = MersenneTwister(seed),
        properties = Dict(:goal => goal, :pathfinder => pathfinder)
    )
    for _ in 1:10
        ## Place runners in the low-lying space in the map.
        runner = add_agent!((rand(abmrng(model), 100:350), rand(abmrng(model), 50:200)), model)
        ## Everyone wants to get to the same place.
        plan_route!(runner, goal, model.pathfinder)
    end
    return model
end

# The example heightmap we use here is a small region of countryside in Sweden, obtained
# with the [Tangram heightmapper](https://github.com/tangrams/heightmapper).

# ## Dynamics
# With the pathfinder in place, and all our runners having a goal position set, stepping
# is now trivial.

agent_step!(agent, model) = move_along_route!(agent, model, model.pathfinder)

######################################################################
# Let's Race – custom video creation (pattern from Scenario 2)
######################################################################

begin
    using CairoMakie
    using Observables
    using Makie

    # Load the sample heightmap and initialize the model
    map_url =
        "https://raw.githubusercontent.com/JuliaDynamics/" *
        "JuliaDynamics/master/videos/agents/runners_heightmap.jpg"
    model = initialize(map_url)

    # Total frames and framerate
    T = 200
    fr = 45

    # Set up figure and axis
    fig = Figure(; size = (700, 700))
    ax = Axis(fig[1, 1]; title = "Mountain runners", aspect = DataAspect())

    # Plot the penalty map as background heatmap
    heat = penaltymap(model.pathfinder)
    heatmap!(ax, heat; colormap = :terrain)

    # Mark the goal position
    scatter!(
        ax,
        [model.goal[1]],
        [model.goal[2]];
        color = (:red, 0.5),
        marker = :x,
        markersize = 20,
    )

    # Initial agent positions
    xs0 = [a.pos[1] for a in allagents(model)]
    ys0 = [a.pos[2] for a in allagents(model)]
    posobs = Observable(Point2f.(xs0, ys0))

    scatter!(
        ax,
        posobs;
        color = :black,
        markersize = 8,
        strokecolor = :white,
        strokewidth = 2,
    )

    # Record video by stepping the model and updating positions
    record(fig, "runners.mp4", 1:T; framerate = fr) do _
        step!(model, agent_step!, dummystep, 1)
        xs = [a.pos[1] for a in allagents(model)]
        ys = [a.pos[2] for a in allagents(model)]
        posobs[] = Point2f.(xs, ys)
    end
end

######################################################################
