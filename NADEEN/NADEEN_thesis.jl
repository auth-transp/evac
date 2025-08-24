using Agents
using Random
using Agents.Pathfinding
using UUIDs
using FileIO
using Images

#parameters definition
seed = 123
heightmap_data = load("Maps/test01.bmp")
goal = (70, 50)
#Agent type
abstract type AbstractHuman <: AbstractAgent end

@agent Agent GridAgent{2} begin
    #speed::Float64
    #age::Int64
    #mass::Float64
    #toxicload::Float64
end

#prepare map
heightmap_data = permutedims(channelview(heightmap_data), [2,3,1])[:,:,1]
heightmap = floor.(Int, convert.(Float64, heightmap_data)*255)
#Create Space and path
space = GridSpace(size(heightmap); periodic = false)
pathfinder = AStar(space; cost_metric = PenaltyMap(heightmap, MaxDistance{2}()))

#prepare model
model = ABM(Agent, space; rng = MersenneTwister(seed), properties = Dict(:goal => goal, :pathfinder => pathfinder))
ag_range_y = (size(heightmap)[1]/2-50):(size(heightmap)[1]/2+50)
ag_range_x = (size(heightmap)[2]/2-50):(size(heightmap)[2]/2+50)

for i in 1:10
    person = add_agent!((rand(model.rng, ag_range_x), rand(model.rng, ag_range_y)),  model)
    plan_route!(person, goal, model.pathfinder)
end

agent_step!(person, model) = move_along_route!(person, model, model.pathfinder)



using InteractiveDynamics
using CairoMakie
using GLMakie

static_preplot!(ax, model) = scatter!(ax, model.goal; color = (:red, 50), marker = 'x')



abm_video(
    "runners_10agents.mp4",
    model,
    agent_step!;
    figurekwargs = (resolution = size(heightmap),),
    frames = 200,
    framerate = 20,
    ac = :black,
    as = 8,
    scatterkwargs = (strokecolor = :white, strokewidth = 2),
    heatarray = model -> penaltymap(model.pathfinder),
    heatkwargs = (colormap = :terrain,),
    static_preplot!
)