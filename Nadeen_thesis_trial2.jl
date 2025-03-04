using Agents
using Random
using Agents.Pathfinding
using UUIDs 
using FileIO
using Images
using DataFrames
using ColorTypes
using InteractiveDynamics
export abmexploration
export abmvideo
using GLMakie
import ImageMagick
using FileIO: load

#parameters definition
seed = 123
heightmap_data = load("Maps/test01.png")
BGmap_data = load("Maps/BG.JPG")
goals = [(446., 181.), (200., 400.)]
speed_range = (4,10) #in kmphr range between 4 and 10
mass_range = (50, 80) #mass in kg
age_range = (22, 60) #age range between 22 and 60 years old
dt = 0.1 # Time step for simulation
speed = 10.

#Agent type
abstract type AbstractHuman <: AbstractAgent end

@agent AgentEscapes ContinuousAgent{2} begin
    age::Float64
    mass::Float64
    toxicload::Float64
end


#prepare map
heightmap_data = permutedims(channelview(heightmap_data), [2,3,1])[:,:,1]
heightmap = floor.(Int, convert.(Float64, heightmap_data)*255)
dims = (size(heightmap))
walkmap = BitArray(trues(dims...))

# #Create Space and path
space = ContinuousSpace(size(heightmap); periodic = false)
pathfinder = AStar(space; walkmap = walkmap, cost_metric = PenaltyMap(heightmap, MaxDistance{2}()))

# #prepare model
model = ABM(AgentEscapes, space; rng = MersenneTwister(seed), properties = Dict(:goal => goals, :pathfinder => pathfinder))
ag_range_y = (size(heightmap)[1]/2-50):(size(heightmap)[1]/2+50)
ag_range_x = (size(heightmap)[2]/2-25):(size(heightmap)[2]/2+25)

remove_all!(model)
 
for i in 1:40
    age = rand(model.rng)*(age_range[2]-age_range[1])
    mass = rand(model.rng) * (mass_range[2]-mass_range[1]) +mass_range[1]
    vel = Tuple(rand(model.rng, 2) .* (speed_range[2]-speed_range[1]) .+ speed_range[1])
    pos = Tuple((rand(model.rng, floor.(ag_range_y)), rand(model.rng, floor.(ag_range_x))))
    person = add_agent!(pos, model, vel, age, mass, 1.) 
end

function agent_step!(person, model)
    plan_route!(person, goals, model.pathfinder)
    move_along_route!(person, model, model.pathfinder, speed, dt)
end

# function agent_step!(person, pos, model)
#     move_agent!(person, pos, model)
# end

function model_step!(model)
    for (a1, a2) in interacting_pairs(model, 0.012, :nearest)
        elastic_collision!(a1, a2, :mass)
    end
end



function static_preplot!(ac, model) 
    scatter!(ac, model.goal; color = (:red, 50), marker = 'o')
end

#original_colors = unique(vec(Float64.(channelview(heightmap_data))))
original_colors = unique(vec(Float64.(channelview(BGmap_data))))

heatarray = original_colors
heatkwargs = (colorrange = (-20, 60), colormap = :thermal)
plotkwargs = (;
    ac = :black, as = 8.0, am = 'o',
    scatterkwargs = (strokewidth = 1.0,),
    heatarray, heatkwargs)
    
 GLMakie.activate!(inline=false)
 fig, ax, abmobs = InteractiveDynamics.abmplot(model; plotkwargs, agent_step!, model_step!) 
 fig
 


InteractiveDynamics.abmvideo(
"EVAC_OUTDOORS.mp4", model, agent_step!, model_step!; plotkwargs, 
 figure = (resolution = size(heightmap_data),), frames = 300,
 framerate = 15,
 am = 'o',
 ac = :black,
 as = 8.0,
 static_preplot!,
 original_colors,
 title = "Evacuation Modeling Outdoors"
 )




