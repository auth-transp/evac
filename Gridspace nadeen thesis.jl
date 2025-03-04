using Agents
using Random
using Agents.Pathfinding
using UUIDs
using FileIO
using Images
using DataFrames
import Pkg

#parameters definition
seed = 123
heightmap_data = load("Maps/map3.bmp")
BGmap_data = load("Maps/BG.JPG")
goals = [(446., 181.), (200., 400.)]
speed_range = (4,10) #in kmphr range between 4 and 10
mass_range = (50, 80) #mass in kg
age_range = (22, 60) #age range between 22 and 60 years old
dt = 1.0 / 24  # Time step for simulation
v0 = 5.0  # Desired velocity
tau = 0.5  # Strength of goal force

relaxation_time = 0.5  # Strength of relaxation force
radius_range = (10,20) #radius in 'cm' between the agents

#Agent type
abstract type AbstractHuman <: AbstractAgent end

@agent AgentEscapes GridAgent{2} begin
    #speed::Float64
    age::Float64
    mass::Float64
    vel::Tuple
    axes::Tuple
    toxicload::Float64
end

#prepare map
heightmap_data = permutedims(channelview(heightmap_data), [2,3,1])[:,:,1]
heightmap = floor.(Int, convert.(Float64, heightmap_data)*255)

#Create Space and path
space = GridSpace(size(heightmap); periodic = false)
pathfinder = AStar(space; cost_metric = PenaltyMap(heightmap, MaxDistance{2}()))

#prepare model
model = ABM(AgentEscapes, space; rng = MersenneTwister(seed), properties = Dict(:goal => goals, :pathfinder => pathfinder))
ag_range_y = (size(heightmap)[1]/2-250):(size(heightmap)[1]/2+250)
ag_range_x = (size(heightmap)[2]/2-445):(size(heightmap)[2]/2+445)

remove_all!(model)
 

for i in 1:40
    age = rand(model.rng)*(age_range[2]-age_range[1])
    mass = rand(model.rng) * (mass_range[2]-mass_range[1]) +mass_range[1]
    vel = Tuple(rand(model.rng, 2) .* (speed_range[2]-speed_range[1]) .+ speed_range[1])
    axes = Tuple((rand(model.rng, floor.(ag_range_y)), rand(model.rng, floor.(ag_range_x))))
    person = add_agent_single!(model, age, mass, vel, axes, 1.) 
end
    
  

function Agents.plan_route!(person, model, pathfinder)
   move_along_route!(person, model, pathfinder, speed_range, dt=0.05)
end


function model_step!(model)
    for (a1, a2) in interacting_pairs(model, 0.012, :nearest)
        elastic_collision!(a1, a2, :mass)
    end
end


using InteractiveDynamics
using GLMakie
import ImageMagick
using FileIO: load

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
 fig, ax, abmobs = InteractiveDynamics.abmplot(model; plotkwargs, Agents.plan_route!, model_step!)
 fig

 
using Makie
using WGLMakie
using FFMPEG


abmvideo(
"EVAC_OUTDOORS.mp4", model, add_agent_step!, model_step!; plotkwargs, 
 figure = (resolution = size(heightmap_data),), frames = 300,
 framerate = 15,
 am = 'o',
 ac = :black,
 as = 8.0,
 static_preplot!,
 original_colors,
 title = "Evacuation Modeling Outdoors"
 )


Agents.abmvideo.("NEW_OUTDOOR_RUNNERS.mp4", model, add_agent_step!, model_step!; plotkwargs, spf = 1,
framerate = 15,
frames = 300,
title = "Evacuation Modeling Outdoors",
showstep = true,
figure = (resolution = size(heightmap_data),),
profile ="high",
static_preplot!,
original_colors)