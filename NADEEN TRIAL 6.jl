using Agents, Agents.Pathfinding
using Random
using ColorTypes
import ImageMagick
using FileIO: load
using GLMakie 
using InteractiveDynamics
using Images
using DataFrames


@agent AgentEscapes ContinuousAgent{2} begin
    age::Float64
    mass::Float64
    toxicload::Float64
end

    # heightmap_url = "Maps/test01.png"
    # heightmap = floor.(Int, Float32.(Gray.(load(heightmap_url))) * 39) .+ 1
    heightmap_data = load("Maps/test01.png")
    heightmap_data = permutedims(channelview(heightmap_data), [2,3,1])[:,:,1]
    heightmap = floor.(Int, convert.(Float64, heightmap_data)*255)
    dt = 0.1   ## discrete timestep each iteration of the model
    seed = 123  # seed for random number generator
    n_agents = 40
    age_range = (22,60) #years old
    speed_range = (4,7) #in km/hr
    mass_range = (50,80) #in kg
    speed = 5.
    # ag_range_y = (size(heightmap)[1]/2-50):(size(heightmap)[1]/2+50)
    # ag_range_x = (size(heightmap)[2]/2-100):(size(heightmap)[2]/2+100)
    ag_range_y = (size(heightmap)[1]/2-50):(size(heightmap)[1]/2+50)
    ag_range_x = (size(heightmap)[2]/2-50):(size(heightmap)[2]/2+50)
    
    dims = (size(heightmap))
    walkmap = BitArray(trues(dims...))
    
    #goals
    dests = [(68., 51.)]#, #(896., 633.)]

    #Generate the RNG for the model
    rng = MersenneTwister(seed)

    ## Note that the dimensions of the space do not have to correspond to the dimensions
    ## of the pathfinder. Discretisation is handled by the pathfinding methods
    space = ContinuousSpace(size(heightmap); periodic = false, spacing = 1)
    

    pathfinder = AStar(space; walkmap = walkmap, cost_metric = PenaltyMap(heightmap, MaxDistance{2}()))
    properties = (
        pathfinder = AStar(space; walkmap = walkmap, cost_metric = PenaltyMap(heightmap, MaxDistance{2}())),
        heightmap = heightmap, 
        dt = dt,
        #speed = speed,
        speed_range = speed_range,
        :goal => dests,
    )

    #pathfinder = AStar(space; walkmap = walkmap, cost_metric = PenaltyMap(heightmap, MaxDistance{2}()))
    #properties = Dict(:goal => dests, :pathfinder => pathfinder)
    #model = ABM(AgentEscapes, space; rng = MersenneTwister(seed), properties = Dict(:goal => dests, :pathfinder => pathfinder))
    model = ABM(AgentEscapes, space; rng, properties)
    

    for _ in 1:n_agents
        age = rand(model.rng)*(age_range[2]-age_range[1])
        mass = rand(model.rng) * (mass_range[2]-mass_range[1]) +mass_range[1]
        vel = Tuple(rand(model.rng, 2) .* (speed_range[2]-speed_range[1]) .+ speed_range[1])
        #vel = Float64(rand(model.rng,))*speed
        pos = Tuple((rand(model.rng, floor.(ag_range_y)), rand(model.rng, floor.(ag_range_x))))
        person = add_agent!(pos, model, vel, age, mass, 1.)
        plan_best_route!(person, dests, model.pathfinder)
    end
   
  
    return model
    
        
function agent_step!(person, model)
    move_along_route!(person, model, model.pathfinder, speed, dt)
end



function model_step!(model)
    for (a1, a2) in interacting_pairs(model, 0.012, :nearest)
        elastic_collision!(a1, a2, :mass)
    end
end



function static_preplot!(ac, model) 
    scatter!(ac, model.goal; color = (:red, 50), marker = 'o')
end




# BGmap_data = load("Maps/BG.JPG")
# original_colors = unique(vec(Float64.(channelview(BGmap_data))))
# heatarray = original_colors
# heatkwargs = (colorrange = (-20, 60), colormap = :thermal)
# plotkwargs = (;
#     ac = :black, as = 8.0, am = 'o',
#     scatterkwargs = (strokewidth = 1.0,),
#     heatarray, heatkwargs)
    
# #  GLMakie.activate!(inline=false)
#     fig, ax, abmobs = InteractiveDynamics.abmplot(model; scatterkwargs = (strokecolor = :white, strokewidth = 1),
#     heatarray = model -> penaltymap(model.pathfinder),
#     heatkwargs = (colormap = :terrain,), agent_step!, model_step!) 
#   fig
 

# InteractiveDynamics.abmvideo(
# "6TH TRIAL EVAC.mp4", model, agent_step!, model_step!; plotkwargs, 
#  #figure = (resolution = size(heightmap),), 
#  figure = (resolution = (800,800),),
#  frames = 1000,
#  framerate = 15,
#  static_preplot!,
#  #original_colors,
#  title = "Evacuation Modeling Outdoors"
#  ) 
 
 InteractiveDynamics.abmvideo(
    "NADEEN TRIAL 6.mp4",
    model,
    agent_step!,
    model_step!;
    figurekwargs = (resolution = size(heightmap),),
    frames = 2000,
    framerate = 15,
    ac = :black,
    as = 8,
    scatterkwargs = (strokecolor = :white, strokewidth = 1),
    heatarray = model -> penaltymap(model.pathfinder),
    heatkwargs = (colormap = :terrain,),
    static_preplot!
) 