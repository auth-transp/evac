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

    heightmap_data = load("Maps/test01.png")
    heightmap_data = permutedims(channelview(heightmap_data), [2,3,1])[:,:,1]
    heightmap = floor.(Int, convert.(Float64, heightmap_data)*255)
    concentration_data = load("Maps/concentrationmap01.jpg")
    concentration_data = permutedims(channelview(concentration_data ), [2,3,1])[:,:,1]
    concentrationmap = floor.(Int, convert.(Float64, concentration_data )*10)
    
    dt = 0.1   ## discrete timestep each iteration of the model
    seed = 123  ## seed for random number generator
    n_agents = 3
    toxicity_rate = 0.1
    age_range = (22,60)
    speed_range = (4,7)
    speed = 5.
    mass_range = (50,80)
    ag_range_y = (size(heightmap)[1]/2-50):(size(heightmap)[1]/2+50)
    ag_range_x = (size(heightmap)[2]/2-100):(size(heightmap)[2]/2+100)
   
    
    dims = (size(heightmap))
    walkmap = BitArray(trues(dims...))
 

    #goals
    dests = [(540., 980.), (540., 609.)]

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
        speed_range = speed_range,
        :goal => dests,
    )

   
    model = ABM(AgentEscapes, space; rng, properties)
    

    for _ in 1:n_agents
        age = rand(model.rng)*(age_range[2]-age_range[1])
        mass = rand(model.rng) * (mass_range[2]-mass_range[1]) +mass_range[1]
        vel = Tuple(rand(model.rng, 2) .* (speed_range[2]-speed_range[1]) .+ speed_range[1])
        pos = Tuple((rand(model.rng, floor.(ag_range_y)), rand(model.rng, floor.(ag_range_x))))
        person = add_agent!(pos, model, vel, age, mass, 1.)
        plan_best_route!(person, dests, model.pathfinder)
    end
                
    return model
   
    function calculate_dispersion(heightmap)
        #dims = (size(heightmap))
        #return rand!(model.rng,zeros(dims))
        return concentrationmap
    end
    concentrationmap = calculate_dispersion(heightmap)  # Calculate toxic load based on position


function update_toxic_load!(person::AgentEscapes, dt::Float64, toxicity_rate::Float64, heightmap::Array{Int, 2})
    position = floor.(Int, person.pos)  # Get agent's position and convert to integer
    # Update toxic load of the agent
    person.toxicload += toxicity_rate * concentrationmap[position[1], position[2]] * dt
end
    
# function update_toxic_load!(person::AgentEscapes, dt::Float64, toxicity_rate::Float64)
#     person.toxicload += toxicity_rate*dt
# end


function agent_step!(person, model)
    update_toxic_load!(person, model.dt, toxicity_rate, heightmap)
    speed = 10. / person.toxicload
    #display("$speed - $(person.toxicload)")
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


function personcolor(person)
    person.toxicload <3. ? :green : :red
end

#BGmap_data = load("Maps/Background.PNG")
#original_colors = unique(vec(Float64.(channelview(heightmap))))
# heatarray = model -> penaltymap(model.pathfinder)
# heatkwargs = (colorrange = (-20, 60), colormap = :thermal)
# plotkwargs = (;
#     ac = :black, as = 8.0, am = 'o',
#     scatterkwargs = (strokewidth = 2.0,),
#     heatarray, 
#     heatkwargs)
    
#  GLMakie.activate!(inline=false)
#  fig, ax, abmobs = InteractiveDynamics.abmplot(model; plotkwargs, agent_step!, model_step!) 
#  fig
 

 
 InteractiveDynamics.abmvideo(
    "NADEEN TRIAL 10.mp4",
    model,
    agent_step!,
    model_step!;
    figurekwargs = (resolution = size(heightmap),),
    frames = 600,
    framerate = 15,
    ac = personcolor,
    as = 8,
    scatterkwargs = (strokecolor = :white, strokewidth = 1),
    heatarray = model -> penaltymap(model.pathfinder),
    heatkwargs = (colormap = :cividis,),
    static_preplot!
) 