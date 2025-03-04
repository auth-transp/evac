using Agents, Agents.Pathfinding
using Random
using ColorTypes
import ImageMagick
using FileIO: load
using GLMakie 
using InteractiveDynamics
using Images
using DataFrames
export abmexploration
export abmvideo

@agent AgentEscapes ContinuousAgent{2} begin
    age::Float64
    mass::Float64
    toxicload::Float64
end


const v0 = (0.0, 0.0, 0.0) 
person(id, pos, age, mass, toxicload) = AgentEscapes(id, pos, v0, age, mass, :person, toxicload)
eunorm(vec) = √sum(vec .^ 2)
#original_colors = unique(vec(Float64.(channelview(BGmap_da))))

function initialize_model(
    heightmap_url = "Maps/map3.bmp",
    dt = 0.05,   ## discrete timestep each iteration of the model
    seed = 123,  ## seed for random number generator
    n_agents = 40,
    age_range = (22,60),
    speed_range = (4,10),
    mass_range = (50,80),
    ag_range_y = (size(heightmap)[1]/2-250):(size(heightmap)[1]/2+250),
    ag_range_x = (size(heightmap)[2]/2-445):(size(heightmap)[2]/2+445),
)
    heightmap = floor.(Int, Float32.(Gray.(load(heightmap_url))) * 39) .+ 1
    #heightmap = floor.(Int, convert.(Float64, heightmap_data)*255)
    dims = (size(heightmap))
    walkmap = BitArray(trues(dims...))
    
  
    ## Generate the RNG for the model
    rng = MersenneTwister(seed)

    ## Note that the dimensions of the space do not have to correspond to the dimensions
    ## of the pathfinder. Discretisation is handled by the pathfinding methods
    space = ContinuousSpace(size(heightmap); periodic = false, spacing = 1)

   
    properties = (
        pathfinder = AStar(space; walkmap = walkmap, cost_metric = PenaltyMap(heightmap, MaxDistance{2}())),
        heightmap = heightmap,
        dt = dt
    )

    model = ABM(AgentEscapes, space; rng, properties)

    for _ in 1:n_agents
        age = rand(model.rng)*(age_range[2]-age_range[1])
        mass = rand(model.rng) * (mass_range[2]-mass_range[1]) +mass_range[1]
        vel = Tuple(rand(model.rng, 2) .* (speed_range[2]-speed_range[1]) .+ speed_range[1])
        pos = Tuple((rand(model.rng, floor.(ag_range_y)), rand(model.rng, floor.(ag_range_x))))
        person = add_agent!(pos, model, vel, age, mass, 1.) 
    end

    return model
end

model = initialize_model()
 

function agent_step!(person, model)
    if is_stationary(person, model.pathfinder)
        plan_route!(
            person,
            random_walkable(person.pos, model, model.pathfinder),
            model.pathfinder
        )
    end
    ## Move along the route planned above
    move_along_route!(person, model, model.pathfinder, model.dt)
end


function model_step!(model)
    for (a1, a2) in interacting_pairs(model, 0.012, :nearest)
        elastic_collision!(a1, a2, :mass)
    end
end

function static_preplot!(ac, model) 
    scatter!(ac, model.goal; color = (:red, 50), marker = 'o')
end

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
 #original_colors,
 title = "Evacuation Modeling Outdoors"
 )

