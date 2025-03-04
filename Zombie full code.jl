using Agents
using Random

@agent Zombie OSMAgent begin
    infected::Bool
    speed::Float64
    color::Symbol
    Size::Int
end

function initialise_zombies(simulation_id; seed = 1234, speed_range=(2.0,7.0), colors=[:black,:red,:blue],
    sizes=[8,10,12],longitude_range=(-180.0,180.0),latitude_range=(-90.0,90.0),step_range=(0.01,0.05))
    map_path = OSM.test_map()
    properties = Dict(:dt => 1 / 60)
    model = ABM(
        Zombie,
        OpenStreetMapSpace(map_path);
        properties = properties,
        rng = Random.MersenneTwister(seed)
    )

    for id in 1:100
        start = random_position(model) # At an intersection
        speed = rand(model.rng) * 5.0 + 2.0 # Random speed from 2-7kmph
        color=colors[mod(simulation_id, length(colors))+1]
        size=sizes[mod(simulation_id, length(sizes))+1]
        human = Zombie(false, speed, color.black, size)
        add_agent_pos!(human, model)
        OSM.plan_random_route!(human, model; limit = 50) # try 50 times to find a random route
    end

    ## We'll add patient zero at a specific (longitude, latitude)
    longitude=rand(model.rng)*(longitude_range[2]-longitude_range[1])+longitude_range[1]
    latitude=rand(model.rng)*(latitude_range[2]-latitude_range[1])+latitude_range[1]
    start = OSM.nearest_road((9.9351811, 51.5328328), model)
    finish = OSM.nearest_node((9.945125635913511, 51.530876112711745), model)

    speed = rand(model.rng) * (speed_range[2]-speed_range[1])+speed_range[1] # Random speed from 2-7kmph
    color=colors[mod(simulation_id, length(colors))+1]
    size=sizes[mod(simulation_id, length(sizes))+1]
    zombie = add_agent!(start, model, true, speed)
    plan_route!(zombie, finish, model)

    step=rand(model,rng)*(step_range[2]-step_range[1])+step_range[1]

    return model,speed_Range,color,size,longitude_range,latitude_range,step
end

function zombie_step!(agent, model)
    distance_left = move_along_route!(agent, model, agent.speed * model.dt)

    if is_stationary(agent, model) && rand(model.rng) < 0.1
        OSM.plan_random_route!(agent, model; limit = 50)
        move_along_route!(agent, model, distance_left)
    end

    if agent.infected
        map(i -> model[i].infected = true, nearby_ids(agent, model, 0.01))
    end

    return
end

#define different sets of parameters
speed_ranges=[(2.0,4.0),(4.0,6.0),(6.0,8.0)]
colors=[:black,:red,:blue,:purple,:yellow]
sizes=[8,10,12,14]
longitude_ranges=[(-180.0,-90.0),(-90.0,0.0),(0.0,90.0),(90.0,180.0)]
latitude_ranges=[(-90.0,-45.0),(-45.0,0.0),(0.0,45.0),(45.0,90.0)]
step_ranges=[(0.01,0.05),(0.05,0.1),(0.1,0.2)]
framerate_range=(10,30)
frames_range=(100,300)

for simulation_id in 1:25
    speed_range=speed_ranges[mod(simulation_id,length(speed_ranges))+1]
    longitude_range=longitude_ranges[mod(simulation_id,length(longitude_ranges))+1]
    latitude_range=latitude_ranges[mod(simulation_id,length(latitude_ranges))+1]
    step_range=step_ranges[mod(simulation_id,length(step_ranges))+1]
    framerate=rand(10:30)
    frames=rand(100:300)
    #zombies,speed_range_sim,color_sim,size_sim,longitude_range_sim,latitude_range_sim,step_sim=initialise_zombies(simulation_id; 
    #seed=simulation_id*1000,speed_range=speed_range,colors=colors,sizes=sizes,longitude_range=longitude_range,
    #latitude_range=latitude_range,step_range=step_range)
    output_file="Zombie outbreak_$simulation_id.mp4"
end  

using CairoMakie, OSMMakie
CairoMakie.activate!()
zombie_color(agent) = agent.infected ? :green : :black
zombie_size(agent) = agent.infected ? 10 : 8
zombies = initialise_zombies(1)

abmvideo("OUTBREAK.mp4", zombies, zombie_step!;
    title = "Zombie outbreak", framerate = 15, frames = 200,
    ac = zombie_color, as = zombie_size)

    
println("Simulation $simulation_id complete. Output saved as $output_file")
println("Simulation $simulation_id parameters:")
println("Speed Range: $speed_range_sim")
println("Color: $color_sim")
println("Size: $size_sim")
println("Longitude Range: $longitude_range_sim")
println("Latitude Range: $latitude_range_sim")
println("Step Range: $step_sim")
println("Framerate: $framerate")
println("Frames: $frames")