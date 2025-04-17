# 03_model_setup.jl

# Parameters
dt = 1.0
seed = 123
n_agents = 3
toxicity_rate = 0.07
age_range = (22,60)
speed_range = (4.0, 7.0)
mass_range = (50,80)
speed = 5.0
MW = 34
dims = size(heightmap)
walkmap = BitArray(trues(dims...))

ag_range_y = (size(heightmap,1) ÷ 2 - 50):(size(heightmap,1) ÷ 2 + 50)
ag_range_x = (size(heightmap,2) ÷ 2 - 50):(size(heightmap,2) ÷ 2 + 50)

dests = [(600., 980.), (100., 200.)]

# Random number generator
rng = MersenneTwister(seed)

# Continuous space
extent = (Float64(size(heightmap, 1)), Float64(size(heightmap, 2)))
space = ContinuousSpace(extent, 1.0; periodic = false)

# Pathfinder with penalty
pathfinder = AStar(space; walkmap = walkmap, cost_metric = PenaltyMap(heightmap, MaxDistance{2}()))

# Model definition
properties = (
    pathfinder = pathfinder,
    heightmap = heightmap,
    dt = dt,
    speed_range = speed_range,
    :goal => dests
)

model = ABM(AgentEscapes, space; rng, properties)

# Add agents
for _ in 1:n_agents
    age = rand(model.rng) * (age_range[2] - age_range[1])
    mass = rand(model.rng) * (mass_range[2] - mass_range[1]) + mass_range[1]
    vel = Tuple(rand(model.rng, 2) .* (speed_range[2] - speed_range[1]) .+ speed_range[1])
    pos = (
        rand(model.rng, collect(floor.(ag_range_y))),
        rand(model.rng, collect(floor.(ag_range_x)))
    )
    person = add_agent!(pos, AgentEscapes, model, vel, age, mass, 1.0, [pos[1]], [pos[2]], [0.0], [0.0], [0.0])
    set_best_target!(person, dests, model.pathfinder)
end