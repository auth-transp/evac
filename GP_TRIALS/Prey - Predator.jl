using Agents, Random
using CairoMakie: Axis

@agent struct Sheep(GridAgent{2})
    energy::Float64
    reproduction_prob::Float64
    Δenergy::Float64
end

@agent struct Wolf(GridAgent{2})
    energy::Float64
    reproduction_prob::Float64
    Δenergy::Float64
end

function initialize_model(;
    n_sheep = 100,
    n_wolves = 50,
    dims = (20, 20),
    regrowth_time = 30,
    Δenergy_sheep = 4,
    Δenergy_wolf = 20,
    sheep_reproduce = 0.04,
    wolf_reproduce = 0.05,
    seed = 23182,
)

rng = MersenneTwister(seed)
space = GridSpace(dims, periodic = true)
# Model properties contain the grass as two arrays: whether it is fully grown
# and the time to regrow. Also have static parameter `regrowth_time`.
# Notice how the properties are a `NamedTuple` to ensure type stability.
properties = (
    fully_grown = falses(dims),
    countdown = zeros(Int, dims),
    regrowth_time = regrowth_time,
)
model = StandardABM(Union{Sheep, Wolf}, space;
    agent_step! = sheepwolf_step!, model_step! = grass_step!,
    properties, rng, scheduler = Schedulers.Randomly(), warn = false
)
# Add agents
for _ in 1:n_sheep
    energy = rand(abmrng(model), 1:(Δenergy_sheep*2)) - 1
    add_agent!(Sheep, model, energy, sheep_reproduce, Δenergy_sheep)
end
for _ in 1:n_wolves
    energy = rand(abmrng(model), 1:(Δenergy_wolf*2)) - 1
    add_agent!(Wolf, model, energy, wolf_reproduce, Δenergy_wolf)
end
# Add grass with random initial growth
for p in positions(model)
    fully_grown = rand(abmrng(model), Bool)
    countdown = fully_grown ? regrowth_time : rand(abmrng(model), 1:regrowth_time) - 1
    model.countdown[p...] = countdown
    model.fully_grown[p...] = fully_grown
end
return model
end

function sheepwolf_step!(sheep::Sheep, model)
    randomwalk!(sheep, model)
    sheep.energy -= 1
    if sheep.energy < 0
        remove_agent!(sheep, model)
        return
    end
    eat!(sheep, model)
    if rand(abmrng(model)) ≤ sheep.reproduction_prob
        sheep.energy /= 2
        replicate!(sheep, model)
    end
end

function sheepwolf_step!(wolf::Wolf, model)
    randomwalk!(wolf, model; ifempty=false)
    wolf.energy -= 1
    if wolf.energy < 0
        remove_agent!(wolf, model)
        return
    end
    # If there is any sheep on this grid cell, it's dinner time!
    dinner = first_sheep_in_position(wolf.pos, model)
    !isnothing(dinner) && eat!(wolf, dinner, model)
    if rand(abmrng(model)) ≤ wolf.reproduction_prob
        wolf.energy /= 2
        replicate!(wolf, model)
    end
end

function first_sheep_in_position(pos, model)
    ids = ids_in_position(pos, model)
    j = findfirst(id -> model[id] isa Sheep, ids)
    isnothing(j) ? nothing : model[ids[j]]::Sheep
end

function eat!(sheep::Sheep, model)
    if model.fully_grown[sheep.pos...]
        sheep.energy += sheep.Δenergy
        model.fully_grown[sheep.pos...] = false
    end
    return
end

function eat!(wolf::Wolf, sheep::Sheep, model)
    remove_agent!(sheep, model)
    wolf.energy += wolf.Δenergy
    return
end

function grass_step!(model)
    @inbounds for p in positions(model) # we don't have to enable bound checking
        if !(model.fully_grown[p...])
            if model.countdown[p...] ≤ 0
                model.fully_grown[p...] = true
                model.countdown[p...] = model.regrowth_time
            else
                model.countdown[p...] -= 1
            end
        end
    end
end

offset(a) = a isa Sheep ? (-0.1, -0.1*rand()) : (+0.1, +0.1*rand())
ashape(a) = a isa Sheep ? :circle : :utriangle
acolor(a) = a isa Sheep ? RGBAf(1.0, 1.0, 1.0, 0.8) : RGBAf(0.2, 0.2, 0.3, 0.8)

grasscolor(model) = model.countdown ./ model.regrowth_time

heatkwargs = (colormap = [:brown, :green], colorrange = (0, 1))

plotkwargs = (;
    agent_color = acolor,
    agent_size = 25,
    agent_marker = ashape,
    offset,
    agentsplotkwargs = (strokewidth = 1.0, strokecolor = :black),
    heatarray = grasscolor,
    heatkwargs = heatkwargs,
)

sheepwolfgrass = initialize_model()

fig, ax, abmobs = abmplot(sheepwolfgrass; plotkwargs...)
fig

sheep(a) = a isa Sheep
wolf(a) = a isa Wolf
count_grass(model) = count(model.fully_grown)

sheepwolfgrass = initialize_model()
steps = 5
adata = [(sheep, count), (wolf, count)]
mdata = [count_grass]
adf, mdf = run!(sheepwolfgrass, steps; adata, mdata)
