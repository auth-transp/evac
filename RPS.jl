using Agents

@agent struct Rock(GridAgent{2}) end
@agent struct Paper(GridAgent{2}) end
@agent struct Scissors(GridAgent{2}) end

@multiagent RPS(Rock, Paper, Scissors)

function attack!(agent, model)
    # Randomly pick a nearby agent
    contender = random_nearby_agent(agent, model)
    # do nothing if there isn't anyone nearby
    isnothing(contender) && return
    # else perform standard rock paper scissors logic
    # and remove the contender if you win.
    attack!(variant(agent), variant(contender), contender, model)
    return
end

attack!(::AbstractAgent, ::AbstractAgent, contender, model) = nothing
attack!(::Rock, ::Scissors, contender, model) = remove_agent!(contender, model)
attack!(::Scissors, ::Paper, contender, model) = remove_agent!(contender, model)
attack!(::Paper, ::Rock, contender, model) = remove_agent!(contender, model)

function move!(agent, model)
    rand_pos = random_nearby_position(agent.pos, model)
    if isempty(rand_pos, model)
        move_agent!(agent, rand_pos, model)
    else
        occupant_id = id_in_position(rand_pos, model)
        occupant = model[occupant_id]
        swap_agents!(agent, occupant, model)
    end
    return
end

function reproduce!(agent, model)
    pos = random_nearby_position(agent, model, 1, pos -> isempty(pos, model))
    isnothing(pos) && return
    # pass target position as a keyword argument
    replicate!(agent, model; pos)
    return
end

# Defining the propensity and timing of the events

attack_propensity = 1.0
movement_propensity = 0.5

function reproduction_propensity(agent, model)
    return cos(abmtime(model))^2
end

# Creating the `AgentEvent` structures

attack_event = AgentEvent(action! = attack!, propensity = attack_propensity)

reproduction_event = AgentEvent(action! = reproduce!, propensity = reproduction_propensity)

function movement_time(agent, model, propensity)
    # `agent` is the agent the event will be applied to,
    # which we do not use in this function!
    t = 0.1 * randn(abmrng(model)) + 1
    return clamp(t, 0, Inf)
end

movement_event = AgentEvent(
    action! = move!, propensity = movement_propensity,
    types = Union{Scissors, Paper}, timing = movement_time
)

events = (attack_event, reproduction_event, movement_event)

space = GridSpaceSingle((100, 100))

using Random: Xoshiro
rng = Xoshiro(42)

model = EventQueueABM(RPS, events, space; rng, warn = false)

const alltypes = (Rock, Paper, Scissors)

for p in positions(model)
    type = rand(abmrng(model), alltypes)
    add_agent!(p, RPS ∘ type, model)
end

abmqueue(model)

function initialize_rps(; n = 100, nx = n, ny = n, seed = 42)
    space = GridSpaceSingle((nx, ny))
    rng = Xoshiro(seed)
    model = EventQueueABM(RPS, events, space; rng, warn = false)
    for p in positions(model)
        type = rand(abmrng(model), alltypes)
        add_agent!(p, RPS ∘ type, model)
    end
    return model
end

step!(model, 123.456)

nagents(model)

function terminate(model, t)
    threshold = 1000
    # Alright, this code snippet loops over all types,
    # and for each it checks if it is less than the threshold.
    # if any is, it returns `true`, otherwise `false.`
    logic = any(alltypes) do type
        n = count(a -> variantof(a) == type, allagents(model))
        return n < threshold
    end
    # For safety, in case this never happens, we also add a trigger
    # regarding the total evolution time
    return logic || (t > 1000.0)
end

step!(model, terminate)

abmtime(model)

model = initialize_rps()

adata = [(a -> variantof(a) === X, count) for X in alltypes]

adf, mdf = run!(model, 100.0; adata, when = 0.5, dt = 0.01)

adf[1:10, :]
