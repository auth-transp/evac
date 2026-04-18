begin   # Φόρτωση των απαραίτητων βιβλιοθηκών
    using Agents
    using Agents.Pathfinding
    using Dates
    using FileIO: load
    using XLSX
    using ImageMagick
    using Images
    using Random
    using Profile
    using BenchmarkTools
end

Agents.@agent struct AgentEscapes(ContinuousAgent{2, Float64})
    age::Float64
    mass::Float64
    toxicload::Float64
    pathX::Vector{Float64}
    pathY::Vector{Float64}
    TL1::Vector{Float64}
    TL2::Vector{Float64}
    TL3::Vector{Float64}
end

begin   # Φόρτωση heightmap και concentration maps 1..7 (ίδιο με Scenario 3 benchmark)
    heightmap_data = load("NADEEN/Maps/Qatargas Map.jpg")
    heightmap_data = permutedims(channelview(heightmap_data), [2, 3, 1])[:, :, 1]
    global heightmap = floor.(Int, convert.(Float64, heightmap_data) * 255)
    heightmap = 255 .- heightmap

    const NUM_CMS = 7
    cm_list = Vector{Array{Float64, 2}}(undef, NUM_CMS)
    for k in 1:NUM_CMS
        fn = joinpath("Concentration Maps", string(k) * ".bmp")
        img = load(fn)
        img = permutedims(channelview(img), [2, 3, 1])[:, :, 1]
        cm_list[k] = convert.(Float64, img) .* 255.0
    end
    for k in 1:NUM_CMS
        @assert size(cm_list[k]) == size(heightmap) "Concentration map $k size mismatch with heightmap"
    end
    global penalty_map = copy(cm_list[1])
end

NPM = heightmap .+ penalty_map
NPM_int = round.(Int, NPM)

begin   # Παράμετροι μοντέλου
    const METERS_TO_PIXELS = 723.37 / 2500.0
    dt = 1.0
    n_agents = 10

    toxicity_rate = 0.07
    age_range = (22, 60)
    speed_range = (4.0, 7.0)
    mass_range = (50, 80)
    ag_range_y = (size(heightmap)[1] / 4):(3 * size(heightmap)[1] / 4)
    ag_range_x = (size(heightmap)[2] / 4):(3 * size(heightmap)[2] / 4)
    dims = (size(NPM))
    walkmap = BitArray(trues(dims...))
    white_threshold = 245
    walkmap[heightmap .> white_threshold] .= false
end

# All 16 candidate goals (fixed order; indices 1:16). Sides are subsets of these indices.
const DESTS_ALL = Tuple{Float64, Float64}[
    (500., 854.), (120., 248.), (691., 750.), (351., 924.),
    (853., 659.), (795., 375.), (703., 187.), (864., 503.),
    (652., 73.), (325., 150.), (223., 198.), (473., 78.),
    (88., 493.), (145., 630.), (209., 787.), (39., 379.),
]

# Side A–D: global indices into DESTS_ALL (max 2 active points per side per run)
const DEST_INDICES_BY_SIDE = Vector{Vector{Int}}([
    [1, 3, 4, 5],
    [6, 7, 8, 9],
    [10, 11, 12, 2],
    [13, 14, 15, 16],
])

"""All ways to pick 0, 1, or 2 distinct indices from one side (≤2 per side)."""
function side_goal_options(side_indices::Vector{Int})
    opts = Vector{Vector{Int}}()
    push!(opts, Int[])
    for i in side_indices
        push!(opts, [i])
    end
    n = length(side_indices)
    for i in 1:(n - 1), j in (i + 1):n
        push!(opts, [side_indices[i], side_indices[j]])
    end
    return opts
end

"""
Enumerate every valid global index set: total goals in 1..8, at most 2 per side.
Sorted by (K, indices lexicographically). Stable index → use for reproducible benchmarks.
"""
function enumerate_valid_goal_combinations(dest_indices_by_side::Vector{Vector{Int}})
    opts = [side_goal_options(s) for s in dest_indices_by_side]
    seen = Set{Vector{Int}}()
    out = Vector{Vector{Int}}()
    for oA in opts[1], oB in opts[2], oC in opts[3], oD in opts[4]
        comb = sort!(Int[oA; oB; oC; oD])
        k = length(comb)
        if 1 <= k <= 8 && comb ∉ seen
            push!(seen, comb)
            push!(out, comb)
        end
    end
    sort!(out; by = x -> (length(x), x))
    return out
end

const GOAL_COMBINATIONS = begin
    u = enumerate_valid_goal_combinations(DEST_INDICES_BY_SIDE)
    println("Benchmark: $(length(u)) valid goal combinations (K in 1:8, max 2 per side); run i uses evenly spaced combo indices (see goal_combo_index_for_run).")
    u
end

"""
Map benchmark run `run_id` (1…`n_runs`) to a combination index in 1…`n_combos`.

When `n_runs` ≤ `n_combos`, indices are **evenly spaced** from 1 to `n_combos` so a batch of runs
samples across the full sorted list (not only the first combinations). When `n_runs` > `n_combos`,
falls back to `mod1(run_id, n_combos)`.
"""
function goal_combo_index_for_run(run_id::Int, n_runs::Int, n_combos::Int)
    @assert 1 <= run_id <= n_runs
    @assert n_combos >= 1
    n_runs > n_combos && return mod1(run_id, n_combos)
    n_runs == 1 && return 1
    return clamp(
        round(Int, 1 + (run_id - 1) * (n_combos - 1) / (n_runs - 1)),
        1,
        n_combos,
    )
end

space = ContinuousSpace(size(NPM); periodic = false, spacing = 1)

begin
    cost_metric_obj = AbsolutePenaltyMap(NPM_int, MaxDistance{2}())
    cost_metric_str = "APM"
    pathfinderPM = AStar(space; walkmap = walkmap, cost_metric = cost_metric_obj)
    properties = (
        pathfinderPM = pathfinderPM,
        heightmap = heightmap,
        dt = dt,
        speed_range = speed_range,
        goal = DESTS_ALL[[1, 2]],  # placeholder; each benchmark run sets goals in `run_one_benchmark`
    )
end

#
# Helper: true if position is within radius of any goal (TL and movement stop when true)
const goal_radius = 10.0
function at_goal(pos, dests, radius = goal_radius)
    px, py = Float64(pos[1]), Float64(pos[2])
    return minimum(hypot(px - d[1], py - d[2]) for d in dests) <= radius
end

function agent_step!(person, model)
    if at_goal(person.pos, model.goal)
        push!(person.pathX, person.pos[1])
        push!(person.pathY, person.pos[2])
        return
    end

    grid_dims = size(penalty_map)
    i = clamp(Int(floor(person.pos[1])), 1, grid_dims[1])
    j = clamp(Int(floor(person.pos[2])), 1, grid_dims[2])
    Ct = penalty_map[i, j]
    TLcurrent = [person.TL1[end], person.TL2[end], person.TL3[end]]
    TL = update_toxic_load(Ct, TLcurrent, model.dt)
    person.toxicload = sum(TL)
    push!(person.TL1, TL[1])
    push!(person.TL2, TL[2])
    push!(person.TL3, TL[3])
    speed = 1.35
    if 0 < person.toxicload <= 1
        speed = 1.35 * exp(0.393 * person.toxicload)
    elseif 1 < person.toxicload < 3
        speed = -1.78 * log(person.toxicload) + 2.063
    elseif person.toxicload >= 3
        speed = 0.0
    end
    move_along_route!(person, model, model.pathfinderPM, speed * METERS_TO_PIXELS, model.dt)
    push!(person.pathX, person.pos[1])
    push!(person.pathY, person.pos[2])
end

function model_step!(model)
    for (a1, a2) in interacting_pairs(model, 0.012, :nearest)
        elastic_collision!(a1, a2, :mass)
    end
end

model = ABM(
    AgentEscapes,
    space;
    rng = MersenneTwister(),
    properties = properties,
    agent_step! = agent_step!,
    model_step! = model_step!,
)

begin
    for _ in 1:n_agents
        age = rand(abmrng(model)) * (age_range[2] - age_range[1]) + age_range[1]
        mass = rand(abmrng(model)) * (mass_range[2] - mass_range[1]) + mass_range[1]
        vel = Tuple(rand(abmrng(model), 2) .* (speed_range[2] - speed_range[1]) .+ speed_range[1])
        max_attempts = 1000
        attempts = 0
        pos = nothing
        while attempts < max_attempts
            candidate_pos = Tuple((rand(abmrng(model), floor.(ag_range_y)), rand(abmrng(model), floor.(ag_range_x))))
            pos_int = floor.(Int, candidate_pos)
            if 1 <= pos_int[1] <= size(walkmap, 1) && 1 <= pos_int[2] <= size(walkmap, 2) && walkmap[pos_int[1], pos_int[2]]
                pos = candidate_pos
                break
            end
            attempts += 1
        end
        if pos === nothing
            println("Warning: Could not find valid walkable spawn position for agent in Scenario 4 benchmark after $max_attempts attempts")
            continue
        end
        add_agent!(
            pos, AgentEscapes, model,
            vel, age, mass, 1.0,
            [pos[1]], [pos[2]],
            [0.0], [0.0], [0.0]
        )
    end
end

function setupToxic()
    Atime = [0.0, 2.0, 5.0, 8.0, 15.0, 30.0]
    Arho = zeros(3, 6)
    Arho[1, 2:6] = [2.87, 2.33, 2.09, 1.81, 1.55]
    Arho[2, 2:6] = [202.38, 164.38, 147.74, 128.10, 109.45]
    Arho[3, 2:6] = [286.71, 232.87, 209.30, 181.47, 155.05]
    MW = 34
    Arho *= MW / 24.04
    Arho = Arho'
    Atime = Atime * 60
    taumin = 200.0
    taumax = 86400.0
    Brho = zeros(7, 3)
    Balpha = zeros(7, 3)
    rhomax = zeros(1, 3)
    rhomin = zeros(1, 3)
    Btime = zeros(7, 3)
    for k = 1:3
        for b = 2:6
            Brho[b, k] = Arho[b, k]
            Btime[b, k] = Atime[b]
        end
        Btime[1, k] = taumin
        Btime[7, k] = taumax
    end
    for k = 1:3
        for b = 3:6
            if Brho[b-1, k] == Brho[b, k]
                Balpha[b, k] = 0.0
            else
                Balpha[b, k] = log(Atime[b] / Atime[b-1]) / log(Brho[b-1, k] / Brho[b, k])
            end
        end
        Balpha[2, k] = Balpha[3, k]
        Balpha[1, k] = Balpha[2, k]
        Balpha[7, k] = Balpha[6, k]
    end
    for k = 1:3
        if Balpha[3, k] == 0
            rhomax[k] = Brho[2, k]
            Brho[1, k] = rhomax[k]
        else
            rhomax[k] = Brho[2, k] * (Btime[2, k] / taumin)^(1 / Balpha[2, k])
            Brho[1, k] = rhomax[k]
        end
        if Brho[5, k] == Brho[6, k]
            rhomin[k] = Brho[6, k]
            Brho[7, k] = rhomin[k]
        else
            rhomin[k] = Brho[6, k] * (Btime[6, k] / taumax)^(1 / Balpha[6, k])
            Brho[7, k] = rhomin[k]
        end
    end
    for k = 1:3
        for b = 2:7
            if Balpha[b, k] == 0
                Btime[b, k] = Btime[b-1, k]
            end
        end
    end
    for k = 1:3
        for b = 3:5
            if Balpha[b-1, k] == 0 && Balpha[b, k] > 0
                Balpha[b, k] = log(Btime[b, k] / Btime[b-1, k]) / log(Brho[b-1, k] / Brho[b, k])
            end
        end
    end
    return Balpha, Btime, Brho'
end

Balpha, Btime, Brho = setupToxic()

function update_toxic_load(Ct, TLcurrent, dt)
    TL = TLcurrent
    TL_rate = 0.0
    for k = 1:3
        Cmin = Brho[k, 7]
        Cmax = Brho[k, 1]
        if Ct > Cmax
            TL_rate = 1 / Btime[1, k]
        elseif Ct < Cmin
            TL_rate = 0.0
        else
            for i in 2:size(Btime, 1)
                if i <= size(Brho, 2)
                    if Brho[k, i-1] < Ct && Ct < Brho[k, i]
                        TL_rate = (1 / Btime[i, k]) * ((Ct / Brho[k, i])^(Balpha[i, k]))
                    end
                end
            end
        end
        TL[k] = TL[k] .+ TL_rate * dt
    end
    return TL
end

# --- Benchmark parameters ---
const BENCHMARK_MAX_RUNS = 50
const BENCHMARK_T_STEPS = 2500
const BENCHMARK_TL_CAP_PER_AGENT = 3.0

const CM_ACTIVE_FRACTION_BENCH = 0.7
const CM_SWITCH_TIMES = collect(range(
    0.0,
    stop = CM_ACTIVE_FRACTION_BENCH * ((BENCHMARK_T_STEPS - 1) * dt),
    length = NUM_CMS,
))

function format_active_goals_for_xlsx(active_idx::Vector{Int}, dests)
    idx_str = join(string.(active_idx), ";")
    xy_str = join(
        ["($(round(x; digits=1)),$(round(y; digits=1)))" for (x, y) in dests],
        "; ",
    )
    return idx_str, xy_str
end

"""
    run_one_benchmark(run_id)

Scenario 4: A* with `AbsolutePenaltyMap`, CM1→CM7 switching on `penalty_map` with replanning.
Active goals are **deterministic**: combination index from `goal_combo_index_for_run(run_id, BENCHMARK_MAX_RUNS, …)`
so batches with `BENCHMARK_MAX_RUNS` ≤ `length(GOAL_COMBINATIONS)` use **evenly spaced** indices across the full sorted list
`GOAL_COMBINATIONS`. Agent placement remains stochastic via `seed` (random per run).
"""
function run_one_benchmark(run_id::Int)
    seed = rand(Random.RandomDevice(), UInt32)
    rng_run = MersenneTwister(seed)

    n_combos = length(GOAL_COMBINATIONS)
    goal_combo_idx = goal_combo_index_for_run(run_id, BENCHMARK_MAX_RUNS, n_combos)
    active_dest_indices = copy(GOAL_COMBINATIONS[goal_combo_idx])
    dests = DESTS_ALL[active_dest_indices]
    K_active = length(active_dest_indices)
    idx_str, xy_str = format_active_goals_for_xlsx(active_dest_indices, dests)

    local_NPM_int = copy(NPM_int)
    local_cost_metric_obj = AbsolutePenaltyMap(local_NPM_int, MaxDistance{2}())
    local_pathfinderPM = AStar(space; walkmap = walkmap, cost_metric = local_cost_metric_obj)
    local_properties = (
        pathfinderPM = local_pathfinderPM,
        heightmap = heightmap,
        dt = dt,
        speed_range = speed_range,
        goal = dests,
    )

    model_run = ABM(
        AgentEscapes,
        space;
        rng = rng_run,
        properties = local_properties,
        agent_step! = agent_step!,
        model_step! = model_step!,
    )

    for _ in 1:n_agents
        age = rand(abmrng(model_run)) * (age_range[2] - age_range[1]) + age_range[1]
        mass = rand(abmrng(model_run)) * (mass_range[2] - mass_range[1]) + mass_range[1]
        vel = Tuple(rand(abmrng(model_run), 2) .* (speed_range[2] - speed_range[1]) .+ speed_range[1])
        max_attempts = 1000
        attempts = 0
        pos = nothing
        while attempts < max_attempts
            candidate_pos = Tuple((rand(abmrng(model_run), floor.(ag_range_y)), rand(abmrng(model_run), floor.(ag_range_x))))
            pos_int = floor.(Int, candidate_pos)
            if 1 <= pos_int[1] <= size(walkmap, 1) && 1 <= pos_int[2] <= size(walkmap, 2) && walkmap[pos_int[1], pos_int[2]]
                pos = candidate_pos
                break
            end
            attempts += 1
        end
        if pos === nothing
            println("Warning: Could not find valid walkable spawn for agent (seed=$seed)")
            continue
        end
        add_agent!(
            pos, AgentEscapes, model_run,
            vel, age, mass, 1.0,
            [pos[1]], [pos[2]],
            [0.0], [0.0], [0.0]
        )
    end

    current_map_idx = 1
    penalty_map .= cm_list[current_map_idx]
    @. local_NPM_int = Int(round(heightmap + penalty_map))
    Pathfinding.penaltymap(local_pathfinderPM) .= local_NPM_int

    initial_pf_time = @elapsed begin
        for a in allagents(model_run)
            plan_best_route!(a, dests, local_pathfinderPM)
        end
    end

    replan_time = 0.0
    sim_step_time = 0.0

    for step_idx in 1:BENCHMARK_T_STEPS
        sim_step_time += @elapsed step!(model_run, 1)
        t_sim = (step_idx - 1) * dt
        new_idx = searchsortedlast(CM_SWITCH_TIMES, t_sim)
        if new_idx != current_map_idx
            current_map_idx = new_idx
            replan_time += @elapsed begin
                penalty_map .= cm_list[current_map_idx]
                @. local_NPM_int = Int(round(heightmap + penalty_map))
                Pathfinding.penaltymap(local_pathfinderPM) .= local_NPM_int
                for a in allagents(model_run)
                    plan_best_route!(a, model_run.goal, local_pathfinderPM)
                end
            end
        end
    end

    incremental_pf_time = replan_time
    simulation_time = sim_step_time

    agents_sorted = sort(collect(allagents(model_run)); by = a -> a.id)
    sum_TL_capped = 0.0
    per_agent_TL_capped = Float64[]
    agent_ids = Int[]
    for a in agents_sorted
        tl_cap = min(a.toxicload, BENCHMARK_TL_CAP_PER_AGENT)
        sum_TL_capped += tl_cap
        push!(per_agent_TL_capped, tl_cap)
        push!(agent_ids, a.id)
    end

    return (
        initial_pf_time,
        incremental_pf_time,
        simulation_time,
        seed,
        goal_combo_idx,
        sum_TL_capped,
        per_agent_TL_capped,
        agent_ids,
        K_active,
        idx_str,
        xy_str,
    )
end

"""
    profile_one_benchmark()
Profile one benchmark run. After running, call Profile.print() or your profiler GUI.
"""
function profile_one_benchmark()
    Profile.clear()
    init_t, incr_t, sim_t, run_seed, _, sum_cap, _, _, _, _, _ = run_one_benchmark(1)
    println("Profiled run (seed=$run_seed): initial PF = $(round(init_t; digits=4)) s, incremental PF = $(round(incr_t; digits=4)) s, simulation = $(round(sim_t; digits=4)) s, sum TL (capped max $(BENCHMARK_TL_CAP_PER_AGENT) per agent) = $(round(sum_cap; digits=4))")
    println("Run Profile.print() or open profiler to inspect hotspots.")
end

"""
    benchmark_run_one_benchmark(; samples = 10)

Use BenchmarkTools to benchmark a single `run_one_benchmark(1)` invocation.
Returns the BenchmarkTools Trial object and also prints a summary.
"""
function benchmark_run_one_benchmark(; samples::Int = 10)
    println("Benchmarking run_one_benchmark(1) with BenchmarkTools (samples = $samples)...")
    result = @benchmark run_one_benchmark(1) samples = samples
    println(result)
    return result
end

# --- Benchmark loop ---
initial_pf_times = Float64[]
incremental_pf_times = Float64[]
simulation_times = Float64[]
sum_TL_capped_values = Float64[]
seeds = UInt32[]
goal_combination_indices = Int[]
n_active_goals_list = Int[]
active_goal_indices_str = String[]
active_goal_xy_str = String[]
agent_tl_runs = Int[]
agent_tl_ids = Int[]
agent_tl_values = Float64[]

println("Benchmark: Scenario 4 with A* (APM + CM replanning; goal combo index = evenly spaced over 1:N when runs ≤ N). Runs = $BENCHMARK_MAX_RUNS, agents = $n_agents, T = $BENCHMARK_T_STEPS, CM_active_fraction = $CM_ACTIVE_FRACTION_BENCH, N_combos = $(length(GOAL_COMBINATIONS)).")
for run_id in 1:BENCHMARK_MAX_RUNS
    t_init, t_incr, t_sim, run_seed, g_combo, sum_TL_capped, per_agent_TL_capped, agent_ids, K_a, idx_s, xy_s = run_one_benchmark(run_id)
    push!(initial_pf_times, t_init)
    push!(incremental_pf_times, t_incr)
    push!(simulation_times, t_sim)
    push!(sum_TL_capped_values, sum_TL_capped)
    push!(seeds, run_seed)
    push!(goal_combination_indices, g_combo)
    push!(n_active_goals_list, K_a)
    push!(active_goal_indices_str, idx_s)
    push!(active_goal_xy_str, xy_s)
    for k in eachindex(per_agent_TL_capped)
        push!(agent_tl_runs, run_id)
        push!(agent_tl_ids, agent_ids[k])
        push!(agent_tl_values, per_agent_TL_capped[k])
    end

    n = length(initial_pf_times)
    avg_initial_pf = sum(initial_pf_times) / n
    avg_incremental_pf = sum(incremental_pf_times) / n
    avg_simulation = sum(simulation_times) / n

    println("  Run $run_id (combo=$g_combo, seed=$run_seed, K=$K_a): initial PF = $(round(t_init; digits=4)) s, incremental PF = $(round(t_incr; digits=4)) s, simulation = $(round(t_sim; digits=4)) s, sum TL capped = $(round(sum_TL_capped; digits=4))  (avg total PF = $(round(avg_initial_pf + avg_incremental_pf; digits=4)) s)")
end

n_runs = length(initial_pf_times)
avg_initial_pf_final = sum(initial_pf_times) / n_runs
avg_incremental_pf_final = sum(incremental_pf_times) / n_runs
avg_pathfinding_final = avg_initial_pf_final + avg_incremental_pf_final
avg_simulation_final = sum(simulation_times) / n_runs

run_timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
xlsx_path = "SCENARIOS/SCENARIO 4/Simulation Results/AStar $(n_agents) Agents Benchmarking_$(run_timestamp).xlsx"
run_ids = collect(1:n_runs)
columns_data = [
    run_ids,
    goal_combination_indices,
    seeds,
    initial_pf_times,
    incremental_pf_times,
    simulation_times,
    sum_TL_capped_values,
    n_active_goals_list,
    active_goal_indices_str,
    active_goal_xy_str,
]
column_names = [
    "Run",
    "GoalCombinationIndex",
    "Seed",
    "InitialPathfindingTime_s",
    "IncrementalPathfindingTime_s",
    "SimulationTime_s",
    "SumTL_capped",
    "N_active_goals",
    "ActiveGoalIndices_1to16",
    "ActiveGoals_xy",
]
agent_columns_data = [agent_tl_runs, agent_tl_ids, agent_tl_values]
agent_column_names = ["Run", "AgentID", "TL_capped_max3"]
XLSX.writetable(
    xlsx_path;
    Runs = (columns_data, column_names),
    AgentTL = (agent_columns_data, agent_column_names),
    overwrite = true,
)
XLSX.openxlsx(xlsx_path, mode = "rw") do xf
    sh = xf["Runs"]
    sh[n_runs + 2, 1] = "Number of runs"
    sh[n_runs + 2, 2] = n_runs
    sh[n_runs + 3, 1] = "Average initial pathfinding time (s)"
    sh[n_runs + 3, 2] = avg_initial_pf_final
    sh[n_runs + 4, 1] = "Average incremental pathfinding time (s)"
    sh[n_runs + 4, 2] = avg_incremental_pf_final
    sh[n_runs + 5, 1] = "Average total pathfinding time (s)"
    sh[n_runs + 5, 2] = avg_pathfinding_final
    sh[n_runs + 6, 1] = "Average simulation time (s)"
    sh[n_runs + 6, 2] = avg_simulation_final
    sh[n_runs + 7, 1] = "BENCHMARK_T_STEPS"
    sh[n_runs + 7, 2] = BENCHMARK_T_STEPS
    sh[n_runs + 8, 1] = "CM_ACTIVE_FRACTION_BENCH"
    sh[n_runs + 8, 2] = CM_ACTIVE_FRACTION_BENCH
    sh[n_runs + 9, 1] = "EnumeratedGoalCombinations_N"
    sh[n_runs + 9, 2] = length(GOAL_COMBINATIONS)
end

println("Benchmark complete. Results written to $xlsx_path")
println("  Total runs: $n_runs | Avg initial PF: $(round(avg_initial_pf_final; digits=4)) s | Avg incremental PF: $(round(avg_incremental_pf_final; digits=4)) s | Avg simulation: $(round(avg_simulation_final; digits=4)) s")
