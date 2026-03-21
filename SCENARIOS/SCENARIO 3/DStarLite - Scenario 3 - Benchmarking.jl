begin   # Φόρτωση των απαραίτητων βιβλιοθηκών
    using Agents
    using Agents.Pathfinding
    using FileIO: load
    using XLSX
    using ImageMagick
    using Images
    using Random
    using StaticArrays
    using Profile
    using BenchmarkTools
end

Agents.@agent struct AgentEscapes(ContinuousAgent{2, Float64})
    age::Float64
    mass::Float64
    toxicload::Float64
    path::Vector{Tuple{Int,Int}}
    pathX::Vector{Float64}
    pathY::Vector{Float64}
    TL1::Vector{Float64}
    TL2::Vector{Float64}
    TL3::Vector{Float64}
end

begin   # Φόρτωση heightmap και concentration maps 1..10
    heightmap_data = load("NADEEN/Maps/Qatargas Map.jpg")
    heightmap_data = permutedims(channelview(heightmap_data), [2,3,1])[:,:,1]
    global heightmap = floor.(Int, convert.(Float64, heightmap_data) * 255)
    heightmap = 255 .- heightmap

    const NUM_CMS = 10
    cm_list = Vector{Array{Float64,2}}(undef, NUM_CMS)
    for k in 1:NUM_CMS
        fn = joinpath("Concentration Maps", string(k) * ".bmp")
        img = load(fn)
        img = permutedims(channelview(img), [2,3,1])[:,:,1]
        cm_list[k] = convert.(Float64, img) .* 500.0
    end
    for k in 1:NUM_CMS
        @assert size(cm_list[k]) == size(heightmap) "Concentration map $k size mismatch with heightmap"
    end
    global penalty_map = copy(cm_list[1])
end

NPM = heightmap .+ penalty_map
NPM_int = round.(Int, NPM)

begin   # Παράμετροι μοντέλου
    const METERS_TO_PIXELS = 0.2692   # 1250 m ≈ 336.5 px on map
    dt = 1.0
    n_agents = 50
    toxicity_rate = 0.07
    age_range = (22, 60)
    speed_range = (4.0, 7.0)
    speed = 5.0
    mass_range = (50, 80)
    ag_range_y = (size(heightmap)[1]/4):(3*size(heightmap)[1]/4)
    ag_range_x = (size(heightmap)[2]/4):(3*size(heightmap)[2]/4)
    dims = (size(NPM))
    walkmap = BitArray(trues(dims...))
    white_threshold = 245
    walkmap[heightmap .> white_threshold] .= false
end

dests = [(600., 980.), (100., 200.)]
space = ContinuousSpace(size(NPM); periodic = false, spacing = 1)

begin
    cost_metric_obj = AbsolutePenaltyMap(NPM_int, MaxDistance{2}())
    cost_metric_str = "APM"
    heuristic_code  = "DF"
    pathfinderPM = DStarLite(space; walkmap = walkmap, cost_metric = cost_metric_obj)
    properties = (
        pathfinderPM = pathfinderPM,
        heightmap    = heightmap,
        dt           = dt,
        speed_range  = speed_range,
        goal         = dests,
    )
end

function move_along_precomputed_path!(agent::AgentEscapes, speed, dt)
    isempty(agent.path) && return
    cell = agent.path[1]
    target = SVector{2,Float64}(cell[1], cell[2])
    dir = target .- agent.pos
    dist = norm(dir)
    if dist < speed * dt
        agent.pos = target
        popfirst!(agent.path)
    else
        agent.pos += (dir / dist) * speed * dt
    end
end

function agent_step!(person, model)
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
    move_along_precomputed_path!(person, speed * METERS_TO_PIXELS, model.dt)
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
    rng          = MersenneTwister(),
    properties   = properties,
    agent_step!  = agent_step!,
    model_step!  = model_step!
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
            println("Warning: Could not find valid walkable spawn position for agent in Scenario 3 after $max_attempts attempts")
            continue
        end
        add_agent!(
            pos, AgentEscapes, model,
            vel, age, mass, 1.0,
            Tuple{Int,Int}[], [pos[1]], [pos[2]],
            [0.0], [0.0], [0.0]
        )
    end
end

function setupToxic()
    Atime = [0.0, 0.17, 0.83, 1.67, 4.17, 8.33]
    Arho = zeros(3, 6)
    Arho[1, 2:6] = [4.85, 4.23, 4.17, 4.06, 3.82]
    Arho[2, 2:6] = [180.79, 157.56, 155.43, 151.37, 142.48]
    Arho[3, 2:6] = [485.62, 423.22, 417.49, 406.59, 382.71]
    MW = 34
    Arho *= MW/24.04
    Arho = Arho'
    Atime = Atime*60
    taumin = 200.
    taumax = 86400.
    Brho = zeros(7,3)
    Balpha = zeros(7,3)
    rhomax = zeros(1,3)
    rhomin = zeros(1,3)
    Btime = zeros(7, 3)
    for k=1:3
        for b=2:6
            Brho[b, k] = Arho[b, k]
            Btime[b, k] = Atime[b]
        end
        Btime[1, k] = taumin
        Btime[7, k] = taumax
    end
    for k=1:3
        for b=3:6
            if Brho[b-1, k]==Brho[b, k]
                Balpha[b, k] = 0.0
            else
                Balpha[b, k] = log(Atime[b]/Atime[b-1])/log(Brho[b-1, k]/Brho[b, k])
            end
        end
        Balpha[2, k] = Balpha[3, k]
        Balpha[1, k] = Balpha[2, k]
        Balpha[7, k] = Balpha[6, k]
    end
    for k=1:3
        if Balpha[3, k]==0
            rhomax[k] = Brho[2, k]
            Brho[1, k] = rhomax[k]
        else
            rhomax[k] = Brho[2, k]*(Btime[2, k]/taumin)^(1/Balpha[2, k])
            Brho[1, k] = rhomax[k]
        end
        if Brho[5, k]==Brho[6, k]
            rhomin[k] = Brho[6, k]
            Brho[7, k] = rhomin[k]
        else
            rhomin[k] = Brho[6, k]*(Btime[6, k]/taumax)^(1/Balpha[6, k])
            Brho[7, k] = rhomin[k]
        end
    end
    for k=1:3
        for b=2:7
            if Balpha[b, k] == 0
                Btime[b, k] = Btime[b-1, k]
            end
        end
    end
    for k=1:3
        for b=3:5
            if Balpha[b-1, k]==0 && Balpha[b, k]>0
                Balpha[b, k]=log(Btime[b, k]/Btime[b-1, k])/log(Brho[b-1, k]/Brho[b, k])
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
                        TL_rate = (1/Btime[i, k])*((Ct/Brho[k, i])^(Balpha[i, k]))
                    end
                end
            end
        end
        TL[k] = TL[k] .+ TL_rate * dt
    end
    TL[1] = TL[1] > 1.0 ? 1.0 : TL[1]
    TL[2] = TL[2] > 1. ? 1.0 : TL[2]
    TL[3] = TL[3] > 1. ? 1.0 : TL[3]
    return TL
end

@inline function world_to_cell(p::Tuple{Float64,Float64}, dims::Tuple{Int,Int})
    i = clamp(Int(floor(p[1])), 1, dims[1])
    j = clamp(Int(floor(p[2])), 1, dims[2])
    return (i, j)
end

@inline function world_to_cell(p::SVector{2,Float64}, dims::Tuple{Int,Int})
    i = clamp(Int(floor(p[1])), 1, dims[1])
    j = clamp(Int(floor(p[2])), 1, dims[2])
    return (i, j)
end

function find_changed_cells(old::AbstractMatrix{Int}, new::AbstractMatrix{Int})
    cells = Tuple{Int,Int}[]
    @inbounds for i in axes(old,1), j in axes(old,2)
        old[i,j] != new[i,j] && push!(cells, (i,j))
    end
    return cells
end

function choose_best_path(paths::Vector{Vector{Tuple{Int,Int}}})
    filter!(!isempty, paths)
    isempty(paths) && return Tuple{Int,Int}[]
    return paths[argmin(length.(paths))]
end

# --- Benchmark parameters (like Scenario 2 Benchmarking) ---
const BENCHMARK_MAX_RUNS = 100
const BENCHMARK_CONVERGENCE_PCT = 0.01
const BENCHMARK_MIN_RUNS = 50
const BENCHMARK_T_STEPS = 1852

"""
    run_one_benchmark() -> (initial_pf_time_s, incremental_pf_time_s, simulation_time_s, seed, total_TL)

Build a fresh model with random seed, add agents, time the initial D* Lite
planning (planner init + first path extraction), then run BENCHMARK_T_STEPS
with dynamic concentration maps (1→10) while timing all incremental replanning
work separately from pure simulation stepping. Returns:

- initial_pf_time_s: time spent on initial pathfinding (s)
- incremental_pf_time_s: time spent on all incremental replans (s)
- simulation_time_s: time spent stepping the model (s)
- seed: RNG seed for this run
- total_TL: total toxic load across agents at the end of the run

No CSV/video is written from this function; higher-level code handles aggregation/Excel.
"""
function run_one_benchmark()
    seed = rand(Random.RandomDevice(), UInt32)
    rng_run = MersenneTwister(seed)

    local_NPM_int = copy(NPM_int)
    local_prev_NPM_int = copy(local_NPM_int)
    local_cost_metric_obj = AbsolutePenaltyMap(local_NPM_int, MaxDistance{2}())
    local_pathfinderPM = DStarLite(space; walkmap = walkmap, cost_metric = local_cost_metric_obj)
    local_properties = (
        pathfinderPM = local_pathfinderPM,
        heightmap    = heightmap,
        dt           = dt,
        speed_range  = speed_range,
        goal         = dests,
    )

    model_run = ABM(
        AgentEscapes,
        space;
        rng         = rng_run,
        properties  = local_properties,
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
            Tuple{Int,Int}[], [pos[1]], [pos[2]],
            [0.0], [0.0], [0.0]
        )
    end

    grid_dims  = size(NPM_int)
    exit_cells = [world_to_cell(g, grid_dims) for g in dests]

    planner_init_time = @elapsed begin
        local_planners = [
            init_planner(local_pathfinderPM, goal_cell)
            for goal_cell in exit_cells
        ]
    end

    path_extract_time = @elapsed begin
        for a in allagents(model_run)
            start = world_to_cell(a.pos, grid_dims)
            paths = [extract_path(p, start) for p in local_planners]
            a.path = choose_best_path(paths)
        end
    end

    replan_time = 0.0
    sim_step_time = 0.0
    NUM_MAPS = 10
    frames_per_map = 60

    # Ensure global penalty_map and local_NPM_int start from CM1 for this run
    current_map_idx = 1
    penalty_map .= cm_list[current_map_idx]
    @. local_NPM_int = Int(round(heightmap + penalty_map))
    local_prev_NPM_int .= local_NPM_int
    local_pathfinderPM.cost_metric.pmap .= local_NPM_int

    for step_idx in 1:BENCHMARK_T_STEPS
        sim_step_time += @elapsed step!(model_run, 1)
        new_idx = min(NUM_MAPS, Int(ceil(step_idx / frames_per_map)))
        if new_idx != current_map_idx
            current_map_idx = new_idx
            replan_time += @elapsed begin
                # Update concentration map and the underlying integer cost map
                penalty_map .= cm_list[current_map_idx]
                @. local_NPM_int = Int(round(heightmap + penalty_map))

                # Incremental D* Lite update: identify changed cells and update planners in-place
                changed_cells = find_changed_cells(local_prev_NPM_int, local_NPM_int)
                local_prev_NPM_int .= local_NPM_int
                local_pathfinderPM.cost_metric.pmap .= local_NPM_int
                for planner in local_planners
                    update_after_cm_change!(planner, changed_cells)
                end

                # Re-extract paths for all agents using the updated planners
                for a in allagents(model_run)
                    start = world_to_cell(a.pos, size(local_NPM_int))
                    paths = [extract_path(p, start) for p in local_planners]
                    a.path = choose_best_path(paths)
                end
            end
        end
    end

    initial_pf_time = planner_init_time + path_extract_time
    incremental_pf_time = replan_time
    simulation_time  = sim_step_time

    # Sum toxic load across all agents at the end of the run
    total_TL = 0.0
    for a in allagents(model_run)
        total_TL += a.toxicload
    end

    return initial_pf_time, incremental_pf_time, simulation_time, seed, total_TL
end

"""
    profile_one_benchmark()
Profile one benchmark run. After running, call Profile.print() or your profiler GUI.
"""
function profile_one_benchmark()
    Profile.clear()
    init_t, incr_t, sim_t, run_seed, total_TL = run_one_benchmark()
    println("Profiled run (seed=$run_seed): initial PF = $(round(init_t; digits=4)) s, incremental PF = $(round(incr_t; digits=4)) s, simulation = $(round(sim_t; digits=4)) s, total TL = $(round(total_TL; digits=4))")
    println("Run Profile.print() or open profiler to inspect hotspots.")
end

"""
    benchmark_run_one_benchmark(; samples = 10)

Use BenchmarkTools to benchmark a single `run_one_benchmark()` invocation.
Returns the BenchmarkTools Trial object and also prints a summary.
"""
function benchmark_run_one_benchmark(; samples::Int = 10)
    println("Benchmarking run_one_benchmark() with BenchmarkTools (samples = $samples)...")
    result = @benchmark run_one_benchmark() samples = samples
    println(result)
    return result
end

# --- Benchmark loop (like Scenario 2 Benchmarking) ---
initial_pf_times   = Float64[]
incremental_pf_times = Float64[]
simulation_times   = Float64[]
total_TL_values    = Float64[]
seeds = UInt32[]
avg_pathfinding_prev = 0.0

println("Benchmark: Scenario 3 with D* Lite (pathfinding + simulation). Max runs = $BENCHMARK_MAX_RUNS, convergence = $(BENCHMARK_CONVERGENCE_PCT*100)%.")
for run_id in 1:BENCHMARK_MAX_RUNS
    t_init, t_incr, t_sim, run_seed, total_TL = run_one_benchmark()
    push!(initial_pf_times,    t_init)
    push!(incremental_pf_times, t_incr)
    push!(simulation_times,     t_sim)
    push!(total_TL_values,      total_TL)
    push!(seeds, run_seed)

    n = length(initial_pf_times)
    avg_initial_pf    = sum(initial_pf_times)    / n
    avg_incremental_pf = sum(incremental_pf_times) / n
    avg_simulation    = sum(simulation_times)    / n
    converged = n >= BENCHMARK_MIN_RUNS && avg_pathfinding_prev > 0 &&
        (abs((avg_initial_pf + avg_incremental_pf) - avg_pathfinding_prev) / avg_pathfinding_prev <= BENCHMARK_CONVERGENCE_PCT)

    println("  Run $run_id (seed=$run_seed): initial PF = $(round(t_init; digits=4)) s, incremental PF = $(round(t_incr; digits=4)) s, simulation = $(round(t_sim; digits=4)) s, total TL = $(round(total_TL; digits=4))  (avg total PF = $(round(avg_initial_pf + avg_incremental_pf; digits=4)) s)")
    global avg_pathfinding_prev = avg_initial_pf + avg_incremental_pf

    if converged
        println("Converged at run $run_id (pathfinding avg change < $(BENCHMARK_CONVERGENCE_PCT*100)%).")
        break
    end
end

n_runs = length(initial_pf_times)
avg_initial_pf_final    = sum(initial_pf_times)    / n_runs
avg_incremental_pf_final = sum(incremental_pf_times) / n_runs
avg_pathfinding_final   = avg_initial_pf_final + avg_incremental_pf_final
avg_simulation_final  = sum(simulation_times)  / n_runs

xlsx_path = "SCENARIOS/SCENARIO 3/Simulation Results/DStarLite_SCENARIO_3_Benchmarking_$(n_agents)_$(cost_metric_str)_$(heuristic_code).xlsx"
run_ids = collect(1:n_runs)
columns_data = [run_ids, seeds, initial_pf_times, incremental_pf_times, simulation_times, total_TL_values]
column_names = ["Run", "Seed", "InitialPathfindingTime_s", "IncrementalPathfindingTime_s", "SimulationTime_s", "Total TL"]
XLSX.writetable(xlsx_path, columns_data, column_names; sheetname = "Runs", overwrite = true)
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
end

println("Benchmark complete. Results written to $xlsx_path")
println("  Total runs: $n_runs | Avg initial PF: $(round(avg_initial_pf_final; digits=4)) s | Avg incremental PF: $(round(avg_incremental_pf_final; digits=4)) s | Avg simulation: $(round(avg_simulation_final; digits=4)) s")
