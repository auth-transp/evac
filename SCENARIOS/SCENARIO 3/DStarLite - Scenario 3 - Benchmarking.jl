begin   # Φόρτωση των απαραίτητων βιβλιοθηκών
    using Agents
    using Agents.Pathfinding
    using Dates
    using FileIO: load
    using XLSX
    using ImageMagick
    using Images
    using Random
    using StaticArrays
    using Profile
    using Statistics
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

begin   # Φόρτωση heightmap και concentration maps 1..7
    heightmap_data = load("NADEEN/Maps/Qatargas Map.jpg")
    heightmap_data = permutedims(channelview(heightmap_data), [2,3,1])[:,:,1]
    global heightmap = floor.(Int, convert.(Float64, heightmap_data) * 255)
    heightmap = 255 .- heightmap

    const NUM_CMS = 7
    cm_list = Vector{Array{Float64,2}}(undef, NUM_CMS)
    for k in 1:NUM_CMS
        fn = joinpath("Concentration Maps", string(k) * ".bmp")
        img = load(fn)
        img = permutedims(channelview(img), [2,3,1])[:,:,1]
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
    n_agents = 50
    
    toxicity_rate = 0.07
    age_range = (22, 60)
    speed_range = (4.0, 7.0)
    mass_range = (50, 80)
    ag_range_y = (size(heightmap)[1]/4):(3*size(heightmap)[1]/4)
    ag_range_x = (size(heightmap)[2]/4):(3*size(heightmap)[2]/4)
    dims = (size(NPM))
    walkmap = BitArray(trues(dims...))
    white_threshold = 245
    walkmap[heightmap .> white_threshold] .= false
end

dests = [(500., 854.), (120., 248.)]
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
    # dist == 0: already on waypoint; must pop without dividing (speed==0 ⇒ 0 < speed*dt is false and 0/0 → NaN)
    if iszero(dist) || dist < speed * dt
        agent.pos = target
        popfirst!(agent.path)
    else
        agent.pos += (dir / dist) * speed * dt
    end
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
    Atime = [0.0, 2.0, 5.0, 8.0, 15.0, 30.0]
    Arho = zeros(3, 6)
    Arho[1, 2:6] = [2.87, 2.33, 2.09, 1.81, 1.55]
    Arho[2, 2:6] = [202.38, 164.38, 147.74, 128.10, 109.45]
    Arho[3, 2:6] = [286.71, 232.87, 209.30, 181.47, 155.05]
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

# --- Benchmark parameters ---
const BENCHMARK_ALGORITHM = "DStarLite"
const BENCHMARK_MAX_RUNS = 100
const BENCHMARK_AGENT_COUNTS = [5, 10, 25, 50, 75, 100]   # one Excel sheet per count
const BENCHMARK_TL_CAP_PER_AGENT = 3.0
const TL_INCAPACITATED = 3.0          # TL at which speed drops to 0 (see agent_step!)
const PROJECTION_MAX_STEPS = 10800    # max extra (untimed) steps (3 h) to resolve stranded agents after T

# Horizon T = 2500 s and CM switch times (s) — Thesis, Table 7: 7 CMs, switching completes at 70% of T.
# t = (step_idx - 1) * dt
const BENCHMARK_T_STEPS = 2500
const CM_SWITCH_TIMES = [0.0, 292.0, 584.0, 876.0, 1168.0, 1460.0, 1752.0]
@assert length(CM_SWITCH_TIMES) == NUM_CMS "CM_SWITCH_TIMES must have one entry per concentration map"

# Warm-up run (JIT compilation) — its results are discarded. It only needs to reach the
# first CM switch so that the incremental replanning code path is compiled too.
const WARMUP_T_STEPS = Int(CM_SWITCH_TIMES[2] / dt) + 2

"""
    run_one_benchmark(; num_agents, t_steps = BENCHMARK_T_STEPS, project = true) -> (run_metrics, agent_rows)

Build a fresh model with a random seed and `num_agents` agents, time the initial D* Lite
planning (planner init + first path extraction), then run `t_steps` steps with dynamic
concentration maps (1→NUM_CMS), timing all incremental replanning separately from pure
simulation stepping. Pathfinding allocations / GC time are measured with `@timed`.
Agent outcomes (evacuated / incapacitated / stranded + projection) are tracked outside the
timed regions — see `evaluate_outcomes!`.

No CSV/video is written from this function; higher-level code handles aggregation/Excel.
"""
function run_one_benchmark(; num_agents::Int, t_steps::Int = BENCHMARK_T_STEPS, project::Bool = true)
    seed = rand(Random.RandomDevice(), UInt32)
    rng_run = MersenneTwister(seed)

    local_NPM_int = copy(NPM_int)
    local_prev_NPM_int = copy(local_NPM_int)
    # Fresh space per run: ContinuousSpace keeps its own spatial index of agent ids, so a shared
    # space would carry stale ids from earlier models (warm-up / previous runs) into this one.
    local_space = ContinuousSpace(size(NPM); periodic = false, spacing = 1)
    local_cost_metric_obj = AbsolutePenaltyMap(local_NPM_int, MaxDistance{2}())
    local_pathfinderPM = DStarLite(local_space; walkmap = walkmap, cost_metric = local_cost_metric_obj)
    local_properties = (
        pathfinderPM = local_pathfinderPM,
        heightmap    = heightmap,
        dt           = dt,
        speed_range  = speed_range,
        goal         = dests,
    )

    model_run = ABM(
        AgentEscapes,
        local_space;
        rng         = rng_run,
        properties  = local_properties,
        agent_step! = agent_step!,
        model_step! = model_step!,
    )

    for _ in 1:num_agents
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

    tm_init = @timed [init_planner(local_pathfinderPM, goal_cell) for goal_cell in exit_cells]
    local_planners = tm_init.value

    tm_extract = @timed for a in allagents(model_run)
        start = world_to_cell(a.pos, grid_dims)
        paths = [extract_path(p, start) for p in local_planners]
        a.path = choose_best_path(paths)
    end

    initial_pf_time = tm_init.time + tm_extract.time
    pf_bytes = tm_init.bytes + tm_extract.bytes
    pf_gc = tm_init.gctime + tm_extract.gctime
    replan_time = 0.0
    sim_step_time = 0.0
    evac_t = Dict{Int,Float64}()
    tl3_t = Dict{Int,Float64}()

    # Ensure global penalty_map and local_NPM_int start from CM1 for this run
    current_map_idx = 1
    penalty_map .= cm_list[current_map_idx]
    @. local_NPM_int = Int(round(heightmap + penalty_map))
    local_prev_NPM_int .= local_NPM_int
    local_pathfinderPM.cost_metric.pmap .= local_NPM_int

    for step_idx in 1:t_steps
        sim_step_time += @elapsed step!(model_run, 1)
        track_outcomes!(evac_t, tl3_t, model_run, step_idx * dt)
        t_sim = (step_idx - 1) * dt
        new_idx = searchsortedlast(CM_SWITCH_TIMES, t_sim)
        if new_idx != current_map_idx
            current_map_idx = new_idx
            tm_replan = @timed begin
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
            replan_time += tm_replan.time
            pf_bytes += tm_replan.bytes
            pf_gc += tm_replan.gctime
        end
    end

    outcomes, agent_rows = evaluate_outcomes!(
        model_run, a -> a.path, evac_t, tl3_t, t_steps * dt; project = project,
    )
    timings = (
        Seed            = Int(seed),
        InitialPF_s     = initial_pf_time,
        IncrementalPF_s = replan_time,
        TotalPF_s       = initial_pf_time + replan_time,
        Simulation_s    = sim_step_time,
        PF_Alloc_MB     = pf_bytes / 2^20,
        PF_GC_s         = pf_gc,
    )
    return merge(timings, outcomes), agent_rows
end

# ---------------- Outcome tracking: Evacuated / Incapacitated / Stranded ----------------

"""
    track_outcomes!(evac_t, tl3_t, model, t_now)

Record the first time `t_now` at which each agent reaches an exit (`evac_t`) or
TL ≥ TL_INCAPACITATED (`tl3_t`). Called after every `step!`, outside the timed regions.
"""
function track_outcomes!(evac_t::Dict{Int,Float64}, tl3_t::Dict{Int,Float64}, model, t_now::Float64)
    for a in allagents(model)
        (haskey(evac_t, a.id) || haskey(tl3_t, a.id)) && continue
        if at_goal(a.pos, model.goal)
            evac_t[a.id] = t_now
        elseif a.toxicload >= TL_INCAPACITATED
            tl3_t[a.id] = t_now
        end
    end
    return nothing
end

"Length (m) of the remaining route from `pos` through `waypoints`."
function route_length_m(pos, waypoints)
    L = 0.0
    px, py = Float64(pos[1]), Float64(pos[2])
    for w in waypoints
        wx, wy = Float64(w[1]), Float64(w[2])
        L += hypot(wx - px, wy - py)
        px, py = wx, wy
    end
    return L / METERS_TO_PIXELS
end

_mean_or_missing(v) = isempty(v) ? missing : mean(v)
_max_or_missing(v) = isempty(v) ? missing : maximum(v)

"""
    evaluate_outcomes!(model_run, route_of, evac_t, tl3_t, t_horizon; project = true)
        -> (outcome_metrics, agent_rows)

Classifies every agent at the horizon T:

- Evacuated:     reached an exit within T
- Incapacitated: reached TL ≥ TL_INCAPACITATED within T
- Stranded:      still en route at T with TL < TL_INCAPACITATED

Stranded agents are then resolved by a projection: the model keeps running after T
under the last concentration map (no replanning, NOT timed) for at most
PROJECTION_MAX_STEPS, until every agent has evacuated, become incapacitated, or has no
route left. `Proj_*` fields describe that projected outcome (for agents already resolved
within T it equals the outcome at T). `route_of(a)` returns the agent's remaining waypoints.
"""
function evaluate_outcomes!(model_run, route_of, evac_t, tl3_t, t_horizon; project::Bool = true)
    agents_sorted = sort(collect(allagents(model_run)); by = a -> a.id)
    snap_tl = Dict(a.id => a.toxicload for a in agents_sorted)
    snap_dist = Dict(a.id => route_length_m(a.pos, route_of(a)) for a in agents_sorted)
    evac_T = copy(evac_t)
    tl3_T = copy(tl3_t)

    proj_steps = 0
    if project
        is_open = a -> !(haskey(evac_t, a.id) || haskey(tl3_t, a.id) || isempty(route_of(a)))
        while proj_steps < PROJECTION_MAX_STEPS && any(is_open, allagents(model_run))
            step!(model_run, 1)
            proj_steps += 1
            track_outcomes!(evac_t, tl3_t, model_run, t_horizon + proj_steps * dt)
        end
    end

    rows = NamedTuple[]
    for a in agents_sorted
        outcome_T = haskey(evac_T, a.id) ? "Evacuated" :
                    haskey(tl3_T, a.id)  ? "Incapacitated" : "Stranded"
        proj_outcome = haskey(evac_t, a.id) ? "Evacuated" :
                       haskey(tl3_t, a.id)  ? "Incapacitated" :
                       isempty(route_of(a)) ? "NoRoute" : "Unresolved"
        proj_time = haskey(evac_t, a.id) ? evac_t[a.id] :
                    haskey(tl3_t, a.id)  ? tl3_t[a.id] : missing
        push!(rows, (
            AgentID           = a.id,
            Outcome_T         = outcome_T,
            TL_T              = snap_tl[a.id],
            TL_T_capped       = min(snap_tl[a.id], BENCHMARK_TL_CAP_PER_AGENT),
            EvacTime_s        = get(evac_T, a.id, missing),
            TL3Time_s         = get(tl3_T, a.id, missing),
            RemainingDist_T_m = outcome_T == "Evacuated" ? 0.0 : snap_dist[a.id],
            Proj_Outcome      = proj_outcome,
            Proj_Time_s       = proj_time,
            Proj_TL           = a.toxicload,
        ))
    end

    stranded = [r for r in rows if r.Outcome_T == "Stranded"]
    metrics = (
        SumTL_capped                  = sum((r.TL_T_capped for r in rows); init = 0.0),
        Evacuated                     = count(r -> r.Outcome_T == "Evacuated", rows),
        Incapacitated                 = count(r -> r.Outcome_T == "Incapacitated", rows),
        Stranded                      = length(stranded),
        EvacTime_mean_s               = _mean_or_missing(collect(values(evac_T))),
        EvacTime_max_s                = _max_or_missing(collect(values(evac_T))),
        Stranded_TL_mean              = _mean_or_missing([r.TL_T for r in stranded]),
        Stranded_RemainingDist_mean_m = _mean_or_missing([r.RemainingDist_T_m for r in stranded]),
        Proj_Evacuated                = count(r -> r.Proj_Outcome == "Evacuated", rows),
        Proj_Incapacitated            = count(r -> r.Proj_Outcome == "Incapacitated", rows),
        Proj_NoRoute                  = count(r -> r.Proj_Outcome == "NoRoute", rows),
        Proj_Unresolved               = count(r -> r.Proj_Outcome == "Unresolved", rows),
        Proj_EvacTime_max_s           = _max_or_missing(collect(values(evac_t))),
        Proj_Steps                    = proj_steps,
    )
    return metrics, rows
end

# ---------------- Excel export: Summary + one sheet per agent count + AgentOutcomes ----------------

const STAT_NAMES = ("Mean", "Median", "Std", "Min", "Max")
const SUMMARY_METRICS = (
    :InitialPF_s, :IncrementalPF_s, :TotalPF_s, :Simulation_s, :PF_Alloc_MB,
    :SumTL_capped, :Evacuated, :Incapacitated, :Stranded,
    :Proj_Evacuated, :Proj_Incapacitated, :Proj_Unresolved,
)

function column_stats(v)
    x = Float64[Float64(y) for y in v if !ismissing(y)]
    isempty(x) && return (missing, missing, missing, missing, missing)
    return (mean(x), median(x), length(x) > 1 ? std(x) : missing, minimum(x), maximum(x))
end

function rows_to_columns(rows)
    ks = collect(keys(first(rows)))
    return ks, [[r[k] for r in rows] for k in ks]
end

"""
    write_benchmark_xlsx(path, algorithm, results)

`results` is a vector of `(n, runs, agents)`. Rewrites the whole workbook, so it is called
after every completed agent count and the file always holds everything finished so far.
"""
function write_benchmark_xlsx(path, algorithm, results)
    XLSX.openxlsx(path, mode = "w") do xf
        summary = xf[1]
        XLSX.rename!(summary, "Summary")
        all_stats = Dict{Symbol,Any}[]

        for res in results
            sh = XLSX.addsheet!(xf, "N$(res.n)")
            ks, cols = rows_to_columns(res.runs)
            XLSX.writetable!(sh, cols, string.(ks))
            # statistics block under the runs (blank row in between), aligned with the columns
            r0 = length(res.runs) + 3
            for (si, sname) in enumerate(STAT_NAMES)
                sh[r0 + si - 1, 1] = sname
            end
            stats = Dict{Symbol,Any}()
            for (ci, k) in enumerate(ks)
                k in (:Run, :Seed) && continue
                st = column_stats(cols[ci])
                stats[k] = st
                for si in eachindex(STAT_NAMES)
                    ismissing(st[si]) || (sh[r0 + si - 1, ci] = st[si])
                end
            end
            push!(all_stats, stats)
        end

        s_names = ["NAgents", "Runs"]
        s_cols = Any[[res.n for res in results], [length(res.runs) for res in results]]
        for m in SUMMARY_METRICS, (si, suffix) in ((1, "mean"), (2, "median"), (3, "std"))
            push!(s_names, "$(m)_$(suffix)")
            push!(s_cols, [stats[m][si] for stats in all_stats])
        end
        XLSX.writetable!(summary, s_cols, s_names)

        config = [
            ("Algorithm", algorithm),
            ("Runs per agent count", BENCHMARK_MAX_RUNS),
            ("Horizon T (steps)", BENCHMARK_T_STEPS),
            ("dt (s)", dt),
            ("CM switch times (s)", join(CM_SWITCH_TIMES, ", ")),
            ("Warm-up (excluded from results)", "1 run, $WARMUP_T_STEPS steps"),
            ("Incapacitation threshold (TL)", TL_INCAPACITATED),
            ("Projection after T", "last CM kept, no replanning, not timed, max $PROJECTION_MAX_STEPS steps"),
            ("Seeds", "random per run (RandomDevice)"),
        ]
        c0 = length(results) + 3
        for (i, (label, value)) in enumerate(config)
            summary[c0 + i - 1, 1] = label
            summary[c0 + i - 1, 2] = value
        end

        ag = XLSX.addsheet!(xf, "AgentOutcomes")
        all_agents = reduce(vcat, [res.agents for res in results])
        ks, cols = rows_to_columns(all_agents)
        XLSX.writetable!(ag, cols, string.(ks))
    end
end

"""
    profile_one_benchmark(; num_agents = first(BENCHMARK_AGENT_COUNTS))
Profile one benchmark run. After running, call Profile.print() or your profiler GUI.
"""
function profile_one_benchmark(; num_agents::Int = first(BENCHMARK_AGENT_COUNTS))
    Profile.clear()
    r, _ = @profile run_one_benchmark(; num_agents = num_agents, project = false)
    println("Profiled run (seed=$(r.Seed), N=$num_agents): initial PF = $(round(r.InitialPF_s; digits=4)) s, incremental PF = $(round(r.IncrementalPF_s; digits=4)) s, simulation = $(round(r.Simulation_s; digits=4)) s")
    println("Run Profile.print() or open profiler to inspect hotspots.")
end

"""
    benchmark_run_one_benchmark(; num_agents = first(BENCHMARK_AGENT_COUNTS), samples = 10)

Use BenchmarkTools to benchmark a single `run_one_benchmark()` invocation.
Returns the BenchmarkTools Trial object and also prints a summary.
"""
function benchmark_run_one_benchmark(; num_agents::Int = first(BENCHMARK_AGENT_COUNTS), samples::Int = 10)
    println("Benchmarking run_one_benchmark() with BenchmarkTools (N = $num_agents, samples = $samples)...")
    result = @benchmark run_one_benchmark(; num_agents = $num_agents, project = false) samples = samples
    println(result)
    return result
end

# ---------------- Benchmark sweep over agent counts ----------------

run_timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
xlsx_path = "SCENARIOS/SCENARIO 3/Simulation Results/$(BENCHMARK_ALGORITHM) Benchmarking_$(run_timestamp).xlsx"

println("Benchmark: Scenario 3 with $BENCHMARK_ALGORITHM. Agent counts = $BENCHMARK_AGENT_COUNTS, runs per count = $BENCHMARK_MAX_RUNS.")

# Warm-up (JIT compilation): executed once, results discarded — not part of the exported statistics.
println("  Warm-up run ($WARMUP_T_STEPS steps, excluded from results)...")
warmup_elapsed = @elapsed run_one_benchmark(; num_agents = first(BENCHMARK_AGENT_COUNTS), t_steps = WARMUP_T_STEPS, project = false)
println("  Warm-up done in $(round(warmup_elapsed; digits=2)) s.")

benchmark_results = NamedTuple[]
for n in BENCHMARK_AGENT_COUNTS
    println("=== N = $n agents ===")
    runs = NamedTuple[]
    agents = NamedTuple[]
    for run_id in 1:BENCHMARK_MAX_RUNS
        r, agent_rows = run_one_benchmark(; num_agents = n)
        push!(runs, merge((Run = run_id,), r))
        append!(agents, [merge((NAgents = n, Run = run_id, Seed = r.Seed), row) for row in agent_rows])
        println("  Run $run_id (seed=$(r.Seed)): initial PF = $(round(r.InitialPF_s; digits=4)) s, incremental PF = $(round(r.IncrementalPF_s; digits=4)) s, simulation = $(round(r.Simulation_s; digits=4)) s, alloc PF = $(round(r.PF_Alloc_MB; digits=1)) MB | evacuated $(r.Evacuated), incapacitated $(r.Incapacitated), stranded $(r.Stranded) → projected evac $(r.Proj_Evacuated) / incap $(r.Proj_Incapacitated) / unresolved $(r.Proj_Unresolved)")
    end
    push!(benchmark_results, (n = n, runs = runs, agents = agents))
    # Saved after every agent count, so a crash never loses completed counts.
    write_benchmark_xlsx(xlsx_path, BENCHMARK_ALGORITHM, benchmark_results)
    tot = [r.TotalPF_s for r in runs]
    println("  N = $n done: total PF mean = $(round(mean(tot); digits=4)) s, median = $(round(median(tot); digits=4)) s. Saved to $xlsx_path")
end

println("Benchmark complete. Results written to $xlsx_path")
