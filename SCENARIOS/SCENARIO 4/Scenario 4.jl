begin   # Φόρτωση των απαραίτητων βιβλιοθηκών
    using Agents
    using Agents.Pathfinding
    using CSV
    using CairoMakie
    using DataFrames
    using Dates
    using FileIO: load
    using ImageMagick
    using Images
    using InteractiveDynamics
    using Makie
    using Observables
    using Random
    using StaticArrays

    include(joinpath(@__DIR__, "..", "append_run_history.jl"))
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


begin   # Φόρτωση του heightmap και των hand-drawn penalty maps (και όλων των CM1..CM7)
    # heightmap (unchanged)
    heightmap_data = load("NADEEN/Maps/Qatargas Map.jpg")
    heightmap_data = permutedims(channelview(heightmap_data), [2,3,1])[:,:,1]
    global heightmap = floor.(Int, convert.(Float64, heightmap_data) * 255)
    heightmap = 255 .- heightmap   # αυτό κάνει την αντιστροφή

    # --- Load all concentration maps 1..7 as Float64 arrays ---
    const NUM_CMS = 7
    cm_list = Vector{Array{Float64,2}}(undef, NUM_CMS)
    for k in 1:NUM_CMS
        fn = joinpath("Concentration Maps", string(k) * ".bmp")
        img = load(fn)
        img = permutedims(channelview(img), [2,3,1])[:,:,1]
        cm_list[k] = convert.(Float64, img) .* 255.0
    end

    # basic check: dimensions match heightmap
    for k in 1:NUM_CMS
        @assert size(cm_list[k]) == size(heightmap) "Concentration map $k size mismatch with heightmap"
    end

    # start with first CM
    global penalty_map = copy(cm_list[1])
end

NPM = heightmap .+ penalty_map
NPM_int = round.(Int, NPM)   # convert to Int for PenaltyMap


begin   # Αρχικοποίηση των παραμέτρων του μοντέλου
    const METERS_TO_PIXELS = 723.37 / 2500.0
    dt = 1.   ## discrete timestep each iteration of the model          # Define the dt variable as 1
    seed = 1544656427  ## seed for random number generator                     # Define the seed variable as 123
    n_agents = 10                                                        # Define the n_agents variable as 3
    toxicity_rate = 0.07                                               # Define the toxicity_rate variable as 0.07
    age_range = (22,60)                                                 # Define the age_range variable as a tuple of 22 and 60
    speed_range = (4.0,7.0)                                            # Define the speed_range variable as a tuple of 4.0 and 7.0
    mass_range = (50,80)                                                # Define the mass_range variable as a tuple of 50 and 80
    ag_range_y = (size(heightmap)[1]/4):(3*size(heightmap)[1]/4)    # Define the ag_range_y variable as a larger range of values from the heightmap array # [1] stands for the 1st row
    ag_range_x = (size(heightmap)[2]/4):(3*size(heightmap)[2]/4)    # Define the ag_range_x variable as a range of values from the heightmap array # [2] stands for the 2nd row
    dims = (size(NPM))                                            # Define the dims variable as the dimensions of the heightmap array (2xn matrix)
    walkmap = BitArray(trues(dims...))                                 # Define the walkmap variable as a BitArray of true values with the dimensions of the heightmap array
    
    # Set walkmap to false in white areas of the heightmap (after reversal, white = high values)
    # White areas are where heightmap value is above threshold (e.g., > 245)
    # Black and grey areas (low to medium values) remain walkable
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
Sorted by (K, indices lexicographically). Index in this list is stable for comparisons.
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
    println("Scenario 4: enumerated $(length(u)) valid goal combinations (K in 1:8, max 2 per side).")
    u
end

# Pick which combination this simulation uses (1:length(GOAL_COMBINATIONS)); edit to compare runs.
# Index 1794 = global goal indices [4, 7, 11, 15] (one per side A–D).
const SCENARIO4_GOAL_COMBO_INDEX = 1794

@assert 1 <= SCENARIO4_GOAL_COMBO_INDEX <= length(GOAL_COMBINATIONS) "SCENARIO4_GOAL_COMBO_INDEX must be in 1:$(length(GOAL_COMBINATIONS))"

active_dest_indices = copy(GOAL_COMBINATIONS[SCENARIO4_GOAL_COMBO_INDEX])
n_active_dests = length(active_dest_indices)
dests = DESTS_ALL[active_dest_indices]
println("Scenario 4 active goals [combination $SCENARIO4_GOAL_COMBO_INDEX / $(length(GOAL_COMBINATIONS))]: K=", n_active_dests, " | global indices ", active_dest_indices)
println("  coordinates: ", dests)

rng = MersenneTwister(seed)

## Note that the dimensions of the space do not have to correspond to the dimensions
## of the pathfinder. Discretisation is handled by the pathfinding methods
space = Pathfinding.ContinuousSpace(size(NPM); periodic = false, spacing = 1)


begin
    cost_metric_obj = AbsolutePenaltyMap(NPM_int, MaxDistance{2}())
    cost_metric_str = "APM"
    heuristic_code  = "DF"   # parity with DStarLite export names (A* uses default heuristic)
    pathfinderPM = AStar(space; walkmap = walkmap, cost_metric = cost_metric_obj)
    properties = (
        pathfinderPM = pathfinderPM,
        heightmap = heightmap,
        dt = dt,
        speed_range = speed_range,
        goal = dests,
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
    # Ct παίρνεται τώρα από το global penalty_map (hand-drawn maps)
    Ct = penalty_map[i, j]
    TLcurrent = [person.TL1[end], person.TL2[end], person.TL3[end]]
    TL = update_toxic_load(Ct, TLcurrent, model.dt)

    person.toxicload = sum(TL)
    push!(person.TL1, TL[1])
    push!(person.TL2, TL[2])
    push!(person.TL3, TL[3])

    # --- Speed update based on toxicload ---
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
  rng          = rng,
  properties   = properties,
  agent_step!  = agent_step!,
  model_step!  = model_step!
)

# Add agents (initial route planning is timed inside the @time block with the animation, like DStarLite)
for _ in 1:n_agents
    age = rand(abmrng(model))*(age_range[2]-age_range[1]) + age_range[1]
    mass = rand(abmrng(model)) * (mass_range[2]-mass_range[1]) + mass_range[1]
    vel = Tuple(rand(abmrng(model), 2) .* (speed_range[2]-speed_range[1]) .+ speed_range[1])
    
    # Keep trying to find a valid spawn position that is walkable (not in white/black areas)
    max_attempts = 1000
    attempts = 0
    pos = nothing
    while attempts < max_attempts
        candidate_pos = Tuple((rand(abmrng(model), floor.(ag_range_y)), rand(abmrng(model), floor.(ag_range_x))))
        pos_int = floor.(Int, candidate_pos)
        # Check if position is within bounds and walkable
        if 1 <= pos_int[1] <= size(walkmap, 1) && 1 <= pos_int[2] <= size(walkmap, 2) && walkmap[pos_int[1], pos_int[2]]
            pos = candidate_pos
            break
        end
        attempts += 1
    end
    
    # If we couldn't find a valid position after max_attempts, skip this agent
    if pos === nothing
        println("Warning: Could not find valid walkable spawn position for agent in Scenario 4 after $max_attempts attempts")
        continue
    end
    
    add_agent!(pos, AgentEscapes, model, vel, age, mass, 1., [pos[1]], [pos[2]], [0.0], [0.0], [0.0])
end


function setupToxic()                                               # Define the setupToxic function
    Atime = [0.0, 2.0, 5.0, 8.0, 15.0, 30.0] #min                # Define the Atime array as a 1x6 matrix                                    # Initialization of the 5 standard AEGL exposure times
    Arho = zeros(3, 6)                                              # Define the Arho array as a 3x6 matrix                                     # the concentration of the three symptoms compared to the AEGL concentrations
    Arho[1, 2:6] = [2.87, 2.33, 2.09, 1.81, 1.55]                   # Define the Arho array for the first row and columns 2 to 6                # odor
    Arho[2, 2:6] = [202.38, 164.38, 147.74, 128.10, 109.45]         # Define the Arho array for the second row and columns 2 to 6               # irritation
    Arho[3, 2:6] = [286.71, 232.87, 209.30, 181.47, 155.05] #ppm    # Define the Arho array for the third row and columns 2 to 6                # edema
    MW = 34 #Molecular weight of H2S in g/mol
    Arho *= MW/24.04 #mg/m^3                                        # Multiply the Arho array by the molecular weight of H2S divided by 24.04
    Arho = Arho'                                                    # Transpose the Arho array 
    Atime = Atime*60 #seconds                                       # Multiply the Atime array by 60 seconds to convert to seconds              
    taumin = 200.                                                   # Define the taumin variable as 200 seconds (Borris&Patnaik, 2014)          # shortest exposure time over which an AEGL 1, 2 or 3 onset can be reached
    taumax = 86400.                                                 # Define the taumax variable as 86400 seconds (Borris&Patnaik, 2014)        # longest exposure time over which an AEGL 1, 2 or 3 onset can be reached
    Brho = zeros(7,3)                                               # Define the Brho array as a 7x3 matrix of zeros                            # in ppm for each AEGL band at every time step ‘Atime’
    Balpha = zeros(7,3)                                             # Define the Balpha array as a 7x3 matrix of zeros                          # power-law exponents
    rhomax = zeros(1,3)                                             # Define the rhomax array as a 1x3 matrix of zeros                          # maximum concentration of H2S exposed by each agent
    rhomin = zeros(1,3)                                             # Define the rhomin array as a 1x3 matrix of zeros                          # minimum concentration of H2S exposed by each agent
    Btime = zeros(7, 3)                                             # Define the Btime array as a 7x3 matrix of zeros                           # represents an array, which is function of ‘taumin’ and ‘taumax’, that changes depending on alpha, which is a corresponsing array of power low exponents interpolating the ‘Brho’ table array

    #Initialize
    for k=1:3
        for b=2:6
            Brho[b, k] = Arho[b, k]
            Btime[b, k] = Atime[b]
        end
        Btime[1, k] = taumin;
         Btime[7, k] = taumax;
    end    
    
    #power low trial values
    for k=1:3
        for b=3:6
            if Brho[b-1, k]==Brho[b, k]
                Balpha[b, k] = 0.0;
            else
                Balpha[b, k] = log(Atime[b]/Atime[b-1])/log(Brho[b-1, k]/Brho[b, k]);
            end
        end

        Balpha[2, k] = Balpha[3, k]
        Balpha[1, k] = Balpha[2, k]
        Balpha[7, k] = Balpha[6, k]
        #Balpha[6, k] = Balpha[5, k]
    end
    #extrapolate for the edges
    for k=1:3
        if Balpha[3, k]==0
            rhomax[k] = Brho[2, k];
            Brho[1, k] = rhomax[k];
        else
            rhomax[k] = Brho[2, k]*(Btime[2, k]/taumin)^(1/Balpha[2, k]);
            Brho[1, k] = rhomax[k];
        end

        if Brho[5, k]==Brho[6, k]
            rhomin[k] = Brho[6, k];
            Brho[7, k] = rhomin[k];
        else
            rhomin[k] = Brho[6, k]*(Btime[6, k]/taumax)^(1/Balpha[6, k]); #I fixed Brho[6] to be Brho[6, k]
            Brho[7, k] = rhomin[k];
        end

    end
    #Correct to account for threshold values
    for k=1:3
        for b=2:7
            if Balpha[b, k] == 0
                Btime[b, k] = Btime[b-1, k];
            end
        end
    end

    for k=1:3
        for b=3:5
            if Balpha[b-1, k]==0 && Balpha[b, k]>0
                Balpha[b, k]=log(Btime[b, k]/Btime[b-1, k])/log(Brho[b-1, k]/Brho[b, k]);
            end
        end
    end

    return Balpha, Btime, Brho'
end

Balpha, Btime, Brho = setupToxic()
# Suffix for exports: align naming with DStarLite - Scenario 4.jl
run_export_tag = "newAEGL_0.5CM_uncappedTL"
run_timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")


function update_toxic_load(Ct, TLcurrent, dt)
   
    #a = alpha
    #Btime = Btime;
    #Brho = Brho;
    TL = TLcurrent
    TL_rate = 0.0

    for k = 1:3
        Cmin = Brho[k, 7]
        Cmax = Brho[k, 1]
        if Ct > Cmax
            TL_rate = 1 / Btime[1];
        elseif Ct < Cmin
            TL_rate = 0.0;
        else
            for i in 2:size(Btime, 1)
                if i <= size(Brho, 2)
                    if Brho[k, i-1] < Ct && Ct < Brho[k, i] #check 
                        TL_rate = (1/Btime[i, k])*((Ct/Brho[k, i])^(Balpha[i, k])) #Toxic Load rate in units of s^-1 
                    end
                end
            end
        end
        TL[k] = TL[k] .+ TL_rate * dt;
        
    end
        #TL[iAg, :] .= sum(TL .> 1) .+ TL[min(3, sum(TL .> 1) + 1)] .* (1 - (TL[3] > 1));
        #TL[iAg, :] .= person.toxicload;
    return TL
end



function static_preplot!(ax, abmplot)
    # 1) Ξεπακετάρουμε το Observable
    model = isa(abmplot, Observable)  ? abmplot[] :
            hasproperty(abmplot, :model) ? abmplot.model[] :
            abmplot

    # 2) Σχεδιάζουμε τα goals
    dests = model.goal
    xs_g = getindex.(dests, 1)
    ys_g = getindex.(dests, 2)
    scatter!(ax, xs_g, ys_g; color = (:red, 50), marker = '●')

    # 3) Σχεδιάζουμε για κάθε agent τη διαδρομή που έχει ήδη κάνει
    for agent in allagents(model)
        xs = agent.pathX
        ys = agent.pathY
        # π.χ. χρώμα ίδια με τον agent, πάχος γραμμής 2
        lines!(ax, xs, ys; linewidth = 2, color = personcolor(agent))
    end
end


function personcolor(person::AgentEscapes)  # Χρώμα του agent ανάλογα με το toxicload
    if person.toxicload >= 3.
        return :red
    elseif person.toxicload <= 1.
        return :green
    else
        return :orange
    end
end

@time begin   # Δημιουργία animation με trails & συλλογή CSV θέσης και toxicload
    const T = 1852
    # Spread CM transitions evenly over the first 8/10 of the simulation time (same as DStarLite).
    # Time counter is t = (frame - 1) * dt.
    const CM_ACTIVE_FRACTION = 0.8
    const NUM_MAPS = NUM_CMS
    const CM_SWITCH_TIMES = collect(range(
        0.0,
        stop = CM_ACTIVE_FRACTION * ((T - 1) * dt),
        length = NUM_MAPS,
    ))
# -- Στήσιμο Figure & Axis --
    fig = Figure(; size = (800,800))
    ax  = Makie.Axis(fig[1,1];
               title  = "Evacuation with Toxic Trail",
               aspect = DataAspect())

    heatmap!(ax, heightmap; colormap=:grays, alpha=0.3)
    
    # --- Concentration Map visualization (contour only, overlay on heightmap) ---
    cm_obs = Observable(penalty_map)
    # Makie errors if min(z)==max(z) (degenerate colorrange). Widen slightly when flat.
    cm_colorrange = @lift let z = $cm_obs
        m = minimum(z)
        M = maximum(z)
        if m == M
            δ = max(1.0, abs(m) * 1e-6)
            (Float32(m - δ), Float32(M + δ))
        else
            (Float32(m), Float32(M))
        end
    end
    cm_contour = contour!(
        ax, cm_obs;
        colormap = cgrad([:yellow, :orange, :red]),
        levels = 10,
        colorrange = cm_colorrange,
        linewidth = 1.5,
        alpha = 0.7,
    )
    
    goals = model.goal
    scatter!(ax,
        getindex.(goals,1),
        getindex.(goals,2);
        color  = (:red,50),
        marker = :circle,
    )

    # --- Overlay time counter + active goals (below clock, fixed for this run) ---
    frame_obs = Observable(0)

    counter_lbl = Label(
        fig,
        @lift("Time elapsed = $((($frame_obs-1)*dt)) s"),
        fontsize = 16,
        padding = (6, 10, 6, 10),
        halign = :left,
    )
    active_dests_text =
        "Active goals ($(length(dests))):\n" *
        join(["  ($(round(x; digits=1)), $(round(y; digits=1)))" for (x, y) in dests], "\n")
    goals_inset = GridLayout()
    goals_inset[1, 1] = counter_lbl
    goals_inset[2, 1] = Label(
        fig,
        active_dests_text;
        fontsize = 11,
        halign = :left,
        padding = (6, 2, 6, 10),
    )
    fig[1, 1, TopLeft()] = goals_inset

    # -- Observables για θέση & χρώμα --

    xs0 = Float64[]  # Initialize empty arrays for positions
    ys0 = Float64[]

    for a in allagents(model)
        push!(xs0, a.pos[1])
        push!(ys0, a.pos[2])
    end

    colors0 = Symbol[]
    for a in allagents(model)
        push!(colors0, personcolor(a))
    end

    posobs = Observable(Point2f.(xs0, ys0))
    colobs = Observable(colors0)

    lines_plots = [
        lines!(
            ax,
            [a.pos[1]], [a.pos[2]];
            color     = personcolor(a),
            linewidth = 2,
        )
        for a in allagents(model)
    ]
    agent_scat = scatter!(
        ax, posobs;
        color       = colobs,
        markersize  = 10,
    )

    agent_id_labels = [string(a.id) for a in allagents(model)]
    text!(
        ax, posobs;
        text       = agent_id_labels,
        fontsize   = 7,
        align      = (:left, :center),
        offset     = (5, 0),
        color      = :black,
        strokewidth = 0.75,
        strokecolor = :white,
    )

    # -- Προετοιμασία DataFrame για θέση & toxicload ανά βήμα --
    df = DataFrame(
        step       = Int[],
        agent_id   = Int[],
        x          = Float64[],
        y          = Float64[],
        toxicload  = Float64[]
    )


video_file = "SCENARIOS/SCENARIO 4/Simulation Results/AStar_SCENARIO_4_$(n_agents)_$(seed)_$(cost_metric_str)_$(heuristic_code)_$(run_export_tag)_$(run_timestamp).mp4"
csv_file   = "SCENARIOS/SCENARIO 4/Simulation Results/AStar_SCENARIO_4_$(n_agents)_$(seed)_$(cost_metric_str)_$(heuristic_code)_$(run_export_tag)_$(run_timestamp).csv"

# keep a variable for the currently active map index
current_map_idx = 1

path_planning_init_time = Ref(0.0)
path_planning_replan_time = Ref(0.0)
simulation_time = Ref(0.0)

# Initial pathfinding: same placement as DStarLite (inside @time block, before record)
path_planning_init_time[] += @elapsed begin
    for a in allagents(model)
        plan_best_route!(a, model.goal, model.pathfinderPM)
    end
end

record(fig, video_file, 1:T; framerate=30) do frame
    # update frame counter observable
    frame_obs[] = frame

    # 1) step the model first (same order as DStarLite - Scenario 4.jl)
    simulation_time[] += @elapsed step!(model, 1)

    # 2) concentration map index for this frame
    global current_map_idx
    t_sim = (frame - 1) * dt
    new_idx = searchsortedlast(CM_SWITCH_TIMES, t_sim)
    if new_idx != current_map_idx
        current_map_idx = new_idx

        path_planning_replan_time[] += @elapsed begin
            penalty_map .= cm_list[current_map_idx]

            NPM = heightmap .+ penalty_map
            NPM_int = round.(Int, NPM)

            global cost_metric_obj
            cost_metric_obj = AbsolutePenaltyMap(NPM_int, MaxDistance{2}())

            pm = Pathfinding.penaltymap(model.pathfinderPM)
            pm .= NPM_int

            for a in allagents(model)
                plan_best_route!(a, model.goal, model.pathfinderPM)
            end
        end

        println("Frame $frame: swapped to concentration map $current_map_idx and replanned paths.")
    end

    # Update concentration map visualization
    cm_obs[] = penalty_map

    # 3) update trails/visuals
    for (i, a) in enumerate(allagents(model))
        lines_plots[i][1][] = Point2f.(a.pathX, a.pathY)
    end

    posobs[] = [Point2f(a.pos[1], a.pos[2]) for a in allagents(model)]
    colobs[] = [personcolor(a) for a in allagents(model)]

    # 4) collect data
    for a in allagents(model)
        push!(df, (
            frame,
            a.id,
            a.pos[1],
            a.pos[2],
            a.toxicload
        ))
    end

    # 5) record runs for T frames (fixed horizon)
    end

    println("Το animation σώθηκε ως $video_file")
    CSV.write(csv_file, df)
    println("Τα δεδομένα θέσης & toxicload αποθηκεύτηκαν ως $csv_file")

    total_path_planning = path_planning_init_time[] + path_planning_replan_time[]
    println("TIMING BREAKDOWN:")
    println("  Pathfinding time (total): ", round(total_path_planning; digits=6), " s")
    println("    - Initial pathfinding: ", round(path_planning_init_time[]; digits=6), " s")
    println("    - Replanning during CM changes: ", round(path_planning_replan_time[]; digits=6), " s")
    println("  Simulation time (step! only): ", round(simulation_time[]; digits=6), " s")
end


begin

    dt = @isdefined(dt) ? dt : 1.0
    seed_str = @isdefined(seed) ? string(seed) : nothing
    folder = joinpath("SCENARIOS","SCENARIO 4", "Simulation Results")

    # Φόρτωση CSV με step, agent_id, toxicload
    csv_file = seed_str === nothing ? nothing : joinpath(folder, "AStar_SCENARIO_4_$(n_agents)_$(seed)_$(cost_metric_str)_$(heuristic_code)_$(run_export_tag)_$(run_timestamp).csv")
    if csv_file === nothing || !isfile(csv_file)
        # αν δεν δοθεί seed, πάρε το πιο πρόσφατο AStar_SCENARIO_4_*.csv
        pattern = Regex("^AStar_SCENARIO_4_.*_$(cost_metric_str)_$(heuristic_code)_$(run_export_tag)_.*\\.csv\$")
        csvs = filter(f -> occursin(pattern, f), readdir(folder))
        @assert !isempty(csvs) "Δεν βρέθηκαν αρχεία *AStar_SCENARIO_4_$(seed)*.csv στο $(folder)."
        stats = stat.(joinpath.(Ref(folder), csvs))
        latest_idx = argmax(getfield.(stats, :mtime))
        csv_file = joinpath(folder, csvs[latest_idx])
        seed_str = "latest"
    end

    df = CSV.read(csv_file, DataFrame)

# --- Σταθερά bins ---
    # Εσωτερικά bins αυστηρά στο (0,3) με βήμα 0.5 -> centers: 0.25, 0.75, ..., 2.75
    edges_inner   = 0:0.5:3.0
    centers_inner = (edges_inner[1:end-1] .+ edges_inner[2:end]) ./ 2
    bin_w_inner   = 0.45

    # Άκρες (0 και 3) σαν ξεχωριστές μπάρες με μικρό offset για να μην επικαλύπτονται
    edge_offset = 0.30
    x0_pos      = 0.0 - edge_offset   # π.χ. -0.30
    x3_pos      = 3.0 + edge_offset   # π.χ.  3.30
    bin_w_edge  = 0.30

    # Counts για ΑΝΟΙΚΤΟ (0,3) χωρίς StatsBase
    function bin_counts_open(values::AbstractVector{<:Real}, edges::AbstractVector{<:Real})
        counts = zeros(Int, length(edges) - 1)
        @inbounds for v in values
            if v > edges[1] && v < edges[end]           # αυστηρά (0,3)
                idx = searchsortedlast(edges, v)
                idx = clamp(idx, 1, length(counts))
                counts[idx] += 1
            end
        end
        return counts
    end

    # Formatter για ticks (χωρίς ->)
    function mytickfmt(x::Real)
        string(round(x; digits=1))
    end
    function mytickfmt(xs::AbstractVector{<:Real})
        string.(round.(xs; digits=1))
    end

    # --- Παράμετροι χρόνου / y-scale ---
    T_avail = maximum(df.step)
    T_play  = min(T, T_avail)               # παίξε μέχρι 400 ή όσο υπάρχει
    n_agents_guess = maximum(combine(groupby(df, :step), nrow).nrow)

    # --- Figure / Axis ---
    fig = Figure(; size = (1000, 600))
    ax  = CairoMakie.Axis(fig[1,1];
        title  = "Toxic Load (X) vs Agents Count (Y) — bins 0 & 3 separate, inner step 0.5",
        xlabel = "Toxic load",
        ylabel = "Agents (count)",
        yticks = 0:10:n_agents_guess          # δείκτες Υ ανά 10
    )
    xlims!(ax, -0.5, 3.5)                      # λίγο περιθώριο για τις άκρες
    ylims!(ax, 0, n_agents_guess)              # προσαρμόσ’ το αν θες

    xt = collect(0.0:0.5:3.0)
    ax.xticks = xt
    ax.xtickformat = mytickfmt
    ax.xticklabelrotation = pi/4               # προαιρετικό

    # Overlay χρόνου
    frame_obs = Observable(0)
    lbl = Label(fig, @lift("Time elapsed = $((($frame_obs-1)*dt)) s"),
                fontsize = 16, padding = (6,10,6,10), halign = :left)
    fig[1,1, TopLeft()] = lbl

    # --- Τρία layers μπαρών: εσωτερικά + 0 + 3 ---
    y_inner = Observable(zeros(Float64, length(centers_inner)))  # (0,3) ανά 0.5
    y0      = Observable([0.0])                                  # bin για 0
    y3      = Observable([0.0])                                  # bin για 3

    # Χρώματα για τα εσωτερικά bins ανά TL ζώνη (ίδια λογική με DStarLite - Scenario 4.jl)
    inner_cols = [ c for x in centers_inner for c in
        (x < 1.0  ? (:dodgerblue,) :          # (0,1)
        x < 2.0  ? (:gold,)       :          # [1,2)
                 (:orange,)) ]            # [2,3)

    # εσωτερικές μπάρες (vector χρωμάτων)
    barplot!(ax, centers_inner, y_inner; width = bin_w_inner, color = inner_cols, strokewidth = 0)

    # άκρες: 0 = γκρι, 3 = κόκκινο
    barplot!(ax, [x0_pos], y0; width = bin_w_edge, color = :gray35,  strokewidth = 0)
    barplot!(ax, [x3_pos], y3; width = bin_w_edge, color = :crimson, strokewidth = 0)

    # --- Εγγραφή βίντεο (αλλάζουν μόνο οι Υ-τιμές) ---
    out_file = joinpath(folder, "AStar_SCENARIO_4_hist__$(n_agents)_$(seed_str)_$(cost_metric_str)_$(heuristic_code)_$(run_export_tag)_$(run_timestamp).mp4")
    record(fig, out_file, 1:T_play; framerate = 30) do frame
        frame_obs[] = frame
        tl = Vector(df[df.step .== frame, :toxicload])

        count0       = count(==(0.0), tl)                 # ακριβώς 0 (όπως DStarLite)
        count3       = count(==(3.0), tl)                 # ακριβώς 3
        counts_inner = bin_counts_open(tl, edges_inner)   # μόνο (0,3) ανά 0.5

        y_inner[] = Float64.(counts_inner)
        y0[]      = [Float64(count0)]
        y3[]      = [Float64(count3)]

        # Προαιρετικό auto-scale μόνο στον Υ:
        # ylims!(ax, 0, max(1, maximum(vcat(y0[][1], y_inner[], y3[][1]))))
    end

    println("Βίντεο με ξεχωριστά bins για 0 & 3 και εσωτερικά ανά 0.5 σώθηκε ως: $out_file")

    append_run_history(
        scenario = "SCENARIO 4",
        runner_script = "SCENARIOS/SCENARIO 4/Scenario 4.jl",
        csv_file = csv_file,
        mp4_file = video_file,
        hist_mp4_file = out_file,
        pathfinding_time = total_path_planning,
        run_timestamp = run_timestamp,
    )
end
