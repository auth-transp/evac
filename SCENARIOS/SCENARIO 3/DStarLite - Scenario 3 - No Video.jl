begin   # Φόρτωση των απαραίτητων βιβλιοθηκών
    using Agents
    using Agents.Pathfinding
    using CSV
    using DataFrames
    using FileIO: load
    using ImageMagick
    using Images
    using Random
    using StaticArrays
end       

Agents.@agent struct AgentEscapes(ContinuousAgent{2, Float64})
    age::Float64
    mass::Float64
    toxicload::Float64
    path::Vector{Tuple{Int,Int}}    # ΜΟΝΟ grid path
    pathX::Vector{Float64}          # μόνο για visualization
    pathY::Vector{Float64}          # μόνο για visualization
    TL1::Vector{Float64}
    TL2::Vector{Float64}
    TL3::Vector{Float64}
end


begin   # Φόρτωση του heightmap και των hand-drawn penalty maps (και όλων των CM1..CM10)
    # heightmap (unchanged)
    heightmap_data = load("NADEEN/Maps/Qatargas Map.jpg")
    heightmap_data = permutedims(channelview(heightmap_data), [2,3,1])[:,:,1]
    global heightmap = floor.(Int, convert.(Float64, heightmap_data) * 255)
    heightmap = 255 .- heightmap   # αυτό κάνει την αντιστροφή

    # --- Load all concentration maps 1..10 as Float64 arrays ---
    const NUM_CMS = 10
    cm_list = Vector{Array{Float64,2}}(undef, NUM_CMS)
    for k in 1:NUM_CMS
        fn = joinpath("Concentration Maps", string(k) * ".bmp")
        img = load(fn)
        img = permutedims(channelview(img), [2,3,1])[:,:,1]
        cm_list[k] = convert.(Float64, img) .* 500.0   # keep same scaling as before
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
    n_agents = 5                                                        # Define the n_agents variable as 3
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


    #goals
    dests = [(600., 980.), (100., 200.)]

    #Generate the RNG for the model
    rng = MersenneTwister(seed)

    ## Note that the dimensions of the space do not have to correspond to the dimensions
    ## of the pathfinder. Discretisation is handled by the pathfinding methods
    space = ContinuousSpace(size(NPM); periodic = false, spacing = 1)


begin
    cost_metric_obj = PenaltyMap(NPM_int, MaxDistance{2}())
    cost_metric_str = "PM"
    heuristic_code  = "DF"   # DF: default Agents.jl heuristic, EY: Euclidean, MN: Manhattan
    pathfinderPM = DStarLite(space; walkmap = walkmap, cost_metric = cost_metric_obj)
    properties = (
        pathfinderPM = pathfinderPM,
        heightmap = heightmap,
        dt = dt,
        speed_range = speed_range,
        goal = dests,
    )
end


function move_along_precomputed_path!(agent::AgentEscapes, speed, dt)
    isempty(agent.path) && return

    # επόμενος grid στόχος
    cell = agent.path[1]
    target = SVector{2,Float64}(cell[1], cell[2])

    dir = target .- agent.pos
    dist = norm(dir)

    if dist < speed * dt
        agent.pos = target
        popfirst!(agent.path)   # πήγες στο cell
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

    # --- Speed update based on toxicload ---
    speed = 1.35
    if 0 < person.toxicload <= 1
        speed = 1.35 * exp(0.393 * person.toxicload)
    elseif 1 < person.toxicload < 3
        speed = -1.78 * log(person.toxicload) + 2.063
    elseif person.toxicload >= 3
        speed = 0.0
    end

    # use the global `pathfinderPM` (do not mutate model properties)
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
  rng          = rng,
  properties   = properties,
  agent_step!  = agent_step!,
  model_step!  = model_step!
)

@time for _ in 1:n_agents
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
        println("Warning: Could not find valid walkable spawn position for agent in Scenario 3 after $max_attempts attempts")
        continue
    end
        
    person = add_agent!(
        pos, AgentEscapes, model,
        vel, age, mass, 1.,
        Tuple{Int,Int}[],     # path
        [pos[1]],             # pathX
        [pos[2]],             # pathY
        [0.0], [0.0], [0.0]
    )        # plan using the global `pathfinderPM`
end



function setupToxic()                                               # Define the setupToxic function
    Atime = [0.0, 0.17, 0.83, 1.67, 4.17, 8.33] #min                # Define the Atime array as a 1x6 matrix                                    # Initialization of the 5 standard AEGL exposure times
    Arho = zeros(3, 6)                                              # Define the Arho array as a 3x6 matrix                                     # the concentration of the three symptoms compared to the AEGL concentrations
    Arho[1, 2:6] = [4.85, 4.23, 4.17, 4.06, 3.82]                   # Define the Arho array for the first row and columns 2 to 6                # odor
    Arho[2, 2:6] = [180.79, 157.56, 155.43, 151.37, 142.48]         # Define the Arho array for the second row and columns 2 to 6               # irritation
    Arho[3, 2:6] = [485.62, 423.22, 417.49, 406.59, 382.71] #ppm    # Define the Arho array for the third row and columns 2 to 6                # edema
    MW = 34 #Molecular weight of H2S in g/mol
    Arho *= MW/24.04 #mg/m^3                                        # Multiply the Arho array by the molecular weight of H2S divided by 24.04
    Arho = Arho'                                                    # Transpose the Arho array 
    Atime = Atime*60 #seconds                                       # Multiply the Atime array by 60 seconds to convert to seconds              
    taumin = 200.                                                   # Define the taumin variable as 200 seconds (Borris&Patnaik, 2014)          # shortest exposure time over which an AEGL 1, 2 or 3 onset can be reached
    taumax = 86400.                                                 # Define the taumax variable as 86400 seconds (Borris&Patnaik, 2014)        # longest exposure time over which an AEGL 1, 2 or 3 onset can be reached
    Brho = zeros(7,3)                                               # Define the Brho array as a 7x3 matrix of zeros                            # in ppm for each AEGL band at every time step 'Atime'
    Balpha = zeros(7,3)                                             # Define the Balpha array as a 7x3 matrix of zeros                          # power-law exponents
    rhomax = zeros(1,3)                                             # Define the rhomax array as a 1x3 matrix of zeros                          # maximum concentration of H2S exposed by each agent
    rhomin = zeros(1,3)                                             # Define the rhomin array as a 1x3 matrix of zeros                          # minimum concentration of H2S exposed by each agent
    Btime = zeros(7, 3)                                             # Define the Btime array as a 7x3 matrix of zeros                           # represents an array, which is function of 'taumin' and 'taumax', that changes depending on alpha, which is a corresponsing array of power low exponents interpolating the 'Brho' table array

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
    TL[1] = TL[1] > 1.0 ? 1.0 : TL[1]
    TL[2] = TL[2] > 1. ? 1.0 : TL[2]
    TL[3] = TL[3] > 1. ? 1.0 : TL[3]

    return TL
end


# Convert continuous world position (Float64, Float64) to grid cell (Int, Int)
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
    return argmin(p -> length(p), paths)
end


# Simulation without video creation - for timing measurements
println("Starting simulation (no video)...")
const T = 1852
frames_per_map = 60
const NUM_MAPS = 10

# -- Προετοιμασία DataFrame για θέση & toxicload ανά βήμα --
df = DataFrame(
    step       = Int[],
    agent_id   = Int[],
    x          = Float64[],
    y          = Float64[],
    toxicload  = Float64[]
)

csv_file = "SCENARIOS/SCENARIO 3/Simulation Results/DStarLite_SCENARIO_3_$(n_agents)_$(seed)_$(cost_metric_str)_$(heuristic_code)_NOVIDEO.csv"

# keep a variable for the currently active map index
current_map_idx = 1

# --- INIT D* Lite planners (one per exit) ---
grid_dims = size(NPM_int)
exit_cells = [world_to_cell(g, grid_dims) for g in model.goal]

# Timing variables (aligned with DStarLite - Scenario 3 - Benchmarking.jl)
planner_init_time = 0.0   # init_planner per exit only
path_extract_time = 0.0   # initial extract_path for all agents only
replan_time = 0.0         # all D* Lite work when CM changes (map update + update_after_cm_change! + extract_path)
sim_step_time = 0.0       # pure step!(model, 1) per frame

# --- Initial D* Lite planning (same split as Benchmarking) ---
planner_init_time = @elapsed begin
    global planners
    planners = [
        init_planner(pathfinderPM, goal_cell)
        for goal_cell in exit_cells
    ]
end

path_extract_time = @elapsed begin
    for a in allagents(model)
        start = world_to_cell(a.pos, grid_dims)
        paths = [extract_path(p, start) for p in planners]
        a.path = choose_best_path(paths)
    end
end

prev_NPM_int = copy(NPM_int)

# Run simulation loop (without video recording)
for frame in 1:T
    # 1) step the model normally (SIMULATION) - same timing as Benchmarking
    sim_step_time += @elapsed step!(model, 1)

    # 2) determine which CM index should be active on this frame
    global current_map_idx
    new_idx = min(NUM_MAPS, Int(ceil(frame / frames_per_map)))

    if new_idx != current_map_idx
        current_map_idx = new_idx

        # --- Full replan block timed as one (same as Benchmarking) ---
        replan_time += @elapsed begin
            penalty_map .= cm_list[current_map_idx]

            NPM = heightmap .+ penalty_map
            NPM_int = round.(Int, NPM)

            global cost_metric_obj
            cost_metric_obj = PenaltyMap(NPM_int, MaxDistance{2}())
            pathfinderPM.cost_metric.pmap .= NPM_int

            changed_cells = find_changed_cells(prev_NPM_int, NPM_int)
            prev_NPM_int .= NPM_int

            for planner in planners
                update_after_cm_change!(planner, changed_cells)
            end

            for a in allagents(model)
                start = world_to_cell(a.pos, grid_dims)
                paths = [extract_path(p, start) for p in planners]
                a.path = choose_best_path(paths)
            end
        end

        println("Frame $frame: swapped to concentration map $current_map_idx and replanned paths.")
    end

    # 3) collect data
    for a in allagents(model)
        push!(df, (
            frame,
            a.id,
            a.pos[1],
            a.pos[2],
            a.toxicload
        ))
    end
end

pathfinding_time = planner_init_time + path_extract_time + replan_time
simulation_time = sim_step_time

CSV.write(csv_file, df)
println("Simulation completed!")
println("=" ^ 60)
println("TIMING BREAKDOWN (aligned with DStarLite - Scenario 3 - Benchmarking.jl):")
println("  Initial planner setup:    $(round(planner_init_time, digits=6)) s")
println("  Initial path extraction: $(round(path_extract_time, digits=6)) s")
println("  Replanning (CM changes): $(round(replan_time, digits=6)) s")
println("  Total pathfinding time:  $(round(pathfinding_time, digits=6)) s ($(round(pathfinding_time/60, digits=2)) min)")
println("  Simulation time (steps): $(round(simulation_time, digits=6)) s ($(round(simulation_time/60, digits=2)) min)")
println("=" ^ 60)
println("Τα δεδομένα θέσης & toxicload αποθηκεύτηκαν ως $csv_file")