begin   # Φόρτωση των απαραίτητων βιβλιοθηκών
    using Agents
    using Agents.Pathfinding
    using CSV
    using CairoMakie
    using DataFrames
    using FileIO: load
    using ImageMagick
    using Images
    using InteractiveDynamics
    using Makie
    using Observables
    using Random
    using StaticArrays

    # FlowField pathfinding module (kept as local include)
    include("../../Agents/src/submodules/pathfinding/flow_field.jl")
    using .FlowField
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
    dt = 1.   ## discrete timestep each iteration of the model          # Define the dt variable as 1
    seed = 123  ## seed for random number generator                     # Define the seed variable as 123
    n_agents = 5                                                       # Define the n_agents variable as 100
    toxicity_rate = 0.07                                               # Define the toxicity_rate variable as 0.07
    age_range = (22,60)                                                 # Define the age_range variable as a tuple of 22 and 60
    speed_range = (4.0,7.0)                                            # Define the speed_range variable as a tuple of 4.0 and 7.0
    speed = 5.                                                         # Define the speed variable as 5 
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
    # Flow Field parameters
    # alpha, beta, gamma: cost parameters
    # lambda: density weighting
    # threshold: unused for now
    # Order: alpha, beta, gamma, λ, threshold
    ff_params = FlowFieldParams(
        1.0,    # alpha
        0.1,    # beta
        0.01,   # gamma (penalty map weight)
        2.0,    # λ (density weighting exponent)
        0.0     # threshold
    )
    
    # Initialize flow field
    # Flow field grid uses cellsize = 1.0 (same as space spacing)
    # Convert world dimensions to match the NPM size
    # NPM is (height, width) in matrix notation
    # Note: init_flowfield adds +1 to dimensions, so we subtract 1 to match map size
    map_width = size(NPM, 2)   # width (x dimension) in grid cells
    map_height = size(NPM, 1)  # height (y dimension) in grid cells
    world_width = Float64(map_width - 1)   # subtract 1 because init_flowfield adds +1
    world_height = Float64(map_height - 1) # subtract 1 because init_flowfield adds +1
    cellsize = 1.0
    
    # Flow field expects penalty_map and walkmap as (ny, nx) = (height, width) in matrix notation
    # Our maps are already (height, width) = (ny, nx), so no transpose needed!
    # Flow field uses [iy, ix] indexing where iy is row (y) and ix is column (x)
    
    # Convert goals to GridGoal format
    # Flow field uses 1-based grid coordinates where:
    # - ix (x) ranges from 1 to nx (width)
    # - iy (y) ranges from 1 to ny (height)
    # Goals are in world coordinates (x, y), need to convert to grid (ix, iy)
    goal_radius = 5  # radius around each goal point in grid cells
    goal_cells = Tuple{Int,Int}[]
    for dest in dests
        gx = Int(floor(dest[1])) + 1  # x -> ix (column, width dimension)
        gy = Int(floor(dest[2])) + 1  # y -> iy (row, height dimension)
        # Add cells in a small region around the goal
        for dx in -goal_radius:goal_radius
            for dy in -goal_radius:goal_radius
                if dx*dx + dy*dy <= goal_radius*goal_radius
                    gx_cell = clamp(gx + dx, 1, map_width)
                    gy_cell = clamp(gy + dy, 1, map_height)
                    push!(goal_cells, (gx_cell, gy_cell))
                end
            end
        end
    end
    
    # Create combined GridGoal from all goal cells
    if !isempty(goal_cells)
        minx = minimum(c -> c[1], goal_cells)
        maxx = maximum(c -> c[1], goal_cells)
        miny = minimum(c -> c[2], goal_cells)
        maxy = maximum(c -> c[2], goal_cells)
        global flow_goal = GridGoal(minx, maxx, miny, maxy)
    else
        # Fallback: use first goal
        gx = Int(floor(dests[1][1])) + 1
        gy = Int(floor(dests[1][2])) + 1
        global flow_goal = GridGoal(gx, gx, gy, gy)
    end
    
    # Initialize flow field with penalty map and walkmap (already in correct format)
    global flowfield = init_flowfield(
        world_width, world_height, cellsize, ff_params;
        penalty_map = penalty_map,  # (ny, nx) = (height, width)
        walkmap = walkmap            # (ny, nx) = (height, width)
    )
    
    cost_metric_str = "FlowField_AbsolutePenaltyMap"
    
    properties = (
        flowfield = flowfield,
        heightmap = heightmap,
        dt = dt,
        speed_range = speed_range,
        goal = dests,
    )
end


function agent_step!(person, model)
    position = floor.(Int, person.pos)
   # Ct παίρνεται τώρα από το global_penalty_map (hand-drawn maps)
    Ct = penalty_map[position[1], position[2]]
    TLcurrent = [person.TL1[end], person.TL2[end], person.TL3[end]]
    TL = update_toxic_load(Ct, TLcurrent, dt)

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

    # Get velocity from flow field
    pos_svec = SVector{2,Float64}(person.pos[1], person.pos[2])
    flow_vel = lookup_velocity(model.flowfield, pos_svec)
    
    # Move agent along flow field direction
    vel_norm = norm(flow_vel)
    if vel_norm > 0.0
        # Scale velocity by agent's speed
        direction = flow_vel / vel_norm
        person.pos += direction * speed * dt
    end
    
    push!(person.pathX, person.pos[1])
    push!(person.pathY, person.pos[2])
end



function model_step!(model)
    # Update flow field before agent steps
    # This recomputes the flow field based on current agent positions
    update_flowfield!(model.flowfield, model, flow_goal;
                     get_pos = a -> SVector{2,Float64}(a.pos[1], a.pos[2]),
                     get_vel = a -> SVector{2,Float64}(0.0, 0.0))
    
    # Handle collisions
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

@time begin
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
        println("Warning: Could not find valid walkable spawn position for agent in Scenario 3 after $max_attempts attempts")
        continue
    end
    
    person = add_agent!(pos, AgentEscapes, model, vel, age, mass, 1., [pos[1]], [pos[2]], [0.0], [0.0], [0.0])
    # No need to plan route - flow field handles navigation
end
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
    const T = 600
frames_per_map = 60
const NUM_MAPS = 10
# -- Στήσιμο Figure & Axis --
    fig = Figure(; size = (800,800))
    ax  = Makie.Axis(fig[1,1];
               title  = "Evacuation with Toxic Trail (Flow Field)",
               aspect = DataAspect())

    heatmap!(ax, heightmap; colormap=:grays, alpha=0.3)
    
    # --- Concentration Map visualization (contour only, overlay on heightmap) ---
    cm_obs = Observable(penalty_map)
    cm_contour = contour!(ax, cm_obs; colormap=:hot, levels=10, linewidth=1.5, alpha=0.7)
    
    goals = model.goal
    scatter!(ax,
        getindex.(goals,1),
        getindex.(goals,2);
        color  = (:red,50),
        marker = :circle,
    )

    # --- Overlay time counter ---
    frame_obs = Observable(0)
    
    counter_lbl = Label(
    fig,
    @lift("Time elapsed = $((($frame_obs-1)*dt)) s"),
    fontsize = 16,
    padding = (6, 10, 6, 10),
    halign = :left
)
    fig[1,1, TopLeft()] = counter_lbl  # αγκίστρωση πάνω-αριστερά στο ίδιο κελί με τον άξονα

    # -- Observables για θέση & χρώμα --

    xs0 = Float64[]  # Initialize empty arrays for positions
    ys0 = Float64[]
    colors0 = Symbol[]  # Initialize empty array for colors

    for a in allagents(model)
        push!(xs0, a.pos[1])
        push!(ys0, a.pos[2])
        push!(colors0, personcolor(a))
    end

    posobs = Observable(Point2f.(xs0, ys0))
    colobs = Observable(colors0)

    lines_plots = [
        lines!(ax,
               [a.pos[1]], [a.pos[2]];
               color     = personcolor(a),
               linewidth = 2)
        for a in allagents(model)
    ]
    agent_scat = scatter!(ax, posobs;
                          color      = colobs,
                          markersize = 10)

    # -- Προετοιμασία DataFrame για θέση & toxicload ανά βήμα --
    df = DataFrame(
        step       = Int[],
        agent_id   = Int[],
        x          = Float64[],
        y          = Float64[],
        toxicload  = Float64[]
    )


video_file = "SCENARIOS/SCENARIO 3/Simulation Results/FlowField_SCENARIO_3_$(n_agents)_$(seed)_$(cost_metric_str).mp4"
csv_file   = "SCENARIOS/SCENARIO 3/Simulation Results/FlowField_SCENARIO_3_$(n_agents)_$(seed)_$(cost_metric_str).csv"  # added

# keep a variable for the currently active map index
current_map_idx = 1

record(fig, video_file, 1:T; framerate=30) do frame
    # update frame counter observable
    frame_obs[] = frame

    # 1) step the model normally (this will update flow field in model_step!)
    step!(model, 1)

    # 2) determine which CM index should be active on this frame
    global current_map_idx  # Declare `current_map_idx` as global
    new_idx = min(NUM_MAPS, Int(ceil(frame / frames_per_map)))
    if new_idx != current_map_idx
        # instant swap of penalty_map
        current_map_idx = new_idx
        penalty_map .= cm_list[current_map_idx]   # in-place replace values

        # Update flow field's penalty map (already in correct format)
        update_penalty_map!(model.flowfield, penalty_map)

        # Flow field will be updated in the next model_step! call
        println("Frame $frame: swapped to concentration map $current_map_idx and updated flow field penalty map.")
    end

    # Update concentration map visualization
    cm_obs[] = penalty_map

    # 3) update trails/visuals as before
    for (i, a) in enumerate(allagents(model))
        lines_plots[i][1][] = Point2f.(a.pathX, a.pathY)
    end

    xs = [a.pos[1] for a in allagents(model)]
    ys = [a.pos[2] for a in allagents(model)]
    posobs[] = Point2f.(xs, ys)
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

    # 5) stop condition: after map 10 completes (frame == NUM_MAPS*frames_per_map) the record ends automatically
    end

    println("Το animation σώθηκε ως $video_file")
    CSV.write(csv_file, df)
    println("Τα δεδομένα θέσης & toxicload αποθηκεύτηκαν ως $csv_file")
end
