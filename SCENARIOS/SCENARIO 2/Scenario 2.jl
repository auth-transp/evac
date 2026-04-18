begin   # Φόρτωση των απαραίτητων βιβλιοθηκών
    using Agents
    using Agents.Pathfinding
    using LinearAlgebra
    using Random                        
    using ColorTypes                      
    using ImageMagick                 
    using FileIO: load                     
    using InteractiveDynamics             
    using Images                    
    using DataFrames           
    using Dates
    using Statistics
    using CairoMakie
    using DelimitedFiles
    using Observables
    using Makie
    using CSV
end                          



@agent struct AgentEscapes(ContinuousAgent{2, Float64}) # Αρχικοποίηση των Agents
    age::Float64
    mass::Float64
    toxicload::Float64
    pathX::Vector{Float64}
    pathY::Vector{Float64}
    TL1::Vector{Float64}
    TL2::Vector{Float64}
    TL3::Vector{Float64}
end

    
begin   # Heightmap για χώρο / A* · CM1..CM7 μόνο για TL και οπτικό overlay (και NPM στο pathfinding)
    heightmap_data = load("NADEEN/Maps/Qatargas Map.jpg")
    heightmap_data = permutedims(channelview(heightmap_data), [2,3,1])[:,:,1]
    global heightmap = floor.(Int, convert.(Float64, heightmap_data) * 255)
    heightmap = 255 .- heightmap

    const NUM_CMS = 7
    global cm_list = Vector{Array{Float64,2}}(undef, NUM_CMS)
    for k in 1:NUM_CMS
        fn = joinpath("Concentration Maps", string(k) * ".bmp")
        img = load(fn)
        img = permutedims(channelview(img), [2, 3, 1])[:, :, 1]
        cm_list[k] = convert.(Float64, img) .* 255.0
        @assert size(cm_list[k]) == size(heightmap) "Concentration map $k size mismatch with heightmap"
    end
    global tl_penalty_map = copy(cm_list[1])
    global NPM = heightmap .+ tl_penalty_map
end

begin   # Αρχικοποίηση των παραμέτρων του μοντέλου
    # Time–speed correlation: distance per step = speed × dt (in space units).
    # Treat dt as "time per step" (e.g. 1 = 1 second). Map scale: 1250 m ≈ 336.5 px.
    const METERS_TO_PIXELS = 723.37 / 2500.0
    dt = 1.   ## discrete timestep each iteration of the model          # Define the dt variable as 1
    seed = 123  ## seed for random number generator                     # Define the seed variable as 123
    run_timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
    n_agents = 50                                                        # Define the n_agents variable as 3
    toxicity_rate = 0.07                                               # Define the toxicity_rate variable as 0.07
    age_range = (22,60)                                                 # Define the age_range variable as a tuple of 22 and 60
    speed_range = (4.0,7.0)                                            # Define the speed_range variable as a tuple of 4.0 and 7.0
    mass_range = (50,80)                                                # Define the mass_range variable as a tuple of 50 and 80
    ag_range_y = (size(heightmap)[1]/4):(3*size(heightmap)[1]/4)    # Define the ag_range_y variable as a larger range of values from the heightmap array # [1] stands for the 1st row
    ag_range_x = (size(heightmap)[2]/4):(3*size(heightmap)[2]/4)    # Define the ag_range_x variable as a range of values from the heightmap array # [2] stands for the 2nd row
    dims = size(heightmap)                                            # Define the dims variable as the dimensions of the heightmap array (2xn matrix)
    walkmap = BitArray(trues(dims...))                                 # Define the walkmap variable as a BitArray of true values with the dimensions of the heightmap array
    
    # Set walkmap to false in white areas of the heightmap (after reversal, white = high values)
    # White areas are where heightmap value is above threshold (e.g., > 245)
    # Black and grey areas (low to medium values) remain walkable
    white_threshold = 245
    walkmap[heightmap .> white_threshold] .= false
end    


    #goals
    dests = [(500., 854.), (120., 248.)]

    #Generate the RNG for the model
    rng = MersenneTwister(seed)

    ## Note that the dimensions of the space do not have to correspond to the dimensions
    ## of the pathfinder. Discretisation is handled by the pathfinding methods
    # ContinuousSpace requires extent to be exactly divisible by spacing in each dimension.
    # Use a nearby compatible spacing (0.25 px) instead of METERS_TO_PIXELS (~0.2893 px).
    space = ContinuousSpace(size(heightmap); periodic = false, spacing = 1)


begin
    pathfinderPM = AStar(space; walkmap = walkmap, cost_metric = PenaltyMap(NPM, MaxDistance{2}()))
    properties = (
        pathfinderPM = pathfinderPM,
        heightmap = heightmap,
        dt = dt,
        speed_range = speed_range,
        goal = dests
    )
end


# Helper: true if position is within radius of any goal (TL and movement are skipped when true)
const goal_radius = 10.0
function at_goal(pos, dests, radius = goal_radius)
    return minimum(norm(pos .- d) for d in dests) ≤ radius
end


function agent_step!(person, model)
    if at_goal(person.pos, model.goal)
        # At goal: TL and movement stop; keep path in sync for visualization
        push!(person.pathX, person.pos[1])
        push!(person.pathY, person.pos[2])
        return
    end

    grid_dims = size(tl_penalty_map)
    i = clamp(Int(floor(person.pos[1])), 1, grid_dims[1])
    j = clamp(Int(floor(person.pos[2])), 1, grid_dims[2])
    Ct = tl_penalty_map[i, j]
    TLcurrent = [person.TL1[end], person.TL2[end], person.TL3[end]]
    TL = update_toxic_load(Ct, TLcurrent, model.dt)

    person.toxicload = sum(TL)
    push!(person.TL1, TL[1])
    push!(person.TL2, TL[2])
    push!(person.TL3, TL[3])

    # --- Speed in m/s (base 1.35 m/s); pathfinder expects pixels/s so scale by METERS_TO_PIXELS ---
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


model = StandardABM(
  AgentEscapes,
  space;
  rng          = rng,
  properties   = properties,
  agent_step!  = agent_step!,
  model_step!  = model_step!
)


begin 
    for _ in 1:n_agents
    age = rand(abmrng(model))*(age_range[2]-age_range[1]) + age_range[1]
    mass = rand(abmrng(model)) * (mass_range[2]-mass_range[1]) + mass_range[1]
    vel = Tuple(rand(abmrng(model), 2) .* (speed_range[2]-speed_range[1]) .+ speed_range[1])
    
    # Keep trying to find a valid spawn position that is walkable (not in black areas)
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
        println("Warning: Could not find valid spawn position for agent after $max_attempts attempts")
        continue
    end
    
    person = add_agent!(pos, AgentEscapes, model, vel, age, mass, 1., [pos[1]], [pos[2]], [0.0], [0.0], [0.0])
    plan_best_route!(person, dests, model.pathfinderPM)
end
end


function setupToxic()                                               # Define the setupToxic function
    Atime = [0.0, 2.0, 5.0, 8.0, 15.0, 30.0] #min                # Initialization of the 5 standard AEGL exposure times
    Arho = zeros(3, 6)                                              # the concentration of the three symptoms compared to the AEGL concentrations
    Arho[1, 2:6] = [2.87, 2.33, 2.09, 1.81, 1.55]                   # odor
    Arho[2, 2:6] = [202.38, 164.38, 147.74, 128.10, 109.45]         # irritation
    Arho[3, 2:6] = [286.71, 232.87, 209.30, 181.47, 155.05] #ppm    # edema
    MW = 34                                                         # Molecular weight of H2S in g/mol
    Arho *= MW/24.04 #mg/m^3                                        # Multiply the Arho array by the molecular weight of H2S divided by 24.04
    Arho = Arho'                                                    # Transpose the Arho array 
    Atime = Atime*60 #seconds                                       # Multiply the Atime array by 60 seconds to convert to seconds              
    taumin = 200.                                                   # shortest exposure time over which an AEGL 1, 2 or 3 onset can be reached
    taumax = 86400.                                                 # longest exposure time over which an AEGL 1, 2 or 3 onset can be reached
    Brho = zeros(7,3)                                               # in ppm for each AEGL band at every time step ‘Atime’
    Balpha = zeros(7,3)                                             # power-law exponents
    rhomax = zeros(1,3)                                             # maximum concentration of H2S exposed by each agent
    rhomin = zeros(1,3)                                             # minimum concentration of H2S exposed by each agent
    Btime = zeros(7, 3)                                             # represents an array, which is function of ‘taumin’ and ‘taumax’, that changes depending on alpha, which is a corresponsing array of power low exponents interpolating the ‘Brho’ table array

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
            TL_rate = 1 / Btime[1, k];
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


begin   # Δημιουργία animation με trails & συλλογή CSV θέσης και toxicload
    T = 3375
    const CM_ACTIVE_FRACTION = 0.7
    cm_switch_times = collect(range(0.0, stop = CM_ACTIVE_FRACTION * ((T - 1) * dt), length = NUM_CMS))
    tl_map_idx = Ref(1)
    tl_penalty_map .= cm_list[tl_map_idx[]]
    NPM .= heightmap .+ tl_penalty_map

    # -- Στήσιμο Figure & Axis --
    fig = Figure(; size = (800,800))
    ax  = Makie.Axis(fig[1,1];
               title  = "Evacuation with Toxic Trail",
               aspect = DataAspect())

    heatmap!(ax, heightmap; colormap=:grays, alpha=0.3)
    
    # --- Concentration Map visualization (contour only, overlay on heightmap) ---
    cm_obs = Observable(copy(tl_penalty_map))
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
    contour!(
        ax, cm_obs;
        colormap = cgrad([:yellow, :orange, :red]),
        colorrange = cm_colorrange,
        levels = 10,
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

    # -- Έναρξη record: video και συλλογή δεδομένων ταυτόχρονα --
    video_file = "SCENARIOS/SCENARIO 2/Simulation Results/SCENARIO_2_$(n_agents)_$(seed)_$(run_timestamp).mp4"
    record(fig, video_file, 1:T; framerate=30) do frame
        # 1) ενημέρωση του frame counter
        frame_obs[] = frame
        t_sim = (frame - 1) * dt
        new_idx = searchsortedlast(cm_switch_times, t_sim)
        if new_idx != tl_map_idx[]
            tl_map_idx[] = new_idx
            tl_penalty_map .= cm_list[new_idx]
            NPM .= heightmap .+ tl_penalty_map
        end
        cm_obs[] = tl_penalty_map
        # 2) βήμα προσομοίωσης
        step!(model, 1)

        # 3) ενημέρωση των trails
        for (i,a) in enumerate(allagents(model))
            lines_plots[i][1][] = Point2f.(a.pathX, a.pathY)
        end

        # 4) ενημέρωση θέσεων & δυναμικού χρώματος
        xs = [a.pos[1] for a in allagents(model)]
        ys = [a.pos[2] for a in allagents(model)]
        posobs[] = Point2f.(xs, ys)
        colobs[] = [personcolor(a) for a in allagents(model)]

        # 5) συλλογή δεδομένων στο DataFrame
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

    println("Το animation σώθηκε ως $video_file")

    # -- Εξαγωγή CSV με θέση & toxicload των agents --
    csv_file = "SCENARIOS/SCENARIO 2/Simulation Results/SCENARIO_2_$(n_agents)_$(seed)_$(run_timestamp).csv"
    CSV.write(csv_file, df)
    println("Τα δεδομένα θέσης & toxicload αποθηκεύτηκαν ως $csv_file")
end