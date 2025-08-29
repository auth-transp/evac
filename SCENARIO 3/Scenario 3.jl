begin
    using Agents
    #using Agents.Pathfinding
    using Random
    using ColorTypes
    using ImageMagick
    using FileIO: load
    using InteractiveDynamics
    using Images
    using DataFrames
    using Statistics
    using CairoMakie
    using DelimitedFiles
    using Observables
    using Makie
    using CSV

    include("DStarLite.jl")
    # import τις βασικές συναρτήσεις που θέλουμε (όχι όλες, μπορούμε να χρησιμοποιήσουμε και qualified calls)
    using .DStarLiteModule: DStarLite, plan_best_route!, penaltymap, handle_map_changes!, compute_shortest_path!, get_next_step
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
    dstar::Any   # αποθηκεύουμε εδώ το per-agent D* instance (AgentDS)
end

# ---------------------- Load maps & penalty maps ----------------------
begin
    # heightmap (sat image)
    heightmap_data = load("Maps/Qatargas Map.jpg")
    heightmap_data = permutedims(channelview(heightmap_data), [2,3,1])[:,:,1]
    global heightmap = floor.(Int, convert.(Float64, heightmap_data) * 255)

    # penalty maps folder
    penalty_dir = joinpath("Penalty Map")
    penalty_files = sort(readdir(penalty_dir))
    @assert !isempty(penalty_files) "Δεν βρέθηκαν penalty maps στον φάκελο $penalty_dir"
    penalty_images = [load(joinpath(penalty_dir, f)) for f in penalty_files]

    global penalty_maps = [
        begin
            img = permutedims(channelview(pi), [2,3,1])[:,:,1]
            floor.(Int, convert.(Float64, img .* 500))
        end for pi in penalty_images
    ]

    for pm in penalty_maps
        @assert size(pm) == size(heightmap) "Penalty map size does not match heightmap size"
    end

    global n_penalties = length(penalty_maps)
    global current_penalty = 1
    global global_penalty_map = deepcopy(penalty_maps[current_penalty])

    # obstacle threshold: πάνω από αυτό θεωρούμε "μπλοκ/ανηχόρ"
    global obstacle_threshold = 250

    function get_changed_nodes(oldmap::AbstractMatrix{<:Number}, newmap::AbstractMatrix{<:Number})
        changed = Tuple{Int,Int}[]
        @inbounds for i in axes(oldmap,1), j in axes(oldmap,2)
            if oldmap[i,j] != newmap[i,j]
                push!(changed, (i,j))
            end
        end
        return changed
    end

    function apply_penalty_index!(model, idx::Integer)
        @assert 1 <= idx <= n_penalties "Penalty index out of range"
        global global_penalty_map

        oldmap = deepcopy(global_penalty_map)
        newmap = penalty_maps[idx]
        global_penalty_map = deepcopy(newmap)

        # Πάρε properties με ασφαλή τρόπο (παρακάμπτουμε πιθανό Agents.getproperty override)
        props = try
            getfield(model, :properties)
        catch _e
            nothing
        end

        # Αποφάσισε obstacle_threshold — αν υπάρχει στα properties το παίρνουμε, αλλιώς default 250
        obstacle_threshold = 250
        if props !== nothing
            if isa(props, NamedTuple)
                if (:obstacle_threshold in propertynames(props))
                    obstacle_threshold = props.obstacle_threshold
                end
            elseif isa(props, Dict)
                obstacle_threshold = get(props, :obstacle_threshold, obstacle_threshold)
            end
        end

        # compute binary walkmaps (true = walkable)
        old_walkmap = oldmap .< obstacle_threshold
        new_walkmap = newmap .< obstacle_threshold

        # changed cells where walkability toggled
        changed_cells = [(i,j) for i in axes(old_walkmap,1), j in axes(old_walkmap,2) if old_walkmap[i,j] != new_walkmap[i,j]]

        # produce a simple 'vc' structure that handle_map_changes! accepts
        vc = [(cell, nothing) for cell in changed_cells]

        # Προσπάθησε να βρεις top-level pathfinder (pf) από props με ασφαλή τρόπο
        pf = nothing
        if props !== nothing
            if isa(props, NamedTuple)
                pf = get(props, :pathfinder, nothing)
            elseif isa(props, Dict)
                pf = get(props, :pathfinder, nothing)
            end
        end

        # ενημέρωση κάθε agent που έχει dstar
        for a in allagents(model)
            if hasfield(a, :dstar) && a.dstar !== nothing
                try
                    # ενημέρωσε sensed_walkmap
                    a.dstar.sensed_walkmap .= new_walkmap

                    # ενημέρωσε incremental map-changes (module function)
                    try
                        DStarLiteModule.handle_map_changes!(a.dstar, vc)
                    catch e_handle
                        @warn "handle_map_changes! απέτυχε για agent $(a.id): $e_handle"
                    end

                    # επανυπολόγισε shortest path (μινιμαλιστική συνάρτηση του module)
                    try
                        DStarLiteModule.compute_shortest_path!(a.dstar)
                    catch e_compute
                        @warn "compute_shortest_path! απέτυχε για agent $(a.id): $e_compute"
                    end

                    # ενημέρωσε την αντιστοιχία στον top-level pathfinder (αν υπάρχει)
                    if pf !== nothing
                        try
                            # Κατασκευάζουμε νέο path από το ds και το γράφουμε στο pf
                            newpath = try
                                DStarLiteModule.build_path_from_ds(a.dstar)
                            catch e_bp
                                @warn "build_path_from_ds απέτυχε για agent $(a.id): $e_bp"
                                nothing
                            end

                            if newpath !== nothing
                                pf.agent_paths[a.id] = newpath
                            else
                                # fallback: κάνουμε πλήρη replanning μέσω της exposed helper plan_best_route! (αν υπάρχει)
                                try
                                    DStarLiteModule.plan_best_route!(a, getfield(model, :goal), pf)
                                catch e_plan
                                    @warn "Fallback plan_best_route! απέτυχε για agent $(a.id): $e_plan"
                                end
                            end
                        catch e
                            @warn "Σφάλμα κατά την ενημέρωση pf.agent_paths για agent $(a.id): $e"
                        end
                    end
                catch e
                    @warn "Σφάλμα ενημέρωσης D* για agent $(a.id): $e"
                end
            end
        end

        # Ενημέρωση pf.cost_metric.penalty_map (αν υπάρχει)
        if pf !== nothing
            try
                if hasproperty(pf, :cost_metric) && pf.cost_metric !== nothing
                    # προτίμηση σε πεδίο penalty_map, fallback σε pmap
                    if hasproperty(pf.cost_metric, :penalty_map)
                        pf.cost_metric.penalty_map .= newmap
                    elseif hasproperty(pf.cost_metric, :pmap)
                        pf.cost_metric.pmap .= newmap
                    else
                        # αν cost_metric δεν έχει αυτά τα πεδία, προσπάθησε να βάλεις νέο πεδίο (προαιρετικά)
                        @debug "cost_metric δεν έχει πεδίο penalty_map ή pmap — δεν ενημερώθηκε"
                    end
                end
            catch e
                @warn "Δεν κατέστη δυνατό να ενημερωθεί το pathfinder.cost_metric.penalty_map: $e"
            end
        end
        return changed_cells
    end

    global sim_time = 0.0
    global penalty_interval = 120.0
    global next_penalty_time = penalty_interval

    function maybe_update_penalty!(model, dt)
        global sim_time, next_penalty_time, current_penalty
        sim_time += dt
        if sim_time >= next_penalty_time
            next_penalty_time += penalty_interval
            current_penalty = (current_penalty % n_penalties) + 1
            @info "Switching to penalty map $current_penalty at sim_time=$(sim_time)s"
            changed = apply_penalty_index!(model, current_penalty)
            return changed
        end
        return Tuple{Int,Int}[]
    end
end

# ------------------- Model parameters -------------------
begin
    dt = 1.0
    seed = 123
    n_agents = 3
    toxicity_rate = 0.07
    age_range = (22,60)
    speed_range = (4.0,7.0)
    speed = 5.0
    mass_range = (50,80)
    ag_range_y = (size(heightmap)[1]÷2-50):(size(heightmap)[1]÷2+50)
    ag_range_x = (size(heightmap)[2]÷2-50):(size(heightmap)[2]÷2+50)
    MW = 34
    dims = (size(heightmap))
    walkmap = BitArray(trues(dims...))
end

dests = [(600., 980.), (100., 200.)]
rng = MersenneTwister(seed)
space = ContinuousSpace(size(heightmap); periodic=false, spacing=1)

# ------------------- Create top-level DStarLite pathfinder (grid dims) -------------------
begin
    # create DStarLite using dims (rows,cols) — όχι ContinuousSpace
    pathfinder = DStarLite((size(heightmap,1), size(heightmap,2)); walkmap=walkmap, cost_metric=PenaltyMap(heightmap, MaxDistance{2}()))
    properties = (
        pathfinder = pathfinder,
        heightmap = heightmap,
        dt = dt,
        speed_range = speed_range,
        goal = dests
    )
end

# ------------------- helper: move using dstar if present -------------------
function dstar_move_one_step!(person::AgentEscapes, model, speed::Real, dt::Real)
    # if no dstar, fallback to existing move_along_route!
    if person.dstar === nothing
        move_along_route!(person, model, model.properties[:pathfinder], speed, dt)
        return
    end

    ds = person.dstar
    # sync ds.s_start with agent current grid cell
    # IMPORTANT: module uses tuple (row, col) indexing; we map continuous pos (x,y) -> (row=Int(y), col=Int(x))
    cur_row = clamp(floor(Int, person.pos[2]), 1, ds.rows)
    cur_col = clamp(floor(Int, person.pos[1]), 1, ds.cols)
    ds.s_start = (cur_row, cur_col)

    # ask ds for next cell (returns (row,col))
    nextcell = DStarLiteModule.get_next_step(ds)
    if nextcell === nothing
        return
    end

    # convert nextcell (row,col) -> target (x,y) continuous coords
    target_x = Float64(nextcell[2])
    target_y = Float64(nextcell[1])

    dx = target_x - person.pos[1]
    dy = target_y - person.pos[2]
    dist = sqrt(dx^2 + dy^2)
    if dist == 0.0
        # we reached center of next cell: update s_start
        ds.s_start = nextcell
        return
    end

    maxstep = speed * dt
    if dist <= maxstep
        # step to center and mark start as the next cell
        person.pos = (target_x, target_y)
        ds.s_start = nextcell
    else
        # partial step
        person.pos = (person.pos[1] + dx / dist * maxstep, person.pos[2] + dy / dist * maxstep)
    end
end

# ------------------- agent_step! (uses dstar_move_one_step!) -------------------
function agent_step!(person, model)
    position = floor.(Int, person.pos)
    Ct = global_penalty_map[position[1], position[2]]
    TLcurrent = [person.TL1[end], person.TL2[end], person.TL3[end]]
    TL = update_toxic_load(Ct, TLcurrent, dt)

    person.toxicload = sum(TL)
    push!(person.TL1, TL[1])
    push!(person.TL2, TL[2])
    push!(person.TL3, TL[3])

    # --- Speed update based on toxicload ---
    spd = 1.35
    if 0 < person.toxicload <= 1
        spd = 1.35 * exp(0.393 * person.toxicload)
    elseif 1 < person.toxicload < 3
        spd = -1.78 * log(person.toxicload) + 2.063
    elseif person.toxicload >= 3
        spd = 0.0
    end

    display("Speed: $spd  -  ToxicLoad: $(person.toxicload)")

    # move using per-agent D*
    dstar_move_one_step!(person, model, spd, dt)

    push!(person.pathX, person.pos[1])
    push!(person.pathY, person.pos[2])
end

# model_step! as before
function model_step!(model)
    for (a1, a2) in interacting_pairs(model, 0.012, :nearest)
        elastic_collision!(a1, a2, :mass)
    end
end

# create model
model = ABM(
  AgentEscapes,
  space;
  rng          = rng,
  properties   = properties,
  agent_step!  = agent_step!,
  model_step!  = model_step!
)



# ensure initial penalty map applied
apply_penalty_index!(model, current_penalty)


# Πάρε raw properties και pathfinder μία φορά
_props = getfield(model, :properties)           # παρακάμπτει Agents.getproperty
pf = isa(_props, NamedTuple) ? _props.pathfinder : (isa(_props, Dict) ? get(_props, :pathfinder, nothing) : nothing)

if pf === nothing
    @warn "Δεν βρέθηκε pathfinder στα model.properties"
end


# create agents and call plan_best_route! (which will attach AgentDS -> agent.dstar)
for _ in 1:n_agents
    # sample properties
    age = rand(abmrng(model))*(age_range[2]-age_range[1]) + age_range[1]
    mass = rand(abmrng(model)) * (mass_range[2]-mass_range[1]) + mass_range[1]
    vel = Tuple(rand(abmrng(model), 2) .* (speed_range[2]-speed_range[1]) .+ speed_range[1])

    # generate integer grid indices but convert to Float64 to match ContinuousAgent{Float64}
    y_i = rand(abmrng(model), floor.(ag_range_y))
    x_i = rand(abmrng(model), floor.(ag_range_x))
    pos = (Float64(y_i), Float64(x_i))   # (y,x) όπως έχεις συνηθίσει

    # 1) Δημιούργησε τον agent: model πρώτο, μετά τύπος, μετά pos
    person = add_agent!(model, AgentEscapes, pos)

    # 2) Θέσε τα υπόλοιπα πεδία ρητά (safer από μεγάλα positional args)
    # - vel είναι πεδίο του ContinuousAgent supertype -> το θέτεις απευθείας
    person.vel = vel
    person.age = age
    person.mass = mass
    person.toxicload = 1.0
    person.pathX = [pos[1]]
    person.pathY = [pos[2]]
    person.TL1 = [0.0]
    person.TL2 = [0.0]
    person.TL3 = [0.0]
    person.dstar = nothing

    # 3) Πάρε pathfinder από model.properties με ασφάλεια
    pf = nothing
    props = try
        getfield(model, :properties)
    catch
        nothing
    end

    if props !== nothing
        if isa(props, NamedTuple) || isa(props, Base.Something) # safe check
            pf = get(props, :pathfinder, nothing)
        elseif isa(props, Dict)
            pf = get(props, :pathfinder, nothing)
        end
    end

    if pf === nothing
        @warn "Αποφεύγω υπολογισμό διαδρομής για agent $(person.id): δεν βρέθηκε pathfinder"
    else
        # καλούμε τη συνάρτηση του D* module που φτιάχνει το AgentDS και βάζει path
        plan_best_route!(person, dests, pf)
    end
end



function setupToxic()                                               # Define the setupToxic function
    Atime = [0.0, 0.17, 0.83, 1.67, 4.17, 8.33] #min                # Define the Atime array as a 1x6 matrix                                    # Initialization of the 5 standard AEGL exposure times
    Arho = zeros(3, 6)                                              # Define the Arho array as a 3x6 matrix                                     # the concentration of the three symptoms compared to the AEGL concentrations
    Arho[1, 2:6] = [4.85, 4.23, 4.17, 4.06, 3.82]                   # Define the Arho array for the first row and columns 2 to 6                # odor
    Arho[2, 2:6] = [180.79, 157.56, 155.43, 151.37, 142.48]         # Define the Arho array for the second row and columns 2 to 6               # irritation
    Arho[3, 2:6] = [485.62, 423.22, 417.49, 406.59, 382.71] #ppm    # Define the Arho array for the third row and columns 2 to 6                # edema
    MW = 34 #Molecular Weight of H2S
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


begin   # Δημιουργία animation με trails & συλλογή CSV θέσης και toxicload
    const T = 1200

    # -- Στήσιμο Figure & Axis --
    fig = Figure(resolution = (800,800))
    ax  = Makie.Axis(fig[1,1];
               title  = "Evacuation with Toxic Trail",
               aspect = DataAspect())

    heatmap!(ax, penaltymap(model.pathfinder); colormap=:grays, alpha=0.3)
    goals = model.goal
    scatter!(ax,
        getindex.(goals,1),
        getindex.(goals,2);
        color  = (:red,50),
        marker = :circle,
    )

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
    video_file = "SCENARIO 3/Simulation Results/SCENARIO_3_$(seed).mp4"
    record(fig, video_file, 1:T; framerate=30) do frame
        # 1) βήμα προσομοίωσης
        maybe_update_penalty!(model, dt)
        step!(model, agent_step!, model_step!, 1)

        # 2) ενημέρωση των trails
        for (i,a) in enumerate(allagents(model))
            lines_plots[i][1][] = Point2f.(a.pathX, a.pathY)
        end

        # 3) ενημέρωση θέσεων & δυναμικού χρώματος
        xs = [a.pos[1] for a in allagents(model)]
        ys = [a.pos[2] for a in allagents(model)]
        posobs[] = Point2f.(xs, ys)
        colobs[] = [personcolor(a) for a in allagents(model)]

        # 4) συλλογή δεδομένων στο DataFrame
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
    csv_file = "SCENARIO 3/Simulation Results/SCENARIO_3_$(seed).csv"
    CSV.write(csv_file, df)
    println("Τα δεδομένα θέσης & toxicload αποθηκεύτηκαν ως $csv_file")
end