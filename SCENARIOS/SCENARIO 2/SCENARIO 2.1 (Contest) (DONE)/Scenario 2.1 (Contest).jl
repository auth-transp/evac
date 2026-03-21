begin   # Φόρτωση των απαραίτητων βιβλιοθηκών
    using Agents
    using Agents.Pathfinding
    using CairoMakie
    using ColorTypes
    using CSV
    using DataFrames
    using DelimitedFiles
    using FileIO: load
    using ImageMagick
    using Images
    using InteractiveDynamics
    using Makie
    using Observables
    using Random
    using Statistics
    using StatsBase
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


begin   # Φόρτωση του heightmap και των hand-drawn penalty maps

    # heightmap
    heightmap_data = load("Maps/Qatargas Map.jpg")
    heightmap_data = permutedims(channelview(heightmap_data), [2,3,1])[:,:,1]
    global heightmap = floor.(Int, convert.(Float64, heightmap_data) * 255)

    # Φόρτωση penalty maps
    penalty_map_data = load("Penalty Map/6.bmp")
    penalty_map_data = permutedims(channelview(penalty_map_data), [2,3,1])[:,:,1]
    global penalty_map = floor.(Int, convert.(Float64, penalty_map_data) * 500)
end

NPM = heightmap + penalty_map # Merging the two maps to create a new penalty map



begin   # Αρχικοποίηση των παραμέτρων του μοντέλου
    const METERS_TO_PIXELS = 0.2692   # 1250 m ≈ 336.5 px on map
    dt = 1.   ## discrete timestep each iteration of the model          # Define the dt variable as 1
    seed = 123  ## seed for random number generator                     # Define the seed variable as 123
    n_agents = 100                                                      # Define the n_agents variable as 100
    toxicity_rate = 0.07                                               # Define the toxicity_rate variable as 0.07
    age_range = (22,60)                                                 # Define the age_range variable as a tuple of 22 and 60
    speed_range = (4.0,7.0)                                            # Define the speed_range variable as a tuple of 4.0 and 7.0
    speed = 5.                                                         # Define the speed variable as 5 
    mass_range = (50,80)                                                # Define the mass_range variable as a tuple of 50 and 80
    ag_range_y = (size(heightmap)[1]/4):(3*size(heightmap)[1]/4)    # Define the ag_range_y variable as a larger range of values from the heightmap array # [1] stands for the 1st row
    ag_range_x = (size(heightmap)[2]/4):(3*size(heightmap)[2]/4)    # Define the ag_range_x variable as a larger range of values from the heightmap array # [2] stands for the 2nd row
    MW = 34 #Molecular weight of H2S in g/mol
    dims = (size(NPM))                                            # Define the dims variable as the dimensions of the heightmap array (2xn matrix)
    walkmap = BitArray(trues(dims...))                                 # Define the walkmap variable as a BitArray of true values with the dimensions of the heightmap array
end    


    #goals
    dests = [(600., 980.), (100., 200.)]

    #Generate the RNG for the model
    rng = MersenneTwister(seed)

    ## Note that the dimensions of the space do not have to correspond to the dimensions
    ## of the pathfinder. Discretisation is handled by the pathfinding methods
    space = ContinuousSpace(size(NPM); periodic = false, spacing = 1)


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



function agent_step!(person, model)
    position = floor.(Int, person.pos)
   # Ct παίρνεται τώρα από το global_penalty_map (hand-drawn maps)
     Ct = penalty_map[position[1], position[2]]
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

    display("Speed: $speed  -  ToxicLoad: $(person.toxicload)")

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

for _ in 1:n_agents
    age = rand(abmrng(model))*(age_range[2]-age_range[1]) + age_range[1]
    mass = rand(abmrng(model)) * (mass_range[2]-mass_range[1]) + mass_range[1]
    vel = Tuple(rand(abmrng(model), 2) .* (speed_range[2]-speed_range[1]) .+ speed_range[1])
    pos = Tuple((rand(abmrng(model), floor.(ag_range_y)), rand(abmrng(model), floor.(ag_range_x))))
    person = add_agent!(pos, AgentEscapes, model, vel, age, mass, 1., [pos[1]], [pos[2]], [0.0], [0.0], [0.0])
    plan_best_route!(person, dests, model.pathfinderPM)
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
    const T = 1852

    # -- Στήσιμο Figure & Axis --
    fig = Figure(; size = (800,800))
    ax  = Makie.Axis(fig[1,1];
               title  = "Evacuation with Toxic Trail",
               aspect = DataAspect())

    heatmap!(ax, heightmap; colormap=:grays, alpha=0.3)
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
    video_file = "SCENARIO 2.1 (Contest) (DONE)/Simulation Results/SCENARIO_2.1_$(seed).mp4"
    record(fig, video_file, 1:T; framerate=30) do frame
        # 1) ενημέρωση του frame counter
        frame_obs[] = frame
        # 2) βήμα προσομοίωσης
        step!(model, agent_step!, model_step!, 1)

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
    csv_file = "SCENARIO 2.1 (Contest) (DONE)/Simulation Results/SCENARIO_2.1_$(seed).csv"
    CSV.write(csv_file, df)
    CSV.write(joinpath("SCENARIO 2.1 (Contest) (DONE)", "Simulation Results", "SCENARIO_2.1_tl_agents_$(seed).csv"),
          select(df, [:step, :agent_id, :toxicload]))
    println("Τα δεδομένα θέσης & toxicload αποθηκεύτηκαν ως $csv_file")
end



begin

    dt = @isdefined(dt) ? dt : 1.0
    seed_str = @isdefined(seed) ? string(seed) : nothing
    folder = joinpath("SCENARIO 2.2 (Contest) (no CM in PM) (DONE)", "Simulation Results")

    # Φόρτωση CSV με step, agent_id, toxicload
    csv_file = seed_str === nothing ? nothing : joinpath(folder, "SCENARIO_2.2_tl_agents_$(seed_str).csv")
    if csv_file === nothing || !isfile(csv_file)
        # αν δεν δοθεί seed, πάρε το πιο πρόσφατο *_tl_agents_*.csv
        csvs = filter(f -> occursin(r"^SCENARIO_2\.2_tl_agents_.*\.csv$", f), readdir(folder))
        @assert !isempty(csvs) "Δεν βρέθηκαν αρχεία *_tl_agents_*.csv στο $(folder)."
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
    T_play  = min(400, T_avail)               # παίξε μέχρι 400 ή όσο υπάρχει
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

    # Χρώματα για τα εσωτερικά bins ανά TL ζώνη (με βάση το center του bin)
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
    out_file = joinpath(folder, "SCENARIO_2.2_hist_$(seed_str).mp4")
    record(fig, out_file, 1:T_play; framerate = 30) do frame
        frame_obs[] = frame
        tl = Vector(df[df.step .== frame, :toxicload])

        count0       = count(==(0.0), tl)                 # ακριβώς 0
        count3       = count(==(3.0), tl)                 # ακριβώς 3
        counts_inner = bin_counts_open(tl, edges_inner)   # μόνο (0,3) ανά 0.5

        y_inner[] = Float64.(counts_inner)
        y0[]      = [Float64(count0)]
        y3[]      = [Float64(count3)]

        # Προαιρετικό auto-scale μόνο στον Υ:
        # ylims!(ax, 0, max(1, maximum(vcat(y0[][1], y_inner[], y3[][1]))))
    end

    println("Βίντεο με ξεχωριστά bins για 0 & 3 και εσωτερικά ανά 0.5 σώθηκε ως: $out_file")
end