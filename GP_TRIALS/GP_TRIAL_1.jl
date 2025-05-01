begin   # Φόρτωση των απαραίτητων βιβλιοθηκών
    using Agents
    using Agents.Pathfinding
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


begin   # Φόρτωση του heightmap και του concentrationmap
    heightmap_data = load("Maps/Qatargas Map.jpg")                                      # Load the heightmap data
    heightmap_data = permutedims(channelview(heightmap_data), [2,3,1])[:,:,1]           # Permute the dimensions of the heightmap data
    heightmap = floor.(Int, convert.(Float64, heightmap_data)*255)                      # Convert the heightmap data to a 2D array of integers
    concentration_data = load("Maps/concentrationmap new.jpg")                          # Load the concentration data 
    concentration_data = permutedims(channelview(concentration_data), [2,3,1])[:,:,1]   # Permute the dimensions of the concentration data
    concentrationmap = floor.(Int, convert.(Float64, concentration_data .*500))         # Convert the concentration data to a 2D array of integers        # Convert the concentration data to a 2D array of integers
end


begin   # Αρχικοποίηση των παραμέτρων του μοντέλου
    dt = 1.   ## discrete timestep each iteration of the model          # Define the dt variable as 1
    seed = 123  ## seed for random number generator                     # Define the seed variable as 123
    n_agents = 3                                                        # Define the n_agents variable as 3
    toxicity_rate = 0.07                                               # Define the toxicity_rate variable as 0.07
    age_range = (22,60)                                                 # Define the age_range variable as a tuple of 22 and 60
    speed_range = (4.0,7.0)                                            # Define the speed_range variable as a tuple of 4.0 and 7.0
    speed = 5.                                                         # Define the speed variable as 5 
    mass_range = (50,80)                                                # Define the mass_range variable as a tuple of 50 and 80
    ag_range_y = (size(heightmap)[1]/2-50):(size(heightmap)[1]/2+50)    # Define the ag_range_y variable as a range of values from the heightmap array # [1] stands for the 1st row
    ag_range_x = (size(heightmap)[2]/2-50):(size(heightmap)[2]/2+50)    # Define the ag_range_x variable as a range of values from the heightmap array # [2] stands for the 2nd row
    MW = 34 #Molecular weight of H2S in g/mol
    dims = (size(heightmap))                                            # Define the dims variable as the dimensions of the heightmap array (2xn matrix)
    walkmap = BitArray(trues(dims...))                                 # Define the walkmap variable as a BitArray of true values with the dimensions of the heightmap array
end    


    #goals
    dests = [(600., 980.), (100., 200.)]

    #Generate the RNG for the model
    rng = MersenneTwister(seed)

    ## Note that the dimensions of the space do not have to correspond to the dimensions
    ## of the pathfinder. Discretisation is handled by the pathfinding methods
    space = ContinuousSpace(size(heightmap); periodic = false, spacing = 1)


begin
    pathfinder = AStar(space; walkmap = walkmap, cost_metric = PenaltyMap(heightmap, MaxDistance{2}()))
    properties = (
        pathfinder = pathfinder,
        heightmap = heightmap,
        dt = dt,
        speed_range = speed_range,
        goal = dests
    )
end



function agent_step!(person, model)
    position = floor.(Int, person.pos)
    Ct = concentrationmap[position[1], position[2]]
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

    display("Speed: $speed  -  ToxicLoad: $(person.toxicload)")

    move_along_route!(person, model, model.pathfinder, speed, dt)
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
    plan_best_route!(person, dests, model.pathfinder)
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
    scatter!(ax, xs_g, ys_g; color = (:red, 50), marker = 'o')

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
    const T = 300

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
        marker = :o,
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
    video_file = "EVAC_TOXIC_TRAIL_$(seed).mp4"
    record(fig, video_file, 1:T) do frame
        # 1) βήμα προσομοίωσης
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
    csv_file = "agent_trajectories_$(seed).csv"
    CSV.write(csv_file, df)
    println("Τα δεδομένα θέσης & toxicload αποθηκεύτηκαν ως $csv_file")
end