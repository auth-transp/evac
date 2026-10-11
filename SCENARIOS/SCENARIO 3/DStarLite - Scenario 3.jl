begin   # Φόρτωση των απαραίτητων βιβλιοθηκών
    using Agents                    # πλαίσιο agent-based μοντελοποίησης (ABM, add_agent!, step!, ...)
    using Agents.Pathfinding        # αλγόριθμοι εύρεσης διαδρομής (DStarLite, AbsolutePenaltyMap, ...)
    using CSV                       # εγγραφή/ανάγνωση αρχείων CSV
    using CairoMakie                # backend σχεδίασης (εικόνες/βίντεο)
    using DataFrames                # πίνακες δεδομένων (DataFrame) για τα αποτελέσματα
    using Dates                     # ημερομηνία/ώρα για το timestamp των αρχείων εξόδου
    using FileIO: load              # φόρτωση εικόνων (heightmap, concentration maps)
    using ImageMagick               # υποστήριξη μορφών εικόνας (jpg, bmp)
    using Images                    # επεξεργασία εικόνων (channelview, ...)
    using InteractiveDynamics       # εργαλεία οπτικοποίησης για Agents.jl
    using Makie                     # βιβλιοθήκη γραφικών
    using Observables               # reactive μεταβλητές για το animation
    using Random                    # γεννήτριες τυχαίων αριθμών (MersenneTwister)
    using StaticArrays              # SVector για γρήγορα διανύσματα σταθερού μεγέθους

    #include("../../Agents/src/Agents.jl")
    #using .Agents               # core Agents.jl API
    #using .Agents.Pathfinding   # DStarLite, AbsolutePenaltyMap, MaxDistance, init_planner, update_after_cm_change!, extract_path, ...

    # φορτώνει τη συνάρτηση append_run_history που καταγράφει κάθε εκτέλεση στο RunHistory
    include(joinpath(@__DIR__, "..", "append_run_history.jl"))
end


# Ορισμός του agent: συνεχής 2D agent (pos, vel προστίθενται αυτόματα από το ContinuousAgent)
Agents.@agent struct AgentEscapes(ContinuousAgent{2, Float64})
    age::Float64                    # ηλικία (έτη)
    mass::Float64                   # μάζα (kg), χρησιμοποιείται στις ελαστικές συγκρούσεις
    toxicload::Float64              # συνολικό τρέχον τοξικό φορτίο (άθροισμα TL1+TL2+TL3)
    path::Vector{Tuple{Int,Int}}    # ΜΟΝΟ grid path
    pathX::Vector{Float64}          # μόνο για visualization
    pathY::Vector{Float64}          # μόνο για visualization
    TL1::Vector{Float64}            # ιστορικό τοξικού φορτίου για το 1ο σύμπτωμα (οσμή)
    TL2::Vector{Float64}            # ιστορικό τοξικού φορτίου για το 2ο σύμπτωμα (ερεθισμός)
    TL3::Vector{Float64}            # ιστορικό τοξικού φορτίου για το 3ο σύμπτωμα (οίδημα)
end


begin   # Φόρτωση του heightmap και των hand-drawn penalty maps (και όλων των CM1..CM7)
    # heightmap (unchanged)
    heightmap_data = load("NADEEN/Maps/Qatargas Map.jpg")                              # φόρτωση της εικόνας του χάρτη
    heightmap_data = permutedims(channelview(heightmap_data), [2,3,1])[:,:,1]          # μετατροπή σε πίνακα (γραμμές, στήλες) και κράτημα του 1ου καναλιού
    global heightmap = floor.(Int, convert.(Float64, heightmap_data) * 255)            # κλιμάκωση σε ακέραιες τιμές 0..255
    heightmap = 255 .- heightmap   # αυτό κάνει την αντιστροφή

    # --- Load all concentration maps 1..7 as Float64 arrays ---
    const NUM_CMS = 7                                               # πλήθος concentration maps (CM1..CM7)
    cm_list = Vector{Array{Float64,2}}(undef, NUM_CMS)              # λίστα που θα κρατά τους 7 χάρτες συγκέντρωσης
    for k in 1:NUM_CMS
        fn = joinpath("Concentration Maps", string(k) * ".bmp")     # διαδρομή αρχείου k.bmp
        img = load(fn)                                              # φόρτωση εικόνας
        img = permutedims(channelview(img), [2,3,1])[:,:,1]         # ίδια μετατροπή σε πίνακα όπως στο heightmap
        cm_list[k] = convert.(Float64, img) .* 255.0                # κλιμάκωση τιμών στο 0..255 (Float64)
    end

    # basic check: dimensions match heightmap
    # έλεγχος ότι κάθε CM έχει τις ίδιες διαστάσεις με το heightmap (αλλιώς σταματά με μήνυμα σφάλματος)
    for k in 1:NUM_CMS
        @assert size(cm_list[k]) == size(heightmap) "Concentration map $k size mismatch with heightmap"
    end

    # start with first CM
    # το penalty_map (τρέχον CM) ξεκινά ως αντίγραφο του 1ου χάρτη
    global penalty_map = copy(cm_list[1])
end

# συνδυασμένος χάρτης κόστους = ανάγλυφο/εμπόδια (heightmap) + τοξική συγκέντρωση (penalty_map)
NPM = heightmap .+ penalty_map
NPM_int = round.(Int, NPM)   # convert to Int for PenaltyMap

begin   # Αρχικοποίηση των παραμέτρων του μοντέλου
    const METERS_TO_PIXELS = 723.37 / 2500.0                            # συντελεστής μετατροπής μέτρων σε pixels του χάρτη
    dt = 1.   ## discrete timestep each iteration of the model          # Define the dt variable as 1
    seed = 123  ## seed for random number generator                     # Define the seed variable as 123
    n_agents = 100                                                        # Define the n_agents variable as 3
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
    # συντεταγμένες (x, y) των δύο εξόδων διαφυγής
    dests = [(500., 854.), (120., 248.)]

    #Generate the RNG for the model
    # γεννήτρια τυχαίων αριθμών με σταθερό seed ώστε οι εκτελέσεις να είναι αναπαραγώγιμες
    rng = MersenneTwister(seed)

    ## Note that the dimensions of the space do not have to correspond to the dimensions
    ## of the pathfinder. Discretisation is handled by the pathfinding methods
    # συνεχής (μη περιοδικός) χώρος με διαστάσεις ίσες με του χάρτη
    space = Pathfinding.ContinuousSpace(size(NPM); periodic = false, spacing = 1)


begin   # Ορισμός του pathfinder D* Lite και των ιδιοτήτων του μοντέλου
    # κόστος κίνησης ανά cell από τον συνδυασμένο χάρτη, με γειτονιά 8 κατευθύνσεων (MaxDistance{2})
    cost_metric_obj = AbsolutePenaltyMap(NPM_int, MaxDistance{2}())
    cost_metric_str = "APM"                       # σύντμηση για τα ονόματα των αρχείων εξόδου
    heuristic_code  = "DF"   # DF: default Agents.jl heuristic, EY: Euclidean, MN: Manhattan
    # D* Lite pathfinder: οι agents δεν περνούν από μη βατά cells (walkmap) και ακολουθούν το cost_metric
    pathfinderPM = DStarLite(space; walkmap = walkmap, cost_metric = cost_metric_obj)
    # ιδιότητες προσβάσιμες μέσα στο μοντέλο ως model.<όνομα>
    properties = (
        pathfinderPM = pathfinderPM,
        heightmap = heightmap,
        dt = dt,
        speed_range = speed_range,
        goal = dests,
    )
end



# Μετακινεί τον agent κατά μήκος ενός προϋπολογισμένου μονοπατιού (λίστα από grid cells), κάνοντας ένα βήμα ταχύτητας `speed` για χρονικό διάστημα `dt` ανά κλήση.
function move_along_precomputed_path!(agent::AgentEscapes, speed, dt)
    # Αν δεν υπάρχει μονοπάτι (άδειο path), δεν κάνει τίποτα
    isempty(agent.path) && return

    # επόμενος grid στόχος: το πρώτο cell του path είναι ο τρέχων στόχος
    cell = agent.path[1]
    target = SVector{2,Float64}(cell[1], cell[2])

    # διάνυσμα και απόσταση από την τρέχουσα θέση προς τον στόχο
    dir = target .- agent.pos
    dist = norm(dir)

    # dist == 0: already on waypoint; must pop without dividing (speed==0 ⇒ 0 < speed*dt is false and 0/0 → NaN)
    # Αν ο agent είναι ήδη στον στόχο (dist == 0) ή τον φτάνει μέσα σε αυτό το βήμα
    # (dist < speed*dt), τον τοποθετεί ακριβώς πάνω του (χωρίς overshoot).
    # Το iszero(dist) είναι απαραίτητο: με speed == 0 η συνθήκη 0 < 0 είναι ψευδής
    # και η διαίρεση dir/dist θα έδινε 0/0 → NaN.
    if iszero(dist) || dist < speed * dt
        agent.pos = target
        popfirst!(agent.path)   # πήγες στο cell: αφαιρείται ώστε την επόμενη κλήση να στοχεύσει το επόμενο
    else
        # Διαφορετικά κινείται κατά speed*dt προς τον στόχο (dir/dist = μοναδιαίο διάνυσμα κατεύθυνσης)
        agent.pos += (dir / dist) * speed * dt
    end
    # Σημείωση: ολοκληρώνεται το πολύ ένα waypoint ανά κλήση· το περισσεύον βήμα δεν μεταφέρεται στο επόμενο cell.
end



# Helper: true if position is within radius of any goal (TL and movement stop when true)
const goal_radius = 10.0    # ακτίνα (σε pixels) γύρω από την έξοδο που θεωρείται ότι ο agent «διέφυγε»
function at_goal(pos, dests, radius = goal_radius)
    px, py = Float64(pos[1]), Float64(pos[2])                                     # θέση του agent ως Float64
    return minimum(hypot(px - d[1], py - d[2]) for d in dests) <= radius          # απόσταση από την πλησιέστερη έξοδο <= ακτίνα
end


# Ένα βήμα του agent σε κάθε χρονική στιγμή: έλεγχος στόχου, ενημέρωση τοξικού φορτίου, ταχύτητα και κίνηση
function agent_step!(person, model)
    # αν ο agent έχει φτάσει σε έξοδο: καταγράφει τη θέση και σταματά (δεν κινείται, δεν συσσωρεύει TL)
    if at_goal(person.pos, model.goal)
        push!(person.pathX, person.pos[1])
        push!(person.pathY, person.pos[2])
        return
    end

    # cell του penalty_map στο οποίο βρίσκεται ο agent (με περιορισμό εντός ορίων)
    grid_dims = size(penalty_map)
    i = clamp(Int(floor(person.pos[1])), 1, grid_dims[1])
    j = clamp(Int(floor(person.pos[2])), 1, grid_dims[2])
    Ct = penalty_map[i, j]                                                       # συγκέντρωση τοξικού στο cell του agent
    TLcurrent = [person.TL1[end], person.TL2[end], person.TL3[end]]              # τελευταίες τιμές TL των τριών συμπτωμάτων
    TL = update_toxic_load(Ct, TLcurrent, model.dt)                              # νέο τοξικό φορτίο μετά από έκθεση dt

    person.toxicload = sum(TL)                                                   # συνολικό τοξικό φορτίο
    push!(person.TL1, TL[1])                                                     # αποθήκευση στο ιστορικό
    push!(person.TL2, TL[2])
    push!(person.TL3, TL[3])

    # --- Speed update based on toxicload ---
    speed = 1.35                                                                 # κανονική ταχύτητα (m/s) χωρίς επίδραση τοξικού
    if 0 < person.toxicload <= 1
        speed = 1.35 * exp(0.393 * person.toxicload)                             # μικρή έκθεση: αύξηση ταχύτητας (πανικός)
    elseif 1 < person.toxicload < 3
        speed = -1.78 * log(person.toxicload) + 2.063                            # μεγαλύτερη έκθεση: η ταχύτητα μειώνεται
    elseif person.toxicload >= 3
        speed = 0.0                                                              # θανατηφόρο φορτίο: ο agent ακινητοποιείται
    end

    # use the global `pathfinderPM` (do not mutate model properties)
    # κίνηση κατά μήκος του path με ταχύτητα μετατραμένη σε pixels
    move_along_precomputed_path!(person, speed * METERS_TO_PIXELS, model.dt)
    push!(person.pathX, person.pos[1])                                           # καταγραφή νέας θέσης για το trail
    push!(person.pathY, person.pos[2])
end



# Βήμα του μοντέλου: ελαστικές συγκρούσεις μεταξύ γειτονικών agents (ακτίνα 0.012, βάσει μάζας)
function model_step!(model)
    for (a1, a2) in interacting_pairs(model, 0.012, :nearest)
        elastic_collision!(a1, a2, :mass)
    end
end

# Δημιουργία του ABM με τον τύπο agent, τον χώρο, τον rng, τις ιδιότητες και τις συναρτήσεις βήματος
model = ABM(
  AgentEscapes,
  space;
  rng          = rng,
  properties   = properties,
  agent_step!  = agent_step!,
  model_step!  = model_step!
)

begin   # Δημιουργία των agents σε τυχαίες βατές θέσεις
    for _ in 1:n_agents
        age = rand(abmrng(model))*(age_range[2]-age_range[1]) + age_range[1]                  # τυχαία ηλικία εντός age_range
        mass = rand(abmrng(model)) * (mass_range[2]-mass_range[1]) + mass_range[1]            # τυχαία μάζα εντός mass_range
        vel = Tuple(rand(abmrng(model), 2) .* (speed_range[2]-speed_range[1]) .+ speed_range[1])  # τυχαία αρχική ταχύτητα (vx, vy)

        # Keep trying to find a valid spawn position that is walkable (not in white/black areas)
        max_attempts = 1000     # μέγιστος αριθμός προσπαθειών εύρεσης βατής θέσης
        attempts = 0
        pos = nothing
        while attempts < max_attempts
            # τυχαία υποψήφια θέση μέσα στην κεντρική περιοχή του χάρτη (ag_range_x/ag_range_y)
            candidate_pos = Tuple((rand(abmrng(model), floor.(ag_range_y)), rand(abmrng(model), floor.(ag_range_x))))
            pos_int = floor.(Int, candidate_pos)                                               # αντίστοιχο cell
            # Check if position is within bounds and walkable
            # αποδοχή της θέσης μόνο αν είναι εντός ορίων και βατή· τότε σταματά η αναζήτηση
            if 1 <= pos_int[1] <= size(walkmap, 1) && 1 <= pos_int[2] <= size(walkmap, 2) && walkmap[pos_int[1], pos_int[2]]
                pos = candidate_pos
                break
            end
            attempts += 1       # αλλιώς μετράει την αποτυχημένη προσπάθεια και ξαναδοκιμάζει
        end

        # If we couldn't find a valid position after max_attempts, skip this agent
        # αν δεν βρέθηκε βατή θέση, εμφανίζει προειδοποίηση και παραλείπει τον agent
        if pos === nothing
            println("Warning: Could not find valid walkable spawn position for agent in Scenario 3 after $max_attempts attempts")
            continue
        end
        
        # προσθήκη του agent στο μοντέλο με αρχικές τιμές για όλα τα πεδία του AgentEscapes
        person = add_agent!(
            pos, AgentEscapes, model,
            vel, age, mass, 1.,        # (αρχικό toxicload = 1.)
            Tuple{Int,Int}[],     # path
            [pos[1]],             # pathX
            [pos[2]],             # pathY
            [0.0], [0.0], [0.0]
        )        # plan using the global `pathfinderPM`
    end
end


# Υπολογίζει τους πίνακες (Balpha, Btime, Brho) του μοντέλου τοξικού φορτίου από τα όρια AEGL του H2S:
# για κάθε σύμπτωμα (οσμή, ερεθισμός, οίδημα) δίνει συγκεντρώσεις, χρόνους και εκθέτες power-law
# που χρησιμοποιεί η update_toxic_load.
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
    # αντιγραφή των πέντε AEGL σημείων (b = 2..6) στους πίνακες Brho/Btime και ορισμός των ακραίων χρόνων taumin/taumax
    for k=1:3
        for b=2:6
            Brho[b, k] = Arho[b, k]
            Btime[b, k] = Atime[b]
        end
        Btime[1, k] = taumin;
         Btime[7, k] = taumax;
    end

    #power low trial values
    # υπολογισμός του εκθέτη power-law (Balpha) μεταξύ διαδοχικών AEGL σημείων· 0 αν οι συγκεντρώσεις είναι ίσες
    for k=1:3
        for b=3:6
            if Brho[b-1, k]==Brho[b, k]
                Balpha[b, k] = 0.0;
            else
                Balpha[b, k] = log(Atime[b]/Atime[b-1])/log(Brho[b-1, k]/Brho[b, k]);
            end
        end

        # επέκταση των εκθετών στα άκρα του πίνακα (b = 1, 2, 7)
        Balpha[2, k] = Balpha[3, k]
        Balpha[1, k] = Balpha[2, k]
        Balpha[7, k] = Balpha[6, k]
        #Balpha[6, k] = Balpha[5, k]
    end
    #extrapolate for the edges
    # εξαγωγή των ακραίων συγκεντρώσεων: rhomax (για taumin) στη θέση 1 και rhomin (για taumax) στη θέση 7
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
    # στα τμήματα με Balpha == 0 (κατώφλι) ο χρόνος παίρνει την τιμή του προηγούμενου σημείου
    for k=1:3
        for b=2:7
            if Balpha[b, k] == 0
                Btime[b, k] = Btime[b-1, k];
            end
        end
    end

    # επαναυπολογισμός του Balpha μετά από κατώφλι (προηγούμενος εκθέτης 0 και τρέχων > 0)
    for k=1:3
        for b=3:5
            if Balpha[b-1, k]==0 && Balpha[b, k]>0
                Balpha[b, k]=log(Btime[b, k]/Btime[b-1, k])/log(Brho[b-1, k]/Brho[b, k]);
            end
        end
    end

    return Balpha, Btime, Brho'
end

# Υπολογισμός των πινάκων τοξικότητας μία φορά (global, χρησιμοποιούνται από την update_toxic_load)
Balpha, Btime, Brho = setupToxic()
# Suffix for exports: new AEGL tables in setupToxic() + 0.5 CM context
run_export_tag = "newAEGL_0.5CM_uncappedTL"                                 # ετικέτα στα ονόματα αρχείων εξόδου
run_timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")            # χρονική σήμανση της εκτέλεσης



# Ενημερώνει το τοξικό φορτίο (TL) των τριών συμπτωμάτων για έκθεση σε συγκέντρωση Ct επί χρόνο dt
function update_toxic_load(Ct, TLcurrent, dt)

    #a = alpha
    #Btime = Btime;
    #Brho = Brho;
    TL = TLcurrent          # τρέχον φορτίο (τροποποιείται in-place)
    TL_rate = 0.0           # ρυθμός αύξησης του TL (s^-1)

    for k = 1:3             # για κάθε σύμπτωμα
        Cmin = Brho[k, 7]   # συγκέντρωση κάτω από την οποία δεν υπάρχει επίδραση
        Cmax = Brho[k, 1]   # συγκέντρωση πάνω από την οποία ο ρυθμός είναι ο μέγιστος
        if Ct > Cmax
            TL_rate = 1 / Btime[1];      # πάνω από Cmax: μέγιστος ρυθμός (1/taumin)
        elseif Ct < Cmin
            TL_rate = 0.0;               # κάτω από Cmin: καμία αύξηση
        else
            # ενδιάμεση συγκέντρωση: βρίσκει το διάστημα (Brho[i-1], Brho[i]) που περιέχει το Ct
            for i in 2:size(Btime, 1)
                if i <= size(Brho, 2)
                    if Brho[k, i-1] < Ct && Ct < Brho[k, i] #check
                        TL_rate = (1/Btime[i, k])*((Ct/Brho[k, i])^(Balpha[i, k])) #Toxic Load rate in units of s^-1
                    end
                end
            end
        end
        TL[k] = TL[k] .+ TL_rate * dt;   # ολοκλήρωση: προσθήκη ρυθμού × χρονικού βήματος

    end
        #TL[iAg, :] .= sum(TL .> 1) .+ TL[min(3, sum(TL .> 1) + 1)] .* (1 - (TL[3] > 1));
        #TL[iAg, :] .= person.toxicload;

    return TL
end



# Στατικό σχέδιο για το plot του ABM: σχεδιάζει τις εξόδους και το trail κάθε agent
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


# Convert continuous world position (Float64, Float64) to grid cell (Int, Int)
# έκδοση για tuple: στρογγυλοποίηση προς τα κάτω και περιορισμός εντός των διαστάσεων του grid
@inline function world_to_cell(p::Tuple{Float64,Float64}, dims::Tuple{Int,Int})
    i = clamp(Int(floor(p[1])), 1, dims[1])
    j = clamp(Int(floor(p[2])), 1, dims[2])
    return (i, j)
end

# ίδια μετατροπή για θέση τύπου SVector (π.χ. agent.pos)
@inline function world_to_cell(p::SVector{2,Float64}, dims::Tuple{Int,Int})
    i = clamp(Int(floor(p[1])), 1, dims[1])
    j = clamp(Int(floor(p[2])), 1, dims[2])
    return (i, j)
end


# Επιστρέφει τα cells (i, j) στα οποία διαφέρουν δύο χάρτες κόστους (χρήσιμο για το replanning του D* Lite)
function find_changed_cells(old::AbstractMatrix{Int}, new::AbstractMatrix{Int})
    cells = Tuple{Int,Int}[]                                    # λίστα αλλαγμένων cells
    @inbounds for i in axes(old,1), j in axes(old,2)
        old[i,j] != new[i,j] && push!(cells, (i,j))             # αν η τιμή άλλαξε, το προσθέτει
    end
    return cells
end


# Διαλέγει το συντομότερο (λιγότερα cells) μονοπάτι ανάμεσα στα μονοπάτια προς τις διάφορες εξόδους
function choose_best_path(paths::Vector{Vector{Tuple{Int,Int}}})
    filter!(!isempty, paths)                                    # αφαιρεί τα άδεια (απροσπέλαστες έξοδοι)
    isempty(paths) && return Tuple{Int,Int}[]                   # αν δεν υπάρχει κανένα, επιστρέφει άδειο path
    return argmin(p -> length(p), paths)                        # το μονοπάτι με το μικρότερο μήκος
end


@time begin   # Δημιουργία animation με trails & συλλογή CSV θέσης και toxicload
    const T = 2500      # συνολικός αριθμός frames/βημάτων της προσομοίωσης
    # CM switch times (s) — Thesis, Table 7: 7 CMs, switching completes at 70% of T = 2500 s.
    # Time counter is t = (frame - 1) * dt.
    const CM_SWITCH_TIMES = [0.0, 292.0, 584.0, 876.0, 1168.0, 1460.0, 1752.0]


    # -- Στήσιμο Figure & Axis --
    fig = Figure(; size = (800,800))
    ax  = Makie.Axis(fig[1,1];
               title  = "Evacuation with Toxic Trail",
               aspect = DataAspect())

    heatmap!(ax, heightmap; colormap=:grays, alpha=0.3)     # φόντο: ο χάρτης (ημιδιαφανής, σε αποχρώσεις του γκρι)

    # --- Concentration Map visualization (contour only, overlay on heightmap) ---
    cm_obs = Observable(penalty_map)    # Observable με το τρέχον CM, ώστε το contour να ανανεώνεται όταν αλλάζει
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
    # ισοϋψείς της συγκέντρωσης (κίτρινο → πορτοκαλί → κόκκινο), 10 επίπεδα
    cm_contour = contour!(
        ax, cm_obs;
        colormap = cgrad([:yellow, :orange, :red]),
        colorrange = cm_colorrange,
        levels = 10,
        linewidth = 1.5,
        alpha = 0.7,
    )

    # σχεδίαση των εξόδων ως κόκκινους κύκλους
    goals = model.goal
    scatter!(ax,
        getindex.(goals,1),
        getindex.(goals,2);
        color  = (:red,50),
        marker = :circle,
    )

    # --- Overlay time counter ---
    frame_obs = Observable(0)       # τρέχον frame (ενημερώνεται σε κάθε βήμα της εγγραφής)

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

    posobs = Observable(Point2f.(xs0, ys0))     # Observable με τις θέσεις όλων των agents
    colobs = Observable(colors0)                # Observable με τα χρώματα των agents

    # One Observable per agent trail: each frame only appends the new points
    # (instead of rebuilding the whole trail from pathX/pathY every frame).
    trail_obs = [Observable([Point2f(a.pos[1], a.pos[2])]) for a in allagents(model)]
    # μία γραμμή (trail) ανά agent, με το χρώμα του
    lines_plots = [
        lines!(ax,
               trail_obs[i];
               color     = personcolor(a),
               linewidth = 2)
        for (i, a) in enumerate(allagents(model))
    ]
    # σημεία των agents στην τρέχουσα θέση τους
    agent_scat = scatter!(ax, posobs;
                          color      = colobs,
                          markersize = 10)

    # ετικέτες με το id κάθε agent δίπλα στο σημείο του
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


    # ονόματα αρχείων εξόδου (βίντεο και CSV) με παραμέτρους της εκτέλεσης και timestamp
    video_file = "SCENARIOS/SCENARIO 3/Simulation Results/DStarLite_SCENARIO_3_$(n_agents)_$(seed)_$(cost_metric_str)_$(heuristic_code)_$(run_export_tag)_$(run_timestamp).mp4"
    csv_file   = "SCENARIOS/SCENARIO 3/Simulation Results/DStarLite_SCENARIO_3_$(n_agents)_$(seed)_$(cost_metric_str)_$(heuristic_code)_$(run_export_tag)_$(run_timestamp).csv"  # added

    # keep a variable for the currently active map index
    # δείκτης του ενεργού concentration map (ξεκινά από το 1ο)
    current_map_idx = 1

    # --- Accumulators for timing (in seconds) ---
    # Pathfinding (initial planners + initial paths)
    path_planning_init_time = Ref(0.0)
    # Pathfinding during concentration-map changes (replanning)
    path_planning_replan_time = Ref(0.0)
    # Simulation time (step! calls only)
    simulation_time = Ref(0.0)
    # Everything inside the record do-block (step!, replanning, observable updates, data collection)
    frame_logic_time = Ref(0.0)

    # --- INIT D* Lite planners (one per exit) ---
    grid_dims = size(NPM_int)
    exit_cells = [world_to_cell(g, grid_dims) for g in model.goal]     # οι έξοδοι ως cells του grid

    # Initial pathfinding: build planners and extract initial best paths
    path_planning_init_time[] += @elapsed begin
        global planners
        # ένας D* Lite planner ανά έξοδο
        planners = [
            init_planner(pathfinderPM, goal_cell)
            for goal_cell in exit_cells
        ]
        for a in allagents(model)
            start = world_to_cell(a.pos, grid_dims)                     # cell εκκίνησης του agent

            paths = [extract_path(p, start) for p in planners]          # μονοπάτι προς κάθε έξοδο
            a.path = choose_best_path(paths)                            # κρατά το συντομότερο

            # Note: Do NOT update pathX/pathY here - they are the traveled trail,
            # not the planned path. They are initialized with the starting position
            # when the agent is created, and will be updated in agent_step! as the agent moves.
        end
    end

    prev_NPM_int = copy(NPM_int)    # αντίγραφο του προηγούμενου χάρτη κόστους, για εντοπισμό αλλαγών


    # Κύρια εγγραφή: για κάθε frame κάνει βήμα προσομοίωσης, ελέγχει αλλαγή CM, ενημερώνει γραφικά και δεδομένα
    record_time = @elapsed record(fig, video_file, 1:T; framerate=30) do frame
        t_frame_start = time_ns()   # αρχή μέτρησης χρόνου του frame
        # update frame counter observable
        frame_obs[] = frame

        # 1) step the model normally (simulation time only)
        simulation_time[] += @elapsed step!(model, 1)

        # 2) determine which CM index should be active on this frame (1..NUM_MAPS)
        global current_map_idx  # Declare `current_map_idx` as global
        t_sim = (frame - 1) * dt                                    # τρέχων χρόνος προσομοίωσης (s)
        new_idx = searchsortedlast(CM_SWITCH_TIMES, t_sim)          # τελευταίο CM του οποίου ο χρόνος ενεργοποίησης έχει περάσει

        # αν άλλαξε το ενεργό CM: ενημέρωση χαρτών, κόστους και επανασχεδιασμός μονοπατιών
        if new_idx != current_map_idx
            current_map_idx = new_idx
            penalty_map .= cm_list[current_map_idx]                 # αντικατάσταση του penalty_map in-place
            # Update concentration map visualization (contour is recomputed only here, when the CM changes)
            cm_obs[] = penalty_map

            # recompute combined penalty map
            NPM = heightmap .+ penalty_map
            NPM_int = round.(Int, NPM)

            # recreate the cost_metric_obj with the new NPM_int
            global cost_metric_obj
            cost_metric_obj = AbsolutePenaltyMap(NPM_int, MaxDistance{2}())

            # update cost metric IN-PLACE (no new DStarLite)
            pathfinderPM.cost_metric.pmap .= NPM_int

            # find changed cells (you already have logic for this)
            changed_cells = find_changed_cells(prev_NPM_int, NPM_int) # <-- use your existing diff logic
            prev_NPM_int .= NPM_int  

            # --- Path planning time (replanning): D* Lite update + per-agent path extraction ---
            path_planning_replan_time[] += @elapsed begin
                # ενημέρωση κάθε planner μόνο για τα cells που άλλαξαν (αυξητικό replanning)
                for planner in planners
                    update_after_cm_change!(planner, changed_cells)
                end
                # νέο μονοπάτι για κάθε agent από την τρέχουσα θέση του
                for a in allagents(model)
                    start = world_to_cell(a.pos, grid_dims)

                    paths = [extract_path(p, start) for p in planners]

                    a.path = choose_best_path(paths)

                    # Note: Do NOT update pathX/pathY here - they are the traveled trail,
                    # not the planned path. They are updated in agent_step! as the agent moves.
                end
            end

            println("Frame $frame: swapped to concentration map $current_map_idx and replanned paths.")
        end

        # 3) update trails/visuals: append only the points added to pathX/pathY since last frame
        for (i, a) in enumerate(allagents(model))
            trail = trail_obs[i][]
            for k in (length(trail) + 1):length(a.pathX)                # μόνο τα νέα σημεία από το προηγούμενο frame
                push!(trail, Point2f(a.pathX[k], a.pathY[k]))
            end
            notify(trail_obs[i])                                        # ειδοποίηση Makie ότι το trail άλλαξε
        end

        # ενημέρωση θέσεων και χρωμάτων των agents στο plot
        xs = [a.pos[1] for a in allagents(model)]
        ys = [a.pos[2] for a in allagents(model)]
        posobs[] = Point2f.(xs, ys)
        colobs[] = [personcolor(a) for a in allagents(model)]

        # 4) collect data
        # καταγραφή (frame, id, x, y, toxicload) κάθε agent στο DataFrame
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
        frame_logic_time[] += (time_ns() - t_frame_start) / 1e9
    end
    # Whatever record() spent outside our do-block is Makie rendering + ffmpeg encoding
    render_time = record_time - frame_logic_time[]      # χρόνος rendering + κωδικοποίησης βίντεο

    println("Το animation σώθηκε ως $video_file")
    CSV.write(csv_file, df)                             # αποθήκευση των δεδομένων θέσης & toxicload
    println("Τα δεδομένα θέσης & toxicload αποθηκεύτηκαν ως $csv_file")

    # --- Timing summary ---
    # συνολικός χρόνος pathfinding = αρχικός σχεδιασμός + replanning
    total_path_planning = path_planning_init_time[] + path_planning_replan_time[]
    println("TIMING BREAKDOWN:")
    println("  Pathfinding time (total): ", round(total_path_planning; digits=6), " s")
    println("    - Initial pathfinding: ", round(path_planning_init_time[]; digits=6), " s")
    println("    - Replanning during CM changes: ", round(path_planning_replan_time[]; digits=6), " s")
    println("  Simulation time (step! only): ", round(simulation_time[]; digits=6), " s")
    println("  Per-frame logic excl. step!/replan (observables, data collection): ",
            round(frame_logic_time[] - simulation_time[] - path_planning_replan_time[]; digits=3), " s")
    println("  Video rendering + encoding (record, excl. per-frame logic): ", round(render_time; digits=3), " s")
    println("  Total record() time: ", round(record_time; digits=3), " s")
end


begin   # Ιστόγραμμα τοξικού φορτίου ανά frame (βίντεο) από το CSV της προσομοίωσης

    dt = @isdefined(dt) ? dt : 1.0                                  # βήμα χρόνου (αν δεν έχει οριστεί, 1 s)
    seed_str = @isdefined(seed) ? string(seed) : nothing            # seed ως string για το όνομα αρχείου
    folder = joinpath("SCENARIOS","SCENARIO 3", "Simulation Results")   # φάκελος αποτελεσμάτων

    # Φόρτωση CSV με step, agent_id, toxicload
    csv_file = seed_str === nothing ? nothing : joinpath(folder, "DStarLite_SCENARIO_3_$(n_agents)_$(seed_str)_$(cost_metric_str)_$(heuristic_code)_$(run_export_tag)_$(run_timestamp).csv")
    if csv_file === nothing || !isfile(csv_file)
        # αν δεν δοθεί seed, πάρε το πιο πρόσφατο *DStarLite_SCENARIO_3_*.csv
        pattern = Regex("^DStarLite_SCENARIO_3_.*_$(cost_metric_str)_$(heuristic_code)_$(run_export_tag)_.*\\.csv\$")
        csvs = filter(f -> occursin(pattern, f), readdir(folder))
        @assert !isempty(csvs) "Δεν βρέθηκαν αρχεία *DStarLite_SCENARIO_3_$(seed)*.csv στο $(folder)."
        stats = stat.(joinpath.(Ref(folder), csvs))
        latest_idx = argmax(getfield.(stats, :mtime))
        csv_file = joinpath(folder, csvs[latest_idx])
        seed_str = "latest"
    end

    df = CSV.read(csv_file, DataFrame)      # ανάγνωση του CSV σε DataFrame

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
    # στρογγυλοποίηση των ticks του άξονα x σε 1 δεκαδικό (για μία τιμή και για διάνυσμα τιμών)
    function mytickfmt(x::Real)
        string(round(x; digits=1))
    end
    function mytickfmt(xs::AbstractVector{<:Real})
        string.(round.(xs; digits=1))
    end

    # --- Παράμετροι χρόνου / y-scale ---
    T_avail = maximum(df.step)              # διαθέσιμα frames στο CSV
    T_play  = min(T, T_avail)               # παίξε μέχρι 400 ή όσο υπάρχει
    n_agents_guess = maximum(combine(groupby(df, :step), nrow).nrow)    # μέγιστος αριθμός agents ανά frame (όριο του άξονα y)

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

    xt = collect(0.0:0.5:3.0)                  # θέσεις ticks στον άξονα x
    ax.xticks = xt
    ax.xtickformat = mytickfmt
    ax.xticklabelrotation = pi/4               # προαιρετικό

    # Overlay χρόνου
    frame_obs = Observable(0)                  # τρέχον frame για το label χρόνου
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
    out_file = joinpath(folder, "DStarLite_SCENARIO_3_hist__$(n_agents)_$(seed_str)_$(cost_metric_str)_$(heuristic_code)_$(run_export_tag)_$(run_timestamp).mp4")   # αρχείο εξόδου του ιστογράμματος
    record(fig, out_file, 1:T_play; framerate = 30) do frame
        frame_obs[] = frame
        tl = Vector(df[df.step .== frame, :toxicload])    # τοξικά φορτία όλων των agents στο συγκεκριμένο frame

        count0       = count(==(0.0), tl)                 # ακριβώς 0
        count3       = count(==(3.0), tl)                 # ακριβώς 3
        counts_inner = bin_counts_open(tl, edges_inner)   # μόνο (0,3) ανά 0.5

        # ενημέρωση των υψών των μπαρών (το Makie ξανασχεδιάζει αυτόματα)
        y_inner[] = Float64.(counts_inner)
        y0[]      = [Float64(count0)]
        y3[]      = [Float64(count3)]

        # Προαιρετικό auto-scale μόνο στον Υ:
        # ylims!(ax, 0, max(1, maximum(vcat(y0[][1], y_inner[], y3[][1]))))
    end

    println("Βίντεο με ξεχωριστά bins για 0 & 3 και εσωτερικά ανά 0.5 σώθηκε ως: $out_file")

    # καταγραφή της εκτέλεσης (αρχεία εξόδου, χρόνος pathfinding, timestamp) στο RunHistory
    append_run_history(
        scenario = "SCENARIO 3",
        runner_script = "SCENARIOS/SCENARIO 3/DStarLite - Scenario 3.jl",
        csv_file = csv_file,
        mp4_file = video_file,
        hist_mp4_file = out_file,
        pathfinding_time = total_path_planning,
        run_timestamp = run_timestamp,
    )
end
