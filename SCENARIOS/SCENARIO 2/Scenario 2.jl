begin   # Φόρτωση των απαραίτητων βιβλιοθηκών
    using Agents  # πλαίσιο agent-based μοντελοποίησης (ABM, add_agent!, step!, ...)
    using Agents.Pathfinding  # αλγόριθμοι εύρεσης διαδρομής (AStar, PenaltyMap, plan_best_route!, move_along_route!, ...)
    using LinearAlgebra  # γραμμική άλγεβρα (norm για αποστάσεις)
    using Random  # γεννήτριες τυχαίων αριθμών (MersenneTwister)
    using ColorTypes  # τύποι χρωμάτων
    using ImageMagick  # υποστήριξη μορφών εικόνας (jpg, bmp)
    using FileIO: load  # φόρτωση εικόνων (heightmap, concentration maps)
    using InteractiveDynamics  # εργαλεία οπτικοποίησης για Agents.jl
    using Images  # επεξεργασία εικόνων (channelview, ...)
    using DataFrames  # πίνακες δεδομένων (DataFrame) για τα αποτελέσματα
    using Dates  # ημερομηνία/ώρα για το timestamp των αρχείων εξόδου
    using Statistics  # στατιστικές συναρτήσεις
    using CairoMakie  # backend σχεδίασης (εικόνες/βίντεο)
    using DelimitedFiles  # ανάγνωση/εγγραφή αρχείων με διαχωριστικά
    using Observables  # reactive μεταβλητές για το animation
    using Makie  # βιβλιοθήκη γραφικών
    using CSV  # εγγραφή/ανάγνωση αρχείων CSV

    # φορτώνει τη συνάρτηση append_run_history που καταγράφει κάθε εκτέλεση στο RunHistory
    include(joinpath(@__DIR__, "..", "append_run_history.jl"))
end



# Ορισμός του agent: συνεχής 2D agent (pos, vel προστίθενται αυτόματα από το ContinuousAgent)
@agent struct AgentEscapes(ContinuousAgent{2, Float64}) # Αρχικοποίηση των Agents
    age::Float64  # ηλικία (έτη)
    mass::Float64  # μάζα (kg), χρησιμοποιείται στις ελαστικές συγκρούσεις
    toxicload::Float64  # συνολικό τρέχον τοξικό φορτίο (άθροισμα TL1+TL2+TL3)
    pathX::Vector{Float64}  # ιστορικό x θέσεων του agent (για το trail στο animation)
    pathY::Vector{Float64}  # ιστορικό y θέσεων του agent (για το trail στο animation)
    TL1::Vector{Float64}  # ιστορικό τοξικού φορτίου για το 1ο σύμπτωμα (οσμή)
    TL2::Vector{Float64}  # ιστορικό τοξικού φορτίου για το 2ο σύμπτωμα (ερεθισμός)
    TL3::Vector{Float64}  # ιστορικό τοξικού φορτίου για το 3ο σύμπτωμα (οίδημα)
end

    
begin   # Heightmap για χώρο / A* · σταθερό CM4 μόνο για το κόστος του pathfinding · CM1..CM7 μόνο για TL και οπτικό overlay
    heightmap_data = load("NADEEN/Maps/Qatargas Map.jpg")  # φόρτωση της εικόνας του χάρτη
    heightmap_data = permutedims(channelview(heightmap_data), [2,3,1])[:,:,1]  # μετατροπή σε πίνακα (γραμμές, στήλες) και κράτημα του 1ου καναλιού
    global heightmap = floor.(Int, convert.(Float64, heightmap_data) * 255)  # κλιμάκωση σε ακέραιες τιμές 0..255
    heightmap = 255 .- heightmap  # αντιστροφή: το μαύρο της εικόνας γίνεται υψηλή τιμή (λευκό στο αντεστραμμένο heightmap, μη βατό στο walkmap) και το λευκό γίνεται 0

    # Fixed map for pathfinding cost: CM4 (Thesis, Scenario 2). It never changes and does NOT affect the TL.
    pf_map_data = load("Concentration Maps/4.bmp")  # φόρτωση του concentration map CM4, που χρησιμοποιείται σταθερά για το κόστος του pathfinding
    pf_map_data = permutedims(channelview(pf_map_data), [2,3,1])[:,:,1]  # μετατροπή σε πίνακα (γραμμές, στήλες) και κράτημα του 1ου καναλιού
    global penalty_map = floor.(Int, convert.(Float64, pf_map_data) * 255.0)  # κλιμάκωση σε ακέραιες τιμές 0..255 (μόνο για το pathfinding)

    # Dynamic maps for TL accumulation only: CM1..CM7

    const NUM_CMS = 7  # πλήθος concentration maps (CM1..CM7)
    global cm_list = Vector{Array{Float64,2}}(undef, NUM_CMS)  # λίστα που κρατά τους 7 χάρτες συγκέντρωσης
    for k in 1:NUM_CMS  # φόρτωση κάθε CM
        fn = joinpath("Concentration Maps", string(k) * ".bmp")  # διαδρομή αρχείου k.bmp
        img = load(fn)  # φόρτωση εικόνας
        img = permutedims(channelview(img), [2, 3, 1])[:, :, 1]  # ίδια μετατροπή σε πίνακα όπως στο heightmap
        cm_list[k] = convert.(Float64, img) .* 255.0  # κλιμάκωση τιμών στο 0..255 (Float64)
        @assert size(cm_list[k]) == size(heightmap) "Concentration map $k size mismatch with heightmap"  # έλεγχος ότι το CM έχει τις ίδιες διαστάσεις με το heightmap
    end
    global tl_penalty_map = copy(cm_list[1])  # ενεργός χάρτης συγκέντρωσης για το τοξικό φορτίο (TL), αρχικά το CM1· εναλλάσσεται CM1→CM7 και επηρεάζει μόνο το TL
    @assert size(penalty_map) == size(heightmap) "penalty_map dimensions $(size(penalty_map)) do not match heightmap dimensions $(size(heightmap))"  # έλεγχος ότι το CM4 έχει τις ίδιες διαστάσεις με το heightmap
end

NPM = heightmap .+ penalty_map  # χάρτης κόστους pathfinding = heightmap + CM4 (σταθερός, ακέραιος)

begin   # Αρχικοποίηση των παραμέτρων του μοντέλου
    # Time–speed correlation: distance per step = speed × dt (in space units).
    # Treat dt as "time per step" (e.g. 1 = 1 second). Map scale: 2500 m ≈ 723.37 px.
    const METERS_TO_PIXELS = 723.37 / 2500.0  # συντελεστής μετατροπής μέτρων σε pixels του χάρτη
    dt = 1.   ## discrete timestep each iteration of the model          # Define the dt variable as 1  # διακριτό χρονικό βήμα του μοντέλου (1 s)
    seed = 123  ## seed for random number generator                     # Define the seed variable as 123  # seed της γεννήτριας τυχαίων αριθμών
    run_timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")  # timestamp για τα ονόματα των αρχείων εξόδου
    n_agents = 50                                                        # πλήθος agents
    age_range = (22,60)                                                 # Define the age_range variable as a tuple of 22 and 60  # εύρος ηλικιών των agents (έτη)
    speed_range = (4.0,7.0)                                            # Define the speed_range variable as a tuple of 4.0 and 7.0  # εύρος της αρχικής ταχύτητας vel των agents· η πραγματική ταχύτητα κίνησης προκύπτει από το τοξικό φορτίο
    mass_range = (50,80)                                                # Define the mass_range variable as a tuple of 50 and 80  # εύρος μάζας των agents (kg)
    ag_range_y = (size(heightmap)[1]/4):(3*size(heightmap)[1]/4)    # Define the ag_range_y variable as a larger range of values from the heightmap array # [1] stands for the 1st dimension (rows)  # περιοχή εμφάνισης agents: κεντρικό μισό του χάρτη (από 1/4 έως 3/4 της 1ης διάστασης)
    ag_range_x = (size(heightmap)[2]/4):(3*size(heightmap)[2]/4)    # Define the ag_range_x variable as a range of values from the heightmap array # [2] stands for the 2nd dimension (columns)  # περιοχή εμφάνισης agents: κεντρικό μισό του χάρτη (από 1/4 έως 3/4 της 2ης διάστασης)
    dims = size(heightmap)                                            # Define the dims variable as the dimensions of the heightmap array (a tuple of rows and columns)  # διαστάσεις του χάρτη
    walkmap = BitArray(trues(dims...))                                 # Define the walkmap variable as a BitArray of true values with the dimensions of the heightmap array  # walkmap: αρχικά όλα τα cells είναι βατά (true)
    
    # Set walkmap to false in white areas of the heightmap (after reversal, white = high values)
    # White areas are where heightmap value is above threshold (e.g., > 245)
    # Black and grey areas (low to medium values) remain walkable
    # Τα cells με τιμή heightmap > white_threshold (λευκά σημεία μετά την αντιστροφή) γίνονται μη βατά
    white_threshold = 245  # κατώφλι τιμής πάνω από το οποίο το cell θεωρείται μη βατό
    walkmap[heightmap .> white_threshold] .= false  # σήμανση των μη βατών cells
end    


    #goals
    dests = [(500., 854.), (120., 248.)]  # συντεταγμένες (x, y) των δύο εξόδων διαφυγής

    #Generate the RNG for the model
    rng = MersenneTwister(seed)  # γεννήτρια τυχαίων αριθμών με σταθερό seed ώστε οι εκτελέσεις να είναι αναπαραγώγιμες

    ## Note that the dimensions of the space do not have to correspond to the dimensions
    ## of the pathfinder. Discretisation is handled by the pathfinding methods
    # ContinuousSpace requires extent to be exactly divisible by spacing in each dimension.
    # Use a compatible spacing (1 px, which always divides an integer extent) instead of METERS_TO_PIXELS (~0.2893 px).
    space = ContinuousSpace(size(heightmap); periodic = false, spacing = 1)  # συνεχής (μη περιοδικός) χώρος με διαστάσεις ίσες με του χάρτη


begin  # Ορισμός του pathfinder A* και των ιδιοτήτων του μοντέλου
    pathfinderPM = AStar(space; walkmap = walkmap, cost_metric = PenaltyMap(NPM, MaxDistance{2}()))  # A* pathfinder: δεν περνά από μη βατά cells (walkmap), κόστος PenaltyMap πάνω στον NPM, με βασική μετρική απόστασης MaxDistance{2}
    properties = (  # ιδιότητες προσβάσιμες μέσα στο μοντέλο ως model.<όνομα>
        pathfinderPM = pathfinderPM,  # ο A* pathfinder
        heightmap = heightmap,  # ο χάρτης ανάγλυφου/εμποδίων
        dt = dt,  # το χρονικό βήμα
        speed_range = speed_range,  # εύρος αρχικής ταχύτητας
        goal = dests  # οι έξοδοι (στόχοι) των agents
    )
end


# Helper: true if position is within radius of any goal (TL and movement are skipped when true)
# Ακτίνα (σε pixels) γύρω από κάθε έξοδο· εντός αυτής ο agent θεωρείται ότι διέφυγε
const goal_radius = 10.0
# Επιστρέφει true αν ο agent βρίσκεται σε απόσταση <= radius από κάποια έξοδο (τότε σταματούν η κίνηση και η συσσώρευση τοξικού φορτίου).
function at_goal(pos, dests, radius = goal_radius)
    return minimum(norm(pos .- d) for d in dests) ≤ radius  # απόσταση από την πλησιέστερη έξοδο <= ακτίνα
end


# Ένα βήμα του agent σε κάθε χρονική στιγμή: έλεγχος εξόδου, ενημέρωση τοξικού φορτίου, υπολογισμός ταχύτητας και κίνηση κατά μήκος της διαδρομής A*.
function agent_step!(person, model)
    if at_goal(person.pos, model.goal)  # ο agent έχει φτάσει σε έξοδο: καταγράφει τη θέση και σταματά
        # At goal: TL and movement stop; keep path in sync for visualization
        push!(person.pathX, person.pos[1])  # καταγραφή θέσης x στο trail
        push!(person.pathY, person.pos[2])  # καταγραφή θέσης y στο trail
        return  # δεν κινείται και δεν συσσωρεύει TL
    end

    grid_dims = size(tl_penalty_map)  # διαστάσεις του ενεργού χάρτη συγκέντρωσης
    i = clamp(Int(floor(person.pos[1])), 1, grid_dims[1])  # δείκτης γραμμής του cell στο οποίο βρίσκεται ο agent (εντός ορίων)
    j = clamp(Int(floor(person.pos[2])), 1, grid_dims[2])  # δείκτης στήλης του cell στο οποίο βρίσκεται ο agent (εντός ορίων)
    Ct = tl_penalty_map[i, j]  # συγκέντρωση τοξικού στο cell του agent από το τρέχον CM
    TLcurrent = [person.TL1[end], person.TL2[end], person.TL3[end]]  # τελευταίες τιμές TL των τριών συμπτωμάτων
    TL = update_toxic_load(Ct, TLcurrent, model.dt)  # νέο τοξικό φορτίο μετά από έκθεση διάρκειας dt

    person.toxicload = sum(TL)  # συνολικό τοξικό φορτίο = άθροισμα των τριών συμπτωμάτων
    push!(person.TL1, TL[1])  # αποθήκευση στο ιστορικό του 1ου συμπτώματος
    push!(person.TL2, TL[2])  # αποθήκευση στο ιστορικό του 2ου συμπτώματος
    push!(person.TL3, TL[3])  # αποθήκευση στο ιστορικό του 3ου συμπτώματος

    # --- Speed in m/s (base 1.35 m/s); pathfinder expects pixels/s so scale by METERS_TO_PIXELS ---
    # Ταχύτητα (m/s) ανάλογα με το συνολικό τοξικό φορτίο: μικρή έκθεση την αυξάνει, μεγαλύτερη τη μειώνει, και για toxicload >= 3 ο agent ακινητοποιείται
    speed = 1.35  # κανονική ταχύτητα (m/s) χωρίς επίδραση τοξικού
    if 0 < person.toxicload <= 1  # μικρή έκθεση (0 < TL <= 1)
        speed = 1.35 * exp(0.393 * person.toxicload)  # αύξηση ταχύτητας (πανικός)
    elseif 1 < person.toxicload < 3  # μεγαλύτερη έκθεση (1 < TL < 3)
        speed = -1.78 * log(person.toxicload) + 2.063  # η ταχύτητα μειώνεται όσο αυξάνεται το TL
    elseif person.toxicload >= 3  # TL >= 3
        speed = 0.0  # ο agent ακινητοποιείται
    end

    move_along_route!(person, model, model.pathfinderPM, speed * METERS_TO_PIXELS, model.dt)  # κίνηση του agent κατά μήκος της διαδρομής του pathfinder, με ταχύτητα μετατραμένη από m/s σε pixels/s
    push!(person.pathX, person.pos[1])  # καταγραφή της νέας θέσης x
    push!(person.pathY, person.pos[2])  # καταγραφή της νέας θέσης y
end



# Βήμα του μοντέλου: ελαστικές συγκρούσεις μεταξύ των πλησιέστερων ζευγών agents (απόσταση 0.012), με βάση τη μάζα τους.
function model_step!(model)
    for (a1, a2) in interacting_pairs(model, 0.012, :nearest)  # ζεύγη agents σε επαφή (πλησιέστερος γείτονας)
        elastic_collision!(a1, a2, :mass)  # ελαστική κρούση που λαμβάνει υπόψη τις μάζες
    end
end


model = StandardABM(  # δημιουργία του agent-based μοντέλου
  AgentEscapes,
  space;
  rng          = rng,  # η γεννήτρια τυχαίων αριθμών
  properties   = properties,  # ιδιότητες του μοντέλου (pathfinder, heightmap, dt, ...)
  agent_step!  = agent_step!,  # συνάρτηση βήματος ανά agent
  model_step!  = model_step!  # συνάρτηση βήματος του μοντέλου
)


begin  # Δημιουργία των agents και προϋπολογισμός των διαδρομών τους
    path_planning_time = Ref(0.0)  # συσσωρευτής του συνολικού χρόνου υπολογισμού διαδρομών
    for _ in 1:n_agents  # δημιουργία n_agents agents
    age = rand(abmrng(model))*(age_range[2]-age_range[1]) + age_range[1]  # τυχαία ηλικία μέσα στο age_range
    mass = rand(abmrng(model)) * (mass_range[2]-mass_range[1]) + mass_range[1]  # τυχαία μάζα μέσα στο mass_range
    vel = Tuple(rand(abmrng(model), 2) .* (speed_range[2]-speed_range[1]) .+ speed_range[1])  # τυχαία αρχική ταχύτητα vel μέσα στο speed_range
    
    # Keep trying to find a valid spawn position that is walkable (not in the non-walkable areas of the walkmap)
    max_attempts = 1000  # μέγιστος αριθμός προσπαθειών εύρεσης βατής θέσης
    attempts = 0  # μετρητής προσπαθειών
    pos = nothing  # η θέση εμφάνισης δεν έχει βρεθεί ακόμα
    while attempts < max_attempts  # επανάληψη μέχρι να βρεθεί βατή θέση
        candidate_pos = Tuple((rand(abmrng(model), floor.(ag_range_y)), rand(abmrng(model), floor.(ag_range_x))))  # τυχαία υποψήφια θέση μέσα στις περιοχές ag_range_y/ag_range_x
        pos_int = floor.(Int, candidate_pos)  # ακέραιες συντεταγμένες (cell) της υποψήφιας θέσης
        # Check if position is within bounds and walkable
        if 1 <= pos_int[1] <= size(walkmap, 1) && 1 <= pos_int[2] <= size(walkmap, 2) && walkmap[pos_int[1], pos_int[2]]  # η θέση είναι εντός ορίων και το cell είναι βατό
            pos = candidate_pos  # δεκτή θέση
            break  # έξοδος από την επανάληψη
        end
        attempts += 1  # αποτυχία: νέα προσπάθεια
    end
    
    # If we couldn't find a valid position after max_attempts, skip this agent
    if pos === nothing  # δεν βρέθηκε βατή θέση μετά από max_attempts προσπάθειες
        println("Warning: Could not find valid spawn position for agent after $max_attempts attempts")  # προειδοποίηση στον χρήστη
        continue  # παράλειψη αυτού του agent
    end
    
    person = add_agent!(pos, AgentEscapes, model, vel, age, mass, 0., [pos[1]], [pos[2]], [0.0], [0.0], [0.0])  # προσθήκη agent: αρχικό toxicload 0., ιστορικό θέσης με την αρχική θέση, ιστορικά TL1..TL3 με αρχική τιμή 0.0
    path_planning_time[] += @elapsed plan_best_route!(person, dests, model.pathfinderPM)  # υπολογισμός της καλύτερης διαδρομής προς μία από τις εξόδους και μέτρηση του χρόνου
end
    println("TIMING BREAKDOWN:")  # εκτύπωση χρόνου υπολογισμού διαδρομών
    println("  Pathfinding time (total): ", round(path_planning_time[]; digits=6), " s")  # συνολικός χρόνος pathfinding σε sec
end


# Υπολογίζει τους πίνακες Balpha (εκθέτες νόμου ισχύος), Btime (χρόνοι) και Brho (συγκεντρώσεις) από τις τιμές AEGL του H2S για τρία συμπτώματα (οσμή, ερεθισμός, οίδημα), με προέκταση στους ακραίους χρόνους taumin και taumax. Επιστρέφει (Balpha, Btime, Brho').
function setupToxic()                                               # Define the setupToxic function
    Atime = [0.0, 2.0, 5.0, 8.0, 15.0, 30.0] #min                # Initialization of the 5 standard AEGL exposure times  # χρόνοι έκθεσης AEGL σε λεπτά (το πρώτο στοιχείο 0 είναι placeholder)
    Arho = zeros(3, 6)                                              # the concentration of the three symptoms compared to the AEGL concentrations  # πίνακας συγκεντρώσεων AEGL: 3 συμπτώματα x 6 χρόνοι
    Arho[1, 2:6] = [2.87, 2.33, 2.09, 1.81, 1.55]                   # odor  # συγκεντρώσεις (ppm) για την οσμή
    Arho[2, 2:6] = [202.38, 164.38, 147.74, 128.10, 109.45]         # irritation  # συγκεντρώσεις (ppm) για τον ερεθισμό
    Arho[3, 2:6] = [286.71, 232.87, 209.30, 181.47, 155.05] #ppm    # edema  # συγκεντρώσεις (ppm) για το οίδημα
    MW = 34                                                         # Molecular weight of H2S in g/mol  # μοριακό βάρος του H2S (g/mol)
    Arho *= MW/24.04 #mg/m^3                                        # Multiply the Arho array by the molecular weight of H2S divided by 24.04  # μετατροπή από ppm σε mg/m^3
    Arho = Arho'                                                    # Transpose the Arho array  # ανάστροφος: γραμμές = χρόνοι, στήλες = συμπτώματα
    Atime = Atime*60 #seconds                                       # Multiply the Atime array by 60 seconds to convert to seconds  # μετατροπή των χρόνων από λεπτά σε sec
    taumin = 200.                                                   # shortest exposure time over which an AEGL 1, 2 or 3 onset can be reached  # ελάχιστος χρόνος έκθεσης (sec) για την εμφάνιση AEGL 1, 2 ή 3
    taumax = 86400.                                                 # longest exposure time over which an AEGL 1, 2 or 3 onset can be reached  # μέγιστος χρόνος έκθεσης (sec) για την εμφάνιση AEGL 1, 2 ή 3
    Brho = zeros(7,3)                                               # in mg/m^3 for each AEGL band at every time step ‘Atime’  # συγκεντρώσεις: 7 γραμμές (taumin, 5 χρόνοι AEGL, taumax) x 3 συμπτώματα
    Balpha = zeros(7,3)                                             # power-law exponents  # εκθέτες νόμου ισχύος (power-law)
    rhomax = zeros(1,3)                                             # μέγιστη συγκέντρωση ανά σύμπτωμα (προέκταση στο taumin)
    rhomin = zeros(1,3)                                             # ελάχιστη συγκέντρωση ανά σύμπτωμα (προέκταση στο taumax)
    Btime = zeros(7, 3)                                             # represents an array, which is function of ‘taumin’ and ‘taumax’, that changes depending on alpha, which is a corresponsing array of power low exponents interpolating the ‘Brho’ table array  # χρόνοι που αντιστοιχούν στις συγκεντρώσεις του Brho

    #Initialize
    # Αρχικοποίηση: αντιγραφή των τιμών AEGL στις γραμμές 2..6 και ορισμός των taumin/taumax στις γραμμές 1 και 7 του Btime
    for k=1:3
        for b=2:6
            Brho[b, k] = Arho[b, k]
            Btime[b, k] = Atime[b]
        end
        Btime[1, k] = taumin;  # ο χρόνος της 1ης γραμμής είναι ο taumin
         Btime[7, k] = taumax;  # ο χρόνος της 7ης γραμμής είναι ο taumax
    end    
    
    #power low trial values
    # Υπολογισμός εκθετών νόμου ισχύος μεταξύ διαδοχικών σημείων AEGL
    for k=1:3
        for b=3:6
            if Brho[b-1, k]==Brho[b, k]  # ίδιες διαδοχικές συγκεντρώσεις: ο εκθέτης είναι 0
                Balpha[b, k] = 0.0;
            else
                Balpha[b, k] = log(Atime[b]/Atime[b-1])/log(Brho[b-1, k]/Brho[b, k]);  # εκθέτης νόμου ισχύος από δύο διαδοχικά σημεία AEGL
            end
        end

        Balpha[2, k] = Balpha[3, k]  # στα άκρα αντιγράφεται ο εκθέτης του πλησιέστερου υπολογισμένου σημείου
        Balpha[1, k] = Balpha[2, k]  # ομοίως για την 1η γραμμή
        Balpha[7, k] = Balpha[6, k]  # ομοίως για την 7η γραμμή
        #Balpha[6, k] = Balpha[5, k]
    end
    #extrapolate for the edges
    # Προέκταση των συγκεντρώσεων στα άκρα (taumin και taumax)
    for k=1:3
        if Balpha[3, k]==0  # μηδενικός εκθέτης: η μέγιστη συγκέντρωση ισούται με αυτήν της 2ης γραμμής
            rhomax[k] = Brho[2, k];
            Brho[1, k] = rhomax[k];
        else
            rhomax[k] = Brho[2, k]*(Btime[2, k]/taumin)^(1/Balpha[2, k]);  # προέκταση της συγκέντρωσης στον ελάχιστο χρόνο taumin
            Brho[1, k] = rhomax[k];
        end

        if Brho[5, k]==Brho[6, k]  # ίδιες συγκεντρώσεις στο τέλος: η ελάχιστη συγκέντρωση ισούται με αυτήν της 6ης γραμμής
            rhomin[k] = Brho[6, k];
            Brho[7, k] = rhomin[k];
        else
            rhomin[k] = Brho[6, k]*(Btime[6, k]/taumax)^(1/Balpha[6, k]); # προέκταση της συγκέντρωσης στον μέγιστο χρόνο taumax
            Brho[7, k] = rhomin[k];
        end

    end
    #Correct to account for threshold values
    # Διόρθωση των χρόνων Btime όπου ο εκθέτης είναι 0 (σταθερή συγκέντρωση)
    for k=1:3
        for b=2:7
            if Balpha[b, k] == 0
                Btime[b, k] = Btime[b-1, k];  # όπου ο εκθέτης είναι 0, ο χρόνος παίρνει την τιμή της προηγούμενης γραμμής
            end
        end
    end

    # Επανυπολογισμός εκθέτη όπου ο προηγούμενος ήταν 0 και ο τρέχων > 0
    for k=1:3
        for b=3:5
            if Balpha[b-1, k]==0 && Balpha[b, k]>0  # ο προηγούμενος εκθέτης ήταν 0 και ο τρέχων είναι θετικός
                Balpha[b, k]=log(Btime[b, k]/Btime[b-1, k])/log(Brho[b-1, k]/Brho[b, k]);  # επανυπολογισμός του εκθέτη με τους διορθωμένους χρόνους
            end
        end
    end

    return Balpha, Btime, Brho'  # επιστρέφει εκθέτες, χρόνους και συγκεντρώσεις (το Brho ανάστροφο: 3x7)
end

Balpha, Btime, Brho = setupToxic()  # καθολικοί πίνακες που χρησιμοποιεί η tl_rate (και μέσω αυτής η update_toxic_load)



# Ρυθμός συσσώρευσης τοξικού φορτίου dTL/dt (s^-1) για πυκνότητα rho (mg/m^3) και επίπεδο AEGL k (1, 2, 3).
# Πιστή μεταφορά της συνάρτησης TL_RATE του Boris & Patnaik (2014), NRL/MR/6003--14-9493, Παράρτημα A:
#   dTL/dt = (rho / Brho(b,k))^alpha(b,k) / Btime(b,k),  όπου b είναι η ζώνη πυκνότητας στην οποία ανήκει το rho.
# Αντιστοίχιση δεικτών: ο δείκτης b του paper (0..6) είναι b+1 εδώ, δηλαδή Brho(b,k) = Brho[k, b+1],
# Btime(b,k) = Btime[b+1, k], alpha(b,k) = Balpha[b+1, k]. Άρα Brho[k, 1] = rhomax και Brho[k, 7] = rhomin.
# Σημείωση: το paper επιστρέφει 1.0E-6 s^-1 ως "καμία συσσώρευση" (αριθμητική προστασία για το 1/TL_RATE),
# ενώ στο κείμενο ορίζει "hard zero" για πυκνότητες κάτω από rhomin· εδώ χρησιμοποιείται 0.0.
function tl_rate(rho, k)
    # rho κάτω από τη χαμηλότερη τιμή του πίνακα AEGL, Brho(5,k)
    if rho < Brho[k, 6]
        Balpha[7, k] == 0.0 && return 0.0                       # alpha(6,k) == 0: threshold, καμία συσσώρευση
        rho >= Brho[k, 7] || return 0.0                         # rho < rhomin: καμία συσσώρευση (hard zero)
        return (rho / Brho[k, 6])^Balpha[6, k] / Btime[6, k]    # επέκταση του power law μέχρι τον taumax
    end

    # rho >= rhomax: μέγιστος ρυθμός συσσώρευσης 1/taumin
    rho >= Brho[k, 1] && return 1.0 / Btime[1, k]

    # επιλογή της ζώνης b (κάτω όριο της περιοχής πυκνότητας)
    if rho < Brho[k, 5] && Brho[k, 5] != Brho[k, 6]
        b = 5
    elseif rho < Brho[k, 4] && Brho[k, 4] != Brho[k, 5]
        b = 4
    elseif rho < Brho[k, 3] && Brho[k, 3] != Brho[k, 4]
        b = 3
    elseif rho < Brho[k, 2]
        b = 2
    else
        b = 1                                                   # Brho(1,k) <= rho < rhomax
    end
    return (rho / Brho[k, b+1])^Balpha[b+1, k] / Btime[b+1, k]
end


# Ενημερώνει το τοξικό φορτίο TL των τριών επιπέδων AEGL για έκθεση σε πυκνότητα Ct επί χρόνο dt.
# TL_k >= 1 σημαίνει ότι έχει επιτευχθεί η έναρξη των συμπτωμάτων του AEGL k.
function update_toxic_load(Ct, TLcurrent, dt)
    TL = TLcurrent
    for k = 1:3
        TL[k] = TL[k] + tl_rate(Ct, k) * dt     # ολοκλήρωση: TL_k += (dTL/dt)_k * dt
    end
    return TL
end



# Επιστρέφει το χρώμα του agent ανάλογα με το toxicload (πράσινο: <= 1, πορτοκαλί: ενδιάμεσα, κόκκινο: >= 3).
function personcolor(person::AgentEscapes)  # Χρώμα του agent ανάλογα με το toxicload
    if person.toxicload >= 3.  # ακινητοποιημένος agent
        return :red  # κόκκινο
    elseif person.toxicload <= 1.  # μικρή έκθεση
        return :green  # πράσινο
    else
        return :orange  # πορτοκαλί (ενδιάμεση έκθεση)
    end
end


begin   # Δημιουργία animation με trails & συλλογή CSV θέσης και toxicload
    T = 2500  # συνολικός αριθμός frames/βημάτων (dt = 1 s)
    # CM switch times (s) — Thesis, Table 7: 7 CMs, switching completes at 70% of T = 2500 s.
    cm_switch_times = [0.0, 292.0, 584.0, 876.0, 1168.0, 1460.0, 1752.0]  # χρόνοι (s) από τους οποίους ισχύει κάθε CM (ένας ανά CM)
    @assert length(cm_switch_times) == NUM_CMS  # έλεγχος ότι υπάρχει χρόνος αλλαγής για κάθε CM
    tl_map_idx = Ref(1)  # δείκτης του CM που ισχύει αυτή τη στιγμή (ξεκινά από το 1)
    tl_penalty_map .= cm_list[tl_map_idx[]]  # φόρτωση του CM1 στον ενεργό χάρτη TL (επί τόπου)

    # -- Στήσιμο Figure & Axis --
    fig = Figure(; size = (800,800))  # δημιουργία του Figure
    ax  = Makie.Axis(fig[1,1];  # άξονας με σταθερή αναλογία πλευρών
               title  = "Evacuation with Toxic Trail",
               aspect = DataAspect())

    heatmap!(ax, heightmap; colormap=:grays, alpha=0.3)  # το heightmap ως ημιδιαφανές φόντο σε αποχρώσεις του γκρι
    
    # --- Concentration Map visualization (contour only, overlay on heightmap) ---
    # Οπτικοποίηση του concentration map: contour πάνω στο heightmap, που ενημερώνεται όταν αλλάζει το CM
    cm_obs = Observable(copy(tl_penalty_map))  # Observable με το τρέχον CM
    # Εύρος χρωμάτων από min/max του τρέχοντος CM· αν min == max, προστίθεται μικρή απόκλιση ώστε το εύρος να μην είναι μηδενικό
    cm_colorrange = @lift let z = $cm_obs
        m = minimum(z)  # ελάχιστη τιμή του CM
        M = maximum(z)  # μέγιστη τιμή του CM
        if m == M  # ομοιόμορφος χάρτης: μηδενικό εύρος
            δ = max(1.0, abs(m) * 1e-6)  # μικρή απόκλιση
            (Float32(m - δ), Float32(M + δ))  # εύρος με απόκλιση προς τις δύο πλευρές
        else
            (Float32(m), Float32(M))  # κανονικό εύρος min..max
        end
    end
    contour!(  # περιγράμματα (contours) του τρέχοντος CM
        ax, cm_obs;
        colormap = cgrad([:yellow, :orange, :red]),  # κλίμακα χρωμάτων κίτρινο, πορτοκαλί, κόκκινο
        colorrange = cm_colorrange,  # δυναμικό εύρος χρωμάτων
        levels = 10,  # πλήθος επιπέδων περιγραμμάτων
        linewidth = 1.5,  # πάχος γραμμής
        alpha = 0.7,  # διαφάνεια
    )
    
    goals = model.goal  # οι έξοδοι από τις ιδιότητες του μοντέλου
    scatter!(ax,  # σχεδίαση των εξόδων
        getindex.(goals,1),
        getindex.(goals,2);
        color  = (:red,50),
        marker = :circle,
    )

    # --- Overlay time counter ---
    frame_obs = Observable(0)  # μετρητής frame για την ετικέτα χρόνου
    
    counter_lbl = Label(  # ετικέτα που δείχνει τον χρόνο που πέρασε
    fig,
    @lift("Time elapsed = $((($frame_obs-1)*dt)) s"),  # κείμενο της ετικέτας: (frame - 1) * dt σε sec
    fontsize = 16,
    padding = (6, 10, 6, 10),
    halign = :left
)
    fig[1,1, TopLeft()] = counter_lbl  # αγκίστρωση πάνω-αριστερά στο ίδιο κελί με τον άξονα

    # -- Observables για θέση & χρώμα --

    xs0 = Float64[]  # Initialize empty arrays for positions  # αρχικές x θέσεις των agents
    ys0 = Float64[]  # αρχικές y θέσεις των agents
    colors0 = Symbol[]  # Initialize empty array for colors  # αρχικά χρώματα των agents

    for a in allagents(model)  # συλλογή αρχικών θέσεων και χρωμάτων
        push!(xs0, a.pos[1])
        push!(ys0, a.pos[2])
        push!(colors0, personcolor(a))
    end

    posobs = Observable(Point2f.(xs0, ys0))  # θέσεις των agents ως Observable ώστε να ενημερώνεται το γράφημα
    colobs = Observable(colors0)  # χρώματα των agents ως Observable

    # Ένα γράφημα γραμμής (trail) ανά agent, με αρχικό σημείο τη θέση του
    lines_plots = [
        lines!(ax,
               [a.pos[1]], [a.pos[2]];
               color     = personcolor(a),
               linewidth = 2)
        for a in allagents(model)
    ]
    agent_scat = scatter!(ax, posobs;  # σημεία που παριστάνουν τους agents
                          color      = colobs,
                          markersize = 10)

    # -- Προετοιμασία DataFrame για θέση & toxicload ανά βήμα --
    df = DataFrame(  # πίνακας αποτελεσμάτων: μία γραμμή ανά agent και ανά βήμα
        step       = Int[],
        agent_id   = Int[],
        x          = Float64[],
        y          = Float64[],
        toxicload  = Float64[]
    )

    # -- Έναρξη record: video και συλλογή δεδομένων ταυτόχρονα --
    video_file = "SCENARIOS/SCENARIO 2/Simulation Results/SCENARIO_2_$(n_agents)_$(seed)_$(run_timestamp).mp4"  # διαδρομή του αρχείου βίντεο εξόδου
    record(fig, video_file, 1:T; framerate=30) do frame  # καταγραφή του animation: ένα frame ανά βήμα προσομοίωσης, 30 fps
        # 1) ενημέρωση του frame counter και του ενεργού CM
        frame_obs[] = frame  # ενημέρωση του μετρητή frame
        t_sim = (frame - 1) * dt  # χρόνος προσομοίωσης (s) του τρέχοντος frame
        new_idx = searchsortedlast(cm_switch_times, t_sim)  # δείκτης του τελευταίου CM με χρόνο έναρξης <= t_sim
        if new_idx != tl_map_idx[]  # αλλαγή CM μόνο όταν αλλάζει ο δείκτης
            tl_map_idx[] = new_idx  # αποθήκευση του νέου δείκτη
            tl_penalty_map .= cm_list[new_idx]  # αντικατάσταση του ενεργού χάρτη TL (τον διαβάζει η agent_step!)· ο χάρτης του pathfinder (CM4) δεν αλλάζει
        end
        cm_obs[] = tl_penalty_map  # ανανέωση του contour overlay
        # 2) βήμα προσομοίωσης
        step!(model, 1)  # ένα βήμα του μοντέλου (agent_step! για κάθε agent και model_step!)

        # 3) ενημέρωση των trails
        for (i,a) in enumerate(allagents(model))  # για κάθε agent i
            lines_plots[i][1][] = Point2f.(a.pathX, a.pathY)  # ενημέρωση του trail με το ιστορικό θέσεων του agent
        end

        # 4) ενημέρωση θέσεων & δυναμικού χρώματος
        xs = [a.pos[1] for a in allagents(model)]  # τρέχουσες x θέσεις
        ys = [a.pos[2] for a in allagents(model)]  # τρέχουσες y θέσεις
        posobs[] = Point2f.(xs, ys)  # ενημέρωση των θέσεων στο γράφημα
        colobs[] = [personcolor(a) for a in allagents(model)]  # ενημέρωση των χρωμάτων στο γράφημα

        # 5) συλλογή δεδομένων στο DataFrame
        for a in allagents(model)  # προσθήκη δεδομένων κάθε agent στο DataFrame
            push!(df, (  # προσθήκη γραμμής στο DataFrame: frame, id, x, y, toxicload
                frame,
                a.id,
                a.pos[1],
                a.pos[2],
                a.toxicload
            ))
        end
    end

    println("Το animation σώθηκε ως $video_file")  # μήνυμα ότι το animation αποθηκεύτηκε

    # -- Εξαγωγή CSV με θέση & toxicload των agents --
    csv_file = "SCENARIOS/SCENARIO 2/Simulation Results/SCENARIO_2_$(n_agents)_$(seed)_$(run_timestamp).csv"  # διαδρομή του αρχείου CSV εξόδου
    CSV.write(csv_file, df)  # εγγραφή του DataFrame σε αρχείο CSV
    println("Τα δεδομένα θέσης & toxicload αποθηκεύτηκαν ως $csv_file")  # μήνυμα ότι τα δεδομένα αποθηκεύτηκαν

    # Καταγραφή της εκτέλεσης στο RunHistory
    append_run_history(
        scenario = "SCENARIO 2",  # όνομα σεναρίου
        runner_script = "SCENARIOS/SCENARIO 2/Scenario 2.jl",  # το script που εκτελέστηκε
        csv_file = csv_file,  # αρχείο CSV αποτελεσμάτων
        mp4_file = video_file,  # αρχείο βίντεο
        pathfinding_time = path_planning_time[],  # συνολικός χρόνος υπολογισμού διαδρομών (s)
        run_timestamp = run_timestamp,  # timestamp της εκτέλεσης
    )
end