begin   # Φόρτωση των απαραίτητων βιβλιοθηκών
    using Agents  # πλαίσιο agent-based μοντελοποίησης (ABM, add_agent!, step!, ...)
    using Agents.Pathfinding  # αλγόριθμοι εύρεσης διαδρομής (AStar, AbsolutePenaltyMap, MaxDistance, ...)
    using CSV  # εγγραφή/ανάγνωση αρχείων CSV
    using CairoMakie  # backend σχεδίασης (εικόνες/βίντεο)
    using DataFrames  # πίνακες δεδομένων (DataFrame) για τα αποτελέσματα
    using Dates  # ημερομηνία/ώρα για το timestamp των αρχείων εξόδου
    using FileIO: load  # φόρτωση εικόνων (heightmap, concentration maps)
    using ImageMagick  # υποστήριξη μορφών εικόνας (jpg, bmp)
    using Images  # επεξεργασία εικόνων (channelview, ...)
    using InteractiveDynamics  # εργαλεία οπτικοποίησης για Agents.jl
    using Makie  # βιβλιοθήκη γραφικών
    using Observables  # reactive μεταβλητές (Observable) για το animation
    using Random  # γεννήτριες τυχαίων αριθμών (MersenneTwister)
    using StaticArrays  # φορτώνεται, αλλά δεν χρησιμοποιείται άμεσα σε αυτό το αρχείο

    # φορτώνει τη συνάρτηση append_run_history που καταγράφει κάθε εκτέλεση στο RunHistory
    include(joinpath(@__DIR__, "..", "append_run_history.jl"))
end


# Ορισμός του agent: συνεχής 2D agent (pos, vel προστίθενται αυτόματα από το ContinuousAgent)
Agents.@agent struct AgentEscapes(ContinuousAgent{2, Float64})
    age::Float64  # ηλικία (έτη)
    mass::Float64  # μάζα (kg), χρησιμοποιείται στις ελαστικές συγκρούσεις
    toxicload::Float64  # συνολικό τρέχον τοξικό φορτίο (άθροισμα TL1+TL2+TL3)
    pathX::Vector{Float64}  # ιστορικό της συντεταγμένης x του agent (χρησιμοποιείται για το trail στο animation)
    pathY::Vector{Float64}  # ιστορικό της συντεταγμένης y του agent (χρησιμοποιείται για το trail στο animation)
    TL1::Vector{Float64}  # ιστορικό τοξικού φορτίου για το 1ο σύμπτωμα (οσμή)
    TL2::Vector{Float64}  # ιστορικό τοξικού φορτίου για το 2ο σύμπτωμα (ερεθισμός)
    TL3::Vector{Float64}  # ιστορικό τοξικού φορτίου για το 3ο σύμπτωμα (οίδημα)
end


begin   # Φόρτωση του heightmap και των hand-drawn penalty maps (και όλων των CM1..CM7)
    heightmap_data = load("NADEEN/Maps/Qatargas Map.jpg")  # φόρτωση της εικόνας του χάρτη
    heightmap_data = permutedims(channelview(heightmap_data), [2,3,1])[:,:,1]  # μετατροπή σε πίνακα (γραμμές, στήλες) και κράτημα του 1ου καναλιού
    global heightmap = floor.(Int, convert.(Float64, heightmap_data) * 255)  # κλιμάκωση σε ακέραιες τιμές 0..255
    heightmap = 255 .- heightmap   # αυτό κάνει την αντιστροφή

    # --- Load all concentration maps 1..7 as Float64 arrays ---
    const NUM_CMS = 7  # πλήθος concentration maps (CM1..CM7)
    cm_list = Vector{Array{Float64,2}}(undef, NUM_CMS)  # λίστα που θα κρατά τους 7 χάρτες συγκέντρωσης
    for k in 1:NUM_CMS
        fn = joinpath("Concentration Maps", string(k) * ".bmp")  # διαδρομή αρχείου k.bmp
        img = load(fn)  # φόρτωση εικόνας
        img = permutedims(channelview(img), [2,3,1])[:,:,1]  # ίδια μετατροπή σε πίνακα όπως στο heightmap
        cm_list[k] = convert.(Float64, img) .* 255.0  # κλιμάκωση τιμών στο 0..255 (Float64)
    end

    # έλεγχος ότι κάθε CM έχει τις ίδιες διαστάσεις με το heightmap (αλλιώς σταματά με μήνυμα σφάλματος)
    # basic check: dimensions match heightmap
    for k in 1:NUM_CMS
        @assert size(cm_list[k]) == size(heightmap) "Concentration map $k size mismatch with heightmap"
    end

    # το penalty_map (τρέχον CM) ξεκινά ως αντίγραφο του 1ου χάρτη
    # start with first CM
    global penalty_map = copy(cm_list[1])
end

# συνδυασμένος χάρτης κόστους = ανάγλυφο/εμπόδια (heightmap) + τοξική συγκέντρωση (penalty_map)
NPM = heightmap .+ penalty_map
NPM_int = round.(Int, NPM)   # convert to Int for PenaltyMap


begin   # Αρχικοποίηση των παραμέτρων του μοντέλου
    const METERS_TO_PIXELS = 723.37 / 2500.0  # συντελεστής μετατροπής μέτρων σε pixels του χάρτη
    # dt: διάρκεια του χρονικού βήματος· seed: σπόρος του RNG· n_agents: πλήθος agents (εδώ 10)
    dt = 1.   ## discrete timestep each iteration of the model          # Define the dt variable as 1
    seed = 1544656427  ## seed for random number generator                     # Define the seed variable as 1544656427
    n_agents = 10                                                        # Define the n_agents variable as 10
    # age_range / mass_range: εύρος τυχαίας ηλικίας (έτη) και μάζας (kg)· speed_range: εύρος της τυχαίας αρχικής ταχύτητας vel
    age_range = (22,60)                                                 # Define the age_range variable as a tuple of 22 and 60
    speed_range = (4.0,7.0)                                            # Define the speed_range variable as a tuple of 4.0 and 7.0
    mass_range = (50,80)                                                # Define the mass_range variable as a tuple of 50 and 80
    # περιοχή εμφάνισης των agents: το κεντρικό μισό του χάρτη σε κάθε διάσταση
    ag_range_y = (size(heightmap)[1]/4):(3*size(heightmap)[1]/4)    # εύρος συντεταγμένων από 1/4 έως 3/4 της 1ης διάστασης του heightmap (size(heightmap)[1])
    ag_range_x = (size(heightmap)[2]/4):(3*size(heightmap)[2]/4)    # εύρος συντεταγμένων από 1/4 έως 3/4 της 2ης διάστασης του heightmap (size(heightmap)[2])
    # διαστάσεις του χάρτη κόστους και αρχικά πλήρως βατός χάρτης βατότητας (walkmap)
    dims = (size(NPM))                                            # διαστάσεις (γραμμές, στήλες) του πίνακα NPM, ίδιες με του heightmap
    walkmap = BitArray(trues(dims...))                                 # Define the walkmap variable as a BitArray of true values with the dimensions of the heightmap array
    
    # Τα λευκά σημεία του heightmap (τιμή πάνω από το κατώφλι) δεν είναι βατά· τα υπόλοιπα (μαύρα/γκρι) μένουν βατά
    # Set walkmap to false in white areas of the heightmap (after reversal, white = high values)
    # White areas are where heightmap value is above threshold (e.g., > 245)
    # Black and grey areas (low to medium values) remain walkable
    white_threshold = 245  # κατώφλι για τα «λευκά» (μη βατά) σημεία
    walkmap[heightmap .> white_threshold] .= false  # μηδενίζει τη βατότητα όπου το heightmap ξεπερνά το κατώφλι
end    

# Και οι 16 υποψήφιες έξοδοι (x, y) με σταθερή σειρά, δείκτες 1..16
# All 16 candidate goals (fixed order; indices 1:16). Sides are subsets of these indices.
const DESTS_ALL = Tuple{Float64, Float64}[
    (500., 854.), (120., 248.), (691., 750.), (351., 924.),
    (853., 659.), (795., 375.), (703., 187.), (864., 503.),
    (652., 73.), (325., 150.), (223., 198.), (473., 78.),
    (88., 493.), (145., 630.), (209., 787.), (39., 379.),
]

# Ομαδοποίηση των εξόδων σε 4 πλευρές (A-D): κάθε λίστα περιέχει καθολικούς δείκτες στο DESTS_ALL
# Side A–D: global indices into DESTS_ALL (max 2 active points per side per run)
const DEST_INDICES_BY_SIDE = Vector{Vector{Int}}([
    [1, 3, 4, 5],
    [6, 7, 8, 9],
    [10, 11, 12, 2],
    [13, 14, 15, 16],
])

# Επιστρέφει όλες τις επιλογές 0, 1 ή 2 διαφορετικών δεικτών από μία πλευρά (το πολύ 2 ανά πλευρά)
"""All ways to pick 0, 1, or 2 distinct indices from one side (≤2 per side)."""
function side_goal_options(side_indices::Vector{Int})
    opts = Vector{Vector{Int}}()  # λίστα επιλογών (κάθε επιλογή είναι λίστα δεικτών)
    push!(opts, Int[])  # επιλογή χωρίς καμία έξοδο της πλευράς
    for i in side_indices  # επιλογές με ακριβώς μία έξοδο
        push!(opts, [i])  # μία έξοδος κάθε φορά
    end
    n = length(side_indices)  # πλήθος εξόδων της πλευράς
    for i in 1:(n - 1), j in (i + 1):n  # όλα τα ζεύγη (i < j) διαφορετικών εξόδων της πλευράς
        push!(opts, [side_indices[i], side_indices[j]])  # επιλογή με δύο εξόδους
    end
    return opts  # επιστρέφει όλες τις επιλογές
end

# Παράγει όλα τα έγκυρα σύνολα καθολικών δεικτών εξόδων: 1..8 έξοδοι συνολικά, το πολύ 2 ανά πλευρά, ταξινομημένα κατά (K, λεξικογραφικά)
"""
Enumerate every valid global index set: total goals in 1..8, at most 2 per side.
Sorted by (K, indices lexicographically). Index in this list is stable for comparisons.
"""
function enumerate_valid_goal_combinations(dest_indices_by_side::Vector{Vector{Int}})
    opts = [side_goal_options(s) for s in dest_indices_by_side]  # επιλογές κάθε πλευράς ξεχωριστά
    seen = Set{Vector{Int}}()  # σύνολο που θυμάται ποιους συνδυασμούς έχουμε ήδη δει
    out = Vector{Vector{Int}}()  # λίστα εξόδου με τους έγκυρους συνδυασμούς
    for oA in opts[1], oB in opts[2], oC in opts[3], oD in opts[4]  # καρτεσιανό γινόμενο των επιλογών των 4 πλευρών
        comb = sort!(Int[oA; oB; oC; oD])  # ένωση των δεικτών των 4 πλευρών, ταξινομημένη
        k = length(comb)  # συνολικό πλήθος εξόδων K του συνδυασμού
        if 1 <= k <= 8 && comb ∉ seen  # δεκτός μόνο αν 1 <= K <= 8 και δεν έχει ξαναεμφανιστεί
            push!(seen, comb)  # σημειώνεται ως ήδη δεδομένος
            push!(out, comb)  # προστίθεται στη λίστα εξόδου
        end
    end
    sort!(out; by = x -> (length(x), x))  # ταξινόμηση πρώτα κατά πλήθος K και μετά λεξικογραφικά, ώστε ο δείκτης κάθε συνδυασμού να είναι σταθερός
    return out  # επιστρέφει τη λίστα των έγκυρων συνδυασμών
end

# Λίστα όλων των έγκυρων συνδυασμών εξόδων (υπολογίζεται μία φορά)
const GOAL_COMBINATIONS = begin
    u = enumerate_valid_goal_combinations(DEST_INDICES_BY_SIDE)
    println("Scenario 4: enumerated $(length(u)) valid goal combinations (K in 1:8, max 2 per side).")  # ενημερωτικό μήνυμα με το πλήθος των συνδυασμών
    u
end

# Επιλογή του συνδυασμού που χρησιμοποιεί αυτή η προσομοίωση (δείκτης στη λίστα GOAL_COMBINATIONS)
# Pick which combination this simulation uses (1:length(GOAL_COMBINATIONS)); edit to compare runs.
# Index 1794 = global goal indices [4, 7, 11, 15] (one per side A–D).
const SCENARIO4_GOAL_COMBO_INDEX = 1794  # δείκτης του επιλεγμένου συνδυασμού εξόδων

@assert 1 <= SCENARIO4_GOAL_COMBO_INDEX <= length(GOAL_COMBINATIONS) "SCENARIO4_GOAL_COMBO_INDEX must be in 1:$(length(GOAL_COMBINATIONS))"  # έλεγχος ότι ο δείκτης είναι εντός ορίων της λίστας συνδυασμών

active_dest_indices = copy(GOAL_COMBINATIONS[SCENARIO4_GOAL_COMBO_INDEX])  # οι καθολικοί δείκτες των ενεργών εξόδων
n_active_dests = length(active_dest_indices)  # πλήθος ενεργών εξόδων K
dests = DESTS_ALL[active_dest_indices]  # οι συντεταγμένες (x, y) των ενεργών εξόδων
println("Scenario 4 active goals [combination $SCENARIO4_GOAL_COMBO_INDEX / $(length(GOAL_COMBINATIONS))]: K=", n_active_dests, " | global indices ", active_dest_indices)  # εκτύπωση του επιλεγμένου συνδυασμού και του πλήθους ενεργών εξόδων
println("  coordinates: ", dests)  # εκτύπωση των συντεταγμένων των ενεργών εξόδων

# γεννήτρια τυχαίων αριθμών με σταθερό seed ώστε οι εκτελέσεις να είναι αναπαραγώγιμες
rng = MersenneTwister(seed)

## Note that the dimensions of the space do not have to correspond to the dimensions
## of the pathfinder. Discretisation is handled by the pathfinding methods
# συνεχής (μη περιοδικός) χώρος με διαστάσεις ίσες με του χάρτη
space = Pathfinding.ContinuousSpace(size(NPM); periodic = false, spacing = 1)


# Ορισμός του pathfinder A* και των ιδιοτήτων του μοντέλου
begin
    cost_metric_obj = AbsolutePenaltyMap(NPM_int, MaxDistance{2}())  # κόστος κίνησης ανά cell από τον συνδυασμένο χάρτη, με γειτονιά 8 κατευθύνσεων (MaxDistance{2})
    cost_metric_str = "APM"  # σύντμηση για τα ονόματα των αρχείων εξόδου
    heuristic_code  = "DF"   # parity with DStarLite export names (A* uses default heuristic)
    pathfinderPM = AStar(space; walkmap = walkmap, cost_metric = cost_metric_obj)  # A* pathfinder: οι agents δεν περνούν από μη βατά cells (walkmap) και ακολουθούν το cost_metric
    # ιδιότητες προσβάσιμες μέσα στο μοντέλο ως model.<όνομα>
    properties = (
        pathfinderPM = pathfinderPM,
        heightmap = heightmap,
        dt = dt,
        speed_range = speed_range,
        goal = dests,
    )  # goal: οι ενεργές έξοδοι του επιλεγμένου συνδυασμού
end


# Helper: true if position is within radius of any goal (TL and movement stop when true)
const goal_radius = 10.0  # ακτίνα (σε pixels) γύρω από την έξοδο που θεωρείται ότι ο agent «διέφυγε»
# Ελέγχει αν η θέση pos απέχει το πολύ radius από κάποια από τις εξόδους dests (true = ο agent θεωρείται ότι διέφυγε)
function at_goal(pos, dests, radius = goal_radius)
    px, py = Float64(pos[1]), Float64(pos[2])  # θέση του agent ως Float64
    return minimum(hypot(px - d[1], py - d[2]) for d in dests) <= radius  # απόσταση από την πλησιέστερη έξοδο <= ακτίνα
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
    # Ct παίρνεται από το global penalty_map (hand-drawn maps)
    Ct = penalty_map[i, j]  # συγκέντρωση τοξικού στο cell του agent
    TLcurrent = [person.TL1[end], person.TL2[end], person.TL3[end]]  # τελευταίες τιμές TL των τριών συμπτωμάτων
    TL = update_toxic_load(Ct, TLcurrent, model.dt)  # νέο τοξικό φορτίο μετά από έκθεση dt

    person.toxicload = sum(TL)  # συνολικό τοξικό φορτίο
    push!(person.TL1, TL[1])  # αποθήκευση στο ιστορικό
    push!(person.TL2, TL[2])
    push!(person.TL3, TL[3])

    # --- Speed update based on toxicload ---
    speed = 1.35  # κανονική ταχύτητα (m/s) χωρίς επίδραση τοξικού
    if 0 < person.toxicload <= 1
        speed = 1.35 * exp(0.393 * person.toxicload)  # μικρή έκθεση: αύξηση ταχύτητας
    elseif 1 < person.toxicload < 3
        speed = -1.78 * log(person.toxicload) + 2.063  # μεγαλύτερη έκθεση: η ταχύτητα μειώνεται
    elseif person.toxicload >= 3
        speed = 0.0  # θανατηφόρο φορτίο: ο agent ακινητοποιείται
    end

    # κίνηση κατά μήκος της διαδρομής που έχει υπολογίσει ο A* για τον agent, με ταχύτητα μετατραμένη σε pixels
    move_along_route!(person, model, model.pathfinderPM, speed * METERS_TO_PIXELS, model.dt)
    push!(person.pathX, person.pos[1])  # καταγραφή νέας θέσης για το trail
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

# Δημιουργία των agents σε τυχαίες βατές θέσεις
# Add agents (initial route planning is timed inside the @time block with the animation, like DStarLite)
for _ in 1:n_agents
    age = rand(abmrng(model))*(age_range[2]-age_range[1]) + age_range[1]  # τυχαία ηλικία εντός age_range
    mass = rand(abmrng(model)) * (mass_range[2]-mass_range[1]) + mass_range[1]  # τυχαία μάζα εντός mass_range
    vel = Tuple(rand(abmrng(model), 2) .* (speed_range[2]-speed_range[1]) .+ speed_range[1])  # τυχαία αρχική ταχύτητα (vx, vy)
    
    # Keep trying to find a valid spawn position that is walkable (walkmap == true, δηλαδή όχι στα μη βατά «λευκά» σημεία)
    max_attempts = 1000  # μέγιστος αριθμός προσπαθειών εύρεσης βατής θέσης
    attempts = 0
    pos = nothing
    while attempts < max_attempts
        # τυχαία υποψήφια θέση μέσα στην κεντρική περιοχή του χάρτη (ag_range_x/ag_range_y)
        candidate_pos = Tuple((rand(abmrng(model), floor.(ag_range_y)), rand(abmrng(model), floor.(ag_range_x))))
        pos_int = floor.(Int, candidate_pos)  # αντίστοιχο cell
        # Check if position is within bounds and walkable
        # αποδοχή της θέσης μόνο αν είναι εντός ορίων και βατή· τότε σταματά η αναζήτηση
        if 1 <= pos_int[1] <= size(walkmap, 1) && 1 <= pos_int[2] <= size(walkmap, 2) && walkmap[pos_int[1], pos_int[2]]
            pos = candidate_pos
            break
        end
        attempts += 1  # αλλιώς μετράει την αποτυχημένη προσπάθεια και ξαναδοκιμάζει
    end
    
    # If we couldn't find a valid position after max_attempts, skip this agent
    # αν δεν βρέθηκε βατή θέση, εμφανίζει προειδοποίηση και παραλείπει τον agent
    if pos === nothing
        println("Warning: Could not find valid walkable spawn position for agent in Scenario 4 after $max_attempts attempts")
        continue
    end
    
    # προσθήκη του agent με αρχικές τιμές: vel, age, mass, toxicload = 0., pathX, pathY, TL1, TL2, TL3 (η διαδρομή υπολογίζεται αργότερα με plan_best_route!)
    add_agent!(pos, AgentEscapes, model, vel, age, mass, 0., [pos[1]], [pos[2]], [0.0], [0.0], [0.0])
end


# Υπολογίζει τους πίνακες (Balpha, Btime, Brho) του μοντέλου τοξικού φορτίου από τα όρια AEGL του H2S:
# για κάθε σύμπτωμα (οσμή, ερεθισμός, οίδημα) δίνει συγκεντρώσεις, χρόνους και εκθέτες power-law
# που χρησιμοποιεί η tl_rate (και μέσω αυτής η update_toxic_load).
function setupToxic()                                               # Define the setupToxic function
    Atime = [0.0, 2.0, 5.0, 8.0, 15.0, 30.0] #min                # Define the Atime array as a vector of 6 elements                                   # Initialization of the 5 standard AEGL exposure times
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
    Brho = zeros(7,3)                                               # Define the Brho array as a 7x3 matrix of zeros                            # in mg/m^3 for each AEGL band at every time step ‘Atime’
    Balpha = zeros(7,3)                                             # Define the Balpha array as a 7x3 matrix of zeros                          # power-law exponents
    rhomax = zeros(1,3)                                             # Define the rhomax array as a 1x3 matrix of zeros                          # concentration (mg/m^3) that corresponds to the shortest time taumin (stored in Brho[1, k])
    rhomin = zeros(1,3)                                             # Define the rhomin array as a 1x3 matrix of zeros                          # concentration (mg/m^3) that corresponds to the longest time taumax (stored in Brho[7, k])
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
            rhomin[k] = Brho[6, k]*(Btime[6, k]/taumax)^(1/Balpha[6, k]);
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

# Υπολογισμός των πινάκων τοξικότητας μία φορά (global, χρησιμοποιούνται από την tl_rate)
Balpha, Btime, Brho = setupToxic()
# Suffix των αρχείων εξόδου: ίδια ονοματολογία με το DStarLite - Scenario 3.jl
# Suffix for exports: align naming with DStarLite - Scenario 3.jl
run_export_tag = "newAEGL_0.5CM_uncappedTL"  # ετικέτα στα ονόματα αρχείων εξόδου
run_timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")  # χρονική σήμανση της εκτέλεσης


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



# Επιστρέφει το χρώμα του agent ανάλογα με το τοξικό φορτίο: κόκκινο (>= 3), πράσινο (<= 1), αλλιώς πορτοκαλί
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
    const T = 2500  # συνολικός αριθμός frames/βημάτων της προσομοίωσης
    # CM switch times (s) — Thesis, Table 7: 7 CMs, switching completes at 70% of T = 2500 s.
    # Time counter is t = (frame - 1) * dt.
    # πλήθος χαρτών συγκέντρωσης που εναλλάσσονται
    const NUM_MAPS = NUM_CMS
    const CM_SWITCH_TIMES = [0.0, 292.0, 584.0, 876.0, 1168.0, 1460.0, 1752.0]
# -- Στήσιμο Figure & Axis --
    fig = Figure(; size = (800,800))
    ax  = Makie.Axis(fig[1,1];
               title  = "Evacuation with Toxic Trail",
               aspect = DataAspect())

    heatmap!(ax, heightmap; colormap=:grays, alpha=0.3)  # φόντο: ο χάρτης (ημιδιαφανής, σε αποχρώσεις του γκρι)
    
    # --- Concentration Map visualization (contour only, overlay on heightmap) ---
    cm_obs = Observable(penalty_map)  # Observable με το τρέχον CM, ώστε το contour να ανανεώνεται όταν αλλάζει
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
    # ισοϋψείς της συγκέντρωσης (κίτρινο προς πορτοκαλί προς κόκκινο), 10 επίπεδα
    cm_contour = contour!(
        ax, cm_obs;
        colormap = cgrad([:yellow, :orange, :red]),
        levels = 10,
        colorrange = cm_colorrange,
        linewidth = 1.5,
        alpha = 0.7,
    )
    
    # σχεδίαση των ενεργών εξόδων ως κόκκινους κύκλους
    goals = model.goal
    scatter!(ax,
        getindex.(goals,1),
        getindex.(goals,2);
        color  = (:red,50),
        marker = :circle,
    )

    # --- Overlay time counter + active goals (below clock, fixed for this run) ---
    frame_obs = Observable(0)  # τρέχον frame (ενημερώνεται σε κάθε βήμα της εγγραφής)

    # ετικέτα χρόνου πάνω-αριστερά
    counter_lbl = Label(
        fig,
        @lift("Time elapsed = $((($frame_obs-1)*dt)) s"),
        fontsize = 16,
        padding = (6, 10, 6, 10),
        halign = :left,
    )
    # κείμενο με τη λίστα των ενεργών εξόδων και τις συντεταγμένες τους
    active_dests_text =
        "Active goals ($(length(dests))):\n" *
        join(["  ($(round(x; digits=1)), $(round(y; digits=1)))" for (x, y) in dests], "\n")
    # πλαίσιο που περιέχει το ρολόι και τη λίστα εξόδων και αγκιστρώνεται πάνω-αριστερά στο ίδιο κελί με τον άξονα
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

    xs0 = Float64[]  # Initialize empty arrays for positions  # αρχικές θέσεις x των agents
    ys0 = Float64[]  # αρχικές θέσεις y των agents

    for a in allagents(model)  # συλλογή των αρχικών θέσεων
        push!(xs0, a.pos[1])
        push!(ys0, a.pos[2])
    end

    colors0 = Symbol[]  # αρχικά χρώματα των agents
    for a in allagents(model)
        push!(colors0, personcolor(a))
    end

    posobs = Observable(Point2f.(xs0, ys0))  # Observable με τις θέσεις όλων των agents
    colobs = Observable(colors0)  # Observable με τα χρώματα των agents

    # μία γραμμή (trail) ανά agent, με το χρώμα του
    lines_plots = [
        lines!(
            ax,
            [a.pos[1]], [a.pos[2]];
            color     = personcolor(a),
            linewidth = 2,
        )
        for a in allagents(model)
    ]
    # σημεία των agents στην τρέχουσα θέση τους
    agent_scat = scatter!(
        ax, posobs;
        color       = colobs,
        markersize  = 10,
    )

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


# Ονόματα αρχείων εξόδου: βίντεο (mp4) και CSV με θέση και toxicload ανά βήμα
video_file = "SCENARIOS/SCENARIO 4/Simulation Results/AStar_SCENARIO_4_$(n_agents)_$(seed)_$(cost_metric_str)_$(heuristic_code)_$(run_export_tag)_$(run_timestamp).mp4"
csv_file   = "SCENARIOS/SCENARIO 4/Simulation Results/AStar_SCENARIO_4_$(n_agents)_$(seed)_$(cost_metric_str)_$(heuristic_code)_$(run_export_tag)_$(run_timestamp).csv"

# keep a variable for the currently active map index
current_map_idx = 1  # δείκτης του ενεργού χάρτη συγκέντρωσης

path_planning_init_time = Ref(0.0)  # χρόνος αρχικού υπολογισμού διαδρομών (s)
path_planning_replan_time = Ref(0.0)  # χρόνος επανασχεδιασμού διαδρομών κατά τις αλλαγές CM (s)
simulation_time = Ref(0.0)  # χρόνος εκτέλεσης των step! (s)

# Initial pathfinding: same placement as DStarLite (inside @time block, before record)
# υπολογισμός της αρχικής διαδρομής κάθε agent προς την καλύτερη (συντομότερη) από τις ενεργές εξόδους
path_planning_init_time[] += @elapsed begin
    for a in allagents(model)
        plan_best_route!(a, model.goal, model.pathfinderPM)
    end
end

# Εγγραφή του βίντεο: T frames, ένα βήμα του μοντέλου ανά frame
record(fig, video_file, 1:T; framerate=30) do frame
    # update frame counter observable
    frame_obs[] = frame  # ενημέρωση του μετρητή χρόνου στην οθόνη

    # 1) step the model first (same order as DStarLite - Scenario 3.jl)
    simulation_time[] += @elapsed step!(model, 1)  # ένα βήμα του μοντέλου (με μέτρηση του χρόνου)

    # 2) concentration map index for this frame
    global current_map_idx
    t_sim = (frame - 1) * dt  # χρόνος της προσομοίωσης για αυτό το frame
    new_idx = searchsortedlast(CM_SWITCH_TIMES, t_sim)  # δείκτης του τελευταίου χρόνου εναλλαγής που έχει παρέλθει, δηλαδή ο χάρτης CM που ισχύει τώρα
    if new_idx != current_map_idx  # αν άλλαξε ο χάρτης συγκέντρωσης
        current_map_idx = new_idx  # ο νέος χάρτης γίνεται ο τρέχων

        path_planning_replan_time[] += @elapsed begin
            penalty_map .= cm_list[current_map_idx]  # αντιγραφή του νέου CM στο penalty_map (in-place, ώστε να το βλέπει η agent_step!)

            NPM = heightmap .+ penalty_map  # νέος συνδυασμένος χάρτης κόστους
            NPM_int = round.(Int, NPM)  # μετατροπή σε Int

            pm = Pathfinding.penaltymap(model.pathfinderPM)  # ο πίνακας κόστους που χρησιμοποιεί ο pathfinder
            pm .= NPM_int  # ενημέρωση του πίνακα κόστους του pathfinder στη θέση του

            # επανασχεδιασμός της διαδρομής κάθε agent με τον νέο χάρτη κόστους
            for a in allagents(model)
                plan_best_route!(a, model.goal, model.pathfinderPM)
            end
        end

        println("Frame $frame: swapped to concentration map $current_map_idx and replanned paths.")  # ενημερωτικό μήνυμα για την εναλλαγή χάρτη
    end

    # ενημέρωση της οπτικοποίησης του χάρτη συγκέντρωσης
    # Update concentration map visualization
    cm_obs[] = penalty_map

    # 3) update trails/visuals
    # ενημέρωση των trails των agents (ολόκληρο το ιστορικό pathX/pathY)
    for (i, a) in enumerate(allagents(model))
        lines_plots[i][1][] = Point2f.(a.pathX, a.pathY)
    end

    posobs[] = [Point2f(a.pos[1], a.pos[2]) for a in allagents(model)]  # νέες θέσεις των agents
    colobs[] = [personcolor(a) for a in allagents(model)]  # νέα χρώματα ανάλογα με το τοξικό φορτίο

    # 4) collect data
    # καταγραφή δεδομένων του frame στο DataFrame (βήμα, id, θέση, toxicload)
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
    end  # τέλος του record: η εγγραφή τρέχει για T frames (σταθερός ορίζοντας)

    println("Το animation σώθηκε ως $video_file")  # μήνυμα αποθήκευσης του βίντεο
    CSV.write(csv_file, df)  # εγγραφή του CSV
    println("Τα δεδομένα θέσης & toxicload αποθηκεύτηκαν ως $csv_file")  # μήνυμα αποθήκευσης του CSV

    total_path_planning = path_planning_init_time[] + path_planning_replan_time[]  # συνολικός χρόνος υπολογισμού διαδρομών = αρχικός + επανασχεδιασμός
    # αναφορά των χρόνων εκτέλεσης
    println("TIMING BREAKDOWN:")
    println("  Pathfinding time (total): ", round(total_path_planning; digits=6), " s")
    println("    - Initial pathfinding: ", round(path_planning_init_time[]; digits=6), " s")
    println("    - Replanning during CM changes: ", round(path_planning_replan_time[]; digits=6), " s")
    println("  Simulation time (step! only): ", round(simulation_time[]; digits=6), " s")
end  # τέλος του @time block


begin  # Δημιουργία βίντεο ιστογράμματος τοξικού φορτίου από το CSV της εκτέλεσης

    dt = @isdefined(dt) ? dt : 1.0  # αν δεν υπάρχει ήδη ορισμένο, το dt γίνεται 1.0
    seed_str = @isdefined(seed) ? string(seed) : nothing  # το seed ως string (ή nothing αν δεν υπάρχει)
    folder = joinpath("SCENARIOS","SCENARIO 4", "Simulation Results")  # φάκελος αποτελεσμάτων του Scenario 4

    # Φόρτωση CSV με step, agent_id, toxicload
    csv_file = seed_str === nothing ? nothing : joinpath(folder, "AStar_SCENARIO_4_$(n_agents)_$(seed)_$(cost_metric_str)_$(heuristic_code)_$(run_export_tag)_$(run_timestamp).csv")
    if csv_file === nothing || !isfile(csv_file)
        # αν δεν υπάρχει το CSV αυτής της εκτέλεσης (ή δεν υπάρχει seed), πάρε το πιο πρόσφατο AStar_SCENARIO_4_*.csv
        # μοτίβο (regex) που ταιριάζει τα CSV του AStar Scenario 4 με τις ίδιες ετικέτες εξόδου
        pattern = Regex("^AStar_SCENARIO_4_.*_$(cost_metric_str)_$(heuristic_code)_$(run_export_tag)_.*\\.csv\$")
        csvs = filter(f -> occursin(pattern, f), readdir(folder))  # τα αρχεία του φακέλου που ταιριάζουν με το μοτίβο
        @assert !isempty(csvs) "Δεν βρέθηκαν αρχεία *AStar_SCENARIO_4_$(seed)*.csv στο $(folder)."
        stats = stat.(joinpath.(Ref(folder), csvs))  # στατιστικά (ώρα τροποποίησης) των αρχείων αυτών
        latest_idx = argmax(getfield.(stats, :mtime))  # δείκτης του πιο πρόσφατου αρχείου
        csv_file = joinpath(folder, csvs[latest_idx])
        seed_str = "latest"  # το seed στο όνομα του βίντεο ιστογράμματος γίνεται latest
    end

    df = CSV.read(csv_file, DataFrame)  # ανάγνωση του CSV σε DataFrame

# Σταθερά bins του ιστογράμματος
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

    # Μετράει πόσες τιμές πέφτουν σε κάθε εσωτερικό bin, με αυστηρά ανοικτό διάστημα (0,3)
    # Counts για ΑΝΟΙΚΤΟ (0,3) χωρίς StatsBase
    function bin_counts_open(values::AbstractVector{<:Real}, edges::AbstractVector{<:Real})
        counts = zeros(Int, length(edges) - 1)  # ένας μετρητής ανά bin
        @inbounds for v in values  # έλεγχος κάθε τιμής
            if v > edges[1] && v < edges[end]           # αυστηρά (0,3)
                idx = searchsortedlast(edges, v)  # δείκτης του bin στο οποίο ανήκει η τιμή
                idx = clamp(idx, 1, length(counts))  # περιορισμός του δείκτη εντός ορίων
                counts[idx] += 1  # αύξηση του μετρητή του bin
            end
        end
        return counts  # επιστροφή των μετρητών ανά bin
    end

    # Μορφοποίηση των ticks του άξονα x με ένα δεκαδικό ψηφίο (μία τιμή και διάνυσμα τιμών)
    # Formatter για ticks (χωρίς ->)
    function mytickfmt(x::Real)
        string(round(x; digits=1))
    end
    function mytickfmt(xs::AbstractVector{<:Real})
        string.(round.(xs; digits=1))
    end

    # --- Παράμετροι χρόνου / y-scale ---
    T_avail = maximum(df.step)  # τελευταίο βήμα που υπάρχει στο CSV
    T_play  = min(T, T_avail)               # παίξε μέχρι T frames ή όσα υπάρχουν στο CSV
    n_agents_guess = maximum(combine(groupby(df, :step), nrow).nrow)  # μέγιστο πλήθος γραμμών ανά βήμα, δηλαδή το πλήθος των agents (χρησιμοποιείται ως όριο του άξονα y)

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

    xt = collect(0.0:0.5:3.0)  # θέσεις των ticks του άξονα x
    ax.xticks = xt  # οι θέσεις των ticks
    ax.xtickformat = mytickfmt  # μορφοποίηση των ticks
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

    # Χρώματα για τα εσωτερικά bins ανά TL ζώνη (ίδια λογική με DStarLite - Scenario 3.jl)
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
    out_file = joinpath(folder, "AStar_SCENARIO_4_hist__$(n_agents)_$(seed_str)_$(cost_metric_str)_$(heuristic_code)_$(run_export_tag)_$(run_timestamp).mp4")  # αρχείο εξόδου του βίντεο ιστογράμματος
    record(fig, out_file, 1:T_play; framerate = 30) do frame  # για κάθε frame (βήμα) του CSV μέχρι T_play
        frame_obs[] = frame  # ενημέρωση του ρολογιού
        tl = Vector(df[df.step .== frame, :toxicload])  # οι τιμές toxicload των agents σε αυτό το βήμα

        count0       = count(==(0.0), tl)                 # ακριβώς 0 (όπως DStarLite)
        count3       = count(>=(3.0), tl)                 # TL ≥ 3 (περιλαμβάνει και τους agents με TL > 3)
        counts_inner = bin_counts_open(tl, edges_inner)   # μόνο (0,3) ανά 0.5

        y_inner[] = Float64.(counts_inner)  # ενημέρωση των ύψων των εσωτερικών μπαρών
        y0[]      = [Float64(count0)]  # ενημέρωση της μπάρας του 0
        y3[]      = [Float64(count3)]  # ενημέρωση της μπάρας του 3

        # Προαιρετικό auto-scale μόνο στον Υ:
        # ylims!(ax, 0, max(1, maximum(vcat(y0[][1], y_inner[], y3[][1]))))
    end

    println("Βίντεο με ξεχωριστά bins για 0 & 3 και εσωτερικά ανά 0.5 σώθηκε ως: $out_file")  # μήνυμα αποθήκευσης του βίντεο ιστογράμματος

    # Καταγραφή της εκτέλεσης στο RunHistory (αρχεία εξόδου και χρόνος υπολογισμού διαδρομών)
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
