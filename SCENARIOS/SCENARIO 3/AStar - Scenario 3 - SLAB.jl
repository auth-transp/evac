begin   # Φόρτωση των απαραίτητων βιβλιοθηκών
    using Agents                    # πλαίσιο agent-based μοντελοποίησης (ABM, add_agent!, step!, ...)
    using Agents.Pathfinding        # αλγόριθμοι εύρεσης διαδρομής (AStar, AbsolutePenaltyMap, plan_best_route!, ...)
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

    # φορτώνει τη συνάρτηση append_run_history που καταγράφει κάθε εκτέλεση στο RunHistory
    include(joinpath(@__DIR__, "..", "append_run_history.jl"))
end


# Ορισμός του agent: συνεχής 2D agent (pos, vel προστίθενται αυτόματα από το ContinuousAgent)
# Σημείωση: το πεδίο path (grid cells) χρησιμοποιείται ΜΟΝΟ στον τύπο pathfinder full_field· στον per_agent τη διαδρομή την κρατά ο ίδιος ο AStar pathfinder (move_along_route!)
Agents.@agent struct AgentEscapes(ContinuousAgent{2, Float64})
    age::Float64                    # ηλικία (έτη)
    mass::Float64                   # μάζα (kg), χρησιμοποιείται στις ελαστικές συγκρούσεις
    toxicload::Float64              # συνολικό τρέχον τοξικό φορτίο (άθροισμα TL1+TL2+TL3)
    path::Vector{Tuple{Int,Int}}    # ΜΟΝΟ grid path (μόνο στον τύπο full_field· στον per_agent μένει κενό)
    pathX::Vector{Float64}          # ιστορικό θέσης x (trail, για visualization)
    pathY::Vector{Float64}          # ιστορικό θέσης y (trail, για visualization)
    TL1::Vector{Float64}            # ιστορικό τοξικού φορτίου για το 1ο σύμπτωμα (οσμή)
    TL2::Vector{Float64}            # ιστορικό τοξικού φορτίου για το 2ο σύμπτωμα (ερεθισμός)
    TL3::Vector{Float64}            # ιστορικό τοξικού φορτίου για το 3ο σύμπτωμα (οίδημα)
end


begin   # Φόρτωση του heightmap και των χαρτών συγκέντρωσης CM1..CMn από το μοντέλο διασποράς SLAB (αντί για τα χειρόγραφα CMs)
    # heightmap
    heightmap_data = load("NADEEN/Maps/Qatargas Map.jpg")                              # φόρτωση της εικόνας του χάρτη
    heightmap_data = permutedims(channelview(heightmap_data), [2,3,1])[:,:,1]          # μετατροπή σε πίνακα (γραμμές, στήλες) και κράτημα του 1ου καναλιού
    global heightmap = floor.(Int, convert.(Float64, heightmap_data) * 255)            # κλιμάκωση σε ακέραιες τιμές 0..255
    heightmap = 255 .- heightmap   # αυτό κάνει την αντιστροφή

    # --- Μοντέλο διασποράς SLAB: ξεχωριστό script (module SlabFields) που καλείται από εδώ ---
    include(joinpath(@__DIR__, "..", "..", "scripts", "slab_to_concentration_maps.jl"))   # ορίζει το module SlabFields
    const SF = SlabFields                                           # σύντομο όνομα του module

    # Παράμετροι της περίπτωσης διασποράς· κάθε μία αλλάζει με μεταβλητή περιβάλλοντος, π.χ. EVAC_SLAB_CASE=ag_small_D_5ms
    const SLAB_DIR = joinpath(@__DIR__, "..", "..", "SLAB Cases")   # φάκελος με τα αρχεία .in/.predict και το cases.tsv
    const SLAB_CASE = get(ENV, "EVAC_SLAB_CASE", "ag_small_D_3ms")  # όνομα περίπτωσης: μέγεθος διαρροής, σταθερότητα, ταχύτητα ανέμου
    const SLAB_SOURCE = (parse(Int, get(ENV, "EVAC_SLAB_SRC_I", "360")), parse(Int, get(ENV, "EVAC_SLAB_SRC_J", "352")))   # pixel πηγής (i, j)· (360,352) = κορυφή των χειρόγραφων CM2-CM4
    const SLAB_WIND_DEG = parse(Float64, get(ENV, "EVAC_SLAB_WIND", "70"))   # κατεύθυνση κίνησης του νέφους σε μοίρες από το +i προς το +j (70° = άξονας των χειρόγραφων CMs)
    const SLAB_Z = parse(Float64, get(ENV, "EVAC_SLAB_Z", "2.0"))   # ύψος υπολογισμού της συγκέντρωσης (m), ύψος αναπνοής· ο κλειστός τύπος του SLAB δίνει το πεδίο σε οποιοδήποτε ύψος
    const PENALTY_CAP = parse(Float64, get(ENV, "EVAC_SLAB_PENALTY_CAP", "255.0"))   # άνω όριο του κόστους διαδρομής από τη συγκέντρωση (όπως η κλίμακα 0..255 των χειρόγραφων CMs)
    const MG_PER_PPM = 34.0 / 24.04                                 # ppm -> mg/m^3 με MW = 34 g/mol του setupToxic
    const SLAB_CM_PERIOD = parse(Float64, get(ENV, "EVAC_SLAB_CM_PERIOD", "60"))   # κάθε πόσα frames (1 frame = 1 s) αλλάζει το CM
    const SLAB_WIDEN = parse(Float64, get(ENV, "EVAC_SLAB_WIDEN", "1.3"))   # διεύρυνση του νέφους κατά πλάτος (1 = ακριβώς το SLAB)· διατηρεί τη συνολική εγκάρσια μάζα, άρα μειώνει ανάλογα τη συγκέντρωση
    const SLAB_CM_EDGES = vcat(collect(0.0:SLAB_CM_PERIOD:2499.0), 2500.0)   # όρια ισχύος των CM: ένα CM κάθε SLAB_CM_PERIOD s και το T = 2500 s
    const SLAB_CM_TIMES = (SLAB_CM_EDGES[1:end-1] .+ SLAB_CM_EDGES[2:end]) ./ 2          # χρόνος του SLAB για κάθε CM = μέσο του διαστήματος ισχύος του

    # κλάσμα H2S του μίγματος από το cases.tsv (το SLAB δίνει κλάσμα όγκου του συνόλου του εκλυόμενου μίγματος)
    cases_rows = split.(readlines(joinpath(SLAB_DIR, "cases.tsv")), '\t')   # πίνακας περιπτώσεων: κάθε γραμμή χωρισμένη σε στήλες
    case_row = only(filter(r -> r[1] == SLAB_CASE, cases_rows))     # η γραμμή της επιλεγμένης περίπτωσης
    const SLAB_XH2S = parse(Float64, case_row[findfirst(==("xh2s"), cases_rows[1])])   # μοριακό κλάσμα H2S στο μίγμα

    # --- Χάρτες CM1..CMn από το SLAB ---
    const NUM_CMS = length(SLAB_CM_TIMES)                           # πλήθος χαρτών (ένας ανά SLAB_CM_PERIOD s, π.χ. 42 για 60 s)
    slab_cloud = SF.parse_slab_output(joinpath(SLAB_DIR, SLAB_CASE * ".predict"))   # παράμετροι του νέφους από την έξοδο του SLAB
    exposure_list = Vector{Matrix{Float32}}(undef, NUM_CMS)         # πραγματικές συγκεντρώσεις (mg/m^3) για το τοξικό φορτίο, σε Float32 για οικονομία μνήμης (42 χάρτες)
    cm_list = Vector{Matrix{Float32}}(undef, NUM_CMS)               # κόστος διαδρομής: οι ίδιες συγκεντρώσεις κομμένες στο PENALTY_CAP
    for k in 1:NUM_CMS
        A = SF.concentration_map(slab_cloud; dims = size(heightmap), src = SLAB_SOURCE, wind_deg = SLAB_WIND_DEG, t = SLAB_CM_TIMES[k],
                                 z = SLAB_Z, m_per_px = 2500.0 / 723.37, scale = SLAB_XH2S, widen = SLAB_WIDEN)   # ppm H2S στο πλέγμα του heightmap (1 pixel = 2500/723.37 m)
        exposure_list[k] = A .* Float32(MG_PER_PPM)                 # ppm -> mg/m^3
        cm_list[k] = min.(exposure_list[k], Float32(PENALTY_CAP))   # κοπή για το κόστος διαδρομής
    end

    # basic check: dimensions match heightmap
    # έλεγχος ότι κάθε CM έχει τις ίδιες διαστάσεις με το heightmap (αλλιώς σταματά με μήνυμα σφάλματος)
    for k in 1:NUM_CMS
        @assert size(cm_list[k]) == size(heightmap) "Concentration map $k size mismatch with heightmap"
    end

    # start with first CM
    # το penalty_map (τρέχον CM) ξεκινά ως αντίγραφο του 1ου χάρτη
    global penalty_map = Float64.(cm_list[1])                       # τα τρέχοντα penalty_map/exposure_map είναι Float64, οι λίστες Float32
    global exposure_map = Float64.(exposure_list[1])                # τρέχων χάρτης έκθεσης (mg/m^3) που διαβάζει το τοξικό φορτίο
end

# συνδυασμένος χάρτης κόστους = ανάγλυφο/εμπόδια (heightmap) + τοξική συγκέντρωση (penalty_map)
NPM = heightmap .+ penalty_map
NPM_int = round.(Int, NPM)   # convert to Int for PenaltyMap


begin   # Αρχικοποίηση των παραμέτρων του μοντέλου
    const METERS_TO_PIXELS = 723.37 / 2500.0                            # συντελεστής μετατροπής μέτρων σε pixels του χάρτη
    dt = 1.   ## discrete timestep each iteration of the model          # Define the dt variable as 1
    seed = 123  ## seed for random number generator                     # Define the seed variable as 123
    make_histogram = false  # true για να δημιουργηθεί το βίντεο ιστογράμματος τοξικού φορτίου (default false, ώστε να μη γεμίζουν οι φάκελοι με αρχεία)
    n_agents = 100                                                       # Define the n_agents variable as 50
    age_range = (22,60)                                                 # Define the age_range variable as a tuple of 22 and 60
    speed_range = (4.0,7.0)                                            # Define the speed_range variable as a tuple of 4.0 and 7.0
    mass_range = (50,80)                                                # Define the mass_range variable as a tuple of 50 and 80
    ag_range_y = (size(heightmap)[1]/4):(3*size(heightmap)[1]/4)    # Define the ag_range_y variable as the central range (1/4 to 3/4 of the size) of the heightmap's 1st dimension # [1] stands for the 1st dimension (rows)
    ag_range_x = (size(heightmap)[2]/4):(3*size(heightmap)[2]/4)    # Define the ag_range_x variable as the central range (1/4 to 3/4 of the size) of the heightmap's 2nd dimension # [2] stands for the 2nd dimension (columns)
    dims = (size(NPM))                                            # Define the dims variable as the dimensions (rows, columns) of the NPM array, the same as the heightmap
    walkmap = BitArray(trues(dims...))                                 # Define the walkmap variable as a BitArray of true values with the dimensions of the heightmap array

    # Set walkmap to false in white areas of the heightmap (after reversal, white = high values)
    # White areas are where heightmap value is above threshold (e.g., > 245)
    # Black and grey areas (low to medium values) remain walkable
    white_threshold = 245                                  # κατώφλι: πάνω από αυτήν την τιμή το cell θεωρείται μη βατό
    walkmap[heightmap .> white_threshold] .= false         # μη βατά cells = λευκές περιοχές του (αντεστραμμένου) heightmap
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


begin   # Ορισμός του pathfinder A* και των ιδιοτήτων του μοντέλου
    # κόστος κίνησης ανά cell από τον συνδυασμένο χάρτη, με γειτονιά 8 κατευθύνσεων (MaxDistance{2})
    cost_metric_obj = AbsolutePenaltyMap(NPM_int, MaxDistance{2}())
    # A* pathfinder: οι agents δεν περνούν από μη βατά cells (walkmap) και ακολουθούν το cost_metric
    pathfinderPM = AStar(space; walkmap = walkmap, cost_metric = cost_metric_obj)
    # ιδιότητες προσβάσιμες μέσα στο μοντέλο ως model.<όνομα>
    properties = (
        pathfinderPM = pathfinderPM,
        heightmap = heightmap,
        dt = dt,
        speed_range = speed_range,
        goal = dests,
    )
end


# Pathfinder type: "per_agent" (ο A* του Agents.jl ανά agent: plan_best_route! από την τρέχουσα θέση του agent προς κάθε έξοδο) ή
# "full_field" (A* χωρίς σημείο εκκίνησης, h ≡ 0 = Dijkstra προς τα πίσω από κάθε έξοδο: ένα πεδίο κόστους ανά έξοδο, κοινό για όλους τους agents)
const PATHFINDER_TYPE = lowercase(get(ENV, "EVAC_PATHFINDER", "per_agent"))     # επιλογή τύπου: μεταβλητή περιβάλλοντος EVAC_PATHFINDER=per_agent | full_field
PATHFINDER_TYPE in ("per_agent", "full_field") || error("Άγνωστος τύπος pathfinder «$(PATHFINDER_TYPE)»: χρησιμοποίησε per_agent ή full_field")
println("Pathfinder type: ", PATHFINDER_TYPE == "full_field" ? "Full field" : "Per agent")

# Μετακινεί τον agent κατά μήκος ενός προϋπολογισμένου μονοπατιού (λίστα από grid cells), κάνοντας ένα βήμα ταχύτητας `speed` για χρονικό διάστημα `dt` ανά κλήση.
function move_along_precomputed_path!(agent::AgentEscapes, speed, dt)
    # Ο agent καλύπτει ακριβώς απόσταση speed*dt ανά βήμα: όταν φτάνει σε ένα waypoint, το υπόλοιπο της απόστασης
    # συνεχίζει προς τα επόμενα (ίδια συμπεριφορά με το move_along_route! του A* στο Agents.jl). Χωρίς αυτή τη μεταφορά
    # ο agent θα σταματούσε σε κάθε waypoint και θα έχανε το υπόλοιπο, κινούμενος ~10-15% πιο αργά από το μοντέλο ταχύτητας.
    remaining = speed * dt                                                                          # απόσταση που απομένει να καλυφθεί σε αυτό το βήμα
    pos = agent.pos                                                                                 # τρέχουσα θέση
    while !isempty(agent.path) && remaining > 0                                                     # όσο υπάρχουν waypoints και απομένει απόσταση (με speed == 0 δεν κινείται)
        cell = agent.path[1]                                                                        # επόμενο waypoint (grid cell)
        target = SVector{2,Float64}(cell[1], cell[2])                                               # θέση του waypoint
        d = norm(target .- pos)                                                                     # απόσταση ως το waypoint
        if d <= remaining                                                                           # το waypoint φτάνεται μέσα σε αυτό το βήμα (και όταν d == 0: χωρίς διαίρεση με 0)
            pos = target                                                                            # ο agent φτάνει στο waypoint
            remaining -= d                                                                          # η απόσταση που περισσεύει συνεχίζει προς το επόμενο waypoint
            popfirst!(agent.path)                                                                   # το waypoint αφαιρείται από το path
        else
            pos = pos .+ (target .- pos) ./ d .* remaining                                          # κίνηση κατά την απόσταση που απομένει προς το waypoint (d > remaining > 0)
            remaining = 0.0                                                                         # η απόσταση του βήματος εξαντλήθηκε
        end
    end
    agent.pos = pos                                                                                 # νέα θέση του agent
end


# Helper: true if position is within radius of any goal (TL and movement stop when true)
# Επιστρέφει true αν η θέση απέχει <= radius από οποιαδήποτε έξοδο (τότε σταματούν η κίνηση και η συσσώρευση TL)
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
    # Ct παίρνεται από το global exposure_map (ο τρέχων χάρτης έκθεσης του SLAB, mg/m^3)· το penalty_map (κομμένο στο PENALTY_CAP) είναι μόνο το κόστος διαδρομής
    Ct = exposure_map[i, j]                                                      # συγκέντρωση τοξικού (mg/m^3) στο cell του agent
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

    if PATHFINDER_TYPE == "per_agent"
        # κίνηση κατά μήκος της διαδρομής που έχει σχεδιάσει ο A* (plan_best_route!), με ταχύτητα μετατραμένη σε pixels
        move_along_route!(person, model, model.pathfinderPM, speed * METERS_TO_PIXELS, model.dt)
    else
        # full_field: κίνηση κατά μήκος του προϋπολογισμένου path (cells) με ταχύτητα μετατραμένη σε pixels
        move_along_precomputed_path!(person, speed * METERS_TO_PIXELS, model.dt)
    end
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

# Add agents (initial route planning is timed inside the @time block with the animation, like DStarLite)
# Δημιουργία των agents σε τυχαίες βατές θέσεις (η αρχική διαδρομή υπολογίζεται αργότερα, μέσα στο @time block)
for _ in 1:n_agents
    age = rand(abmrng(model))*(age_range[2]-age_range[1]) + age_range[1]                  # τυχαία ηλικία εντός age_range
    mass = rand(abmrng(model)) * (mass_range[2]-mass_range[1]) + mass_range[1]            # τυχαία μάζα εντός mass_range
    vel = Tuple(rand(abmrng(model), 2) .* (speed_range[2]-speed_range[1]) .+ speed_range[1])  # τυχαία αρχική ταχύτητα (vx, vy)

    # Keep trying to find a valid spawn position that is walkable (walkmap == true)
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
    # (vel, age, mass, toxicload = 0., κενό path, trail pathX/pathY με την αρχική θέση, ιστορικά TL1..TL3 ξεκινούν από 0.0)
    add_agent!(pos, AgentEscapes, model, vel, age, mass, 0., Tuple{Int,Int}[], [pos[1]], [pos[2]], [0.0], [0.0], [0.0])
end


# Υπολογίζει τους πίνακες (Balpha, Btime, Brho) του μοντέλου τοξικού φορτίου από τα όρια AEGL του H2S:
# για κάθε σύμπτωμα (οσμή, ερεθισμός, οίδημα) δίνει συγκεντρώσεις, χρόνους και εκθέτες power-law
# που χρησιμοποιεί η tl_rate (καλείται από την update_toxic_load).
function setupToxic()                                               # Define the setupToxic function
    Atime = [0.0, 2.0, 5.0, 8.0, 15.0, 30.0] #min                # Define the Atime array as a vector of 6 elements                                    # Initialization of the 5 standard AEGL exposure times
    Arho = zeros(3, 6)                                              # Define the Arho array as a 3x6 matrix                                     # the concentration of the three symptoms compared to the AEGL concentrations
    Arho[1, 2:6] = [2.87, 2.33, 2.09, 1.81, 1.55]                   # Define the Arho array for the first row and columns 2 to 6                # odor
    Arho[2, 2:6] = [202.38, 164.38, 147.74, 128.10, 109.45]         # Define the Arho array for the second row and columns 2 to 6               # irritation
    Arho[3, 2:6] = [286.71, 232.87, 209.30, 181.47, 155.05] #ppm    # Define the Arho array for the third row and columns 2 to 6                # edema
    MW = 34 #Molecular weight of H2S in g/mol
    Arho *= MW/24.04 #mg/m^3                                        # Multiply the Arho array by the molecular weight of H2S divided by 24.04
    Arho = Arho'                                                    # Transpose the Arho array
    Atime = Atime*60 #seconds                                       # Multiply the Atime array by 60 to convert from minutes to seconds
    taumin = 200.                                                   # Define the taumin variable as 200 seconds (Boris&Patnaik, 2014)          # shortest exposure time over which an AEGL 1, 2 or 3 onset can be reached
    taumax = 86400.                                                 # Define the taumax variable as 86400 seconds (Boris&Patnaik, 2014)        # longest exposure time over which an AEGL 1, 2 or 3 onset can be reached
    Brho = zeros(7,3)                                               # Define the Brho array as a 7x3 matrix of zeros                            # in mg/m^3 for each AEGL band at every time step ‘Atime’
    Balpha = zeros(7,3)                                             # Define the Balpha array as a 7x3 matrix of zeros                          # power-law exponents
    rhomax = zeros(1,3)                                             # Define the rhomax array as a 1x3 matrix of zeros                          # maximum concentration of H2S per symptom (the one giving an onset time of taumin)
    rhomin = zeros(1,3)                                             # Define the rhomin array as a 1x3 matrix of zeros                          # minimum concentration of H2S per symptom (the one giving an onset time of taumax)
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
    # παρέκταση των ακραίων συγκεντρώσεων: rhomax (για taumin) στη θέση 1 και rhomin (για taumax) στη θέση 7
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
# Suffix for exports: align naming with DStarLite - Scenario 3.jl
run_export_tag = "_SLAB" * (PATHFINDER_TYPE == "full_field" ? "_pfFull" : "")   # επίθημα στα ονόματα αρχείων εξόδου: SLAB (+ pathfinder όταν δεν είναι ο προεπιλεγμένος)
run_timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")            # χρονική σήμανση της εκτέλεσης


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



# Μετατροπή θέσης (x, y) σε cell του πλέγματος με περιορισμό εντός των διαστάσεων dims (μέθοδος για Tuple).
@inline function world_to_cell(p::Tuple{Float64,Float64}, dims::Tuple{Int,Int})
    i = clamp(Int(floor(p[1])), 1, dims[1])                                                         # ακέραιο τμήμα της 1ης συντεταγμένης, εντός ορίων
    j = clamp(Int(floor(p[2])), 1, dims[2])                                                         # ακέραιο τμήμα της 2ης συντεταγμένης, εντός ορίων
    return (i, j)                                                                                   # το cell (i, j)
end

# Ίδια μετατροπή για θέση τύπου SVector (πεδίο pos των agents).
@inline function world_to_cell(p::SVector{2,Float64}, dims::Tuple{Int,Int})
    i = clamp(Int(floor(p[1])), 1, dims[1])                                                         # ακέραιο τμήμα της 1ης συντεταγμένης, εντός ορίων
    j = clamp(Int(floor(p[2])), 1, dims[2])                                                         # ακέραιο τμήμα της 2ης συντεταγμένης, εντός ορίων
    return (i, j)                                                                                   # το cell (i, j)
end

# Από τις διαδρομές προς τις εξόδους επιλέγει τη συντομότερη (λιγότερα cells)· αγνοεί τις κενές και επιστρέφει κενή διαδρομή αν δεν υπάρχει καμία.
function choose_best_path(paths::Vector{Vector{Tuple{Int,Int}}})
    filter!(!isempty, paths)                                                                        # αφαίρεση των κενών διαδρομών (in-place)
    isempty(paths) && return Tuple{Int,Int}[]                                                       # καμία διαδρομή: επιστρέφει κενή
    return paths[argmin(length.(paths))]                                                            # η διαδρομή με το μικρότερο μήκος
end

# ---------------- A* full-field: backward Dijkstra from an exit (A* with no start, h ≡ 0) ----------------

# Πεδίο κόστους g(s) = ελάχιστο κόστος από το cell s ως την έξοδο `goal`, με τα κόστη ακμών (delta_cost), τη γειτονιά
# (get_neighbors) και τον έλεγχο βατότητας (inbounds) του A* pathfinder του Agents.jl, όπως στην find_path.
# Χωρίς σημείο εκκίνησης δεν υπάρχει ευρετική (h ≡ 0): ο A* γίνεται Dijkstra προς τα πίσω από την έξοδο και τρέχει
# μέχρι να αδειάσει η open list, δηλαδή υπολογίζει όλα τα προσβάσιμα cells (full-field). Τα κόστη είναι συμμετρικά.
"""
    astar_full_field(pathfinder::AStar{2}, goal) -> Matrix{Int}

Cost-to-go field of `goal` (`typemax(Int)` = unreachable): A* of `pathfinder` run backwards from `goal` with
no start, so `h ≡ 0` and the search is Dijkstra over the whole walkable grid. Same edge costs, neighborhood
and walkability test as `Pathfinding.find_path`. Not incremental: recomputed from scratch on every call.
"""
function astar_full_field(pathfinder::AStar{2}, goal::Tuple{Int,Int})
    g = fill(typemax(Int), size(pathfinder.walkmap))                                                # g = ∞ (typemax) για όλα τα cells
    Pathfinding.inbounds(goal, pathfinder) || return g                                              # μη βατή έξοδος: κανένα cell δεν τη φτάνει
    g[goal...] = 0                                                                                  # κόστος της εξόδου = 0
    open_list = Pathfinding.PriorityQueue{Tuple{Int,Int},Int}()                                     # open list του A* (ίδια δομή με την find_path)· κλειδί = g, αφού h ≡ 0
    open_list[goal] = 0                                                                             # η αναζήτηση ξεκινά από την έξοδο
    while !isempty(open_list)                                                                       # μέχρι να αδειάσει η open list (full-field)
        cur, g_cur = Pathfinding.dequeue_pair!(open_list)                                           # το cell με το μικρότερο g
        for nbor in Pathfinding.get_neighbors(cur, pathfinder)                                      # γείτονες με τη γειτονιά του A*
            Pathfinding.inbounds(nbor, pathfinder) || continue                                      # μόνο εντός ορίων και βατοί
            new_g = g_cur + Pathfinding.delta_cost(pathfinder, nbor, cur)                           # κόστος από τον γείτονα ως την έξοδο μέσω του cur
            if new_g < g[nbor...]                                                                   # βρέθηκε φθηνότερη διαδρομή
                g[nbor...] = new_g                                                                  # ενημέρωση του g
                open_list[nbor] = new_g                                                             # εισαγωγή/μείωση κλειδιού στην open list
            end
        end
    end
    return g                                                                                        # το πεδίο κόστους της εξόδου
end

# Διαδρομή από το cell `start` ως την έξοδο `goal` πάνω στο πεδίο g: σε κάθε βήμα μετάβαση στον γείτονα που ελαχιστοποιεί
# c(s,s') + g(s'), με την ίδια σειρά γειτόνων και τον ίδιο κανόνα με την extract_path του D* Lite.
"""
    extract_field_path(pathfinder::AStar{2}, g, start, goal; max_steps = 10_000) -> Vector{Tuple{Int,Int}}

Greedy descent of the cost-to-go field `g` from `start` to `goal` (same rule and neighbor order as the
D* Lite `extract_path`). Unreachable neighbors (`typemax(Int)`) are skipped.
"""
function extract_field_path(pathfinder::AStar{2}, g::Matrix{Int}, start::Tuple{Int,Int}, goal::Tuple{Int,Int}; max_steps = 10_000)
    path = Tuple{Int,Int}[start]                                                                    # η διαδρομή ξεκινά από το cell του agent
    cur = start                                                                                     # τρέχον cell
    for _ in 1:max_steps                                                                            # όριο βημάτων για προστασία από ατέρμονο βρόχο
        cur == goal && break                                                                        # έφτασε στην έξοδο
        best = nothing                                                                              # καλύτερος γείτονας
        best_val = typemax(Int)                                                                     # και η τιμή του c(cur,n) + g(n)
        for n in Pathfinding.get_neighbors(cur, pathfinder)                                         # γείτονες με τη γειτονιά του A*
            Pathfinding.inbounds(n, pathfinder) || continue                                         # μόνο εντός ορίων και βατοί
            g[n...] == typemax(Int) && continue                                                     # μη προσβάσιμος γείτονας (αποφυγή υπερχείλισης)
            val = Pathfinding.delta_cost(pathfinder, cur, n) + g[n...]                              # c(cur,n) + g(n)
            if val < best_val                                                                       # νέο ελάχιστο
                best_val = val
                best = n
            end
        end
        best === nothing && break                                                                   # καμία γνωστή διαδρομή
        cur = best                                                                                  # μετάβαση στον καλύτερο γείτονα
        push!(path, cur)                                                                            # η επόμενη κορυφή της διαδρομής
    end
    return path                                                                                     # η διαδρομή (cells) ως την έξοδο
end


function personcolor(person::AgentEscapes)  # Χρώμα του agent ανάλογα με το toxicload
    if person.toxicload >= 3.
        return :red                 # θανατηφόρο φορτίο
    elseif person.toxicload <= 1.
        return :green               # χαμηλό φορτίο
    else
        return :orange              # ενδιάμεσο φορτίο
    end
end

@time begin   # Δημιουργία animation με trails & συλλογή CSV θέσης και toxicload
    const T = 2500      # συνολικός αριθμός frames/βημάτων της προσομοίωσης
    # CM switch times (s): νέος χάρτης SLAB κάθε SLAB_CM_PERIOD s (προεπιλογή 60 s = 60 frames), βλ. το μπλοκ φόρτωσης στην αρχή.
    # Time counter is t = (frame - 1) * dt.
    const NUM_MAPS = NUM_CMS    # πλήθος χαρτών που εναλλάσσονται
    const CM_SWITCH_TIMES = SLAB_CM_EDGES[1:end-1]                  # χρόνοι ενεργοποίησης των CM (τα όρια του μπλοκ φόρτωσης χωρίς το T)
# -- Στήσιμο Figure & Axis --
    fig = Figure(; size = (800,800))
    ax  = Makie.Axis(fig[1,1];
               title  = "Evacuation with Toxic Trail",
               aspect = DataAspect())

    heatmap!(ax, heightmap; colormap=:grays, alpha=0.3)     # φόντο: ο χάρτης (ημιδιαφανής, σε αποχρώσεις του γκρι)

    # --- Concentration Map visualization (contour only, overlay on heightmap) ---
    cm_obs = Observable(penalty_map)    # Observable με το τρέχον CM, ώστε το contour να ανανεώνεται
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
        levels = 10,
        colorrange = cm_colorrange,
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

    # αρχικές θέσεις όλων των agents
    for a in allagents(model)
        push!(xs0, a.pos[1])
        push!(ys0, a.pos[2])
    end

    # αρχικά χρώματα όλων των agents
    colors0 = Symbol[]
    for a in allagents(model)
        push!(colors0, personcolor(a))
    end

    posobs = Observable(Point2f.(xs0, ys0))     # Observable με τις θέσεις όλων των agents
    colobs = Observable(colors0)                # Observable με τα χρώματα των agents

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


# ονόματα αρχείων εξόδου (βίντεο και CSV) με παραμέτρους της εκτέλεσης και timestamp
video_file = "SCENARIOS/SCENARIO 3/Simulation Results/AStar_SCENARIO_3_$(n_agents)_$(seed)$(run_export_tag)_$(run_timestamp).mp4"
csv_file   = "SCENARIOS/SCENARIO 3/Simulation Results/AStar_SCENARIO_3_$(n_agents)_$(seed)$(run_export_tag)_$(run_timestamp).csv"

# keep a variable for the currently active map index
# δείκτης του ενεργού concentration map (ξεκινά από το 1ο)
current_map_idx = 1

# συσσωρευτές χρόνου (σε δευτερόλεπτα)
path_planning_init_time = Ref(0.0)      # αρχικός σχεδιασμός διαδρομών
path_planning_replan_time = Ref(0.0)    # επανασχεδιασμός κατά τις αλλαγές CM
simulation_time = Ref(0.0)              # χρόνος των κλήσεων step! μόνο

# Initial pathfinding: same placement as DStarLite (inside @time block, before record)
# Αρχικός σχεδιασμός των διαδρομών (εξαρτάται από τον τύπο pathfinder)
grid_dims = size(NPM_int)                                           # διαστάσεις του grid
exit_cells = [world_to_cell(g, grid_dims) for g in model.goal]      # οι έξοδοι ως cells του grid
fields_ref = Ref(Matrix{Int}[])                                     # πεδία κόστους ανά έξοδο (μόνο στον τύπο full_field)
path_planning_init_time[] += @elapsed begin
    if PATHFINDER_TYPE == "per_agent"
        # Per agent: ο A* του Agents.jl υπολογίζει για κάθε agent την καλύτερη διαδρομή προς τις εξόδους (dests)
        for a in allagents(model)
            plan_best_route!(a, dests, model.pathfinderPM)
        end
    else
        # Full field: ένα πεδίο κόστους ανά έξοδο (A* χωρίς start, h ≡ 0), κοινό για όλους τους agents
        fields_ref[] = [astar_full_field(model.pathfinderPM, goal_cell) for goal_cell in exit_cells]
        for a in allagents(model)
            start = world_to_cell(a.pos, grid_dims)                 # cell εκκίνησης του agent
            paths = Vector{Vector{Tuple{Int,Int}}}([extract_field_path(model.pathfinderPM, fld, start, goal) for (fld, goal) in zip(fields_ref[], exit_cells)])   # διαδρομή προς κάθε έξοδο
            a.path = choose_best_path(paths)                        # κρατά τη συντομότερη
        end
    end
end

# Κύρια εγγραφή: για κάθε frame ελέγχει πρώτα αλλαγή CM (και επανασχεδιάζει), μετά κάνει βήμα προσομοίωσης, ενημερώνει γραφικά και δεδομένα
record(fig, video_file, 1:T; framerate=30) do frame
    # update frame counter observable
    frame_obs[] = frame

    # 1) concentration map index for this frame
    global current_map_idx
    t_sim = (frame - 1) * dt                                    # τρέχων χρόνος προσομοίωσης (s)
    new_idx = searchsortedlast(CM_SWITCH_TIMES, t_sim)          # τελευταίο CM του οποίου ο χρόνος ενεργοποίησης έχει περάσει
    # αν άλλαξε το ενεργό CM: ενημέρωση χαρτών, κόστους και επανασχεδιασμός των διαδρομών (A* από την αρχή)
    if new_idx != current_map_idx
        current_map_idx = new_idx

        path_planning_replan_time[] += @elapsed begin
            penalty_map .= cm_list[current_map_idx]             # αντικατάσταση του penalty_map in-place
            exposure_map .= exposure_list[current_map_idx]      # αντικατάσταση του exposure_map in-place

            # νέος συνδυασμένος χάρτης κόστους
            NPM = heightmap .+ penalty_map
            NPM_int = round.(Int, NPM)

            # ενημέρωση του penalty map του ίδιου του pathfinder in-place (χωρίς νέο AStar)
            pm = Pathfinding.penaltymap(model.pathfinderPM)
            pm .= NPM_int

            if PATHFINDER_TYPE == "per_agent"
                # νέα διαδρομή για κάθε agent από την τρέχουσα θέση του (A* από την αρχή)
                for a in allagents(model)
                    plan_best_route!(a, model.goal, model.pathfinderPM)
                end
            else
                # Full field: νέος υπολογισμός από την αρχή των πεδίων κόστους (ένα ανά έξοδο) και νέα διαδρομή κάθε agent από τα πεδία
                fields_ref[] = [astar_full_field(model.pathfinderPM, goal_cell) for goal_cell in exit_cells]
                for a in allagents(model)
                    start = world_to_cell(a.pos, grid_dims)         # τρέχον cell του agent
                    paths = Vector{Vector{Tuple{Int,Int}}}([extract_field_path(model.pathfinderPM, fld, start, goal) for (fld, goal) in zip(fields_ref[], exit_cells)])   # διαδρομή προς κάθε έξοδο
                    a.path = choose_best_path(paths)                # κρατά τη συντομότερη
                end
            end
        end

        println("Frame $frame: swapped to concentration map $current_map_idx and replanned paths.")
    end

    # 2) step the model AFTER the CM check: the new CM is already active in the step that starts at
    #    t = CM_SWITCH_TIMES[k] (same order in every scenario and benchmark)
    # Το βήμα εκτελείται μετά τον έλεγχο εναλλαγής CM, ώστε το νέο CM να ισχύει ήδη στο βήμα που ξεκινά στο t = CM_SWITCH_TIMES[k] (ίδια σειρά σε όλα τα σενάρια και benchmarks).
    simulation_time[] += @elapsed step!(model, 1)

    # Update concentration map visualization
    cm_obs[] = penalty_map      # ανανέωση των ισοϋψών του CM (σε κάθε frame)

    # 3) update trails/visuals
    # ξαναχτίζει ολόκληρο το trail κάθε agent από τα pathX/pathY
    for (i, a) in enumerate(allagents(model))
        lines_plots[i][1][] = Point2f.(a.pathX, a.pathY)
    end

    # ενημέρωση θέσεων και χρωμάτων των agents στο plot
    posobs[] = [Point2f(a.pos[1], a.pos[2]) for a in allagents(model)]
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

    end

    println("Το animation σώθηκε ως $video_file")
    CSV.write(csv_file, df)                             # αποθήκευση των δεδομένων θέσης & toxicload
    println("Τα δεδομένα θέσης & toxicload αποθηκεύτηκαν ως $csv_file")

    # --- Timing summary ---
    # συνολικός χρόνος pathfinding = αρχικός σχεδιασμός + replanning
    total_path_planning = path_planning_init_time[] + path_planning_replan_time[]
    println("TIMING BREAKDOWN:  (Pathfinder type: ", PATHFINDER_TYPE == "full_field" ? "Full field" : "Per agent", ")")
    println("  Pathfinding time (total): ", round(total_path_planning; digits=6), " s")
    println("    - Initial pathfinding: ", round(path_planning_init_time[]; digits=6), " s")
    println("    - Replanning during CM changes: ", round(path_planning_replan_time[]; digits=6), " s")
    println("  Simulation time (step! only): ", round(simulation_time[]; digits=6), " s")
end


if make_histogram   # Ιστόγραμμα τοξικού φορτίου ανά frame (βίντεο) από το CSV της προσομοίωσης

    dt = @isdefined(dt) ? dt : 1.0                                  # βήμα χρόνου (αν δεν έχει οριστεί, 1 s)
    seed_str = @isdefined(seed) ? string(seed) : nothing            # seed ως string για το όνομα αρχείου
    folder = joinpath("SCENARIOS","SCENARIO 3", "Simulation Results")   # φάκελος αποτελεσμάτων

    # Φόρτωση CSV με step, agent_id, toxicload
    csv_file = seed_str === nothing ? nothing : joinpath(folder, "AStar_SCENARIO_3_$(n_agents)_$(seed)$(run_export_tag)_$(run_timestamp).csv")
    if csv_file === nothing || !isfile(csv_file)
        # αν δεν δοθεί seed ή δεν υπάρχει το αρχείο, πάρε το πιο πρόσφατο AStar_SCENARIO_3_*.csv
        pattern = Regex("^AStar_SCENARIO_3_\\d+_\\d+$(run_export_tag)_\\d{4}-\\d{2}-\\d{2}_\\d{2}-\\d{2}-\\d{2}\\.csv\$")
        csvs = filter(f -> occursin(pattern, f), readdir(folder))
        @assert !isempty(csvs) "Δεν βρέθηκαν αρχεία *AStar_SCENARIO_3_$(seed)*.csv στο $(folder)."
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

    # Άκρες (TL = 0 και TL ≥ 3) σαν ξεχωριστές μπάρες με μικρό offset για να μην επικαλύπτονται
    edge_offset = 0.30
    x0_pos      = 0.0 - edge_offset   # π.χ. -0.30
    x3_pos      = 3.0 + edge_offset   # π.χ.  3.30
    bin_w_edge  = 0.30

    # Counts για ΑΝΟΙΚΤΟ (0,3) χωρίς StatsBase
    # Μετράει πόσες τιμές πέφτουν σε κάθε εσωτερικό bin (edges), μόνο για τιμές αυστηρά μέσα στο (edges[1], edges[end])
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

    # Formatter για ticks
    # στρογγυλοποίηση των ticks του άξονα x σε 1 δεκαδικό (για μία τιμή και για διάνυσμα τιμών)
    function mytickfmt(x::Real)
        string(round(x; digits=1))
    end
    function mytickfmt(xs::AbstractVector{<:Real})
        string.(round.(xs; digits=1))
    end

    # --- Παράμετροι χρόνου / y-scale ---
    T_avail = maximum(df.step)              # διαθέσιμα frames στο CSV
    T_play  = min(T, T_avail)               # παίξε μέχρι T frames ή όσα υπάρχουν στο CSV (το μικρότερο από τα δύο)
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
    ylims!(ax, 0, n_agents_guess)              # όριο y = μέγιστος αριθμός agents ανά frame

    xt = collect(0.0:0.5:3.0)                  # θέσεις ticks στον άξονα x
    ax.xticks = xt
    ax.xtickformat = mytickfmt
    ax.xticklabelrotation = pi/4               # περιστροφή των ετικετών του x κατά 45°

    # Overlay χρόνου
    frame_obs = Observable(0)                  # τρέχον frame για το label χρόνου
    lbl = Label(fig, @lift("Time elapsed = $((($frame_obs-1)*dt)) s"),
                fontsize = 16, padding = (6,10,6,10), halign = :left)
    fig[1,1, TopLeft()] = lbl

    # --- Τρία layers μπαρών: εσωτερικά + 0 + 3 ---
    y_inner = Observable(zeros(Float64, length(centers_inner)))  # (0,3) ανά 0.5
    y0      = Observable([0.0])                                  # bin για 0
    y3      = Observable([0.0])                                  # bin για TL ≥ 3

    # Χρώματα για τα εσωτερικά bins ανά TL ζώνη (ίδια λογική με DStarLite - Scenario 3.jl)
    inner_cols = [ c for x in centers_inner for c in
        (x < 1.0  ? (:dodgerblue,) :          # (0,1)
        x < 2.0  ? (:gold,)       :          # [1,2)
                 (:orange,)) ]            # [2,3)

    # εσωτερικές μπάρες (vector χρωμάτων)
    barplot!(ax, centers_inner, y_inner; width = bin_w_inner, color = inner_cols, strokewidth = 0)

    # άκρες: TL = 0 → γκρι, TL ≥ 3 → κόκκινο
    barplot!(ax, [x0_pos], y0; width = bin_w_edge, color = :gray35,  strokewidth = 0)
    barplot!(ax, [x3_pos], y3; width = bin_w_edge, color = :crimson, strokewidth = 0)

    # --- Εγγραφή βίντεο (αλλάζουν μόνο οι Υ-τιμές) ---
    out_file = joinpath(folder, "AStar_SCENARIO_3_hist_$(n_agents)_$(seed_str)$(run_export_tag)_$(run_timestamp).mp4")   # αρχείο εξόδου του ιστογράμματος
    record(fig, out_file, 1:T_play; framerate = 30) do frame
        frame_obs[] = frame
        tl = Vector(df[df.step .== frame, :toxicload])    # τοξικά φορτία όλων των agents στο συγκεκριμένο frame

        count0       = count(==(0.0), tl)                 # ακριβώς 0 (όπως DStarLite)
        count3       = count(>=(3.0), tl)                 # TL ≥ 3 (περιλαμβάνει και τους agents με TL > 3)
        counts_inner = bin_counts_open(tl, edges_inner)   # μόνο (0,3) ανά 0.5

        # ενημέρωση των υψών των μπαρών (το Makie ξανασχεδιάζει αυτόματα)
        y_inner[] = Float64.(counts_inner)
        y0[]      = [Float64(count0)]
        y3[]      = [Float64(count3)]

    end

    println("Βίντεο με ξεχωριστά bins για 0 & 3 και εσωτερικά ανά 0.5 σώθηκε ως: $out_file")

end

begin
    # καταγραφή της εκτέλεσης (αρχεία εξόδου, χρόνος pathfinding, timestamp) στο RunHistory
    append_run_history(
        scenario = "SCENARIO 3",
        runner_script = "SCENARIOS/SCENARIO 3/AStar - Scenario 3 - SLAB.jl",
        csv_file = csv_file,
        mp4_file = video_file,
        hist_mp4_file = make_histogram ? out_file : "",  # κενό αν δεν δημιουργήθηκε ιστόγραμμα
        pathfinding_time = total_path_planning,
        run_timestamp = run_timestamp,
    )
end
