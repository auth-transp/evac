begin   # Φόρτωση των απαραίτητων βιβλιοθηκών
    using Agents  # πλαίσιο agent-based μοντελοποίησης (ABM)
    using Agents.Pathfinding  # αλγόριθμοι εύρεσης διαδρομής (AStar, PenaltyMap, MaxDistance, ...)
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
    using DelimitedFiles  # ανάγνωση/εγγραφή αρχείων με διαχωριστές
    using Observables  # reactive μεταβλητές (Observable) για το animation
    using Makie  # βιβλιοθήκη γραφικών
    using CSV  # εγγραφή αρχείων CSV

    # φορτώνει τη συνάρτηση append_run_history που καταγράφει κάθε εκτέλεση στο RunHistory
    include(joinpath(@__DIR__, "..", "append_run_history.jl"))
end


# Ορισμός του agent: συνεχής 2D agent (pos, vel προστίθενται αυτόματα από το ContinuousAgent). Δεν υπάρχει πεδίο path· η διαδρομή κρατείται από τον pathfinder
@agent struct AgentEscapes(ContinuousAgent{2, Float64})
    age::Float64  # ηλικία (έτη)
    mass::Float64  # μάζα (kg), χρησιμοποιείται στις ελαστικές συγκρούσεις
    toxicload::Float64  # συνολικό τρέχον τοξικό φορτίο (άθροισμα TL1+TL2+TL3)
    pathX::Vector{Float64}  # ιστορικό x θέσης (για visualization)
    pathY::Vector{Float64}  # ιστορικό y θέσης (για visualization)
    TL1::Vector{Float64}  # ιστορικό τοξικού φορτίου για το 1ο σύμπτωμα (οσμή)
    TL2::Vector{Float64}  # ιστορικό τοξικού φορτίου για το 2ο σύμπτωμα (ερεθισμός)
    TL3::Vector{Float64}  # ιστορικό τοξικού φορτίου για το 3ο σύμπτωμα (οίδημα)
end


begin   # Heightmap για χώρο / A* · CM1..CM7 μόνο για TL και οπτικό overlay (όχι στο pathfinder)

    heightmap_data = load("NADEEN/Maps/Qatargas Map.jpg")  # φόρτωση της εικόνας του χάρτη
    heightmap_data = permutedims(channelview(heightmap_data), [2, 3, 1])[:, :, 1]  # μετατροπή σε πίνακα (γραμμές, στήλες) και κράτημα του 1ου καναλιού
    global heightmap = floor.(Int, convert.(Float64, heightmap_data) * 255)  # κλιμάκωση σε ακέραιες τιμές 0..255
    heightmap = 255 .- heightmap  # αντιστροφή τιμών: το λευκό γίνεται χαμηλή τιμή (0), το μαύρο υψηλή (255)

    const NUM_CMS = 7  # πλήθος concentration maps (CM1..CM7)
    global cm_list = Vector{Array{Float64,2}}(undef, NUM_CMS)  # λίστα που κρατά τους 7 χάρτες συγκέντρωσης (χρησιμοποιούνται μόνο για το τοξικό φορτίο και το overlay)
    for k in 1:NUM_CMS  # για κάθε χάρτη k = 1..7
        fn = joinpath("Concentration Maps", string(k) * ".bmp")  # διαδρομή αρχείου k.bmp
        img = load(fn)  # φόρτωση εικόνας
        img = permutedims(channelview(img), [2, 3, 1])[:, :, 1]  # ίδια μετατροπή σε πίνακα όπως στο heightmap
        cm_list[k] = convert.(Float64, img) .* 255.0  # κλιμάκωση τιμών στο 0..255 (Float64)
        @assert size(cm_list[k]) == size(heightmap) "Concentration map $k size mismatch with heightmap"  # έλεγχος ότι ο χάρτης έχει τις ίδιες διαστάσεις με το heightmap (αλλιώς σφάλμα)
    end
    global tl_penalty_map = copy(cm_list[1])  # το tl_penalty_map (τρέχον CM για το TL) ξεκινά ως αντίγραφο του CM1· δεν επηρεάζει το κόστος του A*
end


begin   # Αρχικοποίηση των παραμέτρων του μοντέλου
    # Time–speed correlation: distance per step = speed × dt (in space units). — αντιστοιχία χρόνου–ταχύτητας: απόσταση ανά βήμα = ταχύτητα × dt
    # Treat dt as "time per step" (e.g. 1 = 1 second). Map scale: 1250 m ≈ 361.7 px. — το dt ως «χρόνος ανά βήμα» και η κλίμακα του χάρτη
    const METERS_TO_PIXELS = 723.37 / 2500.0  # συντελεστής μετατροπής μέτρων σε pixels του χάρτη
    dt = 1.   ## discrete timestep each iteration of the model — διακριτό χρονικό βήμα ανά επανάληψη του μοντέλου
    seed = 123  ## seed for random number generator — seed της γεννήτριας τυχαίων αριθμών
    run_timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")  # χρονική σήμανση της εκτέλεσης (για τα ονόματα αρχείων εξόδου)
    n_agents = 50  # πλήθος agents
    age_range = (22, 60)  # εύρος ηλικίας (έτη)
    speed_range = (4.0, 7.0)  # εύρος για την τυχαία αρχική ταχύτητα vel κάθε agent
    mass_range = (50, 80)  # εύρος μάζας (kg)
    ag_range_y = (size(heightmap)[1] / 4):(3 * size(heightmap)[1] / 4)  # περιοχή εμφάνισης των agents: από το 1/4 έως το 3/4 της 1ης διάστασης του heightmap
    ag_range_x = (size(heightmap)[2] / 4):(3 * size(heightmap)[2] / 4)  # περιοχή εμφάνισης των agents: από το 1/4 έως το 3/4 της 2ης διάστασης του heightmap
    dims = size(heightmap)  # διαστάσεις του χάρτη
    walkmap = BitArray(trues(dims...))  # walkmap: αρχικά όλα τα cells βατά (true)

    white_threshold = 245  # κατώφλι τιμής heightmap πάνω από το οποίο το cell είναι μη βατό
    walkmap[heightmap .> white_threshold] .= false  # τα cells με τιμή > white_threshold γίνονται μη βατά (false)
end


# συντεταγμένες (x, y) των δύο εξόδων διαφυγής
dests = [(500., 854.), (120., 248.)]

# γεννήτρια τυχαίων αριθμών με σταθερό seed ώστε οι εκτελέσεις να είναι αναπαραγώγιμες
rng = MersenneTwister(seed)

## Note that the dimensions of the space do not have to correspond to the dimensions
## of the pathfinder. Discretisation is handled by the pathfinding methods
# συνεχής (μη περιοδικός) χώρος με διαστάσεις ίσες με του heightmap
space = ContinuousSpace(size(heightmap); periodic = false, spacing = 1)


begin  # Ορισμός του pathfinder A* και των ιδιοτήτων του μοντέλου
    pathfinderPM = AStar(space; walkmap = walkmap, cost_metric = PenaltyMap(heightmap, MaxDistance{2}()))  # A* pathfinder: μη βατά cells από το walkmap· κόστος μόνο από το ανάγλυφο (heightmap) με βασική μετρική απόστασης MaxDistance{2}, χωρίς concentration map
    properties = (  # ιδιότητες προσβάσιμες μέσα στο μοντέλο ως model.<όνομα>
        pathfinderPM = pathfinderPM,  # ο pathfinder
        heightmap = heightmap,  # το heightmap
        dt = dt,  # χρονικό βήμα
        speed_range = speed_range,  # εύρος ταχυτήτων
        goal = dests  # οι έξοδοι διαφυγής (στόχοι)
    )
end


# Helper: true if position is within radius of any goal (TL and movement are skipped when true)
const goal_radius = 10.0  # ακτίνα (σε pixels) γύρω από μια έξοδο μέσα στην οποία ο agent θεωρείται ότι έφτασε
# Επιστρέφει true αν η θέση pos απέχει το πολύ radius από κάποια από τις εξόδους dests
function at_goal(pos, dests, radius = goal_radius)
    return minimum(norm(pos .- d) for d in dests) ≤ radius  # ελάχιστη Ευκλείδεια απόσταση από τις εξόδους ≤ radius
end


# Βήμα ενός agent: αν δεν έχει φτάσει σε έξοδο, ενημερώνει το τοξικό φορτίο από τη συγκέντρωση στη θέση του, υπολογίζει την ταχύτητα και κινείται κατά μήκος της διαδρομής
function agent_step!(person, model)
    if at_goal(person.pos, model.goal)  # αν ο agent είναι ήδη σε έξοδο: δεν υπολογίζεται TL ούτε κίνηση
        push!(person.pathX, person.pos[1])  # καταγράφει μόνο την τρέχουσα x θέση
        push!(person.pathY, person.pos[2])  # καταγράφει μόνο την τρέχουσα y θέση
        return  # τέλος βήματος για αυτόν τον agent
    end

    grid_dims = size(tl_penalty_map)  # διαστάσεις του τρέχοντος χάρτη συγκέντρωσης
    i = clamp(Int(floor(person.pos[1])), 1, grid_dims[1])  # δείκτης γραμμής του cell από την pos[1], περιορισμένος μέσα στα όρια
    j = clamp(Int(floor(person.pos[2])), 1, grid_dims[2])  # δείκτης στήλης του cell από την pos[2], περιορισμένος μέσα στα όρια
    Ct = tl_penalty_map[i, j]  # συγκέντρωση Ct στη θέση του agent από το τρέχον CM
    TLcurrent = [person.TL1[end], person.TL2[end], person.TL3[end]]  # τελευταίες τιμές TL για τα τρία συμπτώματα
    TL = update_toxic_load(Ct, TLcurrent, model.dt)  # νέο τοξικό φορτίο μετά από έκθεση Ct για χρόνο dt

    person.toxicload = sum(TL)  # συνολικό τοξικό φορτίο = άθροισμα των τριών συμπτωμάτων
    push!(person.TL1, TL[1])  # προσθήκη στο ιστορικό TL1
    push!(person.TL2, TL[2])  # προσθήκη στο ιστορικό TL2
    push!(person.TL3, TL[3])  # προσθήκη στο ιστορικό TL3

    # --- Speed in m/s (base 1.35 m/s); pathfinder expects pixels/s so scale by METERS_TO_PIXELS --- — ταχύτητα σε m/s ανάλογα με το τοξικό φορτίο· ο pathfinder θέλει pixels/s, γι αυτό πολλαπλασιάζεται με METERS_TO_PIXELS
    speed = 1.35  # βασική ταχύτητα περπατήματος (1.35 m/s)
    if 0 < person.toxicload <= 1  # για 0 < TL ≤ 1
        speed = 1.35 * exp(0.393 * person.toxicload)  # εκθετική σχέση ταχύτητας–TL
    elseif 1 < person.toxicload < 3  # για 1 < TL < 3
        speed = -1.78 * log(person.toxicload) + 2.063  # λογαριθμική σχέση ταχύτητας–TL (μείωση ταχύτητας)
    elseif person.toxicload >= 3  # για TL ≥ 3: ο agent ακινητοποιείται
        speed = 0.0  # μηδενική ταχύτητα
    end

    move_along_route!(person, model, model.pathfinderPM, speed * METERS_TO_PIXELS, model.dt)  # κίνηση κατά μήκος της διαδρομής του A* με ταχύτητα σε pixels/s για χρόνο dt
    push!(person.pathX, person.pos[1])  # καταγραφή της νέας x θέσης
    push!(person.pathY, person.pos[2])  # καταγραφή της νέας y θέσης
end


# Βήμα του μοντέλου: ελαστικές συγκρούσεις (με βάση τη μάζα) μεταξύ των πλησιέστερων ζευγών agents σε απόσταση ≤ 0.012
function model_step!(model)
    for (a1, a2) in interacting_pairs(model, 0.012, :nearest)  # ζεύγη agents σε μικρή απόσταση (πλησιέστερος γείτονας)
        elastic_collision!(a1, a2, :mass)  # ελαστική σύγκρουση με βάση τις μάζες
    end
end


# Δημιουργία του μοντέλου ABM με τον χώρο, το rng, τις ιδιότητες και τις συναρτήσεις βήματος
model = StandardABM(
    AgentEscapes,  # τύπος agent
    space;  # ο συνεχής χώρος
    rng = rng,  # η γεννήτρια τυχαίων αριθμών
    properties = properties,  # οι ιδιότητες του μοντέλου
    agent_step! = agent_step!,  # βήμα κάθε agent
    model_step! = model_step!,  # βήμα του μοντέλου (συγκρούσεις)
)


begin  # Δημιουργία των agents και προϋπολογισμός της αρχικής διαδρομής τους
    path_planning_time = Ref(0.0)  # συσσωρευτής του συνολικού χρόνου εύρεσης διαδρομής
    for _ in 1:n_agents  # για κάθε agent
        age = rand(abmrng(model)) * (age_range[2] - age_range[1]) + age_range[1]  # τυχαία ηλικία μέσα στο age_range
        mass = rand(abmrng(model)) * (mass_range[2] - mass_range[1]) + mass_range[1]  # τυχαία μάζα μέσα στο mass_range
        vel = Tuple(rand(abmrng(model), 2) .* (speed_range[2] - speed_range[1]) .+ speed_range[1])  # τυχαία αρχική ταχύτητα (2 συνιστώσες) μέσα στο speed_range

        max_attempts = 1000  # μέγιστος αριθμός προσπαθειών εύρεσης βατής θέσης εκκίνησης
        attempts = 0  # μετρητής προσπαθειών
        pos = nothing  # θέση εκκίνησης (nothing μέχρι να βρεθεί βατή)
        while attempts < max_attempts  # επανάληψη μέχρι να βρεθεί βατή θέση ή να εξαντληθούν οι προσπάθειες
            candidate_pos = Tuple((rand(abmrng(model), floor.(ag_range_y)), rand(abmrng(model), floor.(ag_range_x))))  # τυχαία υποψήφια θέση μέσα στις περιοχές ag_range
            pos_int = floor.(Int, candidate_pos)  # ακέραιες συντεταγμένες cell της υποψήφιας θέσης
            if 1 <= pos_int[1] <= size(walkmap, 1) && 1 <= pos_int[2] <= size(walkmap, 2) && walkmap[pos_int[1], pos_int[2]]  # αποδοχή αν το cell είναι μέσα στον χάρτη και βατό στο walkmap
                pos = candidate_pos  # αποδοχή της θέσης
                break  # έξοδος από τον βρόχο
            end
            attempts += 1  # αποτυχημένη προσπάθεια: αύξηση του μετρητή
        end

        if pos === nothing  # αν δεν βρέθηκε βατή θέση
            println("Warning: Could not find valid spawn position for agent after $max_attempts attempts")  # προειδοποίηση στον χρήστη
            continue  # παράλειψη αυτού του agent
        end

        person = add_agent!(pos, AgentEscapes, model, vel, age, mass, 0., [pos[1]], [pos[2]], [0.0], [0.0], [0.0])  # προσθήκη agent με ορίσματα: vel, ηλικία, μάζα, αρχικό toxicload (0.), ιστορικά pathX/pathY και TL1..TL3 που ξεκινούν από 0
        path_planning_time[] += @elapsed plan_best_route!(person, dests, model.pathfinderPM)  # υπολογισμός της καλύτερης διαδρομής προς τις εξόδους και μέτρηση του χρόνου
    end
    println("TIMING BREAKDOWN:")  # εκτύπωση χρόνων
    println("  Pathfinding time (total): ", round(path_planning_time[]; digits=6), " s")  # συνολικός χρόνος εύρεσης διαδρομής
end


# Υπολογίζει τους πίνακες του μοντέλου τοξικού φορτίου: από τα όρια συγκέντρωσης–χρόνου έκθεσης (AEGL) για τα 3 συμπτώματα βγάζει τους εκθέτες power-law (Balpha), τους χρόνους (Btime) και τις συγκεντρώσεις (Brho)
function setupToxic()
    Atime = [0.0, 2.0, 5.0, 8.0, 15.0, 30.0] # min — χρόνοι έκθεσης σε λεπτά
    Arho = zeros(3, 6)  # συγκεντρώσεις ανά σύμπτωμα (3 γραμμές) και χρόνο (6 στήλες)
    Arho[1, 2:6] = [2.87, 2.33, 2.09, 1.81, 1.55]  # 1ο σύμπτωμα: οσμή
    Arho[2, 2:6] = [202.38, 164.38, 147.74, 128.10, 109.45]  # 2ο σύμπτωμα: ερεθισμός
    Arho[3, 2:6] = [286.71, 232.87, 209.30, 181.47, 155.05]  # 3ο σύμπτωμα: οίδημα
    MW = 34  # μοριακό βάρος H2S (g/mol)
    Arho *= MW / 24.04  # μετατροπή ppm σε mg/m^3
    Arho = Arho'  # αναστροφή πίνακα (γίνεται 6×3)
    Atime = Atime * 60  # μετατροπή χρόνων σε δευτερόλεπτα
    taumin = 200.  # ελάχιστος χρόνος έκθεσης (s)
    taumax = 86400.  # μέγιστος χρόνος έκθεσης (s)
    Brho = zeros(7, 3)  # πίνακας συγκεντρώσεων (7×3) ανά σημείο και σύμπτωμα
    Balpha = zeros(7, 3)  # πίνακας εκθετών power-law (7×3)
    rhomax = zeros(1, 3)  # μέγιστη συγκέντρωση ανά σύμπτωμα
    rhomin = zeros(1, 3)  # ελάχιστη συγκέντρωση ανά σύμπτωμα
    Btime = zeros(7, 3)  # πίνακας χρόνων (7×3)

    # αντιγραφή των πέντε AEGL σημείων (b = 2..6) στους πίνακες Brho/Btime και ορισμός των ακραίων χρόνων taumin/taumax
    for k = 1:3
        for b = 2:6  # για τα πέντε σημεία
            Brho[b, k] = Arho[b, k]  # συγκέντρωση του σημείου
            Btime[b, k] = Atime[b]  # χρόνος του σημείου
        end
        Btime[1, k] = taumin  # ο πρώτος χρόνος είναι ο taumin
        Btime[7, k] = taumax  # ο τελευταίος χρόνος είναι ο taumax
    end

    # υπολογισμός του εκθέτη power-law (Balpha) μεταξύ διαδοχικών σημείων· 0 αν οι συγκεντρώσεις είναι ίσες
    for k = 1:3
        for b = 3:6  # για τα διαδοχικά σημεία
            if Brho[b-1, k] == Brho[b, k]  # ίσες συγκεντρώσεις: εκθέτης 0 (κατώφλι)
                Balpha[b, k] = 0.0
            else
                Balpha[b, k] = log(Atime[b] / Atime[b-1]) / log(Brho[b-1, k] / Brho[b, k])  # εκθέτης από τον λόγο των λογαρίθμων χρόνου και συγκέντρωσης
            end
        end
        Balpha[2, k] = Balpha[3, k]  # επέκταση των εκθετών στα άκρα του πίνακα (b = 1, 2, 7)
        Balpha[1, k] = Balpha[2, k]
        Balpha[7, k] = Balpha[6, k]
    end

    # υπολογισμός των ακραίων συγκεντρώσεων: rhomax (για taumin) στη θέση 1 και rhomin (για taumax) στη θέση 7
    for k = 1:3
        if Balpha[3, k] == 0
            rhomax[k] = Brho[2, k]  # rhomax ίσο με τη συγκέντρωση του 2ου σημείου
            Brho[1, k] = rhomax[k]  # αποθήκευση στη θέση 1
        else
            rhomax[k] = Brho[2, k] * (Btime[2, k] / taumin)^(1 / Balpha[2, k])  # rhomax με extrapolation μέσω του power-law
            Brho[1, k] = rhomax[k]  # αποθήκευση στη θέση 1
        end

        if Brho[5, k] == Brho[6, k]  # αν οι δύο τελευταίες συγκεντρώσεις είναι ίσες
            rhomin[k] = Brho[6, k]  # rhomin ίσο με την τελευταία συγκέντρωση
            Brho[7, k] = rhomin[k]  # αποθήκευση στη θέση 7
        else
            rhomin[k] = Brho[6, k] * (Btime[6, k] / taumax)^(1 / Balpha[6, k])  # rhomin με extrapolation μέσω του power-law
            Brho[7, k] = rhomin[k]  # αποθήκευση στη θέση 7
        end
    end

    # στα τμήματα με Balpha == 0 (κατώφλι) ο χρόνος παίρνει την τιμή του προηγούμενου σημείου
    for k = 1:3
        for b = 2:7
            if Balpha[b, k] == 0
                Btime[b, k] = Btime[b-1, k]
            end
        end
    end

    # επαναυπολογισμός του Balpha μετά από κατώφλι (προηγούμενος εκθέτης 0 και τρέχων > 0)
    for k = 1:3
        for b = 3:5
            if Balpha[b-1, k] == 0 && Balpha[b, k] > 0
                Balpha[b, k] = log(Btime[b, k] / Btime[b-1, k]) / log(Brho[b-1, k] / Brho[b, k])
            end
        end
    end

    return Balpha, Btime, Brho'  # επιστρέφει τους εκθέτες, τους χρόνους και τις συγκεντρώσεις (ανάστροφος: 3×7)
end

# Υπολογισμός των πινάκων τοξικότητας μία φορά (global, χρησιμοποιούνται από την tl_rate, που καλείται από την update_toxic_load)
Balpha, Btime, Brho = setupToxic()


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


# Χρώμα του agent ανάλογα με το τοξικό φορτίο: κόκκινο (TL ≥ 3), πράσινο (TL ≤ 1), πορτοκαλί (ενδιάμεσα)
function personcolor(person::AgentEscapes)
    if person.toxicload >= 3.  # TL ≥ 3: ακινητοποιημένος
        return :red  # κόκκινο
    elseif person.toxicload <= 1.  # TL ≤ 1
        return :green  # πράσινο
    else
        return :orange  # πορτοκαλί (ενδιάμεσο TL)
    end
end


begin   # Δημιουργία animation με trails & συλλογή CSV θέσης και toxicload
    T = 2500  # συνολικά βήματα προσομοίωσης (με dt = 1 s ισοδυναμεί με 2500 s)
    # CM switch times (s) — Thesis, Table 7: 7 CMs, the last switch happens at 1752 s ≈ 70% of T = 2500 s. — χρόνοι αλλαγής χάρτη συγκέντρωσης
    cm_switch_times = [0.0, 292.0, 584.0, 876.0, 1168.0, 1460.0, 1752.0]  # έναρξη κάθε CM (s)
    @assert length(cm_switch_times) == NUM_CMS  # έλεγχος ότι υπάρχει ένας χρόνος ανά CM
    tl_map_idx = Ref(1)  # δείκτης του τρέχοντος CM
    tl_penalty_map .= cm_list[tl_map_idx[]]  # φόρτωση του CM1 στο tl_penalty_map

    fig = Figure(; size = (800, 800))  # το σχήμα του animation
    ax = Makie.Axis(fig[1, 1];  # ο άξονας με τίτλο και ίσες κλίμακες x/y
        title = "Evacuation with Toxic Trail (Scenario 1 — A*: heightmap only, TL: CM1–7)",
        aspect = DataAspect())

    heatmap!(ax, heightmap; colormap = :grays, alpha = 0.3)  # το heightmap ως ημιδιαφανές φόντο

    cm_obs = Observable(copy(tl_penalty_map))  # Observable με το τρέχον CM για το overlay
    cm_colorrange = @lift let z = $cm_obs  # δυναμικό εύρος χρωμάτων από min/max του τρέχοντος CM
        m = minimum(z)
        M = maximum(z)
        if m == M
            δ = max(1.0, abs(m) * 1e-6)  # αν ο χάρτης είναι σταθερός, διευρύνει το εύρος ώστε να μην είναι εκφυλισμένο
            (Float32(m - δ), Float32(M + δ))
        else
            (Float32(m), Float32(M))
        end
    end
    contour!(  # περιγράμματα (contour) του CM ως overlay
        ax, cm_obs;
        colormap = cgrad([:yellow, :orange, :red]),
        colorrange = cm_colorrange,
        levels = 10,
        linewidth = 1.5,
        alpha = 0.7,
    )

    goals = model.goal  # οι έξοδοι του μοντέλου
    scatter!(ax,  # σχεδίαση των εξόδων ως κόκκινους κύκλους
        getindex.(goals, 1),
        getindex.(goals, 2);
        color = (:red, 50),
        marker = :circle,
    )

    frame_obs = Observable(0)  # Observable με τον αριθμό του τρέχοντος frame

    counter_lbl = Label(  # ετικέτα που δείχνει τον χρόνο που πέρασε
        fig,
        @lift("Time elapsed = $((($frame_obs - 1) * dt)) s"),  # χρόνος = (frame − 1) × dt σε s
        fontsize = 16,
        padding = (6, 10, 6, 10),
        halign = :left
    )
    fig[1, 1, TopLeft()] = counter_lbl  # τοποθέτηση της ετικέτας πάνω αριστερά

    xs0 = Float64[]  # αρχικές x θέσεις των agents
    ys0 = Float64[]  # αρχικές y θέσεις των agents
    colors0 = Symbol[]  # αρχικά χρώματα των agents

    for a in allagents(model)  # συλλογή αρχικών θέσεων και χρωμάτων
        push!(xs0, a.pos[1])
        push!(ys0, a.pos[2])
        push!(colors0, personcolor(a))
    end

    posobs = Observable(Point2f.(xs0, ys0))  # Observable με τις θέσεις των agents
    colobs = Observable(colors0)  # Observable με τα χρώματα των agents

    lines_plots = [  # ένα αντικείμενο lines! ανά agent για το ίχνος της διαδρομής του
        lines!(ax,
            [a.pos[1]], [a.pos[2]];
            color = personcolor(a),
            linewidth = 2)
        for a in allagents(model)
    ]
    agent_scat = scatter!(ax, posobs;  # scatter με τις τρέχουσες θέσεις των agents
        color = colobs,
        markersize = 10)

    df = DataFrame(  # DataFrame για τη συλλογή θέσης και toxicload ανά βήμα και agent
        step = Int[],  # αριθμός βήματος
        agent_id = Int[],  # id του agent
        x = Float64[],  # θέση x
        y = Float64[],  # θέση y
        toxicload = Float64[]  # τοξικό φορτίο
    )

    video_file = "SCENARIOS/SCENARIO 1/Simulation Results/SCENARIO_1_$(n_agents)_$(seed)_$(run_timestamp).mp4"  # όνομα αρχείου βίντεο
    record(fig, video_file, 1:T; framerate = 30) do frame  # εγγραφή του animation για T frames, ένα βήμα προσομοίωσης ανά frame
        frame_obs[] = frame  # ενημέρωση του μετρητή frame

        t_sim = (frame - 1) * dt  # χρόνος προσομοίωσης (s) στο συγκεκριμένο frame
        new_idx = searchsortedlast(cm_switch_times, t_sim)  # δείκτης του τελευταίου CM που έχει ήδη ξεκινήσει
        if new_idx != tl_map_idx[]  # αν άλλαξε ο χάρτης
            tl_map_idx[] = new_idx  # ενημέρωση του δείκτη
            tl_penalty_map .= cm_list[new_idx]  # φόρτωση του νέου CM στο tl_penalty_map
        end
        cm_obs[] = tl_penalty_map  # ενημέρωση του overlay

        step!(model, 1)  # ένα βήμα του μοντέλου

        for (i, a) in enumerate(allagents(model))  # ενημέρωση των ιχνών διαδρομής
            lines_plots[i][1][] = Point2f.(a.pathX, a.pathY)
        end

        xs = [a.pos[1] for a in allagents(model)]  # τρέχουσες x θέσεις
        ys = [a.pos[2] for a in allagents(model)]  # τρέχουσες y θέσεις
        posobs[] = Point2f.(xs, ys)  # ενημέρωση των θέσεων του scatter
        colobs[] = [personcolor(a) for a in allagents(model)]  # ενημέρωση των χρωμάτων

        for a in allagents(model)  # καταγραφή θέσης και toxicload κάθε agent στο DataFrame
            push!(df, (
                frame,
                a.id,
                a.pos[1],
                a.pos[2],
                a.toxicload
            ))
        end
    end

    println("Το animation σώθηκε ως $video_file")  # μήνυμα ότι το animation αποθηκεύτηκε

    csv_file = "SCENARIOS/SCENARIO 1/Simulation Results/SCENARIO_1_$(n_agents)_$(seed)_$(run_timestamp).csv"  # όνομα αρχείου CSV
    CSV.write(csv_file, df)  # εγγραφή του DataFrame σε CSV
    println("Τα δεδομένα θέσης & toxicload αποθηκεύτηκαν ως $csv_file")  # μήνυμα ότι τα δεδομένα αποθηκεύτηκαν

    append_run_history(  # καταγραφή της εκτέλεσης στο RunHistory
        scenario = "SCENARIO 1",  # όνομα σεναρίου
        runner_script = "SCENARIOS/SCENARIO 1/Scenario 1.jl",  # script που εκτελέστηκε
        csv_file = csv_file,  # αρχείο CSV εξόδου
        mp4_file = video_file,  # αρχείο βίντεο εξόδου
        pathfinding_time = path_planning_time[],  # συνολικός χρόνος pathfinding
        run_timestamp = run_timestamp,  # χρονική σήμανση της εκτέλεσης
    )
end
