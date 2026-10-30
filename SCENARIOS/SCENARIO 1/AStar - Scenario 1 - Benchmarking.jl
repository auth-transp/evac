begin   # Φόρτωση των απαραίτητων βιβλιοθηκών
    using Agents  # πλαίσιο agent-based μοντελοποίησης (ABM)
    using Agents.Pathfinding  # αλγόριθμοι εύρεσης διαδρομής (AStar, PenaltyMap, MaxDistance, ...)
    using Dates  # ημερομηνία/ώρα για το timestamp του αρχείου εξόδου
    using FileIO: load  # φόρτωση εικόνων (heightmap, concentration maps)
    using XLSX  # εγγραφή/ανάγνωση αρχείων Excel (.xlsx) για τα αποτελέσματα
    using ImageMagick  # υποστήριξη μορφών εικόνας (jpg, bmp)
    using Images  # επεξεργασία εικόνων (channelview, ...)
    using Random  # γεννήτριες τυχαίων αριθμών (MersenneTwister, RandomDevice)
    using Profile  # profiler για εντοπισμό σημείων καθυστέρησης
    using BenchmarkTools  # @benchmark για στατιστικές μετρήσεις χρόνου
end

# Ορισμός του agent: συνεχής 2D agent (pos, vel προστίθενται αυτόματα από το ContinuousAgent). Δεν υπάρχει πεδίο path· η διαδρομή κρατείται από τον pathfinder
Agents.@agent struct AgentEscapes(ContinuousAgent{2, Float64})
    age::Float64  # ηλικία (έτη)
    mass::Float64  # μάζα (kg), χρησιμοποιείται στις ελαστικές συγκρούσεις
    toxicload::Float64  # συνολικό τρέχον τοξικό φορτίο (άθροισμα TL1+TL2+TL3)
    pathX::Vector{Float64}  # ιστορικό x θέσης
    pathY::Vector{Float64}  # ιστορικό y θέσης
    TL1::Vector{Float64}  # ιστορικό τοξικού φορτίου για το 1ο σύμπτωμα (οσμή)
    TL2::Vector{Float64}  # ιστορικό τοξικού φορτίου για το 2ο σύμπτωμα (ερεθισμός)
    TL3::Vector{Float64}  # ιστορικό τοξικού φορτίου για το 3ο σύμπτωμα (οίδημα)
end

begin   # Heightmap μόνο για χώρο / pathfinding — χωρίς CM στο κόστος A*
    heightmap_data = load("NADEEN/Maps/Qatargas Map.jpg")  # φόρτωση της εικόνας του χάρτη
    heightmap_data = permutedims(channelview(heightmap_data), [2, 3, 1])[:, :, 1]  # μετατροπή σε πίνακα (γραμμές, στήλες) και κράτημα του 1ου καναλιού
    global heightmap = floor.(Int, convert.(Float64, heightmap_data) * 255)  # κλιμάκωση σε ακέραιες τιμές 0..255
    heightmap = 255 .- heightmap  # αντιστροφή τιμών: το λευκό γίνεται χαμηλή τιμή (0), το μαύρο υψηλή (255)

    # Δυναμικά CM μόνο για TL (CM1..CM7), όπως Scenario 2 / 3 benchmark
    const NUM_CMS = 7  # πλήθος concentration maps (CM1..CM7)
    cm_list = Vector{Array{Float64,2}}(undef, NUM_CMS)  # λίστα που κρατά τους 7 χάρτες συγκέντρωσης
    for k in 1:NUM_CMS  # για κάθε χάρτη k = 1..7
        fn = joinpath("Concentration Maps", string(k) * ".bmp")  # διαδρομή αρχείου k.bmp
        img = load(fn)  # φόρτωση εικόνας
        img = permutedims(channelview(img), [2, 3, 1])[:, :, 1]  # ίδια μετατροπή σε πίνακα όπως στο heightmap
        cm_list[k] = convert.(Float64, img) .* 255.0  # κλιμάκωση τιμών στο 0..255 (Float64)
        @assert size(cm_list[k]) == size(heightmap) "Concentration map $k size mismatch with heightmap"  # έλεγχος ότι ο χάρτης έχει τις ίδιες διαστάσεις με το heightmap (αλλιώς σφάλμα)
    end
    global tl_penalty_map = copy(cm_list[1])  # το tl_penalty_map (τρέχον CM για το TL) ξεκινά ως αντίγραφο του CM1· δεν επηρεάζει το κόστος του A*
end

begin   # Παράμετροι μοντέλου (ίδιες με Scenario 2 A* benchmark)
    const METERS_TO_PIXELS = 723.37 / 2500.0  # συντελεστής μετατροπής μέτρων σε pixels του χάρτη
    dt = 1.0  # διακριτό χρονικό βήμα ανά επανάληψη του μοντέλου
    n_agents = 5  # πλήθος agents ανά εκτέλεση benchmark

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
dests = [(500.0, 854.0), (120.0, 248.0)]
# συνεχής (μη περιοδικός) χώρος με διαστάσεις ίσες με του heightmap
space = ContinuousSpace(size(heightmap); periodic = false, spacing = 1)

const goal_radius = 10.0  # ακτίνα (σε pixels) γύρω από μια έξοδο μέσα στην οποία ο agent θεωρείται ότι έφτασε
# Επιστρέφει true αν η θέση pos απέχει το πολύ radius από κάποια από τις εξόδους dests
function at_goal(pos, dests, radius = goal_radius)
    px, py = Float64(pos[1]), Float64(pos[2])  # συντεταγμένες της θέσης ως Float64
    return minimum(hypot(px - d[1], py - d[2]) for d in dests) <= radius  # ελάχιστη απόσταση (hypot) από τις εξόδους ≤ radius
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
    speed = 1.35  # βασική ταχύτητα περπατήματος (1.35 m/s)
    if 0 < person.toxicload <= 1  # για 0 < TL ≤ 1
        speed = 1.35 * exp(0.393 * person.toxicload)  # εκθετική σχέση ταχύτητας–TL
    elseif 1 < person.toxicload < 3  # για 1 < TL < 3
        speed = -1.78 * log(person.toxicload) + 2.063  # λογαριθμική σχέση ταχύτητας–TL (μείωση ταχύτητας)
    elseif person.toxicload >= 3  # για TL ≥ 3: ο agent ακινητοποιείται
        speed = 0.0  # μηδενική ταχύτητα
    end
    move_along_route!(person, model, model.pathfinderPM, speed * METERS_TO_PIXELS, model.dt)  # κίνηση κατά μήκος της διαδρομής του A* με ταχύτητα σε pixels/s (m/s × METERS_TO_PIXELS) για χρόνο dt
    push!(person.pathX, person.pos[1])  # καταγραφή της νέας x θέσης
    push!(person.pathY, person.pos[2])  # καταγραφή της νέας y θέσης
end

# Βήμα του μοντέλου: ελαστικές συγκρούσεις (με βάση τη μάζα) μεταξύ των πλησιέστερων ζευγών agents σε απόσταση ≤ 0.012
function model_step!(model)
    for (a1, a2) in interacting_pairs(model, 0.012, :nearest)  # ζεύγη agents σε μικρή απόσταση (πλησιέστερος γείτονας)
        elastic_collision!(a1, a2, :mass)  # ελαστική σύγκρουση με βάση τις μάζες
    end
end

# Υπολογίζει τους πίνακες του μοντέλου τοξικού φορτίου: από τα όρια συγκέντρωσης–χρόνου έκθεσης (AEGL) για τα 3 συμπτώματα βγάζει τους εκθέτες power-law (Balpha), τους χρόνους (Btime) και τις συγκεντρώσεις (Brho)
function setupToxic()
    Atime = [0.0, 2.0, 5.0, 8.0, 15.0, 30.0]  # χρόνοι έκθεσης σε λεπτά
    Arho = zeros(3, 6)  # συγκεντρώσεις ανά σύμπτωμα (3 γραμμές) και χρόνο (6 στήλες)
    Arho[1, 2:6] = [2.87, 2.33, 2.09, 1.81, 1.55]  # 1ο σύμπτωμα: οσμή
    Arho[2, 2:6] = [202.38, 164.38, 147.74, 128.10, 109.45]  # 2ο σύμπτωμα: ερεθισμός
    Arho[3, 2:6] = [286.71, 232.87, 209.30, 181.47, 155.05]  # 3ο σύμπτωμα: οίδημα
    MW = 34  # μοριακό βάρος H2S (g/mol)
    Arho *= MW / 24.04  # μετατροπή ppm σε mg/m^3
    Arho = Arho'  # αναστροφή πίνακα (γίνεται 6×3)
    Atime = Atime * 60  # μετατροπή χρόνων σε δευτερόλεπτα
    taumin = 200.0  # ελάχιστος χρόνος έκθεσης (s)
    taumax = 86400.0  # μέγιστος χρόνος έκθεσης (s)
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

const BENCHMARK_MAX_RUNS = 100  # πλήθος εκτελέσεων του benchmark (εκτελούνται όλες, χωρίς πρόωρο τερματισμό)
const BENCHMARK_T_STEPS = 2500  # βήματα προσομοίωσης ανά εκτέλεση (2500 με dt = 1 s)
const BENCHMARK_TL_CAP_PER_AGENT = 3.0  # άνω όριο του TL ανά agent κατά την αναφορά των αποτελεσμάτων
# CM switch times (s) — Thesis, Table 7: 7 CMs, the last switch happens at 1752 s ≈ 70% of T = 2500 s.
const CM_SWITCH_TIMES = [0.0, 292.0, 584.0, 876.0, 1168.0, 1460.0, 1752.0]  # χρόνοι έναρξης κάθε CM (s)
@assert length(CM_SWITCH_TIMES) == NUM_CMS  # έλεγχος ότι υπάρχει ένας χρόνος ανά CM

# Μία εκτέλεση του benchmark: δημιουργεί νέο μοντέλο με A* (κόστος μόνο από το heightmap), τοποθετεί τους agents, υπολογίζει τις αρχικές διαδρομές και τρέχει T βήματα με εναλλαγή των CM για το TL. Επιστρέφει χρόνους pathfinding/προσομοίωσης, seed και μετρικές TL
"""
    run_one_benchmark()

Scenario 1: A* χρησιμοποιεί μόνο `heightmap` στο `PenaltyMap` (χωρίς CM στο pathfinding).
TL accumulation uses dynamic CM1→CM7 switching on `tl_penalty_map` (same schedule as Scenario 2 benchmark).
No path replanning; `incremental_pf` stays zero for spreadsheet parity.
"""
function run_one_benchmark()
    seed = rand(Random.RandomDevice(), UInt32)  # τυχαίο seed από τη γεννήτρια του λειτουργικού σε κάθε εκτέλεση
    rng_run = MersenneTwister(seed)  # rng της συγκεκριμένης εκτέλεσης

    # Μόνο terrain στο κόστος — όχι concentration map
    local_cost_metric_obj = PenaltyMap(heightmap, MaxDistance{2}())  # μετρική κόστους: μόνο heightmap, με βασική μετρική απόστασης MaxDistance{2}
    local_pathfinderPM = AStar(space; walkmap = walkmap, cost_metric = local_cost_metric_obj)  # νέος A* pathfinder για αυτήν την εκτέλεση
    local_properties = (  # ιδιότητες του μοντέλου της εκτέλεσης
        pathfinderPM = local_pathfinderPM,  # ο pathfinder
        heightmap = heightmap,  # το heightmap
        dt = dt,  # χρονικό βήμα
        speed_range = speed_range,  # εύρος ταχυτήτων
        goal = dests,  # οι έξοδοι διαφυγής (στόχοι)
    )

    model_run = StandardABM(  # νέο μοντέλο ABM για αυτήν την εκτέλεση
        AgentEscapes,  # τύπος agent
        space;  # ο συνεχής χώρος
        rng = rng_run,  # η γεννήτρια τυχαίων αριθμών της εκτέλεσης
        properties = local_properties,  # οι ιδιότητες του μοντέλου
        agent_step! = agent_step!,  # βήμα κάθε agent
        model_step! = model_step!,  # βήμα του μοντέλου (συγκρούσεις)
    )

    for _ in 1:n_agents  # δημιουργία των agents
        age = rand(abmrng(model_run)) * (age_range[2] - age_range[1]) + age_range[1]  # τυχαία ηλικία μέσα στο age_range
        mass = rand(abmrng(model_run)) * (mass_range[2] - mass_range[1]) + mass_range[1]  # τυχαία μάζα μέσα στο mass_range
        vel = Tuple(rand(abmrng(model_run), 2) .* (speed_range[2] - speed_range[1]) .+ speed_range[1])  # τυχαία αρχική ταχύτητα (2 συνιστώσες) μέσα στο speed_range
        max_attempts = 1000  # μέγιστος αριθμός προσπαθειών εύρεσης βατής θέσης εκκίνησης
        attempts = 0  # μετρητής προσπαθειών
        pos = nothing  # θέση εκκίνησης (nothing μέχρι να βρεθεί βατή)
        while attempts < max_attempts  # επανάληψη μέχρι να βρεθεί βατή θέση ή να εξαντληθούν οι προσπάθειες
            candidate_pos = Tuple((rand(abmrng(model_run), floor.(ag_range_y)), rand(abmrng(model_run), floor.(ag_range_x))))  # τυχαία υποψήφια θέση μέσα στις περιοχές ag_range
            pos_int = floor.(Int, candidate_pos)  # ακέραιες συντεταγμένες cell της υποψήφιας θέσης
            if 1 <= pos_int[1] <= size(walkmap, 1) && 1 <= pos_int[2] <= size(walkmap, 2) && walkmap[pos_int[1], pos_int[2]]  # αποδοχή αν το cell είναι μέσα στον χάρτη και βατό στο walkmap
                pos = candidate_pos  # αποδοχή της θέσης
                break  # έξοδος από τον βρόχο
            end
            attempts += 1  # αποτυχημένη προσπάθεια: αύξηση του μετρητή
        end
        if pos === nothing  # αν δεν βρέθηκε βατή θέση
            println("Warning: Could not find valid walkable spawn for agent (seed=$seed)")  # προειδοποίηση στον χρήστη
            continue  # παράλειψη αυτού του agent
        end
        add_agent!(  # προσθήκη agent με ορίσματα: vel, ηλικία, μάζα, αρχικό toxicload (0.0), ιστορικά pathX/pathY και TL1..TL3 που ξεκινούν από 0
            pos, AgentEscapes, model_run,
            vel, age, mass, 0.0,
            [pos[1]], [pos[2]],
            [0.0], [0.0], [0.0]
        )
    end

    initial_pf_time = @elapsed begin  # μέτρηση χρόνου της αρχικής εύρεσης διαδρομής όλων των agents
        for a in allagents(model_run)  # για κάθε agent
            plan_best_route!(a, dests, local_pathfinderPM)  # υπολογισμός της καλύτερης διαδρομής προς τις εξόδους
        end
    end

    incremental_pf_time = 0.0  # χρόνος επαναπρογραμματισμού διαδρομής: μένει 0 (δεν γίνεται replanning)
    sim_step_time = 0.0  # σωρευτικός χρόνος των βημάτων προσομοίωσης
    current_map_idx = 1  # δείκτης του τρέχοντος CM
    tl_map_switch_count = 0  # πλήθος εναλλαγών CM
    tl_penalty_map .= cm_list[current_map_idx]  # φόρτωση του CM1 στο tl_penalty_map
    for step_idx in 1:BENCHMARK_T_STEPS  # βρόχος στα βήματα προσομοίωσης
        t_sim = (step_idx - 1) * dt  # χρόνος προσομοίωσης (s) στο συγκεκριμένο βήμα
        new_idx = searchsortedlast(CM_SWITCH_TIMES, t_sim)  # δείκτης του τελευταίου CM που έχει ήδη ξεκινήσει
        if new_idx != current_map_idx  # αν άλλαξε ο χάρτης
            current_map_idx = new_idx  # ενημέρωση του δείκτη
            tl_map_switch_count += 1  # αύξηση του μετρητή εναλλαγών
            tl_penalty_map .= cm_list[current_map_idx]  # φόρτωση του νέου CM στο tl_penalty_map
        end
        sim_step_time += @elapsed step!(model_run, 1)  # ένα βήμα του μοντέλου με μέτρηση χρόνου
    end
    simulation_time = sim_step_time  # συνολικός χρόνος προσομοίωσης

    agents_sorted = sort(collect(allagents(model_run)), by = a -> a.id)  # agents ταξινομημένοι κατά id
    sum_TL_capped = 0.0  # άθροισμα TL με άνω όριο ανά agent
    per_agent_TL_capped = Float64[]  # TL με άνω όριο ανά agent
    agent_ids = Int[]  # id των agents
    for a in agents_sorted  # για κάθε agent
        tl_cap = min(a.toxicload, BENCHMARK_TL_CAP_PER_AGENT)  # TL περιορισμένο στο BENCHMARK_TL_CAP_PER_AGENT
        sum_TL_capped += tl_cap  # προσθήκη στο συνολικό άθροισμα
        push!(per_agent_TL_capped, tl_cap)  # αποθήκευση του TL του agent
        push!(agent_ids, a.id)  # αποθήκευση του id
    end

    return initial_pf_time, incremental_pf_time, simulation_time, seed, sum_TL_capped, current_map_idx, tl_map_switch_count, per_agent_TL_capped, agent_ids  # επιστροφή χρόνων, seed και μετρικών TL της εκτέλεσης
end

# Καθαρίζει τα δεδομένα του Profile, τρέχει μία εκτέλεση του benchmark και τυπώνει σύνοψη (χρόνοι, TL, εναλλαγές CM)· η εκτέλεση δεν περιβάλλεται από @profile, άρα δεν συλλέγονται δείγματα profiling
function profile_one_benchmark()
    Profile.clear()  # καθαρισμός προηγούμενων δεδομένων profiling
    init_t, incr_t, sim_t, run_seed, sum_cap, final_map_idx, switch_count, _, _ = run_one_benchmark()  # εκτέλεση του benchmark (τα στοιχεία ανά agent αγνοούνται)
    println("Profiled run (seed=$run_seed): initial PF = $(round(init_t; digits=4)) s, incremental PF = $(round(incr_t; digits=4)) s, simulation = $(round(sim_t; digits=4)) s, sum TL (capped max $(BENCHMARK_TL_CAP_PER_AGENT) per agent) = $(round(sum_cap; digits=4))")  # σύνοψη χρόνων και αθροίσματος TL
    println("TL CM sequence summary: final map index = $final_map_idx, switches = $switch_count")  # σύνοψη της εναλλαγής των CM
    println("Run Profile.print() or open profiler to inspect hotspots.")  # υπόδειξη για την επιθεώρηση του profile
end

# Μετρά την run_one_benchmark() με το BenchmarkTools (samples δείγματα), τυπώνει και επιστρέφει το αποτέλεσμα
function benchmark_run_one_benchmark(; samples::Int = 10)
    println("Benchmarking run_one_benchmark() with BenchmarkTools (samples = $samples)...")  # μήνυμα έναρξης
    result = @benchmark run_one_benchmark() samples = samples  # στατιστική μέτρηση του χρόνου εκτέλεσης
    println(result)  # εκτύπωση αποτελέσματος
    return result  # επιστροφή αποτελέσματος
end

initial_pf_times = Float64[]  # χρόνοι αρχικού pathfinding ανά εκτέλεση
incremental_pf_times = Float64[]  # χρόνοι επαναπρογραμματισμού ανά εκτέλεση
simulation_times = Float64[]  # χρόνοι προσομοίωσης ανά εκτέλεση
sum_TL_capped_values = Float64[]  # άθροισμα TL (με όριο) ανά εκτέλεση
final_tl_map_indices = Int[]  # τελικός δείκτης CM ανά εκτέλεση
tl_map_switch_counts = Int[]  # πλήθος εναλλαγών CM ανά εκτέλεση
seeds = UInt32[]  # seeds των εκτελέσεων
agent_tl_runs = Int[]  # στήλη Run για τον πίνακα ανά agent
agent_tl_ids = Int[]  # στήλη AgentID για τον πίνακα ανά agent
agent_tl_values = Float64[]  # στήλη TL (με όριο) για τον πίνακα ανά agent

println("Benchmark: Scenario 1 with A* (PenaltyMap = heightmap only; TL via CM1..CM7). Runs = $BENCHMARK_MAX_RUNS.")  # μήνυμα έναρξης του benchmark
for run_id in 1:BENCHMARK_MAX_RUNS  # κύριος βρόχος: BENCHMARK_MAX_RUNS εκτελέσεις
    t_init, t_incr, t_sim, run_seed, sum_TL_capped, final_tl_map_idx, tl_map_switch_count, per_agent_TL_capped, agent_ids = run_one_benchmark()  # μία εκτέλεση και λήψη των μετρικών της
    push!(initial_pf_times, t_init)  # αποθήκευση αρχικού χρόνου pathfinding
    push!(incremental_pf_times, t_incr)  # αποθήκευση χρόνου επαναπρογραμματισμού
    push!(simulation_times, t_sim)  # αποθήκευση χρόνου προσομοίωσης
    push!(sum_TL_capped_values, sum_TL_capped)  # αποθήκευση αθροίσματος TL
    push!(final_tl_map_indices, final_tl_map_idx)  # αποθήκευση τελικού δείκτη CM
    push!(tl_map_switch_counts, tl_map_switch_count)  # αποθήκευση πλήθους εναλλαγών CM
    push!(seeds, run_seed)  # αποθήκευση seed
    for k in eachindex(per_agent_TL_capped)  # για κάθε agent της εκτέλεσης
        push!(agent_tl_runs, run_id)  # αριθμός εκτέλεσης
        push!(agent_tl_ids, agent_ids[k])  # id του agent
        push!(agent_tl_values, per_agent_TL_capped[k])  # TL του agent (με όριο)
    end

    n = length(initial_pf_times)  # πλήθος εκτελέσεων μέχρι τώρα
    avg_initial_pf = sum(initial_pf_times) / n  # τρέχων μέσος αρχικού pathfinding
    avg_incremental_pf = sum(incremental_pf_times) / n  # τρέχων μέσος επαναπρογραμματισμού
    avg_simulation = sum(simulation_times) / n  # τρέχων μέσος χρόνου προσομοίωσης

    println("  Run $run_id (seed=$run_seed): initial PF = $(round(t_init; digits=4)) s, incremental PF = $(round(t_incr; digits=4)) s, simulation = $(round(t_sim; digits=4)) s, sum TL capped = $(round(sum_TL_capped; digits=4))  (avg total PF = $(round(avg_initial_pf + avg_incremental_pf; digits=4)) s)")  # εκτύπωση αποτελεσμάτων της εκτέλεσης και του τρέχοντος μέσου συνολικού pathfinding
end

n_runs = length(initial_pf_times)  # συνολικό πλήθος εκτελέσεων
avg_initial_pf_final = sum(initial_pf_times) / n_runs  # τελικός μέσος αρχικού pathfinding
avg_incremental_pf_final = sum(incremental_pf_times) / n_runs  # τελικός μέσος επαναπρογραμματισμού
avg_pathfinding_final = avg_initial_pf_final + avg_incremental_pf_final  # τελικός μέσος συνολικού pathfinding
avg_simulation_final = sum(simulation_times) / n_runs  # τελικός μέσος χρόνου προσομοίωσης

run_timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")  # χρονική σήμανση για το όνομα του αρχείου Excel
xlsx_path = "SCENARIOS/SCENARIO 1/Simulation Results/AStar $(n_agents) Agents Benchmarking_$(run_timestamp).xlsx"  # διαδρομή του αρχείου Excel
run_ids = collect(1:n_runs)  # αριθμοί εκτελέσεων 1..n_runs
columns_data = [run_ids, seeds, initial_pf_times, incremental_pf_times, simulation_times, sum_TL_capped_values, final_tl_map_indices, tl_map_switch_counts]  # δεδομένα των στηλών του φύλλου Runs
column_names = ["Run", "Seed", "InitialPathfindingTime_s", "IncrementalPathfindingTime_s", "SimulationTime_s", "SumTL_capped", "FinalTLMapIndex", "TLMapSwitchCount"]  # ονόματα των στηλών του φύλλου Runs
agent_columns_data = [agent_tl_runs, agent_tl_ids, agent_tl_values]  # δεδομένα των στηλών του φύλλου AgentTL
agent_column_names = ["Run", "AgentID", "TL_capped_max3"]  # ονόματα των στηλών του φύλλου AgentTL
XLSX.writetable(xlsx_path;  # εγγραφή του αρχείου Excel με δύο φύλλα (Runs, AgentTL), με αντικατάσταση αν υπάρχει
    Runs=(columns_data, column_names),
    AgentTL=(agent_columns_data, agent_column_names),
    overwrite=true,
)
XLSX.openxlsx(xlsx_path, mode = "rw") do xf  # άνοιγμα του αρχείου για προσθήκη γραμμών σύνοψης στο φύλλο Runs
    sh = xf["Runs"]  # το φύλλο Runs
    sh[n_runs + 2, 1] = "Number of runs"  # γραμμή: πλήθος εκτελέσεων
    sh[n_runs + 2, 2] = n_runs
    sh[n_runs + 3, 1] = "Average initial pathfinding time (s)"  # γραμμή: μέσος αρχικός χρόνος pathfinding
    sh[n_runs + 3, 2] = avg_initial_pf_final
    sh[n_runs + 4, 1] = "Average incremental pathfinding time (s)"  # γραμμή: μέσος χρόνος επαναπρογραμματισμού
    sh[n_runs + 4, 2] = avg_incremental_pf_final
    sh[n_runs + 5, 1] = "Average total pathfinding time (s)"  # γραμμή: μέσος συνολικός χρόνος pathfinding
    sh[n_runs + 5, 2] = avg_pathfinding_final
    sh[n_runs + 6, 1] = "Average simulation time (s)"  # γραμμή: μέσος χρόνος προσομοίωσης
    sh[n_runs + 6, 2] = avg_simulation_final
end

println("Benchmark complete. Results written to $xlsx_path")  # μήνυμα ολοκλήρωσης
println("  Total runs: $n_runs | Avg initial PF: $(round(avg_initial_pf_final; digits=4)) s | Avg incremental PF: $(round(avg_incremental_pf_final; digits=4)) s | Avg simulation: $(round(avg_simulation_final; digits=4)) s")  # σύνοψη μέσων χρόνων
