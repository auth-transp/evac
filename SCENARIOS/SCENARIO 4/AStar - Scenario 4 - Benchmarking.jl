begin   # Φόρτωση των απαραίτητων βιβλιοθηκών
    using Agents  # πλαίσιο agent-based μοντελοποίησης (ABM, add_agent!, step!, ...)
    using Agents.Pathfinding  # αλγόριθμοι εύρεσης διαδρομής (AStar, AbsolutePenaltyMap, MaxDistance, ...)
    using Dates  # ημερομηνία/ώρα για το timestamp του αρχείου εξόδου
    using FileIO: load  # φόρτωση εικόνων (heightmap, concentration maps)
    using XLSX  # εγγραφή των αποτελεσμάτων σε αρχείο Excel (xlsx)
    using ImageMagick  # υποστήριξη μορφών εικόνας (jpg, bmp)
    using Images  # επεξεργασία εικόνων (channelview, ...)
    using Random  # γεννήτριες τυχαίων αριθμών (MersenneTwister, RandomDevice)
    using Profile  # εργαλεία profiling (χρησιμοποιούνται στη profile_one_benchmark)
    using BenchmarkTools  # εργαλεία benchmarking (χρησιμοποιούνται στη benchmark_run_one_benchmark)
end

# Ορισμός του agent: συνεχής 2D agent (pos, vel προστίθενται αυτόματα από το ContinuousAgent)
Agents.@agent struct AgentEscapes(ContinuousAgent{2, Float64})
    age::Float64  # ηλικία (έτη)
    mass::Float64  # μάζα (kg), χρησιμοποιείται στις ελαστικές συγκρούσεις
    toxicload::Float64  # συνολικό τρέχον τοξικό φορτίο (άθροισμα TL1+TL2+TL3)
    pathX::Vector{Float64}  # ιστορικό της συντεταγμένης x του agent
    pathY::Vector{Float64}  # ιστορικό της συντεταγμένης y του agent
    TL1::Vector{Float64}  # ιστορικό τοξικού φορτίου για το 1ο σύμπτωμα (οσμή)
    TL2::Vector{Float64}  # ιστορικό τοξικού φορτίου για το 2ο σύμπτωμα (ερεθισμός)
    TL3::Vector{Float64}  # ιστορικό τοξικού φορτίου για το 3ο σύμπτωμα (οίδημα)
end

begin   # Φόρτωση heightmap και concentration maps 1..7 (ίδιο με Scenario 3 benchmark)
    heightmap_data = load("NADEEN/Maps/Qatargas Map.jpg")  # φόρτωση της εικόνας του χάρτη
    heightmap_data = permutedims(channelview(heightmap_data), [2, 3, 1])[:, :, 1]  # μετατροπή σε πίνακα (γραμμές, στήλες) και κράτημα του 1ου καναλιού
    global heightmap = floor.(Int, convert.(Float64, heightmap_data) * 255)  # κλιμάκωση σε ακέραιες τιμές 0..255
    heightmap = 255 .- heightmap  # αντιστροφή των τιμών (255 - τιμή)

    const NUM_CMS = 7  # πλήθος concentration maps (CM1..CM7)
    cm_list = Vector{Array{Float64, 2}}(undef, NUM_CMS)  # λίστα που θα κρατά τους 7 χάρτες συγκέντρωσης
    for k in 1:NUM_CMS
        fn = joinpath("Concentration Maps", string(k) * ".bmp")  # διαδρομή αρχείου k.bmp
        img = load(fn)  # φόρτωση εικόνας
        img = permutedims(channelview(img), [2, 3, 1])[:, :, 1]  # ίδια μετατροπή σε πίνακα όπως στο heightmap
        cm_list[k] = convert.(Float64, img) .* 255.0  # κλιμάκωση τιμών στο 0..255 (Float64)
    end
    # έλεγχος ότι κάθε CM έχει τις ίδιες διαστάσεις με το heightmap (αλλιώς σταματά με μήνυμα σφάλματος)
    for k in 1:NUM_CMS
        @assert size(cm_list[k]) == size(heightmap) "Concentration map $k size mismatch with heightmap"
    end
    # το penalty_map (τρέχον CM) ξεκινά ως αντίγραφο του 1ου χάρτη
    global penalty_map = copy(cm_list[1])
end

# συνδυασμένος χάρτης κόστους = ανάγλυφο/εμπόδια (heightmap) + τοξική συγκέντρωση (penalty_map)
NPM = heightmap .+ penalty_map
NPM_int = round.(Int, NPM)

begin   # Παράμετροι μοντέλου
    const METERS_TO_PIXELS = 723.37 / 2500.0  # συντελεστής μετατροπής μέτρων σε pixels του χάρτη
    dt = 1.0  # διάρκεια του χρονικού βήματος
    n_agents = 10  # πλήθος agents ανά εκτέλεση του benchmark

    age_range = (22, 60)  # εύρος τυχαίας ηλικίας (έτη)
    speed_range = (4.0, 7.0)  # εύρος της τυχαίας αρχικής ταχύτητας vel
    mass_range = (50, 80)  # εύρος τυχαίας μάζας (kg)
    ag_range_y = (size(heightmap)[1] / 4):(3 * size(heightmap)[1] / 4)  # περιοχή εμφάνισης των agents: κεντρικό μισό του χάρτη (1η διάσταση)
    ag_range_x = (size(heightmap)[2] / 4):(3 * size(heightmap)[2] / 4)  # περιοχή εμφάνισης των agents: κεντρικό μισό του χάρτη (2η διάσταση)
    dims = (size(NPM))  # διαστάσεις του χάρτη κόστους
    walkmap = BitArray(trues(dims...))  # αρχικά πλήρως βατός χάρτης βατότητας (walkmap)
    white_threshold = 245  # κατώφλι για τα «λευκά» (μη βατά) σημεία του heightmap
    walkmap[heightmap .> white_threshold] .= false  # μηδενίζει τη βατότητα όπου το heightmap ξεπερνά το κατώφλι
end

# All 16 candidate goals (fixed order; indices 1:16). Sides are subsets of these indices.
# Και οι 16 υποψήφιες έξοδοι (x, y) με σταθερή σειρά, δείκτες 1..16
const DESTS_ALL = Tuple{Float64, Float64}[
    (500., 854.), (120., 248.), (691., 750.), (351., 924.),
    (853., 659.), (795., 375.), (703., 187.), (864., 503.),
    (652., 73.), (325., 150.), (223., 198.), (473., 78.),
    (88., 493.), (145., 630.), (209., 787.), (39., 379.),
]

# Side A–D: global indices into DESTS_ALL (max 2 active points per side per run)
# Ομαδοποίηση των εξόδων σε 4 πλευρές (A-D): κάθε λίστα περιέχει καθολικούς δείκτες στο DESTS_ALL
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
Sorted by (K, indices lexicographically). Stable index → use for reproducible benchmarks.
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
    println("Benchmark: $(length(u)) valid goal combinations (K in 1:8, max 2 per side); run i uses evenly spaced combo indices (see goal_combo_index_for_run).")  # ενημερωτικό μήνυμα με το πλήθος των συνδυασμών
    u
end

# Αντιστοιχίζει το run_id (1..n_runs) σε δείκτη συνδυασμού εξόδων (1..n_combos): αν n_runs <= n_combos οι δείκτες είναι ισαπέχοντες από το 1 ως το n_combos, αλλιώς επιστρέφεται mod1(run_id, n_combos)
"""
Map benchmark run `run_id` (1…`n_runs`) to a combination index in 1…`n_combos`.

When `n_runs` ≤ `n_combos`, indices are **evenly spaced** from 1 to `n_combos` so a batch of runs
samples across the full sorted list (not only the first combinations). When `n_runs` > `n_combos`,
falls back to `mod1(run_id, n_combos)`.
"""
function goal_combo_index_for_run(run_id::Int, n_runs::Int, n_combos::Int)
    @assert 1 <= run_id <= n_runs  # έλεγχος ότι το run_id είναι έγκυρο
    @assert n_combos >= 1  # έλεγχος ότι υπάρχει τουλάχιστον ένας συνδυασμός
    n_runs > n_combos && return mod1(run_id, n_combos)  # περισσότερα runs από συνδυασμούς: κυκλική επανάληψη των συνδυασμών
    n_runs == 1 && return 1  # ένα μόνο run: χρησιμοποιεί τον πρώτο συνδυασμό (αποφυγή διαίρεσης με το 0)
    # γραμμική παρεμβολή του run_id στο 1..n_combos, στρογγυλοποίηση και περιορισμός εντός ορίων
    return clamp(
        round(Int, 1 + (run_id - 1) * (n_combos - 1) / (n_runs - 1)),
        1,
        n_combos,
    )
end

# Helper: true if position is within radius of any goal (TL and movement stop when true)
const goal_radius = 10.0  # ακτίνα (σε pixels) γύρω από την έξοδο που θεωρείται ότι ο agent «διέφυγε»
# Ελέγχει αν η θέση pos απέχει το πολύ radius από κάποια από τις εξόδους dests
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
    Ct = penalty_map[i, j]  # συγκέντρωση τοξικού στο cell του agent
    TLcurrent = [person.TL1[end], person.TL2[end], person.TL3[end]]  # τελευταίες τιμές TL των τριών συμπτωμάτων
    TL = update_toxic_load(Ct, TLcurrent, model.dt)  # νέο τοξικό φορτίο μετά από έκθεση dt
    person.toxicload = sum(TL)  # συνολικό τοξικό φορτίο
    push!(person.TL1, TL[1])  # αποθήκευση στο ιστορικό
    push!(person.TL2, TL[2])
    push!(person.TL3, TL[3])
    # Ενημέρωση της ταχύτητας ανάλογα με το τοξικό φορτίο
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
    push!(person.pathX, person.pos[1])  # καταγραφή νέας θέσης
    push!(person.pathY, person.pos[2])
end

# Βήμα του μοντέλου: ελαστικές συγκρούσεις μεταξύ γειτονικών agents (ακτίνα 0.012, βάσει μάζας)
function model_step!(model)
    for (a1, a2) in interacting_pairs(model, 0.012, :nearest)
        elastic_collision!(a1, a2, :mass)
    end
end

# Υπολογίζει τους πίνακες (Balpha, Btime, Brho) του μοντέλου τοξικού φορτίου από τα όρια AEGL του H2S:
# για κάθε σύμπτωμα (οσμή, ερεθισμός, οίδημα) δίνει συγκεντρώσεις, χρόνους και εκθέτες power-law
# που χρησιμοποιεί η tl_rate (και μέσω αυτής η update_toxic_load).
function setupToxic()
    Atime = [0.0, 2.0, 5.0, 8.0, 15.0, 30.0]  # τυποποιημένοι χρόνοι έκθεσης AEGL (λεπτά)
    Arho = zeros(3, 6)  # συγκεντρώσεις AEGL για τα 3 συμπτώματα και τους 6 χρόνους
    Arho[1, 2:6] = [2.87, 2.33, 2.09, 1.81, 1.55]  # οσμή
    Arho[2, 2:6] = [202.38, 164.38, 147.74, 128.10, 109.45]  # ερεθισμός
    Arho[3, 2:6] = [286.71, 232.87, 209.30, 181.47, 155.05]  # οίδημα (ppm)
    MW = 34  # μοριακό βάρος του H2S (g/mol)
    Arho *= MW / 24.04  # μετατροπή από ppm σε mg/m^3
    Arho = Arho'  # αναστροφή του πίνακα
    Atime = Atime * 60  # μετατροπή των χρόνων σε δευτερόλεπτα
    taumin = 200.0  # ελάχιστος χρόνος έκθεσης (s) για την εκδήλωση AEGL
    taumax = 86400.0  # μέγιστος χρόνος έκθεσης (s) για την εκδήλωση AEGL
    Brho = zeros(7, 3)  # συγκεντρώσεις ανά σημείο και σύμπτωμα
    Balpha = zeros(7, 3)  # εκθέτες power-law
    rhomax = zeros(1, 3)  # μέγιστη συγκέντρωση ανά σύμπτωμα
    rhomin = zeros(1, 3)  # ελάχιστη συγκέντρωση ανά σύμπτωμα
    Btime = zeros(7, 3)  # χρόνοι ανά σημείο και σύμπτωμα
    # αντιγραφή των πέντε AEGL σημείων (b = 2..6) στους πίνακες Brho/Btime και ορισμός των ακραίων χρόνων taumin/taumax
    for k = 1:3
        for b = 2:6
            Brho[b, k] = Arho[b, k]
            Btime[b, k] = Atime[b]
        end
        Btime[1, k] = taumin
        Btime[7, k] = taumax
    end
    # υπολογισμός του εκθέτη power-law (Balpha) μεταξύ διαδοχικών AEGL σημείων· 0 αν οι συγκεντρώσεις είναι ίσες
    for k = 1:3
        for b = 3:6
            if Brho[b-1, k] == Brho[b, k]
                Balpha[b, k] = 0.0
            else
                Balpha[b, k] = log(Atime[b] / Atime[b-1]) / log(Brho[b-1, k] / Brho[b, k])
            end
        end
        # επέκταση των εκθετών στα άκρα του πίνακα (b = 1, 2, 7)
        Balpha[2, k] = Balpha[3, k]
        Balpha[1, k] = Balpha[2, k]
        Balpha[7, k] = Balpha[6, k]
    end
    # εξαγωγή των ακραίων συγκεντρώσεων: rhomax (για taumin) στη θέση 1 και rhomin (για taumax) στη θέση 7
    for k = 1:3
        if Balpha[3, k] == 0
            rhomax[k] = Brho[2, k]
            Brho[1, k] = rhomax[k]
        else
            rhomax[k] = Brho[2, k] * (Btime[2, k] / taumin)^(1 / Balpha[2, k])
            Brho[1, k] = rhomax[k]
        end
        if Brho[5, k] == Brho[6, k]
            rhomin[k] = Brho[6, k]
            Brho[7, k] = rhomin[k]
        else
            rhomin[k] = Brho[6, k] * (Btime[6, k] / taumax)^(1 / Balpha[6, k])
            Brho[7, k] = rhomin[k]
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
    return Balpha, Btime, Brho'
end

# Υπολογισμός των πινάκων τοξικότητας μία φορά (global, χρησιμοποιούνται από την tl_rate)
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

# --- Benchmark parameters ---
const BENCHMARK_MAX_RUNS = 50  # μέγιστο πλήθος εκτελέσεων (runs) του benchmark
const BENCHMARK_T_STEPS = 2500  # πλήθος βημάτων προσομοίωσης ανά εκτέλεση
const BENCHMARK_TL_CAP_PER_AGENT = 3.0  # άνω όριο του τοξικού φορτίου ανά agent που μετράει στα αποτελέσματα

# CM switch times (s) — Thesis, Table 7: 7 CMs, switching completes at 70% of T = 2500 s.
const CM_SWITCH_TIMES = [0.0, 292.0, 584.0, 876.0, 1168.0, 1460.0, 1752.0]  # χρόνοι (s) εναλλαγής των 7 χαρτών συγκέντρωσης
@assert length(CM_SWITCH_TIMES) == NUM_CMS  # έλεγχος ότι υπάρχει ένας χρόνος εναλλαγής ανά CM

# Μετατρέπει τους ενεργούς δείκτες και τις συντεταγμένες των εξόδων σε δύο strings για καταγραφή στο xlsx
function format_active_goals_for_xlsx(active_idx::Vector{Int}, dests)
    idx_str = join(string.(active_idx), ";")  # οι δείκτες χωρισμένοι με ερωτηματικό
    xy_str = join(  # οι συντεταγμένες (x, y) κάθε εξόδου, χωρισμένες με ερωτηματικό
        ["($(round(x; digits=1)),$(round(y; digits=1)))" for (x, y) in dests],
        "; ",
    )
    return idx_str, xy_str  # επιστρέφει τα δύο strings
end

# Εκτελεί μία εκτέλεση του benchmark και επιστρέφει τους χρόνους, το seed, τον συνδυασμό εξόδων και το τοξικό φορτίο των agents
"""
    run_one_benchmark(run_id)

Scenario 4: A* with `AbsolutePenaltyMap`, CM1→CM7 switching on `penalty_map` with replanning.
Active goals are **deterministic**: combination index from `goal_combo_index_for_run(run_id, BENCHMARK_MAX_RUNS, …)`
so batches with `BENCHMARK_MAX_RUNS` ≤ `length(GOAL_COMBINATIONS)` use **evenly spaced** indices across the full sorted list
`GOAL_COMBINATIONS`. Agent placement remains stochastic via `seed` (random per run).
"""
function run_one_benchmark(run_id::Int)
    seed = rand(Random.RandomDevice(), UInt32)  # τυχαίο seed για αυτή την εκτέλεση
    rng_run = MersenneTwister(seed)  # γεννήτρια τυχαίων αριθμών της εκτέλεσης

    n_combos = length(GOAL_COMBINATIONS)  # πλήθος διαθέσιμων συνδυασμών εξόδων
    goal_combo_idx = goal_combo_index_for_run(run_id, BENCHMARK_MAX_RUNS, n_combos)  # δείκτης συνδυασμού εξόδων για αυτό το run
    active_dest_indices = copy(GOAL_COMBINATIONS[goal_combo_idx])  # οι καθολικοί δείκτες των ενεργών εξόδων
    dests = DESTS_ALL[active_dest_indices]  # οι συντεταγμένες των ενεργών εξόδων
    K_active = length(active_dest_indices)  # πλήθος ενεργών εξόδων K
    idx_str, xy_str = format_active_goals_for_xlsx(active_dest_indices, dests)  # strings δεικτών και συντεταγμένων για το xlsx

    # τοπικό αντίγραφο του χάρτη κόστους και τοπικός pathfinder, ώστε κάθε εκτέλεση να έχει δικά της δεδομένα
    local_NPM_int = copy(NPM_int)
    # Fresh space per run: ContinuousSpace keeps its own spatial index of agent ids, so a shared
    # space would carry stale ids from earlier models (previous runs) into this one.
    local_space = ContinuousSpace(size(NPM); periodic = false, spacing = 1)  # νέος χώρος για κάθε run (βλ. σχόλιο παραπάνω)
    local_cost_metric_obj = AbsolutePenaltyMap(local_NPM_int, MaxDistance{2}())  # κόστος κίνησης από το τοπικό αντίγραφο
    local_pathfinderPM = AStar(local_space; walkmap = walkmap, cost_metric = local_cost_metric_obj)  # τοπικός A* pathfinder
    # ιδιότητες του μοντέλου αυτής της εκτέλεσης
    local_properties = (
        pathfinderPM = local_pathfinderPM,
        heightmap = heightmap,
        dt = dt,
        speed_range = speed_range,
        goal = dests,  # goal: οι ενεργές έξοδοι του συνδυασμού
    )

    # δημιουργία του ABM της εκτέλεσης με τον τοπικό pathfinder και το δικό της rng
    model_run = ABM(
        AgentEscapes,
        local_space;
        rng = rng_run,
        properties = local_properties,
        agent_step! = agent_step!,
        model_step! = model_step!,
    )

    # δημιουργία των agents σε τυχαίες βατές θέσεις (όπως παραπάνω, αλλά στο model_run)
    for _ in 1:n_agents
        age = rand(abmrng(model_run)) * (age_range[2] - age_range[1]) + age_range[1]  # τυχαία ηλικία εντός age_range
        mass = rand(abmrng(model_run)) * (mass_range[2] - mass_range[1]) + mass_range[1]  # τυχαία μάζα εντός mass_range
        vel = Tuple(rand(abmrng(model_run), 2) .* (speed_range[2] - speed_range[1]) .+ speed_range[1])  # τυχαία αρχική ταχύτητα (vx, vy)
        max_attempts = 1000  # μέγιστος αριθμός προσπαθειών εύρεσης βατής θέσης
        attempts = 0
        pos = nothing
        while attempts < max_attempts
            # τυχαία υποψήφια θέση μέσα στην κεντρική περιοχή του χάρτη
            candidate_pos = Tuple((rand(abmrng(model_run), floor.(ag_range_y)), rand(abmrng(model_run), floor.(ag_range_x))))
            pos_int = floor.(Int, candidate_pos)  # αντίστοιχο cell
            # αποδοχή της θέσης μόνο αν είναι εντός ορίων και βατή
            if 1 <= pos_int[1] <= size(walkmap, 1) && 1 <= pos_int[2] <= size(walkmap, 2) && walkmap[pos_int[1], pos_int[2]]
                pos = candidate_pos
                break
            end
            attempts += 1  # αλλιώς μετράει την αποτυχημένη προσπάθεια
        end
        # αν δεν βρέθηκε βατή θέση, εμφανίζει προειδοποίηση και παραλείπει τον agent
        if pos === nothing
            println("Warning: Could not find valid walkable spawn for agent (seed=$seed)")
            continue
        end
        # προσθήκη του agent με αρχικές τιμές: vel, age, mass, toxicload = 0.0, pathX, pathY, TL1, TL2, TL3
        add_agent!(
            pos, AgentEscapes, model_run,
            vel, age, mass, 0.0,
            [pos[1]], [pos[2]],
            [0.0], [0.0], [0.0]
        )
    end

    # ξεκίνημα με τον 1ο χάρτη συγκέντρωσης: ενημέρωση του (κοινού) penalty_map και του χάρτη κόστους του pathfinder
    current_map_idx = 1
    penalty_map .= cm_list[current_map_idx]  # αντιγραφή του CM1 στο penalty_map
    @. local_NPM_int = Int(round(heightmap + penalty_map))  # νέος συνδυασμένος χάρτης κόστους (heightmap + penalty_map) σε Int, στο τοπικό αντίγραφο
    Pathfinding.penaltymap(local_pathfinderPM) .= local_NPM_int  # ο πίνακας κόστους του pathfinder ενημερώνεται στη θέση του

    # μέτρηση του χρόνου της αρχικής εύρεσης διαδρομών (κάθε agent προς την καλύτερη έξοδο)
    initial_pf_time = @elapsed begin
        for a in allagents(model_run)
            plan_best_route!(a, dests, local_pathfinderPM)
        end
    end

    # συσσώρευση χρόνων: επανασχεδιασμού διαδρομών και βημάτων προσομοίωσης
    replan_time = 0.0
    sim_step_time = 0.0

    # κύριος βρόχος: ένα βήμα προσομοίωσης ανά επανάληψη
    for step_idx in 1:BENCHMARK_T_STEPS
        sim_step_time += @elapsed step!(model_run, 1)  # ένα βήμα του μοντέλου (με μέτρηση του χρόνου)
        t_sim = (step_idx - 1) * dt  # χρόνος της προσομοίωσης για αυτό το βήμα
        new_idx = searchsortedlast(CM_SWITCH_TIMES, t_sim)  # δείκτης του χάρτη συγκέντρωσης που ισχύει τώρα, με βάση τους χρόνους εναλλαγής
        if new_idx != current_map_idx  # αν άλλαξε ο χάρτης συγκέντρωσης
            current_map_idx = new_idx  # ο νέος χάρτης γίνεται ο τρέχων
            # μέτρηση του χρόνου επανασχεδιασμού: ενημέρωση χαρτών και νέα διαδρομή για κάθε agent
            replan_time += @elapsed begin
                penalty_map .= cm_list[current_map_idx]  # αντιγραφή του νέου CM στο penalty_map
                @. local_NPM_int = Int(round(heightmap + penalty_map))  # νέος χάρτης κόστους στο τοπικό αντίγραφο
                Pathfinding.penaltymap(local_pathfinderPM) .= local_NPM_int  # ενημέρωση του πίνακα κόστους του pathfinder στη θέση του
                for a in allagents(model_run)
                    plan_best_route!(a, model_run.goal, local_pathfinderPM)  # επανασχεδιασμός της διαδρομής του agent
                end
            end
        end
    end

    incremental_pf_time = replan_time  # συνολικός χρόνος επανασχεδιασμού κατά τις αλλαγές CM
    simulation_time = sim_step_time  # συνολικός χρόνος των βημάτων προσομοίωσης

    # συλλογή του τοξικού φορτίου κάθε agent με άνω όριο BENCHMARK_TL_CAP_PER_AGENT, ταξινομημένου κατά id
    agents_sorted = sort(collect(allagents(model_run)); by = a -> a.id)
    sum_TL_capped = 0.0  # άθροισμα των περιορισμένων τοξικών φορτίων
    per_agent_TL_capped = Float64[]  # τα περιορισμένα φορτία ανά agent
    agent_ids = Int[]  # τα id των agents
    for a in agents_sorted
        tl_cap = min(a.toxicload, BENCHMARK_TL_CAP_PER_AGENT)  # περιορισμός του φορτίου του agent στο όριο
        sum_TL_capped += tl_cap
        push!(per_agent_TL_capped, tl_cap)
        push!(agent_ids, a.id)
    end

    return (  # επιστροφή: χρόνοι, seed, δείκτης συνδυασμού, άθροισμα TL, TL ανά agent, id, K, δείκτες και συντεταγμένες εξόδων
        initial_pf_time,
        incremental_pf_time,
        simulation_time,
        seed,
        goal_combo_idx,
        sum_TL_capped,
        per_agent_TL_capped,
        agent_ids,
        K_active,
        idx_str,
        xy_str,
    )
end

# Εκτελεί μία εκτέλεση run_one_benchmark(1) για profiling· η συνάρτηση δεν καλεί η ίδια @profile ή Profile.print() (καλούνται χειροκίνητα από τον χρήστη)
"""
    profile_one_benchmark()
Profile one benchmark run. After running, call Profile.print() or your profiler GUI.
"""
function profile_one_benchmark()
    Profile.clear()  # καθαρισμός προηγούμενων δεδομένων του profiler
    init_t, incr_t, sim_t, run_seed, _, sum_cap, _, _, _, _, _ = run_one_benchmark(1)  # εκτέλεση της run_one_benchmark(1) (δεδομένα profiling συλλέγονται μόνο αν η συνάρτηση καλείται μέσα σε @profile)
    println("Profiled run (seed=$run_seed): initial PF = $(round(init_t; digits=4)) s, incremental PF = $(round(incr_t; digits=4)) s, simulation = $(round(sim_t; digits=4)) s, sum TL (capped max $(BENCHMARK_TL_CAP_PER_AGENT) per agent) = $(round(sum_cap; digits=4))")  # εκτύπωση των χρόνων και του αθροίσματος TL της εκτέλεσης
    println("Run Profile.print() or open profiler to inspect hotspots.")  # υπόδειξη για την εκτύπωση των αποτελεσμάτων του profiler
end

# Μετράει με το BenchmarkTools μία κλήση της run_one_benchmark(1) και επιστρέφει το αποτέλεσμα (Trial)
"""
    benchmark_run_one_benchmark(; samples = 10)

Use BenchmarkTools to benchmark a single `run_one_benchmark(1)` invocation.
Returns the BenchmarkTools Trial object and also prints a summary.
"""
function benchmark_run_one_benchmark(; samples::Int = 10)
    println("Benchmarking run_one_benchmark(1) with BenchmarkTools (samples = $samples)...")  # ενημερωτικό μήνυμα
    result = @benchmark run_one_benchmark(1) samples = samples  # εκτέλεση του benchmark με samples δείγματα
    println(result)  # εκτύπωση της σύνοψης
    return result  # επιστροφή του αποτελέσματος
end

# --- Benchmark loop ---
# Λίστες όπου συγκεντρώνονται τα αποτελέσματα όλων των εκτελέσεων
initial_pf_times = Float64[]  # χρόνος αρχικής εύρεσης διαδρομών ανά run
incremental_pf_times = Float64[]  # χρόνος επανασχεδιασμού ανά run
simulation_times = Float64[]  # χρόνος προσομοίωσης ανά run
sum_TL_capped_values = Float64[]  # άθροισμα περιορισμένου TL ανά run
seeds = UInt32[]  # seed ανά run
goal_combination_indices = Int[]  # δείκτης συνδυασμού εξόδων ανά run
n_active_goals_list = Int[]  # πλήθος ενεργών εξόδων ανά run
active_goal_indices_str = String[]  # δείκτες ενεργών εξόδων (string) ανά run
active_goal_xy_str = String[]  # συντεταγμένες ενεργών εξόδων (string) ανά run
agent_tl_runs = Int[]  # για το φύλλο AgentTL: αριθμός run
agent_tl_ids = Int[]  # για το φύλλο AgentTL: id του agent
agent_tl_values = Float64[]  # για το φύλλο AgentTL: περιορισμένο TL του agent

println("Benchmark: Scenario 4 with A* (APM + CM replanning; goal combo index = evenly spaced over 1:N when runs ≤ N). Runs = $BENCHMARK_MAX_RUNS, agents = $n_agents, T = $BENCHMARK_T_STEPS, CM switch times = $CM_SWITCH_TIMES, N_combos = $(length(GOAL_COMBINATIONS)).")  # μήνυμα με τις παραμέτρους του benchmark
# Βρόχος των εκτελέσεων: κάθε run αποθηκεύει τα αποτελέσματά του στις λίστες
for run_id in 1:BENCHMARK_MAX_RUNS
    t_init, t_incr, t_sim, run_seed, g_combo, sum_TL_capped, per_agent_TL_capped, agent_ids, K_a, idx_s, xy_s = run_one_benchmark(run_id)  # εκτέλεση του run και λήψη των αποτελεσμάτων του
    push!(initial_pf_times, t_init)  # αποθήκευση των αποτελεσμάτων του run
    push!(incremental_pf_times, t_incr)
    push!(simulation_times, t_sim)
    push!(sum_TL_capped_values, sum_TL_capped)
    push!(seeds, run_seed)
    push!(goal_combination_indices, g_combo)
    push!(n_active_goals_list, K_a)
    push!(active_goal_indices_str, idx_s)
    push!(active_goal_xy_str, xy_s)
    for k in eachindex(per_agent_TL_capped)  # καταγραφή του περιορισμένου TL κάθε agent
        push!(agent_tl_runs, run_id)
        push!(agent_tl_ids, agent_ids[k])
        push!(agent_tl_values, per_agent_TL_capped[k])
    end

    n = length(initial_pf_times)  # μέσοι όροι ως τώρα, για την εκτύπωση προόδου
    avg_initial_pf = sum(initial_pf_times) / n
    avg_incremental_pf = sum(incremental_pf_times) / n
    avg_simulation = sum(simulation_times) / n

    println("  Run $run_id (combo=$g_combo, seed=$run_seed, K=$K_a): initial PF = $(round(t_init; digits=4)) s, incremental PF = $(round(t_incr; digits=4)) s, simulation = $(round(t_sim; digits=4)) s, sum TL capped = $(round(sum_TL_capped; digits=4))  (avg total PF = $(round(avg_initial_pf + avg_incremental_pf; digits=4)) s)")  # γραμμή προόδου του run
end

# Τελικά στατιστικά: πλήθος runs και μέσοι χρόνοι
n_runs = length(initial_pf_times)
avg_initial_pf_final = sum(initial_pf_times) / n_runs
avg_incremental_pf_final = sum(incremental_pf_times) / n_runs
avg_pathfinding_final = avg_initial_pf_final + avg_incremental_pf_final
avg_simulation_final = sum(simulation_times) / n_runs

run_timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")  # χρονική σήμανση για το όνομα του xlsx
xlsx_path = "SCENARIOS/SCENARIO 4/Simulation Results/AStar $(n_agents) Agents Benchmarking_$(run_timestamp).xlsx"  # διαδρομή του αρχείου xlsx
run_ids = collect(1:n_runs)
columns_data = [  # στήλες του φύλλου Runs
    run_ids,
    goal_combination_indices,
    seeds,
    initial_pf_times,
    incremental_pf_times,
    simulation_times,
    sum_TL_capped_values,
    n_active_goals_list,
    active_goal_indices_str,
    active_goal_xy_str,
]
column_names = [  # ονόματα των στηλών του φύλλου Runs
    "Run",
    "GoalCombinationIndex",
    "Seed",
    "InitialPathfindingTime_s",
    "IncrementalPathfindingTime_s",
    "SimulationTime_s",
    "SumTL_capped",
    "N_active_goals",
    "ActiveGoalIndices_1to16",
    "ActiveGoals_xy",
]
agent_columns_data = [agent_tl_runs, agent_tl_ids, agent_tl_values]  # στήλες του φύλλου AgentTL
agent_column_names = ["Run", "AgentID", "TL_capped_max3"]  # ονόματα των στηλών του φύλλου AgentTL
# εγγραφή του xlsx με δύο φύλλα: Runs και AgentTL
XLSX.writetable(
    xlsx_path;
    Runs = (columns_data, column_names),
    AgentTL = (agent_columns_data, agent_column_names),
    overwrite = true,
)
# προσθήκη στο φύλλο Runs, κάτω από τα δεδομένα, γραμμών σύνοψης (μέσοι χρόνοι και παράμετροι του benchmark)
XLSX.openxlsx(xlsx_path, mode = "rw") do xf
    sh = xf["Runs"]
    sh[n_runs + 2, 1] = "Number of runs"
    sh[n_runs + 2, 2] = n_runs
    sh[n_runs + 3, 1] = "Average initial pathfinding time (s)"
    sh[n_runs + 3, 2] = avg_initial_pf_final
    sh[n_runs + 4, 1] = "Average incremental pathfinding time (s)"
    sh[n_runs + 4, 2] = avg_incremental_pf_final
    sh[n_runs + 5, 1] = "Average total pathfinding time (s)"
    sh[n_runs + 5, 2] = avg_pathfinding_final
    sh[n_runs + 6, 1] = "Average simulation time (s)"
    sh[n_runs + 6, 2] = avg_simulation_final
    sh[n_runs + 7, 1] = "BENCHMARK_T_STEPS"
    sh[n_runs + 7, 2] = BENCHMARK_T_STEPS
    sh[n_runs + 8, 1] = "CM_SWITCH_TIMES_s"
    sh[n_runs + 8, 2] = join(CM_SWITCH_TIMES, ", ")
    sh[n_runs + 9, 1] = "EnumeratedGoalCombinations_N"
    sh[n_runs + 9, 2] = length(GOAL_COMBINATIONS)
end

println("Benchmark complete. Results written to $xlsx_path")  # μήνυμα ολοκλήρωσης
println("  Total runs: $n_runs | Avg initial PF: $(round(avg_initial_pf_final; digits=4)) s | Avg incremental PF: $(round(avg_incremental_pf_final; digits=4)) s | Avg simulation: $(round(avg_simulation_final; digits=4)) s")  # σύνοψη των μέσων χρόνων
