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
    using Statistics                                                                                # στατιστικά (mean, median, std) για τα αποτελέσματα
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

# --- Benchmark parameters ---
const BENCHMARK_SCENARIO = "SCENARIO 1"                                                             # σενάριο (φάκελος) του benchmark: στο όνομα και στη θέση του αρχείου Excel και στο Summary
const BENCHMARK_ALGORITHM = "AStar"                                                                 # όνομα αλγορίθμου (στις παραμέτρους του Summary και στο όνομα του αρχείου Excel)
const BENCHMARK_DESCRIPTION = "A* on PenaltyMap = heightmap only (no CM in pathfinding, no replanning); TL from CM1..CM7"  # σύντομη περιγραφή του σεναρίου (γράφεται στο Summary)
const BENCHMARK_MAX_RUNS = 100                                                                      # αριθμός runs ανά πλήθος agents
const BENCHMARK_AGENT_COUNTS = [5, 10, 25, 50, 75, 100]                                             # πλήθη agents που δοκιμάζονται (ένα φύλλο Excel για κάθε πλήθος)
const BENCHMARK_TL_CAP_PER_AGENT = 3.0                                                              # άνω όριο TL ανά agent για το άθροισμα SumTL_capped
const TL_INCAPACITATED = 3.0                                                                        # TL από το οποίο ο agent θεωρείται ανίκανος (ταχύτητα 0, βλ. agent_step!)
const PROJECTION_MAX_STEPS = 10800                                                                  # μέγιστα επιπλέον βήματα της προβολής (3 ώρες, χωρίς χρονομέτρηση) για την επίλυση των stranded agents μετά το T
const BENCHMARK_EXTRA_CONFIG = Tuple{String,Any}[]                                                  # επιπλέον παράμετροι του σεναρίου για το Summary (καμία σε αυτό το σενάριο)

# Horizon T = 2500 s and CM switch times (s) — Thesis, Table 7: 7 CMs, switching completes at 70% of T.
# t = (step_idx - 1) * dt. The CM of each switch time is applied BEFORE the step that starts at that time (same in every benchmark).
const BENCHMARK_T_STEPS = 2500                                                                      # ορίζοντας T σε βήματα (2500 s με dt = 1)
const CM_SWITCH_TIMES = [0.0, 292.0, 584.0, 876.0, 1168.0, 1460.0, 1752.0]                          # χρόνοι (s) στους οποίους ενεργοποιείται κάθε CM (CM1..CM7)
@assert length(CM_SWITCH_TIMES) == NUM_CMS "CM_SWITCH_TIMES must have one entry per concentration map"  # έλεγχος ότι υπάρχει χρόνος εναλλαγής για κάθε CM

# Warm-up run (JIT compilation) — its results are discarded. It only needs to reach the
# first CM switch so that the CM-switch (and replanning) code path is compiled too.
# Το warm-up τρέχει μέχρι λίγο μετά την πρώτη αλλαγή CM ώστε να γίνει JIT compilation και του κώδικα της εναλλαγής CM / του επανασχεδιασμού (τα αποτελέσματά του απορρίπτονται).
const WARMUP_T_STEPS = Int(CM_SWITCH_TIMES[2] / dt) + 2                                             # βήματα του warm-up: μέχρι λίγο μετά την 1η εναλλαγή CM

# Ένα πλήρες benchmark run (Σενάριο 1): νέο μοντέλο με τυχαίο seed και num_agents agents, χρονομέτρηση του αρχικού σχεδιασμού διαδρομών με A* μόνο πάνω στο heightmap, t_steps βήματα με εναλλαγή των CM1..CM7 μόνο για το τοξικό φορτίο (χωρίς επανασχεδιασμό). Επιστρέφει τις μετρικές του run και τις γραμμές ανά agent.
"""
    run_one_benchmark(; num_agents, t_steps = BENCHMARK_T_STEPS, project = true) -> (run_metrics, agent_rows)

Scenario 1: A* uses only the `heightmap` in its `PenaltyMap` (no CM in pathfinding), so the routes are
planned once at the start and never replanned (`IncrementalPF_s` = 0). The toxic load accumulates on
`tl_penalty_map`, which switches CM1→CM7 at `CM_SWITCH_TIMES` exactly as in every other benchmark.
Agent outcomes (evacuated / incapacitated / stranded + projection) are tracked outside the
timed regions — see `evaluate_outcomes!`.

No CSV/video is written from this function; higher-level code handles aggregation/Excel.
"""
function run_one_benchmark(; num_agents::Int, t_steps::Int = BENCHMARK_T_STEPS, project::Bool = true)
    seed = rand(Random.RandomDevice(), UInt32)                                                      # τυχαίο seed από RandomDevice: κάθε run είναι ανεξάρτητο και το seed καταγράφεται στα αποτελέσματα
    rng_run = MersenneTwister(seed)                                                                 # γεννήτρια τυχαίων αριθμών του run

    # Fresh space per run: ContinuousSpace keeps its own spatial index of agent ids, so a shared
    # space would carry stale ids from earlier models (warm-up / previous runs) into this one.
    local_space = ContinuousSpace(size(heightmap); periodic = false, spacing = 1)                   # νέος χώρος για κάθε run (βλ. σχόλιο παραπάνω)
    local_cost_metric_obj = PenaltyMap(heightmap, MaxDistance{2}())                                 # PenaltyMap μόνο πάνω στο heightmap: το CM δεν επηρεάζει το κόστος του A*
    local_pathfinderPM = AStar(local_space; walkmap = walkmap, cost_metric = local_cost_metric_obj)  # τοπικός A* pathfinder για αυτό το run
    local_properties = (                                                                            # ιδιότητες του μοντέλου του run
        pathfinderPM = local_pathfinderPM,
        heightmap    = heightmap,
        dt           = dt,
        speed_range  = speed_range,
        goal         = dests,
    )

    model_run = ABM(                                                                                # δημιουργία του μοντέλου του run (χρησιμοποιεί τις global agent_step! και model_step!)
        AgentEscapes,
        local_space;
        rng         = rng_run,
        properties  = local_properties,
        agent_step! = agent_step!,
        model_step! = model_step!,
    )

    for _ in 1:num_agents                                                                           # δημιουργία των agents σε τυχαίες βατές θέσεις
        age = rand(abmrng(model_run)) * (age_range[2] - age_range[1]) + age_range[1]                # τυχαία ηλικία εντός age_range
        mass = rand(abmrng(model_run)) * (mass_range[2] - mass_range[1]) + mass_range[1]            # τυχαία μάζα εντός mass_range
        vel = Tuple(rand(abmrng(model_run), 2) .* (speed_range[2] - speed_range[1]) .+ speed_range[1])  # τυχαία αρχική ταχύτητα (vx, vy)
        max_attempts = 1000                                                                         # μέγιστος αριθμός προσπαθειών εύρεσης βατής θέσης
        attempts = 0                                                                                # μετρητής προσπαθειών
        pos = nothing                                                                               # θέση spawn (nothing μέχρι να βρεθεί βατή)
        while attempts < max_attempts                                                               # επανάληψη μέχρι να βρεθεί βατή θέση ή να εξαντληθούν οι προσπάθειες
            candidate_pos = Tuple((rand(abmrng(model_run), floor.(ag_range_y)), rand(abmrng(model_run), floor.(ag_range_x))))  # τυχαία υποψήφια θέση μέσα στην κεντρική περιοχή του χάρτη (ag_range_x/ag_range_y)
            pos_int = floor.(Int, candidate_pos)                                                    # αντίστοιχο cell
            if 1 <= pos_int[1] <= size(walkmap, 1) && 1 <= pos_int[2] <= size(walkmap, 2) && walkmap[pos_int[1], pos_int[2]]  # αποδοχή της θέσης μόνο αν είναι εντός ορίων και βατή
                pos = candidate_pos                                                                 # βρέθηκε βατή θέση
                break                                                                               # σταματά η αναζήτηση
            end
            attempts += 1                                                                           # αλλιώς μετράει την αποτυχημένη προσπάθεια και ξαναδοκιμάζει
        end
        if pos === nothing                                                                          # αν δεν βρέθηκε βατή θέση, εμφανίζει προειδοποίηση και παραλείπει τον agent
            println("Warning: Could not find valid walkable spawn for agent (seed=$seed)")          # προειδοποίηση
            continue                                                                                # επόμενος agent
        end
        # προσθήκη του agent με αρχικές τιμές για όλα τα πεδία του AgentEscapes
        add_agent!(
            pos, AgentEscapes, model_run,
            vel, age, mass, 0.0,                                                                    # (αρχικό toxicload = 0.0)
            [pos[1]], [pos[2]],                                                                     # pathX, pathY: αρχική θέση
            [0.0], [0.0], [0.0]                                                                     # TL1, TL2, TL3: αρχικό φορτίο 0
        )
    end

    # Το TL του run ξεκινά από το CM1 (το global tl_penalty_map το διαβάζει η agent_step!)
    current_map_idx = 1                                                                             # δείκτης του τρέχοντος concentration map (ξεκινά από το CM1)
    tl_penalty_map .= cm_list[current_map_idx]                                                      # επαναφορά του global tl_penalty_map στο CM1 για αυτό το run

    tm_init = @timed for a in allagents(model_run)                                                  # χρονομέτρηση του αρχικού σχεδιασμού: καλύτερη διαδρομή προς μία από τις εξόδους για κάθε agent
        plan_best_route!(a, dests, local_pathfinderPM)                                              # σχεδιασμός διαδρομής του agent προς την καλύτερη έξοδο (plan_best_route!)
    end

    initial_pf_time = tm_init.time                                                                  # χρόνος αρχικού σχεδιασμού (s)
    pf_bytes = tm_init.bytes                                                                        # bytes που δεσμεύτηκαν (allocations) στον σχεδιασμό
    pf_gc = tm_init.gctime                                                                          # χρόνος garbage collection στον σχεδιασμό
    replan_time = 0.0                                                                               # συνολικός χρόνος επανασχεδιασμού (s)
    sim_step_time = 0.0                                                                             # συνολικός χρόνος των βημάτων προσομοίωσης (s), χωρίς τον σχεδιασμό
    evac_t = Dict{Int,Float64}()                                                                    # χρόνος διαφυγής ανά agent id
    tl3_t = Dict{Int,Float64}()                                                                     # χρόνος που ο agent έφτασε TL >= TL_INCAPACITATED, ανά agent id

    for step_idx in 1:t_steps                                                                       # κύριος βρόχος προσομοίωσης: σε κάθε βήμα ελέγχεται πρώτα αν πρέπει να αλλάξει το CM και έπειτα εκτελείται (και χρονομετρείται) το βήμα
        t_sim = (step_idx - 1) * dt                                                                 # χρόνος προσομοίωσης (s) στην αρχή αυτού του βήματος
        new_idx = searchsortedlast(CM_SWITCH_TIMES, t_sim)                                          # δείκτης του τελευταίου CM του οποίου ο χρόνος εναλλαγής έχει παρέλθει
        if new_idx != current_map_idx                                                               # αν άλλαξε το CM
            current_map_idx = new_idx                                                               # νέο τρέχον CM
            tl_penalty_map .= cm_list[current_map_idx]                                              # νέο CM μόνο για το TL· το κόστος του A* δεν εξαρτάται από το CM, άρα δεν γίνεται επανασχεδιασμός (IncrementalPF_s = 0)
        end
        # Το βήμα εκτελείται μετά τον έλεγχο εναλλαγής CM: το νέο CM ισχύει ήδη στο βήμα που ξεκινά στο t = CM_SWITCH_TIMES[k], την ίδια στιγμή σε όλα τα benchmarks.
        sim_step_time += @elapsed step!(model_run, 1)                                               # ένα βήμα του μοντέλου (χρονομετρείται με @elapsed)
        track_outcomes!(evac_t, tl3_t, model_run, step_idx * dt)                                    # καταγραφή διαφυγών/ανικανοτήτων (εκτός χρονομέτρησης)
    end

    route_of = a -> get(local_pathfinderPM.agent_paths, a.id, ())                                   # συνάρτηση που επιστρέφει τα εναπομείναντα waypoints του agent από τον pathfinder (agent_paths· κενή αν δεν υπάρχει)
    outcomes, agent_rows = evaluate_outcomes!(                                                      # ταξινόμηση των agents σε Evacuated/Incapacitated/Stranded στο T και προβολή (βλ. evaluate_outcomes!)
        model_run, route_of, evac_t, tl3_t, t_steps * dt; project = project,
    )
    timings = (                                                                                     # χρονομετρήσεις του run
        Seed            = Int(seed),                                                                # seed του run
        InitialPF_s     = initial_pf_time,                                                          # χρόνος αρχικού σχεδιασμού (s)
        IncrementalPF_s = replan_time,                                                              # χρόνος επανασχεδιασμού (s)
        TotalPF_s       = initial_pf_time + replan_time,                                            # συνολικός χρόνος σχεδιασμού (s)
        Simulation_s    = sim_step_time,                                                            # χρόνος προσομοίωσης (s)
        PF_Alloc_MB     = pf_bytes / 2^20,                                                          # allocations του σχεδιασμού σε MB
        PF_GC_s         = pf_gc,                                                                    # χρόνος GC του σχεδιασμού (s)
    )
    return merge(timings, outcomes), agent_rows                                                     # ένωση χρονομετρήσεων και αποτελεσμάτων· επιστροφή και των γραμμών ανά agent
end

# ---------------- Outcome tracking: Evacuated / Incapacitated / Stranded ----------------

# Καταγράφει, για κάθε agent χωρίς ήδη καταγεγραμμένο αποτέλεσμα, τον πρώτο χρόνο t_now που έφτασε σε έξοδο (evac_t) ή σε TL >= TL_INCAPACITATED (tl3_t). Καλείται μετά από κάθε step!, εκτός των χρονομετρούμενων τμημάτων.
"""
    track_outcomes!(evac_t, tl3_t, model, t_now)

Record the first time `t_now` at which each agent reaches an exit (`evac_t`) or
TL ≥ TL_INCAPACITATED (`tl3_t`). Called after every `step!`, outside the timed regions.
"""
function track_outcomes!(evac_t::Dict{Int,Float64}, tl3_t::Dict{Int,Float64}, model, t_now::Float64)
    for a in allagents(model)                                                                       # έλεγχος κάθε agent
        (haskey(evac_t, a.id) || haskey(tl3_t, a.id)) && continue                                   # αν έχει ήδη καταγραφεί αποτέλεσμα (διαφυγή ή ανικανότητα), τον προσπερνά
        if at_goal(a.pos, model.goal)                                                               # έφτασε σε έξοδο: καταγράφεται ο χρόνος διαφυγής
            evac_t[a.id] = t_now                                                                    # χρόνος διαφυγής
        elseif a.toxicload >= TL_INCAPACITATED                                                      # TL >= TL_INCAPACITATED: ο agent είναι ανίκανος
            tl3_t[a.id] = t_now                                                                     # χρόνος ανικανότητας
        end
    end
    return nothing                                                                                  # δεν επιστρέφει τιμή
end

# Μήκος (σε μέτρα) της εναπομείνασας διαδρομής από τη θέση pos μέσω των waypoints.
"Length (m) of the remaining route from `pos` through `waypoints`."
function route_length_m(pos, waypoints)
    L = 0.0                                                                                         # συσσωρευμένο μήκος της διαδρομής (σε pixels)
    px, py = Float64(pos[1]), Float64(pos[2])                                                       # τρέχον σημείο = θέση του agent
    for w in waypoints                                                                              # για κάθε επόμενο waypoint
        wx, wy = Float64(w[1]), Float64(w[2])                                                       # συντεταγμένες του waypoint
        L += hypot(wx - px, wy - py)                                                                # προσθήκη της ευκλείδιας απόστασης
        px, py = wx, wy                                                                             # το waypoint γίνεται το τρέχον σημείο
    end
    return L / METERS_TO_PIXELS                                                                     # μετατροπή από pixels σε μέτρα
end

# Βοηθητικές: μέσος όρος / μέγιστο ή missing όταν το διάνυσμα είναι κενό.
_mean_or_missing(v) = isempty(v) ? missing : mean(v)
_max_or_missing(v) = isempty(v) ? missing : maximum(v)

# Ταξινομεί κάθε agent στον ορίζοντα T (Evacuated / Incapacitated / Stranded) και, αν project=true, προβάλλει τους stranded agents μέχρι να επιλυθούν (χωρίς επανασχεδιασμό, χωρίς χρονομέτρηση). Επιστρέφει τις μετρικές του run και μία γραμμή ανά agent.
"""
    evaluate_outcomes!(model_run, route_of, evac_t, tl3_t, t_horizon; project = true)
        -> (outcome_metrics, agent_rows)

Classifies every agent at the horizon T:

- Evacuated:     reached an exit within T
- Incapacitated: reached TL ≥ TL_INCAPACITATED within T
- Stranded:      still en route at T with TL < TL_INCAPACITATED

Stranded agents are then resolved by a projection: the model keeps running after T
under the last concentration map (no replanning, NOT timed) for at most
PROJECTION_MAX_STEPS, until every agent has evacuated, become incapacitated, or has no
route left. `Proj_*` fields describe that projected outcome (for agents already resolved
within T it equals the outcome at T). `route_of(a)` returns the agent's remaining waypoints.
"""
function evaluate_outcomes!(model_run, route_of, evac_t, tl3_t, t_horizon; project::Bool = true)
    agents_sorted = sort(collect(allagents(model_run)); by = a -> a.id)                             # agents ταξινομημένοι κατά id (σταθερή σειρά στα αποτελέσματα)
    snap_tl = Dict(a.id => a.toxicload for a in agents_sorted)                                      # στιγμιότυπο του TL κάθε agent στον ορίζοντα T (πριν την προβολή)
    snap_dist = Dict(a.id => route_length_m(a.pos, route_of(a)) for a in agents_sorted)             # απομένουσα απόσταση (m) κάθε agent στο T, από την υπάρχουσα διαδρομή του
    evac_T = copy(evac_t)                                                                           # αντίγραφα των αποτελεσμάτων στο T (η προβολή θα τροποποιήσει τα evac_t/tl3_t)
    tl3_T = copy(tl3_t)                                                                             # αντίγραφο των χρόνων ανικανότητας στο T

    proj_steps = 0                                                                                  # βήματα προβολής που εκτελέστηκαν
    if project                                                                                      # η προβολή μετά το T είναι προαιρετική
        is_open = a -> !(haskey(evac_t, a.id) || haskey(tl3_t, a.id) || isempty(route_of(a)))       # agent «ανοιχτός»: χωρίς διαφυγή/ανικανότητα και με διαδρομή που δεν έχει εξαντληθεί
        while proj_steps < PROJECTION_MAX_STEPS && any(is_open, allagents(model_run))               # συνεχίζει μέχρι PROJECTION_MAX_STEPS ή μέχρι να μην υπάρχει ανοιχτός agent
            step!(model_run, 1)                                                                     # ένα βήμα προσομοίωσης (χωρίς επανασχεδιασμό, εκτός χρονομέτρησης)
            proj_steps += 1                                                                         # μέτρηση των βημάτων προβολής
            track_outcomes!(evac_t, tl3_t, model_run, t_horizon + proj_steps * dt)                  # καταγραφή αποτελεσμάτων με χρόνο μετά το T
        end
    end

    rows = NamedTuple[]                                                                             # γραμμές αποτελεσμάτων ανά agent
    for a in agents_sorted                                                                          # για κάθε agent με σειρά id
        outcome_T = haskey(evac_T, a.id) ? "Evacuated" :                                            # αποτέλεσμα στο T: Evacuated / Incapacitated / Stranded
                    haskey(tl3_T, a.id)  ? "Incapacitated" : "Stranded"
        proj_outcome = haskey(evac_t, a.id) ? "Evacuated" :                                         # αποτέλεσμα μετά την προβολή (NoRoute: δεν έχει απομείνει διαδρομή· Unresolved: ακόμη εν κινήσει)
                       haskey(tl3_t, a.id)  ? "Incapacitated" :
                       isempty(route_of(a)) ? "NoRoute" : "Unresolved"
        proj_time = haskey(evac_t, a.id) ? evac_t[a.id] :                                           # χρόνος του τελικού αποτελέσματος (missing αν δεν υπάρχει)
                    haskey(tl3_t, a.id)  ? tl3_t[a.id] : missing
        push!(rows, (                                                                               # γραμμή αποτελεσμάτων του agent
            AgentID           = a.id,                                                               # id του agent
            Outcome_T         = outcome_T,                                                          # αποτέλεσμα στο T
            TL_T              = snap_tl[a.id],                                                      # τοξικό φορτίο στο T
            TL_T_capped       = min(snap_tl[a.id], BENCHMARK_TL_CAP_PER_AGENT),                     # τοξικό φορτίο στο T με άνω όριο BENCHMARK_TL_CAP_PER_AGENT
            EvacTime_s        = get(evac_T, a.id, missing),                                         # χρόνος διαφυγής (αν διέφυγε εντός T)
            TL3Time_s         = get(tl3_T, a.id, missing),                                          # χρόνος ανικανότητας (αν έγινε ανίκανος εντός T)
            RemainingDist_T_m = outcome_T == "Evacuated" ? 0.0 : snap_dist[a.id],                   # απομένουσα απόσταση στο T (0 για όσους διέφυγαν)
            Proj_Outcome      = proj_outcome,                                                       # αποτέλεσμα μετά την προβολή
            Proj_Time_s       = proj_time,                                                          # χρόνος του αποτελέσματος μετά την προβολή
            Proj_TL           = a.toxicload,                                                        # τοξικό φορτίο μετά την προβολή
        ))
    end

    stranded = [r for r in rows if r.Outcome_T == "Stranded"]                                       # οι agents που είναι ακόμη εν κινήσει στο T
    metrics = (                                                                                     # συγκεντρωτικές μετρικές του run
        SumTL_capped                  = sum((r.TL_T_capped for r in rows); init = 0.0),             # άθροισμα TL (με άνω όριο ανά agent)
        Evacuated                     = count(r -> r.Outcome_T == "Evacuated", rows),               # πλήθος agents που διέφυγαν εντός T
        Incapacitated                 = count(r -> r.Outcome_T == "Incapacitated", rows),           # πλήθος agents που έγιναν ανίκανοι εντός T
        Stranded                      = length(stranded),                                           # πλήθος stranded agents στο T
        EvacTime_mean_s               = _mean_or_missing(collect(values(evac_T))),                  # μέσος χρόνος διαφυγής
        EvacTime_max_s                = _max_or_missing(collect(values(evac_T))),                   # μέγιστος χρόνος διαφυγής
        Stranded_TL_mean              = _mean_or_missing([r.TL_T for r in stranded]),               # μέσο TL των stranded agents
        Stranded_RemainingDist_mean_m = _mean_or_missing([r.RemainingDist_T_m for r in stranded]),  # μέση απομένουσα απόσταση (m) των stranded agents
        Proj_Evacuated                = count(r -> r.Proj_Outcome == "Evacuated", rows),            # πλήθος διαφυγών μετά την προβολή
        Proj_Incapacitated            = count(r -> r.Proj_Outcome == "Incapacitated", rows),        # πλήθος ανικανοτήτων μετά την προβολή
        Proj_NoRoute                  = count(r -> r.Proj_Outcome == "NoRoute", rows),              # πλήθος agents χωρίς διαδρομή μετά την προβολή
        Proj_Unresolved               = count(r -> r.Proj_Outcome == "Unresolved", rows),           # πλήθος agents που παραμένουν ανεπίλυτοι μετά την προβολή
        Proj_EvacTime_max_s           = _max_or_missing(collect(values(evac_t))),                   # μέγιστος χρόνος διαφυγής μετά την προβολή
        Proj_Steps                    = proj_steps,                                                 # βήματα προβολής
    )
    return metrics, rows                                                                            # επιστροφή των μετρικών του run και των γραμμών ανά agent
end

# ---------------- Excel export: Summary + one sheet per agent count + AgentOutcomes ----------------

# Στατιστικά που υπολογίζονται κάτω από τον πίνακα runs κάθε φύλλου.
const STAT_NAMES = ("Mean", "Median", "Std", "Min", "Max")                                          # ονόματα των στατιστικών που γράφονται κάτω από τα runs κάθε φύλλου
# Μετρικές που συνοψίζονται στο φύλλο Summary (mean/median/std ανά πλήθος agents).
const SUMMARY_METRICS = (
    :InitialPF_s, :IncrementalPF_s, :TotalPF_s, :Simulation_s, :PF_Alloc_MB,
    :SumTL_capped, :Evacuated, :Incapacitated, :Stranded,
    :Proj_Evacuated, :Proj_Incapacitated, :Proj_Unresolved,
)

# Υπολογίζει (mean, median, std, min, max) μιας στήλης αγνοώντας τα missing· επιστρέφει missing αν δεν υπάρχουν τιμές.
function column_stats(v)
    x = Float64[Float64(y) for y in v if !ismissing(y)]                                             # τιμές ως Float64 χωρίς τα missing
    isempty(x) && return (missing, missing, missing, missing, missing)                              # αν δεν υπάρχει καμία τιμή, όλα τα στατιστικά είναι missing
    return (mean(x), median(x), length(x) > 1 ? std(x) : missing, minimum(x), maximum(x))           # (mean, median, std, min, max)· η std μόνο όταν υπάρχουν περισσότερες από μία τιμές
end

# Μετατρέπει ένα διάνυσμα NamedTuples σε (ονόματα στηλών, στήλες) για το XLSX.writetable!.
function rows_to_columns(rows)
    ks = collect(keys(first(rows)))                                                                 # ονόματα των στηλών από τα κλειδιά της 1ης γραμμής
    return ks, [[r[k] for r in rows] for k in ks]                                                   # επιστροφή ονομάτων και στηλών (διάνυσμα τιμών ανά όνομα)
end

# Γράφει το αρχείο Excel: φύλλο Summary, ένα φύλλο N<πλήθος> ανά πλήθος agents (runs + στατιστικά) και φύλλο AgentOutcomes.
"""
    write_benchmark_xlsx(path, algorithm, results)

`results` is a vector of `(n, runs, agents)`. Rewrites the whole workbook, so it is called
after every completed agent count and the file always holds everything finished so far.
"""
function write_benchmark_xlsx(path, algorithm, results)
    XLSX.openxlsx(path, mode = "w") do xf                                                           # άνοιγμα (δημιουργία) του workbook σε λειτουργία εγγραφής
        summary = xf[1]                                                                             # το 1ο φύλλο γίνεται το Summary
        XLSX.rename!(summary, "Summary")
        all_stats = Dict{Symbol,Any}[]                                                              # στατιστικά ανά πλήθος agents (για το Summary)

        for res in results                                                                          # ένα φύλλο ανά πλήθος agents
            sh = XLSX.addsheet!(xf, "N$(res.n)")                                                    # νέο φύλλο N<πλήθος>
            ks, cols = rows_to_columns(res.runs)                                                    # ονόματα και στήλες των μετρικών των runs
            XLSX.writetable!(sh, cols, string.(ks))                                                 # γράφει τον πίνακα των runs
            # statistics block under the runs (blank row in between), aligned with the columns
            r0 = length(res.runs) + 3                                                               # πρώτη γραμμή του μπλοκ στατιστικών
            for (si, sname) in enumerate(STAT_NAMES)                                                # ονόματα των στατιστικών στην 1η στήλη
                sh[r0 + si - 1, 1] = sname
            end
            stats = Dict{Symbol,Any}()                                                              # στατιστικά ανά μετρική αυτού του πλήθους agents
            for (ci, k) in enumerate(ks)                                                            # για κάθε στήλη μετρικής
                k in (:Run, :Seed) && continue                                                      # δεν υπολογίζονται στατιστικά για τις στήλες Run και Seed
                all(x -> ismissing(x) || x isa Real, cols[ci]) || continue                          # ούτε για στήλες με μη αριθμητικές τιμές (π.χ. κείμενο, όπως οι έξοδοι στο Σενάριο 4)
                st = column_stats(cols[ci])                                                         # στατιστικά της στήλης
                stats[k] = st                                                                       # αποθήκευση για το Summary
                for si in eachindex(STAT_NAMES)                                                     # για κάθε στατιστικό
                    ismissing(st[si]) || (sh[r0 + si - 1, ci] = st[si])                             # γράφεται το στατιστικό (αν δεν είναι missing) κάτω από τη στήλη της μετρικής
                end
            end
            push!(all_stats, stats)                                                                 # αποθήκευση των στατιστικών αυτού του πλήθους agents
        end

        s_names = ["NAgents", "Runs"]                                                               # στήλες του Summary: NAgents, Runs και mean/median/std κάθε SUMMARY_METRICS
        s_cols = Any[[res.n for res in results], [length(res.runs) for res in results]]             # τιμές των στηλών NAgents και Runs
        for m in SUMMARY_METRICS, (si, suffix) in ((1, "mean"), (2, "median"), (3, "std"))          # για κάθε μετρική: mean, median, std
            push!(s_names, "$(m)_$(suffix)")                                                        # όνομα στήλης
            push!(s_cols, [stats[m][si] for stats in all_stats])                                    # τιμές της στήλης για όλα τα πλήθη agents
        end
        XLSX.writetable!(summary, s_cols, s_names)                                                  # γράφει τον πίνακα του Summary

        config = Tuple{String,Any}[                                                                 # παράμετροι της εκτέλεσης που καταγράφονται κάτω από τον πίνακα του Summary
            ("Scenario", BENCHMARK_SCENARIO),
            ("Algorithm", algorithm),
            ("Description", BENCHMARK_DESCRIPTION),
            ("Runs per agent count", BENCHMARK_MAX_RUNS),
            ("Horizon T (steps)", BENCHMARK_T_STEPS),
            ("dt (s)", dt),
            ("CM switch times (s)", join(CM_SWITCH_TIMES, ", ")),
            ("Warm-up (excluded from results)", "1 run, $WARMUP_T_STEPS steps"),
            ("Incapacitation threshold (TL)", TL_INCAPACITATED),
            ("Projection after T", "last CM kept, no replanning, not timed, max $PROJECTION_MAX_STEPS steps"),
            ("Seeds", "random per run (RandomDevice)"),
            ("CM switch timing", "applied before the step that starts at each switch time"),
        ]
        append!(config, BENCHMARK_EXTRA_CONFIG)                                                     # επιπλέον παράμετροι του σεναρίου (π.χ. οι συνδυασμοί εξόδων στο Σενάριο 4)
        c0 = length(results) + 3                                                                    # πρώτη γραμμή του μπλοκ παραμέτρων (μετά από μία κενή γραμμή)
        for (i, (label, value)) in enumerate(config)                                                # για κάθε παράμετρο
            summary[c0 + i - 1, 1] = label                                                          # όνομα παραμέτρου
            summary[c0 + i - 1, 2] = value                                                          # τιμή παραμέτρου
        end

        ag = XLSX.addsheet!(xf, "AgentOutcomes")                                                    # φύλλο με τα αποτελέσματα όλων των agents από όλα τα runs
        all_agents = reduce(vcat, [res.agents for res in results])                                  # συνένωση των γραμμών agents όλων των πλήθων
        ks, cols = rows_to_columns(all_agents)                                                      # ονόματα και στήλες
        XLSX.writetable!(ag, cols, string.(ks))                                                     # γράφει τον πίνακα AgentOutcomes
    end
end

# Κάνει profile ενός benchmark run (χωρίς προβολή) για εντοπισμό hotspots.
"""
    profile_one_benchmark(; num_agents = first(BENCHMARK_AGENT_COUNTS))
Profile one benchmark run. After running, call Profile.print() or your profiler GUI.
"""
function profile_one_benchmark(; num_agents::Int = first(BENCHMARK_AGENT_COUNTS))
    Profile.clear()                                                                                 # καθαρισμός προηγούμενων δεδομένων του profiler
    r, _ = @profile run_one_benchmark(; num_agents = num_agents, project = false)                   # profiling ενός run (χωρίς προβολή)
    println("Profiled run (seed=$(r.Seed), N=$num_agents): initial PF = $(round(r.InitialPF_s; digits=4)) s, incremental PF = $(round(r.IncrementalPF_s; digits=4)) s, simulation = $(round(r.Simulation_s; digits=4)) s")  # εκτύπωση συνοπτικών χρόνων του run
    println("Run Profile.print() or open profiler to inspect hotspots.")                            # οδηγία για την εξέταση των hotspots
end

# Μετράει ένα run_one_benchmark() με το BenchmarkTools (@benchmark) και επιστρέφει το Trial.
"""
    benchmark_run_one_benchmark(; num_agents = first(BENCHMARK_AGENT_COUNTS), samples = 10)

Use BenchmarkTools to benchmark a single `run_one_benchmark()` invocation.
Returns the BenchmarkTools Trial object and also prints a summary.
"""
function benchmark_run_one_benchmark(; num_agents::Int = first(BENCHMARK_AGENT_COUNTS), samples::Int = 10)
    println("Benchmarking run_one_benchmark() with BenchmarkTools (N = $num_agents, samples = $samples)...")  # ενημερωτικό μήνυμα
    result = @benchmark run_one_benchmark(; num_agents = $num_agents, project = false) samples = samples  # BenchmarkTools: samples επαναλήψεις του run_one_benchmark (χωρίς προβολή)
    println(result)                                                                                 # εκτύπωση του αποτελέσματος
    return result                                                                                   # επιστροφή του Trial
end

# ---------------- Benchmark sweep over agent counts ----------------

# Εκτέλεση του benchmark: warm-up και έπειτα runs για κάθε πλήθος agents, με εγγραφή του Excel μετά από κάθε πλήθος.
run_timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")                                    # χρονική σήμανση της εκτέλεσης
xlsx_path = "SCENARIOS/$(BENCHMARK_SCENARIO)/Simulation Results/$(BENCHMARK_ALGORITHM) Benchmarking_$(run_timestamp).xlsx"  # διαδρομή του αρχείου Excel εξόδου

println("Benchmark: $(BENCHMARK_SCENARIO) with $BENCHMARK_ALGORITHM ($BENCHMARK_DESCRIPTION). Agent counts = $BENCHMARK_AGENT_COUNTS, runs per count = $BENCHMARK_MAX_RUNS.")  # μήνυμα έναρξης του benchmark

# Warm-up (JIT compilation): executed once, results discarded — not part of the exported statistics.
println("  Warm-up run ($WARMUP_T_STEPS steps, excluded from results)...")                          # μήνυμα warm-up
warmup_elapsed = @elapsed run_one_benchmark(; num_agents = first(BENCHMARK_AGENT_COUNTS), t_steps = WARMUP_T_STEPS, project = false)  # warm-up με το μικρότερο πλήθος agents και t_steps = WARMUP_T_STEPS (χωρίς προβολή)
println("  Warm-up done in $(round(warmup_elapsed; digits=2)) s.")                                  # μήνυμα ολοκλήρωσης warm-up

benchmark_results = NamedTuple[]                                                                    # αποτελέσματα όλων των πλήθων agents
# Βρόχος πάνω στα πλήθη agents· για κάθε πλήθος εκτελούνται BENCHMARK_MAX_RUNS runs.
for n in BENCHMARK_AGENT_COUNTS                                                                     # για κάθε πλήθος agents
    println("=== N = $n agents ===")                                                                # μήνυμα για το τρέχον πλήθος
    runs = NamedTuple[]                                                                             # μετρικές των runs
    agents = NamedTuple[]                                                                           # γραμμές αποτελεσμάτων ανά agent όλων των runs
    for run_id in 1:BENCHMARK_MAX_RUNS                                                              # επαναλήψεις του benchmark για αυτό το πλήθος agents
        r, agent_rows = run_one_benchmark(; num_agents = n)                                         # ένα πλήρες benchmark run
        push!(runs, merge((Run = run_id,), r))                                                      # προσθήκη των μετρικών του run με τον αριθμό του run
        append!(agents, [merge((NAgents = n, Run = run_id, Seed = r.Seed), row) for row in agent_rows])  # προσθήκη των γραμμών agents με το πλήθος, το run και το seed
        println("  Run $run_id (seed=$(r.Seed)): initial PF = $(round(r.InitialPF_s; digits=4)) s, incremental PF = $(round(r.IncrementalPF_s; digits=4)) s, simulation = $(round(r.Simulation_s; digits=4)) s, alloc PF = $(round(r.PF_Alloc_MB; digits=1)) MB | evacuated $(r.Evacuated), incapacitated $(r.Incapacitated), stranded $(r.Stranded) → projected evac $(r.Proj_Evacuated) / incap $(r.Proj_Incapacitated) / unresolved $(r.Proj_Unresolved)")  # εκτύπωση συνοπτικών αποτελεσμάτων του run
    end
    push!(benchmark_results, (n = n, runs = runs, agents = agents))                                 # αποθήκευση των αποτελεσμάτων αυτού του πλήθους agents
    # Saved after every agent count, so a crash never loses completed counts.
    write_benchmark_xlsx(xlsx_path, BENCHMARK_ALGORITHM, benchmark_results)                         # εγγραφή (ξαναγράφεται ολόκληρο) του Excel
    tot = [r.TotalPF_s for r in runs]                                                               # συνολικός χρόνος σχεδιασμού κάθε run
    println("  N = $n done: total PF mean = $(round(mean(tot); digits=4)) s, median = $(round(median(tot); digits=4)) s. Saved to $xlsx_path")  # εκτύπωση μέσου και διάμεσου του συνολικού χρόνου σχεδιασμού
end

println("Benchmark complete. Results written to $xlsx_path")                                        # μήνυμα ολοκλήρωσης
