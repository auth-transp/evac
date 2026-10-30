begin   # Φόρτωση των απαραίτητων βιβλιοθηκών
    using Agents  # πλαίσιο agent-based μοντελοποίησης (ABM, add_agent!, step!, ...)
    using Agents.Pathfinding  # αλγόριθμοι εύρεσης διαδρομής (AStar, PenaltyMap, plan_best_route!, move_along_route!, ...)
    using Dates  # ημερομηνία/ώρα για το timestamp των αρχείων εξόδου
    using FileIO: load  # φόρτωση εικόνων (heightmap, concentration maps)
    using XLSX  # εγγραφή των αποτελεσμάτων σε αρχείο Excel (xlsx)
    using ImageMagick  # υποστήριξη μορφών εικόνας (jpg, bmp)
    using Images  # επεξεργασία εικόνων (channelview, ...)
    using Random  # γεννήτριες τυχαίων αριθμών (MersenneTwister)
    using Profile  # profiler για την ανάλυση χρόνου εκτέλεσης
    using BenchmarkTools  # εργαλεία μέτρησης απόδοσης (@benchmark)
end

# Ορισμός του agent: συνεχής 2D agent (pos, vel προστίθενται αυτόματα από το ContinuousAgent)
Agents.@agent struct AgentEscapes(ContinuousAgent{2, Float64})
    age::Float64  # ηλικία (έτη)
    mass::Float64  # μάζα (kg), χρησιμοποιείται στις ελαστικές συγκρούσεις
    toxicload::Float64  # συνολικό τρέχον τοξικό φορτίο (άθροισμα TL1+TL2+TL3)
    pathX::Vector{Float64}  # ιστορικό x θέσεων του agent (για το trail στο animation)
    pathY::Vector{Float64}  # ιστορικό y θέσεων του agent (για το trail στο animation)
    TL1::Vector{Float64}  # ιστορικό τοξικού φορτίου για το 1ο σύμπτωμα (οσμή)
    TL2::Vector{Float64}  # ιστορικό τοξικού φορτίου για το 2ο σύμπτωμα (ερεθισμός)
    TL3::Vector{Float64}  # ιστορικό τοξικού φορτίου για το 3ο σύμπτωμα (οίδημα)
end

begin   # Φόρτωση heightmap, fixed PF map (CM4), και TL CMs 1..7  # Φόρτωση του heightmap, του σταθερού χάρτη κόστους pathfinding (CM4) και των CM1..CM7 για το TL
    heightmap_data = load("NADEEN/Maps/Qatargas Map.jpg")  # φόρτωση της εικόνας του χάρτη
    heightmap_data = permutedims(channelview(heightmap_data), [2,3,1])[:,:,1]  # μετατροπή σε πίνακα (γραμμές, στήλες) και κράτημα του 1ου καναλιού
    global heightmap = floor.(Int, convert.(Float64, heightmap_data) * 255)  # κλιμάκωση σε ακέραιες τιμές 0..255
    heightmap = 255 .- heightmap  # αντιστροφή: το μαύρο της εικόνας γίνεται υψηλή τιμή (λευκό στο αντεστραμμένο heightmap, μη βατό στο walkmap) και το λευκό γίνεται 0

    # Fixed map for pathfinding cost (as requested): CM4
    pf_map_data = load("Concentration Maps/4.bmp")  # φόρτωση του concentration map CM4, που χρησιμοποιείται σταθερά για το κόστος του pathfinding
    pf_map_data = permutedims(channelview(pf_map_data), [2,3,1])[:,:,1]  # μετατροπή σε πίνακα (γραμμές, στήλες) και κράτημα του 1ου καναλιού
    global penalty_map = floor.(Int, convert.(Float64, pf_map_data) * 255.0)  # κλιμάκωση σε ακέραιες τιμές 0..255

    # Dynamic maps for TL accumulation only: CM1..CM7
    const NUM_CMS = 7  # πλήθος concentration maps (CM1..CM7)
    cm_list = Vector{Array{Float64,2}}(undef, NUM_CMS)  # λίστα που κρατά τους 7 χάρτες συγκέντρωσης
    for k in 1:NUM_CMS  # φόρτωση κάθε CM
        fn = joinpath("Concentration Maps", string(k) * ".bmp")  # διαδρομή αρχείου k.bmp
        img = load(fn)  # φόρτωση εικόνας
        img = permutedims(channelview(img), [2,3,1])[:,:,1]  # ίδια μετατροπή σε πίνακα όπως στο heightmap
        cm_list[k] = convert.(Float64, img) .* 255.0  # κλιμάκωση τιμών στο 0..255 (Float64)
        @assert size(cm_list[k]) == size(heightmap) "Concentration map $k size mismatch with heightmap"  # έλεγχος ότι το CM έχει τις ίδιες διαστάσεις με το heightmap
    end
    global tl_penalty_map = copy(cm_list[1])  # ενεργός χάρτης συγκέντρωσης για το τοξικό φορτίο (TL), αρχικά το CM1

    @assert size(penalty_map) == size(heightmap) "penalty_map dimensions $(size(penalty_map)) do not match heightmap dimensions $(size(heightmap))"  # έλεγχος ότι το penalty_map (CM4) έχει τις ίδιες διαστάσεις με το heightmap
end

NPM = heightmap .+ penalty_map  # χάρτης κόστους pathfinding = heightmap + CM4 (σταθερός)

begin   # Παράμετροι μοντέλου (match Scenario 3 A* benchmark)
    const METERS_TO_PIXELS = 723.37 / 2500.0  # συντελεστής μετατροπής μέτρων σε pixels του χάρτη
    dt = 1.0  # διακριτό χρονικό βήμα του μοντέλου (1 s)
    n_agents = 5  # πλήθος agents ανά εκτέλεση του benchmark

    age_range = (22, 60)  # εύρος ηλικιών των agents (έτη)
    speed_range = (4.0, 7.0)  # εύρος της αρχικής ταχύτητας vel των agents· η πραγματική ταχύτητα κίνησης προκύπτει από το τοξικό φορτίο
    mass_range = (50, 80)  # εύρος μάζας των agents (kg)
    ag_range_y = (size(heightmap)[1]/4):(3*size(heightmap)[1]/4)  # περιοχή εμφάνισης agents: κεντρικό μισό του χάρτη (από 1/4 έως 3/4 της 1ης διάστασης)
    ag_range_x = (size(heightmap)[2]/4):(3*size(heightmap)[2]/4)  # περιοχή εμφάνισης agents: κεντρικό μισό του χάρτη (από 1/4 έως 3/4 της 2ης διάστασης)
    dims = (size(NPM))  # διαστάσεις του χάρτη
    walkmap = BitArray(trues(dims...))  # walkmap: αρχικά όλα τα cells είναι βατά (true)
    white_threshold = 245  # κατώφλι τιμής πάνω από το οποίο το cell θεωρείται μη βατό
    walkmap[heightmap .> white_threshold] .= false  # σήμανση των λευκών (μη βατών) cells
end

dests = [(500., 854.), (120., 248.)]  # συντεταγμένες (x, y) των δύο εξόδων διαφυγής
space = ContinuousSpace(size(NPM); periodic = false, spacing = 1)  # συνεχής (μη περιοδικός) χώρος με διαστάσεις ίσες με του χάρτη

#
# Helper: true if position is within radius of any goal (TL and movement stop when true)
# Ακτίνα (σε pixels) γύρω από κάθε έξοδο· εντός αυτής ο agent θεωρείται ότι διέφυγε
const goal_radius = 10.0
# Επιστρέφει true αν ο agent βρίσκεται σε απόσταση <= radius από κάποια έξοδο (τότε σταματούν η κίνηση και η συσσώρευση τοξικού φορτίου).
function at_goal(pos, dests, radius = goal_radius)
    px, py = Float64(pos[1]), Float64(pos[2])  # θέση του agent ως Float64
    return minimum(hypot(px - d[1], py - d[2]) for d in dests) <= radius  # απόσταση από την πλησιέστερη έξοδο <= ακτίνα
end


# Ένα βήμα του agent σε κάθε χρονική στιγμή: έλεγχος εξόδου, ενημέρωση τοξικού φορτίου, υπολογισμός ταχύτητας και κίνηση κατά μήκος της διαδρομής A*.
function agent_step!(person, model)
    if at_goal(person.pos, model.goal)  # ο agent έχει φτάσει σε έξοδο: καταγράφει τη θέση και σταματά
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

# Υπολογίζει τους πίνακες Balpha (εκθέτες νόμου ισχύος), Btime (χρόνοι) και Brho (συγκεντρώσεις) από τις τιμές AEGL του H2S για τρία συμπτώματα (οσμή, ερεθισμός, οίδημα), με προέκταση στους ακραίους χρόνους taumin και taumax. Επιστρέφει (Balpha, Btime, Brho').
function setupToxic()
    Atime = [0.0, 2.0, 5.0, 8.0, 15.0, 30.0]  # χρόνοι έκθεσης AEGL σε λεπτά (το πρώτο στοιχείο 0 είναι placeholder)
    Arho = zeros(3, 6)  # πίνακας συγκεντρώσεων AEGL: 3 συμπτώματα x 6 χρόνοι
    Arho[1, 2:6] = [2.87, 2.33, 2.09, 1.81, 1.55]  # συγκεντρώσεις (ppm) για την οσμή
    Arho[2, 2:6] = [202.38, 164.38, 147.74, 128.10, 109.45]  # συγκεντρώσεις (ppm) για τον ερεθισμό
    Arho[3, 2:6] = [286.71, 232.87, 209.30, 181.47, 155.05]  # συγκεντρώσεις (ppm) για το οίδημα
    MW = 34  # μοριακό βάρος του H2S (g/mol)
    Arho *= MW/24.04  # μετατροπή από ppm σε mg/m^3
    Arho = Arho'  # ανάστροφος: γραμμές = χρόνοι, στήλες = συμπτώματα
    Atime = Atime*60  # μετατροπή των χρόνων από λεπτά σε sec
    taumin = 200.  # ελάχιστος χρόνος έκθεσης (sec) για την εμφάνιση AEGL 1, 2 ή 3
    taumax = 86400.  # μέγιστος χρόνος έκθεσης (sec) για την εμφάνιση AEGL 1, 2 ή 3
    Brho = zeros(7,3)  # συγκεντρώσεις: 7 γραμμές (taumin, 5 χρόνοι AEGL, taumax) x 3 συμπτώματα
    Balpha = zeros(7,3)  # εκθέτες νόμου ισχύος (power-law)
    rhomax = zeros(1,3)  # μέγιστη συγκέντρωση ανά σύμπτωμα (προέκταση στο taumin)
    rhomin = zeros(1,3)  # ελάχιστη συγκέντρωση ανά σύμπτωμα (προέκταση στο taumax)
    Btime = zeros(7, 3)  # χρόνοι που αντιστοιχούν στις συγκεντρώσεις του Brho
    # Αρχικοποίηση: αντιγραφή των τιμών AEGL στις γραμμές 2..6 και ορισμός των taumin/taumax στις γραμμές 1 και 7 του Btime
    for k=1:3
        for b=2:6
            Brho[b, k] = Arho[b, k]
            Btime[b, k] = Atime[b]
        end
        Btime[1, k] = taumin  # ο χρόνος της 1ης γραμμής είναι ο taumin
        Btime[7, k] = taumax  # ο χρόνος της 7ης γραμμής είναι ο taumax
    end
    # Υπολογισμός εκθετών νόμου ισχύος μεταξύ διαδοχικών σημείων AEGL
    for k=1:3
        for b=3:6
            if Brho[b-1, k]==Brho[b, k]  # ίδιες διαδοχικές συγκεντρώσεις: ο εκθέτης είναι 0
                Balpha[b, k] = 0.0
            else
                Balpha[b, k] = log(Atime[b]/Atime[b-1])/log(Brho[b-1, k]/Brho[b, k])  # εκθέτης νόμου ισχύος από δύο διαδοχικά σημεία AEGL
            end
        end
        Balpha[2, k] = Balpha[3, k]  # στα άκρα αντιγράφεται ο εκθέτης του πλησιέστερου υπολογισμένου σημείου
        Balpha[1, k] = Balpha[2, k]  # ομοίως για την 1η γραμμή
        Balpha[7, k] = Balpha[6, k]  # ομοίως για την 7η γραμμή
    end
    # Προέκταση των συγκεντρώσεων στα άκρα (taumin και taumax)
    for k=1:3
        if Balpha[3, k]==0  # μηδενικός εκθέτης: η μέγιστη συγκέντρωση ισούται με αυτήν της 2ης γραμμής
            rhomax[k] = Brho[2, k]
            Brho[1, k] = rhomax[k]
        else
            rhomax[k] = Brho[2, k]*(Btime[2, k]/taumin)^(1/Balpha[2, k])  # προέκταση της συγκέντρωσης στον ελάχιστο χρόνο taumin
            Brho[1, k] = rhomax[k]
        end
        if Brho[5, k]==Brho[6, k]  # ίδιες συγκεντρώσεις στο τέλος: η ελάχιστη συγκέντρωση ισούται με αυτήν της 6ης γραμμής
            rhomin[k] = Brho[6, k]
            Brho[7, k] = rhomin[k]
        else
            rhomin[k] = Brho[6, k]*(Btime[6, k]/taumax)^(1/Balpha[6, k])  # προέκταση της συγκέντρωσης στον μέγιστο χρόνο taumax
            Brho[7, k] = rhomin[k]
        end
    end
    # Διόρθωση των χρόνων Btime όπου ο εκθέτης είναι 0 (σταθερή συγκέντρωση)
    for k=1:3
        for b=2:7
            if Balpha[b, k] == 0
                Btime[b, k] = Btime[b-1, k]  # όπου ο εκθέτης είναι 0, ο χρόνος παίρνει την τιμή της προηγούμενης γραμμής
            end
        end
    end
    # Επανυπολογισμός εκθέτη όπου ο προηγούμενος ήταν 0 και ο τρέχων > 0
    for k=1:3
        for b=3:5
            if Balpha[b-1, k]==0 && Balpha[b, k]>0  # ο προηγούμενος εκθέτης ήταν 0 και ο τρέχων είναι θετικός
                Balpha[b, k]=log(Btime[b, k]/Btime[b-1, k])/log(Brho[b-1, k]/Brho[b, k])  # επανυπολογισμός του εκθέτη με τους διορθωμένους χρόνους
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

# --- Benchmark parameters (match Scenario 3 A* benchmark) ---
# Παράμετροι του benchmark
const BENCHMARK_MAX_RUNS = 100  # πλήθος εκτελέσεων (runs) του benchmark
const BENCHMARK_T_STEPS = 2500  # βήματα προσομοίωσης ανά run (dt = 1 s)
const BENCHMARK_TL_CAP_PER_AGENT = 3.0  # άνω όριο του TL ανά agent στο άθροισμα (3 = όριο ακινητοποίησης)
# CM switch times (s) — Thesis, Table 7: 7 CMs, switching completes at 70% of T = 2500 s.
const CM_SWITCH_TIMES = [0.0, 292.0, 584.0, 876.0, 1168.0, 1460.0, 1752.0]  # χρόνοι (s) από τους οποίους ισχύει κάθε CM (ένας ανά CM)
@assert length(CM_SWITCH_TIMES) == NUM_CMS  # έλεγχος ότι υπάρχει χρόνος αλλαγής για κάθε CM

# Εκτελεί ένα run του benchmark: δημιουργεί μοντέλο και agents, μετρά τον χρόνο αρχικού υπολογισμού διαδρομών (A* με σταθερό CM4) και τον χρόνο της προσομοίωσης (με εναλλαγή CM1..CM7 για το TL). Επιστρέφει χρόνους, seed, αθροιστικό TL (capped) και στοιχεία ανά agent.
"""
    run_one_benchmark() -> (initial_pf, incremental_pf, sim_time, seed, sum_TL_capped, final_tl_map_idx, tl_map_switch_count, per_agent_TL_capped, agent_ids)

Scenario 2 keeps pathfinding fixed on CM4 (single `plan_best_route!` pass per agent at start),
while TL accumulation uses dynamic CM switching (CM1→CM7) over time like Scenario 3.
No path replanning is performed, so `incremental_pf` remains zero for spreadsheet parity.
"""
function run_one_benchmark()
    seed = rand(Random.RandomDevice(), UInt32)  # τυχαίο seed από το σύστημα, διαφορετικό σε κάθε run (τοπική μεταβλητή)
    rng_run = MersenneTwister(seed)  # γεννήτρια τυχαίων αριθμών του run

    local_NPM = copy(NPM)  # τοπικό αντίγραφο του χάρτη κόστους (heightmap + CM4)
    local_cost_metric_obj = PenaltyMap(local_NPM, MaxDistance{2}())  # κόστος κίνησης από το τοπικό NPM, με MaxDistance{2}
    local_pathfinderPM = AStar(space; walkmap = walkmap, cost_metric = local_cost_metric_obj)  # A* pathfinder του run (μη βατά cells από το walkmap)
    local_properties = (  # ιδιότητες του μοντέλου του run
        pathfinderPM = local_pathfinderPM,
        heightmap    = heightmap,
        dt           = dt,
        speed_range  = speed_range,
        goal         = dests,
    )

    model_run = StandardABM(  # δημιουργία του μοντέλου του run
        AgentEscapes,
        space;
        rng         = rng_run,
        properties  = local_properties,
        agent_step! = agent_step!,
        model_step! = model_step!,
    )

    for _ in 1:n_agents  # δημιουργία n_agents agents σε τυχαίες βατές θέσεις
        age = rand(abmrng(model_run)) * (age_range[2] - age_range[1]) + age_range[1]  # τυχαία ηλικία μέσα στο age_range
        mass = rand(abmrng(model_run)) * (mass_range[2] - mass_range[1]) + mass_range[1]  # τυχαία μάζα μέσα στο mass_range
        vel = Tuple(rand(abmrng(model_run), 2) .* (speed_range[2] - speed_range[1]) .+ speed_range[1])  # τυχαία αρχική ταχύτητα vel μέσα στο speed_range
        max_attempts = 1000  # μέγιστος αριθμός προσπαθειών εύρεσης βατής θέσης
        attempts = 0  # μετρητής προσπαθειών
        pos = nothing  # η θέση εμφάνισης δεν έχει βρεθεί ακόμα
        while attempts < max_attempts  # επανάληψη μέχρι να βρεθεί βατή θέση
            candidate_pos = Tuple((rand(abmrng(model_run), floor.(ag_range_y)), rand(abmrng(model_run), floor.(ag_range_x))))  # τυχαία υποψήφια θέση μέσα στις περιοχές ag_range_y/ag_range_x
            pos_int = floor.(Int, candidate_pos)  # ακέραιες συντεταγμένες (cell) της υποψήφιας θέσης
            if 1 <= pos_int[1] <= size(walkmap, 1) && 1 <= pos_int[2] <= size(walkmap, 2) && walkmap[pos_int[1], pos_int[2]]  # η θέση είναι εντός ορίων και το cell είναι βατό
                pos = candidate_pos  # δεκτή θέση
                break  # έξοδος από την επανάληψη
            end
            attempts += 1  # αποτυχία: νέα προσπάθεια
        end
        if pos === nothing  # δεν βρέθηκε βατή θέση μετά από max_attempts προσπάθειες
            println("Warning: Could not find valid walkable spawn for agent (seed=$seed)")  # προειδοποίηση στον χρήστη
            continue  # παράλειψη αυτού του agent
        end
        add_agent!(  # προσθήκη agent: αρχικό toxicload 0.0, ιστορικό θέσης με την αρχική θέση, ιστορικά TL1..TL3 με αρχική τιμή 0.0
            pos, AgentEscapes, model_run,
            vel, age, mass, 0.0,
            [pos[1]], [pos[2]],
            [0.0], [0.0], [0.0]
        )
    end

    initial_pf_time = @elapsed begin  # μέτρηση του χρόνου αρχικού υπολογισμού διαδρομών για όλους τους agents
        for a in allagents(model_run)  # για κάθε agent
            plan_best_route!(a, dests, local_pathfinderPM)  # υπολογισμός της καλύτερης διαδρομής προς μία από τις εξόδους (μία φορά, στην αρχή)
        end
    end

    incremental_pf_time = 0.0  # δεν γίνεται επανασχεδιασμός διαδρομών, άρα παραμένει 0
    sim_step_time = 0.0  # συσσωρευτής του χρόνου των βημάτων προσομοίωσης
    current_map_idx = 1  # δείκτης του CM που ισχύει αυτή τη στιγμή
    tl_map_switch_count = 0  # μετρητής αλλαγών CM
    tl_penalty_map .= cm_list[current_map_idx]  # επαναφορά του ενεργού χάρτη TL στο CM1 στην αρχή του run
    for step_idx in 1:BENCHMARK_T_STEPS  # βήματα προσομοίωσης
        t_sim = (step_idx - 1) * dt  # χρόνος προσομοίωσης (s) του τρέχοντος βήματος
        new_idx = searchsortedlast(CM_SWITCH_TIMES, t_sim)  # δείκτης του τελευταίου CM με χρόνο έναρξης <= t_sim
        if new_idx != current_map_idx  # αλλαγή CM μόνο όταν αλλάζει ο δείκτης
            current_map_idx = new_idx  # αποθήκευση του νέου δείκτη
            tl_map_switch_count += 1  # καταμέτρηση της αλλαγής
            tl_penalty_map .= cm_list[current_map_idx]  # αντικατάσταση του ενεργού χάρτη TL (τον διαβάζει η agent_step!)
        end
        sim_step_time += @elapsed step!(model_run, 1)  # μετράται μόνο ο χρόνος του step!, όχι η αλλαγή CM
    end
    simulation_time = sim_step_time  # συνολικός χρόνος προσομοίωσης του run

    agents_sorted = sort(collect(allagents(model_run)), by = a -> a.id)  # agents ταξινομημένοι κατά id
    sum_TL_capped = 0.0  # άθροισμα του TL όλων των agents, με άνω όριο ανά agent
    per_agent_TL_capped = Float64[]  # TL (με άνω όριο) ανά agent
    agent_ids = Int[]  # id των agents
    for a in agents_sorted  # για κάθε agent
        tl_cap = min(a.toxicload, BENCHMARK_TL_CAP_PER_AGENT)  # το TL του agent περιορίζεται στο BENCHMARK_TL_CAP_PER_AGENT
        sum_TL_capped += tl_cap  # προσθήκη στο συνολικό άθροισμα
        push!(per_agent_TL_capped, tl_cap)  # αποθήκευση του TL του agent
        push!(agent_ids, a.id)  # αποθήκευση του id του agent
    end

    return initial_pf_time, incremental_pf_time, simulation_time, seed, sum_TL_capped, current_map_idx, tl_map_switch_count, per_agent_TL_capped, agent_ids  # επιστρέφει χρόνους, seed, αθροιστικό TL, τελικό δείκτη CM, αριθμό αλλαγών CM και τα δεδομένα ανά agent
end

# Καθαρίζει τα δεδομένα του Profile, εκτελεί ένα run του benchmark και εκτυπώνει τα αποτελέσματά του (η ίδια η συνάρτηση δεν καλεί @profile· για profiling πρέπει να κληθεί μέσα από @profile).
"""
    profile_one_benchmark()
Profile one benchmark run. After running, call Profile.print() or your profiler GUI.
"""
function profile_one_benchmark()
    Profile.clear()  # καθαρισμός προηγούμενων δεδομένων profiling
    init_t, incr_t, sim_t, run_seed, sum_cap, final_map_idx, switch_count, _, _ = run_one_benchmark()  # εκτέλεση ενός run (αγνοούνται τα δεδομένα ανά agent)
    println("Profiled run (seed=$run_seed): initial PF = $(round(init_t; digits=4)) s, incremental PF = $(round(incr_t; digits=4)) s, simulation = $(round(sim_t; digits=4)) s, sum TL (capped max $(BENCHMARK_TL_CAP_PER_AGENT) per agent) = $(round(sum_cap; digits=4))")  # εκτύπωση χρόνων του run και αθροιστικού TL
    println("TL CM sequence summary: final map index = $final_map_idx, switches = $switch_count")  # εκτύπωση του τελικού δείκτη CM και του αριθμού αλλαγών
    println("Run Profile.print() or open profiler to inspect hotspots.")  # οδηγία για την προβολή των αποτελεσμάτων του profiler
end

# Μετρά με το BenchmarkTools ένα run του benchmark και επιστρέφει το Trial.
"""
    benchmark_run_one_benchmark(; samples = 10)

Use BenchmarkTools to benchmark a single `run_one_benchmark()` invocation.
Returns the BenchmarkTools Trial object and also prints a summary.
"""
function benchmark_run_one_benchmark(; samples::Int = 10)
    println("Benchmarking run_one_benchmark() with BenchmarkTools (samples = $samples)...")  # ενημέρωση για την έναρξη της μέτρησης
    result = @benchmark run_one_benchmark() samples = samples  # μέτρηση της run_one_benchmark() με το δοσμένο πλήθος samples
    println(result)  # εκτύπωση της σύνοψης
    return result  # επιστροφή του αντικειμένου Trial
end

# --- Benchmark loop ---
# Πίνακες που συγκεντρώνουν τα αποτελέσματα όλων των runs
initial_pf_times   = Float64[]  # χρόνος αρχικού pathfinding ανά run
incremental_pf_times = Float64[]  # χρόνος incremental pathfinding ανά run (πάντα 0 εδώ)
simulation_times   = Float64[]  # χρόνος προσομοίωσης ανά run
sum_TL_capped_values = Float64[]  # αθροιστικό TL (με όριο ανά agent) ανά run
final_tl_map_indices = Int[]  # τελικός δείκτης CM ανά run
tl_map_switch_counts = Int[]  # αριθμός αλλαγών CM ανά run
seeds = UInt32[]  # seed κάθε run
agent_tl_runs = Int[]  # αριθμός run για κάθε εγγραφή TL agent
agent_tl_ids = Int[]  # id agent για κάθε εγγραφή
agent_tl_values = Float64[]  # TL (με όριο) κάθε agent

println("Benchmark: Scenario 2 with A* (static PenaltyMap; pathfinding + simulation). Runs = $BENCHMARK_MAX_RUNS.")  # ενημέρωση για την έναρξη του benchmark
for run_id in 1:BENCHMARK_MAX_RUNS  # επανάληψη για κάθε run
    t_init, t_incr, t_sim, run_seed, sum_TL_capped, final_tl_map_idx, tl_map_switch_count, per_agent_TL_capped, agent_ids = run_one_benchmark()  # εκτέλεση ενός run και λήψη των αποτελεσμάτων
    push!(initial_pf_times,    t_init)  # αποθήκευση αρχικού χρόνου pathfinding
    push!(incremental_pf_times, t_incr)  # αποθήκευση incremental χρόνου pathfinding
    push!(simulation_times,     t_sim)  # αποθήκευση χρόνου προσομοίωσης
    push!(sum_TL_capped_values, sum_TL_capped)  # αποθήκευση αθροιστικού TL
    push!(final_tl_map_indices, final_tl_map_idx)  # αποθήκευση τελικού δείκτη CM
    push!(tl_map_switch_counts, tl_map_switch_count)  # αποθήκευση αριθμού αλλαγών CM
    push!(seeds, run_seed)  # αποθήκευση seed
    for k in eachindex(per_agent_TL_capped)  # εγγραφή του TL κάθε agent
        push!(agent_tl_runs, run_id)  # αριθμός run
        push!(agent_tl_ids, agent_ids[k])  # id agent
        push!(agent_tl_values, per_agent_TL_capped[k])  # TL του agent
    end

    n = length(initial_pf_times)  # πλήθος runs μέχρι τώρα
    avg_initial_pf    = sum(initial_pf_times)    / n  # τρέχων μέσος όρος αρχικού pathfinding
    avg_incremental_pf = sum(incremental_pf_times) / n  # τρέχων μέσος όρος incremental pathfinding
    avg_simulation    = sum(simulation_times)    / n  # τρέχων μέσος όρος χρόνου προσομοίωσης

    println("  Run $run_id (seed=$run_seed): initial PF = $(round(t_init; digits=4)) s, incremental PF = $(round(t_incr; digits=4)) s, simulation = $(round(t_sim; digits=4)) s, sum TL capped = $(round(sum_TL_capped; digits=4))  (avg total PF = $(round(avg_initial_pf + avg_incremental_pf; digits=4)) s)")  # εκτύπωση αποτελεσμάτων του run και του τρέχοντος μέσου χρόνου pathfinding
end

n_runs = length(initial_pf_times)  # συνολικό πλήθος runs
avg_initial_pf_final    = sum(initial_pf_times)    / n_runs  # τελικός μέσος όρος αρχικού pathfinding
avg_incremental_pf_final = sum(incremental_pf_times) / n_runs  # τελικός μέσος όρος incremental pathfinding
avg_pathfinding_final   = avg_initial_pf_final + avg_incremental_pf_final  # τελικός μέσος συνολικός χρόνος pathfinding
avg_simulation_final  = sum(simulation_times)  / n_runs  # τελικός μέσος χρόνος προσομοίωσης

# Εγγραφή των αποτελεσμάτων σε αρχείο Excel
run_timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")  # timestamp για το όνομα του αρχείου
xlsx_path = "SCENARIOS/SCENARIO 2/Simulation Results/AStar $(n_agents) Agents Benchmarking_$(run_timestamp).xlsx"  # διαδρομή του αρχείου xlsx εξόδου
run_ids = collect(1:n_runs)  # αριθμοί των runs
columns_data = [run_ids, seeds, initial_pf_times, incremental_pf_times, simulation_times, sum_TL_capped_values, final_tl_map_indices, tl_map_switch_counts]  # στήλες του φύλλου Runs
column_names = ["Run", "Seed", "InitialPathfindingTime_s", "IncrementalPathfindingTime_s", "SimulationTime_s", "SumTL_capped", "FinalTLMapIndex", "TLMapSwitchCount"]  # ονόματα των στηλών του φύλλου Runs
agent_columns_data = [agent_tl_runs, agent_tl_ids, agent_tl_values]  # στήλες του φύλλου AgentTL
agent_column_names = ["Run", "AgentID", "TL_capped_max3"]  # ονόματα των στηλών του φύλλου AgentTL
XLSX.writetable(xlsx_path;  # δημιουργία του xlsx με δύο φύλλα (Runs, AgentTL), με αντικατάσταση αν υπάρχει
    Runs=(columns_data, column_names),
    AgentTL=(agent_columns_data, agent_column_names),
    overwrite=true,
)
XLSX.openxlsx(xlsx_path, mode = "rw") do xf  # ανοίγει το αρχείο για εγγραφή των συνοπτικών στατιστικών κάτω από τον πίνακα Runs
    sh = xf["Runs"]  # το φύλλο Runs
    sh[n_runs + 2, 1] = "Number of runs"  # ετικέτα
    sh[n_runs + 2, 2] = n_runs  # πλήθος runs
    sh[n_runs + 3, 1] = "Average initial pathfinding time (s)"  # ετικέτα
    sh[n_runs + 3, 2] = avg_initial_pf_final  # μέσος αρχικός χρόνος pathfinding
    sh[n_runs + 4, 1] = "Average incremental pathfinding time (s)"  # ετικέτα
    sh[n_runs + 4, 2] = avg_incremental_pf_final  # μέσος incremental χρόνος pathfinding
    sh[n_runs + 5, 1] = "Average total pathfinding time (s)"  # ετικέτα
    sh[n_runs + 5, 2] = avg_pathfinding_final  # μέσος συνολικός χρόνος pathfinding
    sh[n_runs + 6, 1] = "Average simulation time (s)"  # ετικέτα
    sh[n_runs + 6, 2] = avg_simulation_final  # μέσος χρόνος προσομοίωσης
end

println("Benchmark complete. Results written to $xlsx_path")  # μήνυμα ολοκλήρωσης
println("  Total runs: $n_runs | Avg initial PF: $(round(avg_initial_pf_final; digits=4)) s | Avg incremental PF: $(round(avg_incremental_pf_final; digits=4)) s | Avg simulation: $(round(avg_simulation_final; digits=4)) s")  # σύνοψη μέσων χρόνων
