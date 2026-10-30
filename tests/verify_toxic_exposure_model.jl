# Verification tests για το μοντέλο έκθεσης/απόκρισης (τοξικό φορτίο TL → ταχύτητα του agent).
# Δεν χρειάζεται Agents.jl ή χάρτες: οι συναρτήσεις διαβάζονται ΑΥΤΟΥΣΙΕΣ από το σενάριο, ώστε το test να δένεται με τον πραγματικό κώδικα.
# Τρέξιμο (από τη ρίζα του repo):  julia tests/verify_toxic_exposure_model.jl
#
# T1: η σχέση ταχύτητας–TL του κώδικα έναντι της Εξίσωσης 3 της διπλωματικής (Shaikh et al., 2020, Fig. 3).
# T2: αναπαραγωγή των Test Cases A του Shaikh et al. (2020): ένας agent, ευθεία διαδρομή 300 m, σταθερή συγκέντρωση H2S.
# T3: διαγνωστικό: χρόνος μέχρι TL >= 3 από το άθροισμα των ΜΗ περιορισμένων TL1+TL2+TL3 (όπως στο σενάριο) έναντι περιορισμένων TL_k <= 1.
# T4: συνέπεια tl_rate με τους πίνακες AEGL: σε σταθερή συγκέντρωση Brho[k,b] το TL_k πρέπει να φτάνει το 1 σε χρόνο Btime[b,k].
# T5: μονοτονία της tl_rate ως προς τη συγκέντρωση (περισσότερο αέριο δεν πρέπει να δίνει μικρότερο ρυθμό).
# T6: ευαισθησία των Test Cases A στην παράμετρο taumin (το σενάριο χρησιμοποιεί 200 s).
using Printf

const REPO = normpath(joinpath(@__DIR__, ".."))                                          # ρίζα του repo
const SRC = joinpath(REPO, "SCENARIOS", "SCENARIO 3", "DStarLite - Scenario 3.jl")       # σενάριο από το οποίο διαβάζονται οι συναρτήσεις
const SRC_LINES = readlines(SRC)                                                         # οι γραμμές του σεναρίου (readlines αφαιρεί και τα CRLF)

# Επιστρέφει το κείμενο μιας top-level συνάρτησης (από `function name(` μέχρι το πρώτο `end` στη στήλη 0).
function extract_function(name)
    i = findfirst(l -> startswith(l, "function $name("), SRC_LINES)                      # γραμμή έναρξης
    i === nothing && error("Δεν βρέθηκε η συνάρτηση $name στο $SRC")
    j = findnext(l -> rstrip(l) == "end", SRC_LINES, i)                                  # πρώτο `end` χωρίς εσοχή = τέλος της συνάρτησης
    return join(SRC_LINES[i:j], "\n")
end

# Επιστρέφει το μπλοκ υπολογισμού ταχύτητας μέσα στην agent_step! (από `speed = 1.35` μέχρι το `end` του if).
function extract_speed_block()
    i = findfirst(l -> occursin(r"^\s*speed = 1\.35\s+#", l), SRC_LINES)                 # η γραμμή με την κανονική ταχύτητα
    i === nothing && error("Δεν βρέθηκε το μπλοκ ταχύτητας στο $SRC")
    j = findnext(l -> occursin(r"^\s*end\s*$", l), SRC_LINES, i)                         # κλείσιμο του if/elseif
    return join(SRC_LINES[i:j], "\n")
end

# Χτίζει ένα module με τον κώδικα του σεναρίου (setupToxic, tl_rate, update_toxic_load, speed_code) χωρίς να τρέχει η προσομοίωση.
# Το `taumin` του setupToxic μπορεί να αλλάξει (το σενάριο έχει 200 s) για την ανάλυση ευαισθησίας T6.
function build_model(; taumin = 200.0)
    m = Module(:Model)                                                                   # καινούριο module
    setup_src = extract_function("setupToxic")
    if taumin != 200.0
        patched = replace(setup_src, "taumin = 200." => "taumin = $(taumin)")           # αντικατάσταση της τιμής 200 s
        patched == setup_src && error("Δεν βρέθηκε το `taumin = 200.` στο setupToxic")
        setup_src = patched
    end
    Base.include_string(m, setup_src)
    Base.include_string(m, extract_function("tl_rate"))
    Base.include_string(m, extract_function("update_toxic_load"))
    Base.include_string(m, "Balpha, Btime, Brho = setupToxic()")                         # οι πίνακες που χρησιμοποιεί η tl_rate (όπως στο σενάριο)
    Base.include_string(m, "function speed_code(person)\n" * extract_speed_block() * "\n    return speed\nend")
    return m
end

const M = build_model()                                                                  # το μοντέλο ακριβώς όπως στο σενάριο (taumin = 200 s)
speed_code(TL) = Base.invokelatest(M.speed_code, (toxicload = TL,))                      # ταχύτητα από τον κώδικα (m/s)

# Αποτελέσματα των ελέγχων: (όνομα, πέρασε;)
const RESULTS = Tuple{String,Bool}[]
check(name, ok) = (push!(RESULTS, (name, ok)); println(ok ? "  [PASS] " : "  [FAIL] ", name))

# Εξίσωση 3 της διπλωματικής (Shaikh et al., 2020, Fig. 3): TL ∈ [0,1): 1.35·e^{0.4·TL}, TL ∈ [1,3]: 1.35·(e^{-0.45·(TL-3)} - 1)
speed_thesis(TL) = TL < 1 ? 1.35 * exp(0.4 * TL) : (TL <= 3 ? 1.35 * (exp(-0.45 * (TL - 3)) - 1) : 0.0)

println("== T1: ταχύτητα–TL (κώδικας έναντι Εξίσωσης 3) ==")
@printf("  %6s %10s %10s %10s\n", "TL", "κώδικας", "Εξ. 3", "διαφορά")
maxdiff = 0.0
for TL in 0.0:0.25:3.0                                                                   # πλέγμα TL στο [0,3]
    vc = speed_code(TL)                                                                  # ταχύτητα από τον κώδικα (m/s)
    vt = speed_thesis(TL)                                                                # ταχύτητα από την Εξίσωση 3 (m/s)
    global maxdiff = max(maxdiff, abs(vc - vt))
    @printf("  %6.2f %10.3f %10.3f %10.3f\n", TL, vc, vt, vc - vt)
end
@printf("  μέγιστη απόκλιση = %.3f m/s\n", maxdiff)
check("T1 κώδικας ≈ Εξ. 3 (|Δv| ≤ 0.05 m/s σε όλο το [0,3])", maxdiff <= 0.05)
check("T1 v(0) = 1.35 m/s", isapprox(speed_code(0.0), 1.35; atol = 1e-9))
check("T1 v(TL=3) = 0 (ακινητοποίηση)", speed_code(3.0) == 0.0)

# Προσομοίωση ενός agent σε ευθεία διαδρομή L μέτρων με σταθερή συγκέντρωση (όπως στην agent_step!: πρώτα TL, μετά ταχύτητα, μετά κίνηση).
#   unit  = :raw  → η συγκέντρωση περνά στην tl_rate όπως στο σενάριο (τιμή του CM, 1 μονάδα = 1 ppm κατά τη διπλωματική)
#         = :mgm3 → μετατροπή ppm → mg/m^3 (×34/24.04), δηλαδή στις μονάδες των πινάκων Brho της setupToxic
#   cap   = true  → κάθε TL_k περιορίζεται στο 1 πριν το άθροισμα (TL ∈ [0,3] όπως στο Shaikh), false → όπως στο σενάριο (uncapped)
function run_case(ppm; model = M, unit = :raw, cap = false, L = 300.0, dt = 1.0, tmax = 1500)
    C = unit == :mgm3 ? ppm * 34.0 / 24.04 : float(ppm)                                  # τιμή που δίνεται στην tl_rate
    TL = [0.0, 0.0, 0.0]                                                                 # TL1, TL2, TL3
    x = 0.0                                                                              # διανυθείσα απόσταση (m)
    t1 = t2 = t3 = NaN                                                                   # χρόνοι πρώτης υπέρβασης TL ≥ 1, 2, 3
    for t in 1:tmax
        TL = Base.invokelatest(model.update_toxic_load, C, copy(TL), dt)                 # νέο TL μετά από έκθεση dt
        TLs = cap ? sum(min.(TL, 1.0)) : sum(TL)                                         # συνολικό TL
        isnan(t1) && TLs >= 1 && (t1 = t)
        isnan(t2) && TLs >= 2 && (t2 = t)
        isnan(t3) && TLs >= 3 && (t3 = t)
        x += Base.invokelatest(model.speed_code, (toxicload = TLs,)) * dt                # κίνηση με ταχύτητα v(TL)
        x >= L && return (exit = float(t), t1 = t1, t2 = t2, t3 = t3, TL = TLs, x = L)
    end
    return (exit = NaN, t1 = t1, t2 = t2, t3 = t3, TL = cap ? sum(min.(TL, 1.0)) : sum(TL), x = x)
end

# Τιμές αναφοράς από το Shaikh et al. (2020), §4.1 (Test Cases A, ευθεία διαδρομή 300 m):
#   A1 (0 ppm): έξοδος σε ~225 s · A2 (5 ppm) και A3 (140 ppm): έξοδος σε ~160 s ·
#   A4 (450 ppm): TL ≈ 3 στα ~80 s, αποτυχία στα ~130 s (ο agent δεν φτάνει στην έξοδο).
const SHAIKH = Dict(0 => 225.0, 5 => 160.0, 140 => 160.0)

println("\n== T2: Shaikh et al. (2020), Test Cases A (300 m, σταθερό H2S) — στήλες: έξοδος · χρόνοι υπέρβασης TL ≥ 1/2/3 · θέση στο τέλος ==")
@printf("  %-18s %5s %9s %7s %7s %7s %10s\n", "παραλλαγή", "ppm", "έξοδος(s)", "TL≥1", "TL≥2", "TL≥3", "x_τέλος(m)")
for (unit, cap) in ((:raw, false), (:raw, true), (:mgm3, false), (:mgm3, true))          # 4 παραλλαγές μονάδων/περιορισμού
    label = "$(unit), cap=$(cap)"
    for ppm in (0, 5, 140, 450)
        r = run_case(ppm; unit, cap)
        @printf("  %-18s %5d %9.0f %7.0f %7.0f %7.0f %10.1f\n", label, ppm, r.exit, r.t1, r.t2, r.t3, r.x)
        if haskey(SHAIKH, ppm) && unit == :raw && !cap                                   # έλεγχος μόνο για την παραλλαγή του σεναρίου (raw, uncapped)
            ok = !isnan(r.exit) && abs(r.exit - SHAIKH[ppm]) <= 0.15 * SHAIKH[ppm]       # ανοχή ±15% (το Shaikh έχει δυναμική κοινωνικών δυνάμεων, τ = 1 s)
            check("T2 A($(ppm) ppm): έξοδος ≈ $(SHAIKH[ppm]) s ±15% (κώδικας: $(r.exit) s)", ok)
        end
        if ppm == 450 && unit == :raw && !cap
            check("T2 A4 (450 ppm): ο agent ΔΕΝ φτάνει στην έξοδο", isnan(r.exit))
            check("T2 A4 (450 ppm): TL ≥ 3 μέσα στα πρώτα ~100 s (Shaikh: ~80 s)", !isnan(r.t3) && r.t3 <= 100)
        end
    end
end

println("\n== T3: διαγνωστικό — χρόνος μέχρι TL ≥ 3 (άθροισμα uncapped έναντι capped) ==")
@printf("  %6s %14s %14s\n", "ppm", "uncapped (s)", "capped (s)")
for ppm in (1, 2, 5, 10, 25, 50, 100)
    ru = run_case(ppm; cap = false, L = 1.0e9, tmax = 3000)                              # L τεράστιο: ο agent δεν «βγαίνει» ποτέ, μετράμε μόνο το TL
    rc = run_case(ppm; cap = true, L = 1.0e9, tmax = 3000)
    @printf("  %6d %14s %14s\n", ppm, isnan(ru.t3) ? ">3000" : string(Int(ru.t3)), isnan(rc.t3) ? ">3000" : string(Int(rc.t3)))
end

println("\n== T4: συνέπεια tl_rate με τους πίνακες AEGL (σε σταθερό Brho[k,b]: TL_k = 1 τη χρονική στιγμή Btime[b,k]) ==")
@printf("  %3s %3s %12s %12s %12s\n", "k", "b", "Brho(mg/m3)", "Btime(s)", "1/ρυθμός(s)")
bad = 0
for k in 1:3, b in 2:6
    rho = M.Brho[k, b]                                                                   # συγκέντρωση του σημείου AEGL b για το σύμπτωμα k
    target = M.Btime[b, k]                                                               # χρόνος που πρέπει να χρειάζεται για TL_k = 1
    rate = Base.invokelatest(M.tl_rate, rho, k)                                          # ρυθμός συσσώρευσης TL_k
    t_hit = 1 / rate                                                                     # χρόνος μέχρι TL_k = 1 με σταθερό ρυθμό
    @printf("  %3d %3d %12.2f %12.0f %12.1f\n", k, b, rho, target, t_hit)
    abs(t_hit - target) / target > 0.02 && (global bad += 1)
end
check("T4 σε όλα τα σημεία AEGL ο χρόνος TL_k = 1 ισούται με Btime (±2%)  [αποκλίσεις: $bad]", bad == 0)

println("\n== T5: μονοτονία tl_rate ως προς τη συγκέντρωση ==")
nviol = 0                                                                                # πλήθος παραβιάσεων μονοτονίας
first_viol = nothing                                                                     # πρώτη παραβίαση (k, rho)
for k in 1:3
    prev = 0.0                                                                           # ρυθμός στο προηγούμενο σημείο του πλέγματος
    for rho in 0.0:0.05:800.0
        r = Base.invokelatest(M.tl_rate, rho, k)
        if r < prev - 1e-12
            global nviol += 1
            if first_viol === nothing
                global first_viol = (k, rho)
            end
        end
        prev = r
    end
end
println("  παραβιάσεις = $nviol", first_viol === nothing ? "" : "  (πρώτη: k=$(first_viol[1]), ρ=$(first_viol[2]) mg/m3)")
check("T5 η tl_rate είναι μη φθίνουσα ως προς τη συγκέντρωση", nviol == 0)

println("\n== T6: ευαισθησία των Test Cases A στο taumin (έξοδος σε s · χρόνος TL ≥ 3) ==")
@printf("  %8s %14s %14s %14s %16s\n", "taumin", "5 ppm", "140 ppm", "450 ppm έξοδος", "450 ppm TL≥3")
for taumin in (200.0, 120.0, 60.0, 30.0)
    m = build_model(; taumin)                                                            # μοντέλο με άλλο taumin
    r5, r140, r450 = (run_case(p; model = m) for p in (5, 140, 450))
    @printf("  %8.0f %14.0f %14.0f %14.0f %16.0f\n", taumin, r5.exit, r140.exit, r450.exit, r450.t3)
end
println("  (Shaikh et al.: 5 ppm ≈ 160 s, 140 ppm ≈ 160 s, 450 ppm: TL ≈ 3 στα ~80 s και αποτυχία στα ~130 s)")

println("\n== Σύνοψη ==")
nfail = count(!last, RESULTS)
println("  $(length(RESULTS) - nfail)/$(length(RESULTS)) έλεγχοι πέρασαν")
nfail > 0 && exit(1)
