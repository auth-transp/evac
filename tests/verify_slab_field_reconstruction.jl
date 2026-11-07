# Verification tests για το scripts/slab_to_concentration_maps.jl (ανασύνθεση 2D πεδίου συγκέντρωσης από την έξοδο του SLAB).
# Δεν χρειάζεται Agents.jl, χάρτες ή το SLAB: χρησιμοποιεί το αποθηκευμένο παράδειγμα tests/fixtures/slab_h2s/{input,predict}
# (SLAB 1990 έκδοση EPA SCRAM, διαρροή H2S: idspl=2, 10 kg/s, 600 s, ουδέτερη ευστάθεια D, άνεμος 4 m/s).
# Τρέξιμο (από τη ρίζα του repo):  julia tests/verify_slab_field_reconstruction.jl
#
# S1: ακρίβεια των erf/erfc (γνωστές τιμές + ανεξάρτητη αριθμητική ολοκλήρωση + συνέχεια στο σημείο αλλαγής μεθόδου).
# S2: ανάγνωση της εξόδου του SLAB (αριθμός γραμμών, μονοτονία, πρώτη/τελευταία γραμμή, παράμετροι εισόδου).
# S3: ανασύνθεση έναντι του πίνακα «concentration in the z = 2.00 plane» που τυπώνει το ίδιο το SLAB (μέγιστο στον χρόνο, y/bbc = 0 … 2.5).
#     Προσοχή: το SLAB τυπώνει εκεί ΜΕΣΟΡΟΥΜΕΝΗ στον χρόνο τιμή με άλλον τύπο (cvbrx), άρα οι δύο υπολογισμοί συμφωνούν μόνο προσεγγιστικά·
#     τα όρια ανοχής είναι εμπειρικά (παρατηρήθηκε −1% … +4% στον άξονα, μέχρι ±10% στην ουρά y/bbc = 2,5) και ελέγχουν συνέπεια, ΟΧΙ επικύρωση έναντι πειραμάτων.
# S4: γεωμετρία του χάρτη: μηδενικό ανάντη της πηγής, συμμετρία ως προς τον άξονα του ανέμου, περιστροφή 90°/45°, μονάδες m/pixel.
# S5: χρονική εξέλιξη (πριν τη διαρροή = 0, προέκταση πέρα από τον τελευταίο χρόνο του πίνακα), αποθήκευση/ανάγνωση Float32.
# S6: ο δημιουργός αρχείου εισόδου slab_input_text αναπαράγει αριθμητικά το fixture `input`· το CLI γράφει τα αναμενόμενα αρχεία.
# S7: σενάριο πυκνού acid gas (scripts/slab_acid_gas_cases.jl): ιδιότητες μίγματος, ψευδο-πηγή choked flow, περιπτώσεις, είσοδος έναντι fixture, τάξη μεγέθους έναντι Gaussian.
using Printf

include(joinpath(@__DIR__, "..", "scripts", "slab_to_concentration_maps.jl"))            # το module SlabFields (το main του δεν τρέχει όταν γίνεται include)
using .SlabFields
const SF = SlabFields
include(joinpath(@__DIR__, "..", "scripts", "slab_acid_gas_cases.jl"))                   # το module AcidGasCases (έχει δικό του αντίγραφο του SlabFields)
const AG = AcidGasCases

const FIX = joinpath(@__DIR__, "fixtures", "slab_h2s")                                   # φάκελος με το παράδειγμα
const PREDICT = joinpath(FIX, "predict")
const INPUT = joinpath(FIX, "input")

# Αποτελέσματα των ελέγχων: (όνομα, πέρασε;)
const RESULTS = Tuple{String,Bool}[]
check(name, ok) = (push!(RESULTS, (name, ok)); println(ok ? "  [PASS] " : "  [FAIL] ", name))

# ---------------------------------------------------------------------------------------------
println("== S1: ακρίβεια erf / erfc ==")
# Ανεξάρτητη αναφορά: κανόνας Simpson στο 2/√π ∫₀ˣ e^{−s²} ds με 4000 διαστήματα
function erf_simpson(x::Float64; n::Int = 4000)
    h = x / n
    f(s) = exp(-s * s)
    acc = f(0.0) + f(x)
    for k in 1:n-1
        acc += (isodd(k) ? 4.0 : 2.0) * f(k * h)
    end
    return 2 / sqrt(pi) * acc * h / 3
end
xs = [0.05, 0.1, 0.3, 0.5, 0.8, 1.0, 1.3, 1.7, 2.0, 2.3, 2.49]
err_simpson = maximum(abs(SF.erf_scalar(x) - erf_simpson(x)) for x in xs)
@printf("  max |erf − Simpson| για x ∈ [0.05, 2.49] = %.2e\n", err_simpson)
check("S1a erf συμφωνεί με αριθμητική ολοκλήρωση (< 1e-12)", err_simpson < 1e-12)

known = [(:erf, 0.1, 0.1124629160182849), (:erf, 0.5, 0.5204998778130465), (:erf, 1.0, 0.8427007929497149),
         (:erf, 2.0, 0.9953222650189527), (:erfc, 3.0, 2.209049699858544e-5), (:erfc, 5.0, 1.5374597944280349e-12),
         (:erfc, 6.0, 2.1519736712498913e-17)]                                          # τιμές από πίνακες (Abramowitz–Stegun)
worst_rel = 0.0
for (f, x, ref) in known
    val = f === :erf ? SF.erf_scalar(x) : SF.erfc_scalar(x)
    global worst_rel = max(worst_rel, abs(val / ref - 1))
end
@printf("  μέγιστο σχετικό σφάλμα έναντι γνωστών τιμών (και ουρές erfc) = %.2e\n", worst_rel)
check("S1b γνωστές τιμές erf/erfc, μαζί με τις ουρές (σχετικό σφάλμα < 1e-10)", worst_rel < 1e-10)

odd_err = maximum(abs(SF.erf_scalar(-x) + SF.erf_scalar(x)) for x in 0.0:0.37:5.0)
check("S1c erf περιττή συνάρτηση", odd_err == 0.0)

# Συνέχεια στο σημείο αλλαγής μεθόδου (2,5): σειρά έναντι συνεχούς κλάσματος
jump = abs(SF.erf_series(2.5) - (1 - SF.erfc_cf(2.5)))
@printf("  |σειρά − συνεχές κλάσμα| στο x = 2.5: %.2e\n", jump)
check("S1d οι δύο μέθοδοι ταυτίζονται στο όριο 2,5 (< 1e-13)", jump < 1e-13)

# Η διαφορά erf(a) − erf(b) στις ουρές δεν πρέπει να χάνει ακρίβεια (το απλό erf(a) − erf(b) δίνει 0 ή θόρυβο)
tail = SF.erf_diff(8.0, 7.0)                                                             # = erfc(7) − erfc(8) ≈ 4.18e-23
tail_ref = SF.erfc_scalar(7.0) - SF.erfc_scalar(8.0)
check("S1e διαφορά στις ουρές χωρίς απώλεια ακρίβειας", tail > 0 && abs(tail / tail_ref - 1) < 1e-12 && SF.erf_diff(-7.0, -8.0) ≈ tail)

# ---------------------------------------------------------------------------------------------
println("\n== S2: ανάγνωση της εξόδου του SLAB ==")
c = SF.parse_slab_output(PREDICT)
@printf("  γραμμές πίνακα = %d · tav = %.0f s · tsd = %.0f s · xffm = %.0f m\n", length(c.x), c.tav, c.tsd, c.xffm)
check("S2a 61 γραμμές", length(c.x) == 61)
check("S2b αύξοντα x και t", issorted(c.x) && issorted(c.t))
check("S2c πρώτη γραμμή (x = 1, cc = 0)", c.x[1] == 1.0 && c.cc[1] == 0.0 && c.b[1] ≈ 0.101)
check("S2d τελευταία γραμμή (x = 3120 m, cc = 5.92E-4, t = 991 s)",
      c.x[end] == 3120.0 && c.cc[end] ≈ 5.92e-4 && c.t[end] ≈ 991.0 && c.bx[end] ≈ 1330.0 && c.betax[end] ≈ 518.0)
check("S2e παράμετροι εισόδου (tav = 10 s, tsd = 600 s, xffm = 3000 m)", c.tav == 10.0 && c.tsd == 600.0 && c.xffm == 3000.0)
check("S2f όλες οι τιμές πεπερασμένες και μη αρνητικές", all(v -> all(isfinite, v) && all(>=(0), v),
      (c.cc, c.b, c.betac, c.zc, c.sig, c.t, c.bx, c.betax)))
check("S2g αριθμοί Fortran χωρίς «E» (1.23-105 → 1.23E-105)", SF.parse_fnum("1.23-105") == 1.23e-105 && SF.parse_fnum("4.5E+02") == 450.0)

# ---------------------------------------------------------------------------------------------
println("\n== S3: ανασύνθεση έναντι του πίνακα επιπέδου z = 2 m του SLAB ==")
lines = readlines(PREDICT)
i0 = findfirst(l -> occursin("concentration in the z", l), lines)
plane = Vector{Float64}[]                                                                # γραμμές: x, χρόνος μεγίστου, διάρκεια νέφους, bbc, 6 συγκεντρώσεις
for l in lines[i0+1:end]
    v = SF.numbers_in(l)
    length(v) == 10 || (isempty(plane) ? continue : break)
    push!(plane, v)
end
check("S3a 61 γραμμές στον πίνακα επιπέδου", length(plane) == 61)

# Μέγιστο στον χρόνο του c(x, 0, 2, t) με σάρωση χρόνου· το πλατό της μόνιμης πλούμας είναι επίπεδο, οπότε αρκεί λεπτό πλέγμα χρόνου
function peak_state(c, x, z)
    best, bst = 0.0, nothing
    for t in range(c.t[1], c.t[end] + 2000.0, length = 20001)
        st = SF.cloud_state(c, t)
        v = SF.concentration_at(c, x, 0.0, z, st)
        if v > best
            best, bst = v, st
        end
    end
    return bst
end

devs = [Float64[] for _ in 1:6]                                                          # σχετικές αποκλίσεις ανά y/bbc
for r in plane
    x, bbc, tab = r[1], r[4], r[5:10]
    tab[1] < 1e-6 && continue                                                            # αγνοούνται οι γραμμές εγγύς πεδίου με αμελητέα συγκέντρωση
    st = peak_state(c, x, 2.0)
    for k in 1:6
        tab[k] > 1e-12 || continue
        mine = SF.concentration_at(c, x, 0.5 * (k - 1) * bbc, 2.0, st)
        push!(devs[k], mine / tab[k] - 1)
    end
end
tols = [0.05, 0.05, 0.06, 0.07, 0.09, 0.12]                                             # εμπειρικά όρια ανά y/bbc (βλ. επικεφαλίδα)
@printf("  %-8s %5s %10s %10s %10s   %s\n", "y/bbc", "n", "min", "max", "rms", "όριο")
for k in 1:6
    d = devs[k]
    @printf("  %-8.1f %5d %+10.4f %+10.4f %10.4f   ±%.2f\n", 0.5 * (k - 1), length(d), minimum(d), maximum(d), sqrt(sum(abs2, d) / length(d)), tols[k])
end
check("S3b αρκετές γραμμές με μη αμελητέα συγκέντρωση", length(devs[1]) >= 40)
for k in 1:6
    check(@sprintf("S3c y/bbc = %.1f: |απόκλιση| ≤ %.0f%% σε όλα τα x", 0.5 * (k - 1), 100 * tols[k]), maximum(abs, devs[k]) <= tols[k])
end
check("S3d στον άξονα το rms ≤ 2,5% (συστηματική συμφωνία)", sqrt(sum(abs2, devs[1]) / length(devs[1])) <= 0.025)

# ---------------------------------------------------------------------------------------------
println("\n== S4: γεωμετρία του χάρτη ==")
dims = (201, 201)                                                                        # τετράγωνο πλέγμα ώστε οι περιστροφές 90° να αντιστοιχούν σε αντιμετάθεση δεικτών
src = (101, 101)                                                                         # πηγή στο κέντρο
mpp = 2.0                                                                                # 2 m ανά pixel (κάλυψη ±200 m)
tq = 200.0                                                                               # s
A0 = SF.concentration_map(c; dims, src, wind_deg = 0.0, t = tq, z = 2.0, m_per_px = mpp)
A90 = SF.concentration_map(c; dims, src, wind_deg = 90.0, t = tq, z = 2.0, m_per_px = mpp)
A45 = SF.concentration_map(c; dims, src, wind_deg = 45.0, t = tq, z = 2.0, m_per_px = mpp)
@printf("  μέγιστο A0 = %.1f ppm · pixels > 100 ppm: %d\n", maximum(A0), count(>(100), A0))
check("S4a το πεδίο δεν είναι κενό και περιέχει συγκεντρώσεις > 100 ppm", maximum(A0) > 100)
check("S4b μηδέν ανάντη της πηγής (άνεμος +i: όλα τα i < i0)", all(iszero, A0[1:src[1]-1, :]))
check("S4c συμμετρία ως προς τον άξονα του ανέμου (A0[i, j0+d] = A0[i, j0−d])", A0[:, src[2]+1:end] == reverse(A0[:, 1:src[2]-1], dims = 2))
check("S4d περιστροφή 90°: A90[i0+a, j0+b] = A0[i0+b, j0+a]", all(A90[src[1]+a, src[2]+b] == A0[src[1]+b, src[2]+a] for a in -100:100, b in -100:100))
sym45 = all(A45[src[1]+a, src[2]+b] ≈ A45[src[1]+b, src[2]+a] for a in -100:100, b in -100:100)
check("S4e άνεμος 45°: συμμετρία ως προς τη διαγώνιο i − i0 = j − j0", sym45)
# Μονάδες: ο άξονας του νέφους στη θέση x = 100 m (50 pixels) πρέπει να έχει την τιμή του c(100, 0, 2, 200 s)
ref100 = 1e6 * SF.concentration_at(c, 100.0, 0.0, 2.0, SF.cloud_state(c, tq))
check(@sprintf("S4f μονάδες m/pixel: A0 σε 50 pixels (= 100 m) ισούται με τον τύπο (%.2f ppm)", ref100), A0[src[1]+50, src[2]] ≈ Float32(ref100))
cx_pix = argmax(A0[:, src[2]]) - src[1]                                                 # θέση του μεγίστου στον άξονα, σε pixels από την πηγή
check("S4g το μέγιστο στον άξονα βρίσκεται κατά τον άνεμο (>0 pixels)", cx_pix > 0)
A0s = SF.concentration_map(c; dims, src, wind_deg = 0.0, t = tq, z = 2.0, m_per_px = mpp, scale = 0.4)
check("S4h scale = 0.4 (κλάσμα H2S στο μίγμα) πολλαπλασιάζει τον χάρτη", A0s ≈ 0.4f0 .* A0 && maximum(A0s) > 0)

# ---------------------------------------------------------------------------------------------
println("\n== S5: χρονική εξέλιξη και αποθήκευση ==")
check("S5a πριν τη διαρροή (t < t₁) το πεδίο είναι μηδέν", maximum(SF.concentration_map(c; dims = (21, 21), src = (11, 11), wind_deg = 0.0, t = -1.0)) == 0.0)
# Αφού περάσει το νέφος (t = 900 s, ουρά πάνω από 600 s μετά την άφιξη) η συγκέντρωση στο σημείο των 100 m πρέπει να έχει πέσει κατά >2 τάξεις μεγέθους.
# Δεν είναι μηδέν: στο t = 900 s το βx έχει κορεστεί (1,33 km) ενώ το βetax είναι ~440 m, άρα erf(xa) − erf(xb) ≈ 0,008 έναντι 2 μέσα στο νέφος (≈0,4%).
# Αυτή η «ουρά» είναι ιδιότητα του εξομαλυμένου κλειστού τύπου του SLAB και όχι σφάλμα της ανασύνθεσης.
c_early = SF.concentration_ppm(c, 100.0, 0.0, 2.0, 200.0)
c_late = SF.concentration_ppm(c, 100.0, 0.0, 2.0, 900.0)
@printf("  x = 100 m: t = 200 s → %.1f ppm · t = 900 s → %.3g ppm (λόγος %.2e)\n", c_early, c_late, c_late / c_early)
check("S5b το σημείο των 100 m: >100 ppm όσο διαρκεί η διαρροή και <1% αφού περάσει το νέφος", c_early > 100 && c_late < 0.01 * c_early)
# Προέκταση: για t > t_last (991 s) το νέφος συνεχίζει να κινείται· η τιμή πρέπει να παραμένει πεπερασμένη και μη αρνητική
vals_ext = [SF.concentration_ppm(c, x, 0.0, 2.0, 1500.0) for x in 0.0:100.0:3000.0]
check("S5c προέκταση πέρα από τον πίνακα: πεπερασμένες, μη αρνητικές τιμές", all(isfinite, vals_ext) && all(>=(0), vals_ext))
tmp = mktempdir()
rt = SF.concentration_map(c; dims = (60, 70), src = (10, 35), wind_deg = 30.0, t = 300.0, m_per_px = 5.0)
SF.save_concentration_map(joinpath(tmp, "rt.f32"), rt)
check("S5d αποθήκευση/ανάγνωση Float32 χωρίς απώλεια", SF.load_concentration_map(joinpath(tmp, "rt.f32"), (60, 70)) == rt && filesize(joinpath(tmp, "rt.f32")) == 4 * 60 * 70)

# ---------------------------------------------------------------------------------------------
println("\n== S6: δημιουργός εισόδου SLAB και γραμμή εντολών ==")
txt = SF.slab_input_text(; idspl = 2, wms = 0.03408, cps = 1000.0, tbp = 212.8, cmed0 = 0.0, dhe = 548000.0, cpsl = 2200.0, rhosl = 950.0,
                         spb = -1.0, spc = 0.0, ts = 300.0, qs = 10.0, as = 0.05, tsd = 600.0, qtis = 0.0, hs = 1.0, tav = 10.0, xffm = 3000.0,
                         zp = (2.0, 0.0, 0.0, 0.0), z0 = 0.03, za = 10.0, ua = 4.0, ta = 305.0, rh = 50.0, stab = 4.0)
mine_nums = [parse(Float64, strip(l)) for l in split(strip(txt), '\n')]
fix_nums = [parse(Float64, strip(l)) for l in readlines(INPUT) if !isempty(strip(l))]
check("S6a slab_input_text ≡ fixture input (αριθμητικά, 30 γραμμές)", mine_nums == fix_nums && length(fix_nums) == 30)
check("S6b stab = 0 απαιτεί ala", try SF.slab_input_text(; idspl = 2, wms = 1.0, cps = 1.0, tbp = 1.0, dhe = 1.0, cpsl = 1.0, rhosl = 1.0, ts = 1.0, qs = 1.0, as = 1.0, tsd = 1.0, tav = 1.0, xffm = 1.0, z0 = 1.0, za = 1.0, ua = 1.0, ta = 1.0, rh = 1.0, stab = 0.0); false catch; true end)

outdir = joinpath(tmp, "cli")
cd(tmp) do
    SF.main([PREDICT, outdir, "--source=450,500", "--wind=20", "--times=100,300,700", "--z=2.0"])
end
files = sort(readdir(outdir))
check("S6c το CLI γράφει cm_1..3.f32 και maps_meta.txt", files == ["cm_1.f32", "cm_2.f32", "cm_3.f32", "maps_meta.txt"])
check("S6d μέγεθος αρχείου = 4·915·983 bytes", all(filesize(joinpath(outdir, "cm_$k.f32")) == 4 * 915 * 983 for k in 1:3))
back = SF.load_concentration_map(joinpath(outdir, "cm_2.f32"))
direct = SF.concentration_map(c; src = (450, 500), wind_deg = 20.0, t = 300.0, z = 2.0)
check("S6e τα αρχεία του CLI ισούνται με την άμεση κλήση concentration_map", back == direct && maximum(back) > 0)
rm(tmp; recursive = true, force = true)

# ---------------------------------------------------------------------------------------------
println("\n== S7: σενάριο πυκνού acid gas (H2S + CO2) ==")
M_AIR = 0.02896                                                                          # kg/mol (όπως στο SLAB)
mix = AG.mixture(0.40)
check("S7a καθαρό H2S: M = 0,03408 kg/mol και κλάσμα μάζας 1", AG.mixture(1.0).M == 0.03408 && AG.mixture(1.0).w_h2s == 1.0)
check("S7b x_H2S = 0,4: M = 0,040038 kg/mol, κλάσμα μάζας H2S = 0,3405 και βαρύτερο από τον αέρα", isapprox(mix.M, 0.040038; atol = 1e-6) && isapprox(mix.w_h2s, 0.34049; atol = 1e-4) && mix.M > M_AIR)
check("S7c γ = cp/(cp − R) εντός του εύρους H2S/CO2 (1,29 … 1,35)", 1.29 < mix.gamma < 1.35)

src = AG.choked_pseudo_source(260.0, mix, 300.15)
csound = sqrt(mix.gamma * AG.R_GAS * src.Ts / mix.M)                                     # ταχύτητα ήχου στη θερμοκρασία εξόδου
check("S7d choked flow: η ταχύτητα εξόδου ισούται με την ταχύτητα ήχου στη θερμοκρασία εξόδου", src.us ≈ csound)
check("S7e θερμοκρασία εξόδου = 2·T0/(γ+1) < T0 και πυκνότητα στην ατμοσφαιρική πίεση", src.Ts ≈ 2 * 300.15 / (mix.gamma + 1) && src.Ts < 300.15 &&
      src.rhos ≈ 101325.0 * mix.M / (AG.R_GAS * src.Ts))
check("S7f διατήρηση μάζας: ρ·u·A = qs", src.rhos * src.us * src.area ≈ 260.0)

cases = AG.release_cases()
names = [x.name for x in cases]
check("S7g 10 περιπτώσεις (2 εντάσεις × {C3, C5, D3, D5, F3}) και μοναδικά ονόματα", length(cases) == 10 && length(unique(names)) == 10 && !any(occursin("_F_5ms", n) for n in names))
check("S7h η παροχή μάζας H2S είναι 0,77 και 88,55 kg/s για κάθε περίπτωση", all(x.qs * x.mix.w_h2s ≈ x.qh2s for x in cases) &&
      Set(round.([x.qh2s for x in cases]; digits = 2)) == Set([0.77, 88.55]))

# Η είσοδος που παράγεται για την περίπτωση small/D/5 m/s ταυτίζεται με το αποθηκευμένο fixture (προστασία από ακούσιες αλλαγές στις υποθέσεις)
fixdir = joinpath(@__DIR__, "fixtures", "slab_acid_gas")
case = only(filter(x -> x.name == "ag_small_D_5ms", cases))
check("S7i η είσοδος ag_small_D_5ms ταυτίζεται με το fixture", AG.case_input_text(case) == read(joinpath(fixdir, "ag_small_D_5ms.in"), String))

# Τάξη μεγέθους: το SLAB (μακρινό πεδίο, 1 km) έναντι απλής Gaussian πλούμας Briggs για ουδέτερη ατμόσφαιρα, πηγή στο έδαφος. Ελέγχει μονάδες και x_H2S, όχι ακρίβεια.
cd2 = SF.parse_slab_output(joinpath(fixdir, "ag_small_D_5ms.predict"))
ppm_h2s = 0.40 * SF.concentration_ppm(cd2, 1000.0, 0.0, 2.0, 700.0)                      # t = 700 s: το νέφος έχει περάσει το x = 1 km (άφιξη ≈ 200 s, διάρκεια 600 s)
σy = 0.08 * 1000 / sqrt(1 + 0.1)                                                         # Briggs ανοιχτή ύπαιθρος, D: σy = 0,08x/√(1+10⁻⁴x)
σz = 0.06 * 1000 / sqrt(1 + 1.5)                                                         #                              σz = 0,06x/√(1+1,5·10⁻³x)
gauss_ppm = 0.77 / (pi * 5.0 * σy * σz) / 0.03408 * (8.31431 * 300.15 / 101325.0) * 1e6 # kg/m³ → mol/m³ → ppm κατά όγκο
@printf("  x = 1 km, D, 5 m/s, 0,77 kg/s H2S: SLAB %.1f ppm · Gaussian %.1f ppm · λόγος %.2f\n", ppm_h2s, gauss_ppm, ppm_h2s / gauss_ppm)
check("S7j SLAB και Gaussian στο 1 km διαφέρουν λιγότερο από παράγοντα 5", 0.2 < ppm_h2s / gauss_ppm < 5)

println("\n== Σύνοψη ==")
nfail = count(!last, RESULTS)
println("  $(length(RESULTS) - nfail)/$(length(RESULTS)) έλεγχοι πέρασαν")
nfail > 0 && exit(1)
