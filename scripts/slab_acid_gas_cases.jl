# Δημιουργία των εισόδων SLAB για το σενάριο ΠΥΚΝΟΥ acid gas (μίγμα H2S + CO2, βαρύτερο από τον αέρα) και, προαιρετικά, εκτέλεση του SLAB.
#
# ΥΠΟΘΕΣΕΙΣ (δεν προέρχονται από δεδομένα της εγκατάστασης· κάθε μία αλλάζει με επιλογή στη γραμμή εντολών, βλ. «Χρήση» παρακάτω):
#   - σύνθεση: x_H2S = 40% mol H2S, υπόλοιπο CO2 (--xh2s=…)· τυπικό εύρος αερίου από μονάδα αφαίρεσης οξέων, ΟΧΙ μετρημένη σύνθεση Qatargas
#   - ρυθμός διαρροής: ίδιος ρυθμός ΜΑΖΑΣ H2S με τα σενάρια sour gas της ομάδας (0,77% w/w H2S σε μεθάνιο): μικρή 100 kg/s × 0,77% = 0,77 kg/s H2S (600 s),
#     μεγάλη 11.500 kg/s × 0,77% = 88,55 kg/s H2S (3600 s). Έτσι η σύγκριση sour gas (ελαφρύ) έναντι acid gas (πυκνό) γίνεται για την ίδια τοξική εκπομπή.
#   - πηγή: οριζόντιος πίδακας (idspl = 2) στο 1 m· «ψευδο-πηγή» στην ατμοσφαιρική πίεση με τις συνθήκες εξόδου στο όριο choked flow
#     (T* = 2·T0/(γ+1), u* = √(2γ/(γ+1)·R·T0/M)), δηλαδή αγνοείται η επιπλέον επέκταση του πίδακα ως την ατμοσφαιρική πίεση. Επηρεάζει μόνο το εγγύς πεδίο (δεκάδες m).
#   - μετεωρολογία: z0 = 0,1 m, za = 10 m, T = 300,15 K (27 °C), RH = 50%· σταθερότητες C/D/F, άνεμος 3 και 5 m/s (ο συνδυασμός F με άνεμο > 3,5 m/s παραλείπεται)
#   - χρόνος μέσου όρου tav = 10 s· xffm = 5000 m ώστε να καλύπτεται ο χάρτης για κάθε διεύθυνση ανέμου από πηγή οπουδήποτε μέσα του
# Το SLAB δίνει κλάσμα όγκου του ΣΥΝΟΛΟΥ του εκλυόμενου μίγματος· για συγκέντρωση H2S πολλαπλασιάζεται με x_H2S (βλ. --scale στο slab_to_concentration_maps.jl).
#
# Χρήση από τη ρίζα του repo (μόνο standard library):
#   julia scripts/slab_acid_gas_cases.jl <φάκελος_εξόδου> [επιλογές]
# Επιλογές (όλες προαιρετικές, εντός παρένθεσης η προεπιλογή):
#   --xh2s=      μοριακό κλάσμα H2S στο μίγμα H2S + CO2 (0.40)
#   --releases=  διαρροές ως όνομα:ρυθμός_H2S_kg/s:διάρκεια_s, χωρισμένες με κόμμα (small:0.77:600,large:88.55:3600)
#   --stabs=     κατηγορίες σταθερότητας A–F (C,D,F)          --winds=  ταχύτητες ανέμου στο ύψος za, m/s (3,5)
#   --z0=        τραχύτητα εδάφους, m (0.1)                   --za=     ύψος μέτρησης ανέμου, m (10)
#   --ta=        θερμοκρασία περιβάλλοντος, K (300.15)         --rh=     σχετική υγρασία, % (50)
#   --tav=       χρόνος μέσου όρου, s (10)                    --xffm=   μέγιστη απόσταση, m (5000)
#   --hs=        ύψος κέντρου του πίδακα, m (1)
#   --slab=      διαδρομή του Slab.exe· αν δοθεί, το SLAB εκτελείται για κάθε περίπτωση και αποθηκεύεται το <case>.predict
# Χωρίς --slab γράφονται μόνο τα αρχεία <case>.in και το cases.tsv (που καταγράφει και τις τιμές όλων των παραπάνω παραμέτρων).
# Δεν αλλάζουν από τη γραμμή εντολών (θέλουν αλλαγή κώδικα): τα cp των συστατικών, ο τύπος πηγής (οριζόντιος πίδακας) και ο τρόπος υπολογισμού του όρου πηγής.

module AcidGasCases

using Printf

const SCRIPTS = @__DIR__
include(joinpath(SCRIPTS, "slab_to_concentration_maps.jl"))                              # module SlabFields (slab_input_text)
using .SlabFields

export mixture, choked_pseudo_source, release_cases, case_input_text, run_slab, main

const R_GAS = 8.31431                                                                    # σταθερά αερίων όπως στο SLAB (J/mol/K)
const P_ATM = 101325.0                                                                   # Pa
const M_H2S = 0.03408                                                                    # kg/mol
const M_CO2 = 0.04401                                                                    # kg/mol
const CP_H2S = 34.2                                                                      # J/mol/K γύρω στους 300 K (τυπική τιμή βιβλιογραφίας)
const CP_CO2 = 37.2                                                                      # J/mol/K γύρω στους 300 K (τυπική τιμή βιβλιογραφίας)
const STAB_CODE = Dict("A" => 1.0, "B" => 2.0, "C" => 3.0, "D" => 4.0, "E" => 5.0, "F" => 6.0)   # κωδικοί σταθερότητας του SLAB

# Ιδιότητες του μίγματος H2S + CO2 με μοριακό κλάσμα H2S x
function mixture(x::Real)
    0 < x <= 1 || error("Το x_H2S πρέπει να είναι στο (0, 1]")
    M = x * M_H2S + (1 - x) * M_CO2                                                      # μοριακό βάρος (kg/mol)
    cpm = x * CP_H2S + (1 - x) * CP_CO2                                                  # cp σε J/mol/K
    return (M = M, cp_molar = cpm, cp_mass = cpm / M, gamma = cpm / (cpm - R_GAS), w_h2s = x * M_H2S / M)
end

# Ψευδο-πηγή για αέριο που εκρέει choked από δεξαμενή T0: συνθήκες εξόδου σε ατμοσφαιρική πίεση και επιφάνεια από διατήρηση μάζας qs = ρ·u·A
function choked_pseudo_source(qs::Real, mix, T0::Real)
    γ = mix.gamma
    Ts = 2T0 / (γ + 1)                                                                   # θερμοκρασία εξόδου
    us = sqrt(2γ / (γ + 1) * R_GAS * T0 / mix.M)                                         # ταχύτητα εξόδου = ταχύτητα ήχου στις συνθήκες εξόδου
    ρs = P_ATM * mix.M / (R_GAS * Ts)                                                    # πυκνότητα στην ατμοσφαιρική πίεση (όπως τη υπολογίζει το SLAB: rhos = wms·pa/(R·ts))
    return (Ts = Ts, us = us, rhos = ρs, area = qs / (ρs * us))
end

# Οι δύο εντάσεις-διάρκειες της ομάδας (kg/s H2S, s): 100 και 11.500 kg/s αερίου με 0,77% w/w H2S
const RELEASES = ["small" => (100.0 * 0.0077, 600.0), "large" => (11500.0 * 0.0077, 3600.0)]

# Όλες οι περιπτώσεις: (όνομα, παράμετροι SLAB, περιγραφή)
function release_cases(; xh2s = 0.40, stabs = ["C", "D", "F"], winds = [3.0, 5.0], T0 = 300.15, rh = 50.0, z0 = 0.1, za = 10.0,
                       tav = 10.0, xffm = 5000.0, hs = 1.0, releases = RELEASES)
    mix = mixture(xh2s)
    cases = NamedTuple[]
    for (rname, (qh2s, tsd)) in releases, st in stabs, u in winds
        st == "F" && u > 3.5 && continue                                                 # φυσικά ασυνεπής συνδυασμός (F απαιτεί ασθενή άνεμο)
        qs = qh2s / mix.w_h2s                                                            # ολική παροχή μάζας acid gas ώστε η παροχή H2S να είναι qh2s
        src = choked_pseudo_source(qs, mix, T0)
        name = @sprintf("ag_%s_%s_%gms", rname, st, u)
        push!(cases, (name = name, release = rname, stab = st, wind = u, qh2s = qh2s, qs = qs, tsd = tsd, xh2s = xh2s, mix = mix, src = src,
                      T0 = T0, rh = rh, z0 = z0, za = za, tav = tav, xffm = xffm, hs = hs))
    end
    return cases
end

# Κείμενο εισόδου SLAB για μία περίπτωση (τα υγρά μεγέθη cpsl/rhosl/dhe/tbp είναι του H2S και δεν χρησιμοποιούνται: cmed0 = 0 και ts > tbp)
case_input_text(c) = slab_input_text(; idspl = 2, wms = c.mix.M, cps = c.mix.cp_mass, tbp = 212.8, cmed0 = 0.0, dhe = 548000.0, cpsl = 2200.0,
                                    rhosl = 950.0, spb = -1.0, spc = 0.0, ts = c.src.Ts, qs = c.qs, as = c.src.area, tsd = c.tsd, qtis = 0.0,
                                    hs = c.hs, tav = c.tav, xffm = c.xffm, zp = (2.0, 0.0, 0.0, 0.0), z0 = c.z0, za = c.za, ua = c.wind,
                                    ta = c.T0, rh = c.rh, stab = STAB_CODE[c.stab])

# Τρέχει το SLAB για μία περίπτωση σε προσωρινό φάκελο (το SLAB διαβάζει πάντα «input» και γράφει «predict»)· επιστρέφει το κείμενο του predict
function run_slab(exe::AbstractString, text::AbstractString)
    dir = mktempdir()
    try
        write(joinpath(dir, "input"), text)
        run(pipeline(Cmd(`$exe`; dir = dir); stdout = devnull, stderr = devnull))
        pred = joinpath(dir, "predict")
        isfile(pred) || error("Το SLAB δεν δημιούργησε αρχείο predict")
        return read(pred, String)
    finally
        rm(dir; recursive = true, force = true)
    end
end

function main(args)
    isempty(args) && error("Χρήση: julia scripts/slab_acid_gas_cases.jl <φάκελος_εξόδου> [--xh2s= --releases= --stabs= --winds= --z0= --za= --ta= --rh= --tav= --xffm= --hs= --slab=] (βλ. την κεφαλίδα του αρχείου)")
    outdir = args[1]
    opts = Dict{String,String}()
    known = ("xh2s", "releases", "stabs", "winds", "z0", "za", "ta", "rh", "tav", "xffm", "hs", "slab")
    for a in args[2:end]
        m = match(r"^--([a-z0-9]+)=(.*)$", a)
        m === nothing && error("Άγνωστο όρισμα: $a")
        m.captures[1] in known || error("Άγνωστη επιλογή --$(m.captures[1]) (έγκυρες: $(join(known, ", ")))")
        opts[m.captures[1]] = m.captures[2]
    end
    num(k, default) = haskey(opts, k) ? parse(Float64, opts[k]) : default                # αριθμητική επιλογή ή προεπιλογή
    xh2s = num("xh2s", 0.40)
    stabs = haskey(opts, "stabs") ? String.(split(opts["stabs"], ",")) : ["C", "D", "F"]
    winds = haskey(opts, "winds") ? parse.(Float64, split(opts["winds"], ",")) : [3.0, 5.0]
    all(s -> haskey(STAB_CODE, s), stabs) || error("Άγνωστη κατηγορία σταθερότητας στο --stabs=$(opts["stabs"])")
    releases = RELEASES
    if haskey(opts, "releases")                                                           # μορφή όνομα:ρυθμός_H2S:διάρκεια, χωρισμένα με κόμμα
        releases = map(split(opts["releases"], ",")) do item
            f = split(item, ":")
            length(f) == 3 || error("Η διαρροή «$item» πρέπει να έχει τη μορφή όνομα:ρυθμός_H2S_kg/s:διάρκεια_s")
            String(f[1]) => (parse(Float64, f[2]), parse(Float64, f[3]))
        end
    end
    mkpath(outdir)
    cases = release_cases(; xh2s, stabs, winds, releases, z0 = num("z0", 0.1), za = num("za", 10.0), T0 = num("ta", 300.15), rh = num("rh", 50.0),
                          tav = num("tav", 10.0), xffm = num("xffm", 5000.0), hs = num("hs", 1.0))
    open(joinpath(outdir, "cases.tsv"), "w") do tsv
        println(tsv, "case\trelease\tstab\twind_ms\tqh2s_kgs\tqs_kgs\tduration_s\txh2s\tM_kgmol\tTs_K\tus_ms\trhos_kgm3\tarea_m2\tz0_m\tza_m\tta_K\trh_pct\ttav_s\txffm_m\ths_m")
        for c in cases
            write(joinpath(outdir, c.name * ".in"), case_input_text(c))
            @printf(tsv, "%s\t%s\t%s\t%g\t%.4g\t%.5g\t%g\t%g\t%.5f\t%.2f\t%.2f\t%.4f\t%.5f\t%g\t%g\t%g\t%g\t%g\t%g\t%g\n", c.name, c.release, c.stab, c.wind, c.qh2s, c.qs,
                    c.tsd, c.xh2s, c.mix.M, c.src.Ts, c.src.us, c.src.rhos, c.src.area, c.z0, c.za, c.T0, c.rh, c.tav, c.xffm, c.hs)
            if haskey(opts, "slab")
                write(joinpath(outdir, c.name * ".predict"), run_slab(opts["slab"], case_input_text(c)))
            end
            println("  ", c.name, haskey(opts, "slab") ? "  (SLAB εκτελέστηκε)" : "")
        end
    end
    println("Γράφτηκαν ", length(cases), " περιπτώσεις στο ", outdir)
end

end # module AcidGasCases

if abspath(PROGRAM_FILE) == @__FILE__
    AcidGasCases.main(ARGS)
end
