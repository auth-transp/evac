# Μετατροπή της εξόδου του SLAB (Ermak, EPA SCRAM) σε χάρτες συγκέντρωσης (CMs) στο πλέγμα του χάρτη της εγκατάστασης.
#
# Το SLAB εκτυπώνει στο αρχείο PREDICT την κλειστή μορφή του πεδίου συγκέντρωσης (κλάσμα όγκου):
#     c(x,y,z,t) = cc(x) · (erf(xa) − erf(xb)) · (erf(ya) − erf(yb)) · (exp(−za²) + exp(−zb²))
#     xa = (x − xc + bx)/(√2·βx)   xb = (x − xc − bx)/(√2·βx)    ← παράμετροι σε συνάρτηση του χρόνου t: xc(t), bx(t), βx(t)
#     ya = (y + b)/(√2·βc)         yb = (y − b)/(√2·βc)          ← παράμετροι σε συνάρτηση του x: cc(x), b(x), βc(x), zc(x), sig(x)
#     za = (z − zc)/(√2·sig)       zb = (z + zc)/(√2·sig)
# Άρα το πεδίο μπορεί να ανασυντεθεί ακριβώς σε οποιοδήποτε πλέγμα, χρόνο, ύψος και διεύθυνση ανέμου.
#
# Χρήση από τη ρίζα του repo (μόνο standard library, δεν χρειάζεται project):
#   julia scripts/slab_to_concentration_maps.jl <predict> <φάκελος_εξόδου> --source=i,j --wind=θ [--times=t1,t2,...] [--z=2.0] [--dims=915,983] [--mpp=3.456] [--scale=1.0]
# Ορίσματα: --source=i,j (pixel πηγής, υποχρεωτικό) · --wind=θ (μοίρες, υποχρεωτικό) · --times=… (s) · --z=… (m) · --dims=… · --mpp=… (m ανά pixel)
#           --scale=… (πολλαπλασιαστής του κλάσματος όγκου του SLAB, π.χ. 0.4 = x_H2S για ppm H2S σε μίγμα H2S + CO2)
#
# Συμβάσεις συντεταγμένων:
#   - Τα (i, j) είναι οι δείκτες του πίνακα [i, j] (όπως στα σενάρια: pos[1] → i, pos[2] → j).
#   - Η γωνία θ ορίζει προς τα πού κινείται το νέφος στο επίπεδο των δεικτών: θ = 0 → προς αύξηση του i, θ = 90 → προς αύξηση του j.
#     Η αντιστοίχιση με πραγματικό προσανατολισμό (βορράς/ανατολή) εξαρτάται από τον προσανατολισμό της εικόνας του χάρτη.
#   - Το πεδίο είναι μηδέν ανάντη της πηγής και πέρα από το τελευταίο x του πίνακα του SLAB (λίγο μεγαλύτερο από το xffm της εισόδου, π.χ. 3120 m για xffm = 3000 m).
#   - Το πεδίο είναι ο κλειστός τύπος του SLAB (μέσος όρος tav): δεν περιλαμβάνει διακυμάνσεις συγκέντρωσης (K_L) ούτε εμπόδια/κτίρια, και η ουρά του κόβεται μόνο από το erf (~0,3% της τιμής μέσα στο νέφος αφού περάσει).
#
# Δημιουργία της εισόδου και εκτέλεση του SLAB (δεν περιλαμβάνεται στο repo· κατέβασμα από EPA SCRAM, Windows):
#   1) τεκμηρίωση/λήψη: https://www.epa.gov/scram/air-quality-dispersion-modeling-alternative-models (slab.zip)
#   2) γράψε το αρχείο εισόδου με την `slab_input_text`, αντίγραψέ το ως `input` δίπλα στο Slab.exe, τρέξε Slab.exe
#   3) το αποτέλεσμα είναι το `predict` (το SLAB δεν αντικαθιστά υπάρχον predict· μετονόμασέ το πριν την επόμενη εκτέλεση)

module SlabFields

using Printf

export SlabCloud, parse_slab_output, slab_input_text, concentration_ppm, concentration_map,
       erf_scalar, erfc_scalar, erf_diff, save_concentration_map, load_concentration_map,
       DEFAULT_DIMS, DEFAULT_M_PER_PX, DEFAULT_TIMES

# ---------------------------------------------------------------------------------------------
# Σταθερές του χάρτη της εγκατάστασης (διπλωματική §2.7.2: 2.500 m ↔ 723,37 pixels, πλέγμα 915×983)
# ---------------------------------------------------------------------------------------------
const DEFAULT_DIMS = (915, 983)                         # διαστάσεις του πίνακα (i, j) όπως στα σενάρια
const DEFAULT_M_PER_PX = 2500.0 / 723.37                # μέτρα ανά pixel (≈ 3,456 m)
# Μέσα των διαστημάτων εναλλαγής CM των σεναρίων (0, 292, …, 1752 s) και του T = 2500 s
const DEFAULT_TIMES = [146.0, 438.0, 730.0, 1022.0, 1314.0, 1606.0, 2126.0]

# ---------------------------------------------------------------------------------------------
# Συναρτήσεις σφάλματος (μόνο standard library): erf, erfc και σταθερή διαφορά erf(a) − erf(b)
# ---------------------------------------------------------------------------------------------

# erf(x) για x ≥ 0 με σειρά θετικών όρων: erf(x) = 2/√π · e^{−x²} · Σ 2ⁿ x^{2n+1}/(2n+1)!! (χωρίς απαλοιφή ψηφίων)
function erf_series(x::Float64)
    term = x                                            # πρώτος όρος: x
    s = x                                               # τρέχον άθροισμα
    n = 0
    while term > 1e-18 * s && n < 400
        term *= 2x * x / (2n + 3)                       # επόμενος όρος: πολλαπλασιασμός με 2x²/(2n+3)
        s += term
        n += 1
    end
    return 2 / sqrt(pi) * exp(-x * x) * s
end

# erfc(x) για x ≥ 2,5 με συνεχές κλάσμα: erfc(x) = e^{−x²}/√π · 1/(x + (1/2)/(x + 1/(x + (3/2)/(x + …))))
function erfc_cf(x::Float64)
    f = x                                               # κόψιμο της ουράς του κλάσματος
    for k in 80:-1:1                                    # υπολογισμός από το βάθος προς την κορυφή
        f = x + (k / 2) / f
    end
    return exp(-x * x) / (sqrt(pi) * f)
end

# erfc(x) για οποιοδήποτε πραγματικό x
function erfc_scalar(x::Real)
    xf = Float64(x)
    xf < 0 && return 2.0 - erfc_scalar(-xf)             # erfc(−x) = 2 − erfc(x)
    return xf < 2.5 ? 1.0 - erf_series(xf) : erfc_cf(xf)
end

# erf(x) για οποιοδήποτε πραγματικό x
function erf_scalar(x::Real)
    xf = Float64(x)
    xf < 0 && return -erf_scalar(-xf)
    return xf < 2.5 ? erf_series(xf) : 1.0 - erfc_cf(xf)
end

# erf(a) − erf(b) για a ≥ b, χωρίς απώλεια ακρίβειας στις ουρές (όπου και οι δύο τιμές του erf είναι ≈ ±1)
function erf_diff(a::Float64, b::Float64)
    if b >= 2.5
        return erfc_scalar(b) - erfc_scalar(a)          # και τα δύο στη θετική ουρά
    elseif a <= -2.5
        return erfc_scalar(-a) - erfc_scalar(-b)        # και τα δύο στην αρνητική ουρά
    else
        return erf_scalar(a) - erf_scalar(b)
    end
end

# ---------------------------------------------------------------------------------------------
# Ανάγνωση της εξόδου του SLAB
# ---------------------------------------------------------------------------------------------

# Οι παράμετροι του πεδίου από το PREDICT: συναρτήσεις του x (cc, b, βc, zc, sig) και του t (xc, bx, βx)
struct SlabCloud
    x::Vector{Float64}                                  # απόσταση κατά τον άνεμο (m)
    cc::Vector{Float64}                                 # cc(x): πλάτος κορυφής συγκέντρωσης (κλάσμα όγκου)
    b::Vector{Float64}                                  # b(x): ημιπλάτος top-hat στον εγκάρσιο άξονα (m)
    betac::Vector{Float64}                              # βc(x): εύρος εξομάλυνσης στον εγκάρσιο άξονα (m)
    zc::Vector{Float64}                                 # zc(x): ύψος του κέντρου του νέφους (m)
    sig::Vector{Float64}                                # sig(x): κατακόρυφο εύρος (m)
    t::Vector{Float64}                                  # t: χρόνος (s)
    xc::Vector{Float64}                                 # xc(t): θέση του κέντρου του νέφους (m)
    bx::Vector{Float64}                                 # bx(t): ημιμήκος του νέφους (m)
    betax::Vector{Float64}                              # βx(t): εύρος εξομάλυνσης των άκρων του νέφους (m)
    tav::Float64                                        # χρόνος μέσου όρου της συγκέντρωσης (s)
    tsd::Float64                                        # διάρκεια της συνεχούς πηγής (s)
    xffm::Float64                                       # μέγιστη απόσταση του υπολογισμού (m)
end

# Μετατρέπει αριθμό του Fortran σε Float64 (δέχεται και τη μορφή 1.23-105 χωρίς το «E» για εκθέτες τριών ψηφίων)
function parse_fnum(s::AbstractString)
    t = strip(s)
    m = match(r"^([-+]?\d*\.?\d+)([-+]\d{3})$", t)
    m !== nothing && return parse(Float64, m.captures[1] * "E" * m.captures[2])
    return parse(Float64, t)
end

const NUM_RE = r"[-+]?\d*\.\d+(?:E[-+]?\d+|[-+]\d{3})?|[-+]?\d+(?:E[-+]?\d+)?"

# Επιστρέφει όλους τους αριθμούς μιας γραμμής
numbers_in(line::AbstractString) = [parse_fnum(m.match) for m in eachmatch(NUM_RE, line)]

# Διαβάζει από το «problem input» του PREDICT την τιμή μιας μεταβλητής, π.χ. "  tsd   =       600."
function input_value(lines::Vector{String}, name::AbstractString)
    for l in lines
        m = match(Regex("^\\s*" * name * "\\s*=\\s*(\\S+)"), l)
        m !== nothing && return parse_fnum(m.captures[1])
    end
    return NaN
end

# Διαβάζει τον πίνακα παραμέτρων "concentration contour parameters" του PREDICT
function parse_slab_output(path::AbstractString)
    lines = readlines(path)
    i0 = findfirst(l -> occursin("concentration contour parameters", l), lines)
    i0 === nothing && error("Δεν βρέθηκε ο πίνακας 'concentration contour parameters' στο $path")
    ih = findnext(l -> occursin("cc(x)", l) && occursin("betax(t)", l), lines, i0)     # γραμμή επικεφαλίδων
    ih === nothing && error("Δεν βρέθηκε η επικεφαλίδα των παραμέτρων cc(x) … betax(t)")
    rows = Vector{Float64}[]
    for l in lines[ih+1:end]
        v = numbers_in(l)
        length(v) == 10 || (isempty(rows) ? continue : break)                            # ο πίνακας τελειώνει στην πρώτη γραμμή χωρίς 10 αριθμούς
        push!(rows, v)
    end
    isempty(rows) && error("Ο πίνακας παραμέτρων είναι κενός")
    col(k) = [r[k] for r in rows]
    x, t = col(1), col(7)
    issorted(x) || error("Τα x του πίνακα δεν είναι αύξοντα")
    issorted(t) || error("Οι χρόνοι t του πίνακα δεν είναι αύξοντες")
    return SlabCloud(x, col(2), col(3), col(4), col(5), col(6), t, col(8), col(9), col(10),
                     input_value(lines, "tav"), input_value(lines, "tsd"), input_value(lines, "xffm"))
end

# ---------------------------------------------------------------------------------------------
# Ανασύνθεση του πεδίου
# ---------------------------------------------------------------------------------------------

# Γραμμική παρεμβολή y(x) πάνω σε αύξοντα x· εκτός ορίων επιστρέφει nothing
function interp(xs::Vector{Float64}, ys::Vector{Float64}, x::Float64)
    (x < xs[1] || x > xs[end]) && return nothing
    k = clamp(searchsortedlast(xs, x), 1, length(xs) - 1)
    x1, x2 = xs[k], xs[k+1]
    w = x2 > x1 ? (x - x1) / (x2 - x1) : 0.0
    return ys[k] + w * (ys[k+1] - ys[k])
end

# Κατάσταση του νέφους (xc, bx, βx) τη στιγμή t· πέρα από τον τελευταίο χρόνο του πίνακα γίνεται γραμμική προέκταση με τον τελευταίο ρυθμό
function cloud_state(c::SlabCloud, t::Float64)
    t < c.t[1] && return nothing                        # πριν την έναρξη της διαρροής
    if t <= c.t[end]
        return (interp(c.t, c.xc, t), interp(c.t, c.bx, t), interp(c.t, c.betax, t))
    end
    n = length(c.t)
    dt = c.t[n] - c.t[n-1]
    f = dt > 0 ? (t - c.t[n]) / dt : 0.0                # πόσα «τελευταία βήματα» μακριά βρισκόμαστε
    return (c.xc[n] + f * (c.xc[n] - c.xc[n-1]),
            max(0.0, c.bx[n] + f * (c.bx[n] - c.bx[n-1])),
            max(0.0, c.betax[n] + f * (c.betax[n] - c.betax[n-1])))
end

# erf(a) − erf(b) με τα όρια ± ημιπλάτος και εξομάλυνση β· για β → 0 δίνει τη βηματική συνάρτηση (2 μέσα στο διάστημα, 1 στα άκρα)
function smoothed_box(u::Float64, half::Float64, beta::Float64)
    if beta < 1e-12
        a = abs(u)
        return a < half ? 2.0 : (a == half ? 1.0 : 0.0)
    end
    s = sqrt(2.0) * beta
    return erf_diff((u + half) / s, (u - half) / s)
end

# Κλάσμα όγκου c στο σημείο (x, y, z) τη στιγμή που το νέφος έχει κατάσταση st = (xc, bx, βx)
function concentration_at(c::SlabCloud, x::Float64, y::Float64, z::Float64, st)
    st === nothing && return 0.0
    xc, bx, betax = st
    cc = interp(c.x, c.cc, x)
    cc === nothing && return 0.0                        # ανάντη της πηγής ή πέρα από το xffm
    cc == 0.0 && return 0.0
    b = interp(c.x, c.b, x)
    betac = interp(c.x, c.betac, x)
    zc = interp(c.x, c.zc, x)
    sig = interp(c.x, c.sig, x)
    fx = smoothed_box(x - xc, bx, betax)                # (erf(xa) − erf(xb))
    fx == 0.0 && return 0.0
    fy = smoothed_box(y, b, betac)                      # (erf(ya) − erf(yb))
    fy == 0.0 && return 0.0
    s = sqrt(2.0) * sig
    fz = exp(-((z - zc) / s)^2) + exp(-((z + zc) / s)^2) # (exp(−za²) + exp(−zb²))
    return cc * fx * fy * fz
end

# Συγκέντρωση σε ppm (κατά όγκο) στο σημείο (x, y, z) και χρόνο t· x κατά τον άνεμο, y εγκάρσια, z ύψος (m)
concentration_ppm(c::SlabCloud, x::Real, y::Real, z::Real, t::Real) =
    1e6 * concentration_at(c, Float64(x), Float64(y), Float64(z), cloud_state(c, Float64(t)))

# Χάρτης συγκέντρωσης (ppm) πάνω στο πλέγμα [i, j] για χρόνο t και ύψος z
#   src = (i0, j0): pixel της πηγής · wind_deg: διεύθυνση κίνησης του νέφους στο επίπεδο των δεικτών (0° → +i, 90° → +j)
#   scale: το SLAB δίνει κλάσμα όγκου του ΣΥΝΟΛΟΥ του εκλυόμενου μίγματος· για μίγμα H2S + CO2 με x_H2S = 0,4 δώσε scale = 0,4 ώστε να βγει ppm H2S
function concentration_map(c::SlabCloud; dims = DEFAULT_DIMS, src, wind_deg::Real, t::Real, z::Real = 2.0,
                           m_per_px::Real = DEFAULT_M_PER_PX, scale::Real = 1.0)
    st = cloud_state(c, Float64(t))
    out = zeros(Float32, dims...)
    st === nothing && return out
    ew = (cosd(Float64(wind_deg)), sind(Float64(wind_deg)))     # μοναδιαίο διάνυσμα κατά τον άνεμο
    ep = (-ew[2], ew[1])                                        # εγκάρσιο μοναδιαίο διάνυσμα
    for j in 1:dims[2], i in 1:dims[1]
        dx = (i - src[1]) * m_per_px                            # μετατόπιση από την πηγή σε m, κατά i
        dy = (j - src[2]) * m_per_px                            # μετατόπιση από την πηγή σε m, κατά j
        x = dx * ew[1] + dy * ew[2]                             # συντεταγμένη κατά τον άνεμο
        x < c.x[1] && continue                                  # ανάντη της πηγής
        y = dx * ep[1] + dy * ep[2]                             # εγκάρσια συντεταγμένη
        out[i, j] = Float32(scale * 1e6 * concentration_at(c, x, y, Float64(z), st))     # scale = κλάσμα του μίγματος που είναι η τοξική ουσία (π.χ. x_H2S)
    end
    return out
end

# ---------------------------------------------------------------------------------------------
# Αποθήκευση/ανάγνωση χαρτών ως Float32 (τα παλιά CMs είναι 8-bit εικόνες· το Float32 αποφεύγει τον κορεσμό και την κβάντιση)
# ---------------------------------------------------------------------------------------------

# Γράφει τον πίνακα ως ακατέργαστο Float32 (little-endian, column-major)· οι διαστάσεις καταγράφονται στο metadata
save_concentration_map(path::AbstractString, A::Matrix{Float32}) = open(io -> write(io, A), path, "w")

# Διαβάζει πίνακα Float32 γνωστών διαστάσεων
function load_concentration_map(path::AbstractString, dims = DEFAULT_DIMS)
    A = Matrix{Float32}(undef, dims...)
    open(io -> read!(io, A), path, "r")
    return A
end

# ---------------------------------------------------------------------------------------------
# Δημιουργία του αρχείου εισόδου του SLAB (σειρά και μορφή από το SLAB.FOR: μία τιμή ανά γραμμή, f10.3, τέλος με -1)
# ---------------------------------------------------------------------------------------------

# Μορφοποιεί αριθμό σε το πολύ 10 χαρακτήρες (το SLAB διαβάζει πεδία f10.3)
function fmt10(x::Real)
    for fmt in ("%.6f", "%.4f", "%.3f", "%.1f")
        s = Printf.format(Printf.Format(fmt), Float64(x))
        length(s) <= 10 && return s
    end
    error("Η τιμή $x δεν χωρά σε πεδίο 10 χαρακτήρων")
end

# Κείμενο εισόδου για το SLAB. Τύποι πηγής idspl: 1 εξατμιζόμενη λίμνη, 2 οριζόντιος πίδακας, 3 κατακόρυφος πίδακας/στοίβα, 4 στιγμιαία πηγή.
# Οι ιδιότητες του υγρού (cpsl, rhosl, dhe, spb, spc) μετράνε μόνο όταν υπάρχει υγρό (cmed0 > 0 ή λίμνη)· για αέριο στη θερμοκρασία ts > tbp δεν χρησιμοποιούνται.
function slab_input_text(; idspl::Int, wms, cps, tbp, cmed0 = 0.0, dhe, cpsl, rhosl, spb = -1.0, spc = 0.0,
                         ts, qs, as, tsd, qtis = 0.0, hs = 0.0, tav, xffm, zp = (2.0, 0.0, 0.0, 0.0),
                         z0, za, ua, ta, rh, stab, ala = nothing, ncalc::Int = 1)
    io = IOBuffer()
    println(io, lpad(string(idspl), 5))
    println(io, lpad(string(ncalc), 5))
    vals = Any[wms, cps, tbp, cmed0, dhe, cpsl, rhosl, spb, spc, ts, qs, as, tsd, qtis, hs, tav, xffm,
               zp[1], zp[2], zp[3], zp[4], z0, za, ua, ta, rh, stab]
    for v in vals
        println(io, " ", fmt10(v))
    end
    if stab == 0
        ala === nothing && error("Με stab = 0 πρέπει να δοθεί το ala (αντίστροφο μήκος Monin–Obukhov)")
        println(io, " ", fmt10(ala))
    end
    println(io, "-1.")                                      # τέλος εισόδου: αρνητικό z0
    return String(take!(io))
end

# ---------------------------------------------------------------------------------------------
# Γραμμή εντολών
# ---------------------------------------------------------------------------------------------

function parse_list(s::AbstractString, T = Float64)
    return [parse(T, strip(p)) for p in split(s, ",") if !isempty(strip(p))]
end

function main(args)
    length(args) >= 2 || error("Χρήση: julia scripts/slab_to_concentration_maps.jl <predict> <φάκελος_εξόδου> --source=i,j --wind=θ [--times=…] [--z=…] [--dims=…] [--mpp=…]")
    predict, outdir = args[1], args[2]
    opts = Dict{String,String}()
    for a in args[3:end]
        m = match(r"^--([a-z]+)=(.*)$", a)
        m === nothing && error("Άγνωστο όρισμα: $a")
        opts[m.captures[1]] = m.captures[2]
    end
    for k in keys(opts)
        k in ("source", "wind", "times", "z", "dims", "mpp", "scale") || error("Άγνωστη επιλογή --$k")
    end
    haskey(opts, "source") || error("Λείπει το --source=i,j")
    haskey(opts, "wind") || error("Λείπει το --wind=θ")
    src = Tuple(parse_list(opts["source"]))
    wind = parse(Float64, opts["wind"])
    times = haskey(opts, "times") ? parse_list(opts["times"]) : DEFAULT_TIMES
    z = haskey(opts, "z") ? parse(Float64, opts["z"]) : 2.0
    dims = haskey(opts, "dims") ? Tuple(parse_list(opts["dims"], Int)) : DEFAULT_DIMS
    mpp = haskey(opts, "mpp") ? parse(Float64, opts["mpp"]) : DEFAULT_M_PER_PX
    scale = haskey(opts, "scale") ? parse(Float64, opts["scale"]) : 1.0

    cloud = parse_slab_output(predict)
    mkpath(outdir)
    meta = open(joinpath(outdir, "maps_meta.txt"), "w")
    println(meta, "# Χάρτες συγκέντρωσης (ppm κατά όγκο) από SLAB · αρχεία cm_K.f32 = Float32 little-endian, column-major, μέγεθος dims")
    println(meta, "predict = ", predict)
    println(meta, "dims = ", join(dims, ","))
    println(meta, "source_pixel = ", join(src, ","))
    println(meta, "wind_deg = ", wind)
    println(meta, "z_m = ", z)
    println(meta, "m_per_px = ", mpp)
    println(meta, "scale = ", scale, "  (πολλαπλασιαστής του κλάσματος όγκου του SLAB)")
    println(meta, "tav_s = ", cloud.tav, " · tsd_s = ", cloud.tsd, " · xffm_m = ", cloud.xffm)
    println(meta, "t_last_table_s = ", cloud.t[end], "  (πέρα από αυτόν τον χρόνο γίνεται γραμμική προέκταση του νέφους)")
    println(meta, "# k  t(s)  max(ppm)  pixels>10ppm  pixels>100ppm  pixels>1000ppm")
    for (k, t) in enumerate(times)
        A = concentration_map(cloud; dims, src, wind_deg = wind, t, z, m_per_px = mpp, scale)
        save_concentration_map(joinpath(outdir, "cm_$(k).f32"), A)
        line = @sprintf("%d  %.1f  %.1f  %d  %d  %d", k, t, maximum(A), count(>(10), A), count(>(100), A), count(>(1000), A))
        println(meta, line)
        println(line)
        t > cloud.t[end] && println("  προειδοποίηση: t = $t s > t_last = $(cloud.t[end]) s (προέκταση)")
    end
    close(meta)
end

end # module SlabFields

if abspath(PROGRAM_FILE) == @__FILE__
    SlabFields.main(ARGS)
end
