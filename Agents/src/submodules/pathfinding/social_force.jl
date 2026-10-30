# ΣΧΕΣΗ ΜΕ ΤΟ PAPER (Helbing & Molnár, "Social force model for pedestrian dynamics", Phys. Rev. E 51, 4282 (1995),
# arXiv:cond-mat/9805244). Το αρχείο αυτό περιέχει το ίδιο το μοντέλο (τις εξισώσεις του paper) ανεξάρτητα από το ABM:
#   eq (1)   e_α = (r_α^k − r_α)/‖r_α^k − r_α‖          επιθυμητή κατεύθυνση      → `desired_direction` (social_force_continuous.jl)
#   eq (2)   F⁰_α = (v⁰_α e_α − v_α)/τ_α                όρος επιτάχυνσης          → `driving_force`
#   eq (3)   f_αβ = −∇ V_αβ[b(r_αβ)]                    άπωση από πεζό β          → `pedestrian_repulsion`
#   eq (4)   2b = sqrt((‖r_αβ‖+‖r_αβ − v_β Δt e_β‖)² − (v_β Δt)²)  ημιάξονας έλλειψης → `pedestrian_repulsion`
#   eq (5)   F_αB = −∇ U_αB(‖r_αB‖)                     άπωση από τοίχο/όριο      → `wall_repulsion`
#   eq (6)   f_αi = −∇ W_αi(‖r_αi‖, t)                  έλξη (προαιρετική)        → callback `attraction`
#   eq (7)   w(e, f) = 1 αν e·f ≥ ‖f‖ cos φ, αλλιώς c   βάρος οπτικού πεδίου       → `direction_weight`
#   eq (8)   F_αβ = w(e_α, −f_αβ) f_αβ,  F_αi = w(e_α, f_αi) f_αi                  → `social_force`
#   eq (9)   F_α = F⁰_α + Σ_β F_αβ + Σ_B F_αB + Σ_i F_αi                           → `social_force`
#   eq (10)  dw_α/dt = F_α + fluctuations                                          → `social_force_step!`
#   eq (11)  dr_α/dt = v_α = w_α g(v_max/‖w_α‖), eq (12) g = min(1, v_max/‖w‖)      → `cap_velocity`
#   eq (13)  V_αβ(b) = V⁰ e^{−b/σ},  U_αB(‖r‖) = U⁰ e^{−‖r‖/R}                     → `pedestrian_repulsion`, `wall_repulsion`
# Οι προεπιλεγμένες παράμετροι είναι αυτές του Section IV του paper (V⁰=2.1 m²/s², σ=0.3 m, U⁰=10 m²/s², R=0.2 m, τ=0.5 s,
# Δt=2 s, 2φ=200°, c=0.5, v⁰~N(1.34, 0.26) m/s, v_max=1.3 v⁰). Είναι σε SI· αν το μοντέλο σου δουλεύει σε άλλες μονάδες
# μήκους/χρόνου (π.χ. pixels), οι παράμετροι πρέπει να μετατραπούν αντίστοιχα.

const Vec2 = SVector{2,Float64}                  # διάνυσμα 2D (θέσεις, ταχύτητες, δυνάμεις)

# Ένα όριο B (τοίχος) ως ευθύγραμμο τμήμα από το σημείο a στο σημείο b (σε συντεταγμένες του ContinuousSpace)
struct Border
    a::Vec2                                      # αρχή του τμήματος
    b::Vec2                                      # τέλος του τμήματος
    Border(a, b) = new(Vec2(a), Vec2(b))         # δέχεται και tuples
end

# Η κατάσταση ενός πεζού α που δεν υπάρχει ως πεδίο του agent: κρατιέται στο `SocialForce` ανά agent id
mutable struct Pedestrian
    w::Vec2                                      # προτιμώμενη ταχύτητα w_α(t) (η μεταβλητή που ολοκληρώνεται, eq 10)
    v::Vec2                                      # πραγματική ταχύτητα v_α(t) = w g(v_max/‖w‖) (eq 11)
    e::Vec2                                      # επιθυμητή κατεύθυνση e_α(t) του τελευταίου βήματος (eq 1)
    v0::Float64                                  # επιθυμητή ταχύτητα v⁰_α
    vmax::Float64                                # μέγιστη αποδεκτή ταχύτητα v_α^max
end

"""
    SocialForce(pathfinder::GridPathfinder{2}; kwargs...)
    SocialForce(extent::NTuple{2,<:Real}; borders = [], kwargs...)
Social force model of Helbing & Molnár (Phys. Rev. E 51, 4282, 1995; arXiv:cond-mat/9805244).
Holds the model parameters, the borders (walls) and the per-pedestrian state (`w`, `v`, `e`, `v⁰`, `v^max`).
The state is stored here (keyed by agent id), so agents need no extra fields.

The first form takes the walls from the `walkmap` of a `pathfinder` (every boundary between walkable and
non-walkable cells, and the domain edge in non-periodic dimensions, becomes a border; collinear cell edges are
merged into single segments). The second form takes explicit `borders`, a collection of `(a, b)` point pairs.

Advance the pedestrians with [`social_force_step!`](@ref); the destination of every pedestrian comes from a path
planned by [`plan_route!`](@ref) (A*, D* Lite), a [`FlowField`](@ref), a target point or a function (see
[`desired_direction`](@ref)).

## Keywords (defaults are the values of Section IV of the paper, in SI units)
* `tau = 0.5`: relaxation time `τ` (eq 2).
* `V0 = 2.1`, `sigma = 0.3`: strength and range of the pedestrian repulsion `V = V0 exp(-b/σ)` (eq 13).
* `step_time = 2.0`: the time `Δt` of eq 4 that fixes the length `v_β Δt` of the elliptical potential
  (not the integration step).
* `U0 = 10.0`, `R = 0.2`: strength and range of the wall repulsion `U = U0 exp(-‖r‖/R)` (eq 13).
* `angle_of_sight = 200.0`: effective angle `2φ` of sight in degrees (eq 7), `c = 0.5`: weight outside it.
* `v0_mean = 1.34`, `v0_std = 0.26`: Gaussian distribution of the desired speeds `v⁰`;
  `vmax_factor = 1.3`: `v^max = vmax_factor * v⁰`.
* `fluctuation = 0.0`: amplitude of the Langevin noise of eq 10 (the paper's simulations use none).
* `attraction = nothing`: optional `attraction(agent, model)` returning an iterable of attractive forces `f_αi`
  (eq 6); each one is weighted by `w(e_α, f_αi)` (eq 8).
* `interaction_range = 4.0`: pedestrians farther than this are ignored; `wall_range = 5R`: same for walls
  (fixed at construction); `waypoint_radius = 0.5`: distance at which a path waypoint counts as reached.
* `domain_walls = true`, `extra_borders = []`, `offset = (0.0, 0.0)`: (pathfinder form) treat the domain edge as a
  wall / add more borders / shift all walkmap borders by `offset`. Cell `(i, j)` covers `[(i-1) cx, i cx] × [(j-1) cy, j cy]`
  by default; use `offset = (0.5, 0.5)` (with unit cells) if you treat the cell index itself as the cell centre.

Only 2D is supported. The model takes the model's length and time units; the defaults assume metres and seconds.
"""
mutable struct SocialForce
    tau::Float64                                 # χρόνος χαλάρωσης τ (eq 2)
    V0::Float64                                  # ένταση άπωσης πεζών V⁰ (eq 13)
    sigma::Float64                               # εμβέλεια άπωσης πεζών σ (eq 13)
    step_time::Float64                           # Δt του eq 4 (μήκος βήματος v_β Δt της έλλειψης)
    U0::Float64                                  # ένταση άπωσης τοίχων U⁰ (eq 13)
    R::Float64                                   # εμβέλεια άπωσης τοίχων R (eq 13)
    phi::Float64                                 # μισή γωνία όρασης φ σε rad (eq 7)
    c::Float64                                   # βάρος για ό,τι βρίσκεται εκτός οπτικού πεδίου (eq 7)
    v0_mean::Float64                             # μέση επιθυμητή ταχύτητα ⟨v⁰⟩
    v0_std::Float64                              # τυπική απόκλιση της v⁰
    vmax_factor::Float64                         # v^max = vmax_factor · v⁰
    fluctuation::Float64                         # πλάτος του θορύβου (όρος fluctuations του eq 10)
    attraction::Union{Nothing,Function}          # προαιρετική συνάρτηση ελκτικών δυνάμεων (eq 6)
    interaction_range::Float64                   # μέγιστη απόσταση αλληλεπίδρασης με άλλους πεζούς
    wall_range::Float64                          # μέγιστη απόσταση αλληλεπίδρασης με τοίχους (= μέγεθος bucket)
    waypoint_radius::Float64                     # απόσταση στην οποία ένα waypoint θεωρείται ότι επιτεύχθηκε
    borders::Vector{Border}                      # τα όρια B (τοίχοι)
    bucket_size::Float64                         # πλευρά των buckets του χωρικού ευρετηρίου
    buckets::Matrix{Vector{Int}}                 # χωρικό ευρετήριο: ανά bucket οι δείκτες των borders που είναι κοντά
    pedestrians::Dict{Int,Pedestrian}            # κατάσταση ανά agent id
end

function SocialForce(
    extent::Union{NTuple{2,<:Real},SVector{2,<:Real}};
    borders = Border[],
    tau = 0.5,
    V0 = 2.1,
    sigma = 0.3,
    step_time = 2.0,
    U0 = 10.0,
    R = 0.2,
    angle_of_sight = 200.0,
    c = 0.5,
    v0_mean = 1.34,
    v0_std = 0.26,
    vmax_factor = 1.3,
    fluctuation = 0.0,
    attraction = nothing,
    interaction_range = 4.0,
    wall_range = 5R,
    waypoint_radius = 0.5,
)
    all(>(0), (tau, sigma, R, wall_range)) || throw(ArgumentError("tau, sigma, R and wall_range must be positive"))
    brd = Border[b isa Border ? b : Border(b[1], b[2]) for b in borders]       # κανονικοποίηση σε Border
    # Το bucket δεν χρειάζεται να είναι μικρότερο από ~extent/256: μεγαλύτερα buckets μένουν σωστά (απλώς περιέχουν περισσότερα borders)
    # και αποφεύγεται ο τεράστιος πίνακας όταν το wall_range είναι πολύ μικρό σε σχέση με τον χώρο (π.χ. χάρτης σε pixels)
    bucket_size = max(Float64(wall_range), maximum(extent) / 256)
    buckets = build_buckets(brd, Vec2(extent), bucket_size, Float64(wall_range))   # χωρικό ευρετήριο για γρήγορη εύρεση κοντινών τοίχων
    return SocialForce(
        tau, V0, sigma, step_time, U0, R, deg2rad(angle_of_sight) / 2, c, v0_mean, v0_std, vmax_factor,
        fluctuation, attraction, interaction_range, wall_range, waypoint_radius, brd, bucket_size, buckets,
        Dict{Int,Pedestrian}(),
    )
end

# ---------------- Τοίχοι (borders) ----------------

# Δείκτης bucket (i, j) μιας θέσης, με περιορισμό εντός του πίνακα
@inline function bucket_index(pos, range, nb)
    return (clamp(floor(Int, pos[1] / range) + 1, 1, nb[1]), clamp(floor(Int, pos[2] / range) + 1, 1, nb[2]))
end

# Χωρικό ευρετήριο των borders: κάθε border μπαίνει σε όλα τα buckets (πλευράς `bsize`) που απέχουν ≤ range από αυτό,
# ώστε για μια θέση αρκεί να κοιτάξουμε ΕΝΑ bucket για να βρούμε όλα τα borders σε απόσταση ≤ range.
function build_buckets(borders::Vector{Border}, extent::Vec2, bsize::Float64, range::Float64)
    nb = (max(1, ceil(Int, extent[1] / bsize)), max(1, ceil(Int, extent[2] / bsize)))   # πλήθος buckets ανά διάσταση
    buckets = [Int[] for _ in 1:nb[1], _ in 1:nb[2]]
    for (k, b) in enumerate(borders)
        lo = min.(b.a, b.b) .- range                                              # πλαίσιο του border διογκωμένο κατά range
        hi = max.(b.a, b.b) .+ range
        i0, j0 = bucket_index(lo, bsize, nb)
        i1, j1 = bucket_index(hi, bsize, nb)
        for j in j0:j1, i in i0:i1
            push!(buckets[i, j], k)
        end
    end
    return buckets
end

# Το πλησιέστερο σημείο του τμήματος [a, b] στο σημείο p (το r_B^α του paper, eq 5)
@inline function nearest_point(p::Vec2, a::Vec2, b::Vec2)
    ab = b - a
    len2 = dot(ab, ab)
    len2 == 0 && return a                                                         # εκφυλισμένο τμήμα (σημείο)
    t = clamp(dot(p - a, ab) / len2, 0.0, 1.0)                                    # προβολή του p στην ευθεία, περιορισμένη στο τμήμα
    return a + t * ab
end

# true αν το cell (i, j) είναι «τοίχος»: μη βατό cell, ή εκτός grid σε μη περιοδική διάσταση όταν `domain_walls`
function is_blocked(walkmap::BitMatrix, i::Int, j::Int, periodic, domain_walls::Bool)
    nx, ny = size(walkmap)
    if i < 1 || i > nx
        periodic[1] || return domain_walls                                       # εκτός ορίων: τοίχος μόνο αν ζητήθηκε
        i = mod1(i, nx)                                                          # περιοδική διάσταση: wrap-around
    end
    if j < 1 || j > ny
        periodic[2] || return domain_walls
        j = mod1(j, ny)
    end
    return !walkmap[i, j]
end

"""
    borders_from_walkmap(walkmap, dims, periodic = (false, false); domain_walls = true)
Return the borders of the non-walkable cells of a 2D `walkmap` covering a space of size `dims`: every cell edge
between a walkable and a non-walkable cell (or the domain edge, if `domain_walls`) becomes a segment, and
collinear neighbouring edges are merged. Cell `(i, j)` covers `[(i-1) cx, i cx] × [(j-1) cy, j cy]`, as in
`to_discrete_position`.
"""
function borders_from_walkmap(walkmap::BitMatrix, dims, periodic = (false, false); domain_walls::Bool = true)
    nx, ny = size(walkmap)
    cx, cy = dims[1] / nx, dims[2] / ny                                          # μέγεθος cell σε συνεχείς συντεταγμένες
    blocked(i, j) = is_blocked(walkmap, i, j, periodic, domain_walls)
    borders = Border[]
    # κατακόρυφες ακμές στη γραμμή x = i·cx, ανάμεσα στα cells (i, j) και (i+1, j)
    for i in 0:(periodic[1] ? nx - 1 : nx)                                        # σε περιοδική διάσταση η γραμμή x = nx·cx ταυτίζεται με την x = 0
        j = 1
        while j <= ny
            if blocked(i, j) != blocked(i + 1, j)
                j0 = j
                while j <= ny && blocked(i, j) != blocked(i + 1, j)               # συγχώνευση διαδοχικών ακμών σε ένα τμήμα
                    j += 1
                end
                push!(borders, Border(Vec2(i * cx, (j0 - 1) * cy), Vec2(i * cx, (j - 1) * cy)))
            else
                j += 1
            end
        end
    end
    # οριζόντιες ακμές στη γραμμή y = j·cy, ανάμεσα στα cells (i, j) και (i, j+1)
    for j in 0:(periodic[2] ? ny - 1 : ny)
        i = 1
        while i <= nx
            if blocked(i, j) != blocked(i, j + 1)
                i0 = i
                while i <= nx && blocked(i, j) != blocked(i, j + 1)
                    i += 1
                end
                push!(borders, Border(Vec2((i0 - 1) * cx, j * cy), Vec2((i - 1) * cx, j * cy)))
            else
                i += 1
            end
        end
    end
    return borders
end

function SocialForce(
    pathfinder::GridPathfinder{2,P};
    domain_walls::Bool = true,
    extra_borders = Border[],
    offset = (0.0, 0.0),
    kwargs...,
) where {P}
    periodic = P isa Bool ? (P, P) : Tuple(P)                                    # περιοδικότητα ανά διάσταση
    off = Vec2(offset)                                                           # μετατόπιση των τοίχων (σύμβαση συντεταγμένων των cells)
    brd = Border[Border(b.a + off, b.b + off) for b in borders_from_walkmap(pathfinder.walkmap, pathfinder.dims, periodic; domain_walls)]
    return SocialForce(pathfinder.dims; borders = vcat(brd, Border[Border(b[1], b[2]) for b in extra_borders]), kwargs...)
end

# ---------------- Όροι δύναμης ----------------

# eq (2): όρος επιτάχυνσης προς την επιθυμητή ταχύτητα v⁰ e με χρόνο χαλάρωσης τ
@inline driving_force(v::Vec2, e::Vec2, v0::Float64, tau::Float64) = (v0 * e - v) / tau

# eq (3)-(4), (13): άπωση που δέχεται ο α από τον β. `r` = r_αβ = r_α − r_β, `e_b` η επιθυμητή κατεύθυνση του β,
# `v_b` η ταχύτητά του (μέτρο). Ισοδυναμικές γραμμές: ελλείψεις προσανατολισμένες προς την κατεύθυνση κίνησης του β.
# Αναλυτική κλίση: από 4b² = (d + d₂)² − s² με d = ‖r‖, d₂ = ‖r − s e_b‖, s = v_b Δt έχουμε
#   ∇b = (d + d₂)/(4b) · (r/d + (r − s e_b)/d₂),   f = −∇V = (V⁰/σ) e^{−b/σ} ∇b.
function pedestrian_repulsion(r::Vec2, e_b::Vec2, v_b::Float64, V0::Float64, sigma::Float64, step_time::Float64)
    s = v_b * step_time                                                          # μήκος βήματος του β: s_β = v_β Δt
    q = r - s * e_b                                                              # r_αβ − v_β Δt e_β
    d = norm(r)
    d2 = norm(q)
    (d == 0.0 || d2 == 0.0) && return zero(Vec2)                                 # εκφυλισμένες θέσεις: η κλίση δεν ορίζεται
    b = sqrt(max(((d + d2)^2 - s^2) / 4, 0.0))                                   # ημιάξονας b (eq 4)· από την τριγωνική ανισότητα b ≥ 0
    if b < 1e-7
        # Ο α βρίσκεται ακριβώς πάνω στο βήμα του β (b = 0, μέτωπο με μέτωπο): η κλίση του b δεν ορίζεται, αλλά το όριο
        # από οποιαδήποτε πλευρά είναι πεπερασμένο και κάθετο στο e_β, με |∇b| = s/(2 sqrt(x(s − x))) για x = r·e_β ∈ (0, s).
        # Το paper αφήνει την επιλογή πλευράς στις fluctuations· εδώ επιλέγεται ντετερμινιστικά η αριστερή πλευρά του β
        # (άρα η δεξιά του α που έρχεται αντίθετα), ώστε δύο αντίθετοι πεζοί να παρακάμπτονται με την ίδια φορά.
        x = dot(r, e_b)
        (x <= 0.0 || x >= s) && return zero(Vec2)
        return (V0 / sigma) * s / (2 * sqrt(x * (s - x))) * Vec2(-e_b[2], e_b[1])
    end
    grad_b = (d + d2) / (4b) * (r / d + q / d2)                                  # ∇_r b
    return (V0 / sigma) * exp(-b / sigma) * grad_b                               # f_αβ = −∇V, V = V⁰ e^{−b/σ}
end

# eq (5), (13): άπωση από το πλησιέστερο σημείο του ορίου. `r` = r_αB = r_α − r_B^α· U = U⁰ e^{−‖r‖/R}
function wall_repulsion(r::Vec2, U0::Float64, R::Float64)
    d = norm(r)
    d == 0.0 && return zero(Vec2)                                                # ο πεζός πάνω στον τοίχο: η κατεύθυνση δεν ορίζεται
    return (U0 / R) * exp(-d / R) * (r / d)                                      # F_αB = −∇U, μοναδιαίο διάνυσμα από τον τοίχο προς τον πεζό
end

# eq (7): βάρος w(e, f): 1 αν η f είναι μέσα στο οπτικό πεδίο γύρω από την e (e·f ≥ ‖f‖ cos φ), αλλιώς c
@inline direction_weight(e::Vec2, f::Vec2, cosphi::Float64, c::Float64) = dot(e, f) >= norm(f) * cosphi ? 1.0 : c

# eq (11)-(12): πραγματική ταχύτητα v = w g(v_max/‖w‖), με g = 1 αν ‖w‖ ≤ v_max, αλλιώς v_max/‖w‖
@inline function cap_velocity(w::Vec2, vmax::Float64)
    n = norm(w)
    return n <= vmax ? w : w * (vmax / n)
end
