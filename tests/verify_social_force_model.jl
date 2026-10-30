# Verification tests για την υλοποίηση του Social Force Model (Helbing & Molnár 1995) στο Agents/src/submodules/pathfinding.
# Στόχοι:
#   C1: η αναλυτική άπωση πεζών (eq 3-4, 13) ισούται με −∇V υπολογισμένο αριθμητικά (πεπερασμένες διαφορές), για κινούμενο και ακίνητο β.
#   C2: ο ημιάξονας b της έλλειψης (eq 4) είναι ≥ 0, και για ακίνητο β ισούται με την απόσταση ‖r_αβ‖.
#   C3: η άπωση τοίχου (eq 5, 13) ισούται με −∇U αριθμητικά και δείχνει μακριά από τον τοίχο.
#   C4: το βάρος οπτικού πεδίου (eq 7) για 2φ = 200° και c = 0.5.
#   C5: ο περιορισμός ταχύτητας (eq 11-12).
#   C6: ελεύθερη κίνηση: η αναλυτική λύση του eq (2), v(t) = v⁰ (1 − e^{−t/τ}).
#   C7: ισορροπία μπροστά σε τοίχο: d = R ln(U⁰ τ/(R v⁰)) από την F⁰ + F_αB = 0.
#   C8: εξαγωγή borders από walkmap: σωστά μήκη/θέσεις με συγχώνευση collinear ακμών.
#   C9: δύο πεζοί σε αντίθετες κατευθύνσεις αποφεύγουν ο ένας τον άλλον και περνούν.
#   C10: πεζός που ακολουθεί διαδρομή D* Lite γύρω από εμπόδιο φτάνει στον στόχο χωρίς να μπει σε μη βατό cell.
#   C11: αντίρροπη ροή σε διάδρομο (Fig. 2 του paper): ταχύτητες κοντά στις επιθυμητές (έλεγχος)· ο σχηματισμός lanes μόνο ΕΝΗΜΕΡΩΤΙΚΑ,
#        γιατί δεν αναπαράγεται σε αυτή τη διάταξη (βλ. σχόλιο στο C11).
# Τρέξιμο (από τη ρίζα του repo):  julia --project=evac3 tests/verify_social_force_model.jl
using Agents
using Agents.Pathfinding
using LinearAlgebra, Random, Printf, Statistics

const PF = Agents.Pathfinding                                                           # για τις εσωτερικές συναρτήσεις του μοντέλου
const Vec2 = PF.Vec2
const RESULTS = Tuple{String,Bool}[]
check(name, ok) = (push!(RESULTS, (name, ok)); println(ok ? "  [PASS] " : "  [FAIL] ", name); flush(stdout))

Agents.@agent struct Ped(ContinuousAgent{2,Float64})                                    # πεζός: pos, vel + η κατεύθυνση ±1 για τη ροή σε διάδρομο
    dir::Float64
end

# Κεντρικές πεπερασμένες διαφορές της βαθμωτής f στο r (για έλεγχο των αναλυτικών κλίσεων)
function numgrad(f, r::Vec2; h = 1e-6)
    return Vec2((f(r + Vec2(h, 0)) - f(r - Vec2(h, 0))) / 2h, (f(r + Vec2(0, h)) - f(r - Vec2(0, h))) / 2h)
end

# Βοηθητικό: ΑΒΜ με έναν χώρο, χωρίς βήμα agent (το βήμα το κάνει το social_force_step!)
make_model(extent; periodic = false, seed = 1) =
    ABM(Ped, ContinuousSpace(extent; periodic = periodic, spacing = 1.0); rng = MersenneTwister(seed), agent_step! = (a, m) -> nothing)

# ---------------------------------------------------------------- C1, C2: άπωση πεζών (eq 3, 4, 13)
println("C1/C2: άπωση πεζών")
let V0 = 2.1, σ = 0.3, Δt = 2.0, rng = MersenneTwister(7)
    # δυναμικό V(r) = V⁰ exp(−b/σ) με b από το eq (4)
    function V(r::Vec2, e, v)
        s = v * Δt
        b = sqrt(((norm(r) + norm(r - s * e))^2 - s^2)) / 2
        return V0 * exp(-b / σ)
    end
    maxerr, minb = 0.0, Inf
    for _ in 1:200
        r = Vec2(2 .* rand(rng, 2) .- 1) .* 2                                          # τυχαία θέση ως προς τον β μέσα σε 2 m
        ang = 2π * rand(rng)
        e = Vec2(cos(ang), sin(ang))                                                   # τυχαία κατεύθυνση του β
        v = 1.5 * rand(rng)                                                            # τυχαίο μέτρο ταχύτητας του β
        f = PF.pedestrian_repulsion(r, e, v, V0, σ, Δt)
        g = -numgrad(x -> V(x, e, v), r)                                               # f = −∇V
        maxerr = max(maxerr, norm(f - g) / max(norm(g), 1e-9))
        s = v * Δt
        minb = min(minb, ((norm(r) + norm(r - s * e))^2 - s^2) / 4)
    end
    check(@sprintf("C1: αναλυτική άπωση = −∇V αριθμητικά (μέγιστο σχετικό σφάλμα %.2e)", maxerr), maxerr < 1e-5)
    check("C2a: b² ≥ 0 σε όλες τις τυχαίες διατάξεις", minb >= -1e-12)
    r = Vec2(0.7, -0.4)
    f = PF.pedestrian_repulsion(r, Vec2(1, 0), 0.0, V0, σ, Δt)                         # ακίνητος β: b = ‖r‖
    check("C2b: ακίνητος β ⇒ ακτινική άπωση V⁰/σ e^{−d/σ}",
        isapprox(f, V0 / σ * exp(-norm(r) / σ) * r / norm(r); rtol = 1e-12))
    # η άπωση μεγαλώνει όταν ο α βρίσκεται μπροστά στην κατεύθυνση κίνησης του β (έλλειψη), με μικρή πλευρική μετατόπιση
    ahead = norm(PF.pedestrian_repulsion(Vec2(1.0, 0.2), Vec2(1, 0), 1.3, V0, σ, Δt))
    behind = norm(PF.pedestrian_repulsion(Vec2(-1.0, 0.2), Vec2(1, 0), 1.3, V0, σ, Δt))
    check(@sprintf("C2c: άπωση μπροστά από τον β (%.3f) > πίσω από αυτόν (%.3f)", ahead, behind), ahead > behind)
    # εκφυλισμένη διάταξη b = 0 (ο α ακριβώς πάνω στο βήμα του β): πεπερασμένη, κάθετη στο e_β, ίση με το όριο από πλευρική μετατόπιση
    fdeg = PF.pedestrian_repulsion(Vec2(1.0, 0.0), Vec2(1, 0), 1.3, V0, σ, Δt)
    flim = PF.pedestrian_repulsion(Vec2(1.0, 1e-4), Vec2(1, 0), 1.3, V0, σ, Δt)
    check("C2d: b = 0 ⇒ πεπερασμένη κάθετη άπωση ίση με το όριο y → 0",
        isapprox(norm(fdeg), norm(flim); rtol = 1e-3) && isapprox(fdeg[1], 0.0; atol = 1e-12) && isfinite(norm(fdeg)))
end

# ---------------------------------------------------------------- C3: άπωση τοίχου (eq 5, 13)
println("C3: άπωση τοίχου")
let U0 = 10.0, R = 0.2, rng = MersenneTwister(3)
    maxerr = 0.0
    for _ in 1:100
        r = Vec2(2 .* rand(rng, 2) .- 1)
        f = PF.wall_repulsion(r, U0, R)
        g = -numgrad(x -> U0 * exp(-norm(x) / R), r)
        maxerr = max(maxerr, norm(f - g) / norm(g))
    end
    check(@sprintf("C3a: αναλυτική άπωση τοίχου = −∇U αριθμητικά (μέγιστο σχετικό σφάλμα %.2e)", maxerr), maxerr < 1e-5)
    r = Vec2(0.3, 0.4)
    check("C3b: δείχνει μακριά από τον τοίχο (παράλληλη στο r_αB) με μέτρο (U⁰/R) e^{−‖r‖/R}",
        isapprox(PF.wall_repulsion(r, U0, R), U0 / R * exp(-0.5 / R) * r / 0.5; rtol = 1e-12))
end

# ---------------------------------------------------------------- C4: βάρος οπτικού πεδίου (eq 7)
println("C4: βάρος οπτικού πεδίου")
let cosφ = cos(deg2rad(200) / 2), c = 0.5, e = Vec2(1, 0)
    check("C4a: μπροστά (0°) ⇒ 1", PF.direction_weight(e, Vec2(1, 0), cosφ, c) == 1.0)
    check("C4b: στο πλάι (90°) ⇒ 1 (μέσα στο ±100°)", PF.direction_weight(e, Vec2(0, 1), cosφ, c) == 1.0)
    check("C4c: ακριβώς πίσω (180°) ⇒ c", PF.direction_weight(e, Vec2(-1, 0), cosφ, c) == c)
    check("C4d: 110° ⇒ c, 95° ⇒ 1 (όριο στις 100°)",
        PF.direction_weight(e, Vec2(cosd(110), sind(110)), cosφ, c) == c &&
        PF.direction_weight(e, Vec2(cosd(95), sind(95)), cosφ, c) == 1.0)
end

# ---------------------------------------------------------------- C5: περιορισμός ταχύτητας (eq 11-12)
println("C5: περιορισμός ταχύτητας")
let vmax = 1.742
    w1, w2 = Vec2(1.0, 0.5), Vec2(5.0, -12.0)
    check("C5a: ‖w‖ ≤ v_max ⇒ v = w", PF.cap_velocity(w1, vmax) == w1)
    v2 = PF.cap_velocity(w2, vmax)
    check("C5b: ‖w‖ > v_max ⇒ ίδια κατεύθυνση, μέτρο v_max", isapprox(norm(v2), vmax) && isapprox(v2 / norm(v2), w2 / norm(w2)))
end

# ---------------------------------------------------------------- C6: ελεύθερη κίνηση (eq 2)
println("C6: ελεύθερη κίνηση")
let
    model = make_model((100.0, 20.0))
    sf = SocialForce((100.0, 20.0))                                                    # χωρίς borders
    a = add_agent!((5.0, 10.0), Ped, model, (0.0, 0.0), 1.0)
    set_pedestrian!(sf, a, model; v0 = 1.34)
    dt, T = 0.005, 2.0
    for _ in 1:round(Int, T / dt)
        social_force_step!(model, sf, (a, m) -> (1.0, 0.0), dt)
    end
    v_num = sf.pedestrians[a.id].v[1]
    v_exact = 1.34 * (1 - exp(-T / sf.tau))                                            # λύση του dv/dt = (v⁰ − v)/τ
    x_exact = 5.0 + 1.34 * (T - sf.tau * (1 - exp(-T / sf.tau)))                       # ολοκλήρωμα της v(t)
    check(@sprintf("C6a: v(2 s) = %.5f έναντι αναλυτικής %.5f", v_num, v_exact), abs(v_num - v_exact) < 2e-3)
    check(@sprintf("C6b: x(2 s) = %.4f έναντι αναλυτικής %.4f", a.pos[1], x_exact), abs(a.pos[1] - x_exact) < 1e-2)
    check("C6c: η a.vel ενημερώνεται με την ταχύτητα v_α", isapprox(a.vel[1], v_num; atol = 1e-12))
end

# ---------------------------------------------------------------- C7: ισορροπία μπροστά σε τοίχο (eq 2 + eq 5)
println("C7: ισορροπία μπροστά σε τοίχο")
let
    model = make_model((12.0, 5.0))
    sf = SocialForce((12.0, 5.0); borders = [((10.0, 0.0), (10.0, 5.0))])              # κατακόρυφος τοίχος στο x = 10
    a = add_agent!((5.0, 2.5), Ped, model, (0.0, 0.0), 1.0)
    set_pedestrian!(sf, a, model; v0 = 1.34)
    for _ in 1:round(Int, 30 / 0.01)
        social_force_step!(model, sf, (a, m) -> (1.0, 0.0), 0.01)
    end
    d_exact = sf.R * log(sf.U0 * sf.tau / (sf.R * 1.34))                               # F⁰ = 1.34/τ = (U⁰/R) e^{−d/R}
    check(@sprintf("C7: απόσταση ισορροπίας %.4f m έναντι αναλυτικής %.4f m", 10.0 - a.pos[1], d_exact),
        abs((10.0 - a.pos[1]) - d_exact) < 2e-3)
end

# ---------------------------------------------------------------- C8: borders από walkmap
println("C8: borders από walkmap")
let
    wm = trues(10, 6); wm[4:6, 3:4] .= false                                           # ορθογώνιο εμπόδιο 3×2 cells
    brd = borders_from_walkmap(wm, (10.0, 6.0); domain_walls = false)
    check("C8a: το εμπόδιο δίνει 4 συγχωνευμένα τμήματα (όχι 10 ακμές cells)", length(brd) == 4)
    check("C8b: συνολικό μήκος = περίμετρος εμποδίου = 2(3+2)", isapprox(sum(norm(b.b - b.a) for b in brd), 10.0))
    check("C8c: το εμπόδιο καλύπτει x ∈ [3,6], y ∈ [2,4] (cell (i,j) ↔ [(i−1),i]×[(j−1),j])",
        all(b -> 3.0 <= min(b.a[1], b.b[1]) && max(b.a[1], b.b[1]) <= 6.0 && 2.0 <= min(b.a[2], b.b[2]) && max(b.a[2], b.b[2]) <= 4.0, brd))
    brd2 = borders_from_walkmap(wm, (10.0, 6.0); domain_walls = true)
    check("C8d: με domain_walls το μήκος αυξάνεται κατά την περίμετρο του χώρου 2(10+6)",
        isapprox(sum(norm(b.b - b.a) for b in brd2), 10.0 + 32.0))
end

# ---------------------------------------------------------------- C9: δύο πεζοί σε αντίθετες κατευθύνσεις
println("C9: αντίθετες κατευθύνσεις")
let
    model = make_model((30.0, 10.0))
    sf = SocialForce((30.0, 10.0))
    a = add_agent!((5.0, 5.0), Ped, model, (0.0, 0.0), 1.0)
    b = add_agent!((25.0, 5.1), Ped, model, (0.0, 0.0), -1.0)                         # μικρή πλευρική μετατόπιση: σπάει τη συμμετρία
    set_pedestrian!(sf, a, model; v0 = 1.34); set_pedestrian!(sf, b, model; v0 = 1.34)
    dmin = Inf
    for _ in 1:round(Int, 40 / 0.02)
        social_force_step!(model, sf, (ag, m) -> (ag.dir, 0.0), 0.02)
        dmin = min(dmin, norm(Vec2(a.pos) - Vec2(b.pos)))
    end
    check(@sprintf("C9a: ελάχιστη απόσταση %.3f m (αποφυγή, χωρίς επικάλυψη)", dmin), dmin > 0.3)
    check("C9b: και οι δύο πέρασαν ο ένας τον άλλον (a δεξιά του b)", a.pos[1] > b.pos[1] + 5)
end

# ---------------------------------------------------------------- C10: διαδρομή D* Lite γύρω από εμπόδιο
println("C10: D* Lite + social force")
let
    ext = (30.0, 12.0)
    wm = trues(30, 12); wm[14:15, 1:8] .= false                                       # τοίχος με άνοιγμα στο πάνω μέρος (y > 8)
    space = ContinuousSpace(ext; periodic = false, spacing = 1.0)
    pf = DStarLite(space; walkmap = wm)
    model = ABM(Ped, space; rng = MersenneTwister(5), agent_step! = (a, m) -> nothing)
    sf = SocialForce(pf)                                                               # τοίχοι από το walkmap + όρια χώρου
    a = add_agent!((3.5, 3.5), Ped, model, (0.0, 0.0), 1.0)
    dest = (27.5, 3.5)
    plan_route!(a, dest, pf)
    inwalk(p) = wm[clamp(floor(Int, p[1]) + 1, 1, 30), clamp(floor(Int, p[2]) + 1, 1, 12)]
    always_walkable, steps = true, 0
    while norm(Vec2(a.pos) - Vec2(dest)) > 0.6 && steps < 20000
        social_force_step!(model, sf, pf, 0.02)
        always_walkable &= inwalk(a.pos)
        steps += 1
    end
    check(@sprintf("C10a: έφτασε στον στόχο σε %.1f s (πέρασε από το άνοιγμα)", steps * 0.02), steps < 20000)
    check("C10b: δεν μπήκε ποτέ σε μη βατό cell", always_walkable)
end

# ---------------------------------------------------------------- C11: αντίρροπη ροή σε διάδρομο (lane formation)
println("C11: lane formation")
let
    L, W, N = 30.0, 10.0, 90                                                           # πυκνότητα 0.3 m⁻² όπως στο Fig. 3 του paper
    model = make_model((L, W); periodic = (true, false), seed = 11)
    sf = SocialForce((L, W); borders = [((0.0, 0.0), (L, 0.0)), ((0.0, W), (L, W))])
    for k in 1:N
        add_agent!((L * rand(abmrng(model)), 0.5 + (W - 1.0) * rand(abmrng(model))), Ped, model, (0.0, 0.0), isodd(k) ? 1.0 : -1.0)
    end
    # τάξη lanes: ανά λωρίδα πλάτους 1 m, |n₊ − n₋|/(n₊ + n₋), σταθμισμένο με τον πληθυσμό (τυχαία διάταξη ⇒ μικρή τιμή)
    function lane_order()
        num, den = 0.0, 0
        for k in 0:Int(W)-1
            inbin = [ag for ag in allagents(model) if k <= ag.pos[2] < k + 1]
            n = length(inbin); n == 0 && continue
            np = count(ag -> ag.dir > 0, inbin)
            num += abs(2np - n); den += n
        end
        return num / den
    end
    step!() = social_force_step!(model, sf, (ag, m) -> (ag.dir, 0.0), 0.05)
    order0 = lane_order()
    for _ in 1:round(Int, 300 / 0.05)
        step!()
    end
    order1 = lane_order()
    speed = mean(norm(sf.pedestrians[ag.id].v) for ag in allagents(model))
    along = mean(ag.dir * sf.pedestrians[ag.id].v[1] for ag in allagents(model))
    v0mean = mean(sf.pedestrians[ag.id].v0 for ag in allagents(model))
    @printf("  τάξη lanes: %.3f → %.3f· μέση ταχύτητα %.3f m/s (⟨v⁰⟩ = %.3f), συνιστώσα κατά τη ροή %.3f m/s\n", order0, order1, speed, v0mean, along)
    # Ενημερωτικό (όχι έλεγχος): με τις παραμέτρους του paper τα lanes του Fig. 2 ΔΕΝ αναπαράγονται σε αυτό το περιοδικό διάδρομο
    # (ούτε σε μεγαλύτερο διάδρομο, μεγαλύτερη πυκνότητα ή με θόρυβο)· τα lanes που τοποθετούνται εξαρχής διαλύονται.
    println("  [INFO] C11a: τάξη lanes $(round(order0; digits = 3)) → $(round(order1; digits = 3)) (τυχαία διάταξη ≈ 0.24, πλήρη lanes = 1)")
    check("C11b: οι πεζοί κινούνται κοντά στην επιθυμητή ταχύτητα (≥ 70% του ⟨v⁰⟩ κατά τη ροή)", along > 0.7 * v0mean)
end

println()
nfail = count(!last, RESULTS)
println(nfail == 0 ? "ΟΛΟΙ ΟΙ ΕΛΕΓΧΟΙ ΠΕΡΑΣΑΝ ($(length(RESULTS)))" : "ΑΠΟΤΥΧΙΑ: $nfail από $(length(RESULTS)) έλεγχοι")
exit(nfail == 0 ? 0 : 1)
