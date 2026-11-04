# Verification test για τον Pathfinding.FlowField έναντι των άρθρων που παραθέτει το thread του r/gamedev (jfg3gf):
#   [A] howtorts "Generating a path with Dijkstra": flood fill από τον στόχο, μόνο 4 γείτονες, κόστος ακμής 1, τοίχοι = άπειρο.
#   [B] howtorts "Basic Flow Fields": διάνυσμα κάθε cell = κανονικοποιημένη κατεύθυνση προς τον γείτονα (8) με τη μικρότερη απόσταση·
#       οι agents διαβάζουν το πεδίο με διγραμμική παρεμβολή (γειτονικά cells, κανονικοποίηση).
#   [C] howtorts "Flow Field Improvements": η διαγώνια κίνηση επιτρέπεται μόνο αν είναι βατά και τα 2 ορθογώνια cells και το διαγώνιο (όχι corner cutting).
#   [D] howtorts "Flow Fields: Line of Sight" και Game AI Pro κεφ. 23: στα cells με οπτική επαφή ο agent πηγαίνει ευθεία στον στόχο.
# Έλεγχοι:
#   F1: 4 γείτονες -> το πεδίο αποστάσεων ταυτίζεται με ανεξάρτητο BFS [A].
#   F2: 8 γείτονες -> το πεδίο αποστάσεων (ορθογώνιο βήμα 1, διαγώνιο √2, χωρίς corner cutting) ταυτίζεται με ανεξάρτητο Dijkstra [A+B+C].
#   F3: τα διανύσματα ροής: μοναδιαία, προς τον γείτονα με τη μικρότερη απόσταση + κόστος βήματος (συντομότερη διαδρομή), μηδενικά σε στόχο/τοίχους/μη προσβάσιμα cells [B].
#   F4: η διαδρομή που ακολουθεί το πεδίο από κάθε προσβάσιμο cell φτάνει στον στόχο με μήκος ίσο με την απόσταση (βέλτιστη) [B].
#   F5: η διγραμμική παρεμβολή flow_at_continuous ταυτίζεται με τον τύπο του άρθρου [B].
#   F6: καμία διαγώνια κίνηση μέσα από τοίχο (corner cutting) [C].
#   F7: τα cells με LOS έχουν πράγματι ελεύθερη ευθεία προς τον στόχο και η flow_at_continuous δείχνει ευθεία στο κέντρο του στόχου [D].
#   Πληροφοριακά: πληρότητα του LOS έναντι ακριβούς ελέγχου ευθείας, προσομοίωση συνεχούς κίνησης agents.
# Τρέξιμο (από τη ρίζα του repo):  julia --project=evac3 tests/verify_flowfield.jl
using Agents
using DataStructures: PriorityQueue, dequeue_pair!
using LinearAlgebra: norm
using Printf, Random, Statistics

const PF = Agents.Pathfinding                                                            # το module με τους pathfinders
const N1, N2 = 40, 30                                                                    # διαστάσεις του δοκιμαστικού χάρτη
const NBRS4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]                                         # γειτονιά von Neumann
const NBRS8 = [(a, b) for a in -1:1 for b in -1:1 if (a, b) != (0, 0)]                   # γειτονιά Moore
const RESULTS = Tuple{String,Bool}[]
check(name, ok) = (push!(RESULTS, (name, ok)); println(ok ? "  [PASS] " : "  [FAIL] ", name); flush(stdout))

# Πλήθος cells όπου δύο πεδία αποστάσεων διαφέρουν (Inf πρέπει να ταυτίζεται, τα πεπερασμένα με ανοχή λόγω σειράς αθροίσματος).
field_mismatch(a, b) = count(i -> isinf(a[i]) != isinf(b[i]) || (isfinite(a[i]) && abs(a[i] - b[i]) > 1e-9), eachindex(a))

# Δοκιμαστικός χάρτης: τυχαία εμπόδια, μακρύς τοίχος, σχήμα U και ένα απομονωμένο (μη προσβάσιμο) cell.
function make_map(seed)
    rng = MersenneTwister(seed)
    w = BitArray(rand(rng, N1, N2) .> 0.22)                                              # ~22% εμπόδια
    w[10:30, 15] .= false                                                                # μακρύς οριζόντιος τοίχος
    w[30, 15:25] .= false                                                                # κάθετος τοίχος (σχήμα U με τον προηγούμενο)
    w[35:37, 3:5] .= false                                                               # κλειστός χώρος
    w[36, 4] = true                                                                      # απομονωμένο cell μέσα στον κλειστό χώρο
    return w
end

inmap(w, n) = 1 <= n[1] <= size(w, 1) && 1 <= n[2] <= size(w, 2)                        # εντός ορίων χάρτη
# Διαγώνιο βήμα (dx, dy) από το c περνά ανάμεσα από τοίχο;
cuts_corner(w, c, dx, dy) = dx != 0 && dy != 0 && !(w[c[1] + dx, c[2]] && w[c[1], c[2] + dy])

# Ανεξάρτητο Dijkstra από τον στόχο: βάρος ακμής 1 ή √2 (octile) και προαιρετικά απαγόρευση corner cutting.
function ref_dijkstra(w, dest, nbrs; diagcost = 1.0, nocut = false, pm = nothing, abspen = false)
    d = fill(Inf, size(w))
    d[dest...] = 0.0
    pq = PriorityQueue{Tuple{Int,Int},Float64}()
    pq[dest] = 0.0
    while !isempty(pq)
        c, dc = dequeue_pair!(pq)
        dc > d[c...] && continue                                                         # παλιά εγγραφή
        for (dx, dy) in nbrs
            n = (c[1] + dx, c[2] + dy)
            inmap(w, n) && w[n...] || continue                                           # εντός ορίων και βατό
            nocut && cuts_corner(w, c, dx, dy) && continue                               # [C]
            nd = dc + ((dx != 0 && dy != 0) ? diagcost : 1.0) + (pm === nothing ? 0.0 : (abspen ? abs(pm[c...]) + abs(pm[n...]) : abs(pm[c...] - pm[n...])))   # + penalty: |διαφορά| (PenaltyMap) ή |p(u)|+|p(v)| (AbsolutePenaltyMap)
            if nd < d[n...]
                d[n...] = nd
                pq[n] = nd
            end
        end
    end
    return d
end

# Δημιουργία FlowField με στόχο dest (όπως κάνει το plan_route!, χωρίς agent).
function build(w, dest, diag; pm = nothing, abspen = false)
    pf = PF.FlowField(size(w); diagonal_movement = diag, walkmap = w, penalty_map = pm, absolute_penalty = abspen)
    pf.destination = dest
    PF.generate_flow_field!(pf)
    return pf
end

w = make_map(7)
walkable = [Tuple(I) for I in CartesianIndices(w) if w[I]]
dest = (5, 5)
while !w[dest...] ; global dest = (dest[1] + 1, dest[2]) ; end                           # πρώτο βατό cell από το (5,5)
println("Χάρτης $(N1)×$(N2), βατά cells: $(length(walkable)), στόχος: $dest")

# --- F1/F2: πεδίο αποστάσεων έναντι ανεξάρτητου BFS ---
pf4 = build(w, dest, false)
pf8 = build(w, dest, true)
bad1 = field_mismatch(pf4.distance_grid, ref_dijkstra(w, dest, NBRS4))
bad2 = field_mismatch(pf8.distance_grid, ref_dijkstra(w, dest, NBRS8; diagcost = sqrt(2), nocut = true))
reach4 = count(isfinite, pf4.distance_grid)
unreach = count(w[I] && isinf(pf8.distance_grid[I]) for I in CartesianIndices(w))
walls_inf = all(isinf, pf8.distance_grid[.!w])                                           # οι τοίχοι δεν παίρνουν απόσταση
println("  προσβάσιμα cells (4 γείτονες): $reach4 · βατά αλλά μη προσβάσιμα (8 γείτονες): $unreach")

# --- F3/F4: διανύσματα ροής και ακολούθηση του πεδίου ---
function check_flow(pf, nbrs, pm = nothing; abspen = false)
    d = pf.distance_grid
    ecost(c, n) = ((n[1] != c[1] && n[2] != c[2]) ? sqrt(2) : 1.0) + (pm === nothing ? 0.0 : (abspen ? abs(pm[c...]) + abs(pm[n...]) : abs(pm[c...] - pm[n...])))   # κόστος βήματος c -> n
    bad_vec = 0                                                                          # παραβιάσεις κανόνων διανύσματος
    bad_walk = 0                                                                         # διαδρομές που δεν φτάνουν με βήματα = απόσταση
    for I in CartesianIndices(w)
        c = Tuple(I)
        v = PF.flow_at(pf, c)
        if !w[I] || c == pf.destination || isinf(d[I])
            all(==(0.0), v) || (bad_vec += 1)                                            # μηδενικό διάνυσμα
            continue
        end
        abs(norm(v) - 1) < 1e-9 || (bad_vec += 1; continue)                              # μοναδιαίο
        step = (round(Int, v[1]), round(Int, v[2]))
        n = (c[1] + step[1], c[2] + step[2])
        (step in nbrs && inmap(w, n) && w[n...]) || (bad_vec += 1; continue)            # βήμα προς βατό γείτονα
        best = minimum(d[(c[1] + dx, c[2] + dy)...] + ecost(c, (c[1] + dx, c[2] + dy)) for (dx, dy) in nbrs if inmap(w, (c[1] + dx, c[2] + dy)) && w[(c[1] + dx, c[2] + dy)...] && !cuts_corner(w, c, dx, dy))
        (isapprox(d[n...] + ecost(c, n), best; atol = 1e-9) && isapprox(best, d[I]; atol = 1e-9)) || (bad_vec += 1)                                         # ο γείτονας με τη μικρότερη απόσταση
        # ακολούθηση του πεδίου μέχρι τον στόχο
        pos, steps = c, 0.0
        n_it = 0
        while pos != pf.destination && n_it <= length(d)
            vv = PF.flow_at(pf, pos)
            all(==(0.0), vv) && break
            nxt = (pos[1] + round(Int, vv[1]), pos[2] + round(Int, vv[2]))
            steps += ecost(pos, nxt)
            pos = nxt
            n_it += 1
        end
        (pos == pf.destination && abs(steps - d[I]) < 1e-9) || (bad_walk += 1)
    end
    return bad_vec, bad_walk
end
(bv4, bw4) = check_flow(pf4, NBRS4)
(bv8, bw8) = check_flow(pf8, NBRS8)

# --- F5: διγραμμική παρεμβολή (συνεχής χώρος με dims = μέγεθος πλέγματος· το διάνυσμα του cell i δειγματοληπτείται στο κέντρο του, i-0.5) ---
pfc = PF.FlowField((Float64(N1), Float64(N2)); walkmap = w)
pfc.destination = dest
PF.generate_flow_field!(pfc)
rng = MersenneTwister(11)
bad_interp, n_interp = 0, 0
for _ in 1:2000
    p = (0.5 + rand(rng) * (N1 - 2), 0.5 + rand(rng) * (N2 - 2))                             # θέση ώστε να υπάρχουν τα cells i και i+1 (κέντρο cell i στο i-0.5)
    i, j = floor(Int, p[1] + 0.5), floor(Int, p[2] + 0.5)
    (w[i, j] && w[i+1, j] && w[i, j+1] && w[i+1, j+1]) || continue                       # και τα 4 cells βατά
    tx, ty = p[1] + 0.5 - i, p[2] + 0.5 - j
    F = pfc.flow_field
    top = ntuple(k -> F[i, j][k] * (1 - tx) + F[i+1, j][k] * tx, 2)                      # τύπος του [B]
    bot = ntuple(k -> F[i, j+1][k] * (1 - tx) + F[i+1, j+1][k] * tx, 2)
    dir = ntuple(k -> top[k] * (1 - ty) + bot[k] * ty, 2)
    nd = norm(dir)
    ref = nd > 0 ? dir ./ nd : (0.0, 0.0)
    global n_interp += 1
    norm(collect(PF.flow_at_continuous(pfc, p)) .- collect(ref)) < 1e-9 || (global bad_interp += 1)
end

# --- F6: corner cutting [C] ---
cut_steps = 0                                                                            # cells που η ροή τους περνά διαγώνια μέσα από τοίχο
for I in CartesianIndices(w)
    c = Tuple(I)
    v = PF.flow_at(pf8, c)
    all(==(0.0), v) && continue
    dx, dy = round(Int, v[1]), round(Int, v[2])
    cuts_corner(w, c, dx, dy) && (global cut_steps += 1)
end
d_nocut = ref_dijkstra(w, dest, NBRS8; diagcost = sqrt(2), nocut = true)
diff_nocut = field_mismatch(pf8.distance_grid, d_nocut)                                        # cells με διαφορετική απόσταση χωρίς corner cutting
unreach_nocut = count(isinf, d_nocut[w]) - unreach                                       # επιπλέον cells που αποκόβονται

# --- F7: LOS [D] ---
# Ακριβής έλεγχος ευθείας από το κέντρο του cell a στο κέντρο του b (κέντρο cell i στο i): όλα τα cells που αγγίζει το τμήμα πρέπει να είναι βατά.
function clear_segment(wm, a, b)
    len = norm(collect(b) .- collect(a))
    for t in range(0, 1; length = max(2, ceil(Int, len / 0.01) + 1))
        p = a .+ t .* (b .- a)
        for ex in (-1e-9, 1e-9), ey in (-1e-9, 1e-9)
            c = (floor(Int, p[1] + 0.5 + ex), floor(Int, p[2] + 0.5 + ey))
            (inmap(wm, c) && wm[c...]) || return false
        end
    end
    return true
end
los_cells = [Tuple(I) for I in CartesianIndices(w) if pfc.los_grid[I]]
los_unsound = count(c -> !clear_segment(w, c, dest), los_cells)                          # flagged αλλά με εμπόδιο στην ευθεία
reach_cells = [c for c in walkable if isfinite(pfc.distance_grid[c...])]
los_exact = count(c -> clear_segment(w, c, dest), reach_cells)                           # cells με πραγματική ευθεία
los_missed = count(c -> clear_segment(w, c, dest) && !pfc.los_grid[c...], reach_cells)   # ευθεία υπάρχει αλλά δεν flagged (συντηρητικό)
bad_los_dir = 0                                                                          # η ροή στα LOS cells δεν δείχνει ευθεία στον στόχο
goal = (dest[1] - 0.5, dest[2] - 0.5)                                                    # κέντρο του cell-στόχου (συνεχείς συντεταγμένες)
for c in los_cells
    c == dest && continue
    p = (c[1] - 1 + rand(rng), c[2] - 1 + rand(rng))                                     # τυχαίο σημείο μέσα στο cell
    v = collect(goal) .- collect(p)
    want = v ./ norm(v)
    norm(collect(PF.flow_at_continuous(pfc, p)) .- want) < 1e-9 || (global bad_los_dir += 1)
end
w_nolos = PF.FlowField((Float64(N1), Float64(N2)); walkmap = w, line_of_sight = false)
w_nolos.destination = dest
PF.generate_flow_field!(w_nolos)
nolos_empty = !any(w_nolos.los_grid)                                                     # με line_of_sight = false δεν σημειώνεται τίποτα

# --- Πληροφοριακά: προσομοίωση συνεχούς κίνησης (όπως το move_along_route! για ContinuousSpace) ---
function simulate(pf, wm, start; speed = 0.4, maxsteps = 800)
    pos = (start[1] - 0.5, start[2] - 0.5)                                               # κέντρο του cell
    entered_wall = false
    for _ in 1:maxsteps
        cell = (floor(Int, pos[1]) + 1, floor(Int, pos[2]) + 1)
        inmap(wm, cell) || return (:out, entered_wall)
        wm[cell...] || (entered_wall = true)
        cell == pf.destination && return (:reached, entered_wall)
        dir = PF.flow_at_continuous(pf, pos)
        all(≈(0), dir) && return (:stuck, entered_wall)
        pos = (pos[1] + dir[1] * speed, pos[2] + dir[2] * speed)
    end
    return (:timeout, entered_wall)
end
starts = [c for c in walkable if isfinite(pf8.distance_grid[c...])]
sim = [simulate(pfc, w, s) for s in starts]
n_reached = count(r -> r[1] == :reached, sim)
n_wall = count(r -> r[2], sim)
sim_status = Dict(k => count(r -> r[1] == k, sim) for k in (:reached, :stuck, :out, :timeout))
# Ίδια προσομοίωση σε αραιό χάρτη (μόνο ο τοίχος U, χωρίς τυχαία εμπόδια).
w_open = trues(N1, N2); w_open[10:30, 15] .= false; w_open[30, 15:25] .= false
pf_open = PF.FlowField((Float64(N1), Float64(N2)); walkmap = w_open)
pf_open.destination = (5, 5)
PF.generate_flow_field!(pf_open)
open_cells = [Tuple(I) for I in CartesianIndices(w_open) if w_open[I]]
open_exact = count(c -> clear_segment(w_open, c, (5, 5)), open_cells)                   # cells με πραγματική ευθεία στον αραιό χάρτη
open_flag = count(c -> pf_open.los_grid[c...], open_cells)
open_unsound = count(c -> pf_open.los_grid[c...] && !clear_segment(w_open, c, (5, 5)), open_cells)
starts_open = [Tuple(I) for I in CartesianIndices(w_open) if w_open[I] && isfinite(pf_open.distance_grid[I])]
sim_open = [simulate(pf_open, w_open, s) for s in starts_open]

# --- F8: penalty map (κόστος εδάφους, όπως το PenaltyMap του A*) ---
pm = Float64.(rand(MersenneTwister(5), 0:20, N1, N2))                                    # τυχαίο ανάγλυφο
pfp = build(w, dest, true; pm = pm)
bad_pm = field_mismatch(pfp.distance_grid, ref_dijkstra(w, dest, NBRS8; diagcost = sqrt(2), nocut = true, pm = pm))
(bvp, bwp) = check_flow(pfp, NBRS8, pm)
pfa = build(w, dest, true; pm = pm, abspen = true)                                        # AbsolutePenaltyMap: cells με penalty 0 (δρόμοι) είναι δωρεάν
bad_abs = field_mismatch(pfa.distance_grid, ref_dijkstra(w, dest, NBRS8; diagcost = sqrt(2), nocut = true, pm = pm, abspen = true))
(bva, bwa) = check_flow(pfa, NBRS8, pm; abspen = true)
pfa_los = PF.FlowField((Float64(N1), Float64(N2)); walkmap = w, penalty_map = pm, absolute_penalty = true)
pfa_los.destination = dest
PF.generate_flow_field!(pfa_los)
abs_los = count(pfa_los.los_grid)                                                        # με absolute_penalty δεν σημειώνεται LOS
pfp_los = PF.FlowField((Float64(N1), Float64(N2)); walkmap = w, penalty_map = pm)       # συνεχής χώρος με LOS και penalty
pfp_los.destination = dest
PF.generate_flow_field!(pfp_los)
los_bad_pm = count(I -> pfp_los.los_grid[I] && pm[I] != pm[dest...], CartesianIndices(w))   # LOS μόνο ανάμεσα σε cells ίδιου penalty

# --- F9: path_length_at = πλήθος cells της διαδρομής που ακολουθεί το πεδίο ---
bad_len = 0
for c in walkable[1:7:end]
    L = PF.path_length_at(pf8, c)
    if isinf(pf8.distance_grid[c...])
        L == typemax(Int) || (global bad_len += 1)
        continue
    end
    pos, n = c, 0
    while pos != dest
        v = PF.flow_at(pf8, pos)
        pos = (pos[1] + round(Int, v[1]), pos[2] + round(Int, v[2]))
        n += 1
    end
    L == n || (global bad_len += 1)
end

println("\n== Αποτελέσματα ==")
check("F1 [A] 4 γείτονες: πεδίο αποστάσεων = ανεξάρτητο BFS (αποκλίσεις cells = $bad1)", bad1 == 0)
check("F2 [A+B] 8 γείτονες: πεδίο αποστάσεων (διαγώνιο √2) = ανεξάρτητο Dijkstra χωρίς corner cutting (αποκλίσεις cells = $bad2)", bad2 == 0)
check("F2b οι τοίχοι έχουν άπειρη απόσταση και υπάρχουν μη προσβάσιμα cells ($unreach) που μένουν άπειρα", walls_inf && unreach > 0)
check("F3 [B] διανύσματα ροής (4 γείτονες / 8 γείτονες): παραβιάσεις $bv4 / $bv8", bv4 == 0 && bv8 == 0)
check("F4 [B] ακολούθηση του πεδίου φτάνει στον στόχο με μήκος = απόσταση (αποτυχίες $bw4 / $bw8)", bw4 == 0 && bw8 == 0)
check("F5 [B] διγραμμική παρεμβολή = τύπος του άρθρου ($(n_interp - bad_interp)/$n_interp δείγματα)", n_interp > 0 && bad_interp == 0)
check("F7 [D] LOS: cells χωρίς ελεύθερη ευθεία = $los_unsound · λάθος κατεύθυνση ροής = $bad_los_dir · χωρίς LOS όταν είναι απενεργοποιημένο = $nolos_empty", length(los_cells) > 1 && los_unsound == 0 && bad_los_dir == 0 && nolos_empty)
check("F7b [D] LOS στον αραιό χάρτη: flagged $open_flag από $open_exact cells με ευθεία · χωρίς ευθεία = $open_unsound", open_unsound == 0 && open_flag >= 0.8 * open_exact)
check("F8 penalty map: αποστάσεις = ανεξάρτητο Dijkstra με κόστος |Δpenalty| (αποκλίσεις $bad_pm) · διανύσματα/διαδρομές (παραβιάσεις $bvp / $bwp) · LOS σε cells διαφορετικού penalty = $los_bad_pm", bad_pm == 0 && bvp == 0 && bwp == 0 && los_bad_pm == 0)
check("F8b absolute penalty: αποστάσεις = ανεξάρτητο Dijkstra με κόστος |p(u)|+|p(v)| (αποκλίσεις $bad_abs) · διανύσματα/διαδρομές (παραβιάσεις $bva / $bwa) · cells με LOS = $abs_los", bad_abs == 0 && bva == 0 && bwa == 0 && abs_los == 0)
check("F9 path_length_at = πλήθος cells της διαδρομής (αποκλίσεις $bad_len)", bad_len == 0)
check("F6 [C] χωρίς corner cutting: διαγώνια βήματα μέσα από τοίχο = $cut_steps · cells με διαφορετική απόσταση = $diff_nocut · επιπλέον αποκομμένα cells = $unreach_nocut", cut_steps == 0 && diff_nocut == 0)

println("\n== Πληροφοριακά μεγέθη ==")
@printf("  προσομοίωση συνεχούς κίνησης: έφτασαν %d/%d (%.1f%%) · μπήκαν σε τοίχο %d · καταστάσεις %s
", n_reached, length(starts), 100 * n_reached / length(starts), n_wall, sim_status)
@printf("  προσομοίωση σε αραιό χάρτη (μόνο τοίχος U): έφτασαν %d/%d · κόλλησαν %d · timeout %d · μπήκαν σε τοίχο %d
", count(r -> r[1] == :reached, sim_open), length(starts_open), count(r -> r[1] == :stuck, sim_open), count(r -> r[1] == :timeout, sim_open), count(r -> r[2], sim_open))
@printf("  Line of Sight: %d cells flagged · ακριβής ευθεία σε %d/%d προσβάσιμα cells · συντηρητικά χαμένα %d
", length(los_cells), los_exact, length(reach_cells), los_missed)

nfail = count(!last, RESULTS)
println("\n== Σύνοψη ==\n  $(length(RESULTS) - nfail)/$(length(RESULTS)) έλεγχοι πέρασαν")
nfail > 0 && exit(1)
