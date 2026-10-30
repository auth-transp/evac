# Verification tests για τους αλγορίθμους εύρεσης διαδρομής του σεναρίου 3 (A* και D* Lite) πάνω στον πραγματικό χάρτη.
# Στόχοι:
#   C1: ο τύπος κόστους ακμής (1 + |p(u)| + |p(v)|, AbsolutePenaltyMap + MaxDistance) ταυτίζεται με το Pathfinding.delta_cost.
#   C2: ο D* Lite (full-field, όπως στα σενάρια) από την αρχή δίνει το ίδιο πεδίο κόστους με έναν ανεξάρτητο Dijkstra.
#   C3: ο D* Lite μετά από ΑΥΞΗΤΙΚΕΣ αλλαγές CM δίνει το ίδιο πεδίο κόστους με τον ανεξάρτητο Dijkstra.
#   C4: ο A* (Agents.jl) δίνει διαδρομές με το βέλτιστο κόστος.
#   C5: οι διαδρομές που εξάγει ο D* Lite έχουν το βέλτιστο κόστος.
#   Πληροφοριακά: #cells που αλλάζουν σε κάθε αλλαγή CM, χρόνος update έναντι rebuild, και πόσο συχνά η επιλογή εξόδου με
#   «λιγότερα cells» (choose_best_path / plan_best_route!) διαφέρει από την επιλογή εξόδου με το μικρότερο ΚΟΣΤΟΣ.
# Τρέξιμο (από τη ρίζα του repo):  julia --project=evac3 tests/verify_pathfinding_equivalence.jl
using Agents
using FileIO: load
using Images
using ImageMagick
using DataStructures: BinaryMinHeap
using Printf, Random, Statistics

const PF = Agents.Pathfinding                                                            # το module με τους pathfinders
const REPO = normpath(joinpath(@__DIR__, ".."))                                          # ρίζα του repo
const NUM_CMS = 7                                                                        # πλήθος concentration maps (όπως στο σενάριο)
const DESTS = [(500.0, 854.0), (120.0, 248.0)]                                           # οι δύο έξοδοι του σεναρίου (x, y)
const NSTARTS = 12                                                                       # πλήθος τυχαίων σημείων εκκίνησης ανά χάρτη
const RESULTS = Tuple{String,Bool}[]
check(name, ok) = (push!(RESULTS, (name, ok)); println(ok ? "  [PASS] " : "  [FAIL] ", name); flush(stdout))

# Φόρτωση εικόνας ως πίνακα (ίδια διαδικασία με τα σενάρια: 1ο κανάλι, κλίμακα 0..255).
function load_gray(path)
    img = load(path)
    img = permutedims(channelview(img), [2, 3, 1])[:, :, 1]
    return convert.(Float64, img) .* 255.0
end

# Φόρτωση heightmap και των 7 CMs όπως στο σενάριο· επιστρέφει (walkmap, λίστα χαρτών κόστους NPM_int για κάθε CM).
function load_maps()
    hm = floor.(Int, load_gray(joinpath(REPO, "NADEEN", "Maps", "Qatargas Map.jpg")))    # heightmap (0..255)
    hm = 255 .- hm                                                                       # αντιστροφή, όπως στο σενάριο
    cms = [load_gray(joinpath(REPO, "Concentration Maps", string(k) * ".bmp")) for k in 1:NUM_CMS]
    for k in 1:NUM_CMS
        @assert size(cms[k]) == size(hm) "Το CM $k έχει διαφορετικές διαστάσεις από το heightmap"
    end
    walk = BitArray(trues(size(hm)...))
    walk[hm .> 245] .= false                                                             # λευκές περιοχές = μη βατές (όπως στο σενάριο)
    return walk, [round.(Int, hm .+ cm) for cm in cms]
end

# Cells (i, j) στα οποία διαφέρουν δύο χάρτες κόστους (ίδια με find_changed_cells του σεναρίου).
function find_changed_cells(old::AbstractMatrix{Int}, new::AbstractMatrix{Int})
    cells = Tuple{Int,Int}[]
    @inbounds for i in axes(old, 1), j in axes(old, 2)
        old[i, j] != new[i, j] && push!(cells, (i, j))
    end
    return cells
end

# Μετατροπή θέσης εξόδου σε cell (ίδια με world_to_cell του σεναρίου: floor + clamp).
cell_of(p, dims) = (clamp(Int(floor(p[1])), 1, dims[1]), clamp(Int(floor(p[2])), 1, dims[2]))

# Κόστος ακμής όπως ορίζει ο AbsolutePenaltyMap με MaxDistance: 1 (γειτονιά Moore) + |p(u)| + |p(v)|.
edge_cost(pmap, u, v) = max(abs(u[1] - v[1]), abs(u[2] - v[2])) + abs(pmap[u...]) + abs(pmap[v...])

# Συνολικό κόστος μιας διαδρομής (λίστα από cells, με ή χωρίς το αρχικό cell).
function path_cost(pmap, from, path)
    c = 0
    prev = from
    for v in path
        v == prev && continue                                                            # το αρχικό cell μπορεί να εμφανίζεται στη λίστα
        c += edge_cost(pmap, prev, v)
        prev = v
    end
    return c
end

# Ανεξάρτητος Dijkstra από τον στόχο προς όλα τα cells (συμμετρικό κόστος ακμής) → πεδίο κόστους-μέχρι-τον-στόχο.
function reference_dijkstra(pmap::Matrix{Int}, walk::BitMatrix, goal::Tuple{Int,Int})
    h, w = size(pmap)
    dist = fill(typemax(Int), h, w)
    dist[goal...] = 0
    heap = BinaryMinHeap{Tuple{Int,Int,Int}}()                                           # (απόσταση, i, j)
    push!(heap, (0, goal[1], goal[2]))
    while !isempty(heap)
        d, i, j = pop!(heap)
        d > dist[i, j] && continue                                                       # παλιά εγγραφή (lazy deletion)
        for di in -1:1, dj in -1:1
            (di == 0 && dj == 0) && continue
            ni, nj = i + di, j + dj
            (1 <= ni <= h && 1 <= nj <= w && walk[ni, nj]) || continue
            nd = d + 1 + abs(pmap[i, j]) + abs(pmap[ni, nj])
            if nd < dist[ni, nj]
                dist[ni, nj] = nd
                push!(heap, (nd, ni, nj))
            end
        end
    end
    return dist
end

# Σύγκριση του πεδίου g ενός planner με το πεδίο αναφοράς: επιστρέφει (πλήθος αποκλίσεων, μέγιστη απόκλιση).
function compare_field(g::Matrix{Float64}, ref::Matrix{Int})
    nbad = 0
    maxdev = 0.0
    @inbounds for k in eachindex(ref)
        r = ref[k]
        gk = g[k]
        if r == typemax(Int)
            isinf(gk) || (nbad += 1)                                                     # μη προσβάσιμο cell: το g πρέπει να είναι Inf
        else
            dev = abs(gk - r)
            dev > 0 && (nbad += 1)
            maxdev = max(maxdev, isfinite(dev) ? dev : Inf)
        end
    end
    return nbad, maxdev
end

# --- Προθέρμανση (compile) σε μικρό χάρτη, ώστε οι χρόνοι να μην περιέχουν compilation ---
let
    rng = MersenneTwister(1)
    pm = rand(rng, 0:30, 40, 40)
    wk = BitArray(trues(40, 40))
    pf = PF.DStarLite((40, 40); walkmap = wk, cost_metric = PF.AbsolutePenaltyMap(pm, PF.MaxDistance{2}()))
    pl = PF.init_planner(pf, (5, 5))
    pm2 = copy(pm); pm2[10:20, 10:20] .+= 5
    pf.cost_metric.pmap .= pm2
    PF.update_after_cm_change!(pl, find_changed_cells(pm, pm2))
    PF.extract_path(pl, (30, 30))
    PF.find_path(PF.AStar((40, 40); walkmap = wk, cost_metric = PF.AbsolutePenaltyMap(copy(pm2), PF.MaxDistance{2}())), (30, 30), (5, 5))
    reference_dijkstra(pm2, wk, (5, 5))
end

println("Φόρτωση χαρτών ...")
walk, NPM = load_maps()
dims = size(walk)
exits = [cell_of(d, dims) for d in DESTS]                                                # οι έξοδοι ως cells
println("  διαστάσεις χάρτη = $dims · βατά cells = $(count(walk)) · έξοδοι (cells) = $exits")
for e in exits
    @assert walk[e...] "Η έξοδος $e δεν είναι βατή"
end

# Τυχαία σημεία εκκίνησης (κεντρική περιοχή 1/4..3/4, όπως στο σενάριο), μόνο βατά cells.
rng = MersenneTwister(123)
starts = Tuple{Int,Int}[]
while length(starts) < NSTARTS
    i = rand(rng, (dims[1] ÷ 4):(3 * dims[1] ÷ 4))
    j = rand(rng, (dims[2] ÷ 4):(3 * dims[2] ÷ 4))
    walk[i, j] && push!(starts, (i, j))
end

# --- C1: κόστος ακμής έναντι Pathfinding.delta_cost ---
println("\n== C1: κόστος ακμής (δικός μου τύπος έναντι Pathfinding.delta_cost) ==")
pmap_live = copy(NPM[1])                                                                 # ο «ζωντανός» χάρτης κόστους του D* Lite (αλλάζει in-place όπως στο σενάριο)
pf_d = PF.DStarLite(dims; walkmap = walk, cost_metric = PF.AbsolutePenaltyMap(pmap_live, PF.MaxDistance{2}()))
let
    bad = 0
    for _ in 1:2000
        u = (rand(rng, 2:dims[1]-1), rand(rng, 2:dims[2]-1))
        v = (u[1] + rand(rng, -1:1), u[2] + rand(rng, -1:1))
        v == u && continue
        edge_cost(pmap_live, u, v) == PF.delta_cost(pf_d, u, v) || (bad += 1)
    end
    check("C1 το κόστος ακμής ταυτίζεται με το Pathfinding.delta_cost (2000 τυχαίες ακμές, αποκλίσεις = $bad)", bad == 0)
end

# --- Κύριο πείραμα: για κάθε CM, ανεξάρτητος Dijkstra, D* Lite (initial/incremental/rebuild), A* ---
println("\n== Κύριο πείραμα (7 CMs × 2 έξοδοι) ==")
@printf("  %3s %5s %10s %9s %10s %10s %8s %9s %9s\n", "CM", "έξοδ", "αλλαγ.cells", "%βατών", "update(s)", "rebuild(s)", "upd/reb", "g-αποκλ.", "rebuild≟ref")
planners = Any[]                                                                         # ένας planner ανά έξοδο
init_times = Float64[]
upd_times = Float64[]
reb_times = Float64[]
c2_bad = 0                                                                               # αποκλίσεις στον πρώτο χάρτη (C2)
c3_bad = 0                                                                               # αποκλίσεις μετά από αυξητικές αλλαγές (C3)
c4_subopt = 0                                                                            # πλήθος μη βέλτιστων διαδρομών A* (C4)
c5_subopt = 0                                                                            # πλήθος μη βέλτιστων διαδρομών D* Lite (C5)
n_paths = 0                                                                              # πλήθος συγκρίσεων διαδρομών
astar_times = Float64[]
exit_mismatch = 0                                                                        # επιλογή εξόδου με λιγότερα cells ≠ επιλογή με μικρότερο κόστος
exit_excess = Float64[]                                                                  # σχετική επιβάρυνση κόστους όταν υπάρχει ασυμφωνία
n_exitchoice = 0
excess_rel_astar = Float64[]                                                             # σχετική επιβάρυνση κόστους των μη βέλτιστων διαδρομών A*

for (k, npm) in enumerate(NPM)
    refs = [reference_dijkstra(npm, walk, e) for e in exits]                             # πεδία αναφοράς για τις δύο εξόδους
    if k == 1
        pmap_live .= npm                                                                 # αρχικός χάρτης
        for e in exits
            t = @elapsed pl = PF.init_planner(pf_d, e)                                   # αρχικός σχεδιασμός (full-field)
            push!(planners, pl); push!(init_times, t)
        end
        for (ei, pl) in enumerate(planners)
            nbad, md = compare_field(pl.g, refs[ei])
            global c2_bad += nbad
            @printf("  %3d %5d %10s %9s %10.2f %10s %8s %9d %9s\n", k, ei, "-", "-", init_times[ei], "-", "-", nbad, "-")
        end
    else
        prev = copy(pmap_live)                                                           # χάρτης πριν την αλλαγή
        pmap_live .= npm                                                                 # in-place αλλαγή, όπως στο σενάριο
        changed = find_changed_cells(prev, npm)
        for (ei, pl) in enumerate(planners)
            t_upd = @elapsed PF.update_after_cm_change!(pl, changed)                     # αυξητικό replanning
            # rebuild από την αρχή πάνω στον νέο χάρτη (ίδιο κόστος με την init_planner)
            pf_new = PF.DStarLite(dims; walkmap = walk, cost_metric = PF.AbsolutePenaltyMap(copy(npm), PF.MaxDistance{2}()))
            local pl_new
            t_reb = @elapsed pl_new = PF.init_planner(pf_new, exits[ei])
            nbad, md = compare_field(pl.g, refs[ei])
            nbad_reb, _ = compare_field(pl_new.g, refs[ei])
            global c3_bad += nbad
            push!(upd_times, t_upd); push!(reb_times, t_reb)
            @printf("  %3d %5d %10d %8.1f%% %10.2f %10.2f %8.2f %9d %9s\n", k, ei, length(changed), 100 * length(changed) / count(walk), t_upd, t_reb, t_upd / t_reb, nbad, nbad_reb == 0 ? "ναι" : "ΟΧΙ")
        end
    end

    # --- διαδρομές: A* και D* Lite έναντι βέλτιστου κόστους ---
    pf_a = PF.AStar(dims; walkmap = walk, cost_metric = PF.AbsolutePenaltyMap(copy(npm), PF.MaxDistance{2}()))
    for s in starts
        lens = Int[]                                                                     # πλήθος cells της διαδρομής προς κάθε έξοδο (όπως στο choose_best_path)
        costs = Int[]                                                                    # κόστος της βέλτιστης διαδρομής προς κάθε έξοδο (από τον Dijkstra)
        for (ei, e) in enumerate(exits)
            ref_cost = refs[ei][s...]
            ref_cost == typemax(Int) && continue
            global n_paths += 1
            ta = @elapsed pa = PF.find_path(pf_a, s, e)                                  # A* (Agents.jl)
            push!(astar_times, ta)
            if pa !== nothing
                ca = path_cost(npm, s, collect(pa))
                if ca != ref_cost
                    global c4_subopt += 1
                    push!(excess_rel_astar, (ca - ref_cost) / ref_cost)
                end
            end
            pd = PF.extract_path(planners[ei], s)                                        # D* Lite: διαδρομή από το πεδίο g
            cd = path_cost(pmap_live, s, pd)                                             # κόστος της διαδρομής του D* Lite στον τρέχοντα χάρτη
            cd != ref_cost && (global c5_subopt += 1)
            push!(lens, length(pd)); push!(costs, ref_cost)
        end
        if length(lens) == 2
            global n_exitchoice += 1
            by_len = argmin(lens)                                                        # επιλογή όπως στο choose_best_path / plan_best_route!
            by_cost = argmin(costs)                                                      # επιλογή με το μικρότερο κόστος (TL-aware)
            if costs[by_len] != costs[by_cost]
                global exit_mismatch += 1
                push!(exit_excess, (costs[by_len] - costs[by_cost]) / costs[by_cost])
            end
        end
    end
end

println("\n== Αποτελέσματα ==")
check("C2 ο D* Lite (initial) ταυτίζεται με τον ανεξάρτητο Dijkstra (αποκλίσεις cells = $c2_bad)", c2_bad == 0)
check("C3 ο D* Lite (incremental) ταυτίζεται με τον ανεξάρτητο Dijkstra μετά από 6 αλλαγές CM × 2 έξοδοι (αποκλίσεις cells = $c3_bad)", c3_bad == 0)
check("C4 ο A* βρίσκει βέλτιστο κόστος ($(n_paths - c4_subopt)/$n_paths διαδρομές βέλτιστες)", c4_subopt == 0)
check("C5 οι διαδρομές του D* Lite έχουν βέλτιστο κόστος ($(n_paths - c5_subopt)/$n_paths διαδρομές βέλτιστες)", c5_subopt == 0)
!isempty(excess_rel_astar) && @printf("  A*: μη βέλτιστες διαδρομές με μέση σχετική επιβάρυνση κόστους %.2f%% (μέγιστη %.2f%%)\n", 100 * mean(excess_rel_astar), 100 * maximum(excess_rel_astar))

println("\n== Πληροφοριακά μεγέθη ==")
@printf("  αρχικός σχεδιασμός D* Lite (ανά έξοδο): μέσος %.1f s\n", mean(init_times))
@printf("  incremental update (ανά έξοδο και αλλαγή CM): μέσος %.1f s · rebuild από την αρχή: μέσος %.1f s · λόγος update/rebuild: %.2f\n", mean(upd_times), mean(reb_times), mean(upd_times) / mean(reb_times))
@printf("  A* ανά (agent, έξοδο): μέσος %.2f s\n", mean(astar_times))
@printf("  επιλογή εξόδου με λιγότερα cells αντί για μικρότερο κόστος: διαφέρει σε %d/%d περιπτώσεις (%.0f%%)", exit_mismatch, n_exitchoice, 100 * exit_mismatch / max(n_exitchoice, 1))
println(isempty(exit_excess) ? "" : @sprintf(" · μέση επιβάρυνση κόστους όταν διαφέρει: %.0f%%", 100 * mean(exit_excess)))

nfail = count(!last, RESULTS)
println("\n== Σύνοψη ==\n  $(length(RESULTS) - nfail)/$(length(RESULTS)) έλεγχοι πέρασαν")
nfail > 0 && exit(1)
