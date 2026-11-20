# ==============================================================================
# Incremental LPA* (Lifelong Planning A*) — optimized version (Koenig, Likhachev
# & Furcy, "Lifelong Planning A*", preprint submitted to Elsevier Science, 24 May
# 2005, Figure 7, see LITERATURE/LPAStar.pdf). Comments `{NN'}` refer to its lines.
# Το Figure 6 του paper είναι η βασική εκδοχή· το Figure 7 δεν αλλάζει ποιες κορυφές
# επεκτείνονται ούτε με ποια σειρά (Section 6), και είναι η εκδοχή που ακολουθεί και ο
# D* Lite του repo (dstar_lite_planner.jl).
#
# ΤΙ ΕΙΝΑΙ Ο LPA* (περίληψη του paper)
# Αυξητική (incremental) εκδοχή του A* για σταθερή αρχή s_start και στόχο s_goal, σε γράφο του
# οποίου τα κόστη ακμών (ή η βατότητα των κορυφών) αλλάζουν μεταξύ των αναζητήσεων. Αντί να
# ξαναλύνει το πρόβλημα από την αρχή μετά από κάθε αλλαγή (όπως ο A*), ξαναχρησιμοποιεί τις
# τιμές g της προηγούμενης αναζήτησης και ξαναϋπολογίζει ΜΟΝΟ τις κορυφές που επηρεάστηκαν.
#
# Βασικές ιδέες:
#  - Αναζητά ΕΥΘΕΙΑ, από την s_start προς την s_goal (αντίθετα με το D* Lite).
#  - g(s):   η τρέχουσα εκτίμηση της απόστασης s_start → s (πίνακας `pl.g`).
#  - rhs(s): η τιμή «ματιά ενός βήματος πίσω»: 0 για την s_start, αλλιώς
#            min_{s'∈pred(s)} (g(s') + c(s',s)) (πίνακας `pl.rhs`, Invariant 1, εξ. (2)).
#  - p(s):   ο προκάτοχος που πετυχαίνει το rhs(s) (πίνακας `pl.p`)· rhs(s) = g(p(s)) + c(p(s),s).
#  - Τοπική συνέπεια: g(s) = rhs(s). Αν g > rhs η κορυφή είναι «υπερσυνεπής»
#    (overconsistent, βρέθηκε φθηνότερος δρόμος), αν g < rhs «υποσυνεπής»
#    (underconsistent, ο παλιός δρόμος έγινε ακριβότερος/μπλοκαρίστηκε).
#  - Η ουρά προτεραιότητας U περιέχει ΜΟΝΟ τις τοπικά ασυνεπείς κορυφές (Invariant 2), με κλειδί
#    [min(g,rhs) + h(s) ; min(g,rhs)] (λεξικογραφική σειρά). Η h(s) εκτιμά την απόσταση s → s_goal.
#  - Όταν αλλάξουν κόστη ακμών (u,v): το rhs(v) διορθώνεται με βάση το c_old / c_new {32'-44'} και
#    ξανατρέχει το ComputeShortestPath.
#  - Τερματισμός {11'}: U.TopKey() ≥ CalculateKey(s_goal) και rhs(s_goal) ≤ g(s_goal). Η s_goal μπορεί
#    να μείνει υπερσυνεπής (g(s_goal) = ∞ ενώ rhs(s_goal) < ∞)· τότε η απόσταση του στόχου είναι το
#    rhs(s_goal) (Section 6), και η ανακατασκευή της διαδρομής δεν χρειάζεται το g(s_goal).
#
# ΑΝΤΙΣΤΟΙΧΙΑ ΚΩΔΙΚΑ ↔ PAPER (Figure 7)
#   calc_key                 {01'}         CalculateKey(s)
#   init_planner             {02'-07'}     Initialize()  (+ {28'}, {30'} από το Main)
#   update_vertex!           {08'-10'}     UpdateVertex(u)
#   compute_shortest_path!   {11'-27'}     ComputeShortestPath()
#   update_after_cm_change!  {31'-44'}     αλλαγή κόστους ακμών (η «Main» επανάληψη)
#   extract_path             —             ανακατασκευή διαδρομής προς τα πίσω από την s_goal (Theorem 5)
#
# ΑΝΤΙΣΤΟΙΧΙΑ ΔΟΜΩΝ
#   S (κορυφές)              cells του grid όπου `walkmap` == true
#   succ(s) / pred(s)        `_neighbors(pl, s)` (γειτονιά Moore/von Neumann· είναι
#                            συμμετρική, άρα pred(s) = succ(s))
#   c(u,v)                   `_cost(pl, u, v)`: `delta_cost` (cost_metric του LPAStar), ή ∞ αν κάποιο από
#                            τα δύο cells δεν είναι βατό (στο paper οι μη βατές κορυφές είναι ακμές κόστους ∞)
#   h(s)                     `heuristic(pl, s)`
#   U                        `CellHeap` (reuse από το dstar_lite_planner.jl)
#   g, rhs, p, s_start, s_goal   πεδία `pl.g`, `pl.rhs`, `pl.p`, `pl.start`, `pl.goal`
# ==============================================================================

# p(s) = NULL {05'}
const LPA_NULL = (0, 0)

# Η κατάσταση του αλγορίθμου LPA* για ΕΝΑ ζεύγος (s_start, s_goal).
mutable struct LPAStarPlanner{PF<:LPAStar}
    pf::PF                          # γράφος: walkmap (S), γειτονιά (succ/pred), cost_metric (c)
    g::Matrix{Float64}              # g(s) για κάθε cell
    rhs::Matrix{Float64}            # rhs(s) για κάθε cell
    p::Matrix{Tuple{Int,Int}}       # p(s): ο προκάτοχος που δίνει το rhs(s) (`LPA_NULL` = NULL)
    U::CellHeap                     # ουρά προτεραιότητας U με τις τοπικά ασυνεπείς κορυφές
    start::Tuple{Int,Int}           # s_start
    goal::Tuple{Int,Int}            # s_goal
    heuristic_kind::Symbol          # ποια ευρετική h χρησιμοποιείται (:delta_cost, :manhattan, :euclidean, :none)
    # Penalty map και walkmap από τα οποία διαβάστηκαν τελευταία τα κόστη των ακμών: δίνουν το c_old στο
    # {33'} όταν ο καλών έχει ήδη γράψει τις νέες τιμές στο ζωντανό penalty map / walkmap.
    known_pmap::Union{Nothing,Matrix{Int}}
    known_walk::BitMatrix
end

# Η ευρετική h(from, to): εκτίμηση του κόστους από το `from` στο `to`, που δεν πρέπει
# ποτέ να το υπερεκτιμά (admissible) και να ικανοποιεί την τριγωνική ανισότητα (consistent).
@inline function heuristic_cost(
    pl::LPAStarPlanner,
    from::Tuple{Int,Int},
    to::Tuple{Int,Int},
)
    hk = pl.heuristic_kind
    if hk === :delta_cost
        # απόσταση με το βασικό metric (χωρίς penalties), π.χ. MaxDistance
        return Float64(delta_cost(pl.pf, _heuristic_metric(pl.pf.cost_metric), from, to))
    elseif hk === :manhattan
        d = position_delta(pl.pf, from, to)
        return Float64(d[1] + d[2])
    elseif hk === :euclidean
        d = position_delta(pl.pf, from, to)
        return sqrt(Float64(d[1]^2 + d[2]^2))
    else
        return 0.0      # :none → h ≡ 0 (ο LPA* γίνεται «αυξητικός Dijkstra»)
    end
end

# h(s) του paper: εκτίμηση της απόστασης s → s_goal
@inline heuristic(pl::LPAStarPlanner, s::Tuple{Int,Int}) = heuristic_cost(pl, s, pl.goal)

# CalculateKey(s) {01'}: [min(g(s), rhs(s)) + h(s) ; min(g(s), rhs(s))]
@inline function calc_key(pl::LPAStarPlanner, s::Tuple{Int,Int})
    m = min(pl.g[s...], pl.rhs[s...])           # min(g(s), rhs(s))
    return (m + heuristic(pl, s), m)            # (k1, k2)
end

# c(u,v): το κόστος της ακμής (u,v). Ακμή από/προς μη βατό cell έχει κόστος ∞ (0 < c ≤ ∞ στο paper).
@inline _cost(pl::LPAStarPlanner, u::Tuple{Int,Int}, v::Tuple{Int,Int}) =
    (pl.pf.walkmap[u...] && pl.pf.walkmap[v...]) ? Float64(delta_cost(pl.pf, u, v)) : Inf

# c_old: το κόστος της ακμής όπως το γνώριζε ο planner πριν την αλλαγή ({33'}), από τα αντίγραφα known_*.
@inline function _known_cost(pl::LPAStarPlanner, u::Tuple{Int,Int}, v::Tuple{Int,Int})
    (pl.known_walk[u...] && pl.known_walk[v...]) || return Inf
    return pl.known_pmap === nothing ? Float64(delta_cost(pl.pf, u, v)) :
           Float64(_edge_cost(pl.pf, pl.pf.cost_metric, pl.known_pmap, u, v))
end

# argmin_{s' ∈ pred(u)} (g(s') + c(s',u)) και η τιμή του (χρησιμοποιείται στα {25'-26'}, {42'-43'}).
# Αν κανένας προκάτοχος δεν δίνει πεπερασμένη τιμή επιστρέφει (NULL, ∞).
@inline function _argmin_pred(pl::LPAStarPlanner, u::Tuple{Int,Int})
    best = LPA_NULL
    best_val = Inf
    for s in _neighbors(pl, u)                  # pred(u): συμμετρική γειτονιά
        val = pl.g[s...] + _cost(pl, s, u)
        if val < best_val
            best_val = val
            best = s
        end
    end
    return best, best_val
end

# p(v) = argmin_{s'∈pred(v)} (g(s') + c(s',v)); rhs(v) = g(p(v)) + c(p(v),v)  ({25'-26'}, {42'-43'})
@inline function _reset_parent!(pl::LPAStarPlanner, v::Tuple{Int,Int})
    pl.p[v...], pl.rhs[v...] = _argmin_pred(pl, v)
    return nothing
end

# UpdateVertex(u) {08'-10'}: μόνο συγχρονίζει την ιδιότητα μέλους/κλειδί του u στην U με τα g/rhs· το rhs
# ορίζεται στα σημεία κλήσης. Η U.Update γίνεται επί τόπου από το heap.
@inline function update_vertex!(pl::LPAStarPlanner, u::Tuple{Int,Int})
    if pl.g[u...] != pl.rhs[u...]
        pl.U[u] = calc_key(pl, u)                 # {08'} U.Update αν u ∈ U / {09'} U.Insert αν u ∉ U
    else
        delete!(pl.U, u)                          # {10'} U.Remove(u) (no-op αν u ∉ U)
    end
    return nothing
end

# ComputeShortestPath {11'-27'}. U.TopKey() μιας κενής U είναι [∞; ∞].
# Επεξεργάζεται την U μέχρι η s_goal να μην είναι υποσυνεπής και το κλειδί της να μην ξεπερνά την κορυφή της U.
function compute_shortest_path!(pl::LPAStarPlanner)
    goal = pl.goal
    while !isempty(pl.U)
        # {11'} συνθήκη while: U.TopKey() < CalculateKey(s_goal) Ή rhs(s_goal) > g(s_goal)
        (_key_lt(heap_top(pl.U)[2], calc_key(pl, goal)) || pl.rhs[goal...] > pl.g[goal...]) || break
        u = heap_top(pl.U)[1]                                               # {12'} u = U.Top()
        if pl.g[u...] > pl.rhs[u...]                                        # {13'} υπερσυνεπής: βρέθηκε φθηνότερος δρόμος
            pl.g[u...] = pl.rhs[u...]                                       # {14'} g(u) = rhs(u)
            delete!(pl.U, u)                                                # {15'} U.Remove(u)
            for s in _neighbors(pl, u)                                      # {16'} succ(u)
                c = pl.g[u...] + _cost(pl, u, s)
                if pl.rhs[s...] > c                                         # {17'} rhs(s) > g(u) + c(u,s)
                    pl.p[s...] = u                                          # {18'} p(s) = u
                    pl.rhs[s...] = c                                        # {19'} rhs(s) = g(u) + c(u,s)
                    update_vertex!(pl, s)                                   # {20'}
                end
            end
        else                                                                # {21'} υποσυνεπής: το g(u) ήταν υπερβολικά χαμηλό
            pl.g[u...] = Inf                                                # {22'} g(u) = ∞
            for s in _neighbors(pl, u)                                      # {23'} succ(u)
                if s != pl.start && pl.p[s...] == u                         # {24'} το rhs(s) στηριζόταν στο u
                    _reset_parent!(pl, s)                                   # {25'-26'} p(s) = argmin…, rhs(s) = g(p(s)) + c(p(s),s)
                end
                update_vertex!(pl, s)                                       # {27'}
            end
            # {23'} περιλαμβάνει και το s = u: δεν υπάρχει ακμή (u,u), άρα p(u) ≠ u και το {24'} δεν ισχύει·
            # το rhs(u) δεν εξαρτάται από το g(u), αλλάζει μόνο η U.
            update_vertex!(pl, u)                                           # {27'}
        end
    end
    return pl
end

# Η ευρετική είναι επιτρεπτή (admissible) όταν h(s,s') ≤ c(s,s') για κάθε κίνηση ενός βήματος. Δεν ανήκει στο
# Figure 7, αλλά είναι η προϋπόθεση του paper (consistent h, Section 4) ώστε ο LPA* να βρίσκει βέλτιστη διαδρομή.
function _check_heuristic(pl::LPAStarPlanner)
    pl.heuristic_kind === :none && return nothing           # h ≡ 0 είναι πάντα επιτρεπτή
    pl.heuristic_kind in (:delta_cost, :manhattan, :euclidean) ||
        throw(ArgumentError("unknown heuristic :$(pl.heuristic_kind)"))
    base = _heuristic_metric(pl.pf.cost_metric)             # βασικό metric χωρίς penalties
    u = pl.goal
    for β in pl.pf.neighborhood                             # κάθε δυνατή κίνηση ενός βήματος
        v = (u[1] + β[1], u[2] + β[2])
        h, c = heuristic_cost(pl, u, v), delta_cost(pl.pf, base, u, v)
        h <= c || throw(ArgumentError(                      # απαιτείται h(s,s') ≤ c(s,s')
            "heuristic :$(pl.heuristic_kind) is not admissible for " *
            "$(pl.pf.cost_metric): h = $h > c = $c for the move $(β.I); " *
            "LPA* requires h(s,s') ≤ c*(s,s')"))
    end
    return nothing
end

"""
    init_planner(pf::LPAStar{2}, start, goal; heuristic = :delta_cost)

Build an LPA* planner from `start` to `goal`. Η σειρά ορισμάτων είναι `(start, goal)`, όπως στο paper —
αντίθετα με το `init_planner(::DStarLite, goal, start)`.

Runs Initialize() and the first ComputeShortestPath() ({28'}, {30'}). `heuristic` is `h(s)`, the
estimate of the distance `s → goal`:

- `:delta_cost`: base-metric distance (penalty maps only add non-negative terms, so they are left out)
- `:manhattan` / `:euclidean`: grid distance from `s` to `goal`
- `:none`: `h = 0`

The distance `start → goal` is `planner.rhs[goal...]`: under the optimized termination {11'} the goal may
stay overconsistent (`g(goal) = ∞`) although `rhs(goal)` is finite (paper, Section 6).

An `ArgumentError` is thrown if the heuristic is not admissible for the cost metric, or if `start`
or `goal` is out of bounds or not walkable.
"""
function init_planner(
    pf::LPAStar{2},
    start::Tuple{Int,Int},
    goal::Tuple{Int,Int};
    heuristic::Symbol = :delta_cost,
)
    sz = size(pf.walkmap)
    for s in (start, goal)                                                  # αρχή και στόχος εντός ορίων και βατά
        (1 <= s[1] <= sz[1] && 1 <= s[2] <= sz[2] && pf.walkmap[s...]) ||
            throw(ArgumentError("cell $s is out of bounds or not walkable"))
    end
    U = CellHeap(sz)                                                        # {02'} U = ∅
    pl = LPAStarPlanner(pf, fill(Inf, sz), fill(Inf, sz), fill(LPA_NULL, sz), U,   # {03'-05'} g = rhs = ∞, p = NULL για όλα τα s
        start, goal, heuristic, _known_pmap(pf.cost_metric), copy(pf.walkmap))
    _check_heuristic(pl)
    pl.rhs[start...] = 0.0                                                  # {06'} rhs(s_start) = 0
    U[start] = calc_key(pl, start)                                          # {07'} U.Insert(s_start, [h(s_start); 0])
    compute_shortest_path!(pl)                                              # {30'} (το {28'} Initialize() είναι οι παραπάνω γραμμές)
    return pl
end

# Γείτονες του x στη γειτονιά του pathfinder, εντός ορίων αλλά ΑΝΕΞΑΡΤΗΤΑ από το αν είναι βατοί:
# οι ακμές προς/από μη βατά cells αλλάζουν κόστος (∞ ↔ πεπερασμένο) όταν αλλάζει το walkmap.
function _topology_neighbors(pf::LPAStar, x::Tuple{Int,Int})
    s1, s2 = size(pf.walkmap)
    P = _periodicity(pf)
    p1, p2 = P isa Bool ? (P, P) : (P[1], P[2])                 # περιοδικότητα ανά άξονα
    res = Tuple{Int,Int}[]
    for β in pf.neighborhood
        n1 = p1 ? mod1(x[1] + β[1], s1) : x[1] + β[1]
        n2 = p2 ? mod1(x[2] + β[2], s2) : x[2] + β[2]
        1 <= n1 <= s1 && 1 <= n2 <= s2 && push!(res, (n1, n2))
    end
    return res
end

"""
    update_after_cm_change!(planner::LPAStarPlanner, changed_cells)

Handle edge-cost changes, lines {31'-44'} of the paper. The caller has already written the new
penalties into the pathfinder's penalty map (and/or flipped entries of its `walkmap`);
`changed_cells` are the cells whose penalty or walkability changed. `start` and `goal` must stay walkable.

1. for every directed edge (u,v) touching a changed cell whose cost changed {32'}: take `c_old` from
   the planner's copies of the previous penalties / walkmap {33'}, adopt the new cost {34'}, and repair
   `rhs(v)` exactly as in {35'-44'}. An edge to or from a blocked cell costs `∞`.
2. ComputeShortestPath() {30'}

Deviation from the paper: all edges adopt their new cost before the loop, because a cell's penalty
moves all its incident edges at once. {35'-44'} stay correct, since every changed edge is still
visited once with its own `c_old`.
"""
function update_after_cm_change!(
    pl::LPAStarPlanner,
    changed_cells::Vector{Tuple{Int,Int}},
)
    # {32'-33'}: οι κατευθυνόμενες ακμές που αγγίζουν αλλαγμένα cells, μαζί με το παλιό κόστος c_old
    seen = Set{NTuple{2,Tuple{Int,Int}}}()                  # αποφυγή διπλής επεξεργασίας της ίδιας ακμής
    edges = Tuple{Tuple{Int,Int},Tuple{Int,Int},Float64}[]  # (u, v, c_old)
    for x in changed_cells, n in _topology_neighbors(pl.pf, x), e in ((x, n), (n, x))   # και οι δύο κατευθύνσεις
        e in seen && continue
        push!(seen, e)
        push!(edges, (e[1], e[2], _known_cost(pl, e[1], e[2])))     # {33'} c_old = c(u,v)
    end

    # {34'}: adopt the new costs — ενημέρωση των αντιγράφων known_* με τις νέες τιμές
    for x in changed_cells
        pl.known_walk[x...] = pl.pf.walkmap[x...]
        pl.known_pmap === nothing || (pl.known_pmap[x...] = pl.pf.cost_metric.pmap[x...])
    end

    # Επεξεργασία κάθε αλλαγμένης ακμής {35'-44'}: διορθώνει το rhs(v) και ενημερώνει την U
    for (u, v, c_old) in edges
        c_new = _cost(pl, u, v)                         # το νέο κόστος c(u,v)
        c_old == c_new && continue                      # {32'} μόνο ακμές με αλλαγμένο κόστος
        if c_old > c_new                                # {35'} η ακμή έγινε φθηνότερη
            c = pl.g[u...] + c_new
            if pl.rhs[v...] > c                         # {36'} rhs(v) > g(u) + c(u,v)
                pl.p[v...] = u                          # {37'} p(v) = u
                pl.rhs[v...] = c                        # {38'} rhs(v) = g(u) + c(u,v)
                update_vertex!(pl, v)                   # {39'}
            end
        elseif v != pl.start && pl.p[v...] == u         # {41'} η ακμή έγινε ακριβότερη και το rhs(v) στηριζόταν σε αυτήν
            _reset_parent!(pl, v)                       # {42'-43'} p(v) = argmin…, rhs(v) = g(p(v)) + c(p(v),v)
            update_vertex!(pl, v)                       # {44'}
        end
    end

    compute_shortest_path!(pl)                          # {30'} επανυπολογισμός μόνο όπου χρειάζεται
    return pl
end

"""
    extract_path(planner::LPAStarPlanner; max_steps = 10_000)

Walk backwards from `planner.goal`, always to a predecessor `s'` minimizing `g(s') + c(s',s)`, until
`planner.start` is reached (Theorem 5); returns the cells `[start, …, goal]`, a shortest path once
ComputeShortestPath() has terminated, which `init_planner` and `update_after_cm_change!` guarantee.
Returns `nothing` if there is no path (`rhs(goal) = ∞`).
"""
function extract_path(planner::LPAStarPlanner; max_steps = 10_000)
    planner.rhs[planner.goal...] == Inf && return nothing   # δεν υπάρχει γνωστή διαδρομή (Section 6: rhs, όχι g, του στόχου)
    path = Tuple{Int,Int}[planner.goal]
    cur = planner.goal
    for _ in 1:max_steps                      # όριο βημάτων για προστασία από ατέρμονο βρόχο
        cur == planner.start && break         # έφτασε στην αρχή
        best, _ = _argmin_pred(planner, cur)  # s' = argmin_{s'∈pred(cur)} g(s') + c(s',cur)
        best == LPA_NULL && return nothing
        cur = best
        push!(path, cur)                      # ο προηγούμενος κόμβος της διαδρομής
    end
    cur == planner.start || return nothing
    return reverse!(path)
end
