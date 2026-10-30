# ==============================================================================
# Incremental D* Lite — optimized version (Koenig & Likhachev, AAAI 2002,
# Figure 4, see LITERATURE/DStarLite.pdf). Comments `{NN''}` refer to its lines.
#
# Two modes share one struct:
# - Focused (`s_start` set): Figure 4 as written.
# - Full-field (`s_start === nothing`): the same procedures with h ≡ 0 and
#   CalculateKey(s_start) ≡ [∞; ∞], so {10''} runs until U is empty and `g` is
#   the cost-to-go of every cell (one planner per exit shared by all agents,
#   Scenario 3). This is the "D* Lite without heuristic search" of Figs. 6–8.
#
# ΤΙ ΕΙΝΑΙ ΤΟ D* LITE (περίληψη του paper)
# Αυξητικός (incremental) ευρετικός αλγόριθμος εύρεσης διαδρομής για agent που
# κινείται προς έναν στόχο σε γράφο του οποίου τα κόστη ακμών αλλάζουν (εδώ: οι
# συγκεντρώσεις τοξικού αλλάζουν όταν αλλάζει το concentration map). Αντί να
# ξαναλύνει όλο το πρόβλημα μετά από κάθε αλλαγή (όπως ο A*), ξαναχρησιμοποιεί
# την προηγούμενη αναζήτηση και ξαναϋπολογίζει ΜΟΝΟ τις κορυφές που επηρεάστηκαν.
#
# Βασικές ιδέες:
#  - Αναζητά ΑΝΑΠΟΔΑ, από τον στόχο s_goal προς την αρχή s_start. Έτσι οι τιμές
#    g που υπολογίστηκαν παραμένουν έγκυρες όταν ο agent αλλάζει θέση.
#  - g(s):   η τρέχουσα εκτίμηση του κόστους s → s_goal (πίνακας `pl.g`).
#  - rhs(s): η τιμή «ματιά ενός βήματος μπροστά»: 0 για τον στόχο, αλλιώς
#            min_{s'∈Succ(s)} (c(s,s') + g(s')) (πίνακας `pl.rhs`).
#  - Τοπική συνέπεια: g(s) = rhs(s). Αν g > rhs η κορυφή είναι «υπερσυνεπής»
#    (overconsistent, βρέθηκε φθηνότερος δρόμος), αν g < rhs «υποσυνεπής»
#    (underconsistent, ο παλιός δρόμος έγινε ακριβότερος/μπλοκαρίστηκε).
#  - Η ουρά προτεραιότητας U περιέχει ΜΟΝΟ τις τοπικά ασυνεπείς κορυφές, με
#    κλειδί [min(g,rhs) + h(s_start,s) + k_m ; min(g,rhs)] (λεξικογραφική σειρά).
#  - Το k_m (`pl.km`) αθροίζει τις μετακινήσεις του agent h(s_last,s_start), ώστε
#    τα κλειδιά που ήδη βρίσκονται στην U να μην χρειάζεται να ξαναϋπολογιστούν
#    όταν ο agent κινείται (αποφυγή «heap reordering»).
#  - ComputeShortestPath επεξεργάζεται την U με σειρά κλειδιού μέχρι η s_start
#    να γίνει τοπικά συνεπής με κλειδί ≤ της κορυφής της U. Τότε (Theorem 8) ο
#    agent ακολουθεί greedy argmin_{s'} c(s,s') + g(s') και έχει συντομότερη διαδρομή.
#  - Όταν αλλάξουν κόστη ακμών: ενημερώνεται k_m, διορθώνεται το rhs των
#    επηρεασμένων κορυφών και ξανατρέχει ComputeShortestPath.
#
# ΑΝΤΙΣΤΟΙΧΙΑ ΚΩΔΙΚΑ ↔ PAPER (Figure 4)
#   calc_key                 {01''}       CalculateKey(s)
#   _make_planner            {02''-06''}  Initialize()  (+ {29''}, {31''} από το Main)
#   update_vertex!           {07''-09''}  UpdateVertex(u)
#   compute_shortest_path!   {10''}       συνθήκη του while στο ComputeShortestPath
#   _process_top!            {11''-28''}  σώμα του while στο ComputeShortestPath
#   update_start!            {34''-35''}  ο agent μετακινείται στην επόμενη κορυφή
#   update_after_cm_change!  {36''-48''}  αλλαγή κόστους ακμών (η «Main» επανάληψη)
#   extract_path             {34''}       greedy argmin c(s,s') + g(s') (Theorem 8)
#
# ΑΝΤΙΣΤΟΙΧΙΑ ΔΟΜΩΝ
#   S (κορυφές)              cells του grid όπου `walkmap` == true
#   Succ(s) / Pred(s)        `_neighbors(pl, s)` (γειτονιά Moore/von Neumann· είναι
#                            συμμετρική, άρα Pred(s) = Succ(s))
#   c(u,v)                   `delta_cost(pl.pf, u, v)` (cost_metric του DStarLite)
#   h(s_start, s)            `heuristic(pl, s)`
#   U                        `CellHeap` (δυαδικό min-heap με ευρετήριο ανά cell)
#   g, rhs, k_m, s_last      πεδία `pl.g`, `pl.rhs`, `pl.km`, `pl.s_last`
#   s_goal                   `pl.goal`
# ==============================================================================

# s_start μπορεί να είναι `nothing` (full-field mode) ή cell (i, j) (focused mode)
const DStarStart = Union{Nothing,Tuple{Int,Int}}
# κλειδί προτεραιότητας [k1; k2] του paper ως tuple (k1, k2)
const DStarKey = Tuple{Float64,Float64}

# Λεξικογραφική σύγκριση κλειδιών: [a1;a2] < [b1;b2] ⇔ a1 < b1 ή (a1 = b1 και a2 < b2).
# Είναι η σύγκριση «<» που χρησιμοποιεί το paper στα {10''} και {14''} και στην ουρά U.
@inline function _key_lt(a::DStarKey, b::DStarKey)
    return (a[1] < b[1]) || (a[1] == b[1] && a[2] < b[2])
end

# ---------------- Indexed binary min-heap over grid cells ----------------
# Replaces DataStructures.PriorityQueue: `pos[i,j]` is the cell's slot in the
# heap (0 = absent), so membership / decrease-key need no hashing, and a key
# change is an in-place sift instead of delete + reinsert.
#
# Υλοποιεί την ουρά προτεραιότητας U του paper. Αντιστοιχία πράξεων:
#   U.Insert / U.Update  → `U[u] = key`  (Base.setindex!)
#   U.Remove(u)          → `delete!(U, u)`
#   u ∈ U                → `haskey(U, u)`
#   U.Top / U.TopKey     → `heap_top(U)`  (η κενή U έχει TopKey [∞; ∞])

struct CellHeap
    cells::Vector{Tuple{Int,Int}}   # τα cells του heap, στη σειρά του δυαδικού δέντρου
    keys::Vector{DStarKey}          # το κλειδί κάθε cell (ίδια θέση με το `cells`)
    pos::Matrix{Int32}              # θέση του cell (i,j) στο heap· 0 = δεν ανήκει στην U
end

# κενό heap για grid διαστάσεων `sz`
CellHeap(sz::Tuple{Int,Int}) = CellHeap(Tuple{Int,Int}[], DStarKey[], zeros(Int32, sz))

Base.isempty(h::CellHeap) = isempty(h.cells)
Base.length(h::CellHeap) = length(h.cells)
Base.haskey(h::CellHeap, u::Tuple{Int,Int}) = @inbounds h.pos[u[1], u[2]] != 0   # u ∈ U

"Smallest (cell, key) in the heap; the heap must not be empty."
@inline heap_top(h::CellHeap) = @inbounds (h.cells[1], h.keys[1])   # U.Top(), U.TopKey() {11''} {12''}

# ανταλλαγή δύο στοιχείων του heap, με ενημέρωση του ευρετηρίου `pos`
@inline function _heap_swap!(h::CellHeap, i::Int, j::Int)
    @inbounds begin
        ci, cj = h.cells[i], h.cells[j]
        h.cells[i], h.cells[j] = cj, ci
        h.keys[i], h.keys[j] = h.keys[j], h.keys[i]
        h.pos[cj[1], cj[2]] = i
        h.pos[ci[1], ci[2]] = j
    end
    return nothing
end

# ανεβάζει το στοιχείο i προς τη ρίζα όσο το κλειδί του είναι μικρότερο από του γονέα
@inline function _sift_up!(h::CellHeap, i::Int)
    @inbounds while i > 1
        p = i >> 1
        _key_lt(h.keys[i], h.keys[p]) || break
        _heap_swap!(h, i, p)
        i = p
    end
    return i
end

# κατεβάζει το στοιχείο i προς τα φύλλα όσο κάποιο παιδί έχει μικρότερο κλειδί
@inline function _sift_down!(h::CellHeap, i::Int)
    n = length(h.cells)
    @inbounds while true
        l = 2i
        l > n && break
        c = (l < n && _key_lt(h.keys[l+1], h.keys[l])) ? l + 1 : l   # το μικρότερο από τα δύο παιδιά
        _key_lt(h.keys[c], h.keys[i]) || break
        _heap_swap!(h, i, c)
        i = c
    end
    return i
end

"Insert `u` with key `k`, or change its key in place if already present."
function Base.setindex!(h::CellHeap, k::DStarKey, u::Tuple{Int,Int})   # U.Insert / U.Update
    i = Int(@inbounds h.pos[u[1], u[2]])
    if i == 0
        # δεν υπάρχει στην U → U.Insert: προσθήκη στο τέλος και sift-up
        push!(h.cells, u)
        push!(h.keys, k)
        i = length(h.cells)
        @inbounds h.pos[u[1], u[2]] = i
        _sift_up!(h, i)
    else
        # υπάρχει ήδη → U.Update: αλλαγή κλειδιού επί τόπου και sift προς την κατάλληλη κατεύθυνση
        @inbounds old = h.keys[i]
        @inbounds h.keys[i] = k
        _key_lt(k, old) ? _sift_up!(h, i) : _sift_down!(h, i)
    end
    return h
end

# αφαιρεί το στοιχείο στη θέση i: το ανταλλάσσει με το τελευταίο, το πετάει και επαναφέρει την ιδιότητα του heap
function _heap_remove_at!(h::CellHeap, i::Int)
    n = length(h.cells)
    i != n && _heap_swap!(h, i, n)
    u = pop!(h.cells)
    pop!(h.keys)
    @inbounds h.pos[u[1], u[2]] = 0
    if i < n
        _sift_up!(h, i) == i && _sift_down!(h, i)
    end
    return u
end

"Remove `u` if present (no-op otherwise)."
function Base.delete!(h::CellHeap, u::Tuple{Int,Int})   # U.Remove(u)
    i = Int(@inbounds h.pos[u[1], u[2]])
    i != 0 && _heap_remove_at!(h, i)
    return h
end

"Remove and return the smallest cell."
heap_pop!(h::CellHeap) = _heap_remove_at!(h, 1)

# ---------------- Allocation-free walkable neighbors ----------------
# Same cells, in the same order, as `walkable_neighbors(u, pf)` (so tie-breaks
# and extracted paths are unchanged), but yielded lazily instead of collected
# into a fresh Vector on every call.
#
# Υλοποιεί τα Succ(u) και Pred(u) του paper: τους βατούς γείτονες του u στη
# γειτονιά του pathfinder (`pf.neighborhood`). Η γειτονιά είναι συμμετρική, άρα
# Succ(u) = Pred(u)· τα κόστη c(s,u) και c(u,s) υπολογίζονται πάντα χωριστά.

struct WalkableNeighbors{PF<:DStarLite}
    pf::PF                  # ο pathfinder (walkmap, neighborhood, periodicity)
    u::Tuple{Int,Int}       # το cell του οποίου τους γείτονες διατρέχουμε
end

Base.IteratorSize(::Type{<:WalkableNeighbors}) = Base.SizeUnknown()
Base.eltype(::Type{<:WalkableNeighbors}) = Tuple{Int,Int}

_periodicity(::DStarLite{D,P}) where {D,P} = P

# επιστρέφει τον επόμενο βατό γείτονα (και το νέο k) ή `nothing` όταν τελειώσουν
@inline function Base.iterate(it::WalkableNeighbors, k::Int = 1)
    pf = it.pf
    nb = pf.neighborhood
    wm = pf.walkmap
    s1, s2 = size(wm)
    P = _periodicity(pf)
    p1, p2 = P isa Bool ? (P, P) : (P[1], P[2])        # περιοδικότητα ανά άξονα
    u1, u2 = it.u
    @inbounds while k <= length(nb)
        β = nb[k]                                       # k-οστή μετατόπιση της γειτονιάς
        k += 1
        n1 = p1 ? mod1(u1 + β[1], s1) : u1 + β[1]       # συντεταγμένες γείτονα (με wrap αν περιοδικό)
        n2 = p2 ? mod1(u2 + β[2], s2) : u2 + β[2]
        if 1 <= n1 <= s1 && 1 <= n2 <= s2 && wm[n1, n2]    # εντός ορίων και βατός
            return ((n1, n2), k)
        end
    end
    return nothing
end

# Succ(u) / Pred(u) του planner `pl`
@inline _neighbors(pl, u::Tuple{Int,Int}) = WalkableNeighbors(pl.pf, u)

# ---------------- Planner ----------------

# `PF` is the concrete pathfinder type so `delta_cost`/`walkable_neighbors`
# dispatch statically in the hot loop (an abstract `DStarLite{D}` field would not).
#
# Η κατάσταση του αλγορίθμου D* Lite για ΕΝΑΝ στόχο (μία έξοδο).
mutable struct DStarLitePlanner{PF<:DStarLite}
    pf::PF                          # γράφος: walkmap (S), γειτονιά (Succ/Pred), cost_metric (c)
    g::Matrix{Float64}              # g(s) για κάθε cell
    rhs::Matrix{Float64}            # rhs(s) για κάθε cell
    U::CellHeap                     # ουρά προτεραιότητας U με τις τοπικά ασυνεπείς κορυφές
    km::Float64                     # k_m: συσσωρευμένη μετακίνηση του agent {03''} {38''}
    goal::Tuple{Int,Int}            # s_goal
    s_start::DStarStart             # s_start (`nothing` στο full-field mode)
    s_last::DStarStart              # s_last: η s_start στην τελευταία αλλαγή κόστους {29''} {39''}
    heuristic_kind::Symbol          # ποια ευρετική h χρησιμοποιείται (:delta_cost, :manhattan, :euclidean, :none)
    # Penalty map the current edge costs were last read from: gives c_old in
    # {41''} when the caller has already overwritten the live penalty map.
    known_pmap::Union{Nothing,Matrix{Int}}
end

# Η ευρετική h(from, to): εκτίμηση του κόστους από το `from` στο `to`, που δεν πρέπει
# ποτέ να το υπερεκτιμά (admissible) και να ικανοποιεί την τριγωνική ανισότητα.
@inline function heuristic_cost(
    pl::DStarLitePlanner,
    from::Tuple{Int,Int},
    to::Tuple{Int,Int},
)
    hk = pl.heuristic_kind
    if hk === :delta_cost
        # απόσταση με το βασικό metric (χωρίς penalties), π.χ. MaxDistance — η ευρετική των πειραμάτων του paper
        return Float64(delta_cost(pl.pf, _heuristic_metric(pl.pf.cost_metric), from, to))
    elseif hk === :manhattan
        d = position_delta(pl.pf, from, to)
        return Float64(d[1] + d[2])
    elseif hk === :euclidean
        d = position_delta(pl.pf, from, to)
        return sqrt(Float64(d[1]^2 + d[2]^2))
    else
        return 0.0      # :none → h ≡ 0 (το D* Lite χωρίς ευρετική αναζήτηση)
    end
end

# h(s_start, s) του paper. Στο full-field mode (χωρίς s_start) είναι 0.
@inline function heuristic(pl::DStarLitePlanner, s::Tuple{Int,Int})
    pl.s_start === nothing && return 0.0
    return heuristic_cost(pl, pl.s_start, s)
end

# CalculateKey(s) {01''}: [min(g(s), rhs(s)) + h(s_start, s) + k_m ; min(g(s), rhs(s))]
@inline function calc_key(pl::DStarLitePlanner, s::Tuple{Int,Int})
    m = min(pl.g[s...], pl.rhs[s...])           # min(g(s), rhs(s))
    return (m + heuristic(pl, s) + pl.km, m)    # (k1, k2)
end

# min_{s' ∈ Succ(u)} (c(u,s') + g(s')), as in {27''} / {46''}.
# Επαναυπολογισμός του rhs(u) από την αρχή: το καλύτερο κόστος ενός βήματος + g του γείτονα.
@inline function _min_succ(pl::DStarLitePlanner, u::Tuple{Int,Int})
    m = Inf
    for v in _neighbors(pl, u)
        m = min(m, delta_cost(pl.pf, u, v) + pl.g[v...])
    end
    return m
end

# UpdateVertex {07''-09''}: only syncs u's queue membership/key with g/rhs; the
# optimized version sets rhs at the call sites. The heap applies U.Update in place.
# Στο Figure 4 το rhs ενημερώνεται έξω από την UpdateVertex (στα σημεία κλήσης), οπότε εδώ
# η συνάρτηση απλώς κρατά την U συνεπή: ασυνεπής κορυφή (g ≠ rhs) → στην U με νέο κλειδί,
# συνεπής (g = rhs) → αφαιρείται από την U.
@inline function update_vertex!(pl::DStarLitePlanner, u::Tuple{Int,Int})
    if pl.g[u...] != pl.rhs[u...]
        pl.U[u] = calc_key(pl, u)                 # {07''} / {08''} U.Update αν u ∈ U, αλλιώς U.Insert
    else
        delete!(pl.U, u)                          # {09''} (no-op if u ∉ U)
    end
    return nothing
end

# One iteration of the ComputeShortestPath loop body {11''-28''}.
# Επεξεργάζεται την κορυφή u με το μικρότερο κλειδί της U. Τρεις περιπτώσεις:
#   (α) το κλειδί της U ήταν παλιό (k_old < k_new)  → ξαναμπαίνει με το σωστό κλειδί {14''-15''}
#   (β) u υπερσυνεπής (g > rhs)                       → g(u) = rhs(u), ενημέρωση προγόνων {16''-21''}
#   (γ) u υποσυνεπής  (g < rhs)                       → g(u) = ∞, επανυπολογισμός rhs {22''-28''}
function _process_top!(pl::DStarLitePlanner)
    u, k_old = heap_top(pl.U)                     # {11''} {12''} u = U.Top(), k_old = U.TopKey()
    k_new = calc_key(pl, u)                       # {13''} το κλειδί του u με τα τρέχοντα g, rhs, k_m
    if _key_lt(k_old, k_new)                      # {14''} το κλειδί στην U ήταν ξεπερασμένο (λόγω k_m)
        pl.U[u] = k_new                           # {15''} U.Update(u, k_new)
    elseif pl.g[u...] > pl.rhs[u...]              # {16''} υπερσυνεπής: βρέθηκε φθηνότερος δρόμος
        pl.g[u...] = pl.rhs[u...]                 # {17''} g(u) = rhs(u) → u γίνεται τοπικά συνεπής
        delete!(pl.U, u)                          # {18''} U.Remove(u)
        for s in _neighbors(pl, u)                # {19''} Pred(u) — όσοι μπορούν να φτάσουν στο u
            if s != pl.goal                       # {20''} ο στόχος έχει πάντα rhs = 0
                pl.rhs[s...] = min(pl.rhs[s...], delta_cost(pl.pf, s, u) + pl.g[u...])   # {20''} rhs(s) = min(rhs(s), c(s,u) + g(u))
            end
            update_vertex!(pl, s)                 # {21''}
        end
    else
        # υποσυνεπής: το g(u) ήταν υπερβολικά χαμηλό (π.χ. ακριβότερη/μπλοκαρισμένη ακμή)
        g_old = pl.g[u...]                        # {23''} κρατάμε το παλιό g για τον έλεγχο {26''}
        pl.g[u...] = Inf                          # {24''} g(u) = ∞ (το u γίνεται υπερσυνεπές ή συνεπές)
        for s in _neighbors(pl, u)                # {25''} Pred(u)
            if pl.rhs[s...] == delta_cost(pl.pf, s, u) + g_old      # {26''} το rhs(s) στηριζόταν στο παλιό g(u)
                s != pl.goal && (pl.rhs[s...] = _min_succ(pl, s))   # {27''} rhs(s) = min_{s'∈Succ(s)} c(s,s') + g(s')
            end
            update_vertex!(pl, s)                 # {28''}
        end
        # {25''} also includes s = u. There is no edge (u,u), so c(u,u) = ∞ and
        # {26''} cannot hold: rhs(u) does not depend on g(u), only U changes.
        update_vertex!(pl, u)                     # {28''}
    end
    return nothing
end

# ComputeShortestPath {10''-28''}. U.TopKey() of an empty queue is [∞; ∞], so
# the loop also ends when U is empty (always the exit in full-field mode).
# Επεξεργάζεται την U μέχρι η s_start να είναι τοπικά συνεπής (rhs ≤ g) και το κλειδί της
# να μην είναι μεγαλύτερο από την κορυφή της U — τότε το g στην s_start είναι βέλτιστο.
function compute_shortest_path!(pl::DStarLitePlanner)
    while !isempty(pl.U)
        s = pl.s_start
        # {10''} συνθήκη while: U.TopKey() < CalculateKey(s_start) Ή rhs(s_start) > g(s_start).
        # Αν δεν ισχύει καμία από τις δύο, σταματάμε. Στο full-field mode (s === nothing) τρέχει μέχρι να αδειάσει η U.
        if s !== nothing &&
           !(_key_lt(heap_top(pl.U)[2], calc_key(pl, s)) || pl.rhs[s...] > pl.g[s...])  # {10''}
            break
        end
        _process_top!(pl)       # {11''-28''} σώμα του while
    end
    return pl
end

"""
    update_start!(planner, new_start)

"Move to s_start" {34''-35''} of a focused planner: the agent steps to
`new_start`, normally `argmin_{s'} c(s_start,s') + g(s')`. As in the paper,
neither `km` nor `s_last` changes here; they are updated only when edge cost
changes are detected ({37''-39''}, see [`update_after_cm_change!`](@ref)), and
no replanning is needed while costs stay the same (Theorem 8).
"""
function update_start!(pl::DStarLitePlanner, new_start::Tuple{Int,Int})   # {34''-35''}
    pl.s_start === nothing &&
        throw(ArgumentError("update_start! needs a focused planner (init_planner with a start)"))
    pl.s_start = new_start      # {35''} «Move to s_start»: μόνο η s_start αλλάζει (k_m και s_last όχι)
    return pl
end

# The paper requires h(s,s') ≤ c*(s,s') and h(s,s'') ≤ h(s,s') + h(s',s'')
# ("Heap Reordering", p. 480). All heuristic kinds are norms, so the triangle
# inequality holds, and since every edge cost is at least its base-metric cost,
# h ≤ c* follows from checking a single step in each neighborhood direction.
#
# Έλεγχος ότι η ευρετική είναι επιτρεπτή (admissible): δεν ανήκει στο Figure 4, αλλά
# εγγυάται την προϋπόθεση του paper ώστε τα κλειδιά να είναι κάτω φράγματα και ο
# αλγόριθμος να βρίσκει βέλτιστη διαδρομή (Theorems 1-8).
function _check_heuristic(pl::DStarLitePlanner)
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
            "D* Lite requires h(s,s') ≤ c*(s,s')"))
    end
    return nothing
end

# Αντίγραφο του penalty map που χρησιμοποιήθηκε για τα τρέχοντα κόστη ακμών (ή `nothing` αν δεν υπάρχει penalty map)
_known_pmap(m::Union{PenaltyMap{2},AbsolutePenaltyMap{2}}) = copy(m.pmap)
_known_pmap(::CostMetric) = nothing

# Cost of edge (u,v) under the penalty values in `pmap` (same formula as delta_cost).
# Κόστος ακμής (u,v) με βάση το δοσμένο penalty map (PenaltyMap: διαφορά penalties, AbsolutePenaltyMap: άθροισμα)
_edge_cost(pf, m::PenaltyMap, pmap, u, v) =
    delta_cost(pf, m.base_metric, u, v) + abs(pmap[u...] - pmap[v...])
_edge_cost(pf, m::AbsolutePenaltyMap, pmap, u, v) =
    delta_cost(pf, m.base_metric, u, v) + abs(pmap[u...]) + abs(pmap[v...])

# c(u,v) as the planner last knew it (c_old in {41''}).
# Το παλιό κόστος c_old = c(u,v) πριν την αλλαγή, από το αντίγραφο `known_pmap`
# (το ζωντανό penalty map έχει ήδη αλλάξει όταν καλείται η update_after_cm_change!).
@inline _known_cost(pl::DStarLitePlanner, u, v) =
    pl.known_pmap === nothing ? delta_cost(pl.pf, u, v) :
    _edge_cost(pl.pf, pl.pf.cost_metric, pl.known_pmap, u, v)

# Initialize() {02''-06''} + {29''} + πρώτο ComputeShortestPath {31''} για ΕΝΑΝ στόχο.
function _make_planner(
    pf::DStarLite{2},
    goal::Tuple{Int,Int};
    start::DStarStart = nothing,
    heuristic::Symbol = :none,
)
    sz = size(pf.walkmap)
    hk = start === nothing ? :none : heuristic                         # χωρίς s_start δεν υπάρχει ευρετική
    U = CellHeap(sz)                                                   # {02''} U = ∅
    pl = DStarLitePlanner(pf, fill(Inf, sz), fill(Inf, sz), U, 0.0,    # {03''} k_m = 0 · {04''} g = rhs = ∞ για όλα τα s
        goal, start, start, hk, _known_pmap(pf.cost_metric))           # {29''} s_last = s_start
    _check_heuristic(pl)
    pl.rhs[goal...] = 0.0                                              # {05''} rhs(s_goal) = 0
    h_goal = start === nothing ? 0.0 : heuristic_cost(pl, start, goal)
    U[goal] = (h_goal, 0.0)                                            # {06''} U.Insert(s_goal, [h(s_start,s_goal); 0])
    compute_shortest_path!(pl)                                         # {31''} (η {30''} Initialize() είναι οι παραπάνω γραμμές)
    return pl
end

"""
    init_planner(pf, goal)
    init_planner(pf, goal, start; heuristic = :delta_cost)
    init_planner(pf, goal, heuristic::Symbol)
    init_planner(pf, goal; start = nothing, heuristic = :none)

Build a D* Lite planner aimed at `goal`.

This runs Initialize() and the first ComputeShortestPath() ({29''-31''}).

With no `start`, the queue is emptied so `g` is a full cost-to-go field
(Scenario 3 / many agents sharing an exit). With `start`, search is focused
and `heuristic` is `h(start, s)`:

- `:delta_cost`: base-metric distance (penalty maps only add non-negative
  terms, so they are left out; with `MaxDistance` this is the heuristic of
  the paper's experiments)
- `:manhattan` / `:euclidean`: grid distance from `start` to `s`
- `:none`: `h = 0`

An `ArgumentError` is thrown if the heuristic is not admissible for the cost
metric, e.g. `:manhattan` with diagonal moves under `MaxDistance`.
"""
function init_planner(
    pf::DStarLite{2},
    goal::Tuple{Int,Int};
    start::DStarStart = nothing,
    heuristic::Symbol = :none,
)
    # Κύρια μορφή: Initialize() + πρώτο ComputeShortestPath() (Main {29''-31''})
    return _make_planner(pf, goal; start, heuristic)
end

# Εναλλακτική κλήση με θέση εκκίνησης (focused mode, προεπιλεγμένη ευρετική :delta_cost)
function init_planner(
    pf::DStarLite{2},
    goal::Tuple{Int,Int},
    start::Tuple{Int,Int};
    heuristic::Symbol = :delta_cost,
)
    return _make_planner(pf, goal; start, heuristic)
end

# Εναλλακτική κλήση χωρίς s_start (full-field mode)· το `heuristic` εδώ αγνοείται λόγω start = nothing
function init_planner(pf::DStarLite{2}, goal::Tuple{Int,Int}, heuristic::Symbol)
    return _make_planner(pf, goal; start = nothing, heuristic)
end

"""
    update_after_cm_change!(planner, changed_cells)
    update_after_cm_change!(planner, changed_cells, new_start)

Handle edge-cost changes, lines {37''-48''} of the paper. The caller has already
written the new penalties into the pathfinder's penalty map; `changed_cells`
are the cells whose penalty changed. If the agent has moved since the last
call ({35''}), pass its cell as `new_start`.

1. `km += h(s_last, s_start)`, `s_last = s_start` {38''-39''} (focused only)
2. for every directed edge (u,v) touching a changed cell whose cost changed
   {40''}: take `c_old` from the planner's copy of the previous penalties
   {41''}, adopt the new cost {42''}, and repair `rhs(u)` exactly as in
   {43''-47''}
3. ComputeShortestPath() {48''}

Deviation from the paper: all edges adopt their new cost before the loop,
because a cell's penalty moves all its incident edges at once. {43''-47''} stay
correct, since every changed edge is still visited once with its own `c_old`.
"""
function update_after_cm_change!(
    pl::DStarLitePlanner,
    changed_cells::Vector{Tuple{Int,Int}},
    new_start::DStarStart = nothing,
)
    # Αντιστοιχεί στο σώμα του «if any edge costs changed» {37''-48''} της Main.
    # Το βήμα {38''}: k_m += h(s_last, s_start) ώστε τα κλειδιά της U να παραμείνουν κάτω φράγματα.
    if pl.s_start !== nothing
        new_start !== nothing && (pl.s_start = new_start)          # {35''} ο agent μετακινήθηκε
        pl.km += heuristic_cost(pl, pl.s_last, pl.s_start)         # {38''} k_m = k_m + h(s_last, s_start)
        pl.s_last = pl.s_start                                     # {39''} s_last = s_start
    end

    # {40''-41''}: directed edges incident to changed cells, with c_old
    # Συλλογή των κατευθυνόμενων ακμών (u,v) που αγγίζουν αλλαγμένα cells, μαζί με το παλιό κόστος c_old.
    wm = pl.pf.walkmap
    seen = Set{NTuple{2,Tuple{Int,Int}}}()                  # αποφυγή διπλής επεξεργασίας της ίδιας ακμής
    edges = Tuple{Tuple{Int,Int},Tuple{Int,Int},Float64}[]  # (u, v, c_old)
    for x in changed_cells
        wm[x...] || continue                  # a blocked cell has no edges
        for n in _neighbors(pl, x), e in ((x, n), (n, x))   # και οι δύο κατευθύνσεις x→n, n→x
            e in seen && continue
            push!(seen, e)
            push!(edges, (e[1], e[2], _known_cost(pl, e[1], e[2])))   # {41''} c_old = c(u,v)
        end
    end

    # {42''}: adopt the new costs
    # Ενημέρωση του αντιγράφου `known_pmap` με τις νέες τιμές (η «Update the edge cost c(u,v)» του paper)
    if pl.known_pmap !== nothing
        live = pl.pf.cost_metric.pmap
        for x in changed_cells
            pl.known_pmap[x...] = live[x...]
        end
    end

    # Επεξεργασία κάθε αλλαγμένης ακμής {43''-47''}: διορθώνει το rhs(u) και ενημερώνει την U
    for (u, v, c_old) in edges
        c_new = delta_cost(pl.pf, u, v)       # το νέο κόστος c(u,v)
        c_old == c_new && continue            # {40''} only changed edges
        if c_old > c_new                                                   # {43''} η ακμή έγινε φθηνότερη
            u != pl.goal && (pl.rhs[u...] = min(pl.rhs[u...], c_new + pl.g[v...]))  # {44''} rhs(u) = min(rhs(u), c(u,v) + g(v))
        elseif pl.rhs[u...] == c_old + pl.g[v...]                          # {45''} η ακμή έγινε ακριβότερη και το rhs(u) στηριζόταν σε αυτήν
            u != pl.goal && (pl.rhs[u...] = _min_succ(pl, u))              # {46''} rhs(u) = min_{s'∈Succ(u)} c(u,s') + g(s')
        end
        update_vertex!(pl, u)                                              # {47''}
    end

    compute_shortest_path!(pl)                                             # {48''} επανυπολογισμός μόνο όπου χρειάζεται
    return pl
end

"""
    extract_path(planner, start; max_steps = 10_000)

Walk from `start` to `planner.goal`, always moving to a successor `s'` that
minimizes `c(s,s') + g(s')` ({34''}). Theorem 8 guarantees this is a shortest
path from the `s_start` that ComputeShortestPath() last ran for, and from any
cell on that path.

For a focused planner, a different `start` means the agent was relocated to a
cell that might not be on that path. The planner then runs
{38''-39''} and {48''} with no changed edges, i.e. what the paper's main loop
does on its next scan. For moves along the path, call
[`update_start!`](@ref) and then `extract_path(planner, planner.s_start)`.
"""
function extract_path(
    planner::DStarLitePlanner,
    start::Tuple{Int,Int};
    max_steps = 10_000,
)
    # Focused mode: αν ο agent βρίσκεται αλλού από την s_start, ο planner πρώτα προσαρμόζεται
    # ({38''-39''} και {48''} χωρίς αλλαγμένες ακμές), όπως θα έκανε το επόμενο scan της Main.
    if planner.s_start !== nothing && start != planner.s_start
        planner.s_start = start
        update_after_cm_change!(planner, Tuple{Int,Int}[])
    end

    # Η ανακατασκευή της διαδρομής είναι η γραμμή {34''}: s_start = argmin_{s'∈Succ(s_start)} c(s_start,s') + g(s'),
    # επαναλαμβανόμενη μέχρι τον στόχο (Theorem 8: η διαδρομή αυτή είναι συντομότερη).
    path = Tuple{Int,Int}[start]
    cur = start
    for _ in 1:max_steps                      # όριο βημάτων για προστασία από ατέρμονο βρόχο
        cur == planner.goal && break          # έφτασε στον στόχο ({32''} while s_start ≠ s_goal)
        best = nothing
        best_val = Inf
        for n in _neighbors(planner, cur)     # Succ(cur)
            val = delta_cost(planner.pf, cur, n) + planner.g[n...]    # c(cur,n) + g(n)
            if val < best_val
                best_val = val
                best = n
            end
        end
        best === nothing && break             # {33''}: no known path
        cur = best
        push!(path, cur)                      # η επόμενη κορυφή της διαδρομής
    end
    return path
end
