export DStarLite, find_path
# D* Lite pathfinder for Agents.jl GridSpace / ContinuousSpace
#
# ΣΧΕΣΗ ΜΕ ΤΟ PAPER (Koenig & Likhachev, AAAI 2002, LITERATURE/DStarLite.pdf)
# Αυτό το αρχείο έχει δύο μέρη:
#  1. Ο τύπος `DStarLite` (struct + constructors): ΔΕΝ είναι αλγόριθμος του paper. Είναι ο
#     «γράφος» πάνω στον οποίο τρέχει το D* Lite, στη μορφή που περιμένει το Agents.jl:
#       - S, το σύνολο των κορυφών         → τα cells του `walkmap` (true = βατό)
#       - Succ(s) / Pred(s), οι γείτονες    → `neighborhood` (Moore: 8 κατευθύνσεις, ή von Neumann: 4)
#       - c(s,s'), το κόστος ακμής          → `cost_metric` (π.χ. AbsolutePenaltyMap) μέσω `delta_cost`
#       - h(s,s'), η ευρετική               → το βασικό metric του cost_metric (`_heuristic_metric`)
#  2. Η `find_path`: one-shot εκδοχή του Figure 4 (Initialize + ComputeShortestPath + greedy
#     ανακατασκευή διαδρομής). Η επαναληπτική εκδοχή με το main loop (αλλαγές κόστους κατά την κίνηση)
#     υλοποιείται στο dstar_lite_planner.jl (`DStarLitePlanner`).
# Οι υπόλοιπες συναρτήσεις (plan_route!, move_along_route!, κ.λπ.) είναι ο κολλητικός κώδικας του
# Agents.jl και δεν έχουν αντίστοιχο στο paper.

# Ο pathfinder: κρατά τον γράφο (walkmap, neighborhood, cost_metric) και τα αποθηκευμένα paths των agents.
struct DStarLite{D,P,M,T,C<:CostMetric{D}} <: GridPathfinder{D,P,M}
    agent_paths::Dict{Int,Path{D,T}}    # διαδρομή ανά agent (id → path), για plan_route!/move_along_route!
    dims::NTuple{D,T}                   # διαστάσεις του grid
    neighborhood::Vector{CartesianIndex{D}}   # μετατοπίσεις γειτόνων → ορίζει τα Succ(s)/Pred(s)
    admissibility::Float64      # κρατιέται για συμβατότητα, δεν χρησιμοποιείται ιδιαίτερα εδώ
    walkmap::BitArray{D}                # S: ποια cells είναι βατά (κορυφές του γράφου)
    cost_metric::C                      # c(s,s'): πώς υπολογίζεται το κόστος κίνησης

    # Εσωτερικός constructor: ελέγχει ότι οι παράμετροι είναι συνεπείς πριν δημιουργηθεί ο pathfinder
    function DStarLite{D,P,M,T,C}(
        agent_paths::Dict,
        dims::NTuple{D,T},
        neighborhood::Vector{CartesianIndex{D}},
        admissibility::Float64,
        walkmap::BitArray{D},
        cost_metric::C,
    ) where {D,P,M,C,T}
        @assert all(dims .> 0) "Invalid pathfinder dimensions: $(dims)"                       # θετικές διαστάσεις
        T <: Integer && @assert size(walkmap) == dims "Walkmap must be same dimensions as grid"   # walkmap ίδιου μεγέθους με το grid
        @assert admissibility >= 0 "Invalid value for admissibility: $admissibility ≱ 0"

        # έλεγχος ότι το cost_metric είναι συμβατό με το walkmap (penalty maps ίδιου μεγέθους, κ.λπ.)
        if cost_metric isa PenaltyMap{D}
            @assert size(cost_metric.pmap) == size(walkmap) "Penaltymap dimensions must be same as walkable map"
        elseif cost_metric isa AbsolutePenaltyMap{D}
            @assert size(cost_metric.pmap) == size(walkmap) "Penaltymap dimensions must be same as walkable map"
        elseif cost_metric isa DirectDistance{D}
            if M
                @assert length(cost_metric.direction_costs) >= D "DirectDistance direction_costs must have as many values as dimensions"
            else
                @assert length(cost_metric.direction_costs) >= 1 "DirectDistance direction_costs must have non-zero length"
            end
        end

        new(agent_paths, dims, neighborhood, admissibility, walkmap, cost_metric)
    end
end

"""
    DStarLite(dims::NTuple{D,T}; kwargs...)

Δημιουργεί pathfinder τύπου D* Lite για grid `dims`.
"""
function DStarLite(
    dims::NTuple{D,T};
    periodic::Union{Bool,NTuple{D,Bool}} = false,
    diagonal_movement::Bool = true,
    admissibility::Float64 = 0.0,
    walkmap::BitArray{D} = trues(dims),
    cost_metric::CostMetric{D} = DirectDistance{D}(),
) where {D,T}
    # Succ/Pred: 8 γείτονες (Moore) με διαγώνιες κινήσεις, αλλιώς 4 (von Neumann)
    neighborhood = diagonal_movement ? moore_neighborhood(D) : vonneumann_neighborhood(D)
    return DStarLite{D,periodic,diagonal_movement,T,typeof(cost_metric)}(
        Dict{Int,Path{D,T}}(),
        dims,
        neighborhood,
        admissibility,
        walkmap,
        cost_metric,
    )
end

# Constructor από GridSpace: παίρνει τις διαστάσεις του χώρου
DStarLite(
    space::GridSpace{D,periodic};
    diagonal_movement::Bool = true,
    admissibility::Float64 = 0.0,
    walkmap::BitArray{D} = trues(spacesize(space)),
    cost_metric::CostMetric{D} = DirectDistance{D}(),
) where {D,periodic} = DStarLite(size(space); periodic, diagonal_movement, admissibility, walkmap, cost_metric)

# Constructor από ContinuousSpace: ο συνεχής χώρος διακριτοποιείται σε grid με τις διαστάσεις του walkmap/penalty map
# (η διακριτοποίηση γίνεται από τις to_discrete_position / to_continuous_position, βλ. dstar_lite_continuous.jl)
function DStarLite(
    space::ContinuousSpace{D,periodic};
    walkmap::Union{BitArray{D},Nothing} = nothing,
    admissibility::Float64 = 0.0,
    cost_metric::CostMetric{D} = DirectDistance{D}(),
) where {D,periodic}
    @assert walkmap isa BitArray{D} || cost_metric isa PenaltyMap || cost_metric isa AbsolutePenaltyMap "Pathfinding in ContinuousSpace requires either walkmap to be specified or cost_metric to be a PenaltyMap or AbsolutePenaltyMap"
    isnothing(walkmap) && (walkmap = BitArray(trues(size(cost_metric.pmap))))
    DStarLite(
        Tuple(Agents.spacesize(space));
        periodic,
        diagonal_movement = true,
        admissibility,
        walkmap,
        cost_metric,
    )
end

# Εμφάνιση του pathfinder στο REPL (περιγραφή διαστάσεων, γειτονιάς και cost_metric)
function Base.show(io::IO, pathfinder::DStarLite{D,P,M}) where {D,P,M}
    periodic = get_periodic_type(pathfinder)
    moore = M ? "diagonal, " : "orthogonal, "
    s = "D* Lite in $(D) dimensions, $(periodic)$(moore)ϵ=$(pathfinder.admissibility), " *
        "metric=$(pathfinder.cost_metric)"
    print(io, s)
end

get_periodic_type(::DStarLite{D,false,M}) where {D,M} = ""
get_periodic_type(::DStarLite{D,true,M}) where {D,M} = "periodic, "
get_periodic_type(::DStarLite{D,P,M}) where {D,P,M} = "mixed periodicity, "

# ---------------- D* Lite core: find_path ----------------

# Metric used for the heuristic h(s,s'). Penalty maps only add non-negative
# terms to edge costs, so their base metric satisfies h(s,s') ≤ c*(s,s') and
# the triangle inequality (Koenig & Likhachev 2002, p. 480) with h(s,s) = 0.
#
# Το paper ("Heap Reordering", σ. 480) απαιτεί h(s,s') ≤ c*(s,s') και h(s,s'') ≤ h(s,s') + h(s',s'').
# Επειδή τα penalty maps μόνο προσθέτουν μη αρνητικούς όρους στο κόστος, ως ευρετική αρκεί
# το βασικό metric (χωρίς penalties): δεν υπερεκτιμά ποτέ το πραγματικό κόστος.
_heuristic_metric(m::CostMetric) = m
_heuristic_metric(m::PenaltyMap) = m.base_metric
_heuristic_metric(m::AbsolutePenaltyMap) = m.base_metric

"""
    find_path(pathfinder::DStarLite{D}, from::Dims{D}, to::Dims{D})

Υπολογίζει διαδρομή `from → to` με τον αλγόριθμο D* Lite (optimized version,
Figure 4 του paper): Initialize() + ComputeShortestPath() και μετά greedy
`argmin c(s,s') + g(s')` (Theorem 8). Είναι one-shot, χωρίς το main loop
{32''-48''}; για επαναληπτικό σχεδιασμό βλ. [`init_planner`](@ref).
Επιστρέφει `Path{D,Int}` ή `nothing` αν δεν υπάρχει διαδρομή.
"""
function find_path(pathfinder::DStarLite{D}, from::Dims{D}, to::Dims{D}) where {D}
    # bounds / walkable όπως στο AStar
    # έλεγχος ότι αρχή και στόχος είναι εντός ορίων και βατά, αλλιώς δεν υπάρχει διαδρομή
    if !all(1 .<= from .<= size(pathfinder.walkmap)) ||
       !all(1 .<= to .<= size(pathfinder.walkmap)) ||
       !pathfinder.walkmap[from...] || !pathfinder.walkmap[to...]
        return
    end

    # const INF = Inf

    # Πίνακες g και rhs του paper, αραιοί (Dict): όποια κορυφή λείπει έχει τιμή ∞ (βλ. getg/getrhs).
    # Αντιστοιχούν στο {04''} «for all s ∈ S: rhs(s) = g(s) = ∞».
    g   = Dict{Dims{D},Float64}()
    rhs = Dict{Dims{D},Float64}()

    s_start = from      # s_start: η αρχή του agent
    s_goal  = to        # s_goal: ο στόχος (το D* Lite αναζητά ΑΝΑΠΟΔΑ, από τον στόχο προς την αρχή)
    km = 0.0            # {03''} k_m = 0 (στο one-shot δεν αλλάζει ποτέ, αφού ο agent δεν κινείται)

    # U: ουρά προτεραιότητας με κλειδιά [k1; k2], ως tuple (k1, k2)· {02''} U = ∅
    U = PriorityQueue{Dims{D},Tuple{Float64,Float64}}()

    getg(s)   = get(g,   s, Inf)       # g(s)   (∞ αν δεν έχει οριστεί)
    getrhs(s) = get(rhs, s, Inf)       # rhs(s) (∞ αν δεν έχει οριστεί)
    setg!(s, v)   = (g[s]   = v)
    setrhs!(s, v) = (rhs[s] = v)

    # h(s_start, s): η ευρετική του paper, εδώ το βασικό metric χωρίς penalties
    hmetric = _heuristic_metric(pathfinder.cost_metric)
    h(s) = Float64(delta_cost(pathfinder, hmetric, s_start, s))   # h(s_start, s)

    # CalculateKey(s) {01''}: [min(g(s), rhs(s)) + h(s_start, s) + k_m ; min(g(s), rhs(s))]
    function calc_key(s)                                          # {01''}
        m = min(getg(s), getrhs(s))
        return (m + h(s) + km, m)
    end

    # λεξικογραφική σύγκριση κλειδιών (η «<» του paper στα {10''}, {14''})
    key_lt(a::Tuple{T,T}, b::Tuple{T,T}) where {T<:Real} =
        (a[1] < b[1]) || (a[1] == b[1] && a[2] < b[2])

    # Το rhs μιας κορυφής από τους γείτονές της: το καλύτερο c(u,s') + g(s') (χρησιμοποιείται στα {27''} {46''})
    function min_succ(u)                  # min_{s' ∈ Succ(u)} (c(u,s') + g(s'))
        m = Inf
        for sp in walkable_neighbors(u, pathfinder)
            m = min(m, delta_cost(pathfinder, u, sp) + getg(sp))
        end
        return m
    end

    # UpdateVertex(u) {07''-09''}: κρατά την U συνεπή με τα g/rhs (μόνο οι τοπικά ασυνεπείς κορυφές βρίσκονται στην U)
    function update_vertex!(u)                                    # {07''-09''}
        if getg(u) != getrhs(u)
            U[u] = calc_key(u)                                    # U.Update / U.Insert {07''} {08''}
        elseif haskey(U, u)
            delete!(U, u)                                         # U.Remove {09''}
        end
        return nothing
    end

    # Initialize() {02''-06''}
    # rhs(s_goal) = 0 {05''} και ο στόχος μπαίνει στην U με κλειδί [h(s_start,s_goal); 0] {06''}
    setrhs!(s_goal, 0.0)
    U[s_goal] = (h(s_goal), 0.0)

    # ComputeShortestPath() {10''-28''}; an empty U has TopKey [∞; ∞]
    # Επεξεργάζεται την U με σειρά κλειδιού μέχρι η s_start να γίνει τοπικά συνεπή με κλειδί ≤ της κορυφής της U
    while !isempty(U)
        u, k_old = peek(U)                                        # {11''} {12''} u = U.Top(), k_old = U.TopKey()
        # {10''} συνθήκη while: συνεχίζει όσο U.TopKey() < CalculateKey(s_start) Ή rhs(s_start) > g(s_start)
        key_lt(k_old, calc_key(s_start)) || getrhs(s_start) > getg(s_start) || break  # {10''}
        k_new = calc_key(u)                                       # {13''}
        if key_lt(k_old, k_new)                                   # {14''} ξεπερασμένο κλειδί
            U[u] = k_new                                          # {15''} U.Update(u, k_new)
        elseif getg(u) > getrhs(u)                                # {16''} υπερσυνεπής κορυφή
            setg!(u, getrhs(u))                                   # {17''} g(u) = rhs(u)
            delete!(U, u)                                         # {18''} U.Remove(u)
            for s in walkable_neighbors(u, pathfinder)            # {19''} Pred(u)
                s != s_goal &&                                    # {20''}
                    setrhs!(s, min(getrhs(s), delta_cost(pathfinder, s, u) + getg(u)))   # rhs(s) = min(rhs(s), c(s,u) + g(u))
                update_vertex!(s)                                 # {21''}
            end
        else
            # υποσυνεπής κορυφή (g(u) < rhs(u)): το g(u) ήταν υπερβολικά χαμηλό
            g_old = getg(u)                                       # {23''}
            setg!(u, Inf)                                         # {24''} g(u) = ∞
            for s in walkable_neighbors(u, pathfinder)            # {25''} Pred(u)
                if getrhs(s) == delta_cost(pathfinder, s, u) + g_old   # {26''} το rhs(s) στηριζόταν στο παλιό g(u)
                    s != s_goal && setrhs!(s, min_succ(s))        # {27''} ξαναϋπολογισμός rhs(s)
                end
                update_vertex!(s)                                 # {28''}
            end
            update_vertex!(u)   # s = u in {25''}: no edge (u,u), so {26''} fails
        end
    end

    # {33''}. The optimized termination may leave s_start overconsistent with
    # g(s_start) = ∞ although a path exists (p. 481), so test rhs(s_start),
    # which is exactly the value the greedy walk below minimizes.
    # Αν rhs(s_start) = ∞ δεν υπάρχει γνωστή διαδρομή (σχόλιο {33''} του paper)
    if getrhs(s_start) == Inf
        return
    end

    # Ανακατασκευή διαδρομής: greedy, όπως στο κλασικό D* Lite
    # Γραμμή {34''} του Main: από την s_start πηγαίνουμε διαδοχικά στον γείτονα s' που ελαχιστοποιεί c(s,s') + g(s')
    # μέχρι τον στόχο (Theorem 8: αυτό δίνει συντομότερη διαδρομή).
    agent_path = Path{D,Int64}()
    cur = s_start

    while cur != s_goal                                           # {32''} while s_start ≠ s_goal
        best = nothing
        best_cost = Inf
        for n in walkable_neighbors(cur, pathfinder)              # Succ(cur)
            cost = delta_cost(pathfinder, cur, n) + getg(n)       # c(cur,n) + g(n)
            if cost < best_cost
                best_cost = cost
                best = n
            end
        end

        best === nothing && return   # δεν βρέθηκε συνέχεια

        push!(agent_path, best)      # επόμενη κορυφή της διαδρομής ({34''} + {35''} «Move to s_start»)
        cur = best
    end

    return agent_path
end

# ---------------- Grid helpers (ίδια API με AStar) ----------------
# Από εδώ και κάτω: το interface του Agents.jl για pathfinding (plan_route!, plan_best_route!,
# move_along_route!, κ.λπ.). ΔΕΝ ανήκουν στον αλγόριθμο του paper· απλώς καλούν την `find_path`
# και αποθηκεύουν/καταναλώνουν τη διαδρομή του κάθε agent στο `pathfinder.agent_paths`.

"""
    Agents.plan_route!(agent, dest, pathfinder::DStarLite{D})

Υπολογίζει και αποθηκεύει την διαδρομή `agent.pos → dest` με D* Lite.
"""
function Agents.plan_route!(
    agent::AbstractAgent,
    dest,
    pathfinder::DStarLite{D},
) where {D}
    path = find_path(pathfinder, agent.pos, dest)
    isnothing(path) && return
    pathfinder.agent_paths[Agents.getid(agent)] = path
end

"""
    Agents.plan_best_route!(agent, dests, pathfinder::DStarLite{D}; condition=:shortest)

Όπως για AStar, αλλά χρησιμοποιεί D* Lite για κάθε υποψήφιο προορισμό.
"""
function Agents.plan_best_route!(
    agent::AbstractAgent,
    dests,
    pathfinder::DStarLite{D,P,M,Int64};
    condition::Symbol = :shortest,
) where {D,P,M}
    @assert condition ∈ (:shortest, :longest)
    compare = condition == :shortest ? (a, b) -> a < b : (a, b) -> a > b

    best_path   = Path{D,Int64}()
    best_target = nothing

    for target in dests
        path = find_path(pathfinder, agent.pos, target)
        isnothing(path) && continue

        if isempty(best_path) || compare(length(path), length(best_path))
            best_path   = path
            best_target = target
        end
    end

    isnothing(best_target) && return
    pathfinder.agent_paths[Agents.getid(agent)] = best_path
    return best_target
end

"""
    Agents.move_along_route!(agent, model::ABM{<:GridSpace{D}}, pathfinder::DStarLite{D})

Κινεί τον agent ένα βήμα κατά μήκος της αποθηκευμένης διαδρομής.
"""
function Agents.move_along_route!(
    agent::AbstractAgent,
    model::ABM{<:GridSpace{D}},
    pathfinder::DStarLite{D},
) where {D}
    isempty(Agents.getid(agent), pathfinder) && return
    move_agent!(agent, first(pathfinder.agent_paths[Agents.getid(agent)]), model)
    popfirst!(pathfinder.agent_paths[Agents.getid(agent)])
end

"""
    Pathfinding.nearby_walkable(position, model, pathfinder::DStarLite{D}, r)

Ίδιο με την εκδοχή για AStar.
"""
nearby_walkable(
    position,
    model::ABM{<:GridSpace{D}},
    pathfinder::DStarLite{D},
    r = 1,
) where {D} = Iterators.filter(
    x -> pathfinder.walkmap[x...] == 1,
    nearby_positions(position, model, r),
)

"""
    Pathfinding.random_walkable(model, pathfinder::DStarLite{D})

Τυχαία walkable θέση, ίδιο API με AStar.
"""
function random_walkable(model::ABM{<:GridSpace{D}}, pathfinder::DStarLite{D}) where {D}
    return Tuple(
        rand(
            abmrng(model),
            filter(x -> pathfinder.walkmap[x], CartesianIndices(abmspace(model).stored_ids)),
        ),
    )
end

"""
    Pathfinding.penaltymap(pathfinder::DStarLite)

Ό,τι και για AStar.
"""
function penaltymap(pathfinder::DStarLite)
    if pathfinder.cost_metric isa PenaltyMap
        return pathfinder.cost_metric.pmap
    elseif pathfinder.cost_metric isa AbsolutePenaltyMap
        return pathfinder.cost_metric.pmap
    else
        return nothing
    end
end

"""
    Agents.remove_agent!(agent, model, pathfinder::DStarLite)

Σβήνει και τα path δεδομένα του agent.
"""
function Agents.remove_agent!(agent::AbstractAgent, model::ABM, pathfinder::DStarLite)
    delete!(pathfinder.agent_paths, Agents.getid(agent))
    Agents.remove_agent_from_container!(agent, model)
    Agents.remove_agent_from_space!(agent, model)
end

Base.isempty(id::Int, pathfinder::DStarLite) =
    !haskey(pathfinder.agent_paths, id) || isempty(pathfinder.agent_paths[id])

Agents.is_stationary(agent::AbstractAgent, pathfinder::DStarLite) =
    isempty(Agents.getid(agent), pathfinder)