export LPAStar
# LPA* (Lifelong Planning A*) pathfinder for Agents.jl GridSpace / ContinuousSpace
#
# ΣΧΕΣΗ ΜΕ ΤΟ PAPER (Koenig, Likhachev & Furcy, "Lifelong Planning A*", preprint Elsevier Science 2005,
# LITERATURE/LPAStar.pdf). Η υλοποίηση ακολουθεί τη βελτιστοποιημένη εκδοχή του Figure 7 (γραμμές {01'}-{44'}),
# όπως ο D* Lite του repo ακολουθεί το Figure 4 του δικού του paper.
# Αυτό το αρχείο έχει δύο μέρη:
#  1. Ο τύπος `LPAStar` (struct + constructors): ΔΕΝ είναι αλγόριθμος του paper. Είναι ο «γράφος» πάνω στον
#     οποίο τρέχει ο LPA*, στη μορφή που περιμένει το Agents.jl (ίδιος με του `DStarLite`):
#       - S, το σύνολο των κορυφών         → τα cells του `walkmap` (true = βατό)
#       - succ(s) / pred(s), οι γείτονες    → `neighborhood` (Moore: 8 κατευθύνσεις, ή von Neumann: 4)
#       - c(s,s'), το κόστος ακμής          → `cost_metric` (π.χ. AbsolutePenaltyMap) μέσω `delta_cost`
#       - h(s), η ευρετική προς τον στόχο   → το βασικό metric του cost_metric (`_heuristic_metric`)
#  2. Η `find_path`: one-shot εκδοχή (Initialize + ComputeShortestPath + ανακατασκευή διαδρομής).
#     Η επαναληπτική εκδοχή (αλλαγές κόστους/εμποδίων μεταξύ αναζητήσεων) είναι στο lpa_star_planner.jl.
# Οι υπόλοιπες συναρτήσεις (plan_route!, move_along_route!, κ.λπ.) είναι ο κολλητικός κώδικας του
# Agents.jl και δεν έχουν αντίστοιχο στο paper.
#
# ΔΙΑΦΟΡΑ ΑΠΟ ΤΟ D* LITE: ο LPA* αναζητά ΕΥΘΕΙΑ (από την s_start προς την s_goal) με σταθερή αρχή και στόχο·
# δεν υποστηρίζει κινούμενο agent (δεν υπάρχει k_m ούτε heap reordering). Όπως και ο D* Lite, χρειάζεται το
# παλιό κόστος c_old των ακμών που άλλαξαν ({33'})· το κρατά ο planner σε αντίγραφα του penalty map / walkmap.

# Ο pathfinder: κρατά τον γράφο (walkmap, neighborhood, cost_metric) και τα αποθηκευμένα paths των agents.
struct LPAStar{D,P,M,T,C<:CostMetric{D}} <: GridPathfinder{D,P,M}
    agent_paths::Dict{Int,Path{D,T}}    # διαδρομή ανά agent (id → path), για plan_route!/move_along_route!
    dims::NTuple{D,T}                   # διαστάσεις του grid
    neighborhood::Vector{CartesianIndex{D}}   # μετατοπίσεις γειτόνων → ορίζει τα succ(s)/pred(s)
    admissibility::Float64              # κρατιέται για συμβατότητα με τα AStar/DStarLite, δεν χρησιμοποιείται
    walkmap::BitArray{D}                # S: ποια cells είναι βατά (κορυφές του γράφου)
    cost_metric::C                      # c(s,s'): πώς υπολογίζεται το κόστος κίνησης

    # Εσωτερικός constructor: ελέγχει ότι οι παράμετροι είναι συνεπείς πριν δημιουργηθεί ο pathfinder
    function LPAStar{D,P,M,T,C}(
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
    LPAStar(dims::NTuple{D,T}; kwargs...)

Δημιουργεί pathfinder τύπου LPA* (Lifelong Planning A*) για grid `dims`. Δέχεται τα ίδια keywords με
το [`DStarLite`](@ref): `periodic`, `diagonal_movement`, `admissibility`, `walkmap`, `cost_metric`.
Η επαναληπτική χρήση (αλλαγές κόστους μεταξύ αναζητήσεων) γίνεται μέσω [`init_planner`](@ref).
"""
function LPAStar(
    dims::NTuple{D,T};
    periodic::Union{Bool,NTuple{D,Bool}} = false,
    diagonal_movement::Bool = true,
    admissibility::Float64 = 0.0,
    walkmap::BitArray{D} = trues(dims),
    cost_metric::CostMetric{D} = DirectDistance{D}(),
) where {D,T}
    # succ/pred: 8 γείτονες (Moore) με διαγώνιες κινήσεις, αλλιώς 4 (von Neumann)
    neighborhood = diagonal_movement ? moore_neighborhood(D) : vonneumann_neighborhood(D)
    return LPAStar{D,periodic,diagonal_movement,T,typeof(cost_metric)}(
        Dict{Int,Path{D,T}}(),
        dims,
        neighborhood,
        admissibility,
        walkmap,
        cost_metric,
    )
end

# Constructor από GridSpace: παίρνει τις διαστάσεις του χώρου
LPAStar(
    space::GridSpace{D,periodic};
    diagonal_movement::Bool = true,
    admissibility::Float64 = 0.0,
    walkmap::BitArray{D} = trues(spacesize(space)),
    cost_metric::CostMetric{D} = DirectDistance{D}(),
) where {D,periodic} = LPAStar(size(space); periodic, diagonal_movement, admissibility, walkmap, cost_metric)

# Constructor από ContinuousSpace: ο συνεχής χώρος διακριτοποιείται σε grid με τις διαστάσεις του walkmap/penalty map
# (η διακριτοποίηση γίνεται από τις to_discrete_position / to_continuous_position, βλ. lpa_star_continuous.jl)
function LPAStar(
    space::ContinuousSpace{D,periodic};
    walkmap::Union{BitArray{D},Nothing} = nothing,
    admissibility::Float64 = 0.0,
    cost_metric::CostMetric{D} = DirectDistance{D}(),
) where {D,periodic}
    @assert walkmap isa BitArray{D} || cost_metric isa PenaltyMap || cost_metric isa AbsolutePenaltyMap "Pathfinding in ContinuousSpace requires either walkmap to be specified or cost_metric to be a PenaltyMap or AbsolutePenaltyMap"
    isnothing(walkmap) && (walkmap = BitArray(trues(size(cost_metric.pmap))))
    LPAStar(
        Tuple(Agents.spacesize(space));
        periodic,
        diagonal_movement = true,
        admissibility,
        walkmap,
        cost_metric,
    )
end

# Εμφάνιση του pathfinder στο REPL (περιγραφή διαστάσεων, γειτονιάς και cost_metric)
function Base.show(io::IO, pathfinder::LPAStar{D,P,M}) where {D,P,M}
    periodic = get_periodic_type(pathfinder)
    moore = M ? "diagonal, " : "orthogonal, "
    s = "LPA* in $(D) dimensions, $(periodic)$(moore)ϵ=$(pathfinder.admissibility), " *
        "metric=$(pathfinder.cost_metric)"
    print(io, s)
end

get_periodic_type(::LPAStar{D,false,M}) where {D,M} = ""
get_periodic_type(::LPAStar{D,true,M}) where {D,M} = "periodic, "
get_periodic_type(::LPAStar{D,P,M}) where {D,P,M} = "mixed periodicity, "

# ---------------- LPA* core: find_path ----------------

"""
    find_path(pathfinder::LPAStar{2}, from::Dims{2}, to::Dims{2})

Υπολογίζει διαδρομή `from → to` με τον LPA* (Figure 7 του paper): Initialize() + ComputeShortestPath()
και ανακατασκευή της διαδρομής προς τα πίσω από τον στόχο. Είναι one-shot (χτίζει νέο planner σε κάθε
κλήση, κόστος O(|S|) αρχικοποίησης)· για επαναληπτικό σχεδιασμό με αλλαγές κόστους βλ.
[`init_planner`](@ref). Επιστρέφει `Path{2,Int}` (χωρίς την αρχική θέση, όπως τα `AStar`/`DStarLite`)
ή `nothing` αν δεν υπάρχει διαδρομή.
"""
function find_path(pathfinder::LPAStar{2}, from::Dims{2}, to::Dims{2})
    # έλεγχος ότι αρχή και στόχος είναι εντός ορίων και βατά, αλλιώς δεν υπάρχει διαδρομή
    if !all(1 .<= from .<= size(pathfinder.walkmap)) ||
       !all(1 .<= to .<= size(pathfinder.walkmap)) ||
       !pathfinder.walkmap[from...] || !pathfinder.walkmap[to...]
        return
    end

    planner = init_planner(pathfinder, from, to)     # Initialize() + ComputeShortestPath() {28'}-{30'}
    cells = extract_path(planner)                    # προς τα πίσω από τον στόχο, ως την αρχή
    isnothing(cells) && return                       # g(s_goal) = ∞: δεν υπάρχει διαδρομή

    agent_path = Path{2,Int64}()
    for c in Iterators.drop(cells, 1)                # χωρίς την αρχική θέση (ο agent βρίσκεται ήδη εκεί)
        push!(agent_path, c)
    end
    return agent_path
end

# ---------------- Grid helpers (ίδια API με AStar / DStarLite) ----------------
# Από εδώ και κάτω: το interface του Agents.jl για pathfinding (plan_route!, plan_best_route!,
# move_along_route!, κ.λπ.). ΔΕΝ ανήκουν στον αλγόριθμο του paper· απλώς καλούν την `find_path`
# και αποθηκεύουν/καταναλώνουν τη διαδρομή του κάθε agent στο `pathfinder.agent_paths`.

"""
    Agents.plan_route!(agent, dest, pathfinder::LPAStar{2})

Υπολογίζει και αποθηκεύει την διαδρομή `agent.pos → dest` με LPA*.
"""
function Agents.plan_route!(
    agent::AbstractAgent,
    dest,
    pathfinder::LPAStar{2},
)
    path = find_path(pathfinder, agent.pos, dest)
    isnothing(path) && return
    pathfinder.agent_paths[Agents.getid(agent)] = path
end

"""
    Agents.plan_best_route!(agent, dests, pathfinder::LPAStar{2}; condition=:shortest)

Όπως για AStar, αλλά χρησιμοποιεί LPA* για κάθε υποψήφιο προορισμό.
"""
function Agents.plan_best_route!(
    agent::AbstractAgent,
    dests,
    pathfinder::LPAStar{2,P,M,Int64};
    condition::Symbol = :shortest,
) where {P,M}
    @assert condition ∈ (:shortest, :longest)
    compare = condition == :shortest ? (a, b) -> a < b : (a, b) -> a > b

    best_path   = Path{2,Int64}()
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
    Agents.move_along_route!(agent, model::ABM{<:GridSpace{2}}, pathfinder::LPAStar{2})

Κινεί τον agent ένα βήμα κατά μήκος της αποθηκευμένης διαδρομής.
"""
function Agents.move_along_route!(
    agent::AbstractAgent,
    model::ABM{<:GridSpace{2}},
    pathfinder::LPAStar{2},
)
    isempty(Agents.getid(agent), pathfinder) && return
    move_agent!(agent, first(pathfinder.agent_paths[Agents.getid(agent)]), model)
    popfirst!(pathfinder.agent_paths[Agents.getid(agent)])
end

"""
    Pathfinding.nearby_walkable(position, model, pathfinder::LPAStar{D}, r)

Ίδιο με την εκδοχή για AStar.
"""
nearby_walkable(
    position,
    model::ABM{<:GridSpace{D}},
    pathfinder::LPAStar{D},
    r = 1,
) where {D} = Iterators.filter(
    x -> pathfinder.walkmap[x...] == 1,
    nearby_positions(position, model, r),
)

"""
    Pathfinding.random_walkable(model, pathfinder::LPAStar{D})

Τυχαία walkable θέση, ίδιο API με AStar.
"""
function random_walkable(model::ABM{<:GridSpace{D}}, pathfinder::LPAStar{D}) where {D}
    return Tuple(
        rand(
            abmrng(model),
            filter(x -> pathfinder.walkmap[x], CartesianIndices(abmspace(model).stored_ids)),
        ),
    )
end

"""
    Pathfinding.penaltymap(pathfinder::LPAStar)

Ό,τι και για AStar.
"""
function penaltymap(pathfinder::LPAStar)
    if pathfinder.cost_metric isa PenaltyMap
        return pathfinder.cost_metric.pmap
    elseif pathfinder.cost_metric isa AbsolutePenaltyMap
        return pathfinder.cost_metric.pmap
    else
        return nothing
    end
end

"""
    Agents.remove_agent!(agent, model, pathfinder::LPAStar)

Σβήνει και τα path δεδομένα του agent.
"""
function Agents.remove_agent!(agent::AbstractAgent, model::ABM, pathfinder::LPAStar)
    delete!(pathfinder.agent_paths, Agents.getid(agent))
    Agents.remove_agent_from_container!(agent, model)
    Agents.remove_agent_from_space!(agent, model)
end

Base.isempty(id::Int, pathfinder::LPAStar) =
    !haskey(pathfinder.agent_paths, id) || isempty(pathfinder.agent_paths[id])

Agents.is_stationary(agent::AbstractAgent, pathfinder::LPAStar) =
    isempty(Agents.getid(agent), pathfinder)
