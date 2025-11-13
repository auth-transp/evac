using DataStructures
using DataStructures: PriorityQueue, enqueue!, dequeue!, peek, haskey, delete!
using Agents
using LinearAlgebra

# ---------------------------
# D*-Lite style implementation, kept as Dlite name (replace earlier Dlite)
# Requires: using DataStructures
# ---------------------------

# Path alias already defined earlier in your module:
const Path{D,T} = MutableLinkedList{NTuple{D,T}}


moore_neighborhood(D) = [
    CartesianIndex(a)
    for a in Iterators.product([-1:1 for _ in 1:D]...)
    if a != ntuple(_->0, D)
]

function vonneumann_neighborhood(D)
    hypercube = CartesianIndices((repeat([-1:1], D)...,))
    [β for β in hypercube if LinearAlgebra.norm(β.I) == 1]
end


mutable struct Dlite{D,P,M,T,C<:CostMetric{D}} <: GridPathfinder{D,P,M}
    agent_paths::Dict{Int,Path{D,T}}
    dims::NTuple{D,T}
    neighborhood::Vector{CartesianIndex{D}}
    admissibility::Float64
    walkmap::BitArray{D}
    cost_metric::C

    # --- D* Lite state ---
    g::Dict{NTuple{D,Int},Float64}
    rhs::Dict{NTuple{D,Int},Float64}
    km::Float64
    open::PriorityQueue{NTuple{D,Int},Tuple{Float64,Float64}}  # (k1,k2)
    lookup::Dict{NTuple{D,Int},Tuple{Float64,Float64}}         # mirror of priorities

    # book-keeping για main loop
    last_start::Union{Nothing,NTuple{D,Int}}
    last_goal::Union{Nothing,NTuple{D,Int}}

    function Dlite{D,P,M,T,C}(
        agent_paths::Dict{Int,Path{D,T}},
        dims::NTuple{D,T},
        neighborhood::Vector{CartesianIndex{D}},
        admissibility::Float64,
        walkmap::BitArray{D},
        cost_metric::C,
    ) where {D,P,M,T,C}
        @assert all(dims .> 0)
        T <: Integer && @assert size(walkmap) == dims
        new(
            agent_paths, dims, neighborhood, admissibility, walkmap, cost_metric,
            Dict{NTuple{D,Int},Float64}(),
            Dict{NTuple{D,Int},Float64}(),
            0.0,
            PriorityQueue{NTuple{D,Int},Tuple{Float64,Float64}}(),
            Dict{NTuple{D,Int},Tuple{Float64,Float64}}(),
            nothing, nothing
        )
    end
end

# ---------- Constructors (ίδιο pattern με A*) ----------
function Dlite(
    dims::NTuple{D,T};
    periodic::Union{Bool,NTuple{D,Bool}} = false,
    diagonal_movement::Bool = true,
    admissibility::Float64 = 0.0,
    walkmap::BitArray{D} = trues(dims),
    cost_metric::CostMetric{D} = DirectDistance{D}(),
) where {D,T}
    neighborhood = diagonal_movement ? moore_neighborhood(D) : vonneumann_neighborhood(D)
    return Dlite{D,periodic,diagonal_movement,T,typeof(cost_metric)}(
        Dict{Int,Path{D,T}}(),
        dims, neighborhood, admissibility, walkmap, cost_metric
    )
end

Dlite(
    space::GridSpace{D,periodic};
    diagonal_movement::Bool = true,
    admissibility::Float64 = 0.0,
    walkmap::BitArray{D} = trues(spacesize(space)),
    cost_metric::CostMetric{D} = DirectDistance{D}(),
) where {D,periodic} =
    Dlite(size(space); periodic, diagonal_movement, admissibility, walkmap, cost_metric)

function Dlite(
    space::ContinuousSpace{D,periodic};
    walkmap::Union{BitArray{D},Nothing} = nothing,
    admissibility::Float64 = 0.0,
    cost_metric::CostMetric{D} = DirectDistance{D}(),
) where {D,periodic}
    @assert (walkmap isa BitArray{D}) || (cost_metric isa PenaltyMap)
    isnothing(walkmap) && (walkmap = BitArray(trues(size(cost_metric.pmap))))
    # χρησιμοποίησε ακέραιες διαστάσεις από τη walkmap
    dims_int = size(walkmap)::NTuple{D,Int}
    return Dlite(dims_int; periodic, diagonal_movement = true,
                 admissibility, walkmap, cost_metric)
end

Base.show(io::IO, pf::Dlite{D,P,M,T,C}) where {D,P,M,T,C} =
    print(io, "D*Lite in $D dimensions, ",
          (P==true ? "periodic, " : (P==false ? "" : "mixed periodicity, ")),
          (M ? "diagonal" : "orthogonal"),
          ", ϵ=$(pf.admissibility), metric=$(pf.cost_metric)")

# ---------- Βοηθητικά ----------
@inline function dl_walkable(pf::Dlite{D}, u::NTuple{D,Int}) where {D}
    all(1 .<= u .<= size(pf.walkmap)) && pf.walkmap[u...]
end

@inline function dl_neighbors(u::NTuple{D,Int}, pf::Dlite{D,true}) where {D}
    (mod1.(u .+ β.I, size(pf.walkmap)) for β in pf.neighborhood)
end
@inline function dl_neighbors(u::NTuple{D,Int}, pf::Dlite{D,false}) where {D}
    (u .+ β.I for β in pf.neighborhood)
end
@inline function dl_neighbors(u::NTuple{D,Int}, pf::Dlite{D,P}) where {D,P}
    s = size(pf.walkmap)
    ( ntuple(i -> P[i] ? mod1(u[i]+β[i], s[i]) : u[i]+β[i], D) for β in pf.neighborhood )
end

# Κόστος ακμής c(u,v): αν εμπόδιο -> ∞, αλλιώς delta_cost(...)
@inline function dl_c(pf::Dlite{D}, u::NTuple{D,Int}, v::NTuple{D,Int}) where {D}
    (dl_walkable(pf,u) && dl_walkable(pf,v)) || return Inf
    return delta_cost(pf, u, v)  # ήδη υπολογίζει metric+penalty
end

# ---------- CalculateKey ----------
@inline function dl_calculate_key(pf::Dlite{D}, s::NTuple{D,Int}, sstart::NTuple{D,Int}) where {D}
    g  = get(pf.g, s, Inf)
    rhs = get(pf.rhs, s, Inf)
    h  = delta_cost(pf, sstart, s)  # ευρετική στο ίδιο metric
    k1 = min(g, rhs) + h + pf.km
    k2 = min(g, rhs)
    return (k1, k2)
end

# ---------- Open set helpers ----------
@inline function dl_open_set!(pf::Dlite, s::NTuple, key::Tuple{Float64,Float64})
    pf.open[s] = key; pf.lookup[s] = key
end
@inline function dl_open_del!(pf::Dlite, s::NTuple)
    if haskey(pf.lookup, s)
        delete!(pf.open, s); delete!(pf.lookup, s)
    end
end

# ---------- Initialize ----------
function dl_initialize!(pf::Dlite{D}, sstart::NTuple{D,Int}, sgoal::NTuple{D,Int}) where {D}
    empty!(pf.open); empty!(pf.lookup); empty!(pf.g); empty!(pf.rhs)
    pf.km = 0.0
    pf.last_start = sstart
    pf.last_goal  = sgoal
    pf.rhs[sgoal] = 0.0
    dl_open_set!(pf, sgoal, dl_calculate_key(pf, sgoal, sstart))
    return nothing
end

# ---------- UpdateVertex (07–09) ----------
function dl_update_vertex!(pf::Dlite{D}, u::NTuple{D,Int}, sstart::NTuple{D,Int}) where {D}
    gu  = get(pf.g, u, Inf)
    rhsu = get(pf.rhs, u, Inf)
    if gu != rhsu && haskey(pf.lookup, u)
        dl_open_set!(pf, u, dl_calculate_key(pf, u, sstart))
    elseif gu != rhsu && !haskey(pf.lookup, u)
        dl_open_set!(pf, u, dl_calculate_key(pf, u, sstart))
    elseif gu == rhsu && haskey(pf.lookup, u)
        dl_open_del!(pf, u)
    end
    return nothing
end

# ---------- ComputeShortestPath (10–28) ----------
function dl_compute_shortest_path!(pf::Dlite{D}) where {D}
    sstart = pf.last_start::NTuple{D,Int}
    sgoal  = pf.last_goal::NTuple{D,Int}
    while !isempty(pf.open)
        # έλεγχος cond (10)
        u_top = peek(pf.open)
        topkey = pf.open[u_top]   # αυτό είναι το (k1, k2)        
        sstart_key = dl_calculate_key(pf, sstart, sstart)
        if (topkey[1] >= sstart_key[1] && topkey[2] >= sstart_key[2]) &&
           (get(pf.rhs, sstart, Inf) <= get(pf.g, sstart, Inf))
            break
        end

        u = dequeue!(pf.open)
        key_old = pf.lookup[u]
        delete!(pf.lookup, u)

        if (key_old[1] < key_new[1]) || (key_old[1] ≈ key_new[1] && key_old[2] < key_new[2])
            dl_open_set!(pf, u, key_new)       # (15)
            continue
        end

        gu  = get(pf.g, u, Inf)
        rhsu = get(pf.rhs, u, Inf)

        if gu > rhsu                            # (16–21)
            pf.g[u] = rhsu
            for s in dl_neighbors(u, pf)
                s == sgoal || (pf.rhs[s] = min(get(pf.rhs,s,Inf), dl_c(pf, s, u) + pf.g[u]))
                dl_update_vertex!(pf, s, sstart)
            end
        else                                    # (23–28)
            gold = gu
            pf.g[u] = Inf
            preds = [u]; append!(preds, collect(dl_neighbors(u, pf)))
            for s in preds
                if get(pf.rhs, s, Inf) == dl_c(pf, s, u) + gold
                    if s != sgoal
                        minrhs = Inf
                        for sp in dl_neighbors(s, pf)
                            minrhs = min(minrhs, dl_c(pf, s, sp) + get(pf.g, sp, Inf))
                        end
                        pf.rhs[s] = minrhs
                    end
                end
                dl_update_vertex!(pf, s, sstart)
            end
        end
    end
    return nothing
end

# ---------- “Main loop” helper (34–48) ----------
"""
    dl_move_and_replan!(pf::Dlite, start, goal; changed_edges=nothing)

Εκτελεί ένα βήμα “main loop”: υπολογίζει/επανυπολογίζει και επιστρέφει path (λίστα NTuple{D,Int}).
Αν `changed_edges` δοθεί σαν `Vector{Tuple{NTuple{D,Int},NTuple{D,Int}}}` (= ακμές που άλλαξαν),
γίνεται ενημέρωση (41–47) πριν το recompute.
"""
function dl_move_and_replan!(pf::Dlite{D},
                             start::NTuple{D,Int},
                             goal::NTuple{D,Int};
                             changed_edges::Union{Nothing,Vector{Tuple{NTuple{D,Int},NTuple{D,Int}}}}=nothing
) where {D}
    if pf.last_goal != goal || pf.last_start == nothing
        dl_initialize!(pf, start, goal)
    else
        pf.km += delta_cost(pf, pf.last_start::NTuple{D,Int}, start) # (38)
        pf.last_start = start
    end

    # apply changes (41–47)
    if changed_edges !== nothing && !isempty(changed_edges)
        for (u,v) in changed_edges
            c_old = dl_c(pf, u, v)  # εδώ: παλιό κόστος -> αν έχεις log, πέρασέ το αλλιώς
            c_new = dl_c(pf, u, v)  # με βάση pf.walkmap/pmap (ή ενημέρωσέ τα πριν)
            if c_old > c_new
                u != goal && (pf.rhs[u] = min(get(pf.rhs,u,Inf), dl_c(pf,u,v) + get(pf.g,v,Inf)))
            elseif get(pf.rhs,u,Inf) == c_old + get(pf.g,v,Inf)
                if u != goal
                    minrhs = Inf
                    for s_ in dl_neighbors(u, pf)
                        minrhs = min(minrhs, dl_c(pf, u, s_) + get(pf.g, s_, Inf))
                    end
                    pf.rhs[u] = minrhs
                end
            end
            dl_update_vertex!(pf, u, start)
        end
    end

    dl_compute_shortest_path!(pf)

    # greedy extraction από g-values
    path = NTuple{D,Int}[]
    cur = start
    push!(path, cur)
    maxsteps = prod(pf.dims) # ασφαλιστική δικλείδα
    steps = 0
    while cur != goal && steps ≤ maxsteps
        steps += 1
        best = nothing; bestval = Inf
        for nb in dl_neighbors(cur, pf)
            dl_walkable(pf, nb) || continue
            val = dl_c(pf, cur, nb) + get(pf.g, nb, Inf)
            if val < bestval
                bestval = val; best = nb
            end
        end
        best === nothing && break
        push!(path, best); cur = best
    end
    return path
end

# ---------- Συμβατότητες με plotting/helpers ----------
function penaltymap(pf::Dlite)
    pf.cost_metric isa PenaltyMap ? pf.cost_metric.pmap : nothing
end

Base.isempty(id::Int, pf::Dlite) =
    !haskey(pf.agent_paths, id) || isempty(pf.agent_paths[id])