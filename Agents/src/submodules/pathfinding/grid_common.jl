# Shared definitions for grid-based pathfinders (A*; D* Lite; Anytime D*).
# Neighborhoods, path type, and neighbor iteration are common across these algorithms.

"""
    Path{D,T}
Alias of `MutableLinkedList{NTuple{D,T}}`. Used to represent the path to be
taken by an agent in a `D` dimensional space.
"""
const Path{D,T} = MutableLinkedList{NTuple{D,T}}

"""
    moore_neighborhood(D)
Return a vector of `CartesianIndex{D}` for the Moore (3^D - 1) neighborhood in `D` dimensions
(i.e. all adjacent cells including diagonals, excluding the center).
"""
moore_neighborhood(D) = [
    CartesianIndex(a)
    for a in Iterators.product([-1:1 for _ in 1:D]...) if a != Tuple(zeros(Int, D))
]

"""
    vonneumann_neighborhood(D)
Return a vector of `CartesianIndex{D}` for the von Neumann neighborhood in `D` dimensions
(i.e. orthogonal neighbors only, L1 norm equal to 1).
"""
function vonneumann_neighborhood(D)
    hypercube = CartesianIndices((repeat([-1:1], D)...,))
    [β for β ∈ hypercube if LinearAlgebra.norm(β.I) == 1]
end

# Neighbor iteration for any GridPathfinder (periodic / non-periodic / mixed)

@inline get_neighbors(cur, pathfinder::GridPathfinder{D,true}) where {D} =
    (mod1.(cur .+ β.I, size(pathfinder.walkmap)) for β in pathfinder.neighborhood)

@inline get_neighbors(cur, pathfinder::GridPathfinder{D,false}) where {D} =
    (cur .+ β.I for β in pathfinder.neighborhood)

@inline function get_neighbors(cur, pathfinder::GridPathfinder{D,P}) where {D,P}
    s = size(pathfinder.walkmap)
    (
        ntuple(i -> P[i] ? mod1(cur[i] + β[i], s[i]) : cur[i] + β[i], D)
        for β in pathfinder.neighborhood
    )
end

"""
    inbounds(n, pathfinder::GridPathfinder)
Return true if position `n` is within the grid and walkable according to `pathfinder`.
"""
@inline inbounds(n, pathfinder::GridPathfinder) =
    all(1 .<= n .<= size(pathfinder.walkmap)) && pathfinder.walkmap[n...]

"""
    walkable_neighbors(cur, pathfinder::GridPathfinder{D,P,M})
Return a vector of neighbor positions of `cur` that are in bounds and walkable.
"""
function walkable_neighbors(cur::Dims{D}, pathfinder::GridPathfinder{D,P,M}) where {D,P,M}
    s = size(pathfinder.walkmap)
    res = Dims{D}[]
    for n in get_neighbors(cur, pathfinder)
        if all(1 .<= n .<= s) && pathfinder.walkmap[n...]
            push!(res, n)
        end
    end
    return res
end
