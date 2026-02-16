module FlowField

# Flow-field style pathfinding for crowd motion, inspired by the JS prototype.
# Designed so it can be used as an "environment" step with Agents.jl.
#
# You are expected to:
#   1. Build a FlowFieldState for your model's continuous space.
#   2. Call update_flowfield!(...) each model-step.
#   3. For each agent, look up velocity via lookup_velocity and update the agent.
#
# Notes:
#   - Uses 2D vectors via StaticArrays.SVector.
#   - Uses DataStructures.PriorityQueue for the fast marching step.

using StaticArrays
using LinearAlgebra
using DataStructures: PriorityQueue, dequeue!
using ..Agents: allagents

export FlowFieldParams,
       GridGoal,
       FlowFieldState,
       init_flowfield,
       update_flowfield!,
       lookup_velocity,
       enforce_pairwise_distance!,
       update_penalty_map!

# Direction indices: 1-based (Julia) to mirror JS 0=E,1=S,2=W,3=N
const EAST  = 1
const SOUTH = 2
const WEST  = 3
const NORTH = 4

"Hyperparameters for the flow-field cost model."
struct FlowFieldParams
    alpha::Float64
    beta::Float64
    gamma::Float64
    λ::Float64
    threshold::Float64
end

"Rectangular goal region in grid coordinates (1-based indices)."
struct GridGoal
    minx::Int
    maxx::Int
    miny::Int
    maxy::Int
end

"""
State of the flow field on a regular grid.

nx, ny      : number of cells in x and y
cellsize    : cell size in world units (e.g. pixels, meters)
params      : FlowFieldParams
penalty_map : optional penalty map (Array{Float64,2}) for cost calculation
walkmap     : optional walkability map (BitArray{2}) - false means non-walkable

All per-cell arrays are length nx*ny, flattened row-major:
  index(i,j) = (j-1)*nx + i,  with i∈[1,nx], j∈[1,ny].

density     : scalar density per cell
avg_vel     : average agent velocity per cell (density-weighted)
speeds      : 4×N directional speeds (E,S,W,N) per cell
costs       : 4×N directional costs per cell
potential   : scalar potential φ per cell
gradphi     : 4×N directional potential differences (φ_neighbor - φ)
total_vel   : resulting flow velocity per cell
state       : per-cell state for fast marching: 0 known, 1 candidate, 2 far
"""
mutable struct FlowFieldState
    nx::Int
    ny::Int
    cellsize::Float64
    params::FlowFieldParams
    penalty_map::Union{Nothing, Matrix{Float64}}
    walkmap::Union{Nothing, BitMatrix}

    density::Vector{Float64}
    avg_vel::Vector{SVector{2,Float64}}

    speeds::Matrix{Float64}     # (4, N), directions E,S,W,N
    costs::Matrix{Float64}      # (4, N)
    potential::Vector{Float64}
    gradphi::Matrix{Float64}    # (4, N)
    total_vel::Vector{SVector{2,Float64}}
    state::Vector{Int8}
end

# ----------------------------------------------------------
# Indexing helpers
# ----------------------------------------------------------

@inline function index(ff::FlowFieldState, ix::Int, iy::Int)
    @inbounds (iy - 1) * ff.nx + ix
end

@inline function inbounds(ff::FlowFieldState, ix::Int, iy::Int)
    1 <= ix <= ff.nx && 1 <= iy <= ff.ny
end

@inline function ind2xy(ff::FlowFieldState, idx::Int)
    @inbounds begin
        ix = (idx - 1) % ff.nx + 1
        iy = (idx - 1) ÷ ff.nx + 1
    end
    return ix, iy
end

# ----------------------------------------------------------
# Initialization / clearing
# ----------------------------------------------------------

"""
Create a new FlowFieldState for a rectangular continuous domain.

width, height : total world dimensions (same units as agent positions)
cellsize      : grid cell size
params        : FlowFieldParams
penalty_map   : optional penalty map (Array{Float64,2}) for cost calculation
walkmap       : optional walkability map (BitArray{2}) - false means non-walkable

nx and ny are computed such that nx*cellsize ≥ width, ny*cellsize ≥ height.
If penalty_map or walkmap are provided, their dimensions must match (nx, ny).
"""
function init_flowfield(width::Real, height::Real, cellsize::Real,
                        params::FlowFieldParams;
                        penalty_map::Union{Nothing, AbstractMatrix{Float64}} = nothing,
                        walkmap::Union{Nothing, BitMatrix} = nothing)
    nx = Int(floor(width / cellsize)) + 1
    ny = Int(floor(height / cellsize)) + 1
    N  = nx * ny

    # Validate penalty_map and walkmap dimensions if provided
    # Note: penalty_map and walkmap should be stored as (ny, nx) = (height, width) in matrix notation
    # to match Julia's [row, column] indexing where row = iy (y coordinate) and column = ix (x coordinate)
    if penalty_map !== nothing
        @assert size(penalty_map) == (ny, nx) "penalty_map size $(size(penalty_map)) must match flow field grid size (ny=$ny, nx=$nx) as (height, width)"
    end
    if walkmap !== nothing
        @assert size(walkmap) == (ny, nx) "walkmap size $(size(walkmap)) must match flow field grid size (ny=$ny, nx=$nx) as (height, width)"
    end

    density   = zeros(Float64, N)
    avg_vel   = [SVector{2,Float64}(0.0, 0.0) for _ in 1:N]
    speeds    = zeros(Float64, 4, N)
    costs     = zeros(Float64, 4, N)
    potential = fill(Inf, N)
    gradphi   = fill(Inf, 4, N)
    total_vel = [SVector{2,Float64}(0.0, 0.0) for _ in 1:N]
    state     = fill(Int8(2), N)  # 2 = far/unknown

    FlowFieldState(nx, ny, float(cellsize), params, penalty_map, walkmap,
                   density, avg_vel,
                   speeds, costs, potential, gradphi, total_vel, state)
end

"Reset all per-cell quantities (called at each update)."
function clear!(ff::FlowFieldState)
    N = ff.nx * ff.ny
    fill!(ff.density, 0.0)
    fill!(ff.potential, Inf)
    fill!(ff.state, Int8(2))
    for i in 1:N
        ff.avg_vel[i]   = SVector(0.0, 0.0)
        ff.total_vel[i] = SVector(0.0, 0.0)
    end
    fill!(ff.speeds, 0.0)
    fill!(ff.costs, 0.0)
    fill!(ff.gradphi, Inf)
    return ff
end

# ----------------------------------------------------------
# Density and average velocity from agents
# ----------------------------------------------------------

"Clamp into [a,b]."
@inline function clampf(x::Float64, a::Float64, b::Float64)
    x < a && return a
    x > b && return b
    return x
end

"Small positive to avoid zero/negative speeds."
@inline small(x) = x <= 0.0 ? 1e-2 : x

"""
Compute density and average velocity field from agents.

model   : Agents.jl-like model, or anything where `allagents(model)` works.
get_pos : function a -> SVector{2,Float64}, agent position
get_vel : function a -> SVector{2,Float64}, agent velocity
"""
function calc_densities!(ff::FlowFieldState, model;
                         get_pos = a -> SVector{2,Float64}(a.pos),
                         get_vel = a -> SVector{2,Float64}(0.0, 0.0))
    λ = ff.params.λ
    cs = ff.cellsize

    for a in allagents(model)
        pos = get_pos(a)
        vel = get_vel(a)

        # Grid index of "closest" cell center
        gx = clamp(Int(floor(pos[1] / cs)) + 1, 1, ff.nx)
        gy = clamp(Int(floor(pos[2] / cs)) + 1, 1, ff.ny)
        cx = (gx - 0.5) * cs
        cy = (gy - 0.5) * cs
        closest = SVector(cx, cy)

        δ = (pos - closest) / cs  # roughly in [-0.5,0.5]^2

        # A (gx, gy)
        if inbounds(ff, gx, gy)
            wA = (min(1 - abs(δ[1]), 1 - abs(δ[2])))^λ
            if wA > 0
                idxA = index(ff, gx, gy)
                ff.density[idxA] += wA
                ff.avg_vel[idxA] += vel * wA
            end
        end

        # B (gx+1, gy)
        if inbounds(ff, gx + 1, gy)
            wB = (min(abs(δ[1]), 1 - abs(δ[2])))^λ
            if wB > 0
                idxB = index(ff, gx + 1, gy)
                ff.density[idxB] += wB
                ff.avg_vel[idxB] += vel * wB
            end
        end

        # C (gx+1, gy-1)
        if inbounds(ff, gx + 1, gy - 1)
            wC = (min(abs(δ[1]), abs(δ[2])))^λ
            if wC > 0
                idxC = index(ff, gx + 1, gy - 1)
                ff.density[idxC] += wC
                ff.avg_vel[idxC] += vel * wC
            end
        end

        # D (gx, gy-1)
        if inbounds(ff, gx, gy - 1)
            wD = (min(1 - abs(δ[1]), abs(δ[2])))^λ
            if wD > 0
                idxD = index(ff, gx, gy - 1)
                ff.density[idxD] += wD
                ff.avg_vel[idxD] += vel * wD
            end
        end
    end

    # Normalize avg_vel by density
    for i in eachindex(ff.avg_vel)
        d = ff.density[i]
        if d > 0
            ff.avg_vel[i] /= d
        end
    end

    return ff
end

# ----------------------------------------------------------
# Directional speeds and costs
# ----------------------------------------------------------

"""
Compute directional speeds and costs from density and avg_vel.

Speeds:
  - Topographic speed speedT = 1.0 at low density.
  - Blend with flow velocity based on density in [min, max].

Costs (per direction d):
  cost = (α * speed_d + β + γ * discomfort_d) / speed_d

Discomfort is computed from penalty_map if provided, otherwise 0.
Non-walkable cells (from walkmap) get infinite cost.
"""
function calc_costs!(ff::FlowFieldState; min_density = 0.0, max_density = 2.0)
    α = ff.params.alpha
    β = ff.params.beta
    γ = ff.params.gamma

    speedT = 1.0
    nx, ny = ff.nx, ff.ny

    for iy in 1:ny
        for ix in 1:nx
            idx = index(ff, ix, iy)

            # Check if current cell is walkable
            # Note: ix is x (column), iy is y (row)
            # Julia matrices use [row, column] = [iy, ix]
            is_walkable = (ff.walkmap === nothing) || ff.walkmap[iy, ix]

            # Neighbor indices for directional speeds & densities
            # East neighbor cell (ix+1, iy)
            if ix < nx
                idxE = index(ff, ix + 1, iy)
                is_walkable_E = (ff.walkmap === nothing) || ff.walkmap[iy, ix + 1]
                if !is_walkable_E
                    sE = 0.0  # Non-walkable
                else
                    flowE = clampf(small(ff.avg_vel[idxE][1]), 0.0, 1.0)
                    ρE    = ff.density[idxE]
                    wE    = clampf((ρE - min_density) / (max_density - min_density), 0.0, 1.0)
                    sE    = speedT + wE * (flowE - speedT)
                end
            else
                sE = speedT
            end

            # South neighbor cell (ix, iy+1)
            if iy < ny
                idxS = index(ff, ix, iy + 1)
                is_walkable_S = (ff.walkmap === nothing) || ff.walkmap[iy + 1, ix]
                if !is_walkable_S
                    sS = 0.0  # Non-walkable
                else
                    flowS = clampf(small(ff.avg_vel[idxS][2]), 0.0, 1.0)
                    ρS    = ff.density[idxS]
                    wS    = clampf((ρS - min_density) / (max_density - min_density), 0.0, 1.0)
                    sS    = speedT + wS * (flowS - speedT)
                end
            else
                sS = speedT
            end

            # West neighbor cell (ix-1, iy)
            if ix > 1
                idxW = index(ff, ix - 1, iy)
                is_walkable_W = (ff.walkmap === nothing) || ff.walkmap[iy, ix - 1]
                if !is_walkable_W
                    sW = 0.0  # Non-walkable
                else
                    flowW = clampf(small(-ff.avg_vel[idxW][1]), 0.0, 1.0)
                    ρW    = ff.density[idxW]
                    wW    = clampf((ρW - min_density) / (max_density - min_density), 0.0, 1.0)
                    sW    = speedT + wW * (flowW - speedT)
                end
            else
                sW = speedT
            end

            # North neighbor cell (ix, iy-1)
            if iy > 1
                idxN = index(ff, ix, iy - 1)
                is_walkable_N = (ff.walkmap === nothing) || ff.walkmap[iy - 1, ix]
                if !is_walkable_N
                    sN = 0.0  # Non-walkable
                else
                    flowN = clampf(small(-ff.avg_vel[idxN][2]), 0.0, 1.0)
                    ρN    = ff.density[idxN]
                    wN    = clampf((ρN - min_density) / (max_density - min_density), 0.0, 1.0)
                    sN    = speedT + wN * (flowN - speedT)
                end
            else
                sN = speedT
            end

            # Store speeds (E,S,W,N)
            ff.speeds[EAST,  idx] = sE
            ff.speeds[SOUTH, idx] = sS
            ff.speeds[WEST,  idx] = sW
            ff.speeds[NORTH, idx] = sN

            # Compute discomfort from penalty_map (if provided)
            # Use absolute penalty map style: sum of absolute values
            # Note: ix is x (column), iy is y (row)
            # Julia matrices use [row, column] = [iy, ix]
            if ff.penalty_map !== nothing
                penalty_here = abs(ff.penalty_map[iy, ix])
                disE = (ix < nx) ? (penalty_here + abs(ff.penalty_map[iy, ix + 1])) / 2.0 : penalty_here
                disS = (iy < ny) ? (penalty_here + abs(ff.penalty_map[iy + 1, ix])) / 2.0 : penalty_here
                disW = (ix > 1)  ? (penalty_here + abs(ff.penalty_map[iy, ix - 1])) / 2.0 : penalty_here
                disN = (iy > 1)  ? (penalty_here + abs(ff.penalty_map[iy - 1, ix])) / 2.0 : penalty_here
            else
                disE = 0.0
                disS = 0.0
                disW = 0.0
                disN = 0.0
            end

            # Costs: infinite for non-walkable directions, otherwise use formula
            ff.costs[EAST,  idx] = (!is_walkable || sE == 0.0) ? Inf : (α * sE + β + γ * disE) / sE
            ff.costs[SOUTH, idx] = (!is_walkable || sS == 0.0) ? Inf : (α * sS + β + γ * disS) / sS
            ff.costs[WEST,  idx] = (!is_walkable || sW == 0.0) ? Inf : (α * sW + β + γ * disW) / sW
            ff.costs[NORTH, idx] = (!is_walkable || sN == 0.0) ? Inf : (α * sN + β + γ * disN) / sN
        end
    end

    return ff
end

# ----------------------------------------------------------
# Fast marching: potential field
# ----------------------------------------------------------

"Return (φE, φS, φW, φN) for neighbors of idx, Inf if out-of-bounds."
function neighboring_potential(ff::FlowFieldState, idx::Int)
    ix, iy = ind2xy(ff, idx)
    φE = (ix < ff.nx) ? ff.potential[index(ff, ix + 1, iy)] : Inf
    φS = (iy < ff.ny) ? ff.potential[index(ff, ix, iy + 1)] : Inf
    φW = (ix > 1)     ? ff.potential[index(ff, ix - 1, iy)] : Inf
    φN = (iy > 1)     ? ff.potential[index(ff, ix, iy - 1)] : Inf
    return φE, φS, φW, φN
end

"Return neighbor indices of idx (N,S,E,W) in-bounds."
function neighbors(ff::FlowFieldState, idx::Int)
    ix, iy = ind2xy(ff, idx)
    res = Int[]
    iy > 1     && push!(res, index(ff, ix, iy - 1))   # North
    iy < ff.ny && push!(res, index(ff, ix, iy + 1))   # South
    ix < ff.nx && push!(res, index(ff, ix + 1, iy))   # East
    ix > 1     && push!(res, index(ff, ix - 1, iy))   # West
    return res
end

"Safe quadratic solver using the '+' branch."
@inline function solve_quadratic(a::Float64, b::Float64, c::Float64)
    disc = b*b - 4a*c
    disc < 0 && return Inf
    return (-b + sqrt(disc)) / (2a)
end

"Compute updated local potential at cell idx from its neighbors and costs."
function compute_local_phi(ff::FlowFieldState, idx::Int)
    φE, φS, φW, φN = neighboring_potential(ff, idx)
    cE = ff.costs[EAST,  idx]
    cS = ff.costs[SOUTH, idx]
    cW = ff.costs[WEST,  idx]
    cN = ff.costs[NORTH, idx]

    # choose min-horizontal and min-vertical directions
    # based on (φ + cost)
    hor = (cE + φE > cW + φW) ? WEST : EAST
    ver = (cS + φS > cN + φN) ? NORTH : SOUTH

    φ = Inf

    getφ(d) = d === EAST  ? φE :
              d === SOUTH ? φS :
              d === WEST  ? φW :
                            φN

    getc(d) = d === EAST  ? cE :
              d === SOUTH ? cS :
              d === WEST  ? cW :
                            cN

    φh = getφ(hor)
    φv = getφ(ver)
    ch = getc(hor)
    cv = getc(ver)

    if isfinite(φh) && isfinite(φv)
        ah = 1 / ch
        av = 1 / cv
        a = ah^2 + av^2
        b = -2 * (φh * ah^2 + φv * av^2)
        c = (φh / ch)^2 + (φv / cv)^2 - 1
        φ = solve_quadratic(a, b, c)

        # fallback if discriminant < 0 (solve_quadratic returns Inf)
        if !isfinite(φ)
            val = min(φh, φv)
            if isfinite(val)
                φ = (φh <= φv) ? (val + ch) : (val + cv)
            end
        end
    elseif isfinite(φh) && !isfinite(φv)
        ah = 1 / ch
        a = ah^2
        b = -2 * φh * ah^2
        c = (φh / ch)^2 - 1
        φ = solve_quadratic(a, b, c)
        if !isfinite(φ)
            φ = φh + ch
        end
    elseif !isfinite(φh) && isfinite(φv)
        av = 1 / cv
        a = av^2
        b = -2 * φv * av^2
        c = (φv / cv)^2 - 1
        φ = solve_quadratic(a, b, c)
        if !isfinite(φ)
            φ = φv + cv
        end
    else
        # All neighbors are Inf: leave as Inf
        φ = Inf
    end

    return φ
end

"""
Compute potential field φ over the grid using a fast marching–like algorithm.

Goal cells (in grid coords) are initialized with φ=0 and state=0 (known).
All other cells start at φ=Inf, state=2 (far).
"""
function calculate_potential!(ff::FlowFieldState, goal::GridGoal)
    nx, ny = ff.nx, ff.ny

    # Clear potentials and state (in case not already cleared)
    fill!(ff.potential, Inf)
    fill!(ff.state, Int8(2))

    pq = PriorityQueue{Int,Float64}()

    # Initialize goal region (only walkable cells)
    for iy in goal.miny:goal.maxy, ix in goal.minx:goal.maxx
        if inbounds(ff, ix, iy)
            # Only set goal if cell is walkable (or no walkmap provided)
            # Note: ix is x (column), iy is y (row), Julia uses [row, column] = [iy, ix]
            is_walkable = (ff.walkmap === nothing) || ff.walkmap[iy, ix]
            if is_walkable
                idx = index(ff, ix, iy)
                ff.potential[idx] = 0.0
                ff.state[idx]     = 0
                pq[idx] = 0.0
            end
        end
    end

    # Helper: update neighbors of a given cell
    function update_neighbors!(idx::Int)
        for nb in neighbors(ff, idx)
            ff.state[nb] == 0 && continue  # already accepted
            φ = compute_local_phi(ff, nb)
            if φ < ff.potential[nb]
                ff.potential[nb] = φ
                if ff.state[nb] == 2
                    ff.state[nb] = 1  # candidate
                end
                pq[nb] = φ
            end
        end
    end

    # First wave from the goal cells (only walkable ones)
    for iy in goal.miny:goal.maxy, ix in goal.minx:goal.maxx
        if inbounds(ff, ix, iy)
            is_walkable = (ff.walkmap === nothing) || ff.walkmap[iy, ix]
            if is_walkable
                update_neighbors!(index(ff, ix, iy))
            end
        end
    end

    # Main fast marching loop
    while !isempty(pq)
        pair = peek(pq)  # get min φ - returns Pair{Int,Float64}
        idx = pair.first
        φ = pair.second
        dequeue!(pq)  # remove from queue
        if ff.state[idx] == 0
            continue
        end
        ff.state[idx] = 0
        update_neighbors!(idx)
    end

    return ff
end

# ----------------------------------------------------------
# Gradients and flow velocities
# ----------------------------------------------------------

"""
Compute potential gradients and resulting total_vel per cell.

total_vel is directed along -∇φ, with magnitude interpolated from the
directional speeds on the four faces.
"""
function calculate_gradients!(ff::FlowFieldState)
    nxny = ff.nx * ff.ny
    τ = π / 2

    for idx in 1:nxny
        φ = ff.potential[idx]
        if !isfinite(φ)
            ff.total_vel[idx] = SVector(0.0, 0.0)
            ff.gradphi[:, idx] .= Inf
            continue
        end

        φE, φS, φW, φN = neighboring_potential(ff, idx)

        gE = isfinite(φE) ? φE - φ : Inf
        gS = isfinite(φS) ? φS - φ : Inf
        gW = isfinite(φW) ? φW - φ : Inf
        gN = isfinite(φN) ? φN - φ : Inf

        ff.gradphi[EAST,  idx] = gE
        ff.gradphi[SOUTH, idx] = gS
        ff.gradphi[WEST,  idx] = gW
        ff.gradphi[NORTH, idx] = gN

        # dx
        dx = if isfinite(gE) && isfinite(gW)
            (gE - gW) / 2
        elseif isfinite(gE)
            gE
        elseif isfinite(gW)
            -gW
        else
            0.0
        end

        # dy
        dy = if isfinite(gS) && isfinite(gN)
            (gS - gN) / 2
        elseif isfinite(gS)
            gS
        elseif isfinite(gN)
            -gN
        else
            0.0
        end

        if dx == 0.0 && dy == 0.0
            ff.total_vel[idx] = SVector(0.0, 0.0)
            continue
        end

        g = SVector(dx, dy)
        ng = g / norm(g)

        # Interpolate scalar speed from directional speeds
        sE = ff.speeds[EAST,  idx]
        sS = ff.speeds[SOUTH, idx]
        sW = ff.speeds[WEST,  idx]
        sN = ff.speeds[NORTH, idx]

        θ = atan(ng[2], ng[1])
        speed =
            if 0.0 < θ <= τ
                # between east and north
                (1 - θ/τ) * sE + (θ/τ) * sN
            elseif τ < θ <= π
                # between north and west
                (1 - (θ - τ)/τ) * sN + ((θ - τ)/τ) * sW
            elseif -π < θ <= -τ
                # between west and south
                (1 - (θ + π)/τ) * sW + ((θ + π)/τ) * sS
            elseif -τ < θ <= 0.0
                # between south and east
                (1 - (θ + τ)/τ) * sS + ((θ + τ)/τ) * sE
            else
                sE
            end

        ff.total_vel[idx] = -ng * speed
    end

    return ff
end

# ----------------------------------------------------------
# Lookup + interpolation for agent motion
# ----------------------------------------------------------

"""
Bilinear interpolation of the flow velocity at a continuous position pos.

pos is in world coordinates (same units as cellsize). The grid is assumed to
cover [0, nx*cellsize]×[0, ny*cellsize].

Returns SVector{2,Float64}.
"""
function lookup_velocity(ff::FlowFieldState, pos::SVector{2,Float64})
    cs = ff.cellsize
    nx, ny = ff.nx, ff.ny

    # Continuous "center indices" (centers at (i-0.5)*cs)
    gx = pos[1] / cs + 0.5
    gy = pos[2] / cs + 0.5

    # Clamp to interior so ix+1, iy+1 exist
    ix = clamp(Int(floor(gx)), 1, nx - 1)
    iy = clamp(Int(floor(gy)), 1, ny - 1)

    tx = clampf(gx - ix, 0.0, 1.0)
    ty = clampf(gy - iy, 0.0, 1.0)

    idx11 = index(ff, ix,     iy)
    idx21 = index(ff, ix + 1, iy)
    idx12 = index(ff, ix,     iy + 1)
    idx22 = index(ff, ix + 1, iy + 1)

    v11 = ff.total_vel[idx11]
    v21 = ff.total_vel[idx21]
    v12 = ff.total_vel[idx12]
    v22 = ff.total_vel[idx22]

    # bilinear interpolation
    v1 = (1 - tx) * v11 + tx * v21
    v2 = (1 - tx) * v12 + tx * v22
    return (1 - ty) * v1 + ty * v2
end

# ----------------------------------------------------------
# Pairwise distance enforcement (optional)
# ----------------------------------------------------------

"""
Enforce a minimum Euclidean distance between all agent pairs.

model     : Agents.jl-like model
threshold : minimum allowed distance
get_pos   : function a -> SVector{2,Float64}
set_pos!  : function (a, newpos::SVector{2,Float64}) -> nothing
"""
function enforce_pairwise_distance!(model, threshold::Real;
                                    get_pos = a -> SVector{2,Float64}(a.pos),
                                    set_pos! = (a, p) -> (a.pos = Tuple(p)))
    ags = collect(allagents(model))
    n = length(ags)
    thr = float(threshold)

    for i in 1:n-1
        a1 = ags[i]
        p1 = get_pos(a1)
        for j in i+1:n
            a2 = ags[j]
            p2 = get_pos(a2)
            δ = p1 - p2
            dist = norm(δ)
            if dist < thr && dist > 0
                diff = (thr - dist) / 2
                dir  = δ / dist
                new1 = p1 + dir * diff
                new2 = p2 - dir * diff
                set_pos!(a1, new1)
                set_pos!(a2, new2)
                p1 = new1  # update cached p1 for further pairs
            end
        end
    end

    return model
end

# ----------------------------------------------------------
# High-level update
# ----------------------------------------------------------

"""
Full flow-field update for one simulation step.

ff    : FlowFieldState
model : Agents.jl-like model (used only for density/avg_vel)
goal  : GridGoal in grid coordinates (see docstring)
get_pos, get_vel : accessors for agent position and velocity

This function:
  1. Clears fields
  2. Recomputes density + average velocity from agents
  3. Recomputes speeds + costs (incorporating penalty_map and walkmap)
  4. Recomputes potential on the grid
  5. Recomputes gradients and total_vel

After calling this, you typically:
  - For each agent, read v = lookup_velocity(ff, pos),
  - Update agent's velocity and position accordingly.
"""
function update_flowfield!(ff::FlowFieldState, model, goal::GridGoal;
                           get_pos = a -> SVector{2,Float64}(a.pos),
                           get_vel = a -> SVector{2,Float64}(0.0, 0.0))
    clear!(ff)
    calc_densities!(ff, model; get_pos = get_pos, get_vel = get_vel)
    calc_costs!(ff)
    calculate_potential!(ff, goal)
    calculate_gradients!(ff)
    return ff
end

"""
Update the penalty map in-place (useful for dynamic environments).
Note: penalty_map should be (ny, nx) = (height, width) in matrix notation.
"""
function update_penalty_map!(ff::FlowFieldState, new_penalty_map::AbstractMatrix{Float64})
    @assert size(new_penalty_map) == (ff.ny, ff.nx) "penalty_map size $(size(new_penalty_map)) must match flow field grid size (ny=$(ff.ny), nx=$(ff.nx)) as (height, width)"
    if ff.penalty_map === nothing
        ff.penalty_map = copy(new_penalty_map)
    else
        ff.penalty_map .= new_penalty_map
    end
    return ff
end

end # module
