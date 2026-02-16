#=
Social Force Model (SFM) for pedestrian motion, integrated with Agent-based Models.

Based on:
  - "The Integration of Agent-based Model and Social Force Model: Realistic Pedestrian
    Simulation" — the ABM supplies goals/routes and thus desired direction e_desired and
    desired speed v_desired; SFM implements the force-based motion.
  - Helbing & Molnár (1995) — force forms; here we use a simplified radial exponential
    repulsion instead of the ellipsoidal one.

The ABM is responsible for computing e_desired and v_desired (from pathfinding, flow
fields, or goals). This module takes those as input and returns accelerations or
updates vel/pos via the low-level force model.
=#

module SFM

using StaticArrays: SVector
using LinearAlgebra: norm
using ..Agents: nearby_ids

export SFMParams,
       sfm_acceleration,
       sfm_step!,
       repulsive_force_pedestrian,
       repulsive_force_wall

# --------------------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------------------

"""
    SFMParams(; A=2000.0, B=0.08, τ=0.5, r_ped=0.25, interaction_radius=3.0,
               A_wall=2000.0, B_wall=0.08)

Parameter set for the Social Force Model. Values are inspired by Helbing & Molnár (1995),
scaled for typical use.

- `A`, `B`: pedestrian–pedestrian repulsion: `A * exp((r_i + r_j - d_ij) / B)`
- `τ`: relaxation time for the desired-force term `(v_desired * e_desired - vel) / τ`
- `r_ped`: default pedestrian radius (used when agent has no `radius` or when not
  overridden by `get_radius`)
- `interaction_radius`: maximum distance for pedestrian–pedestrian repulsion (passed to
  `nearby_ids`)
- `A_wall`, `B_wall`: wall repulsion: `A_wall * exp((r_ped - d) / B_wall)`
"""
struct SFMParams
    A::Float64
    B::Float64
    τ::Float64
    r_ped::Float64
    interaction_radius::Float64
    A_wall::Float64
    B_wall::Float64
end

function SFMParams(;
    A::Real = 2000.0,
    B::Real = 0.08,
    τ::Real = 0.5,
    r_ped::Real = 0.25,
    interaction_radius::Real = 3.0,
    A_wall::Real = 2000.0,
    B_wall::Real = 0.08,
)
    SFMParams(
        Float64(A), Float64(B), Float64(τ), Float64(r_ped),
        Float64(interaction_radius), Float64(A_wall), Float64(B_wall),
    )
end

# --------------------------------------------------------------------------------------
# Pedestrian–pedestrian repulsion
# --------------------------------------------------------------------------------------

"""
    repulsive_force_pedestrian(agent_i, agent_j, A, B, r_i, r_j) → SVector{2,Float64}

Repulsive force on `agent_i` from `agent_j`:

    f = A * exp((r_i + r_j - d_ij) / B) * e_ij

where `d_ij = ‖pos_i - pos_j‖`, `e_ij = (pos_i - pos_j) / d_ij` (unit vector from j
toward i, i.e. away from j). Returns zero if `d_ij` is negligible (avoids division by
zero).
"""
function repulsive_force_pedestrian(agent_i, agent_j, A::Real, B::Real, r_i::Real, r_j::Real)
    pos_i = SVector{2,Float64}(agent_i.pos)
    pos_j = SVector{2,Float64}(agent_j.pos)
    r_ij = pos_i - pos_j
    d_ij = norm(r_ij)
    if d_ij < 1e-10
        return SVector(0.0, 0.0)
    end
    e_ij = r_ij / d_ij
    f_mag = A * exp((Float64(r_i) + Float64(r_j) - d_ij) / Float64(B))
    return f_mag * e_ij
end

# --------------------------------------------------------------------------------------
# Wall repulsion
# --------------------------------------------------------------------------------------

"""
    repulsive_force_wall(agent, walkmap, grid_dims, A_wall, B_wall, r_ped;
                        wall_distance_threshold=0.5) → SVector{2,Float64}

Total repulsive force on `agent` from non-walkable cells in `walkmap`.

- `walkmap`: matrix where `walkmap[j, i] == false` means cell (i, j) is a wall.
  Dimensions must match `grid_dims` as (ny, nx), i.e. `size(walkmap) == (grid_dims[2], grid_dims[1])`.
- `grid_dims`: `(nx, ny)` number of cells in x and y.

Agent `pos` is assumed in the same units as the grid: cell (i, j) in 1-based indexing
covers `[i, i+1) × [j, j+1)`, with center at `(i + 0.5, j + 0.5)`. For each
neighboring cell of `floor.(Int, agent.pos)` that is out-of-bounds or has
`!walkmap[nj, ni]`, the distance `d` from the agent to the cell center is computed;
if `d < wall_distance_threshold`, a force `A_wall * exp((r_ped - d) / B_wall) * e` is
added, with `e` pointing from the wall cell center to the agent.
"""
function repulsive_force_wall(
    agent,
    walkmap::AbstractMatrix,
    grid_dims::Tuple{Int,Int},
    A_wall::Real,
    B_wall::Real,
    r_ped::Real;
    wall_distance_threshold::Real = 0.5,
)
    nx, ny = grid_dims
    @assert size(walkmap) == (ny, nx) "walkmap size $(size(walkmap)) must match grid_dims as (ny=$ny, nx=$nx)"

    pos = SVector{2,Float64}(agent.pos)
    i0 = clamp(floor(Int, pos[1]), 1, nx)
    j0 = clamp(floor(Int, pos[2]), 1, ny)

    F = SVector(0.0, 0.0)
    Aw = Float64(A_wall)
    Bw = Float64(B_wall)
    rp = Float64(r_ped)
    thr = Float64(wall_distance_threshold)

    for nj in (j0 - 1):(j0 + 1), ni in (i0 - 1):(i0 + 1)
        (ni < 1 || ni > nx || nj < 1 || nj > ny) && (F += _one_wall_force(pos, ni, nj, Aw, Bw, rp, thr); continue)
        walkmap[nj, ni] && continue  # walkable, not a wall

        F += _one_wall_force(pos, ni, nj, Aw, Bw, rp, thr)
    end
    return F
end

function _one_wall_force(
    pos::SVector{2,Float64},
    ni::Int, nj::Int,
    A_wall::Float64, B_wall::Float64, r_ped::Float64,
    wall_distance_threshold::Float64,
)
    # Cell (ni, nj) in 1-based: center at (ni + 0.5, nj + 0.5)
    center = SVector(Float64(ni) + 0.5, Float64(nj) + 0.5)
    r = pos - center
    d = norm(r)
    if d < 1e-10 || d >= wall_distance_threshold
        return SVector(0.0, 0.0)
    end
    e = r / d
    f_mag = A_wall * exp((r_ped - d) / B_wall)
    return f_mag * e
end

# --------------------------------------------------------------------------------------
# SFM acceleration and step
# --------------------------------------------------------------------------------------

"""
    sfm_acceleration(agent, model, e_desired, v_desired;
                    params=SFMParams(), walkmap=nothing, grid_dims=nothing,
                    get_radius=nothing) → SVector{2,Float64}

Compute SFM acceleration for `agent`:

    a = F_desired + F_ped + F_wall

- `F_desired = (v_desired * e_desired - agent.vel) / τ`
- `F_ped`: sum of `repulsive_force_pedestrian(agent, model[id], ...)` over
  `nearby_ids(agent, model, params.interaction_radius)` (excludes the agent).
- `F_wall`: if `walkmap !== nothing`, `repulsive_force_wall(...)`; else zero. If
  `walkmap` is set and `grid_dims === nothing`, `grid_dims` is taken from
  `(size(walkmap, 2), size(walkmap, 1))`.

`e_desired` and `v_desired` must be supplied by the ABM (from pathfinding, flow field, or
goals). `get_radius(agent) → Float64` is optional; if `nothing`, `params.r_ped` is used
for all agents.
"""
function sfm_acceleration(
    agent,
    model,
    e_desired::Union{SVector{2,Float64},NTuple{2,Float64}},
    v_desired::Real;
    params::SFMParams = SFMParams(),
    walkmap = nothing,
    grid_dims = nothing,
    get_radius = nothing,
)
    e = SVector{2,Float64}(e_desired)
    vd = Float64(v_desired)
    vel = SVector{2,Float64}(agent.vel)
    τ = params.τ

    F_desired = (vd * e - vel) / τ

    r_i = get_radius !== nothing ? Float64(get_radius(agent)) : params.r_ped
    F_ped = SVector(0.0, 0.0)
    for id in nearby_ids(agent, model, params.interaction_radius)
        j = model[id]
        r_j = get_radius !== nothing ? Float64(get_radius(j)) : params.r_ped
        F_ped += repulsive_force_pedestrian(agent, j, params.A, params.B, r_i, r_j)
    end

    if walkmap === nothing
        F_wall = SVector(0.0, 0.0)
    else
        gd = grid_dims === nothing ? (size(walkmap, 2), size(walkmap, 1)) : grid_dims
        F_wall = repulsive_force_wall(agent, walkmap, gd, params.A_wall, params.B_wall, params.r_ped)
    end

    return F_desired + F_ped + F_wall
end

"""
    sfm_step!(agent, model, e_desired, v_desired, dt;
              params=SFMParams(), walkmap=nothing, grid_dims=nothing, v_max=nothing,
              get_radius=nothing)

Update `agent.vel` and `agent.pos` using the Social Force Model.

1. `a = sfm_acceleration(agent, model, e_desired, v_desired; params, walkmap, grid_dims, get_radius)`
2. `agent.vel = agent.vel + a * dt`; if `v_max !== nothing`, clamp `norm(agent.vel)` to `v_max`.
3. `agent.pos = agent.pos + agent.vel * dt`

Supports both `SVector` and `Tuple` for `agent.vel` and `agent.pos`: the internal step
uses `SVector` and then assigns back in a compatible form (tuple or SVector) to preserve
the agent's field types when possible.
"""
function sfm_step!(
    agent,
    model,
    e_desired,
    v_desired::Real,
    dt::Real;
    params::SFMParams = SFMParams(),
    walkmap = nothing,
    grid_dims = nothing,
    v_max = nothing,
    get_radius = nothing,
)
    a = sfm_acceleration(agent, model, e_desired, v_desired;
                         params, walkmap, grid_dims, get_radius)
    dtf = Float64(dt)

    v = SVector{2,Float64}(agent.vel) .+ a .* dtf
    if v_max !== nothing
        nv = norm(v)
        if nv > Float64(v_max) && nv > 1e-10
            v = v * (Float64(v_max) / nv)
        end
    end

    p = SVector{2,Float64}(agent.pos) .+ v .* dtf

    # Preserve field types: SVector vs Tuple
    if agent.vel isa SVector
        agent.vel = v
    else
        agent.vel = (v[1], v[2])
    end
    if agent.pos isa SVector
        agent.pos = p
    else
        agent.pos = (p[1], p[2])
    end
    return agent
end

end # module
