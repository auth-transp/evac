"""
    gaussian_plume_map(dims, source;
                       Q::Real=1.0,
                       u::Real=1.0,
                       pixel_size::Real=1.0)

Compute a 2D concentration field on a regular grid using a
steady-state Gaussian plume model.

Assumptions:
- Continuous point source at ground level (H = 0).
- Neutral stability (Pasquill–Gifford class D) using Briggs formulas.
- Uniform wind speed `u` blowing along the +x direction.
- No vertical reflection and no mixing-height effects.

Arguments
- `dims::Tuple{Int,Int}`: `(ny, nx)` size of the output grid.
- `source::Tuple{Int,Int}`: `(iy, ix)` index of the source in grid cells.
- `Q::Real`: emission rate in kg/s.
- `u::Real`: wind speed in m/s (kept constant in the current usage).
- `pixel_size::Real`: linear size of a grid cell in meters.

Returns
- `Array{Float64,2}`: concentration field in kg·m⁻³ (up to the
  map scale implied by `pixel_size` and the chosen `Q`).
"""
function gaussian_plume_map(
    dims::Tuple{Int,Int},
    source::Tuple{Int,Int};
    Q::Real = 10.0,
    u::Real = 5.0,
    pixel_size::Real = 1.0,
)
    ny, nx = dims
    sy, sx = source

    Q_val = float(Q)
    u_val = max(float(u), eps()) # avoid division by zero
    px = float(pixel_size)

    C = zeros(Float64, ny, nx)

    @inbounds for iy in 1:ny
        for ix in 1:nx
            # Map indices to distances (meters) relative to source.
            x = (ix - sx) * px    # downwind distance (m), +x is wind direction
            y = (iy - sy) * px    # crosswind distance (m)

            # Only downwind (x > 0) receives plume.
            x <= 0 && continue

            σy, σz = _sigmas_PG_D(x)
            # Standard Gaussian plume without vertical reflection, H = 0, z = 0.
            prefactor = Q_val / (2π * u_val * σy * σz)
            exponent_y = -0.5 * (y / σy)^2
            # z = 0, H = 0 ⇒ exponent_z = 0

            C[iy, ix] = prefactor * exp(exponent_y)
        end
    end

    return C
end

"""
    _sigmas_PG_D(x)

Briggs-type parameterization of lateral and vertical dispersion
coefficients (σᵧ, σ_z) for Pasquill–Gifford stability class D
(neutral conditions).

`x` is the downwind distance in meters.
"""
function _sigmas_PG_D(x::Real)
    # Guard against x ≈ 0 which would give extremely small sigmas.
    x_m = max(float(x), 1.0)

    # Common Briggs formulas for class D (neutral):
    σy = 0.16 * x_m * (1 + 0.0004 * x_m)^(-0.5)
    σz = 0.14 * x_m * (1 + 0.0003 * x_m)^(-0.5)

    return σy, σz
end

