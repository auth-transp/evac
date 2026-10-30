# ContinuousSpace API for the social force model: desired directions from the pathfinders, the total social force
# on an agent (eq 8-9 του paper) and the synchronous time step (eq 10-12).
# Το αρχείο αυτό είναι ο προσαρμογέας του ABM: τα μαθηματικά βρίσκονται στο social_force.jl. Ο pathfinder (A*, D* Lite,
# FlowField, σημείο ή συνάρτηση) δίνει μόνο την επιθυμητή κατεύθυνση e_α (eq 1)· τη θέση την ορίζει το μοντέλο δυνάμεων.

# μετατροπή του Vec2 στον τύπο θέσης/ταχύτητας του agent (SVector ή NTuple)
@inline to_agent_vec(::Type{T}, x::Vec2) where {T<:Tuple} = Tuple(x)
@inline to_agent_vec(::Type{T}, x::Vec2) where {T} = T(x)

"""
    set_pedestrian!(sf::SocialForce, agent, model; v0, vmax, velocity)
Create or update the social-force state of `agent`. `v0` is the desired speed (drawn from the Gaussian
`N(v0_mean, v0_std)` of `sf` with the model's rng if missing), `vmax` the maximal speed (default
`vmax_factor * v0`) and `velocity` the initial (or, for a registered agent, the new) velocity (default: at rest for a
new agent, unchanged for a registered one). Agents not registered by this function
are registered automatically by [`social_force_step!`](@ref).
"""
function set_pedestrian!(
    sf::SocialForce,
    agent::AbstractAgent,
    model::ABM{<:ContinuousSpace{2}};
    v0::Union{Real,Nothing} = nothing,
    vmax::Union{Real,Nothing} = nothing,
    velocity = nothing,
)
    id = Agents.getid(agent)
    ped = get(sf.pedestrians, id, nothing)
    if isnothing(v0)
        if isnothing(ped)
            v0 = sf.v0_mean + sf.v0_std * randn(abmrng(model))                    # v⁰ ~ N(⟨v⁰⟩, std) όπως στο paper
            while v0 <= 0                                                        # η ουρά της Gaussian κάτω από 0 δεν έχει φυσικό νόημα
                v0 = sf.v0_mean + sf.v0_std * randn(abmrng(model))
            end
        else
            v0 = ped.v0                                                          # διατήρηση της υπάρχουσας τιμής
        end
    end
    vmax = isnothing(vmax) ? sf.vmax_factor * v0 : vmax                          # v^max = 1.3 v⁰ στο paper
    if isnothing(ped)                                                            # νέος πεζός: ξεκινά ακίνητος ή με την ταχύτητα που δόθηκε
        v = isnothing(velocity) ? zero(Vec2) : Vec2(velocity)
        sf.pedestrians[id] = Pedestrian(v, v, zero(Vec2), Float64(v0), Float64(vmax))
    else                                                                         # υπάρχων πεζός: ενημέρωση v⁰, v^max· η w/v αλλάζει μόνο αν δοθεί velocity
        ped.v0 = Float64(v0)
        ped.vmax = Float64(vmax)
        isnothing(velocity) || (ped.w = ped.v = Vec2(velocity))
    end
    return sf.pedestrians[id]
end

# ---------------- Επιθυμητή κατεύθυνση e_α (eq 1) ----------------

"""
    desired_direction(agent, model, destination, sf::SocialForce)
Return the desired direction `e_α` (a unit `SVector{2}`, or zero if the agent has nowhere to go) of `agent`.
Methods:
* `destination::Union{AStar,DStarLite}`: towards the next waypoint of the path stored by [`plan_route!`](@ref);
  waypoints closer than `sf.waypoint_radius` are dropped (the agent decelerates to rest after the last one).
* `destination::FlowField`: the (interpolated) flow direction at the agent's position.
* `destination::Union{NTuple{2},SVector{2}}`: straight towards that point (eq 1 for a single target point).
* `destination::Function`: `destination(agent, model)` must return the direction.
"""
function desired_direction end

function desired_direction(agent, model, pathfinder::Union{AStar{2},DStarLite{2}}, sf::SocialForce)
    path = get(pathfinder.agent_paths, Agents.getid(agent), nothing)             # η διαδρομή του agent (αν έχει σχεδιαστεί)
    isnothing(path) && return zero(Vec2)
    pos = Vec2(agent.pos)
    while !isempty(path)
        r = Vec2(get_direction(pos, first(path), model))                         # διάνυσμα προς το επόμενο waypoint r_α^k − r_α
        d = norm(r)
        if d <= sf.waypoint_radius
            popfirst!(path)                                                      # το waypoint επιτεύχθηκε: επόμενο
            continue
        end
        return r / d                                                             # eq (1)
    end
    return zero(Vec2)                                                            # τέλος διαδρομής: χωρίς επιθυμητή κατεύθυνση
end

desired_direction(agent, model, pathfinder::FlowField{2}, sf::SocialForce) =
    Vec2(flow_at_continuous(pathfinder, agent.pos))

function desired_direction(agent, model, target::Union{NTuple{2,<:Real},SVector{2,<:Real}}, sf::SocialForce)
    r = Vec2(get_direction(agent.pos, target, model))                            # r_α^0 − r_α
    d = norm(r)
    return d <= sf.waypoint_radius ? zero(Vec2) : r / d                          # eq (1)
end

desired_direction(agent, model, f::Function, sf::SocialForce) = Vec2(f(agent, model))

# ---------------- Συνολική δύναμη (eq 8-9) ----------------

"""
    social_force(agent, model, sf::SocialForce)
Total social force `F_α` (eq 9) on `agent`, computed from the stored state (`v`, `e`, `v⁰`) of the agent and of
its neighbours: the acceleration term (eq 2), the repulsion from other pedestrians within `sf.interaction_range`
(eq 3-4), from the borders within `sf.wall_range` (eq 5) and the optional attraction terms (eq 6), the repulsive
and attractive effects being weighted by the angle of sight (eq 7-8). `agent` must have been registered.
"""
function social_force(agent::AbstractAgent, model::ABM{<:ContinuousSpace{2}}, sf::SocialForce)
    id = Agents.getid(agent)
    ped = sf.pedestrians[id]
    pos = Vec2(agent.pos)
    cosphi = cos(sf.phi)
    F = driving_force(ped.v, ped.e, ped.v0, sf.tau)                              # F⁰_α, eq (2)
    # Άπωση από τους άλλους πεζούς β: F_αβ = w(e_α, −f_αβ) f_αβ, eq (8)
    for jd in nearby_ids(agent, model, sf.interaction_range)
        jd == id && continue
        other = get(sf.pedestrians, jd, nothing)
        isnothing(other) && continue                                             # agents εκτός μοντέλου δυνάμεων αγνοούνται
        r = Vec2(get_direction(model[jd].pos, agent.pos, model))                 # r_αβ = r_α − r_β
        norm(r) > sf.interaction_range && continue                               # η αναζήτηση γειτόνων είναι προσεγγιστική (υπερεκτιμά)· ακριβές cutoff
        f =pedestrian_repulsion(r, other.e, norm(other.v), sf.V0, sf.sigma, sf.step_time)
        F += direction_weight(ped.e, -f, cosphi, sf.c) * f
    end
    # Άπωση από τα όρια B: F_αB = w(e_α, −F) F με r_αB = r_α − r_B^α (πλησιέστερο σημείο κάθε τμήματος), eq (5), (9)
    if !isempty(sf.borders)
        for k in sf.buckets[bucket_index(pos, sf.bucket_size, size(sf.buckets))...]
            brd = sf.borders[k]
            r = pos - nearest_point(pos, brd.a, brd.b)
            norm(r) > sf.wall_range && continue                                  # πέρα από την εμβέλεια η άπωση είναι αμελητέα
            f = wall_repulsion(r, sf.U0, sf.R)
            F += direction_weight(ped.e, -f, cosphi, sf.c) * f
        end
    end
    # Ελκτικές δυνάμεις f_αi (προαιρετικές): F_αi = w(e_α, f_αi) f_αi, eq (8)
    if !isnothing(sf.attraction)
        for fi in sf.attraction(agent, model)
            f = Vec2(fi)
            F += direction_weight(ped.e, f, cosphi, sf.c) * f
        end
    end
    return F
end

# ---------------- Βήμα χρόνου (eq 10-12) ----------------

"""
    social_force_step!(model::ABM{<:ContinuousSpace{2}}, sf::SocialForce, destination, dt; agents = allagents(model))
Advance all `agents` by one explicit Euler step `dt` of the social force model:
```
dw/dt = F_α + fluctuations      (eq 10)
v_α   = w_α g(v_max/‖w_α‖)      (eq 11-12)
dr/dt = v_α
```
The update is synchronous (every force is computed from the state at the beginning of the step), as in the
paper. Call it once per step from `model_step!`, with a no-op `agent_step!`, instead of
[`move_along_route!`](@ref). `destination` determines the desired direction of every agent, see
[`desired_direction`](@ref); for a pathfinder, plan the paths beforehand with [`plan_route!`](@ref).
Choose `dt` well below the relaxation time `sf.tau` (e.g. 0.05 s for `τ = 0.5 s`).

If the agent type has a field `vel`, it is set to the realised velocity `v_α`. The model does not forbid walking
through walls (they only repel); make sure `dt` and the speeds keep agents from crossing them.
"""
function social_force_step!(
    model::ABM{<:ContinuousSpace{2}},
    sf::SocialForce,
    destination,
    dt::Real;
    agents = allagents(model),
)
    list = collect(agents)
    # 1) εγγραφή των νέων πεζών και υπολογισμός της επιθυμητής κατεύθυνσης e_α(t) όλων (πριν κινηθεί οποιοσδήποτε)
    for a in list
        haskey(sf.pedestrians, Agents.getid(a)) || set_pedestrian!(sf, a, model)
        sf.pedestrians[Agents.getid(a)].e = desired_direction(a, model, destination, sf)
    end
    # 2) δυνάμεις από την κατάσταση του τρέχοντος βήματος → νέα w, v, θέση για κάθε πεζό (χωρίς να εφαρμοστούν ακόμη)
    updates = map(list) do a
        ped = sf.pedestrians[Agents.getid(a)]
        w = ped.w + social_force(a, model, sf) * dt                              # dw = F dt (eq 10)
        if sf.fluctuation > 0
            w += sf.fluctuation * sqrt(dt) * Vec2(randn(abmrng(model)), randn(abmrng(model)))   # Langevin θόρυβος
        end
        v = cap_velocity(w, ped.vmax)                                            # eq (11)-(12)
        (w, v, Vec2(a.pos) + v * dt)                                             # r(t+dt) = r + v dt
    end
    # 3) εφαρμογή των νέων καταστάσεων και μετακίνηση των agents
    for (a, (w, v, newpos)) in zip(list, updates)
        ped = sf.pedestrians[Agents.getid(a)]
        ped.w = w
        ped.v = v
        T = typeof(a.pos)
        move_agent!(a, Agents.normalize_position(to_agent_vec(T, newpos), model), model)
        hasproperty(a, :vel) && (a.vel = to_agent_vec(typeof(a.vel), v))        # αποθήκευση της ταχύτητας στον agent (π.χ. για οπτικοποίηση)
    end
    return nothing
end
