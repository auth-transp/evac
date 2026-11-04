# Προσομοίωση εκκένωσης (D* Lite + Social Force Model + τοξικό φορτίο) πάνω σε heightmap του Live Service.
# Είναι η συναρτησιακή μορφή του DStarLite - Scenario 3.jl: το μοντέλο, ο D* Lite planner ανά έξοδο, το SFM, το τοξικό φορτίο
# και η εναλλαγή των 7 concentration maps με αυξητικό replanning είναι ίδια· αλλάζουν μόνο η είσοδος (heightmap, έξοδοι,
# ζώνες spawn από το config του project) και η έξοδος (frames.bin για την οπτικοποίηση στον browser, CSV).
module LiveSim

using Agents                    # πλαίσιο ABM (ABM, add_agent!, step!, ...)
using Agents.Pathfinding        # DStarLite, AbsolutePenaltyMap, SocialForce, init_planner, extract_path, ...
using StaticArrays              # SVector για τη θέση των agents
using LinearAlgebra: norm       # νόρμα διανύσματος
using Random                    # MersenneTwister
using Images                    # φόρτωση/αλλαγή μεγέθους εικόνων, label_components
using FileIO                    # load/save
using ImageIO, ImageMagick      # αποκωδικοποίηση png/bmp
using JSON3                     # config.json και progress.json

include("Toxic.jl")

# Agent: συνεχής 2D agent (pos, vel προστίθενται αυτόματα). Δεν κρατιέται ιστορικό θέσης (το καταγράφει το frames.bin)
Agents.@agent struct AgentEscapes(ContinuousAgent{2, Float64})
    age::Float64                    # ηλικία (έτη)
    mass::Float64                   # μάζα (kg)
    toxicload::Float64              # συνολικό τρέχον τοξικό φορτίο (άθροισμα TL1+TL2+TL3)
    path::Vector{Tuple{Int,Int}}    # grid path προς την πλησιέστερη έξοδο
    TL1::Vector{Float64}            # ιστορικό τοξικού φορτίου, 1ο σύμπτωμα (οσμή)
    TL2::Vector{Float64}            # 2ο σύμπτωμα (ερεθισμός)
    TL3::Vector{Float64}            # 3ο σύμπτωμα (οίδημα)
end

const NUM_CMS = 7                                                   # πλήθος concentration maps (CM1..CM7)
const CM_SWITCH_TIMES = [0.0, 0.0, 292.0, 584.0, 876.0, 1168.0, 1460.0]   # εκκίνηση κατευθείαν από το CM2 (το CM1 παραλείπεται), μετά εναλλαγή ανά 292 s
const GOAL_RADIUS_M = 5.0                                           # ακτίνα (m) γύρω από την έξοδο που θεωρείται «διέφυγε» (στο σενάριο: 10 pixels)
const WHITE_THRESHOLD = 245                                         # (255 - τιμή heightmap) > 245: μη βατό
const SFM_SUBSTEPS = 20                                             # υπο-βήματα SFM ανά βήμα μοντέλου (dt_sfm = 0.05 s << τ = 0.5 s)
const SFM_WAYPOINT_RADIUS = 0.5                                     # (pixels) απόσταση στην οποία ένα waypoint θεωρείται επιτευγμένο
const DEFAULT_CM_DIR = joinpath(@__DIR__, "..", "..", "Concentration Maps")   # τα 7 CMs των προσομοιώσεων της διατριβής

# --- Φόρτωση εισόδων ---

# Heightmap όπως στο σενάριο: πρώτο κανάλι, κλίμακα 0..255, αντιστροφή (κόστος = 255 - τιμή)
function load_heightmap(file)
    img = load(file)
    data = permutedims(channelview(img), [2, 3, 1])[:, :, 1]
    return 255 .- floor.(Int, convert.(Float64, data) * 255)
end

# Τα 7 concentration maps (0..255), προσαρμοσμένα (resample) στις διαστάσεις του heightmap όταν διαφέρουν.
# ΠΡΟΣΩΡΙΝΟ: τα CMs είναι φτιαγμένα για το Qatargas· σε άλλη εγκατάσταση απλώς τεντώνονται ώστε να καλύπτουν το grid.
function load_cms(cm_dir, dims::Tuple{Int,Int})
    cms = Vector{Matrix{Float64}}(undef, NUM_CMS)
    for k in 1:NUM_CMS
        img = load(joinpath(cm_dir, string(k) * ".bmp"))
        a = Float64.(red.(RGB{N0f8}.(img))) .* 255.0
        cms[k] = size(a) == dims ? a : clamp.(imresize(a, dims), 0.0, 255.0)
    end
    return cms
end

@inline function world_to_cell(p, dims::Tuple{Int,Int})
    return (clamp(Int(floor(p[1])), 1, dims[1]), clamp(Int(floor(p[2])), 1, dims[2]))
end

at_goal(pos, dests, radius) = minimum(hypot(pos[1] - d[1], pos[2] - d[2]) for d in dests) <= radius

function find_changed_cells(old::AbstractMatrix{Int}, new::AbstractMatrix{Int})
    cells = Tuple{Int,Int}[]
    @inbounds for i in axes(old, 1), j in axes(old, 2)
        old[i, j] != new[i, j] && push!(cells, (i, j))
    end
    return cells
end

# Το συντομότερο μονοπάτι ανάμεσα στα μονοπάτια προς τις διάφορες εξόδους
function choose_best_path(paths::Vector{Vector{Tuple{Int,Int}}})
    filter!(!isempty, paths)
    isempty(paths) && return Tuple{Int,Int}[]
    return argmin(p -> length(p), paths)
end

# Επιθυμητή κατεύθυνση e_α για το SFM: μοναδιαίο διάνυσμα προς το επόμενο waypoint του a.path
function sfm_desired_direction(agent, model)
    while !isempty(agent.path)
        cell = agent.path[1]
        r = SVector{2,Float64}(cell[1] - agent.pos[1], cell[2] - agent.pos[2])
        d = norm(r)
        if d <= SFM_WAYPOINT_RADIUS
            popfirst!(agent.path)
            continue
        end
        return r / d
    end
    return SVector{2,Float64}(0.0, 0.0)
end

# --- Βήματα του μοντέλου ---

function agent_step!(person, model)
    at_goal(person.pos, model.goal, model.goal_radius) && return                                   # διέφυγε: δεν κινείται, δεν συσσωρεύει TL

    pm = model.penalty_map
    i = clamp(Int(floor(person.pos[1])), 1, size(pm, 1))
    j = clamp(Int(floor(person.pos[2])), 1, size(pm, 2))
    Ct = pm[i, j]                                                                # συγκέντρωση τοξικού στο cell του agent
    TL = update_toxic_load(model.tox, Ct, [person.TL1[end], person.TL2[end], person.TL3[end]], model.dt)
    person.toxicload = sum(TL)
    push!(person.TL1, TL[1]); push!(person.TL2, TL[2]); push!(person.TL3, TL[3])

    speed = 1.35                                                                 # κανονική ταχύτητα (m/s)
    if 0 < person.toxicload <= 1
        speed = 1.35 * exp(0.393 * person.toxicload)                             # μικρή έκθεση: αύξηση ταχύτητας (πανικός)
    elseif 1 < person.toxicload < 3
        speed = -1.78 * log(person.toxicload) + 2.063                            # η ταχύτητα μειώνεται
    elseif person.toxicload >= 3
        speed = 0.0                                                              # θανατηφόρο φορτίο: ακινητοποίηση
    end
    set_pedestrian!(model.sfm, person, model; v0 = speed * model.mpx)            # επιθυμητή ταχύτητα v⁰ σε pixels/s
end

function model_step!(model)
    sfm = model.sfm
    movers = AgentEscapes[]
    for a in allagents(model)
        if at_goal(a.pos, model.goal, model.goal_radius)
            delete!(sfm.pedestrians, a.id)                                       # διέφυγε: δεν απωθεί πια
        elseif a.toxicload >= 3.0
            ped = sfm.pedestrians[a.id]                                          # ακινητοποιημένος: ακίνητο εμπόδιο
            ped.w = zero(ped.w); ped.v = zero(ped.v); ped.e = zero(ped.e)
        else
            push!(movers, a)
        end
    end
    for _ in 1:SFM_SUBSTEPS
        social_force_step!(model, sfm, sfm_desired_direction, model.dt / SFM_SUBSTEPS; agents = movers)
    end
end

# --- Έξοδοι αρχείων ---

# Εγγραφή JSON ατομικά (πρώτα σε προσωρινό αρχείο) ώστε ο server να μη διαβάσει μισογραμμένο αρχείο
function write_json(file, obj)
    tmp = file * ".tmp"
    open(io -> JSON3.write(io, obj), tmp, "w")
    mv(tmp, file; force = true)
end

# --- Κύρια συνάρτηση ---

"""
    run_simulation(dir; T = 2500, seed = 123, cm_dir = DEFAULT_CM_DIR, progress_every = 10)

Τρέχει την προσομοίωση για το project στον φάκελο `dir` (heightmap.png, spawn_mask.png, config.json από το Live Service).
Γράφει στον ίδιο φάκελο: meta.json, cm_1..7.png, frames.bin (ανά frame και agent: col, row, toxicload, escaped ως Float32),
trajectories.csv (step, agent_id, x, y, toxicload, με x = γραμμή, y = στήλη όπως στα σενάρια), progress.json και result.json.
"""
function run_simulation(dir::AbstractString; T::Int = 2500, seed::Int = 123, cm_dir = DEFAULT_CM_DIR, progress_every::Int = 10)
    t_start = time()
    progress(status, frame; extra...) = write_json(joinpath(dir, "progress.json"),
        (; status, frame, T, elapsed = round(time() - t_start; digits = 1), extra...))
    progress("starting", 0)

    cfg = JSON3.read(read(joinpath(dir, "config.json"), String))
    heightmap = load_heightmap(joinpath(dir, "heightmap.png"))
    dims = size(heightmap)
    @assert (dims[1], dims[2]) == (cfg.rows, cfg.cols) "το heightmap δεν ταιριάζει με το config.json"
    mpx = Float64(cfg.meters_to_pixels)                                          # METERS_TO_PIXELS του μοντέλου
    walkmap = BitArray(trues(dims...))
    walkmap[heightmap .> WHITE_THRESHOLD] .= false                               # μη βατά: τοίχοι και εκτός εγκατάστασης

    cm_list = load_cms(cm_dir, dims)
    penalty_map = copy(cm_list[searchsortedlast(CM_SWITCH_TIMES, 0.0)])                                               # το τρέχον CM λειτουργεί ως penalty_map
    NPM_int = round.(Int, heightmap .+ penalty_map)                              # ανάγλυφο + τοξική συγκέντρωση

    # έξοδοι: το goal_cell (γραμμή, στήλη) που υπολόγισε το Live Service, ως σημεία (x = γραμμή, y = στήλη)
    exits = [e for e in cfg.exits if e.goal_cell !== nothing]
    isempty(exits) && error("δεν υπάρχει έξοδος με βατό cell")
    dests = [(Float64(e.goal_cell[1]), Float64(e.goal_cell[2])) for e in exits]
    exit_cells = [world_to_cell(g, dims) for g in dests]

    # cells spawn: ζώνες του χρήστη ∩ βατά ∩ συνδεδεμένα με κάποια έξοδο (αλλιώς ο agent δεν έχει διαδρομή)
    spawn_img = load(joinpath(dir, "spawn_mask.png"))
    spawn_mask = permutedims(channelview(spawn_img), [2, 3, 1])[:, :, 1] .> 0.5
    comp = label_components(walkmap, trues(3, 3))
    exit_comps = Set(comp[c...] for c in exit_cells)
    0 in exit_comps && error("μια έξοδος δεν βρίσκεται σε βατό cell")
    spawn_cells = [c for c in findall(spawn_mask .& walkmap) if comp[c] in exit_comps]
    isempty(spawn_cells) && error("δεν υπάρχουν βατά cells στις ζώνες spawn που να συνδέονται με έξοδο")

    rng = MersenneTwister(seed)
    space = Pathfinding.ContinuousSpace(size(NPM_int); periodic = false, spacing = 1)
    prev_NPM = copy(NPM_int)                                                     # προηγούμενος χάρτης κόστους (για τα αλλαγμένα cells)
    cost_metric_obj = AbsolutePenaltyMap(NPM_int, MaxDistance{2}())              # ο pathfinder κρατά και ενημερώνει τον NPM_int in-place
    pathfinderPM = DStarLite(space; walkmap = walkmap, cost_metric = cost_metric_obj)

    # Social Force Model (Helbing & Molnár, 1995): παράμετροι σε m/s, μετατροπή σε pixels με mpx όπως στο σενάριο
    sfm = SocialForce(pathfinderPM;
        offset = (0.5, 0.5),
        tau = 0.5,
        V0 = 2.1 * mpx^2, sigma = 0.3 * mpx,
        step_time = 2.0,
        U0 = 10.0 * mpx^2, R = 0.2 * mpx,
        angle_of_sight = 200.0, c = 0.5,
        vmax_factor = 1.3,
        interaction_range = 4.0 * mpx,
        waypoint_radius = SFM_WAYPOINT_RADIUS,
    )

    goal_radius = GOAL_RADIUS_M * mpx                                            # ακτίνα διαφυγής σε pixels του grid
    properties = (; pathfinderPM, dt = 1.0, goal = dests, goal_radius, sfm, penalty_map, mpx, tox = setup_toxic())
    model = ABM(AgentEscapes, space; rng = rng, properties = properties, agent_step! = agent_step!, model_step! = model_step!)

    n_agents = Int(get(cfg, :n_agents, 100))
    age_range, speed_range, mass_range = (22, 60), (4.0, 7.0), (50, 80)
    for _ in 1:n_agents
        age = rand(abmrng(model)) * (age_range[2] - age_range[1]) + age_range[1]
        mass = rand(abmrng(model)) * (mass_range[2] - mass_range[1]) + mass_range[1]
        vel = Tuple(rand(abmrng(model), 2) .* (speed_range[2] - speed_range[1]) .+ speed_range[1])
        cell = rand(abmrng(model), spawn_cells)
        # θέση μέσα στο cell με τον κανόνα του μοντέλου (floor) και του SFM (κέντρο στο ακέραιο σημείο): [i, i + 0.49]
        pos = (cell[1] + 0.49 * rand(abmrng(model)), cell[2] + 0.49 * rand(abmrng(model)))
        person = add_agent!(pos, AgentEscapes, model, vel, age, mass, 0.0, Tuple{Int,Int}[], [0.0], [0.0], [0.0])
        set_pedestrian!(sfm, person, model; v0 = 1.35 * mpx, velocity = (0.0, 0.0))
    end
    ids = sort!([a.id for a in allagents(model)])

    # D* Lite planners (ένας ανά έξοδο) και αρχικά μονοπάτια
    planners = [init_planner(pathfinderPM, goal_cell) for goal_cell in exit_cells]
    for a in allagents(model)
        a.path = choose_best_path([extract_path(p, world_to_cell(a.pos, dims)) for p in planners])
    end
    no_path = count(a -> isempty(a.path), allagents(model))
    no_path > 0 && println("Προσοχή: $no_path agents δεν βρήκαν διαδρομή προς καμία έξοδο")

    # meta.json και εικόνες CM για την οπτικοποίηση (συντεταγμένες εικόνας: x = στήλη, y = γραμμή, αρχή πάνω αριστερά)
    for k in 1:NUM_CMS
        save(joinpath(dir, "cm_$k.png"), Gray{N0f8}.(clamp.(cm_list[k] ./ 255.0, 0.0, 1.0)))
    end
    write_json(joinpath(dir, "meta.json"), (; n_agents = length(ids), T, dt = 1.0, seed, rows = dims[1], cols = dims[2],
        meters_per_pixel = 1 / mpx, cm_switch_times = CM_SWITCH_TIMES, num_cms = NUM_CMS, goal_radius, goal_radius_m = GOAL_RADIUS_M,
        goals = [(; col = d[2] - 0.5, row = d[1] - 0.5) for d in dests], stride = 4, no_path))

    # κύριος βρόχος: βήμα μοντέλου, εναλλαγή CM με replanning, καταγραφή frame
    escaped_at = Dict{Int,Int}()                                                 # id => frame διαφυγής
    current_map_idx = searchsortedlast(CM_SWITCH_TIMES, 0.0)
    t_sim_total = 0.0
    t_replan_total = 0.0
    buf = Vector{Float32}(undef, 4 * length(ids))
    open(joinpath(dir, "frames.bin"), "w") do fbin
    open(joinpath(dir, "trajectories.csv"), "w") do fcsv
        println(fcsv, "step,agent_id,x,y,toxicload")
        for frame in 1:T
            t_sim_total += @elapsed step!(model, 1)

            t_sim = (frame - 1) * 1.0
            new_idx = searchsortedlast(CM_SWITCH_TIMES, t_sim)
            if new_idx != current_map_idx
                current_map_idx = new_idx
                penalty_map .= cm_list[current_map_idx]
                NPM_int_new = round.(Int, heightmap .+ penalty_map)
                pathfinderPM.cost_metric.pmap .= NPM_int_new                     # ενημέρωση του κόστους in-place (χωρίς νέο DStarLite)
                changed = find_changed_cells(prev_NPM, NPM_int_new)
                prev_NPM .= NPM_int_new
                t_replan_total += @elapsed begin
                    for planner in planners
                        update_after_cm_change!(planner, changed)                # αυξητικό replanning μόνο στα αλλαγμένα cells
                    end
                    for a in allagents(model)
                        a.path = choose_best_path([extract_path(p, world_to_cell(a.pos, dims)) for p in planners])
                    end
                end
                println("Frame $frame: CM $current_map_idx, replanning ($(length(changed)) cells)")
            end

            for (k, id) in enumerate(ids)
                a = model[id]
                esc = at_goal(a.pos, dests, goal_radius)
                esc && !haskey(escaped_at, id) && (escaped_at[id] = frame)
                buf[4k-3] = a.pos[2] - 0.5                                       # στήλη (x της εικόνας)
                buf[4k-2] = a.pos[1] - 0.5                                       # γραμμή (y της εικόνας)
                buf[4k-1] = a.toxicload
                buf[4k] = esc ? 1.0f0 : 0.0f0
                println(fcsv, frame, ',', id, ',', a.pos[1], ',', a.pos[2], ',', a.toxicload)
            end
            write(fbin, buf)
            if frame % progress_every == 0 || frame == T
                flush(fbin)
                progress("running", frame; escaped = length(escaped_at))
            end
        end
    end
    end

    incapacitated = count(a -> a.toxicload >= 3.0, allagents(model))
    result = (; status = "done", n_agents = length(ids), T, escaped = length(escaped_at), incapacitated,
        mean_escape_time = isempty(escaped_at) ? nothing : sum(values(escaped_at)) / length(escaped_at),
        no_path, simulation_time = round(t_sim_total; digits = 2), replanning_time = round(t_replan_total; digits = 2),
        total_time = round(time() - t_start; digits = 1))
    write_json(joinpath(dir, "result.json"), result)
    progress("done", T; escaped = length(escaped_at))
    return result
end

end # module
