begin   # Φόρτωση των απαραίτητων βιβλιοθηκών
    using Agents
    using Agents.Pathfinding
    using CSV
    using CairoMakie
    using DataFrames
    using Dates
    using FileIO: load
    using ImageMagick
    using Images
    using InteractiveDynamics
    using Makie
    using Observables
    using Random
    using StaticArrays
end


Agents.@agent struct AgentEscapes(ContinuousAgent{2, Float64})
    age::Float64
    mass::Float64
    toxicload::Float64
    pathX::Vector{Float64}
    pathY::Vector{Float64}
    TL1::Vector{Float64}
    TL2::Vector{Float64}
    TL3::Vector{Float64}
    goal_idx::Int   # 1 or 2: which flow field (destination) this agent follows
end


begin   # Φόρτωση του heightmap και των hand-drawn penalty maps (και όλων των CM1..CM8)
    heightmap_data = load("NADEEN/Maps/Qatargas Map.jpg")
    heightmap_data = permutedims(channelview(heightmap_data), [2,3,1])[:,:,1]
    global heightmap = floor.(Int, convert.(Float64, heightmap_data) * 255)
    heightmap = 255 .- heightmap

    const NUM_CMS = 8
    cm_list = Vector{Array{Float64,2}}(undef, NUM_CMS)
    for k in 1:NUM_CMS
        fn = joinpath("Concentration Maps", string(k) * ".bmp")
        img = load(fn)
        img = permutedims(channelview(img), [2,3,1])[:,:,1]
        cm_list[k] = convert.(Float64, img) .* 255.0
    end

    for k in 1:NUM_CMS
        @assert size(cm_list[k]) == size(heightmap) "Concentration map $k size mismatch with heightmap"
    end

    global CM_GLOBAL_MIN, CM_GLOBAL_MAX, CM_LEVELS
    CM_GLOBAL_MIN = minimum(minimum.(cm_list))
    CM_GLOBAL_MAX = maximum(maximum.(cm_list))
    if CM_GLOBAL_MIN < CM_GLOBAL_MAX
        CM_LEVELS = collect(range(CM_GLOBAL_MIN, CM_GLOBAL_MAX, length = 10))
    else
        CM_LEVELS = Float64[CM_GLOBAL_MIN]
    end

    global penalty_map = copy(cm_list[1])
end

NPM = heightmap .+ penalty_map
NPM_int = round.(Int, NPM)


begin   # Αρχικοποίηση των παραμέτρων του μοντέλου
    const METERS_TO_PIXELS = 723.37 / 2500.0
    dt = 1.
    seed = 123
    run_timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
    n_agents = 50
    toxicity_rate = 0.07
    age_range = (22,60)
    speed_range = (4.0,7.0)
    mass_range = (50,80)
    ag_range_y = (size(heightmap)[1]/4):(3*size(heightmap)[1]/4)
    ag_range_x = (size(heightmap)[2]/4):(3*size(heightmap)[2]/4)
    dims = (size(NPM))
    walkmap = BitArray(trues(dims...))

    white_threshold = 245
    walkmap[heightmap .> white_threshold] .= false
end

# goals (same as A* scenario)
dests = [(500., 854.), (120., 248.)]

rng = MersenneTwister(seed)
space = ContinuousSpace(size(NPM); periodic = false, spacing = 1)


begin
    # FlowField: one pathfinder per goal (no cost metric — walkmap only; obstacles from heightmap)
    pathfinder_goal1 = FlowField(space; walkmap = walkmap)
    pathfinder_goal2 = FlowField(space; walkmap = walkmap)

    # Set destinations and generate flow fields (no agent needed; set destination then generate)
    for (pf, dest) in [(pathfinder_goal1, dests[1]), (pathfinder_goal2, dests[2])]
        disc = Tuple(Pathfinding.to_discrete_position(dest, pf))
        if all(1 .<= disc .<= size(pf.walkmap)) && pf.walkmap[disc...]
            pf.destination = disc
            Pathfinding.generate_flow_field!(pf)
        end
    end

    cost_metric_str = "FlowField"
    properties = (
        pathfinder_goal1 = pathfinder_goal1,
        pathfinder_goal2 = pathfinder_goal2,
        heightmap = heightmap,
        dt = dt,
        speed_range = speed_range,
        goal = dests,
    )
end


#
# Helper: true if position is within radius of any goal (TL and movement stop when true)
const goal_radius = 10.0
function at_goal(pos, dests, radius = goal_radius)
    px, py = Float64(pos[1]), Float64(pos[2])
    return minimum(hypot(px - d[1], py - d[2]) for d in dests) <= radius
end


function agent_step!(person, model)
    if at_goal(person.pos, model.goal)
        push!(person.pathX, person.pos[1])
        push!(person.pathY, person.pos[2])
        return
    end

    position = floor.(Int, person.pos)
    # Clamp to valid 1-based indices (agent can be near boundary)
    position = (clamp(position[1], 1, size(penalty_map, 1)), clamp(position[2], 1, size(penalty_map, 2)))
    Ct = penalty_map[position[1], position[2]]
    TLcurrent = [person.TL1[end], person.TL2[end], person.TL3[end]]
    TL = update_toxic_load(Ct, TLcurrent, model.dt)

    person.toxicload = sum(TL)
    push!(person.TL1, TL[1])
    push!(person.TL2, TL[2])
    push!(person.TL3, TL[3])

    speed = 1.35
    if 0 < person.toxicload <= 1
        speed = 1.35 * exp(0.393 * person.toxicload)
    elseif 1 < person.toxicload < 3
        speed = -1.78 * log(person.toxicload) + 2.063
    elseif person.toxicload >= 3
        speed = 0.0
    end

    # Move along the flow field for this agent's chosen goal
    pf = person.goal_idx == 1 ? model.pathfinder_goal1 : model.pathfinder_goal2
    move_along_route!(person, model, pf, speed * METERS_TO_PIXELS, model.dt)
    push!(person.pathX, person.pos[1])
    push!(person.pathY, person.pos[2])
end


function model_step!(model)
    for (a1, a2) in interacting_pairs(model, 0.012, :nearest)
        elastic_collision!(a1, a2, :mass)
    end
end

model = ABM(
  AgentEscapes,
  space;
  rng          = rng,
  properties   = properties,
  agent_step!  = agent_step!,
  model_step!  = model_step!
)

# ===== PATHFINDING TIMING: FlowField =====
# With FlowField, "planning" is setting destination + generate_flow_field! (done once per goal above).
# Per-agent we only assign goal_idx (nearest goal by distance_at); no per-agent path computation.
initial_pathfinding_time = 0.0

for _ in 1:n_agents
    age = rand(abmrng(model))*(age_range[2]-age_range[1]) + age_range[1]
    mass = rand(abmrng(model)) * (mass_range[2]-mass_range[1]) + mass_range[1]
    vel = Tuple(rand(abmrng(model), 2) .* (speed_range[2]-speed_range[1]) .+ speed_range[1])

    max_attempts = 1000
    attempts = 0
    pos = nothing
    while attempts < max_attempts
        candidate_pos = Tuple((rand(abmrng(model), floor.(ag_range_y)), rand(abmrng(model), floor.(ag_range_x))))
        pos_int = floor.(Int, candidate_pos)
        if 1 <= pos_int[1] <= size(walkmap, 1) && 1 <= pos_int[2] <= size(walkmap, 2) && walkmap[pos_int[1], pos_int[2]]
            pos = candidate_pos
            break
        end
        attempts += 1
    end

    if pos === nothing
        println("Warning: Could not find valid walkable spawn position for agent in Scenario 3 after $max_attempts attempts")
        continue
    end

    # Assign agent to the goal (flow field) with smaller distance at spawn
    disc = Tuple(floor.(Int, pos))
    d1 = Pathfinding.distance_at(model.pathfinder_goal1, disc)
    d2 = Pathfinding.distance_at(model.pathfinder_goal2, disc)
    goal_idx = d1 <= d2 ? 1 : 2

    person = add_agent!(pos, AgentEscapes, model, vel, age, mass, 1., [pos[1]], [pos[2]], [0.0], [0.0], [0.0], goal_idx)
end

println("Initial flow field setup: both destinations set and generated (no per-agent planning).")


function setupToxic()
    Atime = [0.0, 2.0, 5.0, 8.0, 15.0, 30.0]
    Arho = zeros(3, 6)
    Arho[1, 2:6] = [2.87, 2.33, 2.09, 1.81, 1.55]
    Arho[2, 2:6] = [202.38, 164.38, 147.74, 128.10, 109.45]
    Arho[3, 2:6] = [286.71, 232.87, 209.30, 181.47, 155.05]
    MW = 34
    Arho *= MW/24.04
    Arho = Arho'
    Atime = Atime*60
    taumin = 200.
    taumax = 86400.
    Brho = zeros(7,3)
    Balpha = zeros(7,3)
    rhomax = zeros(1,3)
    rhomin = zeros(1,3)
    Btime = zeros(7, 3)

    for k=1:3
        for b=2:6
            Brho[b, k] = Arho[b, k]
            Btime[b, k] = Atime[b]
        end
        Btime[1, k] = taumin
        Btime[7, k] = taumax
    end

    for k=1:3
        for b=3:6
            if Brho[b-1, k]==Brho[b, k]
                Balpha[b, k] = 0.0
            else
                Balpha[b, k] = log(Atime[b]/Atime[b-1])/log(Brho[b-1, k]/Brho[b, k])
            end
        end
        Balpha[2, k] = Balpha[3, k]
        Balpha[1, k] = Balpha[2, k]
        Balpha[7, k] = Balpha[6, k]
    end

    for k=1:3
        if Balpha[3, k]==0
            rhomax[k] = Brho[2, k]
            Brho[1, k] = rhomax[k]
        else
            rhomax[k] = Brho[2, k]*(Btime[2, k]/taumin)^(1/Balpha[2, k])
            Brho[1, k] = rhomax[k]
        end
        if Brho[5, k]==Brho[6, k]
            rhomin[k] = Brho[6, k]
            Brho[7, k] = rhomin[k]
        else
            rhomin[k] = Brho[6, k]*(Btime[6, k]/taumax)^(1/Balpha[6, k])
            Brho[7, k] = rhomin[k]
        end
    end

    for k=1:3
        for b=2:7
            if Balpha[b, k] == 0
                Btime[b, k] = Btime[b-1, k]
            end
        end
    end

    for k=1:3
        for b=3:5
            if Balpha[b-1, k]==0 && Balpha[b, k]>0
                Balpha[b, k]=log(Btime[b, k]/Btime[b-1, k])/log(Brho[b-1, k]/Brho[b, k])
            end
        end
    end

    return Balpha, Btime, Brho'
end

Balpha, Btime, Brho = setupToxic()


function update_toxic_load(Ct, TLcurrent, dt)
    TL = TLcurrent
    TL_rate = 0.0

    for k = 1:3
        Cmin = Brho[k, 7]
        Cmax = Brho[k, 1]
        if Ct > Cmax
            TL_rate = 1 / Btime[1]
        elseif Ct < Cmin
            TL_rate = 0.0
        else
            for i in 2:size(Btime, 1)
                if i <= size(Brho, 2)
                    if Brho[k, i-1] < Ct && Ct < Brho[k, i]
                        TL_rate = (1/Btime[i, k])*((Ct/Brho[k, i])^(Balpha[i, k]))
                    end
                end
            end
        end
        TL[k] = TL[k] .+ TL_rate * dt
    end
    return TL
end


function static_preplot!(ax, abmplot)
    model = isa(abmplot, Observable)  ? abmplot[] :
            hasproperty(abmplot, :model) ? abmplot.model[] :
            abmplot

    dests = model.goal
    xs_g = getindex.(dests, 1)
    ys_g = getindex.(dests, 2)
    scatter!(ax, xs_g, ys_g; color = (:red, 50), marker = '●')

    for agent in allagents(model)
        xs = agent.pathX
        ys = agent.pathY
        lines!(ax, xs, ys; linewidth = 2, color = personcolor(agent))
    end
end


function personcolor(person::AgentEscapes)
    if person.toxicload >= 3.
        return :red
    elseif person.toxicload <= 1.
        return :green
    else
        return :orange
    end
end

agent_marker_color(a::AgentEscapes) = Makie.to_color(personcolor(a))


@time begin
    const T = 1852
    # Simulation time t matches the on-screen counter: t = (frame - 1) * dt. Switch CMs at these t (s).
    const CM_SWITCH_TIMES = Float64[0, 9, 48, 85, 100, 120, 215, 300]
    const NUM_MAPS = length(CM_SWITCH_TIMES)

    fig = Figure(; size = (800,800))
    ax  = Makie.Axis(fig[1,1];
               title  = "Evacuation with Toxic Trail (FlowField)",
               aspect = DataAspect())

    heatmap!(ax, heightmap; colormap=:grays, alpha=0.3)

    cm_obs = Observable(penalty_map)
    cm_colorrange = @lift let z = $cm_obs
        m = minimum(z)
        M = maximum(z)
        if m == M
            δ = max(1.0, abs(m) * 1e-6)
            (Float32(m - δ), Float32(M + δ))
        else
            (Float32(m), Float32(M))
        end
    end
    cm_contour = contour!(
        ax, cm_obs;
        colormap = cgrad([:yellow, :orange, :red]),
        levels = CM_LEVELS,
        colorrange = cm_colorrange,
        linewidth = 1.5,
        alpha = 0.7,
    )

    goals = model.goal
    scatter!(ax,
        getindex.(goals,1),
        getindex.(goals,2);
        color  = (:red,50),
        marker = :circle,
    )

    frame_obs = Observable(0)
    counter_lbl = Label(
        fig,
        @lift("Time elapsed = $((($frame_obs-1)*dt)) s"),
        fontsize = 16,
        padding = (6, 10, 6, 10),
        halign = :left
    )
    fig[1,1, TopLeft()] = counter_lbl

    xs0 = Float64[]
    ys0 = Float64[]

    for a in allagents(model)
        push!(xs0, a.pos[1])
        push!(ys0, a.pos[2])
    end

    colors0 = [agent_marker_color(a) for a in allagents(model)]

    posobs = Observable(Point2f.(xs0, ys0))
    colobs = Observable(colors0)

    agents_vec = collect(allagents(model))
    line_color_obs = [Observable(agent_marker_color(a)) for a in agents_vec]
    lines_plots = [
        lines!(
            ax,
            [a.pos[1]], [a.pos[2]];
            color     = line_color_obs[i],
            linewidth = 2,
        )
        for (i, a) in enumerate(agents_vec)
    ]
    agent_scat = scatter!(
        ax, posobs;
        color       = colobs,
        markersize  = 10,
        strokewidth = 0,
    )

    agent_id_labels = [string(a.id) for a in allagents(model)]
    text!(
        ax, posobs;
        text       = agent_id_labels,
        fontsize   = 7,
        align      = (:left, :center),
        offset     = (5, 0),
        color      = :black,
        strokewidth = 0.75,
        strokecolor = :white,
    )

    df = DataFrame(
        step       = Int[],
        agent_id   = Int[],
        x          = Float64[],
        y          = Float64[],
        toxicload  = Float64[]
    )

    video_file = "SCENARIOS/SCENARIO 3/Simulation Results/FlowField_SCENARIO_3_$(n_agents)_$(seed)_$(cost_metric_str)_$(run_timestamp).mp4"
    csv_file   = "SCENARIOS/SCENARIO 3/Simulation Results/FlowField_SCENARIO_3_$(n_agents)_$(seed)_$(cost_metric_str)_$(run_timestamp).csv"

    current_map_idx = 1
    replanning_time = 0.0

    # FlowField has no penalty map; flow is based only on walkmap. On map change we do not regenerate flow fields.
    record(fig, video_file, 1:T; framerate=30) do frame
        frame_obs[] = frame

        global current_map_idx
        t_sim = (frame - 1) * dt
        new_idx = searchsortedlast(CM_SWITCH_TIMES, t_sim)
        if new_idx != current_map_idx
            current_map_idx = new_idx
            penalty_map .= cm_list[current_map_idx]
            # FlowField does not use penalty/cost; no pathfinder update or replanning.
            println("Frame $frame: swapped to concentration map $current_map_idx (FlowField: no replanning).")
        end

        step!(model, 1)

        cm_obs[] = penalty_map

        agents_now = collect(allagents(model))
        for (i, a) in enumerate(agents_now)
            lines_plots[i][1][] = Point2f.(a.pathX, a.pathY)
            line_color_obs[i][] = agent_marker_color(a)
        end

        posobs[] = [Point2f(a.pos[1], a.pos[2]) for a in agents_now]
        colobs[] = [agent_marker_color(a) for a in agents_now]

        for a in agents_now
            push!(df, (
                frame,
                a.id,
                a.pos[1],
                a.pos[2],
                a.toxicload
            ))
        end
    end

    total_pathfinding_time = initial_pathfinding_time + replanning_time
    println("\n=== Pathfinding Timing Summary (FlowField) ===")
    println("Initial flow field setup: $(round(initial_pathfinding_time; digits=4)) s")
    println("Replanning time (FlowField does not replan on map change): $(round(replanning_time; digits=4)) s")
    println("Total pathfinding time: $(round(total_pathfinding_time; digits=4)) s")
    println("===================================\n")

    println("Το animation σώθηκε ως $video_file")
    CSV.write(csv_file, df)
    println("Τα δεδομένα θέσης & toxicload αποθηκεύτηκαν ως $csv_file")
end


begin
    dt = @isdefined(dt) ? dt : 1.0
    seed_str = @isdefined(seed) ? string(seed) : nothing
    folder = joinpath("SCENARIOS","SCENARIO 3", "Simulation Results")

    csv_file = seed_str === nothing ? nothing : joinpath(folder, "FlowField_SCENARIO_3_$(n_agents)_$(seed)_$(cost_metric_str)_$(run_timestamp).csv")
    if csv_file === nothing || !isfile(csv_file)
        pattern = Regex("^FlowField_SCENARIO_3_.*_$(cost_metric_str)_.*\\.csv\$")
        csvs = filter(f -> occursin(pattern, f), readdir(folder))
        @assert !isempty(csvs) "Δεν βρέθηκαν αρχεία *FlowField_SCENARIO_3_*.csv στο $(folder)."
        stats = stat.(joinpath.(Ref(folder), csvs))
        latest_idx = argmax(getfield.(stats, :mtime))
        csv_file = joinpath(folder, csvs[latest_idx])
        seed_str = "latest"
    end

    df = CSV.read(csv_file, DataFrame)

    edges_inner   = 0:0.5:3.0
    centers_inner = (edges_inner[1:end-1] .+ edges_inner[2:end]) ./ 2
    bin_w_inner   = 0.45
    edge_offset = 0.30
    x0_pos      = 0.0 - edge_offset
    x3_pos      = 3.0 + edge_offset
    bin_w_edge  = 0.30

    function bin_counts_open(values::AbstractVector{<:Real}, edges::AbstractVector{<:Real})
        counts = zeros(Int, length(edges) - 1)
        @inbounds for v in values
            if v > edges[1] && v < edges[end]
                idx = searchsortedlast(edges, v)
                idx = clamp(idx, 1, length(counts))
                counts[idx] += 1
            end
        end
        return counts
    end

    function mytickfmt(x::Real)
        string(round(x; digits=1))
    end
    function mytickfmt(xs::AbstractVector{<:Real})
        string.(round.(xs; digits=1))
    end

    T_avail = maximum(df.step)
    T_play  = min(T, T_avail)
    n_agents_guess = maximum(combine(groupby(df, :step), nrow).nrow)

    fig = Figure(; size = (1000, 600))
    ax  = CairoMakie.Axis(fig[1,1];
        title  = "Toxic Load (X) vs Agents Count (Y) — FlowField",
        xlabel = "Toxic load",
        ylabel = "Agents (count)",
        yticks = 0:10:n_agents_guess
    )
    xlims!(ax, -0.5, 3.5)
    ylims!(ax, 0, n_agents_guess)
    xt = collect(0.0:0.5:3.0)
    ax.xticks = xt
    ax.xtickformat = mytickfmt
    ax.xticklabelrotation = pi/4

    frame_obs = Observable(0)
    lbl = Label(fig, @lift("Time elapsed = $((($frame_obs-1)*dt)) s"),
                fontsize = 16, padding = (6,10,6,10), halign = :left)
    fig[1,1, TopLeft()] = lbl

    y_inner = Observable(zeros(Float64, length(centers_inner)))
    y0      = Observable([0.0])
    y3      = Observable([0.0])

    inner_cols = [Makie.to_color(x < 1.0 ? :green : :orange) for x in centers_inner]

    barplot!(ax, centers_inner, y_inner; width = bin_w_inner, color = inner_cols, strokewidth = 0)
    barplot!(ax, [x0_pos], y0; width = bin_w_edge, color = :gray35,  strokewidth = 0)
    barplot!(ax, [x3_pos], y3; width = bin_w_edge, color = :crimson, strokewidth = 0)

    out_file = joinpath(folder, "FlowField_SCENARIO_3_hist__$(n_agents)_$(seed_str)_$(cost_metric_str)_$(run_timestamp).mp4")
    record(fig, out_file, 1:T_play; framerate = 30) do frame
        frame_obs[] = frame
        tl = Vector(df[df.step .== frame, :toxicload])
        count0       = count(v -> v <= 1e-9 || isapprox(v, 0.0; atol = 1e-6), tl)
        count3       = count(v -> v >= 3.0 - 1e-9 || isapprox(v, 3.0; atol = 1e-4), tl)
        counts_inner = bin_counts_open(tl, edges_inner)
        y_inner[] = Float64.(counts_inner)
        y0[]      = [Float64(count0)]
        y3[]      = [Float64(count3)]
    end

    println("Βίντεο με ξεχωριστά bins για 0 & 3 και εσωτερικά ανά 0.5 σώθηκε ως: $out_file")
end
