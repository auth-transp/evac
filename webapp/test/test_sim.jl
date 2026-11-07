using Test
using JSON3, HTTP, Base64
include(joinpath(@__DIR__, "..", "LiveService.jl"))
include(joinpath(@__DIR__, "..", "src", "Simulation.jl"))
using .LiveService
using .LiveSim
const LS = LiveService

req(path, body) = HTTP.Request("POST", path, [], body isa AbstractString ? Vector{UInt8}(body) : body)
resp(r) = JSON3.read(String(r.body))

# Περιοχή 600 × 400 m γύρω από το (25.90, 51.54) και στοιχεία OSM στη μορφή της απάντησης του Overpass (`out geom`):
# ένα κτίριο, ένας δρόμος που τέμνει το όριο και μια πύλη
const POLY = [(25.8982, 51.5370), (25.9018, 51.5370), (25.9018, 51.5430), (25.8982, 51.5430)]
geom(pts) = [Dict("lat" => p[1], "lon" => p[2]) for p in pts]
const ELEMENTS = JSON3.read(JSON3.write([
    Dict("type" => "way", "id" => 1, "tags" => Dict("building" => "yes"), "geometry" => geom([(25.8995, 51.5395), (25.9005, 51.5395), (25.9005, 51.5405), (25.8995, 51.5405), (25.8995, 51.5395)])),
    Dict("type" => "way", "id" => 2, "tags" => Dict("highway" => "service"), "geometry" => geom([(25.8990, 51.5360), (25.8990, 51.5440)])),
    Dict("type" => "node", "id" => 4, "lat" => 25.9010, "lon" => 51.5370, "tags" => Dict("barrier" => "gate")),
]))
# ο Overpass αντικαθίσταται με τα παραπάνω στοιχεία (χωρίς δίκτυο)
Core.eval(LS, :(fetch_overpass(q::AbstractString; mirrors = OVERPASS_MIRRORS, rounds::Int = 2, pause::Real = 4.0, hedge::Real = 15.0,
                              cache_ttl::Real = OVERPASS_CACHE_TTL_S) = Main.JSON3.read("{\"elements\":" * Main.JSON3.write(Main.ELEMENTS) * "}")))

# project του Live Service από δεδομένα OSM: εισαγωγή, προτεινόμενες έξοδοι (πύλη και δρόμος δυτικά, δρόμος ανατολικά) και μία ζώνη spawn
function make_project(n_agents; target_mpp = 3.46)
    body = Dict{String,Any}("polygon" => [collect(p) for p in POLY])
    target_mpp === nothing || (body["target_mpp"] = target_mpp)                     # nothing: αυτόματη ανάλυση του grid
    r = resp(LS.osm_route(req("/api/osm", JSON3.write(body))))
    s = 3.46 / r.grid.mpp                                                           # η ζώνη spawn ορίζεται σε μέτρα (στο grid των 3.46 m: [20..50] × [70..100] cells)
    build = JSON3.write(Dict("id" => r.id, "grid" => r.grid, "labels" => r.labels, "n_agents" => n_agents, "exits" => r.exits,
        "spawn_zones" => [Dict("points" => [[20 * s, 70 * s], [50 * s, 70 * s], [50 * s, 100 * s], [20 * s, 100 * s]])]))
    br = resp(LS.build_route(req("/api/build", build)))
    return r.id, joinpath(LS.DATA_DIR, r.id), br
end

# περιμένει να αλλάξει η κατάσταση του job σε κάτι εκτός των starting/running (ή λήξη χρόνου)
function wait_job(id; timeout = 240)
    t0 = time()
    while time() - t0 < timeout
        st = resp(LS.job_status_route(id))
        st.status in ("starting", "running") || return st
        sleep(2)
    end
    return resp(LS.job_status_route(id))
end

@testset "τοξικό φορτίο ίδιο με το Scenario 3" begin
    # τα αυθεντικά setupToxic/tl_rate/update_toxic_load του σεναρίου εκτελούνται σε ξεχωριστό module και συγκρίνονται
    src = read(joinpath(@__DIR__, "..", "..", "SCENARIOS", "SCENARIO 3", "DStarLite - Scenario 3.jl"), String)
    i0 = first(findfirst("function setupToxic()", src))
    i1 = first(findfirst("# Επιθυμητή κατεύθυνση e_α", src))
    ref = Module(:ScenarioRef)
    Core.eval(ref, :(using Dates))
    Base.include_string(ref, src[i0:i1-1], "Scenario3-toxic")
    tab = LiveSim.setup_toxic()
    @test tab.Balpha == ref.Balpha && tab.Btime == ref.Btime && tab.Brho == Matrix(ref.Brho)
    rhos = vcat(10 .^ range(-3, 5; length = 400), tab.Brho[:])                 # και ακριβώς στα όρια των ζωνών
    @test all(LiveSim.tl_rate(tab, r, k) == Base.invokelatest(ref.tl_rate, r, k) for r in rhos, k in 1:3)
    @test Base.invokelatest(ref.update_toxic_load, 50.0, [0.1, 0.2, 0.3], 1.0) == LiveSim.update_toxic_load(tab, 50.0, [0.1, 0.2, 0.3], 1.0)
end

@testset "κλιμάκωση κόστους διαδρομής με τη διάσταση του cell" begin
    hm = [0 102 255; 153 0 195]
    pm = zeros(2, 3)
    @test LiveSim.path_costs(hm, pm, 1.0) == hm                                 # στα 3.46 m/pixel όπως πριν
    @test LiveSim.path_costs(hm, pm, 0.5) == round.(Int, hm .* 0.5)
    @test LiveSim.path_costs(hm, [0.0 0.0 0.0; 0.0 300.0 0.0], 0.5)[2, 2] == round(Int, (0 + max(300.0, LiveSim.MIN_CM_PENALTY)) * 0.5)
    # κόστος ανά μέτρο ίδιο σε κάθε ανάλυση: διασχίζοντας 100 m εδάφους (153 ανά cell των 3.46 m) το ίδιο συνολικό κόστος ανεξάρτητα από το mpp
    for mpp in (1.0, 2.0, 3.46)
        k = mpp / LiveSim.COST_REFERENCE_MPP
        per_cell = LiveSim.path_costs(fill(102, 1, 1) .* 0 .+ 255 .- 102, zeros(1, 1), k)[1, 1]          # κόστος εδάφους ανά cell
        @test abs(per_cell * (100 / mpp) - 153 * (100 / 3.46)) / (153 * (100 / 3.46)) < 0.02
    end
end

@testset "ανίχνευση σκαλώματος" begin
    tr = LiveSim.StuckTracker(15, 0.578)                                        # 2 m = 0.578 pixels στο προεπιλεγμένο grid
    @test LiveSim.check!(tr, 1, (10.0, 10.0), 1, 0.39) == 0                     # πρώτο frame: αρχικοποίηση
    @test LiveSim.check!(tr, 1, (10.0, 10.0), 10, 0.39) == 0                    # δεν πέρασε το παράθυρο
    @test LiveSim.check!(tr, 1, (10.1, 10.0), 16, 0.39) == 1                    # σχεδόν ακίνητος: στάδιο 1
    @test LiveSim.check!(tr, 1, (10.1, 10.0), 31, 0.39) == 2
    @test LiveSim.check!(tr, 1, (10.1, 10.0), 46, 0.39) == 3
    @test LiveSim.check!(tr, 1, (15.0, 10.0), 61, 0.39) == 0                    # κινήθηκε: μηδενισμός σταδίου
    @test LiveSim.check!(tr, 1, (15.0, 10.0), 76, 0.39) == 1                    # ξανασκαλώνει: ξεκινά πάλι από 1
    tr2 = LiveSim.StuckTracker(15, 0.578)                                       # επιθυμητή ταχύτητα τόσο μικρή που δεν θα μπορούσε να κινηθεί: δεν μετρά
    LiveSim.check!(tr2, 2, (5.0, 5.0), 1, 0.01)
    @test LiveSim.check!(tr2, 2, (5.0, 5.0), 16, 0.01) == 0
    tr3 = LiveSim.StuckTracker(15, 0.578)                                       # κίνηση πάνω από το όριο: δεν είναι σκαλωμένος
    LiveSim.check!(tr3, 3, (5.0, 5.0), 1, 0.39)
    @test LiveSim.check!(tr3, 3, (5.0, 5.7), 16, 0.39) == 0
end

@testset "ξεκόλλημα στην προσομοίωση (αναγκαστικό)" begin
    id, dir, br = make_project(20)
    res = LiveSim.run_simulation(dir; T = 60, seed = 3, stuck_window_s = 10, stuck_force = true)   # κάθε 10 s όλοι θεωρούνται σκαλωμένοι
    @test res.status == "done"
    @test res.stuck_events >= 3 * 20 * 0.5 && res.stuck_agents >= 10           # 6 έλεγχοι... πολλές παρεμβάσεις
    actions = Set(e.action for e in res.stuck_log)
    @test "replan" in actions && "push" in actions                              # στάδια 1 και 2 (το 3ο: teleport, αν υπάρχει επόμενο σημείο διαδρομής)
    @test res.stuck_teleports >= 1 && res.stuck_pushes >= 1                      # και το στάδιο 3: μετακίνηση στο επόμενο σημείο της διαδρομής
    @test res.stuck_teleports + res.stuck_pushes <= res.stuck_events
    fr = reinterpret(Float32, read(joinpath(dir, "frames.bin")))
    @test all(isfinite, fr) && filesize(joinpath(dir, "frames.bin")) == 60 * 20 * 16
    meta = JSON3.read(read(joinpath(dir, "meta.json"), String))
    @test all(0 .<= fr[1:4:end] .<= meta.cols) && all(0 .<= fr[2:4:end] .<= meta.rows)   # όλοι οι agents μέσα στο grid και μετά τις παρεμβάσεις
    @test any(==(2.0f0), fr[4:4:end])                                           # σημαίνονται (κατάσταση 2) όσοι μόλις ξεκόλλησαν
    rm(dir; recursive = true)
end

@testset "αυτόματη ανάλυση 1 m/pixel: project, πόρτες και προσομοίωση" begin
    id, dir, br = make_project(12; target_mpp = nothing)                          # χωρίς target_mpp: 600 × 400 m -> 1 m/pixel
    @test br.config.grid.mpp == 1.0 && br.config.cols >= 600 && br.config.rows >= 398          # ~600 × 398 m
    res = LiveSim.run_simulation(dir; T = 40, seed = 5)
    @test res.status == "done" && res.n_agents == 12
    @test res.meters_per_pixel ≈ 1.0 && res.cost_scale ≈ 1.0 / 3.46               # κόστος ανά cell κλιμακωμένο στο cell του 1 m
    fr = reinterpret(Float32, read(joinpath(dir, "frames.bin")))
    meta = JSON3.read(read(joinpath(dir, "meta.json"), String))
    @test all(isfinite, fr) && all(0 .<= fr[1:4:end] .<= meta.cols) && all(0 .<= fr[2:4:end] .<= meta.rows)
    @test meta.goal_radius ≈ 5.0                                                  # 5 m = 5 pixels στο 1 m/pixel
    rm(dir; recursive = true)
end

@testset "χωρίς σκάλωμα η κανονική εκτέλεση δεν κάνει παρεμβάσεις" begin
    id, dir, br = make_project(15)
    res = LiveSim.run_simulation(dir; T = 40, seed = 7)
    @test res.stuck_events == 0 && res.stuck_agents == 0 && res.stuck_window_s == 15 && res.stuck_dist_m == 2.0
    rm(dir; recursive = true)
end

@testset "προσομοίωση από project του Live Service (απευθείας κλήση)" begin
    id, dir, br = make_project(20)
    @test all(e -> e.goal_cell !== nothing, br.config.exits)
    T = 40
    res = LiveSim.run_simulation(dir; T = T, seed = 7)
    @test res.status == "done" && res.n_agents == 20
    meta = JSON3.read(read(joinpath(dir, "meta.json"), String))
    @test meta.stride == 4 && meta.rows == br.config.rows && meta.cols == br.config.cols
    @test meta.goal_radius_m == 5.0 && meta.goal_radius ≈ 5.0 / meta.meters_per_pixel      # ακτίνα διαφυγής 5 m
    @test filesize(joinpath(dir, "frames.bin")) == T * 20 * 4 * 4              # frames × agents × 4 Float32
    fr = reinterpret(Float32, read(joinpath(dir, "frames.bin")))
    @test all(isfinite, fr)
    xs, ys = fr[1:4:end], fr[2:4:end]
    @test all(0 .<= xs .<= meta.cols) && all(0 .<= ys .<= meta.rows)           # όλοι οι agents μέσα στο grid
    @test length(readlines(joinpath(dir, "trajectories.csv"))) == 1 + T * 20
    @test JSON3.read(read(joinpath(dir, "progress.json"), String)).status == "done"
    @test all(isfile(joinpath(dir, "cm_$k.png")) for k in 1:7)
    first_f, last_f = fr[1:80], fr[end-79:end]                                 # οι agents κινούνται
    @test maximum(abs.(first_f[1:4:end] .- last_f[1:4:end]) .+ abs.(first_f[2:4:end] .- last_f[2:4:end])) > 1
    rm(dir; recursive = true)
end

@testset "job σε ξεχωριστή διεργασία: εκκίνηση, πρόοδος, ακύρωση" begin
    id, dir, _ = make_project(15)
    r = resp(LS.run_route(req("/api/run", JSON3.write(Dict("id" => id, "T" => 30, "seed" => 1)))))
    @test r.status == "starting"
    @test_throws ErrorException LS.run_route(req("/api/run", JSON3.write(Dict("id" => id, "T" => 30))))   # μία κάθε φορά
    st = wait_job(id)
    @test st.status == "done"
    @test st.frame == 30
    res = JSON3.read(read(joinpath(dir, "result.json"), String))
    @test res.n_agents == 15 && res.T == 30
    @test filesize(joinpath(dir, "frames.bin")) == 30 * 15 * 16

    # ακύρωση: μεγάλο T, kill της διεργασίας, νέα εκτέλεση μετά την ακύρωση
    resp(LS.run_route(req("/api/run", JSON3.write(Dict("id" => id, "T" => 20000)))))
    sleep(3)
    @test resp(LS.job_cancel_route(id)).status == "cancelled"
    sleep(1)
    @test LS.busy_job() === nothing
    @test resp(LS.job_status_route(id)).status == "cancelled"
    rm(dir; recursive = true)
end
