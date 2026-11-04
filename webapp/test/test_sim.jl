using Test
using Images, FileIO, JSON3, HTTP, Base64
include(joinpath(@__DIR__, "..", "LiveService.jl"))
include(joinpath(@__DIR__, "..", "src", "Simulation.jl"))
using .LiveService
using .LiveSim
const LS = LiveService

req(path, body) = HTTP.Request("POST", path, [], body isa AbstractString ? Vector{UInt8}(body) : body)
resp(r) = JSON3.read(String(r.body))

# project του Live Service με συνθετική εικόνα 120×100 (άμμος, δρόμος, δύο κτίρια), δύο εξόδους και δύο ζώνες spawn
function make_project(n_agents)
    img = fill(RGB{N0f8}(0.80, 0.70, 0.52), 100, 120)
    img[48:52, 5:115] .= RGB{N0f8}(0.45, 0.45, 0.47)
    img[15:35, 20:50] .= RGB{N0f8}(0.92, 0.92, 0.92)
    img[65:90, 70:105] .= RGB{N0f8}(0.30, 0.30, 0.32)
    tmp = joinpath(mktempdir(), "a.png")
    save(tmp, img)
    up = resp(LS.upload_route(req("/api/upload", read(tmp))))
    region = [Dict("mode" => "add", "points" => [[5, 5], [115, 5], [115, 95], [5, 95]])]
    pr = resp(LS.propose_route(req("/api/propose", JSON3.write(Dict("id" => up.id, "src_mpp" => 3.0, "target_mpp" => 3.0, "region" => region)))))
    build = JSON3.write(Dict("id" => up.id, "grid" => pr.grid, "labels" => pr.labels, "n_agents" => n_agents,
        "exits" => [Dict("p1" => [5, 45], "p2" => [5, 55], "thickness_m" => 8), Dict("p1" => [115, 45], "p2" => [115, 55], "thickness_m" => 8)],
        "spawn_zones" => [Dict("points" => [[20, 40], [60, 40], [60, 60], [20, 60]]), Dict("points" => [[70, 10], [100, 10], [100, 30], [70, 30]])]))
    br = resp(LS.build_route(req("/api/build", build)))
    return up.id, joinpath(LS.DATA_DIR, up.id), br
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
