using Test
using JSON3, HTTP, Base64
include(joinpath(@__DIR__, "..", "LiveService.jl"))
using .LiveService
const LS = LiveService

# Περιοχή 600 × 400 m γύρω από το (25.90, 51.54), ως [[lat, lon], ...]
const POLY = [(25.8982, 51.5370), (25.9018, 51.5370), (25.9018, 51.5430), (25.8982, 51.5430)]

geom(pts) = [Dict("lat" => p[1], "lon" => p[2]) for p in pts]
rect(s, w, n, e) = [(s, w), (n, w), (n, e), (s, e), (s, w)]            # κλειστό ορθογώνιο (lat, lon)

# Στοιχεία στη μορφή της απάντησης του Overpass με `out geom`
const ELEMENTS = JSON3.read(JSON3.write([
    Dict("type" => "way", "id" => 1, "tags" => Dict("building" => "yes"), "geometry" => geom(rect(25.8995, 51.5395, 25.9005, 51.5405))),
    Dict("type" => "way", "id" => 2, "tags" => Dict("highway" => "service"), "geometry" => geom([(25.8990, 51.5360), (25.8990, 51.5440)])),
    Dict("type" => "way", "id" => 3, "tags" => Dict("highway" => "footway"), "geometry" => geom([(25.9010, 51.5380), (25.9010, 51.5420)])),
    Dict("type" => "node", "id" => 4, "lat" => 25.9010, "lon" => 51.5370, "tags" => Dict("barrier" => "gate")),
    Dict("type" => "way", "id" => 5, "tags" => Dict("man_made" => "storage_tank"), "geometry" => geom(rect(25.8985, 51.5410, 25.8988, 51.5414))),
    Dict("type" => "relation", "id" => 6, "tags" => Dict("building" => "yes", "type" => "multipolygon"), "members" => [
        Dict("type" => "way", "role" => "outer", "geometry" => geom(rect(25.9008, 51.5410, 25.9014, 51.5425))),
        Dict("type" => "way", "role" => "inner", "geometry" => geom(rect(25.9010, 51.5414, 25.9012, 51.5420)))]),
]))

# κελί του grid για σημείο (lat, lon): το grid αρχίζει στην πάνω-αριστερή γωνία (βορειοδυτικά) του ορθογωνίου POLY
function cell(r, lat, lon)
    g = r.grid
    x = (lon - 51.5370) * cosd(25.9) * 111320.0 / g.mpp
    y = -(lat - 25.9018) * 110574.0 / g.mpp
    return clamp(floor(Int, y) + 1, 1, g.h), clamp(floor(Int, x) + 1, 1, g.w)
end
ground(r, lat, lon) = r.labels[cell(r, lat, lon)...]

@testset "build_from_osm: κλάσεις και στατιστικά" begin
    r = LS.build_from_osm(POLY, ELEMENTS; target_mpp = 3.46)
    @test abs(r.grid.w - 600.6 / 3.46) <= 2 && abs(r.grid.h - 400.3 / 3.46) <= 2
    @test r.counts.buildings == 3 && r.counts.roads == 1 && r.counts.gates == 1       # footway δεν μετρά ως δρόμος
    @test ground(r, 25.9000, 51.5400) == LS.CLASS_BUILDING                            # κέντρο κτιρίου
    @test ground(r, 25.90048, 51.5400) in (LS.CLASS_WALL, LS.CLASS_BUILDING)          # άκρη κτιρίου (τοίχος 1 cell, ή πόρτα· ανά 60 m τοίχου υπάρχει πόρτα)
    @test ground(r, 25.9007, 51.5400) == LS.CLASS_GROUND                              # λίγο έξω από το κτίριο
    @test count(==(LS.CLASS_WALL), r.labels) > 0
    @test ground(r, 25.8990, 51.5390) == LS.CLASS_ROAD                                # δρόμος service
    @test ground(r, 25.9010, 51.5400) == LS.CLASS_GROUND                              # το footway δεν είναι δρόμος
    @test ground(r, 25.8986, 51.5412) in (LS.CLASS_BUILDING, LS.CLASS_WALL)           # δεξαμενή
    @test ground(r, 25.9011, 51.5417) == LS.CLASS_GROUND                              # αυλή (inner) του multipolygon
    @test ground(r, 25.9011, 51.5412) in (LS.CLASS_BUILDING, LS.CLASS_WALL)           # δακτύλιος του multipolygon
    @test isempty(r.warnings)
end

@testset "build_from_osm: προτάσεις εξόδων" begin
    r = LS.build_from_osm(POLY, ELEMENTS; target_mpp = 3.46)
    @test length(r.exits) == 3                                                        # πύλη (δυτικά) + δρόμος δυτικά + δρόμος ανατολικά
    xs = [(e["p1"][1] + e["p2"][1]) / 2 for e in r.exits]
    @test count(x -> x < 6, xs) == 2 && count(x -> x > r.grid.w - 6, xs) == 1          # δύο στο δυτικό όριο, μία στο ανατολικό
    @test all(e -> e["thickness_m"] == 8.0, r.exits)
    # οι έξοδοι είναι κατά μήκος του ορίου: τα άκρα τους έχουν σχεδόν ίδιο x (κατακόρυφο όριο)
    @test all(e -> abs(e["p1"][1] - e["p2"][1]) < 1.0 && hypot(e["p1"][2] - e["p2"][2], 0) > 1.0, r.exits)
    # χωρίς κτίρια/δρόμους/πύλες: προειδοποιήσεις και μόνο έδαφος
    e = LS.build_from_osm(POLY, JSON3.read("[]"); target_mpp = 3.46)
    @test length(e.warnings) == 3 && all(c -> c in (LS.CLASS_GROUND, LS.CLASS_OUTSIDE), e.labels)
end

@testset "έλεγχοι μεγέθους περιοχής και αποτυχία δικτύου" begin
    big = [(25.0, 51.0), (25.0, 51.2), (25.2, 51.2), (25.2, 51.0)]
    @test_throws ErrorException LS.osm_grid(big)
    @test_throws ErrorException LS.osm_grid([(25.9, 51.5), (25.9, 51.50001), (25.90001, 51.50001)])
    @test_throws ErrorException LS.fetch_overpass("[out:json];out;"; mirrors = ["http://127.0.0.1:9/api/interpreter"], rounds = 2, pause = 0.1)   # κλειστή θύρα
    # cache: αποθηκευμένη απάντηση επιστρέφεται χωρίς καμία σύνδεση (ο «server» είναι κλειστός)
    qc = "[out:json];/*test-cache-$(rand(UInt))*/out;"
    cf = LS.overpass_cache_file(qc)
    mkpath(dirname(cf)); write(cf, "{\"elements\":[{\"type\":\"node\",\"id\":1,\"lat\":1.0,\"lon\":2.0}]}")
    @test length(LS.fetch_overpass(qc; mirrors = ["http://127.0.0.1:9/api/interpreter"], rounds = 1).elements) == 1
    @test_throws ErrorException LS.fetch_overpass(qc; mirrors = ["http://127.0.0.1:9/api/interpreter"], rounds = 1, cache_ttl = -1)   # ληγμένη cache: ζητείται ξανά
    rm(cf; force = true)
    q = LS.overpass_query(25.89, 51.53, 25.91, 51.55)
    @test occursin("(25.89,51.53,25.91,51.55)", q) && occursin("out geom;", q) && occursin("highway", q)
end

@testset "αυτόματη ανάλυση του grid" begin
    @test LS.auto_mpp(300) == 1.0 && LS.auto_mpp(700) == 1.0                    # μικρές περιοχές: 1 m/pixel
    @test LS.auto_mpp(1400) ≈ 2.0                                               # ενδιάμεσες: ~700 cells στη μεγαλύτερη διάσταση
    @test LS.auto_mpp(2400) ≈ 2400 / 700 && LS.auto_mpp(2500) == 3.46           # μεγάλες: το βαθμονομημένο 3.46
    @test LS.auto_mpp(5000) == 3.46
    g, _ = LS.osm_grid(POLY)                                                    # 600 × 400 m -> 1 m/pixel
    @test g.mpp == 1.0 && abs(g.w - 600.6) <= 2 && abs(g.h - 400.3) <= 2
    g2, _ = LS.osm_grid(POLY; target_mpp = 3.46)                                # ρητή τιμή (δοκιμές/API)
    @test g2.mpp == 3.46
    wide = [(25.80, 51.50), (25.80, 51.575), (25.82, 51.575), (25.82, 51.50)]   # ~7.5 km πλάτος: ανώτατο όριο cells
    g3, _ = LS.osm_grid(wide; max_side = 1600)
    @test g3.w <= 1601 && g3.mpp > 3.46 && isapprox(g3.mpp, 7507 / 1600; atol = 0.05)
    big = [(25.80, 51.50), (25.80, 51.54), (25.81, 51.54), (25.81, 51.50)]      # ~4 km: 3.46 m/pixel
    @test LS.osm_grid(big)[1].mpp == 3.46
    r = LS.build_from_osm(POLY, ELEMENTS)                                       # χωρίς target_mpp: αυτόματη
    @test r.grid.mpp == 1.0 && r.counts.buildings == 3 && r.counts.roads == 1
end

@testset "hedged αιτήματα σε πολλούς server" begin
    ok_json = "{\"elements\":[{\"type\":\"node\",\"id\":7}]}"
    s1 = HTTP.serve!(req -> (sleep(4); HTTP.Response(200, ok_json)), "127.0.0.1", 8911; verbose = -1)    # αργός
    s2 = HTTP.serve!(req -> HTTP.Response(502, "bad"), "127.0.0.1", 8912; verbose = -1)                  # αποτυγχάνει αμέσως
    s3 = HTTP.serve!(req -> (sleep(0.5); HTTP.Response(200, ok_json)), "127.0.0.1", 8913; verbose = -1)  # γρήγορος
    mirrors = ["http://127.0.0.1:8911/x", "http://127.0.0.1:8912/x", "http://127.0.0.1:8913/x"]
    t0 = time()
    res, errs = LS.race_mirrors("q", mirrors; hedge = 0.4)
    dt = time() - t0
    @test res !== nothing && res.json.elements[1].id == 7
    @test dt < 3.0                                                              # δεν περίμενε τον αργό server (4 s)
    @test any(occursin("HTTP 502", e) for e in errs)
    res2, errs2 = LS.race_mirrors("q", [mirrors[2]]; hedge = 0.2)               # αποτυγχάνουν όλοι: nothing και το σφάλμα
    @test res2 === nothing && length(errs2) == 1 && occursin("502", errs2[1])
    for s in (s1, s2, s3)
        try HTTP.forceclose(s) catch; close(s) end
    end
end

@testset "/api/osm με προσομοιωμένο Overpass και build" begin
    Core.eval(LS, :(fetch_overpass(q::AbstractString; mirrors = OVERPASS_MIRRORS) = Main.JSON3.read("{\"elements\":" * Main.JSON3.write(Main.ELEMENTS) * "}")))
    body = JSON3.write(Dict("polygon" => [collect(p) for p in POLY], "target_mpp" => 3.46))
    r = JSON3.read(String(LS.osm_route(HTTP.Request("POST", "/api/osm", [], Vector{UInt8}(body))).body))
    @test r.counts.buildings == 3 && length(r.exits) == 3 && length(r.region) == 4
    @test length(base64decode(r.labels)) == r.grid.w * r.grid.h
    # το project που προκύπτει περνά κανονικά από το build (grid = εικόνα) και δίνει heightmap και config
    build = JSON3.write(Dict("id" => r.id, "grid" => r.grid, "labels" => r.labels, "n_agents" => 10, "exits" => r.exits,
        "spawn_zones" => [Dict("points" => [[60, 40], [100, 40], [100, 60], [60, 60]])]))
    br = JSON3.read(String(LS.build_route(HTTP.Request("POST", "/api/build", [], Vector{UInt8}(build))).body))
    @test isfile(joinpath(LS.DATA_DIR, r.id, "heightmap.png")) && length(br.config.exits) == 3
    @test all(e -> e.goal_cell !== nothing, br.config.exits)
    # έξοδος εντελώς έξω από την περιοχή: δεν συνδέεται με τον βατό χώρο -> προειδοποίηση
    far = JSON3.write(Dict("id" => r.id, "grid" => r.grid, "labels" => r.labels, "n_agents" => 10,
        "exits" => [Dict("p1" => [-6, 40], "p2" => [-6, 50], "thickness_m" => 3)],
        "spawn_zones" => [Dict("points" => [[60, 40], [100, 40], [100, 60], [60, 60]])]))
    fr = JSON3.read(String(LS.build_route(HTTP.Request("POST", "/api/build", [], Vector{UInt8}(far))).body))
    @test any(w -> occursin("Η έξοδος 1", w), fr.config.warnings)
    rm(joinpath(LS.DATA_DIR, r.id); recursive = true)
end
