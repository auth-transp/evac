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
    @test ground(r, 25.90048, 51.5400) == LS.CLASS_WALL                               # άκρη κτιρίου (τοίχος, 1 cell)
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
    @test_throws ErrorException LS.fetch_overpass("[out:json];out;"; mirrors = ["http://127.0.0.1:9/api/interpreter"])   # κλειστή θύρα
    q = LS.overpass_query(25.89, 51.53, 25.91, 51.55)
    @test occursin("(25.89,51.53,25.91,51.55)", q) && occursin("out geom;", q) && occursin("highway", q)
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
    rm(joinpath(LS.DATA_DIR, r.id); recursive = true)
end
