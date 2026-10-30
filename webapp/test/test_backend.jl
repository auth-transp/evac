using Test
using Images, FileIO, JSON3, HTTP, Base64
include(joinpath(@__DIR__, "..", "LiveService.jl"))
using .LiveService
const LS = LiveService

@testset "polygon fill" begin
    g = LS.Grid(0, 0, 1.0, 10, 10, 1.0)
    m = LS.fill_polygon!(falses(10, 10), [(2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0)], true)
    @test count(m) == 36                                     # τετράγωνο 6×6
    polys = JSON3.read("""[{"mode":"add","points":[[0,0],[10,0],[10,10],[0,10]]},{"mode":"subtract","points":[[4,4],[6,4],[6,6],[4,6]]}]""")
    @test count(LS.region_mask(polys, g)) == 100 - 4         # οπή 2×2
end

@testset "rotated exit rectangle" begin
    g = LS.Grid(0, 0, 1.0, 40, 40, 2.0)
    e = JSON3.read("""{"p1":[10,20],"p2":[30,20],"thickness_m":8}""")
    @test count(LS.exit_mask(e, g)) == 20 * 4                # 20 px × (8 m / 2 m) πάχος
    e2 = JSON3.read("""{"p1":[10,10],"p2":[30,30],"thickness_m":8}""")
    @test count(LS.exit_mask(e2, g)) > 0
end

@testset "morphology" begin
    M = falses(20, 20); M[5:15, 5:15] .= true
    @test count(LS.erode_box(M, 1)) == 81
    @test count(LS.dilate_box(M, 1)) == 169
    @test LS.opening_box(M, 2) == M                          # το τετράγωνο 11×11 επιβιώνει
    thin = falses(20, 20); thin[10, 2:18] .= true
    @test count(LS.opening_box(thin, 1)) == 0                # η γραμμή πάχους 1 εξαφανίζεται
end

@testset "propose on synthetic satellite image" begin
    mpp = 3.0
    h, w = 120, 160
    img = fill(RGB{N0f8}(0.80, 0.70, 0.52), h, w)            # άμμος
    img[58:62, 10:150] .= RGB{N0f8}(0.45, 0.45, 0.47)        # δρόμος πάχους 5 pixels
    img[20:45, 30:60] .= RGB{N0f8}(0.92, 0.92, 0.92)         # κτίριο
    img[75:100, 90:130] .= RGB{N0f8}(0.30, 0.30, 0.32)       # κτίριο (σκούρο)
    g = LS.Grid(0, 0, 1.0, w, h, mpp)
    region = falses(h, w); region[5:115, 5:155] .= true
    L = LS.propose_labels(img, region, g)
    @test all(L[.!region] .== LS.CLASS_OUTSIDE)
    @test L[60, 80] == LS.CLASS_ROAD
    @test L[32, 45] == LS.CLASS_BUILDING
    @test L[20, 45] == LS.CLASS_WALL || L[21, 45] == LS.CLASS_WALL
    @test L[88, 110] == LS.CLASS_BUILDING
    @test L[10, 10] == LS.CLASS_GROUND
end

@testset "compose heightmap + reachability" begin
    labels = fill(LS.CLASS_GROUND, 30, 30)
    labels[1, :] .= LS.CLASS_WALL; labels[30, :] .= LS.CLASS_WALL; labels[:, 1] .= LS.CLASS_WALL; labels[:, 30] .= LS.CLASS_WALL
    exits = falses(30, 30); exits[15, 1] = true               # άνοιγμα στον τοίχο
    hm = LS.compose_heightmap(labels, exits)
    @test hm[1, 5] == 0 && hm[15, 1] == LS.HEIGHT_GROUND && hm[10, 10] == 102
    walk = LS.walkable(hm)
    @test !walk[1, 5] && walk[15, 1] && walk[10, 10]
    # κανόνας του σεναρίου: (255 - τιμή) > 245 είναι μη βατό
    @test all((255 .- Int.(hm) .> 245) .== .!walk)
end

@testset "end-to-end routes" begin
    dir = mktempdir()
    img = fill(RGB{N0f8}(0.80, 0.70, 0.52), 100, 120); img[40:60, 40:80] .= RGB{N0f8}(0.9, 0.9, 0.9)
    save(joinpath(dir, "a.png"), img)
    bytes = read(joinpath(dir, "a.png"))
    r = LS.upload_route(HTTP.Request("POST", "/api/upload", [], bytes))
    up = JSON3.read(String(r.body))
    @test up.width == 120 && up.height == 100
    body = JSON3.write(Dict("id" => up.id, "src_mpp" => 3.0, "target_mpp" => 3.0,
        "region" => [Dict("mode" => "add", "points" => [[5, 5], [115, 5], [115, 95], [5, 95]])]))
    pr = JSON3.read(String(LS.propose_route(HTTP.Request("POST", "/api/propose", [], body)).body))
    @test pr.grid.w == 110 && pr.grid.h == 90
    @test length(base64decode(pr.labels)) == 110 * 90
    build = JSON3.write(Dict("id" => up.id, "grid" => pr.grid, "labels" => pr.labels, "n_agents" => 50,
        "exits" => [Dict("p1" => [5, 40], "p2" => [5, 50], "thickness_m" => 8)],
        "spawn_zones" => [Dict("points" => [[20, 20], [40, 20], [40, 35], [20, 35]])]))
    br = JSON3.read(String(LS.build_route(HTTP.Request("POST", "/api/build", [], build)).body))
    @test isfile(joinpath(LS.DATA_DIR, up.id, "heightmap.png"))
    @test isempty(br.config.warnings)
    hm = load(joinpath(LS.DATA_DIR, up.id, "heightmap.png"))
    @test size(hm) == (90, 110)
    # έξοδος εντελώς έξω από την περιοχή: δεν συνδέεται με τον βατό χώρο -> προειδοποίηση
    far = JSON3.write(Dict("id" => up.id, "grid" => pr.grid, "labels" => pr.labels, "n_agents" => 50,
        "exits" => [Dict("p1" => [0, 40], "p2" => [0, 50], "thickness_m" => 3)],
        "spawn_zones" => [Dict("points" => [[20, 20], [40, 20], [40, 35], [20, 35]])]))
    fr = JSON3.read(String(LS.build_route(HTTP.Request("POST", "/api/build", [], far)).body))
    @test any(w -> occursin("Η έξοδος 1", w), fr.config.warnings)
    rm(joinpath(LS.DATA_DIR, up.id); recursive = true)
end
