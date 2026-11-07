using Test
using JSON3, HTTP, Base64
include(joinpath(@__DIR__, "..", "LiveService.jl"))
using .LiveService
const LS = LiveService

@testset "polygon fill" begin
    m = LS.fill_polygon!(falses(10, 10), [(2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0)], true)
    @test count(m) == 36                                     # τετράγωνο 6×6
    big = LS.fill_polygon!(falses(10, 10), [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)], true)
    @test count(big) == 100
    tri = LS.fill_polygon!(falses(10, 10), [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)], true)
    @test 40 <= count(tri) <= 60                             # ορθογώνιο τρίγωνο: περίπου το μισό
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
    @test count(LS.erode_box(M, 1)) == 81                    # διάβρωση τετραγώνου 11×11 -> 9×9
    @test LS.erode_box(M, 0) == M
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
