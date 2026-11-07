using Test, JSON3, HTTP
include(joinpath(@__DIR__, "..", "LiveService.jl"))
using .LiveService
const LS = LiveService

# Ροή όπως του D* Lite (Moore neighborhood, ΧΩΡΙΣ έλεγχο κοπής γωνίας): ποια cells φτάνει κανείς από τα seeds περνώντας μόνο από βατά cells
function reach(passable::BitMatrix, seeds)
    h, w = size(passable)
    seen = falses(h, w)
    stack = Tuple{Int,Int}[s for s in seeds if passable[s...]]
    foreach(s -> seen[s...] = true, stack)
    while !isempty(stack)
        (i, j) = pop!(stack)
        for di in -1:1, dj in -1:1
            ni, nj = i + di, j + dj
            (1 <= ni <= h && 1 <= nj <= w && passable[ni, nj] && !seen[ni, nj]) || continue
            seen[ni, nj] = true
            push!(stack, (ni, nj))
        end
    end
    return seen
end

# μάσκες σχημάτων: δίσκος, ρόμβος (διαγώνιες πλευρές), ορθογώνιο
disk(h, w, ci, cj, r) = BitMatrix([hypot(i - ci, j - cj) <= r for i in 1:h, j in 1:w])
diamond(h, w, ci, cj, r) = BitMatrix([abs(i - ci) + abs(j - cj) <= r for i in 1:h, j in 1:w])
rectm(h, w, i0, i1, j0, j1) = (m = falses(h, w); m[i0:i1, j0:j1] .= true; m)

@testset "τοίχοι: λεπτότεροι και χωρίς διαρροές" begin
    for (name, b) in (("δίσκος", disk(60, 60, 30, 30, 18)), ("ρόμβος", diamond(60, 60, 30, 30, 20)), ("ορθογώνιο", rectm(60, 60, 12, 45, 10, 50)),
                      ("δίσκος 2", disk(60, 60, 28.5, 31.2, 11.3)))
        walls = LS.building_walls(b)
        interior = b .& .!walls
        @test any(interior)
        # κλειστός: από έξω δεν φτάνει κανείς στο εσωτερικό ούτε διαγώνια (ο D* Lite επιτρέπει διαγώνιες κινήσεις)
        passable = BitMatrix(.!walls)
        seen = reach(passable, [(1, 1)])
        @test !any(seen .& interior)
        @test all(b[walls])
        # ελαχιστότητα: αν αφαιρεθεί οποιοδήποτε cell του τοίχου, δημιουργείται διαρροή (ο τοίχος δεν γίνεται λεπτότερος)
        leaks = 0
        for c in findall(walls)
            w2 = copy(walls); w2[c] = false
            any(reach(BitMatrix(.!w2), [(1, 1)]) .& interior) && (leaks += 1)
        end
        @test leaks >= 0.95 * count(walls)                                          # σχεδόν κάθε cell είναι απαραίτητο (λίγα περιττά μόνο στις ακραίες κορυφές)
    end
end

# εικόνα κλάσεων 40×60: κτίριο (τοίχος + εσωτερικό) στο κέντρο, δρόμος στο νότο
function scene(; road = true)
    h, w = 40, 60
    b = rectm(h, w, 8, 24, 15, 45)
    walls = LS.building_walls(b)
    labels = fill(LS.CLASS_GROUND, h, w)
    road && (labels[34:35, :] .= LS.CLASS_ROAD)
    labels[b .& .!walls] .= LS.CLASS_BUILDING
    labels[walls] .= LS.CLASS_WALL
    return labels
end

@testset "πόρτες" begin
    labels = scene()
    interior = labels .== LS.CLASS_BUILDING
    walkable_cells(l) = BitMatrix((l .== LS.CLASS_GROUND) .| (l .== LS.CLASS_ROAD) .| (l .== LS.CLASS_BUILDING))
    @test !any(reach(walkable_cells(labels), [(1, 1)]) .& interior)             # πριν τις πόρτες το κτίριο είναι κλειστό
    # το κτίριο έχει 88 cells τοίχου που συνορεύουν με τον ανοιχτό χώρο: 88 × 3.46 m ≈ 304 m -> τουλάχιστον μία πόρτα ανά 60 m = 6 πόρτες
    expected = ceil(Int, 88 * 3.46 / LS.DOOR_SPACING_M)
    n_ent, n_auto = LS.open_doors!(labels, [], 3.46)
    @test (n_ent, n_auto) == (0, expected) && expected == 6
    @test all(reach(walkable_cells(labels), [(1, 1)])[interior])                # από έξω φτάνει σε όλο το εσωτερικό
    door_mask = BitMatrix([labels[i, j] == LS.CLASS_BUILDING && !interior[i, j] for i in axes(labels, 1), j in axes(labels, 2)])
    dl, nd = LS.components(door_mask)
    @test nd == 6                                                               # 6 ξεχωριστές πόρτες
    @test any(c -> c[1] == 24, findall(door_mask))                              # η πρώτη στον νότιο τοίχο, προς τον δρόμο
    wc = LS.door_width_cells(3.46)                                              # 6 m πλάτος πόρτας -> 2 cells στο 3.46 m/pixel
    @test wc == 2
    @test count(door_mask) >= 6 * wc && maximum(count(==(k), dl) for k in 1:nd) <= wc * wc   # κάθε πόρτα έως wc cells πλάτος (block wc×wc στις γωνίες)
    centres = [(sum(c[1] for c in findall(dl .== k)) / count(==(k), dl), sum(c[2] for c in findall(dl .== k)) / count(==(k), dl)) for k in 1:nd]
    @test minimum(hypot(a[1] - b[1], a[2] - b[2]) for a in centres for b in centres if a != b) >= 4   # ομοιόμορφα κατανεμημένες, όχι κολλητά
    @test LS.open_doors!(labels, [], 3.46) == (0, 0)                            # ήδη αρκετές: δεν προστίθενται άλλες

    # είσοδος OSM στον βόρειο τοίχο: μετρά ως μία από τις πόρτες, οι υπόλοιπες συμπληρώνονται αυτόματα
    labels = scene()
    n_ent, n_auto = LS.open_doors!(labels, [(30.0, 7.5)], 3.46)                 # (x, y) του grid, κοντά στον βόρειο τοίχο (γραμμή 8)
    @test (n_ent, n_auto) == (1, 5)
    @test count(c -> c[1] == 8 && !interior[c] && labels[c] == LS.CLASS_BUILDING, findall(labels .== LS.CLASS_BUILDING)) >= LS.door_width_cells(3.46)

    # χωρίς δρόμο ανοίγουν πάλι πόρτες προς τον ανοιχτό χώρο
    labels = scene(road = false)
    @test LS.open_doors!(labels, [], 3.46)[2] == 6
    # είσοδος μακριά από τοίχο (πάνω από 12 m) δεν ανοίγει πόρτα στον τοίχο: όλες αυτόματες
    labels = scene()
    @test LS.open_doors!(labels, [(2.0, 2.0)], 3.46) == (0, 6)
end

# πλάτος πόρτας σε μέτρα: 6 m = 3 agents (1 m) δίπλα-δίπλα με χώρο για την άπωση του SFM· ο αριθμός cells προκύπτει από την ανάλυση
@testset "πλάτος πόρτας σε μέτρα" begin
    @test LS.door_width_cells(1.0) == 6 && LS.door_width_cells(3.46) == 2 && LS.door_width_cells(6.0) == 1 && LS.door_width_cells(10.0) == 1
    @test LS.door_width_cells(0.5) == 12 && LS.door_width_cells(2.0) == 3
    # κτίριο 41 × 31 m στο 1 m/pixel: περίμετρος ~136 m -> 3 πόρτες των 6 cells
    h, w = 80, 100
    b = rectm(h, w, 20, 60, 20, 50)
    walls = LS.building_walls(b)
    labels = fill(LS.CLASS_GROUND, h, w)
    labels[b .& .!walls] .= LS.CLASS_BUILDING
    labels[walls] .= LS.CLASS_WALL
    interior = labels .== LS.CLASS_BUILDING
    n_ent, n_auto = LS.open_doors!(labels, [], 1.0)
    @test n_auto == ceil(Int, 136 / LS.DOOR_SPACING_M) == 3
    door_mask = BitMatrix([labels[i, j] == LS.CLASS_BUILDING && !interior[i, j] for i in axes(labels, 1), j in axes(labels, 2)])
    dl, nd = LS.components(door_mask)
    @test nd == 3
    @test all(count(==(k), dl) >= 4 for k in 1:nd)                              # κάθε πόρτα ~6 cells (λιγότερα μόνο αν πέφτει σε γωνία)
    @test maximum(count(==(k), dl) for k in 1:nd) == 6
end

# κλειστός ανοιχτός χώρος μέσα σε κτίριο (αγωνιστικός χώρος): δακτύλιος κτιρίου με αυλή. Ο αγωνιστικός χώρος δεν πρέπει να μένει παγιδευμένος.
@testset "εσωτερικός ανοιχτός χώρος" begin
    h, w = 70, 70
    outer = rectm(h, w, 8, 60, 8, 60)
    inner = rectm(h, w, 22, 46, 22, 46)
    b = outer .& .!inner
    walls = LS.building_walls(b)
    labels = fill(LS.CLASS_GROUND, h, w)
    labels[66:67, :] .= LS.CLASS_ROAD
    labels[b .& .!walls] .= LS.CLASS_BUILDING
    labels[walls] .= LS.CLASS_WALL
    court = (labels .== LS.CLASS_GROUND) .& inner                               # η αυλή (ανοιχτός χώρος μέσα στο κτίριο)
    passable(l) = BitMatrix((l .== LS.CLASS_GROUND) .| (l .== LS.CLASS_ROAD) .| (l .== LS.CLASS_BUILDING))
    @test !any(reach(passable(labels), [(1, 1)]) .& court)                      # πριν: η αυλή είναι απομονωμένη
    LS.open_doors!(labels, [], 3.46)
    seen = reach(passable(labels), [(1, 1)])
    @test all(seen[court])                                                      # μετά: από έξω φτάνει κανείς σε όλη την αυλή
    # πόρτες και προς την αυλή: κελιά κτιρίου που ακουμπούν την αυλή
    inner_doors = [c for c in findall(labels .== LS.CLASS_BUILDING) if any(d -> court[c[1] + d[1], c[2] + d[2]], ((-1, 0), (1, 0), (0, -1), (0, 1)))]
    @test length(inner_doors) >= 2 * LS.door_width_cells(3.46)                  # εσωτερική περίμετρος 100 m+ -> τουλάχιστον 2 πόρτες
end

@testset "Καυτανζόγλειο: πόρτες και λεπτοί τοίχοι" begin
    els = JSON3.read(read(joinpath(@__DIR__, "fixtures", "kaftanzoglio_osm.json"), String)).elements
    poly = [(40.6243, 22.9656), (40.6267, 22.9656), (40.6267, 22.9684), (40.6243, 22.9684)]
    r = LS.build_from_osm(poly, els; target_mpp = 3.46)
    @test r.counts.doors_osm + r.counts.doors_auto >= 1
    walk = BitMatrix(r.labels .∈ Ref((LS.CLASS_GROUND, LS.CLASS_ROAD, LS.CLASS_BUILDING)))
    roads = [Tuple(c) for c in findall(r.labels .== LS.CLASS_ROAD)]
    seen = reach(walk, roads[1:min(end, 50)])
    bcells = r.labels .== LS.CLASS_BUILDING
    @test count(bcells .& .!seen) <= 0.05 * count(bcells)                       # σχεδόν όλα τα εσωτερικά φτάνουν σε δρόμο
    # κανένα κτίριο δεν είναι ανοιχτό από παντού: οι τοίχοι υπάρχουν
    @test count(r.labels .== LS.CLASS_WALL) > 100
    # ο αγωνιστικός χώρος του σταδίου (εσωτερικός ανοιχτός χώρος) συνδέεται με τους δρόμους μέσω της κερκίδας
    rel = first(e for e in els if e.type == "relation" && e.id == 549039)
    inner = LS.osm_pts(first(m for m in rel.members if m.role == "inner").geometry)
    lat_c, lon_c = sum(first.(inner)) / length(inner), sum(last.(inner)) / length(inner)
    x = (lon_c - 22.9656) * cosd(40.6255) * 111320.0 / r.grid.mpp
    y = -(lat_c - 40.6267) * 110574.0 / r.grid.mpp
    pitch = (clamp(floor(Int, y) + 1, 1, r.grid.h), clamp(floor(Int, x) + 1, 1, r.grid.w))
    @test r.labels[pitch...] == LS.CLASS_GROUND
    @test reach(walk, roads[1:min(end, 50)])[pitch...]
    @test r.counts.doors_auto >= 8                                              # η κερκίδα (περίμετρος 500+ m) παίρνει πολλές πόρτες
end
