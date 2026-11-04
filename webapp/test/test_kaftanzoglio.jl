# Τεστ αναφοράς: Εθνικό Καυτανζόγλειο Στάδιο, Θεσσαλονίκη (40.6255, 22.9670).
# Τα δεδομένα είναι αληθινή απάντηση Overpass (`out geom`) αποθηκευμένη στο fixtures/kaftanzoglio_osm.json, ώστε το τεστ να τρέχει χωρίς δίκτυο.
using Test, JSON3, HTTP
include(joinpath(@__DIR__, "..", "LiveService.jl"))
using .LiveService
const LS = LiveService

const ELEMENTS = JSON3.read(read(joinpath(@__DIR__, "fixtures", "kaftanzoglio_osm.json"), String)).elements
# περιοχή περίπου 272 × 265 m γύρω από το στάδιο, ως [(lat, lon), ...]
const POLY = [(40.6243, 22.9656), (40.6267, 22.9656), (40.6267, 22.9684), (40.6243, 22.9684)]

# εμβαδόν δακτυλίου [(lat, lon), ...] σε m² (τύπος του shoelace στην τοπική προβολή)
function ring_area(ring)
    lat0, lon0 = 40.6255, 22.9670
    p = [LS.xy_meters(lat0, lon0, q[1], q[2]) for q in ring]
    return abs(sum(p[k][1] * p[k+1][2] - p[k+1][1] * p[k][2] for k in 1:length(p)-1)) / 2
end

@testset "Καυτανζόγλειο: ένωση δακτυλίων" begin
    rel = first(e for e in ELEMENTS if e.type == "relation" && e.id == 549039)                # building=stadium
    outer = [LS.osm_pts(m.geometry) for m in rel.members if m.role == "outer"]
    inner = [LS.osm_pts(m.geometry) for m in rel.members if m.role == "inner"]
    @test length(outer) == 2 && all(w -> w[1] != w[end], outer)                              # το εξωτερικό περίγραμμα είναι σπασμένο σε 2 ανοιχτές γραμμές
    rings = LS.stitch_rings(outer)
    @test length(rings) == 1 && rings[1][1] == rings[1][end]                                 # ... και ενώνεται σε έναν κλειστό δακτύλιο
    @test length(LS.stitch_rings(inner)) == 1                                                # ο εσωτερικός (αγωνιστικός χώρος) είναι ήδη κλειστός
    @test ring_area(rings[1]) > ring_area(inner[1]) > 1000                                   # το στάδιο έχει δεκάδες χιλιάδες m², ο αγωνιστικός χώρος λιγότερο
end

@testset "Καυτανζόγλειο: εισαγωγή OSM" begin
    r = LS.build_from_osm(POLY, ELEMENTS; target_mpp = 3.46)
    g = r.grid
    @test abs(g.w - 272 / 3.46) < 25 && abs(g.h - 265 / 3.46) < 25
    @test r.counts.buildings >= 30 && r.counts.roads >= 25 && r.counts.gates >= 2
    @test isempty(r.warnings)

    # η κερκίδα του σταδίου (δακτύλιος μείον αγωνιστικός χώρος) καλύπτει το αναμενόμενο εμβαδόν κτιρίου/τοίχου
    rel = first(e for e in ELEMENTS if e.type == "relation" && e.id == 549039)
    stands_m2 = ring_area(LS.stitch_rings([LS.osm_pts(m.geometry) for m in rel.members if m.role == "outer"])[1]) -
                ring_area(LS.osm_pts(first(m for m in rel.members if m.role == "inner").geometry))
    built_cells = count(c -> c in (LS.CLASS_BUILDING, LS.CLASS_WALL), r.labels)
    @test built_cells * g.mpp^2 >= 0.8 * stands_m2                                          # τουλάχιστον η κερκίδα (+ τα υπόλοιπα κτίρια)

    # ο αγωνιστικός χώρος είναι η οπή του πολυγώνου: ανοιχτός χώρος, όχι κτίριο
    inner = LS.osm_pts(first(m for m in rel.members if m.role == "inner").geometry)
    lat_c, lon_c = sum(first.(inner)) / length(inner), sum(last.(inner)) / length(inner)
    x = (lon_c - 22.9656) * cosd(40.6255) * 111320.0 / g.mpp
    y = -(lat_c - 40.6267) * 110574.0 / g.mpp
    @test r.labels[clamp(floor(Int, y) + 1, 1, g.h), clamp(floor(Int, x) + 1, 1, g.w)] == LS.CLASS_GROUND

    # προτάσεις εξόδων: μέσα στο grid, με πάχος και μήκος > 0
    @test length(r.exits) >= 2
    @test all(e -> all(0 <= v <= max(g.w, g.h) + 1 for v in vcat(e["p1"], e["p2"])), r.exits)
end
