# Εισαγωγή δεδομένων OpenStreetMap (Overpass API) για μια περιοχή που ορίζει ο χρήστης στον χάρτη και μετατροπή τους
# στον χάρτη κλάσεων του Live Service: έδαφος, δρόμοι, κτίρια (με τοίχους) και προτάσεις εξόδων.
# Δεδομένα © OpenStreetMap contributors (ODbL). Το grid που παράγεται ταυτίζεται με την «εικόνα» του project (x0 = y0 = 0, f = 1).

# Δημόσιοι server Overpass: δοκιμάζονται με τη σειρά μέχρι να απαντήσει κάποιος (συχνά είναι υπερφορτωμένοι)
const OVERPASS_MIRRORS = [
    "https://lz4.overpass-api.de/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]
const OVERPASS_CACHE_TTL_S = 24 * 3600              # η απάντηση μιας περιοχής ξαναχρησιμοποιείται για 24 ώρες (χωρίς νέο αίτημα)
const OSM_USER_AGENT = "LiveService/1.0 (evacuation simulation research)"
const OSM_MAX_SPAN_M = 8000.0                       # μέγιστη διάσταση περιοχής (m)· μεγαλύτερες περιοχές δεν ζητούνται από το Overpass

# Πλάτος δρόμων (m) ανά κατηγορία highway· οι υπόλοιπες κατηγορίες (footway, path, steps, cycleway, ...) δεν είναι δρόμοι οχημάτων
const ROAD_WIDTH_M = Dict(
    "motorway" => 16.0, "trunk" => 16.0, "primary" => 14.0, "secondary" => 12.0, "tertiary" => 10.0,
    "motorway_link" => 10.0, "trunk_link" => 10.0, "primary_link" => 10.0, "secondary_link" => 9.0, "tertiary_link" => 8.0,
    "unclassified" => 8.0, "residential" => 8.0, "living_street" => 7.0, "road" => 8.0, "service" => 6.0, "track" => 5.0,
)
const GATE_BARRIERS = ("gate", "lift_gate", "swing_gate", "sliding_gate")
const GATE_SNAP_M = 15.0                            # πύλη μέσα σε τόσα m από το όριο της περιοχής θεωρείται έξοδος
const EXIT_DEDUPE_M = 30.0                          # προτάσεις εξόδων πιο κοντά από αυτό ενώνονται
const MAX_EXIT_SUGGESTIONS = 12

# --- Overpass ---

# Ερώτημα για όσα χρειάζονται: κτίρια, δεξαμενές/σιλό, δρόμοι, πύλες και είσοδοι, με γεωμετρία (out geom)
function overpass_query(south, west, north, east)
    b = "($south,$west,$north,$east)"
    return "[out:json][timeout:60];(" *
           "way[\"building\"]$b;relation[\"building\"][\"type\"=\"multipolygon\"]$b;" *
           "way[\"man_made\"~\"^(storage_tank|silo|tank)\$\"]$b;" *
           "way[\"highway\"]$b;" *
           "node[\"barrier\"~\"^(gate|lift_gate|swing_gate|sliding_gate)\$\"]$b;node[\"entrance\"]$b;" *
           ");out geom;"
end

# Αρχείο cache για ένα ερώτημα: οι απαντήσεις αποθηκεύονται στον φάκελο data (δεν μπαίνει στο git)
overpass_cache_file(query) = joinpath(DATA_DIR, "overpass_cache", string(hash(query); base = 16) * ".json")

# Δοκιμάζει τους server με τη σειρά (σε `rounds` γύρους, με παύση `pause` s ανάμεσα)· οι δημόσιοι server Overpass είναι συχνά υπερφορτωμένοι
# (502/504/timeout). Επιτυχημένη απάντηση αποθηκεύεται, ώστε ίδιο ερώτημα (π.χ. άλλο m/pixel στην ίδια περιοχή) να μη ξαναζητηθεί.
function fetch_overpass(query::AbstractString; mirrors = OVERPASS_MIRRORS, rounds::Int = 2, pause::Real = 4.0, hedge::Real = 15.0,
                        cache_ttl::Real = OVERPASS_CACHE_TTL_S)
    cache = overpass_cache_file(query)
    if isfile(cache) && time() - mtime(cache) < cache_ttl
        try
            return JSON3.read(read(cache, String))
        catch
            rm(cache; force = true)                                 # κατεστραμμένο αρχείο: ζητείται ξανά
        end
    end
    errors = String[]
    for round in 1:rounds
        round > 1 && sleep(pause)
        res, errs = race_mirrors(query, mirrors; hedge = hedge)
        if res !== nothing
            try
                mkpath(dirname(cache))
                write(cache, res.body)
            catch
            end
            return res.json
        end
        round == rounds && append!(errors, errs)
    end
    error("Το OpenStreetMap (Overpass) δεν απάντησε σε $(rounds) γύρους (οι δημόσιοι server είναι συχνά υπερφορτωμένοι)· δοκίμασε ξανά σε λίγο ή με μικρότερη περιοχή. ($(join(errors, " | ")))")
end

# Ένα αίτημα σε έναν server: (json, body) ή σφάλμα ως κείμενο
function try_mirror(m::AbstractString, query::AbstractString)
    host = HTTP.URI(m).host
    try
        r = HTTP.post(m, ["Content-Type" => "application/x-www-form-urlencoded", "User-Agent" => OSM_USER_AGENT],
            "data=" * HTTP.URIs.escapeuri(query); readtimeout = 70, connect_timeout = 10, retry = false, status_exception = false)
        r.status != 200 && return (; ok = false, error = "$host: HTTP $(r.status)")
        body = String(r.body)
        j = JSON3.read(body)
        haskey(j, :remark) && occursin("error", lowercase(string(j.remark))) && return (; ok = false, error = "$host: $(j.remark)")   # ο server έκοψε το ερώτημα
        return (; ok = true, json = j, body = body)
    catch e
        return (; ok = false, error = "$host: $(first(sprint(showerror, e), 80))")
    end
end

# Hedged αιτήματα: ξεκινά ο πρώτος server· αν δεν απαντήσει μέσα σε `hedge` s (ή αποτύχει) ξεκινά ο επόμενος παράλληλα, κ.ο.κ.
# Κερδίζει η πρώτη επιτυχής απάντηση. Επιστρέφει (αποτέλεσμα ή nothing, λίστα σφαλμάτων).
function race_mirrors(query::AbstractString, mirrors; hedge::Real = 15.0)
    ch = Channel{Any}(max(1, length(mirrors)))
    started = finished = 0
    errs = String[]
    launch!() = (started += 1; m = mirrors[started]; @async put!(ch, try_mirror(m, query)))
    isempty(mirrors) && return nothing, errs
    launch!()
    while finished < length(mirrors)
        t0 = time()
        while !isready(ch) && time() - t0 < hedge
            sleep(0.2)
        end
        if isready(ch)
            res = take!(ch); finished += 1
            res.ok && return res, errs
            push!(errs, res.error)
            started < length(mirrors) && launch!()                    # ο server απέτυχε: αμέσως ο επόμενος
        elseif started < length(mirrors)
            launch!()                                                 # αργεί: ξεκινά ο επόμενος παράλληλα
        end
    end
    return nothing, errs
end

# --- Γεωμετρία ---

# Τοπική προβολή σε μέτρα γύρω από το κέντρο της περιοχής: x ανατολικά, y προς τα νότια (όπως οι γραμμές της εικόνας)
xy_meters(lat0, lon0, lat, lon) = ((lon - lon0) * cosd(lat0) * 111320.0, -(lat - lat0) * 110574.0)

# Πάχος γραμμής: κάθε cell με κέντρο σε απόσταση ≤ half από το τμήμα a–b γίνεται true
function stamp_segment!(mask::BitMatrix, a, b, half::Real)
    h, w = size(mask)
    i0, i1 = max(1, floor(Int, min(a[2], b[2]) - half)), min(h, ceil(Int, max(a[2], b[2]) + half) + 1)
    j0, j1 = max(1, floor(Int, min(a[1], b[1]) - half)), min(w, ceil(Int, max(a[1], b[1]) + half) + 1)
    dx, dy = b[1] - a[1], b[2] - a[2]
    len2 = dx^2 + dy^2
    for i in i0:i1, j in j0:j1
        px, py = j - 0.5, i - 0.5
        t = len2 == 0 ? 0.0 : clamp(((px - a[1]) * dx + (py - a[2]) * dy) / len2, 0.0, 1.0)
        (px - (a[1] + t * dx))^2 + (py - (a[2] + t * dy))^2 <= half^2 && (mask[i, j] = true)
    end
    return mask
end

# Τομή τμημάτων a–b και c–d: (x, y, u) με u η θέση πάνω στο c–d (0..1), αλλιώς nothing
function seg_intersect(a, b, c, d)
    r = (b[1] - a[1], b[2] - a[2])
    s = (d[1] - c[1], d[2] - c[2])
    den = r[1] * s[2] - r[2] * s[1]
    abs(den) < 1e-12 && return nothing
    qp = (c[1] - a[1], c[2] - a[2])
    t = (qp[1] * s[2] - qp[2] * s[1]) / den
    u = (qp[1] * r[2] - qp[2] * r[1]) / den
    return (0 <= t <= 1 && 0 <= u <= 1) ? (a[1] + t * r[1], a[2] + t * r[2], u) : nothing
end

# Πλησιέστερο σημείο του περιγράμματος poly (λίστα κορυφών, κλειστό) στο p: (απόσταση, σημείο, μοναδιαίο εφαπτόμενο διάνυσμα)
function nearest_on_boundary(poly, p)
    best = (Inf, (0.0, 0.0), (1.0, 0.0))
    n = length(poly)
    for k in 1:n
        c, d = poly[k], poly[k == n ? 1 : k + 1]
        dx, dy = d[1] - c[1], d[2] - c[2]
        len = hypot(dx, dy)
        len == 0 && continue
        t = clamp(((p[1] - c[1]) * dx + (p[2] - c[2]) * dy) / len^2, 0.0, 1.0)
        q = (c[1] + t * dx, c[2] + t * dy)
        dist = hypot(p[1] - q[1], p[2] - q[2])
        dist < best[1] && (best = (dist, q, (dx / len, dy / len)))
    end
    return best
end

# --- Στοιχεία OSM ---

osm_tag(el, key) = (t = get(el, :tags, nothing); t === nothing ? nothing : get(t, Symbol(key), nothing))
osm_pts(geom) = [(Float64(q.lat), Float64(q.lon)) for q in geom]
is_closed(geom) = length(geom) >= 4 && geom[1].lat == geom[end].lat && geom[1].lon == geom[end].lon

# Ενώνει γραμμές (λίστες σημείων (lat, lon)) που μοιράζονται άκρα σε κλειστούς δακτυλίους· ό,τι δεν κλείνει απορρίπτεται
function stitch_rings(ways)
    rings = Vector{Vector{Tuple{Float64,Float64}}}()
    pool = [copy(w) for w in ways if length(w) >= 2]
    while !isempty(pool)
        ring = popfirst!(pool)
        progressed = true
        while ring[1] != ring[end] && progressed
            progressed = false
            for k in eachindex(pool)
                w = pool[k]
                if w[1] == ring[end]
                    append!(ring, w[2:end])
                elseif w[end] == ring[end]
                    append!(ring, reverse(w)[2:end])
                elseif w[end] == ring[1]
                    ring = vcat(w[1:end-1], ring)
                elseif w[1] == ring[1]
                    ring = vcat(reverse(w)[1:end-1], ring)
                else
                    continue
                end
                deleteat!(pool, k)
                progressed = true
                break
            end
        end
        length(ring) >= 4 && ring[1] == ring[end] && push!(rings, ring)
    end
    return rings
end

# Αυτόματη ανάλυση του grid (m/pixel) από τη μεγαλύτερη διάσταση της περιοχής (σε m): το πιο λεπτό grid που κρατά τη μεγαλύτερη διάσταση
# περίπου στα AUTO_GRID_CELLS cells, με όρια AUTO_MPP_MIN..AUTO_MPP_MAX· πάνω από το όριο μεγέθους ο καλούντας το αυξάνει ώστε να μη
# ξεπεραστούν τα max_side cells. Μικρές περιοχές: 1 m/pixel· μεγάλες: 3.46 m/pixel, η τιμή με την οποία είναι βαθμονομημένα τα σενάρια.
const AUTO_GRID_CELLS = 700
const AUTO_MPP_MIN = 1.0
const AUTO_MPP_MAX = 3.46
auto_mpp(span_m::Real) = clamp(span_m / AUTO_GRID_CELLS, AUTO_MPP_MIN, AUTO_MPP_MAX)

# Grid, προβολή και μετατροπή (lat, lon) -> συντεταγμένες grid για το πολύγωνο της περιοχής [(lat, lon), ...]
# target_mpp = nothing: αυτόματη ανάλυση (auto_mpp)
function osm_grid(polygon; target_mpp::Union{Nothing,Real} = nothing, max_side::Int = 1600)
    lat0 = sum(first.(polygon)) / length(polygon)
    lon0 = sum(last.(polygon)) / length(polygon)
    m = [xy_meters(lat0, lon0, p[1], p[2]) for p in polygon]
    xmin, xmax = extrema(first.(m)); ymin, ymax = extrema(last.(m))
    span = max(xmax - xmin, ymax - ymin)
    span > OSM_MAX_SPAN_M && error("Η περιοχή είναι πολύ μεγάλη ($(round(Int, span)) m· μέγιστο $(round(Int, OSM_MAX_SPAN_M)) m)")
    span < 20 && error("Η περιοχή είναι πολύ μικρή")
    mpp = max(target_mpp === nothing ? auto_mpp(span) : Float64(target_mpp), span / max_side)
    g = Grid(0.0, 0.0, 1.0, max(4, ceil(Int, (xmax - xmin) / mpp)), max(4, ceil(Int, (ymax - ymin) / mpp)), mpp)
    togrid(lat, lon) = (p = xy_meters(lat0, lon0, lat, lon); ((p[1] - xmin) / mpp, (p[2] - ymin) / mpp))
    return g, togrid
end

# Χάρτης κλάσεων και προτάσεις εξόδων από τα στοιχεία (elements) μιας απάντησης Overpass
function build_from_osm(polygon, elements; target_mpp::Union{Nothing,Real} = nothing, max_side::Int = 1600)
    g, togrid = osm_grid(polygon; target_mpp, max_side)
    poly_g = [togrid(p[1], p[2]) for p in polygon]
    region = fill_polygon!(falses(g.h, g.w), poly_g, true)

    bmask = falses(g.h, g.w)                                   # κτίρια και δεξαμενές
    rmask = falses(g.h, g.w)                                   # δρόμοι
    road_segs = Tuple{Any,Any,Float64}[]                       # (a, b, πλάτος m) για τις τομές με το όριο
    gates = Tuple{Float64,Float64}[]
    entrances = Tuple{Float64,Float64}[]                       # είσοδοι κτιρίων (entrance=*) -> πόρτες στους τοίχους
    n_buildings = n_roads = 0

    for el in elements
        typ = el.type
        if typ == "node"
            osm_tag(el, "entrance") !== nothing && osm_tag(el, "entrance") != "no" && push!(entrances, togrid(el.lat, el.lon))
            (osm_tag(el, "entrance") !== nothing || osm_tag(el, "barrier") in GATE_BARRIERS) && push!(gates, togrid(el.lat, el.lon))
        elseif typ == "way" && haskey(el, :geometry)
            geom = el.geometry
            hw = osm_tag(el, "highway")
            if hw !== nothing && haskey(ROAD_WIDTH_M, hw)
                width = ROAD_WIDTH_M[hw]
                pts = [togrid(q.lat, q.lon) for q in geom]
                half = max(width / g.mpp / 2, 0.75)            # τουλάχιστον ~1.5 cell ώστε οι διαγώνιοι δρόμοι να μένουν συνεχείς
                for k in 1:(length(pts) - 1)
                    stamp_segment!(rmask, pts[k], pts[k + 1], half)
                    push!(road_segs, (pts[k], pts[k + 1], width))
                end
                n_roads += 1
            elseif (osm_tag(el, "building") !== nothing || osm_tag(el, "man_made") in ("storage_tank", "silo", "tank")) && is_closed(geom)
                fill_polygon!(bmask, [togrid(q.lat, q.lon) for q in geom], true)
                n_buildings += 1
            end
        elseif typ == "relation" && haskey(el, :members) && osm_tag(el, "building") !== nothing
            # multipolygon: οι δακτύλιοι αποτελούνται συχνά από περισσότερες ανοιχτές γραμμές (π.χ. στάδια) που ενώνονται στα άκρα τους
            ways = [(get(mem, :role, "outer") == "inner", osm_pts(mem.geometry)) for mem in el.members
                    if get(mem, :type, "") == "way" && haskey(mem, :geometry)]
            tmp = falses(g.h, g.w)
            for ring in stitch_rings([w for (inner, w) in ways if !inner])
                fill_polygon!(tmp, [togrid(p[1], p[2]) for p in ring], true)
            end
            for ring in stitch_rings([w for (inner, w) in ways if inner])
                hole = fill_polygon!(falses(g.h, g.w), [togrid(p[1], p[2]) for p in ring], true)
                tmp .&= .!hole                                 # οπή του πολυγώνου (αγωνιστικός χώρος, αυλή)
            end
            bmask .|= tmp
            n_buildings += 1
        end
    end

    # κλάσεις με προτεραιότητα: τοίχος > εσωτερικό κτιρίου > δρόμος > έδαφος· εκτός περιοχής μη βατό
    walls = building_walls(bmask)
    interior = bmask .& .!walls
    labels = fill(CLASS_OUTSIDE, g.h, g.w)
    labels[region] .= CLASS_GROUND
    labels[rmask .& region] .= CLASS_ROAD
    labels[interior .& region] .= CLASS_BUILDING
    labels[walls .& region] .= CLASS_WALL
    doors_osm, doors_auto = open_doors!(labels, entrances, g.mpp)   # πόρτες στις εισόδους του OSM και αυτόματα στα κτίρια που μένουν κλειστά

    exits = suggest_exits(poly_g, gates, road_segs, g.mpp)

    warnings = String[]
    n_buildings == 0 && push!(warnings, "Δεν βρέθηκαν κτίρια στο OpenStreetMap για αυτή την περιοχή· σχεδίασέ τα με το πινέλο ή χρησιμοποίησε εικόνα.")
    n_roads == 0 && push!(warnings, "Δεν βρέθηκαν δρόμοι στο OpenStreetMap για αυτή την περιοχή.")
    isempty(exits) && push!(warnings, "Δεν βρέθηκαν πύλες ή δρόμοι που τέμνουν το όριο· όρισε τις εξόδους χειροκίνητα.")
    return (; grid = g, labels, region = [collect(p) for p in poly_g], exits,
        counts = (; buildings = n_buildings, roads = n_roads, gates = length(gates), doors_osm, doors_auto), warnings)
end

# Προτάσεις εξόδων: πρώτα οι πύλες κοντά στο όριο, μετά οι δρόμοι που τέμνουν το όριο. Κάθε πρόταση είναι τμήμα κατά μήκος
# του ορίου (p1–p2, συντεταγμένες grid) και πάχος σε μέτρα, όπως οι έξοδοι που σχεδιάζει ο χρήστης.
function suggest_exits(poly_g, gates, road_segs, mpp)
    cands = Tuple{Tuple{Float64,Float64},Tuple{Float64,Float64},Float64}[]      # (σημείο, εφαπτόμενο, μήκος m)
    for gp in gates
        d, q, t = nearest_on_boundary(poly_g, gp)
        d * mpp <= GATE_SNAP_M && push!(cands, (q, t, 10.0))
    end
    n = length(poly_g)
    for (a, b, width) in sort(road_segs; by = s -> -s[3])                       # πλατύτεροι δρόμοι πρώτα
        for k in 1:n
            c, d = poly_g[k], poly_g[k == n ? 1 : k + 1]
            hit = seg_intersect(a, b, c, d)
            hit === nothing && continue
            len = hypot(d[1] - c[1], d[2] - c[2])
            len == 0 && continue
            push!(cands, ((hit[1], hit[2]), ((d[1] - c[1]) / len, (d[2] - c[2]) / len), max(width + 4.0, 10.0)))
        end
    end
    accepted = typeof(cands[1:0])()
    for cd in cands
        any(a -> hypot(a[1][1] - cd[1][1], a[1][2] - cd[1][2]) * mpp < EXIT_DEDUPE_M, accepted) && continue
        push!(accepted, cd)
        length(accepted) >= MAX_EXIT_SUGGESTIONS && break
    end
    return [Dict("p1" => [q[1] - t[1] * len / mpp / 2, q[2] - t[2] * len / mpp / 2],
                 "p2" => [q[1] + t[1] * len / mpp / 2, q[2] + t[2] * len / mpp / 2], "thickness_m" => 8.0) for (q, t, len) in accepted]
end
