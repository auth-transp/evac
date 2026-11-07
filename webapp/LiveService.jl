module LiveService
# Live Service: τοπικός server για τη δημιουργία heightmap εκκένωσης από δεδομένα OpenStreetMap (Oxygen.jl)
# Εκκίνηση:  julia --project=webapp webapp/LiveService.jl   και άνοιγμα του http://127.0.0.1:8080

using Oxygen            # web framework
using HTTP              # αντικείμενα Request/Response
using JSON3             # ανάγνωση/εγγραφή JSON
using Images            # τύποι εικόνας (RGB, Gray) για την αποθήκευση του heightmap
using FileIO            # save εικόνων
using ImageIO           # κωδικοποίηση png
using Base64            # μεταφορά των labels ως base64
using Random            # τυχαία αναγνωριστικά project

include("src/Heightmap.jl")
include("src/Buildings.jl")
include("src/Osm.jl")

const FRONTEND_DIR = joinpath(@__DIR__, "frontend")
const DATA_DIR = joinpath(@__DIR__, "data")             # αποτελέσματα ανά project και cache του Overpass (δεν μπαίνει στο git)
const MAX_GRID_SIDE = 1600                              # ανώτατη διάσταση του grid ώστε ο D* Lite να μένει γρήγορος
# αρχεία ενός project που επιτρέπεται να κατεβάσει ο browser (heightmap/config από το build, τα υπόλοιπα από την προσομοίωση)
const SERVED_FILES = r"^(heightmap\.png|spawn_mask\.png|config\.json|meta\.json|progress\.json|result\.json|frames\.bin|trajectories\.csv|sim\.log|cm_[1-7]\.png)$"
const CONTENT_TYPES = Dict(".png" => "image/png", ".json" => "application/json", ".bin" => "application/octet-stream", ".csv" => "text/csv; charset=utf-8", ".log" => "text/plain; charset=utf-8")
const SIM_OUTPUTS = ["progress.json", "result.json", "meta.json", "frames.bin", "trajectories.csv", "sim.log"]   # καθαρίζονται σε κάθε νέα εκτέλεση

project_dir(id::AbstractString) = (occursin(r"^[0-9a-f]{12}$", id) || error("άκυρο id"); joinpath(DATA_DIR, id))

# Σώμα του request ως bytes (στο HTTP.jl 2.x το body είναι wrapper)
function body_bytes(req::HTTP.Request)
    b = req.body
    hasfield(typeof(b), :data) && (b = b.data)                         # HTTP.BytesBody
    b isa AbstractVector{UInt8} && return Vector{UInt8}(b)
    return Vector{UInt8}(read(b))
end

json_response(obj; status = 200) = HTTP.Response(status, ["Content-Type" => "application/json"], JSON3.write(obj))

# Εκτέλεση handler με επιστροφή σφάλματος ως JSON αντί για κατάρρευση
function guarded(f, args...)
    try
        return f(args...)
    catch e
        @error "request failed" exception = (e, catch_backtrace())
        return json_response(Dict("error" => sprint(showerror, e)); status = 400)
    end
end

grid_json(g::Grid) = Dict("x0" => g.x0, "y0" => g.y0, "f" => g.f, "w" => g.w, "h" => g.h, "mpp" => g.mpp)
grid_from(j) = Grid(j.x0, j.y0, j.f, j.w, j.h, j.mpp)

# τα labels ταξιδεύουν γραμμή-γραμμή (row-major, όπως το canvas): label[row, col] = byte[(row-1)*w + col]
labels_to_b64(L::Matrix{UInt8}) = base64encode(vec(permutedims(L)))
labels_from_b64(s::AbstractString, g::Grid) = permutedims(reshape(base64decode(s), g.w, g.h))

# POST /api/osm {polygon: [[lat, lon], ...], target_mpp (προαιρετικό)}: νέο project από δεδομένα OpenStreetMap για την περιοχή που ορίστηκε στον χάρτη.
# Το grid παίζει τον ρόλο της «εικόνας» (x0 = y0 = 0, f = 1)· επιστρέφει χάρτη κλάσεων, περίγραμμα περιοχής και προτάσεις εξόδων.
function osm_route(req::HTTP.Request)
    p = JSON3.read(String(body_bytes(req)))
    polygon = [(Float64(q[1]), Float64(q[2])) for q in p.polygon]
    length(polygon) >= 3 || error("όρισε την περιοχή στον χάρτη (τουλάχιστον 3 κορυφές)")
    target = haskey(p, :target_mpp) && p.target_mpp !== nothing ? Float64(p.target_mpp) : nothing      # χωρίς τιμή: αυτόματη ανάλυση (auto_mpp)
    osm_grid(polygon; target_mpp = target, max_side = MAX_GRID_SIDE)       # έλεγχος μεγέθους πριν το αίτημα δικτύου
    margin = 0.0003                                                        # ~30 m: για να πιάνονται δρόμοι που τέμνουν το όριο
    lats, lons = first.(polygon), last.(polygon)
    j = fetch_overpass(overpass_query(minimum(lats) - margin, minimum(lons) - margin, maximum(lats) + margin, maximum(lons) + margin))
    r = build_from_osm(polygon, get(j, :elements, []); target_mpp = target, max_side = MAX_GRID_SIDE)
    id = randstring("0123456789abcdef", 12)
    mkpath(project_dir(id))
    return json_response(Dict("id" => id, "grid" => grid_json(r.grid), "labels" => labels_to_b64(r.labels), "region" => r.region,
        "exits" => r.exits, "counts" => r.counts, "warnings" => r.warnings,
        "attribution" => "© OpenStreetMap contributors"))
end

# POST /api/build: τελικό heightmap από τα (διορθωμένα) labels, τις εξόδους και τις ζώνες spawn
function build_route(req::HTTP.Request)
    p = JSON3.read(String(body_bytes(req)))
    g = grid_from(p.grid)
    labels = Matrix{UInt8}(labels_from_b64(p.labels, g))
    exits = falses(g.h, g.w)
    for e in p.exits
        exits .|= exit_mask(e, g)
    end
    hm = compose_heightmap(labels, exits)
    walk = walkable(hm)
    comp, ncomp = components(walk)
    csize = zeros(Int, ncomp)                                # μέγεθος κάθε συνεκτικής συνιστώσας βατών cells
    for v in comp
        v > 0 && (csize[v] += 1)
    end
    biggest = maximum(csize; init = 0)

    warnings = String[]
    exit_info = []
    exit_comps = Set{Int}()
    for (k, e) in enumerate(p.exits)
        m = exit_mask(e, g)
        cells = findall(m .& walk)
        isempty(cells) && push!(warnings, "Η έξοδος $k δεν περιέχει βατά cells")
        if !isempty(cells) && maximum(csize[comp[c]] for c in cells) < 0.1 * biggest
            push!(warnings, "Η έξοδος $k δεν συνδέεται με τον βατό χώρο της εγκατάστασης (πρέπει να τέμνει το όριο της περιοχής)")
        end
        union!(exit_comps, filter(>(0), unique(comp[cells])))
        (cx, cy) = to_grid(g, ((e.p1[1] + e.p2[1]) / 2, (e.p1[2] + e.p2[2]) / 2))
        # προορισμός για την προσομοίωση: το βατό, συνδεδεμένο cell της εξόδου που είναι πιο κοντά στο κέντρο της
        # (γραμμή, στήλη 1-based· το κέντρο του cell (i, j) είναι το σημείο (i, j) του μοντέλου, άρα το κέντρο της εξόδου είναι (cy + 0.5, cx + 0.5))
        usable = filter(c -> csize[comp[c]] >= 0.1 * biggest, cells)
        goal = isempty(usable) ? nothing : (c = argmin(c -> (c[1] - (cy + 0.5))^2 + (c[2] - (cx + 0.5))^2, usable); [c[1], c[2]])
        push!(exit_info, Dict("id" => k, "col" => cx, "row" => cy, "goal_cell" => goal,
            "p1" => collect(to_grid(g, e.p1)), "p2" => collect(to_grid(g, e.p2)), "thickness_px" => max(e.thickness_m / g.mpp, 1.5)))
    end
    isempty(p.exits) && push!(warnings, "Δεν έχει οριστεί καμία έξοδος")

    spawn_all = falses(g.h, g.w)
    zones = []
    for (k, z) in enumerate(p.spawn_zones)
        zm = fill_polygon!(falses(g.h, g.w), [to_grid(g, q) for q in z.points], true)
        usable = zm .& walk
        spawn_all .|= usable
        cells = findall(usable)
        reachable = count(c -> comp[c] in exit_comps, cells)
        count(zm) > 0 && isempty(cells) && push!(warnings, "Η ζώνη spawn $k δεν περιέχει βατά cells")
        !isempty(cells) && reachable < length(cells) && push!(warnings,
            "Ζώνη spawn $k: $(length(cells) - reachable) από $(length(cells)) βατά cells δεν συνδέονται με καμία έξοδο")
        push!(zones, Dict("id" => k, "polygon" => [collect(to_grid(g, q)) for q in z.points], "walkable_cells" => length(cells), "reachable_cells" => reachable))
    end
    isempty(p.spawn_zones) && push!(warnings, "Δεν έχει οριστεί ζώνη spawn")

    dir = project_dir(p.id)
    save(joinpath(dir, "heightmap.png"), RGB{N0f8}.(Gray{N0f8}.(reinterpret(N0f8, hm))))
    save(joinpath(dir, "spawn_mask.png"), RGB{N0f8}.(Gray{N0f8}.(Float64.(spawn_all))))
    config = Dict(
        "id" => p.id, "grid" => grid_json(g),
        "rows" => g.h, "cols" => g.w,                            # heightmap[row, col]
        "meters_per_pixel" => g.mpp, "meters_to_pixels" => 1 / g.mpp,
        "n_agents" => get(p, :n_agents, 100),
        "exits" => exit_info, "spawn_zones" => zones,
        "coordinates" => "origin top-left, col = x to the right, row = y downwards (grid pixels)",
        "height_of_class" => Dict("outside" => HEIGHT_OF_CLASS[1], "ground" => HEIGHT_OF_CLASS[2], "road" => HEIGHT_OF_CLASS[3],
            "building" => HEIGHT_OF_CLASS[4], "wall" => HEIGHT_OF_CLASS[5]),
        "warnings" => warnings)
    open(joinpath(dir, "config.json"), "w") do io
        JSON3.pretty(io, config)
    end
    return json_response(Dict("config" => config, "walkable_cells" => count(walk)))
end

# --- Προσομοίωση: κάθε εκτέλεση τρέχει σε ξεχωριστή διεργασία Julia (sim_worker.jl) ---
const JOBS = Dict{String,Base.Process}()                # id project => διεργασία του worker

# id της προσομοίωσης που τρέχει τώρα (μία κάθε φορά): ζωντανή διεργασία ΚΑΙ κατάσταση starting/running στο progress.json
# (μετά το "done"/"cancelled" η διεργασία μπορεί να τερματίζει ακόμη, αλλά δεν θεωρείται απασχολημένη)
function busy_job()
    for (id, p) in JOBS
        process_running(p) || continue
        st = try
            JSON3.read(read(joinpath(project_dir(id), "progress.json"), String)).status
        catch
            "starting"
        end
        st in ("starting", "running") && return id
    end
    return nothing
end

write_progress(dir, status, T; extra...) = open(joinpath(dir, "progress.json"), "w") do io
    JSON3.write(io, (; status, frame = 0, T, extra...))
end

# POST /api/run {id, T, seed}: ξεκινά την προσομοίωση για ένα project που έχει περάσει από το build
function run_route(req::HTTP.Request)
    p = JSON3.read(String(body_bytes(req)))
    dir = project_dir(p.id)
    isfile(joinpath(dir, "config.json")) || error("δημιούργησε πρώτα το heightmap (βήμα 7)")
    other = busy_job()
    other === nothing || error("τρέχει ήδη μια προσομοίωση (project $other)· περίμενε να τελειώσει ή ακύρωσέ την")
    T, seed = Int(get(p, :T, 2500)), Int(get(p, :seed, 123))
    (1 <= T <= 20000) || error("άκυρο πλήθος βημάτων T")
    old = get(JOBS, p.id, nothing)
    if old !== nothing && process_running(old)              # προηγούμενος worker του ίδιου project που τερματίζει ακόμη (ή ακυρώθηκε)
        for _ in 1:50
            process_running(old) || break
            sleep(0.1)
        end
        process_running(old) && kill(old)
        wait(old)                                           # στα Windows δεν σβήνονται αρχεία που κρατά ανοιχτά η διεργασία
    end
    for f in SIM_OUTPUTS
        rm(joinpath(dir, f); force = true)
    end
    write_progress(dir, "starting", T)
    log = open(joinpath(dir, "sim.log"), "w")
    cmd = `$(Base.julia_cmd()) --project=$(@__DIR__) $(joinpath(@__DIR__, "sim_worker.jl")) $dir $T $seed`
    JOBS[p.id] = run(pipeline(cmd; stdout = log, stderr = log); wait = false)
    close(log)                                              # ο worker κρατά δικό του handle
    return json_response(Dict("id" => p.id, "status" => "starting", "T" => T))
end

# GET /api/jobs/{id}: κατάσταση της προσομοίωσης (από το progress.json του worker)
function job_status_route(id::AbstractString)
    dir = project_dir(id)
    file = joinpath(dir, "progress.json")
    isfile(file) || return json_response(Dict("status" => "none"))
    st = JSON3.read(read(file, String))
    d = Dict{String,Any}(String(k) => v for (k, v) in pairs(st))
    if d["status"] in ("starting", "running")
        proc = get(JOBS, id, nothing)
        # η διεργασία τελείωσε ή χάθηκε (π.χ. επανεκκίνηση του server) χωρίς να γράψει "done"
        if (proc !== nothing && !process_running(proc)) || (proc === nothing && time() - mtime(file) > 120)
            log = joinpath(dir, "sim.log")
            d["status"] = "failed"
            d["error"] = isfile(log) ? join(last(readlines(log), 8), "\n") : "η διεργασία τερματίστηκε"
        end
    end
    return json_response(d)
end

# POST /api/jobs/{id}/cancel
function job_cancel_route(id::AbstractString)
    dir = project_dir(id)
    proc = get(JOBS, id, nothing)
    proc !== nothing && process_running(proc) && kill(proc)
    write_progress(dir, "cancelled", 0)
    return json_response(Dict("status" => "cancelled"))
end

@get "/" function(req::HTTP.Request)
    return HTTP.Response(200, ["Content-Type" => "text/html; charset=utf-8", "Cache-Control" => "no-store"], read(joinpath(FRONTEND_DIR, "index.html")))
end
@post "/api/build" req -> guarded(build_route, req)
@post "/api/osm" req -> guarded(osm_route, req)
@post "/api/run" req -> guarded(run_route, req)
@get "/api/jobs/{id}" function(req::HTTP.Request, id::String)
    return guarded(job_status_route, id)
end
@post "/api/jobs/{id}/cancel" function(req::HTTP.Request, id::String)
    return guarded(job_cancel_route, id)
end
@get "/api/files/{id}/{name}" function(req::HTTP.Request, id::String, name::String)
    occursin(SERVED_FILES, name) || return HTTP.Response(404, "not found")
    path = joinpath(project_dir(id), name)
    isfile(path) || return HTTP.Response(404, "not found")
    return HTTP.Response(200, ["Content-Type" => CONTENT_TYPES[splitext(name)[2]], "Cache-Control" => "no-store",
        "Content-Disposition" => "inline; filename=\"$name\""], read(path))
end
# Τα αρχεία του frontend διαβάζονται από τον δίσκο σε κάθε αίτημα (χωρίς cache), ώστε οι αλλαγές να φαίνονται αμέσως
const STATIC_TYPES = Dict(".js" => "text/javascript; charset=utf-8", ".css" => "text/css; charset=utf-8", ".html" => "text/html; charset=utf-8",
    ".svg" => "image/svg+xml", ".png" => "image/png", ".webmanifest" => "application/manifest+json")
@get "/static/{name}" function(req::HTTP.Request, name::String)
    path = joinpath(FRONTEND_DIR, name)
    ext = splitext(name)[2]
    (haskey(STATIC_TYPES, ext) && dirname(abspath(path)) == abspath(FRONTEND_DIR) && isfile(path)) || return HTTP.Response(404, "not found")
    return HTTP.Response(200, ["Content-Type" => STATIC_TYPES[ext], "Cache-Control" => "no-store"], read(path))
end

@get "/favicon.ico" function(req::HTTP.Request)                                 # οι browsers ζητούν πάντα /favicon.ico
    return HTTP.Response(200, ["Content-Type" => "image/png", "Cache-Control" => "public, max-age=86400"], read(joinpath(FRONTEND_DIR, "favicon-32.png")))
end
# Το API και τα αρχεία των projects δεν πρέπει να ευρετηριάζονται. TODO: όταν οριστεί το domain, πρόσθεσε γραμμή Sitemap: (η σελίδα είναι ενιαία, άρα το sitemap ανήκει στο γονικό site)
@get "/robots.txt" function(req::HTTP.Request)
    return HTTP.Response(200, ["Content-Type" => "text/plain; charset=utf-8"], "User-agent: *\nDisallow: /api/\nAllow: /\n")
end

# Σελίδα 404 για άγνωστες διαδρομές (browser): το API και τα αρχεία επιστρέφουν απλό 404
function not_found_middleware(handle)
    return function(req::HTTP.Request)
        res = handle(req)
        if res isa HTTP.Response && res.status == 404 && !startswith(req.target, "/api/") && !startswith(req.target, "/static/")
            return HTTP.Response(404, ["Content-Type" => "text/html; charset=utf-8", "Cache-Control" => "no-store"], read(joinpath(FRONTEND_DIR, "404.html")))
        end
        return res
    end
end

start(; port::Int = 8080) = serve(host = "127.0.0.1", port = port, middleware = [not_found_middleware])

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    LiveService.start()
end
