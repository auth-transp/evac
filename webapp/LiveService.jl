module LiveService
# Live Service: τοπικός server για τη δημιουργία heightmap από κάτοψη/δορυφορική εικόνα (Oxygen.jl)
# Εκκίνηση:  julia --project=webapp webapp/LiveService.jl   και άνοιγμα του http://127.0.0.1:8080

using Oxygen            # web framework
using HTTP              # αντικείμενα Request/Response
using JSON3             # ανάγνωση/εγγραφή JSON
using Images            # φόρτωση/επεξεργασία εικόνων (Lab, imresize, ...)
using FileIO            # load/save εικόνων
using ImageIO           # αποκωδικοποίηση png/jpg
using ImageMagick       # αποκωδικοποίηση bmp/tiff
using Base64            # μεταφορά των labels ως base64
using Random            # τυχαία αναγνωριστικά project

include("src/Heightmap.jl")
include("src/Propose.jl")

const FRONTEND_DIR = joinpath(@__DIR__, "frontend")
const DATA_DIR = joinpath(@__DIR__, "data")             # uploads και αποτελέσματα ανά project (δεν μπαίνει στο git)
const DEFAULT_MPP = 2500.0 / 723.37                     # m ανά pixel του Qatargas Map (1 / METERS_TO_PIXELS του σεναρίου)
const MAX_GRID_SIDE = 1600                              # ανώτατη διάσταση του grid ώστε ο D* Lite να μένει γρήγορος
# αρχεία ενός project που επιτρέπεται να κατεβάσει ο browser (heightmap/config από το build, τα υπόλοιπα από την προσομοίωση)
const SERVED_FILES = r"^(heightmap\.png|spawn_mask\.png|config\.json|meta\.json|progress\.json|result\.json|frames\.bin|trajectories\.csv|sim\.log|cm_[1-7]\.png)$"
const CONTENT_TYPES = Dict(".png" => "image/png", ".json" => "application/json", ".bin" => "application/octet-stream", ".csv" => "text/csv; charset=utf-8", ".log" => "text/plain; charset=utf-8")
const SIM_OUTPUTS = ["progress.json", "result.json", "meta.json", "frames.bin", "trajectories.csv", "sim.log"]   # καθαρίζονται σε κάθε νέα εκτέλεση

# Έλεγχος κεφαλίδας αρχείου ώστε να βρεθεί ο τύπος της εικόνας που ανέβηκε
function image_extension(b::Vector{UInt8})
    length(b) >= 4 || error("το αρχείο είναι κενό ή πολύ μικρό")
    b[1:4] == UInt8[0x89, 0x50, 0x4e, 0x47] && return "png"
    b[1:2] == UInt8[0xff, 0xd8] && return "jpg"
    b[1:2] == UInt8[0x42, 0x4d] && return "bmp"
    (b[1:4] == UInt8[0x49, 0x49, 0x2a, 0x00] || b[1:4] == UInt8[0x4d, 0x4d, 0x00, 0x2a]) && return "tif"
    error("μη υποστηριζόμενος τύπος εικόνας (δεκτά: png, jpg, bmp, tif)")
end

project_dir(id::AbstractString) = (occursin(r"^[0-9a-f]{12}$", id) || error("άκυρο id"); joinpath(DATA_DIR, id))

const SOURCE_CACHE = Dict{String,Matrix{RGB{N0f8}}}()   # αποκωδικοποιημένες εικόνες στη μνήμη
function load_source(id::AbstractString)
    get!(SOURCE_CACHE, id) do
        dir = project_dir(id)
        file = first(filter(f -> startswith(f, "source."), readdir(dir)))
        RGB{N0f8}.(load(joinpath(dir, file)))
    end
end

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

# Υπολογισμός grid από την περιοχή: bounding box των πολυγώνων, αναγωγή στην επιθυμητή ανάλυση m/pixel
function make_grid(polys, img_size, src_mpp::Real, target_mpp::Real)
    xs = [q[1] for p in polys for q in p.points]
    ys = [q[2] for p in polys for q in p.points]
    isempty(xs) && error("δεν έχει οριστεί περιοχή εγκατάστασης")
    x0, x1 = max(0.0, floor(minimum(xs))), min(Float64(img_size[2]), ceil(maximum(xs)))
    y0, y1 = max(0.0, floor(minimum(ys))), min(Float64(img_size[1]), ceil(maximum(ys)))
    (x1 - x0 >= 4 && y1 - y0 >= 4) || error("η περιοχή είναι πολύ μικρή")
    mpp = max(Float64(target_mpp), src_mpp * max(x1 - x0, y1 - y0) / MAX_GRID_SIDE)   # όριο μεγέθους grid
    f = src_mpp / mpp
    return Grid(x0, y0, f, max(4, round(Int, (x1 - x0) * f)), max(4, round(Int, (y1 - y0) * f)), mpp)
end

grid_json(g::Grid) = Dict("x0" => g.x0, "y0" => g.y0, "f" => g.f, "w" => g.w, "h" => g.h, "mpp" => g.mpp)
grid_from(j) = Grid(j.x0, j.y0, j.f, j.w, j.h, j.mpp)

# Κομμένο και υπο-δειγματοληπτημένο στο grid κομμάτι της εικόνας
function crop_to_grid(img, g::Grid)
    r0, r1 = Int(g.y0) + 1, min(size(img, 1), Int(g.y0) + max(1, round(Int, g.h / g.f)))
    c0, c1 = Int(g.x0) + 1, min(size(img, 2), Int(g.x0) + max(1, round(Int, g.w / g.f)))
    return imresize(img[r0:r1, c0:c1], (g.h, g.w))
end

# τα labels ταξιδεύουν γραμμή-γραμμή (row-major, όπως το canvas): label[row, col] = byte[(row-1)*w + col]
labels_to_b64(L::Matrix{UInt8}) = base64encode(vec(permutedims(L)))
labels_from_b64(s::AbstractString, g::Grid) = permutedims(reshape(base64decode(s), g.w, g.h))

# POST /api/upload: ανεβάζει την εικόνα (raw bytes) και επιστρέφει id και διαστάσεις
function upload_route(req::HTTP.Request)
    bytes = body_bytes(req)
    ext = image_extension(bytes)
    id = randstring("0123456789abcdef", 12)
    dir = project_dir(id)
    mkpath(dir)
    write(joinpath(dir, "source.$ext"), bytes)
    img = load_source(id)
    return json_response(Dict("id" => id, "width" => size(img, 2), "height" => size(img, 1)))
end

# POST /api/propose: αυτόματη πρόταση χάρτη κλάσεων για την περιοχή που όρισε ο χρήστης
function propose_route(req::HTTP.Request)
    p = JSON3.read(String(body_bytes(req)))
    img = load_source(p.id)
    g = make_grid(p.region, size(img), p.src_mpp, get(p, :target_mpp, DEFAULT_MPP))
    region = region_mask(p.region, g)
    gs = get(p, :ground_sample, nothing)
    sample = gs === nothing ? nothing : to_grid(g, gs)
    labels = propose_labels(crop_to_grid(img, g), region, g;
        sensitivity = get(p, :sensitivity, 14.0), min_building_m = get(p, :min_building_m, 8.0), ground_sample = sample)
    return json_response(Dict("grid" => grid_json(g), "labels" => labels_to_b64(labels), "region_cells" => count(region)))
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
@post "/api/upload" req -> guarded(upload_route, req)
@post "/api/propose" req -> guarded(propose_route, req)
@post "/api/build" req -> guarded(build_route, req)
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
const STATIC_TYPES = Dict(".js" => "text/javascript; charset=utf-8", ".css" => "text/css; charset=utf-8", ".html" => "text/html; charset=utf-8")
@get "/static/{name}" function(req::HTTP.Request, name::String)
    path = joinpath(FRONTEND_DIR, name)
    ext = splitext(name)[2]
    (haskey(STATIC_TYPES, ext) && dirname(abspath(path)) == abspath(FRONTEND_DIR) && isfile(path)) || return HTTP.Response(404, "not found")
    return HTTP.Response(200, ["Content-Type" => STATIC_TYPES[ext], "Cache-Control" => "no-store"], read(path))
end

start(; port::Int = 8080) = serve(host = "127.0.0.1", port = port)

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    LiveService.start()
end
