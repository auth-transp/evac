# Δημιουργία heightmap από χάρτη κλάσεων (labels) + έξοδοι, συμβατό με το DStarLite - Scenario 3.jl

# Κλάσεις του χάρτη labels (ένα byte ανά cell)
const CLASS_OUTSIDE  = 0x00     # εκτός εγκατάστασης (μη βατό)
const CLASS_GROUND   = 0x01     # ανοιχτός χώρος
const CLASS_ROAD     = 0x02     # δρόμος
const CLASS_BUILDING = 0x03     # εσωτερικό κτιρίου (βατό με πεναλτί)
const CLASS_WALL     = 0x04     # τοίχος κτιρίου (μη βατός)

# Τιμές του heightmap όπως στο Qatargas Map.jpg (πριν την αντιστροφή 255 .- heightmap του σεναρίου)
# Το σενάριο θεωρεί μη βατά τα cells με (255 - τιμή) > 245, δηλαδή τιμή < 10
const HEIGHT_OF_CLASS = UInt8[0, 102, 255, 60, 0]       # ευρετήριο: κλάση + 1
const HEIGHT_GROUND = HEIGHT_OF_CLASS[Int(CLASS_GROUND) + 1]
const WHITE_THRESHOLD = 245                             # ίδιο με το white_threshold του σεναρίου

# Μετασχηματισμός από συντεταγμένες της αρχικής εικόνας (pixels, x δεξιά, y κάτω) στο grid του μοντέλου
struct Grid
    x0::Float64     # αριστερό όριο του crop στην αρχική εικόνα
    y0::Float64     # πάνω όριο του crop στην αρχική εικόνα
    f::Float64      # grid pixels ανά pixel της αρχικής εικόνας
    w::Int          # πλάτος του grid (στήλες)
    h::Int          # ύψος του grid (γραμμές)
    mpp::Float64    # μέτρα ανά pixel του grid
end

# σημείο της αρχικής εικόνας -> συνεχείς συντεταγμένες grid (το cell (i, j) καλύπτει [j-1, j] × [i-1, i])
to_grid(g::Grid, p) = ((p[1] - g.x0) * g.f, (p[2] - g.y0) * g.f)

# Γέμισμα πολυγώνου (even-odd) στα κέντρα των cells· pts = λίστα (x, y) σε συντεταγμένες grid
function fill_polygon!(mask::BitMatrix, pts, value::Bool)
    h, w = size(mask)
    n = length(pts)
    n < 3 && return mask
    xs = Float64[]
    for i in 1:h
        yc = i - 0.5                                    # y του κέντρου της γραμμής i
        empty!(xs)
        for k in 1:n
            (xa, ya) = pts[k]
            (xb, yb) = pts[k == n ? 1 : k + 1]
            if (ya <= yc) != (yb <= yc)                 # η ακμή τέμνει την οριζόντια γραμμή
                push!(xs, xa + (yc - ya) / (yb - ya) * (xb - xa))
            end
        end
        sort!(xs)
        for k in 1:2:(length(xs) - 1)
            jmin = max(1, ceil(Int, xs[k] + 0.5))
            jmax = min(w, ceil(Int, xs[k + 1] + 0.5) - 1)
            jmin <= jmax && (mask[i, jmin:jmax] .= value)
        end
    end
    return mask
end

# Μάσκα περιοχής από λίστα πολυγώνων {mode: "add"|"subtract", points}, εφαρμόζονται με τη σειρά
function region_mask(polys, g::Grid)
    mask = falses(g.h, g.w)
    for p in polys
        pts = [to_grid(g, q) for q in p.points]
        tmp = falses(g.h, g.w)
        fill_polygon!(tmp, pts, true)
        if get(p, :mode, "add") == "subtract"
            mask .&= .!tmp
        else
            mask .|= tmp
        end
    end
    return mask
end

# Ορθογώνιο παραλληλόγραμμο από ευθύγραμμο τμήμα p1–p2 (πλάτος της εξόδου) και πάχος σε grid pixels
function rect_corners(p1, p2, thickness_px::Real)
    dx, dy = p2[1] - p1[1], p2[2] - p1[2]
    len = hypot(dx, dy)
    len == 0 && return [p1, p1, p1, p1]
    nx, ny = -dy / len * thickness_px / 2, dx / len * thickness_px / 2     # κάθετο διάνυσμα
    return [(p1[1] + nx, p1[2] + ny), (p2[1] + nx, p2[2] + ny), (p2[1] - nx, p2[2] - ny), (p1[1] - nx, p1[2] - ny)]
end

# Μάσκα ενός ορθογωνίου εξόδου (p1, p2 σε συντεταγμένες αρχικής εικόνας, thickness_m σε μέτρα)
function exit_mask(e, g::Grid)
    pts = rect_corners(to_grid(g, e.p1), to_grid(g, e.p2), max(e.thickness_m / g.mpp, 1.5))
    return fill_polygon!(falses(g.h, g.w), pts, true)
end

# Heightmap (UInt8) από τον χάρτη κλάσεων· οι έξοδοι ανοίγουν βατό πέρασμα στους τοίχους και στο περίγραμμα
function compose_heightmap(labels::Matrix{UInt8}, exits::BitMatrix)
    hm = [HEIGHT_OF_CLASS[min(Int(c), 4) + 1] for c in labels]
    for idx in findall(exits)
        hm[idx] = max(hm[idx], HEIGHT_GROUND)           # το πέρασμα είναι τουλάχιστον τόσο βατό όσο ο ανοιχτός χώρος
    end
    return hm
end

# Βατά cells με τον ίδιο κανόνα του σεναρίου: walkmap[heightmap .> 245] .= false μετά την αντιστροφή
walkable(hm::Matrix{UInt8}) = (255 .- Int.(hm)) .<= WHITE_THRESHOLD

# Συνεκτικές συνιστώσες (8 γειτόνων) ενός BitMatrix· επιστρέφει πίνακα ετικετών (0 = κενό)
function components(M::AbstractMatrix{Bool})
    h, w = size(M)
    lab = zeros(Int, h, w)
    n = 0
    stack = Tuple{Int,Int}[]
    for j in 1:w, i in 1:h
        (M[i, j] && lab[i, j] == 0) || continue
        n += 1
        lab[i, j] = n
        push!(stack, (i, j))
        while !isempty(stack)
            (ci, cj) = pop!(stack)
            for di in -1:1, dj in -1:1
                ni, nj = ci + di, cj + dj
                (1 <= ni <= h && 1 <= nj <= w) || continue
                if M[ni, nj] && lab[ni, nj] == 0
                    lab[ni, nj] = n
                    push!(stack, (ni, nj))
                end
            end
        end
    end
    return lab, n
end
