# Αυτόματη πρόταση χάρτη κλάσεων (ανοιχτός χώρος / δρόμοι / κτίρια / τοίχοι) από δορυφορική εικόνα
# Κλασική επεξεργασία εικόνας: ό,τι διαφέρει χρωματικά από το έδαφος είναι "δομή", και η δομή χωρίζεται
# σε κτίρια (πλατιά blobs) και δρόμους (στενές, επιμήκεις δομές). Είναι πρόταση, ο χρήστης τη διορθώνει.

# Αριθμός true cells σε παράθυρο (2r+1)² γύρω από κάθε cell, με integral image (τα όρια κόβονται)
function window_count(M::BitMatrix, r::Int)
    h, w = size(M)
    S = zeros(Int, h + 1, w + 1)
    for j in 1:w, i in 1:h
        S[i + 1, j + 1] = M[i, j] + S[i, j + 1] + S[i + 1, j] - S[i, j]
    end
    cnt = zeros(Int, h, w)
    full = zeros(Int, h, w)
    for j in 1:w, i in 1:h
        i0, i1, j0, j1 = max(1, i - r), min(h, i + r), max(1, j - r), min(w, j + r)
        cnt[i, j] = S[i1 + 1, j1 + 1] - S[i0, j1 + 1] - S[i1 + 1, j0] + S[i0, j0]
        full[i, j] = (i1 - i0 + 1) * (j1 - j0 + 1)
    end
    return cnt, full
end

dilate_box(M::BitMatrix, r::Int) = r <= 0 ? copy(M) : (window_count(M, r)[1] .> 0)
function erode_box(M::BitMatrix, r::Int)
    r <= 0 && return copy(M)
    cnt, full = window_count(M, r)
    return cnt .== full                                 # ολόκληρο το (κομμένο στα όρια) παράθυρο είναι true
end
opening_box(M::BitMatrix, r::Int) = dilate_box(erode_box(M, r), r)

# Αφαίρεση συνεκτικών συνιστωσών με λιγότερα από minpx cells
function remove_small(M::BitMatrix, minpx::Int)
    minpx <= 1 && return copy(M)
    lab, n = components(M)
    sizes = zeros(Int, n)
    for v in lab
        v > 0 && (sizes[v] += 1)
    end
    return BitMatrix([v > 0 && sizes[v] >= minpx for v in lab])
end

# Χωρίζει τις συνιστώσες σε συμπαγείς (κτίρια) και επιμήκεις/αραιές (δρόμοι): aspect του bounding box ≥ 8 ή γέμισμα < 0.35
function split_elongated(M::BitMatrix; max_aspect = 8.0, min_fill = 0.35)
    lab, n = components(M)
    i0 = fill(typemax(Int), n); i1 = zeros(Int, n); j0 = fill(typemax(Int), n); j1 = zeros(Int, n); area = zeros(Int, n)
    for j in axes(lab, 2), i in axes(lab, 1)
        v = lab[i, j]
        v == 0 && continue
        i0[v] = min(i0[v], i); i1[v] = max(i1[v], i); j0[v] = min(j0[v], j); j1[v] = max(j1[v], j); area[v] += 1
    end
    road = falses(n)
    for v in 1:n
        hh, ww = i1[v] - i0[v] + 1, j1[v] - j0[v] + 1
        road[v] = max(hh, ww) / min(hh, ww) >= max_aspect || area[v] / (hh * ww) < min_fill
    end
    return BitMatrix([v > 0 && !road[v] for v in lab]), BitMatrix([v > 0 && road[v] for v in lab])
end

# Μεσαίο χρώμα Lab των δειγμάτων (πίνακας Nx3)
median_color(samples) = [sort(samples[:, c])[(size(samples, 1) + 1) ÷ 2] for c in 1:3]

# Παράμετροι: sensitivity = κατώφλι χρωματικής απόστασης (ΔE) από το έδαφος· min_building_m = ελάχιστο πλάτος κτιρίου
# ground_sample = προαιρετικά σημείο (στο grid) από το οποίο διαβάζεται το χρώμα του εδάφους, αλλιώς η διάμεσος της περιοχής
function propose_labels(rgb::Matrix{<:Colorant}, region::BitMatrix, g::Grid;
                        sensitivity::Real = 14.0, min_building_m::Real = 8.0, min_area_m2::Real = 40.0,
                        ground_sample = nothing, smooth_px::Real = 1.0)
    smooth_px > 0 && (rgb = imfilter(rgb, Kernel.gaussian(smooth_px), "replicate"))   # εξομάλυνση: λιγότερος θόρυβος (speckle)
    lab = Lab.(rgb)
    L = Float64.(getfield.(lab, :l)); A = Float64.(getfield.(lab, :a)); B = Float64.(getfield.(lab, :b))
    idx = findall(region)
    isempty(idx) && return fill(CLASS_OUTSIDE, g.h, g.w)
    if ground_sample === nothing
        ref = median_color(hcat(L[idx], A[idx], B[idx]))
    else
        gi = clamp(round(Int, ground_sample[2] + 0.5), 1, g.h); gj = clamp(round(Int, ground_sample[1] + 0.5), 1, g.w)
        win = [(i, j) for i in max(1, gi - 2):min(g.h, gi + 2), j in max(1, gj - 2):min(g.w, gj + 2)]
        ref = median_color(hcat([L[i, j] for (i, j) in win], [A[i, j] for (i, j) in win], [B[i, j] for (i, j) in win]))
    end
    dE = sqrt.((L .- ref[1]) .^ 2 .+ (A .- ref[2]) .^ 2 .+ (B .- ref[3]) .^ 2)

    px_area = g.mpp^2
    structure = BitMatrix((dE .> sensitivity) .& region)
    structure = remove_small(structure, max(2, round(Int, min_area_m2 / px_area)))
    r = max(1, round(Int, min_building_m / g.mpp / 2))              # ακτίνα opening (cells)
    buildings = remove_small(opening_box(structure, r), max(3, round(Int, 2 * min_area_m2 / px_area)))
    buildings, elongated = split_elongated(buildings)               # επιμήκεις/αραιές συνιστώσες είναι δρόμοι, όχι κτίρια
    roads = remove_small((structure .& .!buildings) .| elongated, max(3, round(Int, min_area_m2 / px_area)))
    interior = erode_box(buildings, 1)
    walls = buildings .& .!interior

    labels = fill(CLASS_OUTSIDE, g.h, g.w)
    labels[region] .= CLASS_GROUND
    labels[roads .& region] .= CLASS_ROAD
    labels[interior .& region] .= CLASS_BUILDING
    labels[walls .& region] .= CLASS_WALL
    return labels
end
