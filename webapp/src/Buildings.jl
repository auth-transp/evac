# Κτίρια: τοίχοι και πόρτες πάνω στον χάρτη κλάσεων (CLASS_*). Χρησιμοποιείται από την εισαγωγή OpenStreetMap (Osm.jl).

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

# Διάβρωση με τετράγωνο παράθυρο (2r+1)²: ένα cell μένει true όταν ολόκληρο το (κομμένο στα όρια) παράθυρο είναι true
function erode_box(M::BitMatrix, r::Int)
    r <= 0 && return copy(M)
    cnt, full = window_count(M, r)
    return cnt .== full                                 # ολόκληρο το (κομμένο στα όρια) παράθυρο είναι true
end

# --- Τοίχοι και πόρτες κτιρίων ---

# Τοίχος = ο λεπτότερος δυνατός κλειστός δακτύλιος: τα cells του κτιρίου που ακουμπούν τον έξω χώρο, ακόμη και διαγώνια.
# Ο D* Lite έχει γειτονιά Moore χωρίς έλεγχο κοπής γωνίας: ένας agent μπορεί να πάει διαγώνια σε οποιοδήποτε βατό cell, άρα ένα cell του
# κτιρίου με διαγώνιο γείτονα έξω χώρο θα ήταν «διαρροή». Για αυτό ο δακτύλιος είναι 1 cell παχύς σε οριζόντιες/κάθετες πλευρές και 2 cells
# σε πλευρές 45°, και δεν γίνεται λεπτότερος (το test_doors.jl ελέγχει ότι σχεδόν κάθε cell του τοίχου είναι απαραίτητο για να μη δημιουργηθεί διαρροή).
# Για πιο λεπτό τοίχο σε μέτρα χρειάζεται μικρότερο m/pixel στο grid.
building_walls(bmask::BitMatrix) = bmask .& .!erode_box(bmask, 1)

const DOOR_SNAP_M = 12.0                            # είσοδος OSM μέσα σε τόσα m από τοίχο ανοίγει πόρτα σε αυτόν
# Πλάτος πόρτας σε μέτρα: αρκετό ώστε να διασχίζουν ταυτόχρονα 3 agents (διάμετρος 1 m, συν χώρος για την άπωση του SFM, ~2 m ανά agent).
# Σε cells: ceil(DOOR_WIDTH_M / mpp), δηλαδή 6 cells στο 1 m/pixel και 2 cells (6.9 m) στο 3.46 m/pixel· πάντα τουλάχιστον 1 cell.
const DOOR_WIDTH_M = 6.0
door_width_cells(mpp::Real) = max(1, ceil(Int, DOOR_WIDTH_M / mpp - 1e-9))
const DOOR_SPACING_M = 60.0                         # τουλάχιστον μία πόρτα ανά τόσα μέτρα κοινού τοίχου (κτιρίου και ανοιχτού χώρου)

# Ανοίγει πόρτες στους τοίχους ώστε οι agents να μην εγκλωβίζονται μέσα στα κτίρια: ένα cell τοίχου γίνεται CLASS_BUILDING
# (χρώμα κτιρίου) εκεί που συνδέει το εσωτερικό με ανοιχτό χώρο. 1) Στις εισόδους `entrances` (σημεία (x, y) του grid)·
# 2) αυτόματα, ώστε κάθε κτίριο να έχει τουλάχιστον μία πόρτα ανά DOOR_SPACING_M μέτρα τοίχου προς κάθε ανοιχτό χώρο που το περιβάλλει
# (και προς κλειστούς εσωτερικούς χώρους, π.χ. αγωνιστικό χώρο σταδίου): η πρώτη πιο κοντά σε δρόμο, οι υπόλοιπες ομοιόμορφα κατανεμημένες.
# Επιστρέφει (πόρτες από εισόδους, αυτόματες πόρτες).
function open_doors!(labels::Matrix{UInt8}, entrances, mpp::Real)
    h, w = size(labels)
    at(i, j) = (1 <= i <= h && 1 <= j <= w) ? labels[i, j] : CLASS_OUTSIDE
    open_space(c) = c == CLASS_GROUND || c == CLASS_ROAD
    nbrs = ((-1, 0), (1, 0), (0, -1), (0, 1))
    is_candidate(i, j) = labels[i, j] == CLASS_WALL && any(((di, dj),) -> at(i + di, j + dj) == CLASS_BUILDING, nbrs) &&
                         any(((di, dj),) -> open_space(at(i + di, j + dj)), nbrs)

    # η πόρτα ανοίγει στο cell που επιλέχθηκε και στα cells τοίχου γύρω του (block wcells × wcells με κέντρο το cell, μετατοπισμένο κατά
    # ένα cell αν το πλάτος είναι ζυγό): σε ευθύ τοίχο είναι wcells cells κατά μήκος του τοίχου
    wcells = door_width_cells(mpp)
    lo, hi = -((wcells - 1) ÷ 2), wcells ÷ 2
    function open_door!(i, j)
        for di in lo:hi, dj in lo:hi
            (1 <= i + di <= h && 1 <= j + dj <= w && labels[i + di, j + dj] == CLASS_WALL) && (labels[i + di, j + dj] = CLASS_BUILDING)
        end
    end

    n_entrance = 0
    r = ceil(Int, DOOR_SNAP_M / mpp)
    for e in entrances
        ci, cj = floor(Int, e[2]) + 1, floor(Int, e[1]) + 1
        best, bd = nothing, Inf
        for i in max(1, ci - r):min(h, ci + r), j in max(1, cj - r):min(w, cj + r)
            is_candidate(i, j) || continue
            d = hypot(i - ci, j - cj)
            d <= r && d < bd && (bd = d; best = (i, j))
        end
        best === nothing && continue
        open_door!(best...)
        n_entrance += 1
    end

    # Κάθε κτίριο (συνεκτική συνιστώσα του CLASS_BUILDING) συνορεύει με έναν ή περισσότερους ανοιχτούς χώρους: τον κύριο χώρο έξω και
    # ενδεχομένως κλειστούς εσωτερικούς χώρους (π.χ. ο αγωνιστικός χώρος ενός σταδίου). Για κάθε ζεύγος (κτίριο, ανοιχτός χώρος) ανοίγονται
    # πόρτες ώστε να υπάρχει τουλάχιστον μία ανά DOOR_SPACING_M μέτρα κοινού τοίχου (και πάντα τουλάχιστον μία: τίποτα δεν εγκλωβίζεται).
    reg, nreg = components(labels .== CLASS_BUILDING; diag = false)
    ocomp, no = components(BitMatrix(open_space.(labels)))                   # ανοιχτοί χώροι (8 γειτόνων, όπως στον D* Lite)
    osize = zeros(Int, no)
    foreach(v -> v > 0 && (osize[v] += 1), ocomp)
    main = Set(ocomp[i, j] for j in 1:w, i in 1:h if labels[i, j] == CLASS_ROAD)       # κύριος χώρος: αυτός που περιέχει δρόμους, αλλιώς ο μεγαλύτερος
    isempty(main) && no > 0 && push!(main, argmax(osize))

    # απόσταση από δρόμο (BFS, 4 γείτονες) για να επιλεγεί το πλησιέστερο σημείο
    dist = fill(typemax(Int), h, w)
    queue = Tuple{Int,Int}[(i, j) for j in 1:w, i in 1:h if labels[i, j] == CLASS_ROAD]
    foreach(((i, j),) -> dist[i, j] = 0, queue)
    head = 1
    while head <= length(queue)
        (i, j) = queue[head]; head += 1
        for (di, dj) in nbrs
            ni, nj = i + di, j + dj
            (1 <= ni <= h && 1 <= nj <= w && dist[ni, nj] == typemax(Int)) || continue
            dist[ni, nj] = dist[i, j] + 1
            push!(queue, (ni, nj))
        end
    end

    # υποψήφια cells τοίχου ανά ζεύγος (κτίριο, ανοιχτός χώρος)
    inbounds(i, j) = 1 <= i <= h && 1 <= j <= w
    pair_cands = Dict{Tuple{Int,Int},Vector{Tuple{Int,Int}}}()
    for j in 1:w, i in 1:h
        is_candidate(i, j) || continue
        rs = unique(reg[i + di, j + dj] for (di, dj) in nbrs if inbounds(i + di, j + dj) && reg[i + di, j + dj] > 0)
        os = unique(ocomp[i + di, j + dj] for (di, dj) in nbrs if inbounds(i + di, j + dj) && open_space(labels[i + di, j + dj]))
        for r in rs, o in os
            push!(get!(pair_cands, (r, o), Tuple{Int,Int}[]), (i, j))
        end
    end

    # υπάρχουσες πόρτες: cells κτιρίου που ακουμπούν ανοιχτό χώρο, ομαδοποιημένα (8 γείτονες) ανά πόρτα και ζεύγος
    doormask = BitMatrix([labels[i, j] == CLASS_BUILDING && any(((di, dj),) -> open_space(at(i + di, j + dj)), nbrs) for i in 1:h, j in 1:w])
    dcl, _ = components(doormask)
    door_ids = Dict{Tuple{Int,Int},Set{Int}}()
    door_cells = Dict{Tuple{Int,Int},Vector{Tuple{Int,Int}}}()
    for j in 1:w, i in 1:h
        doormask[i, j] || continue
        for (di, dj) in nbrs
            (inbounds(i + di, j + dj) && open_space(labels[i + di, j + dj])) || continue
            key = (reg[i, j], ocomp[i + di, j + dj])
            push!(get!(door_ids, key, Set{Int}()), dcl[i, j])
            push!(get!(door_cells, key, Tuple{Int,Int}[]), (i, j))
        end
    end

    # κατεύθυνση προς τον ανοιχτό χώρο και πλήθος γειτονικών υποψηφίων με την ίδια κατεύθυνση: ευθεία πλευρά (όχι γωνία) προτιμάται
    outdir(i, j) = (sum(di for (di, dj) in nbrs if open_space(at(i + di, j + dj)); init = 0), sum(dj for (di, dj) in nbrs if open_space(at(i + di, j + dj)); init = 0))
    ncand(i, j) = count(is_candidate(i + di, j + dj) && outdir(i + di, j + dj) == outdir(i, j)
                        for di in lo:hi, dj in lo:hi if inbounds(i + di, j + dj))

    n_auto = 0
    for ((r, o), cands) in pair_cands
        (o in main || osize[o] >= 4) || continue                              # μικροί θύλακες ανοιχτού χώρου (1-3 cells) δεν χρειάζονται πόρτα
        perimeter_cells = length(cands) + length(get(door_cells, (r, o), Tuple{Int,Int}[]))     # τοίχος + οι ήδη ανοιγμένες πόρτες
        need = max(1, ceil(Int, perimeter_cells * mpp / DOOR_SPACING_M))       # τουλάχιστον μία ανά DOOR_SPACING_M μέτρα τοίχου
        add = need - length(get(door_ids, (r, o), Set{Int}()))
        chosen = copy(get(door_cells, (r, o), Tuple{Int,Int}[]))
        for _ in 1:add
            avail = [c for c in cands if labels[c[1], c[2]] == CLASS_WALL]
            isempty(avail) && break
            best = if isempty(chosen)                                          # πρώτη πόρτα: το πιο κοντινό σημείο σε δρόμο
                argmin(c -> (dist[c[1], c[2]], -ncand(c[1], c[2])), avail)
            else                                                               # επόμενες: όσο πιο μακριά από τις υπάρχουσες (ομοιόμορφη κατανομή)
                argmin(c -> (-minimum(hypot(c[1] - d[1], c[2] - d[2]) for d in chosen), -ncand(c[1], c[2])), avail)
            end
            !isempty(chosen) && minimum(hypot(best[1] - d[1], best[2] - d[2]) for d in chosen) < wcells + 1 && break   # δεν χωρά άλλη πόρτα
            open_door!(best[1], best[2])
            push!(chosen, best)
            n_auto += 1
        end
    end
    return (n_entrance, n_auto)
end
