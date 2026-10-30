# Σχεδιάζει τις 16 υποψήφιες εξόδους (DESTS_ALL) του Scenario 4 πάνω στην εικόνα Scenario 5.jpg, αριθμημένες και χρωματισμένες ανά πλευρά (δεν είναι προσομοίωση)
# Overlay Scenario 4 goals (DESTS_ALL) on Concentration Maps/Scenario 5.jpg
# Labels: global index 1:16 (indices used in DEST_INDICES_BY_SIDE point into DESTS_ALL).
# Scatter color by side A-D per DEST_INDICES_BY_SIDE.
#
# Run from repo root:
#   julia "SCENARIOS/SCENARIO 4/plot_dests_on_scenario5.jl"
#
# Uses NADEEN/Project.toml (same deps as Scenario 4.jl: CairoMakie, FileIO, Images, …).

# Ενεργοποίηση του περιβάλλοντος (Project.toml) του φακέλου NADEEN, ώστε να βρεθούν τα πακέτα
import Pkg  # διαχείριση πακέτων/περιβαλλόντων
const _REPO_FOR_ENV = dirname(dirname(@__DIR__))  # ρίζα του repo: δύο φάκελοι πάνω από τον φάκελο του script
Pkg.activate(joinpath(_REPO_FOR_ENV, "NADEEN"))  # ενεργοποίηση του περιβάλλοντος NADEEN

using CairoMakie  # σχεδίαση διαγραμμάτων
using FileIO: load  # φόρτωση εικόνων
using ImageMagick  # υποστήριξη μορφών εικόνας
using Images  # επεξεργασία εικόνων (RGB)
using Dates  # ημερομηνία/ώρα για το timestamp του αρχείου εξόδου

# Same as Scenario 4.jl
const DESTS_ALL = Tuple{Float64, Float64}[  # οι 16 υποψήφιες έξοδοι (x, y), δείκτες 1..16
    (500., 854.), (120., 248.), (691., 750.), (351., 924.),
    (853., 659.), (795., 375.), (703., 187.), (864., 503.),
    (652., 73.), (325., 150.), (223., 198.), (473., 78.),
    (88., 493.), (145., 630.), (209., 787.), (39., 379.),
]

# Ομαδοποίηση των εξόδων σε 4 πλευρές: κάθε λίστα περιέχει καθολικούς δείκτες στο DESTS_ALL
const DEST_INDICES_BY_SIDE = Vector{Vector{Int}}([
    [1, 3, 4, 5],
    [6, 7, 8, 9],
    [10, 11, 12, 2],
    [13, 14, 15, 16],
])

const SIDE_LABELS = ['A', 'B', 'C', 'D']  # γράμματα των πλευρών
const SIDE_COLORS = [:crimson, :dodgerblue, :forestgreen, :darkorange]  # χρώμα κάθε πλευράς

# Για έναν καθολικό δείκτη g επιστρέφει (αριθμός πλευράς, γράμμα πλευράς, χρώμα, τοπική θέση μέσα στην πλευρά)
function side_and_local_for_global(g::Int)
    for (s, idxs) in enumerate(DEST_INDICES_BY_SIDE)  # για κάθε πλευρά και τους δείκτες της
        k = findfirst(==(g), idxs)  # θέση του g μέσα στη λίστα της πλευράς (nothing αν δεν υπάρχει)
        if k !== nothing  # αν βρέθηκε
            return s, SIDE_LABELS[s], SIDE_COLORS[s], Int(k)  # επιστροφή των στοιχείων της πλευράς
        end
    end
    error("Global index $g not found in DEST_INDICES_BY_SIDE")  # σφάλμα αν ο δείκτης δεν ανήκει σε καμία πλευρά
end

scenario5_path = joinpath(_REPO_FOR_ENV, "Concentration Maps", "Scenario 5.jpg")  # διαδρομή της εικόνας Scenario 5.jpg
out_dir = joinpath(@__DIR__, "Simulation Results")  # φάκελος εξόδου των αποτελεσμάτων

img_loaded = load(scenario5_path)  # φόρτωση της εικόνας
img_rgb = Matrix(RGB.(img_loaded))  # μετατροπή σε πίνακα RGB
h_img, w_img = size(img_rgb)  # διαστάσεις της εικόνας (πλήθος γραμμών, πλήθος στηλών)

xs = Float64[d[1] for d in DESTS_ALL]  # οι συντεταγμένες x των εξόδων
ys = Float64[d[2] for d in DESTS_ALL]  # οι συντεταγμένες y των εξόδων
colors = Symbol[]  # χρώματα των σημείων
labels_simple = String[]  # ετικέτες των σημείων
# για κάθε έξοδο: χρώμα της πλευράς της και ετικέτα της μορφής δείκτης (γράμμα και τοπική θέση)
for g in 1:16
    _, letter, col, loc = side_and_local_for_global(g)  # στοιχεία της πλευράς της εξόδου
    push!(colors, col)  # αποθήκευση του χρώματος
    push!(labels_simple, "$(g) ($(letter)$(loc))")  # αποθήκευση της ετικέτας
end

fig = Figure(; size = (1100, 1000))  # σχήμα του διαγράμματος
ax = Axis(  # άξονας με ίσες κλίμακες στους δύο άξονες
    fig[1, 1];
    title = "DESTS_ALL (1-16) on Scenario 5.jpg — color by side A-D",
    aspect = DataAspect(),
    xlabel = "x (same frame as Scenario 4)",
    ylabel = "y",
)

# η εικόνα ως φόντο
image!(ax, 1:w_img, 1:h_img, img_rgb)  # οι διαστάσεις w_img, h_img λαμβάνονται αυτόματα από το μέγεθος της εικόνας (size(img_rgb)), οπότε δουλεύει για οποιαδήποτε εικόνα

# οι έξοδοι ως έγχρωμοι κύκλοι
scatter!(
    ax, xs, ys;
    color = colors,
    markersize = 14,
    strokecolor = :black,
    strokewidth = 1.5,
)

# αριθμητικές ετικέτες δίπλα σε κάθε έξοδο
text!(
    ax, xs, ys;
    text = labels_simple,
    fontsize = 9,
    align = (:left, :bottom),
    offset = (6, 6),
    color = :white,
    strokewidth = 1.2,
    strokecolor = :black,
)

xlims!(ax, 0, w_img + 1)  # όρια του άξονα x ίσα με τις διαστάσεις της εικόνας
ylims!(ax, 0, h_img + 1)  # όρια του άξονα y ίσα με τις διαστάσεις της εικόνας

elem = [MarkerElement(color = SIDE_COLORS[i], marker = :circle, markersize = 12) for i in 1:4]  # στοιχεία του υπομνήματος: ένας κύκλος ανά πλευρά
Legend(  # υπόμνημα με τις 4 πλευρές
    fig[1, 2],
    elem,
    ["Side $(SIDE_LABELS[i])" for i in 1:4];
    tellheight = false,
    tellwidth = false,
)

# δημιουργία του φακέλου εξόδου αν δεν υπάρχει
mkpath(out_dir)
ts = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")  # χρονική σήμανση
out_png = joinpath(out_dir, "Scenario5_DESTS_ALL_numbered_$ts.png")  # διαδρομή του αρχείου png
save(out_png, fig)  # αποθήκευση της εικόνας
println("Saved: $out_png")  # μήνυμα αποθήκευσης
println("Image size: $(w_img) x $(h_img) px")  # εκτύπωση των διαστάσεων της εικόνας
# έλεγχος ότι κάθε έξοδος βρίσκεται μέσα στα όρια της εικόνας, αλλιώς εμφανίζεται προειδοποίηση
for g in 1:16
    x, y = DESTS_ALL[g]  # συντεταγμένες της εξόδου
    if x < 1 || x > w_img || y < 1 || y > h_img  # αν η έξοδος είναι εκτός ορίων της εικόνας
        @warn "Point $g = ($x, $y) outside image bounds 1:$w_img x 1:$h_img"  # προειδοποίηση για την έξοδο εκτός ορίων
    end
end
