# Ανάλυση των αποτελεσμάτων του benchmark του Scenario 4 (δεν είναι προσομοίωση): διαβάζει τον πίνακα των runs,
# υπολογίζει στατιστικά ανά πλήθος ενεργών εξόδων (N_active_goals) και σχεδιάζει boxplots και violin plots.
using CSV  # εγγραφή/ανάγνωση αρχείων CSV
using DataFrames  # πίνακες δεδομένων (DataFrame)
using Statistics  # στατιστικά (mean, median)
using CairoMakie  # σχεδίαση διαγραμμάτων
using XLSX  # ανάγνωση αρχείων Excel (xlsx)
using Dates  # ημερομηνία/ώρα για το timestamp των αρχείων εξόδου

# =========================
# Ρυθμίσεις
# =========================
input_file = raw"C:\Users\gavin\Documents\GitHub\evac\SCENARIOS\SCENARIO 4\Simulation Results\AStar 10 Agents Benchmarking_2026-04-17_08-12-13.csv"  # σταθερή διαδρομή σε ένα συγκεκριμένο αρχείο αποτελεσμάτων benchmark (CSV ή XLSX)· για να αναλυθεί άλλο run πρέπει να αλλαχτεί χειροκίνητα εδώ
out_dir    = raw"C:\Users\gavin\Documents\GitHub\evac\SCENARIOS\SCENARIO 4\Simulation Results"  # φάκελος όπου αποθηκεύονται οι έξοδοι (πίνακας σύνοψης και διαγράμματα)

# Αν true, παράγει και violin plots
make_violin = true

# =========================
# Βοηθητικά
# =========================
# Διαβάζει τον πίνακα των runs από αρχείο CSV ή XLSX, ανάλογα με την κατάληξη
function read_runs_table(path::AbstractString)
    lower = lowercase(path)  # κατάληξη του αρχείου με πεζά γράμματα
    if endswith(lower, ".csv")  # περίπτωση CSV
        # Το αρχείο σου είναι ; separated και decimal comma
        return CSV.read(path, DataFrame; delim = ';', decimal = ',')
    elseif endswith(lower, ".xlsx")  # περίπτωση XLSX
        # Περιμένει φύλλο "Runs"
        data, names = XLSX.readtable(path, "Runs")
        return DataFrame(data, Symbol.(names))  # μετατροπή των δεδομένων σε DataFrame με τα ονόματα των στηλών
    else
        error("Unsupported file type: $path")  # σφάλμα για μη υποστηριζόμενη κατάληξη
    end
end

# Κανονικοποιεί ένα όνομα στήλης: αφαιρεί κενά στις άκρες και τον χαρακτήρα BOM (\ufeff)
normalize_colname(s::AbstractString) = replace(strip(s), '\ufeff' => "")

# Βρίσκει ποια στήλη του πίνακα ταιριάζει με κάποιο από τα ονόματα candidates (χωρίς διάκριση πεζών/κεφαλαίων) και επιστρέφει το όνομά της
function resolve_column(df::DataFrame, candidates::Vector{String}; required::Bool = true)
    normalized_map = Dict{String, Symbol}()  # αντιστοίχιση κανονικοποιημένου ονόματος προς το πραγματικό όνομα στήλης
    for n in propertynames(df)  # για κάθε στήλη του πίνακα
        normalized_map[lowercase(normalize_colname(String(n)))] = Symbol(n)  # αποθήκευση του κανονικοποιημένου ονόματος
    end
    for c in candidates  # δοκιμάζει τα υποψήφια ονόματα με τη σειρά
        key = lowercase(normalize_colname(c))  # κανονικοποίηση του υποψήφιου ονόματος
        if haskey(normalized_map, key)  # αν υπάρχει στήλη με αυτό το όνομα
            return normalized_map[key]  # επιστρέφει το πραγματικό όνομα της στήλης
        end
    end
    if required  # αν η στήλη είναι υποχρεωτική και δεν βρέθηκε: εκτυπώνει τις διαθέσιμες στήλες και σταματά με σφάλμα
        println("Available columns: ", propertynames(df))
        error("Could not find any of these columns: $(candidates)")
    end
    return nothing  # αν η στήλη δεν είναι υποχρεωτική και δεν βρέθηκε επιστρέφει nothing
end

# Μετονομάζει (in-place) τις στήλες του πίνακα στα τυπικά ονόματα που χρησιμοποιεί η ανάλυση, δεχόμενη εναλλακτικά ονόματα
function rename_columns_to_standard!(df::DataFrame)
    colmap = Dict{Symbol, Vector{String}}(  # για κάθε τυπικό όνομα: η λίστα των αποδεκτών εναλλακτικών ονομάτων
        :InitialPathfindingTime_s => ["InitialPathfindingTime_s", "InitialPathfindingTime", "Initial PF Time"],
        :IncrementalPathfindingTime_s => ["IncrementalPathfindingTime_s", "IncrementalPathfindingTime", "Incremental PF Time"],
        :SimulationTime_s => ["SimulationTime_s", "SimulationTime"],
        :SumTL_capped => ["SumTL_capped", "SumTL", "Sum TL capped"],
        :N_active_goals => ["N_active_goals", "N Active Goals", "N_active_goal"],
    )

    for (target, aliases) in colmap  # για κάθε τυπικό όνομα
        actual = resolve_column(df, aliases)  # βρίσκει την πραγματική στήλη
        if actual != target  # αν το όνομα διαφέρει
            rename!(df, actual => target)  # μετονομασία στο τυπικό όνομα
        end
    end
    return df  # επιστρέφει τον ίδιο πίνακα
end

# Εξασφαλίζει ότι οι στήλες cols είναι αριθμητικές (αν είναι String με δεκαδικό κόμμα, τις μετατρέπει σε Float64)
function ensure_numeric!(df::DataFrame, cols::Vector{Symbol})
    for c in cols  # για κάθε στήλη
        if !(c in propertynames(df))  # έλεγχος ότι η στήλη υπάρχει
            error("Missing expected column: $(c)")  # σφάλμα αν λείπει η στήλη
        end
        if eltype(df[!, c]) <: Number  # αν είναι ήδη αριθμητική, δεν χρειάζεται μετατροπή
            continue
        end
        # fallback αν ήρθε ως String με decimal comma
        df[!, c] = parse.(Float64, replace.(string.(df[!, c]), ',' => '.'))  # αντικατάσταση του κόμματος με τελεία και μετατροπή σε Float64
    end
end

# Εξασφαλίζει ότι η στήλη col περιέχει ακεραίους (στρογγυλοποιεί αν χρειάζεται)
function sanitize_k!(df::DataFrame, col::Symbol)
    if !(col in propertynames(df))  # έλεγχος ότι η στήλη υπάρχει
        error("Missing expected column: $(col)")  # σφάλμα αν λείπει η στήλη
    end
    if !(eltype(df[!, col]) <: Integer)  # αν δεν είναι ακέραιου τύπου
        df[!, col] = Int.(round.(Float64.(df[!, col])))  # στρογγυλοποίηση και μετατροπή σε Int
    end
end

# =========================
# 1) Φόρτωση
# =========================
df = read_runs_table(input_file)  # ανάγνωση του πίνακα των runs
rename_columns_to_standard!(df)  # μετονομασία των στηλών στα τυπικά ονόματα

required_numeric = [  # στήλες που πρέπει να είναι αριθμητικές
    :InitialPathfindingTime_s,
    :IncrementalPathfindingTime_s,
    :SimulationTime_s,
    :SumTL_capped,
]
ensure_numeric!(df, required_numeric)  # μετατροπή σε αριθμητικές στήλες όπου χρειάζεται
sanitize_k!(df, :N_active_goals)  # το πλήθος ενεργών εξόδων γίνεται ακέραιος

# =========================
# 2) Νέα παράγωγη στήλη
# =========================
# συνολικός χρόνος εύρεσης διαδρομών = αρχικός + incremental (επανασχεδιασμός)
df.TotalPathfinding_s = df.InitialPathfindingTime_s .+ df.IncrementalPathfindingTime_s

# =========================
# 3) Group-by summary
# =========================
gdf = groupby(df, :N_active_goals)  # ομαδοποίηση των runs ανά πλήθος ενεργών εξόδων

# σύνοψη ανά ομάδα: πλήθος runs, μέσος όρος και διάμεσος του SumTL_capped και του συνολικού χρόνου διαδρομών, μέσος χρόνος προσομοίωσης
summary_df = combine(
    gdf,
    nrow => :n_runs,  # πλήθος runs στην ομάδα
    :SumTL_capped => mean => :Mean_SumTL_capped,
    :SumTL_capped => median => :Median_SumTL_capped,
    :TotalPathfinding_s => mean => :Mean_TotalPathfinding_s,
    :TotalPathfinding_s => median => :Median_TotalPathfinding_s,
    :SimulationTime_s => mean => :Mean_SimulationTime_s,
)

sort!(summary_df, :N_active_goals)  # ταξινόμηση κατά πλήθος ενεργών εξόδων

println("=== Summary by N_active_goals ===")  # εκτύπωση του πίνακα σύνοψης
show(summary_df, allrows = true, allcols = true)  # εμφάνιση όλων των γραμμών και στηλών
println()

# save summary table
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")  # χρονική σήμανση για το όνομα του αρχείου
summary_csv = joinpath(out_dir, "Scenario4_GroupBy_N_active_goals_$timestamp.csv")  # διαδρομή του CSV σύνοψης
CSV.write(summary_csv, summary_df)  # αποθήκευση του πίνακα σύνοψης σε CSV
println("Saved summary table to: $summary_csv")

# =========================
# 4) Boxplots
# =========================
# 4a) SumTL_capped vs N_active_goals
fig1 = Figure(size = (1000, 600))  # σχήμα για το boxplot του SumTL_capped ανά πλήθος ενεργών εξόδων
ax1 = Axis(  # άξονας του διαγράμματος
    fig1[1, 1],
    title = "SumTL_capped vs N_active_goals",
    xlabel = "N_active_goals",
    ylabel = "SumTL_capped",
)
boxplot!(ax1, df.N_active_goals, df.SumTL_capped; color = :steelblue, whiskerwidth = 0.5)  # boxplot ανά τιμή του N_active_goals
scatter!(ax1, df.N_active_goals, df.SumTL_capped; color = (:black, 0.18), markersize = 5)  # τα μεμονωμένα σημεία (runs) πάνω από το boxplot
sumtl_box_png = joinpath(out_dir, "Scenario4_Boxplot_SumTL_vs_Nactive_$timestamp.png")  # διαδρομή του αρχείου png
save(sumtl_box_png, fig1)  # αποθήκευση της εικόνας
println("Saved: $sumtl_box_png")

# 4b) TotalPathfinding_s vs N_active_goals
fig2 = Figure(size = (1000, 600))  # σχήμα για το boxplot του συνολικού χρόνου εύρεσης διαδρομών
ax2 = Axis(  # άξονας του διαγράμματος
    fig2[1, 1],
    title = "TotalPathfinding_s vs N_active_goals",
    xlabel = "N_active_goals",
    ylabel = "TotalPathfinding_s",
)
boxplot!(ax2, df.N_active_goals, df.TotalPathfinding_s; color = :darkorange, whiskerwidth = 0.5)  # boxplot ανά τιμή του N_active_goals
scatter!(ax2, df.N_active_goals, df.TotalPathfinding_s; color = (:black, 0.18), markersize = 5)  # τα μεμονωμένα σημεία (runs) πάνω από το boxplot
pf_box_png = joinpath(out_dir, "Scenario4_Boxplot_TotalPF_vs_Nactive_$timestamp.png")  # διαδρομή του αρχείου png
save(pf_box_png, fig2)  # αποθήκευση της εικόνας
println("Saved: $pf_box_png")

# =========================
# 5) Προαιρετικά violin plots
# =========================
if make_violin  # μόνο αν make_violin == true
    fig3 = Figure(size = (1000, 600))  # σχήμα για το violin plot του SumTL_capped
    ax3 = Axis(  # άξονας του διαγράμματος
        fig3[1, 1],
        title = "Violin: SumTL_capped vs N_active_goals",
        xlabel = "N_active_goals",
        ylabel = "SumTL_capped",
    )
    violin!(ax3, df.N_active_goals, df.SumTL_capped; color = (:mediumpurple, 0.6), strokecolor = :black)  # violin plot ανά τιμή του N_active_goals
    sumtl_violin_png = joinpath(out_dir, "Scenario4_Violin_SumTL_vs_Nactive_$timestamp.png")  # διαδρομή του αρχείου png
    save(sumtl_violin_png, fig3)  # αποθήκευση της εικόνας
    println("Saved: $sumtl_violin_png")

    fig4 = Figure(size = (1000, 600))  # σχήμα για το violin plot του συνολικού χρόνου εύρεσης διαδρομών
    ax4 = Axis(  # άξονας του διαγράμματος
        fig4[1, 1],
        title = "Violin: TotalPathfinding_s vs N_active_goals",
        xlabel = "N_active_goals",
        ylabel = "TotalPathfinding_s",
    )
    violin!(ax4, df.N_active_goals, df.TotalPathfinding_s; color = (:seagreen, 0.6), strokecolor = :black)  # violin plot ανά τιμή του N_active_goals
    pf_violin_png = joinpath(out_dir, "Scenario4_Violin_TotalPF_vs_Nactive_$timestamp.png")  # διαδρομή του αρχείου png
    save(pf_violin_png, fig4)  # αποθήκευση της εικόνας
    println("Saved: $pf_violin_png")
end

println("\nDone.")
