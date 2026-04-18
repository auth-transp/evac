using CSV
using DataFrames
using Statistics
using CairoMakie
using XLSX
using Dates

# =========================
# Ρυθμίσεις
# =========================
input_file = raw"C:\Users\gavin\Documents\GitHub\evac\SCENARIOS\SCENARIO 4\Simulation Results\AStar 10 Agents Benchmarking_2026-04-17_08-12-13.csv"
out_dir    = raw"C:\Users\gavin\Documents\GitHub\evac\SCENARIOS\SCENARIO 4\Simulation Results"

# Αν true, παράγει και violin plots
make_violin = true

# =========================
# Βοηθητικά
# =========================
function read_runs_table(path::AbstractString)
    lower = lowercase(path)
    if endswith(lower, ".csv")
        # Το αρχείο σου είναι ; separated και decimal comma
        return CSV.read(path, DataFrame; delim = ';', decimal = ',')
    elseif endswith(lower, ".xlsx")
        # Περιμένει φύλλο "Runs"
        data, names = XLSX.readtable(path, "Runs")
        return DataFrame(data, Symbol.(names))
    else
        error("Unsupported file type: $path")
    end
end

normalize_colname(s::AbstractString) = replace(strip(s), '\ufeff' => "")

function resolve_column(df::DataFrame, candidates::Vector{String}; required::Bool = true)
    normalized_map = Dict{String, Symbol}()
    for n in propertynames(df)
        normalized_map[lowercase(normalize_colname(String(n)))] = Symbol(n)
    end
    for c in candidates
        key = lowercase(normalize_colname(c))
        if haskey(normalized_map, key)
            return normalized_map[key]
        end
    end
    if required
        println("Available columns: ", propertynames(df))
        error("Could not find any of these columns: $(candidates)")
    end
    return nothing
end

function rename_columns_to_standard!(df::DataFrame)
    colmap = Dict{Symbol, Vector{String}}(
        :InitialPathfindingTime_s => ["InitialPathfindingTime_s", "InitialPathfindingTime", "Initial PF Time"],
        :IncrementalPathfindingTime_s => ["IncrementalPathfindingTime_s", "IncrementalPathfindingTime", "Incremental PF Time"],
        :SimulationTime_s => ["SimulationTime_s", "SimulationTime"],
        :SumTL_capped => ["SumTL_capped", "SumTL", "Sum TL capped"],
        :N_active_goals => ["N_active_goals", "N Active Goals", "N_active_goal"],
    )

    for (target, aliases) in colmap
        actual = resolve_column(df, aliases)
        if actual != target
            rename!(df, actual => target)
        end
    end
    return df
end

function ensure_numeric!(df::DataFrame, cols::Vector{Symbol})
    for c in cols
        if !(c in propertynames(df))
            error("Missing expected column: $(c)")
        end
        if eltype(df[!, c]) <: Number
            continue
        end
        # fallback αν ήρθε ως String με decimal comma
        df[!, c] = parse.(Float64, replace.(string.(df[!, c]), ',' => '.'))
    end
end

function sanitize_k!(df::DataFrame, col::Symbol)
    if !(col in propertynames(df))
        error("Missing expected column: $(col)")
    end
    if !(eltype(df[!, col]) <: Integer)
        df[!, col] = Int.(round.(Float64.(df[!, col])))
    end
end

# =========================
# 1) Φόρτωση
# =========================
df = read_runs_table(input_file)
rename_columns_to_standard!(df)

required_numeric = [
    :InitialPathfindingTime_s,
    :IncrementalPathfindingTime_s,
    :SimulationTime_s,
    :SumTL_capped,
]
ensure_numeric!(df, required_numeric)
sanitize_k!(df, :N_active_goals)

# =========================
# 2) Νέα παράγωγη στήλη
# =========================
df.TotalPathfinding_s = df.InitialPathfindingTime_s .+ df.IncrementalPathfindingTime_s

# =========================
# 3) Group-by summary
# =========================
gdf = groupby(df, :N_active_goals)

summary_df = combine(
    gdf,
    nrow => :n_runs,
    :SumTL_capped => mean => :Mean_SumTL_capped,
    :SumTL_capped => median => :Median_SumTL_capped,
    :TotalPathfinding_s => mean => :Mean_TotalPathfinding_s,
    :TotalPathfinding_s => median => :Median_TotalPathfinding_s,
    :SimulationTime_s => mean => :Mean_SimulationTime_s,
)

sort!(summary_df, :N_active_goals)

println("=== Summary by N_active_goals ===")
show(summary_df, allrows = true, allcols = true)
println()

# save summary table
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
summary_csv = joinpath(out_dir, "Scenario4_GroupBy_N_active_goals_$timestamp.csv")
CSV.write(summary_csv, summary_df)
println("Saved summary table to: $summary_csv")

# =========================
# 4) Boxplots
# =========================
# 4a) SumTL_capped vs N_active_goals
fig1 = Figure(size = (1000, 600))
ax1 = Axis(
    fig1[1, 1],
    title = "SumTL_capped vs N_active_goals",
    xlabel = "N_active_goals",
    ylabel = "SumTL_capped",
)
boxplot!(ax1, df.N_active_goals, df.SumTL_capped; color = :steelblue, whiskerwidth = 0.5)
scatter!(ax1, df.N_active_goals, df.SumTL_capped; color = (:black, 0.18), markersize = 5)
sumtl_box_png = joinpath(out_dir, "Scenario4_Boxplot_SumTL_vs_Nactive_$timestamp.png")
save(sumtl_box_png, fig1)
println("Saved: $sumtl_box_png")

# 4b) TotalPathfinding_s vs N_active_goals
fig2 = Figure(size = (1000, 600))
ax2 = Axis(
    fig2[1, 1],
    title = "TotalPathfinding_s vs N_active_goals",
    xlabel = "N_active_goals",
    ylabel = "TotalPathfinding_s",
)
boxplot!(ax2, df.N_active_goals, df.TotalPathfinding_s; color = :darkorange, whiskerwidth = 0.5)
scatter!(ax2, df.N_active_goals, df.TotalPathfinding_s; color = (:black, 0.18), markersize = 5)
pf_box_png = joinpath(out_dir, "Scenario4_Boxplot_TotalPF_vs_Nactive_$timestamp.png")
save(pf_box_png, fig2)
println("Saved: $pf_box_png")

# =========================
# 5) Προαιρετικά violin plots
# =========================
if make_violin
    fig3 = Figure(size = (1000, 600))
    ax3 = Axis(
        fig3[1, 1],
        title = "Violin: SumTL_capped vs N_active_goals",
        xlabel = "N_active_goals",
        ylabel = "SumTL_capped",
    )
    violin!(ax3, df.N_active_goals, df.SumTL_capped; color = (:mediumpurple, 0.6), strokecolor = :black)
    sumtl_violin_png = joinpath(out_dir, "Scenario4_Violin_SumTL_vs_Nactive_$timestamp.png")
    save(sumtl_violin_png, fig3)
    println("Saved: $sumtl_violin_png")

    fig4 = Figure(size = (1000, 600))
    ax4 = Axis(
        fig4[1, 1],
        title = "Violin: TotalPathfinding_s vs N_active_goals",
        xlabel = "N_active_goals",
        ylabel = "TotalPathfinding_s",
    )
    violin!(ax4, df.N_active_goals, df.TotalPathfinding_s; color = (:seagreen, 0.6), strokecolor = :black)
    pf_violin_png = joinpath(out_dir, "Scenario4_Violin_TotalPF_vs_Nactive_$timestamp.png")
    save(pf_violin_png, fig4)
    println("Saved: $pf_violin_png")
end

println("\nDone.")
