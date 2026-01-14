"""
Benchmarking and Profiling Script for Pathfinding in Scenario 2

This script benchmarks and profiles the pathfinding calculation performance.
It can be run independently to measure pathfinding performance without running
the full simulation.

Usage:
    julia benchmark_pathfinding.jl

Results are saved to:
    - SCENARIO 2/Simulation Results/SCENARIO_2_<n_agents>_<PM|APM>_<seed>.json
    - SCENARIO 2/Simulation Results/SCENARIO_2_<n_agents>_<PM|APM>_<seed>_profile.txt
"""

using BenchmarkTools
using Profile
using Dates

# Try to use JSON if available, otherwise use a simpler format
HAS_JSON = false
try
    using JSON
    global HAS_JSON = true
catch
    global HAS_JSON = false
    println("Warning: JSON package not found. Results will be saved in a simpler format.")
end

# Set flag to skip video creation (only run simulation, no visualization)
global SKIP_VIDEO = true

# Load the scenario setup
# This will execute Scenario 2.jl, but video creation will be skipped
# The simulation will still run to collect data
include("Scenario 2.jl")

# ============================================================================
# CONFIGURATION
# ============================================================================

const BENCHMARK_SAMPLE_SIZE = 50  # Number of agents to benchmark
const PROFILE_ITERATIONS = 20    # Number of path calculations to profile
const BENCHMARK_SAMPLES = 5      # Number of samples per benchmark (for statistical accuracy)
const BENCHMARK_EVALS = 1        # Number of evaluations per sample

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# Note: We use the agents already created in Scenario 2.jl
# No need to create new test agents - we'll use the existing ones from the model

"""
Extract statistics from a benchmark result
"""
function extract_benchmark_stats(b)
    return Dict(
        "time_median" => median(b.times) / 1e9,  # Convert nanoseconds to seconds
        "time_mean" => mean(b.times) / 1e9,
        "time_min" => minimum(b.times) / 1e9,
        "time_max" => maximum(b.times) / 1e9,
        "time_std" => std(b.times) / 1e9,
        "memory_allocs" => b.allocs,
        "memory_allocated" => b.memory,
        "gctime" => b.gctime / 1e9,
        "samples" => length(b.times)
    )
end

# ============================================================================
# BENCHMARKING
# ============================================================================

println("\n" * "="^80)
println("PATHFINDING BENCHMARKING")
println("="^80)

# Use the agents already created in Scenario 2.jl
all_agents_list = collect(allagents(model))
actual_n_agents = length(all_agents_list)

if actual_n_agents == 0
    error("No agents found in model. Make sure Scenario 2.jl creates agents before benchmarking.")
end

# Determine sample size (use min of requested sample size and actual number of agents)
sample_size = min(BENCHMARK_SAMPLE_SIZE, actual_n_agents)
println("Using $sample_size agents from the $actual_n_agents agents created in Scenario 2.jl")
println("Benchmarking path calculation...")
println("This may take a few moments...\n")

# Sample agents for benchmarking (use first N agents to ensure consistency)
test_agents = all_agents_list[1:sample_size]

# Benchmark individual path calculations
# Note: These agents already have paths calculated from Scenario 2.jl
# We'll benchmark recalculating their paths
println("\nRunning benchmarks (this may take a while)...")
benchmark_results = []
individual_stats = []

for (i, person) in enumerate(test_agents)
    # Benchmark this agent's path calculation
    # Note: plan_best_route! will recalculate the path for this agent
    b = @benchmark plan_best_route!($person, $dests, $(model.pathfinderPM)) samples=BENCHMARK_SAMPLES evals=BENCHMARK_EVALS
    push!(benchmark_results, b)
    
    # Extract statistics
    stats = extract_benchmark_stats(b)
    push!(individual_stats, stats)
    
    if i % 10 == 0
        println("  Benchmarked $i/$sample_size agents (median: $(round(stats["time_median"], digits=6))s)")
    end
end

# Aggregate statistics
all_median_times = [s["time_median"] for s in individual_stats]
all_mean_times = [s["time_mean"] for s in individual_stats]
all_min_times = [s["time_min"] for s in individual_stats]
all_max_times = [s["time_max"] for s in individual_stats]

# Calculate summary statistics
summary_stats = Dict(
    "sample_size" => sample_size,
    "total_agents_in_model" => actual_n_agents,
    "median_time_per_agent" => mean(all_median_times),
    "mean_time_per_agent" => mean(all_mean_times),
    "min_time_per_agent" => minimum(all_min_times),
    "max_time_per_agent" => maximum(all_max_times),
    "std_deviation" => std(all_median_times),
    "total_memory_allocs" => sum([s["memory_allocs"] for s in individual_stats]),
    "total_memory_allocated" => sum([s["memory_allocated"] for s in individual_stats]),
    "estimated_total_time_100_agents" => mean(all_median_times) * 100,
    "estimated_total_time_$(n_agents)_agents" => mean(all_median_times) * n_agents,
    "timestamp" => string(now()),
    "scenario" => "Scenario 2",
    "pathfinding_algorithm" => "AStar",
    "map_size" => size(NPM),
    "n_goals" => length(dests)
)

# Print summary
println("\n" * "-"^80)
println("BENCHMARK SUMMARY")
println("-"^80)
println("Sample size: $(summary_stats["sample_size"]) agents")
println("Median time per agent: $(round(summary_stats["median_time_per_agent"], digits=6)) seconds")
println("Mean time per agent: $(round(summary_stats["mean_time_per_agent"], digits=6)) seconds")
println("Min time: $(round(summary_stats["min_time_per_agent"], digits=6)) seconds")
println("Max time: $(round(summary_stats["max_time_per_agent"], digits=6)) seconds")
println("Std deviation: $(round(summary_stats["std_deviation"], digits=6)) seconds")
println("\nEstimated total time for 100 agents: $(round(summary_stats["estimated_total_time_100_agents"], digits=4)) seconds")
println("Estimated total time for $n_agents agents: $(round(summary_stats["estimated_total_time_$(n_agents)_agents"], digits=4)) seconds")
println("="^80)

# ============================================================================
# PROFILING
# ============================================================================

println("\n" * "="^80)
println("PATHFINDING PROFILING")
println("="^80)
# Determine profile iterations (use min of requested iterations and actual number of agents)
profile_iterations = min(PROFILE_ITERATIONS, actual_n_agents)
println("Profiling path calculation for $profile_iterations iterations...")
println("Using agents from Scenario 2.jl")
println("This will show where time is spent in the pathfinding code.\n")

# Use agents from Scenario 2.jl for profiling
profile_agents = all_agents_list[1:profile_iterations]

# Run profiling
Profile.clear()
@profile begin
    for person in profile_agents
        plan_best_route!(person, dests, model.pathfinderPM)
    end
end

# Collect profile data
println("Profile results:")
println("-"^80)
Profile.print()

# ============================================================================
# SAVE RESULTS
# ============================================================================

# Create results directory if it doesn't exist
results_dir = "SCENARIO 2/Simulation Results"
if !isdir(results_dir)
    mkpath(results_dir)
end

# Generate filename with format: SCENARIO_2_(n_agents)_(PM or APM)_(seed)
# Note: seed, n_agents, and pathfinderPM are defined in Scenario 2.jl which is included above
metric_type = model.pathfinderPM.cost_metric isa Agents.Pathfinding.PenaltyMap ? "PM" : 
               model.pathfinderPM.cost_metric isa Agents.Pathfinding.AbsolutePenaltyMap ? "APM" : "PM"
benchmark_filename = "SCENARIO_2_$(n_agents)_$(metric_type)_$(seed)"

# Save benchmark results
benchmark_file = joinpath(results_dir, "$benchmark_filename.json")
benchmark_data = Dict(
    "summary" => summary_stats,
    "individual_results" => individual_stats
)

if HAS_JSON
    open(benchmark_file, "w") do f
        JSON.print(f, benchmark_data, 4)
    end
else
    # Fallback: save as a readable text format
    benchmark_file = joinpath(results_dir, "$benchmark_filename.txt")
    open(benchmark_file, "w") do f
        println(f, "Pathfinding Benchmark Results")
        println(f, "Generated: $(now())")
        println(f, "="^80)
        println(f)
        println(f, "SUMMARY:")
        for (key, val) in summary_stats
            println(f, "  $key: $val")
        end
        println(f)
        println(f, "INDIVIDUAL RESULTS:")
        for (i, stats) in enumerate(individual_stats)
            println(f, "  Agent $i:")
            for (key, val) in stats
                println(f, "    $key: $val")
            end
        end
    end
end

println("\n" * "="^80)
println("RESULTS SAVED")
println("="^80)
println("Benchmark results saved to: $benchmark_file")

# Save profile results
profile_file = joinpath(results_dir, "$benchmark_filename_profile.txt")
open(profile_file, "w") do f
    println(f, "Pathfinding Profile Results")
    println(f, "Generated: $(now())")
    println(f, "Iterations: $profile_iterations")
    println(f, "="^80)
    println(f)
    Profile.print(f)
end

println("Profile results saved to: $profile_file")
println("="^80 * "\n")

# ============================================================================
# ADDITIONAL ANALYSIS
# ============================================================================

println("ADDITIONAL ANALYSIS")
println("-"^80)

# Time distribution analysis
sorted_times = sort(all_median_times)
percentiles = Dict(
    "p25" => sorted_times[Int(ceil(0.25 * length(sorted_times)))],
    "p50" => sorted_times[Int(ceil(0.50 * length(sorted_times)))],
    "p75" => sorted_times[Int(ceil(0.75 * length(sorted_times)))],
    "p90" => sorted_times[Int(ceil(0.90 * length(sorted_times)))],
    "p95" => sorted_times[Int(ceil(0.95 * length(sorted_times)))]
)

println("Time percentiles:")
for (p, val) in percentiles
    println("  $p: $(round(val, digits=6)) seconds")
end

println("\nBenchmarking complete! Check the saved files for detailed results.")
