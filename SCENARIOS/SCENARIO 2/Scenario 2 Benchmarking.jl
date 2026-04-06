begin   # Φόρτωση των απαραίτητων βιβλιοθηκών
    using Agents
    using Agents.Pathfinding
    using Random
    using ImageMagick
    using FileIO: load
    using Images
    using XLSX
end



@agent struct AgentEscapes(ContinuousAgent{2, Float64}) # Αρχικοποίηση των Agents
    age::Float64
    mass::Float64
    toxicload::Float64
    pathX::Vector{Float64}
    pathY::Vector{Float64}
    TL1::Vector{Float64}
    TL2::Vector{Float64}
    TL3::Vector{Float64}
end

    
begin   # Φόρτωση του heightmap και των hand-drawn penalty maps

    # heightmap
    heightmap_data = load("NADEEN/Maps/Qatargas Map.jpg")
    heightmap_data = permutedims(channelview(heightmap_data), [2,3,1])[:,:,1]
    global heightmap = floor.(Int, convert.(Float64, heightmap_data) * 255)
    heightmap = 255 .- heightmap   # αυτό κάνει την αντιστροφή

    # Φόρτωση penalty maps
    penalty_map = load("Concentration Maps/6.bmp")
    penalty_map = permutedims(channelview(penalty_map), [2,3,1])[:,:,1]
    global penalty_map = floor.(Int, convert.(Float64, penalty_map) * 500)
    
    # Check dimension consistency
    @assert size(penalty_map) == size(heightmap) "penalty_map dimensions $(size(penalty_map)) do not match heightmap dimensions $(size(heightmap))"
end

NPM = heightmap + penalty_map # Merging the two maps to create a new penalty map

begin   # Αρχικοποίηση των παραμέτρων του μοντέλου
    const METERS_TO_PIXELS = 723.37 / 2500.0
    dt = 1.   ## discrete timestep each iteration of the model          # Define the dt variable as 1
    n_agents = 50                                                         # Define the n_agents variable as 3
    toxicity_rate = 0.07                                               # Define the toxicity_rate variable as 0.07
    age_range = (22,60)                                                 # Define the age_range variable as a tuple of 22 and 60
    speed_range = (4.0,7.0)                                            # Define the speed_range variable as a tuple of 4.0 and 7.0
    mass_range = (50,80)                                                # Define the mass_range variable as a tuple of 50 and 80
    ag_range_y = (size(heightmap)[1]/4):(3*size(heightmap)[1]/4)    # Define the ag_range_y variable as a larger range of values from the heightmap array # [1] stands for the 1st row
    ag_range_x = (size(heightmap)[2]/4):(3*size(heightmap)[2]/4)    # Define the ag_range_x variable as a range of values from the heightmap array # [2] stands for the 2nd row
    dims = (size(NPM))                                            # Define the dims variable as the dimensions of the heightmap array (2xn matrix)
    walkmap = BitArray(trues(dims...))                                 # Define the walkmap variable as a BitArray of true values with the dimensions of the heightmap array
    
    # Set walkmap to false in white areas of the heightmap (after reversal, white = high values)
    # White areas are where heightmap value is above threshold (e.g., > 245)
    # Black and grey areas (low to medium values) remain walkable
    white_threshold = 245
    walkmap[heightmap .> white_threshold] .= false
end    


    #goals
    dests = [(600., 980.), (100., 200.)]

    ## Note that the space of the space do not have to correspond to the dimensions
    ## of the pathfinder. Discretisation is handled by the pathfinding methods
    space = ContinuousSpace(size(NPM); periodic = false, spacing = 1)


begin
    pathfinderPM = AStar(space; walkmap = walkmap, cost_metric = PenaltyMap(NPM, MaxDistance{2}()))
    properties = (
        pathfinderPM = pathfinderPM,
        heightmap = heightmap,
        dt = dt,
        speed_range = speed_range,
        goal = dests
    )
end



#
# Helper: true if position is within radius of any goal (TL and movement stop when true)
const goal_radius = 10.0
function at_goal(pos, dests, radius = goal_radius)
    px, py = Float64(pos[1]), Float64(pos[2])
    return minimum(hypot(px - d[1], py - d[2]) for d in dests) <= radius
end


function agent_step!(person, model)
    if at_goal(person.pos, model.goal)
        push!(person.pathX, person.pos[1])
        push!(person.pathY, person.pos[2])
        return
    end

    position = floor.(Int, person.pos)
   # Ct παίρνεται τώρα από το global_penalty_map (hand-drawn maps)
    Ct = penalty_map[position[1], position[2]]
    TLcurrent = [person.TL1[end], person.TL2[end], person.TL3[end]]
    TL = update_toxic_load(Ct, TLcurrent, model.dt)

    person.toxicload = sum(TL)
    push!(person.TL1, TL[1])
    push!(person.TL2, TL[2])
    push!(person.TL3, TL[3])

    # --- Speed update based on toxicload ---
    speed = 1.35
    if 0 < person.toxicload <= 1
        speed = 1.35 * exp(0.393 * person.toxicload)
    elseif 1 < person.toxicload < 3
        speed = -1.78 * log(person.toxicload) + 2.063
    elseif person.toxicload >= 3
        speed = 0.0
    end

    #display("Speed: $speed  -  ToxicLoad: $(person.toxicload)")

    move_along_route!(person, model, model.pathfinderPM, speed * METERS_TO_PIXELS, model.dt)
    push!(person.pathX, person.pos[1])
    push!(person.pathY, person.pos[2])
end



function model_step!(model)
    for (a1, a2) in interacting_pairs(model, 0.012, :nearest)
        elastic_collision!(a1, a2, :mass)
    end
end


# --- Benchmark parameters ---
const BENCHMARK_MAX_RUNS = 100
const BENCHMARK_CONVERGENCE_PCT = 0.01   # stop when running avg pathfinding time changes by < 1%
const BENCHMARK_MIN_RUNS = 50            # run at least this many times before checking convergence
const BENCHMARK_T_STEPS = 1852

"""
    run_one_benchmark() -> (pathfinding_time_seconds, simulation_time_seconds, total_tl, seed)
Build a fresh model with a random seed, add agents (timing only plan_best_route!), run T steps (timing simulation). No video/CSV.
total_tl = sum of all agents' toxicload at end of simulation.
"""
function run_one_benchmark()
    seed = rand(Random.RandomDevice(), UInt32)
    rng_run = MersenneTwister(seed)
    model = StandardABM(
        AgentEscapes,
        space;
        rng          = rng_run,
        properties   = properties,
        agent_step!  = agent_step!,
        model_step!  = model_step!,
    )

    # Add agents and time only plan_best_route! for each
    pathfinding_time = 0.0
    for _ in 1:n_agents
        age = rand(abmrng(model)) * (age_range[2] - age_range[1]) + age_range[1]
        mass = rand(abmrng(model)) * (mass_range[2] - mass_range[1]) + mass_range[1]
        vel = Tuple(rand(abmrng(model), 2) .* (speed_range[2]-speed_range[1]) .+ speed_range[1])

        max_attempts = 1000
        attempts = 0
        pos = nothing
        while attempts < max_attempts
            candidate_pos = Tuple((rand(abmrng(model), floor.(ag_range_y)), rand(abmrng(model), floor.(ag_range_x))))
            pos_int = floor.(Int, candidate_pos)
            if 1 <= pos_int[1] <= size(walkmap, 1) && 1 <= pos_int[2] <= size(walkmap, 2) && walkmap[pos_int[1], pos_int[2]]
                pos = candidate_pos
                break
            end
            attempts += 1
        end

        if pos === nothing
            println("Warning: Could not find valid spawn position for agent after $max_attempts attempts (seed=$seed)")
            continue
        end

        person = add_agent!(pos, AgentEscapes, model, vel, age, mass, 1., [pos[1]], [pos[2]], [0.0], [0.0], [0.0])
        pathfinding_time += @elapsed plan_best_route!(person, dests, model.pathfinderPM)
    end

    # Time simulation: T steps, no video
    simulation_time = @elapsed for _ in 1:BENCHMARK_T_STEPS
        step!(model, 1)  # StandardABM already knows agent_step! and model_step!
    end

    total_tl = sum(a.toxicload for a in allagents(model))
    return pathfinding_time, simulation_time, total_tl, seed
end


function setupToxic()                                               # Define the setupToxic function
    Atime = [0.0, 2.0, 5.0, 8.0, 15.0, 30.0] #min                # Define the Atime array as a 1x6 matrix                                    # Initialization of the 5 standard AEGL exposure times
    Arho = zeros(3, 6)                                              # Define the Arho array as a 3x6 matrix                                     # the concentration of the three symptoms compared to the AEGL concentrations
    Arho[1, 2:6] = [2.87, 2.33, 2.09, 1.81, 1.55]                   # Define the Arho array for the first row and columns 2 to 6                # odor
    Arho[2, 2:6] = [202.38, 164.38, 147.74, 128.10, 109.45]         # Define the Arho array for the second row and columns 2 to 6               # irritation
    Arho[3, 2:6] = [286.71, 232.87, 209.30, 181.47, 155.05] #ppm    # Define the Arho array for the third row and columns 2 to 6                # edema
    MW = 34 #Molecular weight of H2S in g/mol
    Arho *= MW/24.04 #mg/m^3                                        # Multiply the Arho array by the molecular weight of H2S divided by 24.04
    Arho = Arho'                                                    # Transpose the Arho array 
    Atime = Atime*60 #seconds                                       # Multiply the Atime array by 60 seconds to convert to seconds              
    taumin = 200.                                                   # Define the taumin variable as 200 seconds (Borris&Patnaik, 2014)          # shortest exposure time over which an AEGL 1, 2 or 3 onset can be reached
    taumax = 86400.                                                 # Define the taumax variable as 86400 seconds (Borris&Patnaik, 2014)        # longest exposure time over which an AEGL 1, 2 or 3 onset can be reached
    Brho = zeros(7,3)                                               # Define the Brho array as a 7x3 matrix of zeros                            # in ppm for each AEGL band at every time step 'Atime'
    Balpha = zeros(7,3)                                             # Define the Balpha array as a 7x3 matrix of zeros                          # power-law exponents
    rhomax = zeros(1,3)                                             # Define the rhomax array as a 1x3 matrix of zeros                          # maximum concentration of H2S exposed by each agent
    rhomin = zeros(1,3)                                             # Define the rhomin array as a 1x3 matrix of zeros                          # minimum concentration of H2S exposed by each agent
    Btime = zeros(7, 3)                                             # Define the Btime array as a 7x3 matrix of zeros                           # represents an array, which is function of 'taumin' and 'taumax', that changes depending on alpha, which is a corresponsing array of power low exponents interpolating the 'Brho' table array

    #Initialize
    for k=1:3
        for b=2:6
            Brho[b, k] = Arho[b, k]
            Btime[b, k] = Atime[b]
        end
        Btime[1, k] = taumin;
         Btime[7, k] = taumax;
    end    
    
    #power low trial values
    for k=1:3
        for b=3:6
            if Brho[b-1, k]==Brho[b, k]
                Balpha[b, k] = 0.0;
            else
                Balpha[b, k] = log(Atime[b]/Atime[b-1])/log(Brho[b-1, k]/Brho[b, k]);
            end
        end

        Balpha[2, k] = Balpha[3, k]
        Balpha[1, k] = Balpha[2, k]
        Balpha[7, k] = Balpha[6, k]
        #Balpha[6, k] = Balpha[5, k]
    end
    #extrapolate for the edges
    for k=1:3
        if Balpha[3, k]==0
            rhomax[k] = Brho[2, k];
            Brho[1, k] = rhomax[k];
        else
            rhomax[k] = Brho[2, k]*(Btime[2, k]/taumin)^(1/Balpha[2, k]);
            Brho[1, k] = rhomax[k];
        end

        if Brho[5, k]==Brho[6, k]
            rhomin[k] = Brho[6, k];
            Brho[7, k] = rhomin[k];
        else
            rhomin[k] = Brho[6, k]*(Btime[6, k]/taumax)^(1/Balpha[6, k]); #I fixed Brho[6] to be Brho[6, k]
            Brho[7, k] = rhomin[k];
        end

    end
    #Correct to account for threshold values
    for k=1:3
        for b=2:7
            if Balpha[b, k] == 0
                Btime[b, k] = Btime[b-1, k];
            end
        end
    end

    for k=1:3
        for b=3:5
            if Balpha[b-1, k]==0 && Balpha[b, k]>0
                Balpha[b, k]=log(Btime[b, k]/Btime[b-1, k])/log(Brho[b-1, k]/Brho[b, k]);
            end
        end
    end

    return Balpha, Btime, Brho'
end

Balpha, Btime, Brho = setupToxic()



function update_toxic_load(Ct, TLcurrent, dt)
   
    #a = alpha
    #Btime = Btime;
    #Brho = Brho;
    TL = TLcurrent
    TL_rate = 0.0

    for k = 1:3
        Cmin = Brho[k, 7]
        Cmax = Brho[k, 1]
        if Ct > Cmax
            TL_rate = 1 / Btime[1];
        elseif Ct < Cmin
            TL_rate = 0.0;
        else
            for i in 2:size(Btime, 1)
                if i <= size(Brho, 2)
                    if Brho[k, i-1] < Ct && Ct < Brho[k, i] #check 
                        TL_rate = (1/Btime[i, k])*((Ct/Brho[k, i])^(Balpha[i, k])) #Toxic Load rate in units of s^-1 
                    end
                end
            end
        end
        TL[k] = TL[k] .+ TL_rate * dt;
        
    end
        #TL[iAg, :] .= sum(TL .> 1) .+ TL[min(3, sum(TL .> 1) + 1)] .* (1 - (TL[3] > 1));
        #TL[iAg, :] .= person.toxicload;
    return TL
end



function static_preplot!(ax, abmplot)
    # 1) Ξεπακετάρουμε το Observable
    model = isa(abmplot, Observable)  ? abmplot[] :
            hasproperty(abmplot, :model) ? abmplot.model[] :
            abmplot

    # 2) Σχεδιάζουμε τα goals
    dests = model.goal
    xs_g = getindex.(dests, 1)
    ys_g = getindex.(dests, 2)
    scatter!(ax, xs_g, ys_g; color = (:red, 50), marker = '●')

    # 3) Σχεδιάζουμε για κάθε agent τη διαδρομή που έχει ήδη κάνει
    for agent in allagents(model)
        xs = agent.pathX
        ys = agent.pathY
        # π.χ. χρώμα ίδια με τον agent, πάχος γραμμής 2
        lines!(ax, xs, ys; linewidth = 2, color = personcolor(agent))
    end
end


function personcolor(person::AgentEscapes)  # Χρώμα του agent ανάλογα με το toxicload
    if person.toxicload >= 3.
        return :red
    elseif person.toxicload <= 1.
        return :green
    else
        return :orange
    end
end


# --- Benchmark loop: run at least BENCHMARK_MIN_RUNS, then up to BENCHMARK_MAX_RUNS or until pathfinding avg converges (< 1% change) ---
pathfinding_times = Float64[]
simulation_times  = Float64[]
total_tls = Float64[]
seeds = UInt32[]
avg_pathfinding_prev = 0.0

println("Benchmark: Scenario 2 (pathfinding + simulation). Max runs = $BENCHMARK_MAX_RUNS, convergence = $(BENCHMARK_CONVERGENCE_PCT*100)%.")
for run_id in 1:BENCHMARK_MAX_RUNS
    t_path, t_sim, total_tl, run_seed = run_one_benchmark()
    push!(pathfinding_times, t_path)
    push!(simulation_times, t_sim)
    push!(total_tls, total_tl)
    push!(seeds, run_seed)

    n = length(pathfinding_times)
    avg_pathfinding = sum(pathfinding_times) / n
    avg_simulation  = sum(simulation_times) / n

    # Converged: after minimum runs, running average pathfinding time changed by less than 1%
    converged = n >= BENCHMARK_MIN_RUNS && avg_pathfinding_prev > 0 && (abs(avg_pathfinding - avg_pathfinding_prev) / avg_pathfinding_prev <= BENCHMARK_CONVERGENCE_PCT)

    println("  Run $run_id (seed=$run_seed): pathfinding = $(round(t_path; digits=4)) s, simulation = $(round(t_sim; digits=4)) s  (avg pathfinding = $(round(avg_pathfinding; digits=4)) s)")
    global avg_pathfinding_prev = avg_pathfinding

    if converged
        println("Converged at run $run_id (pathfinding avg change < $(BENCHMARK_CONVERGENCE_PCT*100)%).")
        break
    end
end

n_runs = length(pathfinding_times)
avg_pathfinding_final = sum(pathfinding_times) / n_runs
avg_simulation_final  = sum(simulation_times) / n_runs
avg_total_tl_final    = sum(total_tls) / n_runs

# --- Write Excel: per-run times + summary averages ---
xlsx_path = "SCENARIOS/SCENARIO 2/Simulation Results/Scenario_2_Benchmarking_$(n_agents).xlsx"
run_ids = collect(1:n_runs)
columns_data = [run_ids, seeds, pathfinding_times, simulation_times, total_tls]
column_names = ["Run", "Seed", "PathfindingTime_s", "SimulationTime_s", "Total TL"]
XLSX.writetable(xlsx_path, columns_data, column_names; sheetname = "Runs", overwrite = true)
# Append summary to same sheet (below data)
XLSX.openxlsx(xlsx_path, mode = "rw") do xf
    sh = xf["Runs"]
    sh[n_runs + 2, 1] = "Number of runs"
    sh[n_runs + 2, 2] = n_runs
    sh[n_runs + 3, 1] = "Average pathfinding time (s)"
    sh[n_runs + 3, 2] = avg_pathfinding_final
    sh[n_runs + 4, 1] = "Average simulation time (s)"
    sh[n_runs + 4, 2] = avg_simulation_final
    sh[n_runs + 5, 1] = "Average Total TL (sum toxic load accumulated)"
    sh[n_runs + 5, 2] = avg_total_tl_final
end

println("Benchmark complete. Results written to $xlsx_path")
println("  Total runs: $n_runs | Avg pathfinding: $(round(avg_pathfinding_final; digits=4)) s | Avg simulation: $(round(avg_simulation_final; digits=4)) s | Avg Total TL: $(round(avg_total_tl_final; digits=4))")
