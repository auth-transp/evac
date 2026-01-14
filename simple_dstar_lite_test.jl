"""
Complex D* Lite Test Case with Dynamic Penalty Map and Multiple Agents
This script demonstrates the D* Lite algorithm with:
- Dynamic penalty map that changes during execution
- Multiple agents (3 agents)
- Varying penalty costs across the grid
- Step-by-step logging showing how agents adapt to cost changes
"""

# Load the local Agents package (not from Pkg)
# Go up one directory level since this file is in DStarLite subdirectory
agents_path = joinpath(dirname(@__DIR__), "Agents", "src", "Agents.jl")
include(agents_path)
using .Agents
using .Agents.Pathfinding
using DataStructures

# Enable detailed logging
const VERBOSE = true

# Open output file for writing
const OUTPUT_FILE = open("dstar_lite_test_output.txt", "w")

# Function to print to both console and file (with newline)
function print_both(str = "")
    println(str)
    println(OUTPUT_FILE, str)
    flush(OUTPUT_FILE)  # Ensure it's written immediately
end

# Function to print to both console and file (without newline)
function print_both_no_nl(str = "")
    print(str)
    print(OUTPUT_FILE, str)
    flush(OUTPUT_FILE)
end

function log_step(step_num, message)
    if VERBOSE
        print_both("=" ^ 60)
        print_both("STEP $step_num: $message")
        print_both("=" ^ 60)
    end
end

function log_action(action, details = "")
    if VERBOSE
        print_both("  → $action")
        if details != ""
            print_both("    $details")
        end
    end
end

function log_state(state_name, value)
    if VERBOSE
        print_both("  [$state_name] = $value")
    end
end

# ============================================================================
# SETUP: Create a 7x7 grid with obstacles and varying penalty costs
# ============================================================================
log_step(1, "Setting up the grid environment with penalty map")

# Create a 7x7 grid (larger for more interesting paths)
grid_size = (7, 7)
log_action("Creating grid", "Size: $grid_size")

# Create walkmap - all cells walkable except some obstacles
walkmap = trues(grid_size)
walkmap[3, 3] = false  # Obstacle 1
walkmap[4, 4] = false  # Obstacle 2
walkmap[5, 2] = false  # Obstacle 3
log_action("Creating walkmap", "3 obstacles placed")

# Create initial penalty map with varying costs
# Lower values = easier to traverse, higher values = more costly
penalty_map = zeros(Int, grid_size)
# Set some areas with high penalty (costly to traverse)
penalty_map[2, 2:3] .= 50  # High penalty area 1
penalty_map[3, 5:6] .= 50  # High penalty area 2
penalty_map[6, 3:5] .= 50  # High penalty area 3
# Set some areas with medium penalty
penalty_map[1, 4:5] .= 20  # Medium penalty area
penalty_map[5, 5:6] .= 20  # Medium penalty area
# Most areas have 0 penalty (normal traversal cost)

log_action("Creating penalty map", "High penalty areas: 50, Medium: 20, Normal: 0")

# Visualize the grid with penalties
print_both("\nGrid Layout (X = obstacle, numbers = penalty cost):")
print_both("  " * join([string(i) for i in 1:grid_size[1]], "  "))
for j in 1:grid_size[2]
    print_both_no_nl("$j ")
    for i in 1:grid_size[1]
        if !walkmap[i, j]
            print_both_no_nl(" X ")
        else
            penalty = penalty_map[i, j]
            if penalty >= 50
                print_both_no_nl("H$penalty")
            elseif penalty >= 20
                print_both_no_nl("M$penalty")
            else
                print_both_no_nl(" . ")
            end
        end
    end
    print_both()
end
print_both("  Legend: X=obstacle, H50=high penalty, M20=medium penalty, .=normal")
print_both()

# ============================================================================
# STEP 2: Create D* Lite pathfinder with penalty map
# ============================================================================
log_step(2, "Creating D* Lite pathfinder with PenaltyMap cost metric")

log_action("Calling DStarLite constructor", "dims=$grid_size, with PenaltyMap")
pathfinder = DStarLite(
    grid_size;
    walkmap = walkmap,
    diagonal_movement = true,
    cost_metric = PenaltyMap(penalty_map, DirectDistance{2}())
)
log_state("Pathfinder created", typeof(pathfinder))
log_state("Cost metric", typeof(pathfinder.cost_metric))

# ============================================================================
# STEP 3: Define start and goal positions for 3 agents
# ============================================================================
log_step(3, "Defining start and goal positions for 3 agents")

# Agent 1: Top-left to bottom-right
agent1_start = (1, 1)
agent1_goal = (7, 7)

# Agent 2: Top-right to bottom-left
agent2_start = (7, 1)
agent2_goal = (1, 7)

# Agent 3: Middle-left to middle-right
agent3_start = (1, 4)
agent3_goal = (7, 4)

print_both("\nAgent Positions:")
print_both("  Agent 1: Start $(agent1_start) → Goal $(agent1_goal)")
print_both("  Agent 2: Start $(agent2_start) → Goal $(agent2_goal)")
print_both("  Agent 3: Start $(agent3_start) → Goal $(agent3_goal)")

# ============================================================================
# STEP 4: Create agents and model
# ============================================================================
log_step(4, "Creating agents and model")

space = GridSpace(grid_size; periodic = false)

@agent struct SimpleAgent(GridAgent{2})
end

model = ABM(SimpleAgent, space)

# Create 3 agents
agent1 = add_agent!(agent1_start, model)
agent2 = add_agent!(agent2_start, model)
agent3 = add_agent!(agent3_start, model)

log_action("Agents created", "Agent 1 (ID: $(agent1.id)), Agent 2 (ID: $(agent2.id)), Agent 3 (ID: $(agent3.id))")

# ============================================================================
# STEP 5: Plan initial routes for all agents
# ============================================================================
log_step(5, "Planning initial routes for all agents using D* Lite")

print_both("\n" * "─" ^ 60)
print_both("PLANNING ROUTES FOR ALL AGENTS")
print_both("─" ^ 60)

# Plan routes for all agents
log_action("Planning route for Agent 1", "From $(agent1.pos) to $agent1_goal")
Pathfinding.plan_route!(agent1, agent1_goal, pathfinder)
if haskey(pathfinder.agent_paths, agent1.id)
    log_state("Agent 1 path length", length(pathfinder.agent_paths[agent1.id]))
end

log_action("Planning route for Agent 2", "From $(agent2.pos) to $agent2_goal")
Pathfinding.plan_route!(agent2, agent2_goal, pathfinder)
if haskey(pathfinder.agent_paths, agent2.id)
    log_state("Agent 2 path length", length(pathfinder.agent_paths[agent2.id]))
end

log_action("Planning route for Agent 3", "From $(agent3.pos) to $agent3_goal")
Pathfinding.plan_route!(agent3, agent3_goal, pathfinder)
if haskey(pathfinder.agent_paths, agent3.id)
    log_state("Agent 3 path length", length(pathfinder.agent_paths[agent3.id]))
end

# ============================================================================
# STEP 6: Simulate movement with dynamic penalty map changes
# ============================================================================
log_step(6, "Simulating agent movement with dynamic penalty map updates")

print_both("\n" * "=" ^ 60)
print_both("SIMULATION: Agents moving with dynamic cost changes")
print_both("=" ^ 60)

# Function to update penalty map and show the change
function update_penalty_map!(pmap, changes::Vector{Tuple{Tuple{Int,Int}, Int}})
    print_both("\n" * "─" ^ 60)
    print_both("PENALTY MAP UPDATE:")
    print_both("─" ^ 60)
    for ((i, j), new_penalty) in changes
        old_penalty = pmap[i, j]
        pmap[i, j] = new_penalty
        print_both("  Cell ($i, $j): $old_penalty → $new_penalty")
    end
    print_both("─" ^ 60)
end

# Function to replan routes for all agents after cost map change
function replan_all_routes!(agents, goals, pathfinder, penalty_map)
    print_both("\nReplanning routes for all agents after cost map change...")
    for (agent, goal) in zip(agents, goals)
        Pathfinding.plan_route!(agent, goal, pathfinder)
        if haskey(pathfinder.agent_paths, agent.id)
            print_both("  Agent $(agent.id): New path length = $(length(pathfinder.agent_paths[agent.id]))")
        end
    end
end

# Simulation loop
max_steps = 15
agents = [agent1, agent2, agent3]
goals = [agent1_goal, agent2_goal, agent3_goal]

for step in 1:max_steps
    print_both("\n" * "─" ^ 40)
    print_both("STEP $step")
    print_both("─" ^ 40)
    
    # Move all agents one step
    for agent in agents
        if haskey(pathfinder.agent_paths, agent.id) && !isempty(pathfinder.agent_paths[agent.id])
            old_pos = agent.pos
            Pathfinding.move_along_route!(agent, model, pathfinder)
            print_both("  Agent $(agent.id): $old_pos → $(agent.pos)")
        end
    end
    
    # Dynamic penalty map changes at specific steps
    if step == 3
        # Increase penalty in an area that agents might be approaching
        update_penalty_map!(penalty_map, [
            ((2, 2), 100),  # Very high penalty
            ((2, 3), 100),
            ((3, 2), 100),
        ])
        # Update the cost metric's penalty map
        pathfinder.cost_metric.pmap = penalty_map
        replan_all_routes!(agents, goals, pathfinder, penalty_map)
        print_both("  → Agents will replan to avoid high penalty area")
        
    elseif step == 6
        # Reduce penalty in another area (make it easier)
        update_penalty_map!(penalty_map, [
            ((6, 3), 5),   # Lower penalty
            ((6, 4), 5),
            ((6, 5), 5),
        ])
        pathfinder.cost_metric.pmap = penalty_map
        replan_all_routes!(agents, goals, pathfinder, penalty_map)
        print_both("  → Agents can now use easier path through this area")
        
    elseif step == 9
        # Add new high penalty area
        update_penalty_map!(penalty_map, [
            ((4, 6), 80),  # New high penalty
            ((5, 6), 80),
        ])
        pathfinder.cost_metric.pmap = penalty_map
        replan_all_routes!(agents, goals, pathfinder, penalty_map)
        print_both("  → New obstacle area created, agents must avoid it")
    end
    
    # Check if all agents reached their goals
    all_at_goals = all(agents[i].pos == goals[i] for i in 1:length(agents))
    if all_at_goals
        print_both("\n✅ All agents reached their goals!")
        break
    end
    
    # Safety check
    if step >= max_steps
        print_both("\n⚠️  Reached maximum steps ($max_steps)")
        for (i, agent) in enumerate(agents)
            if agent.pos != goals[i]
                print_both("  Agent $(agent.id) at $(agent.pos), goal: $(goals[i])")
            end
        end
    end
end

# ============================================================================
# SUMMARY
# ============================================================================
print_both("\n" * "=" ^ 60)
print_both("EXECUTION SUMMARY")
print_both("=" ^ 60)
print_both()
print_both("KEY FEATURES DEMONSTRATED:")
print_both("─" ^ 60)
print_both("  1. Multiple agents (3 agents) with different start/goal positions")
print_both("  2. Penalty map with varying costs (high, medium, normal)")
print_both("  3. Dynamic cost map changes during execution:")
print_both("     - Step 3: Increased penalty in area (100)")
print_both("     - Step 6: Decreased penalty in area (5)")
print_both("     - Step 9: Added new high penalty area (80)")
print_both("  4. Agents replan routes when cost map changes")
print_both("  5. D* Lite adapts to dynamic environments")
print_both()
print_both("D* LITE ADVANTAGES:")
print_both("─" ^ 60)
print_both("  • Efficient replanning when costs change")
print_both("  • Only updates affected areas (not full replan)")
print_both("  • Handles multiple agents independently")
print_both("  • Works with dynamic penalty maps")
print_both()
print_both("FINAL AGENT POSITIONS:")
print_both("─" ^ 60)
for (i, agent) in enumerate(agents)
    reached = agent.pos == goals[i] ? "✓" : "✗"
    print_both("  Agent $(agent.id): $(agent.pos) (Goal: $(goals[i])) $reached")
end
print_both()

print_both("\nTest completed successfully! ✓")
print_both("\nOutput saved to: dstar_lite_test_output.txt")

# Close the output file
close(OUTPUT_FILE)
println("\n✓ Output file closed. Results saved to: dstar_lite_test_output.txt")
