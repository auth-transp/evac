module DStarLiteModule

    # Βασικά πακέτα
    using DataStructures  # για PriorityQueue (αν δεν το έχεις: ] add DataStructures)
    using Agents
    using Agents.Pathfinding
    using Agents: MutableLinkedList
    using LinearAlgebra

    export DStarLite, plan_best_route!, penaltymap, handle_map_changes!, compute_shortest_path!, get_next_step

    # ========== ΒΟΗΘΗΤΙΚΕΣ ΔΟΜΕΣ ==========

    # Key struct (παρόμοιο με paper)
    struct Key
        k1::Float64
        k2::Float64
    end

    Base.isless(a::Key,b::Key) = a.k1 < b.k1 || (isapprox(a.k1,b.k1) && a.k2 < b.k2)

    # Min-heap wrapper χρησιμοποιώντας DataStructures.PriorityQueue με Key ordering
    struct MinHeap
        pq::PriorityQueue{Tuple{Int,Int},Key}
        MinHeap() = new(PriorityQueue{Tuple{Int,Int},Key}())
    end

    function isempty(h::MinHeap)
        isempty(h.pq)
    end

    function top_key(h::MinHeap)
        k = peek(h.pq)[2]
        return k
    end

    function push!(h::MinHeap, node::Tuple{Int,Int}, k::Key)
        # PriorityQueue in DataStructures uses highest priority for maximum; we invert by using Key as priority directly
        # But PriorityQueue expects Comparable priorities — we'll use Key and define comparison via pair ordering
        h.pq[node] = k
    end

    function pop!(h::MinHeap)
        pair = pop!(h.pq)  # returns (node=>priority)
        for (n,k) in pair
            return n,k
        end
    end

    # ========== DStarLite τύποι ==========

    # Per-agent D* instance - minimal
    mutable struct AgentDS
        rows::Int
        cols::Int
        s_start::Tuple{Int,Int}
        s_goal::Tuple{Int,Int}
        k_m::Float64
        g::Array{Float64,2}
        rhs::Array{Float64,2}
        openq::MinHeap
        heuristic::Function
        sensed_walkmap::BitArray{2}
        # placeholder for edges changes event
        new_edges_and_old_costs::Union{Nothing,Any}
        AgentDS(rows,cols,s_start,s_goal; heuristic=(u,v)->hypot(u[1]-v[1],u[2]-v[2])) = begin
            g = fill(Inf, rows, cols)
            rhs = fill(Inf, rows, cols)
            openq = MinHeap()
            sensed = trues(rows,cols)
            AgentDS(rows,cols,s_start,s_goal,0.0,g,rhs,openq,heuristic,sensed,nothing)
        end
    end

    # The top-level pathfinder struct used by the rest of the program
    mutable struct DStarLite{D,P,M,T,C} <: Agents.Pathfinding.GridPathfinder{D,P,M}
        agent_paths::Dict{Int,Agents.Pathfinding.Path{D,T}}
        dims::NTuple{D,T}
        neighborhood::Vector{CartesianIndex{D}}
        admissibility::Float64
        walkmap::BitArray{D}
        cost_metric::C
        function DStarLite{D,P,M,T,C}(agent_paths, dims, neighborhood, admissibility, walkmap, cost_metric) where {D,P,M,T,C}
            new(agent_paths, dims, neighborhood, admissibility, walkmap, cost_metric)
        end
    end

    # constructor similar to AStar(space; ...)
    function DStarLite(dims::NTuple{2,Int}; walkmap::BitArray{2}=trues(dims), cost_metric=nothing)
        neighborhood = Agents.Pathfinding.moore_neighborhood(2)
        return DStarLite{2,typeof(false),true,Int,typeof(cost_metric)}(
            Dict{Int,Agents.Pathfinding.Path{2,Int}}(),
            dims,
            neighborhood,
            0.0,
            walkmap,
            cost_metric
        )
    end

    # simple penaltymap accessor
    function penaltymap(pathfinder::DStarLite)
        if pathfinder.cost_metric === nothing
            return nothing
        elseif hasproperty(pathfinder.cost_metric, :pmap)
            return pathfinder.cost_metric.pmap
        else
            return nothing
        end
    end

    # ========= D* core functions (μινιμαλιστικές & επεκτάσιμες) ==========

    # calculate key for a given cell (i,j)
    function calculate_key(ds::AgentDS, s::Tuple{Int,Int})
        i,j = s
        gval = ds.g[i,j]
        rhsval = ds.rhs[i,j]
        k1 = min(gval, rhsval) + ds.heuristic(ds.s_start, s) + ds.k_m
        k2 = min(gval, rhsval)
        return Key(k1, k2)
    end

    # neighbor generator (8-neighborhood)
    function succ(ds::AgentDS, s::Tuple{Int,Int}; avoid_obstacles=true)
        i,j = s
        rows,cols = ds.rows, ds.cols
        out = Tuple{Int,Int}[]
        for di in -1:1, dj in -1:1
            if di==0 && dj==0; continue; end
            ii = i+di; jj = j+dj
            if 1 <= ii <= rows && 1 <= jj <= cols
                if !avoid_obstacles || ds.sensed_walkmap[ii,jj]
                    push!(out, (ii,jj))
                end
            end
        end
        return out
    end

    # cost function c(u,v)
    function c(ds::AgentDS, u::Tuple{Int,Int}, v::Tuple{Int,Int})
        if !ds.sensed_walkmap[u...] || !ds.sensed_walkmap[v...]
            return Inf
        end
        return hypot(u[1]-v[1], u[2]-v[2])
    end

    # update_vertex!
    function update_vertex!(ds::AgentDS, u::Tuple{Int,Int})
        i,j = u
        if ds.g[i,j] != ds.rhs[i,j]
            push!(ds.openq, u, calculate_key(ds, u))
        else
            # if equal, remove from openq - naive approach: re-create pq later or support removal
            # For simplicity, we won't implement sophisticated removal here.
        end
    end

    # compute_shortest_path! (μινιμαλιστική, μπορεί να βελτιωθεί)
    function compute_shortest_path!(ds::AgentDS; max_iters=200000)
        # initialize: goal rhs=0
        gi,gj = ds.s_goal
        ds.rhs[gi,gj] = 0.0
        push!(ds.openq, ds.s_goal, calculate_key(ds, ds.s_goal))

        iters = 0
        while !isempty(ds.openq) && (top_key(ds.openq) < calculate_key(ds, ds.s_start) || ds.rhs[ds.s_start...] > ds.g[ds.s_start...])
            # break timeout
            iters += 1
            if iters > max_iters
                @warn "compute_shortest_path!: max iters reached"
                break
            end

            u,k_old = pop!(ds.openq)
            k_new = calculate_key(ds, u)
            if k_old < k_new
                push!(ds.openq, u, k_new)
            elseif ds.g[u...] > ds.rhs[u...]
                ds.g[u...] = ds.rhs[u...]
                for s in succ(ds, u, avoid_obstacles=true)
                    if s != ds.s_goal
                        ds.rhs[s...] = min(ds.rhs[s...], c(ds, s, u) + ds.g[u...])
                    end
                    update_vertex!(ds, s)
                end
            else
                g_old = ds.g[u...]
                ds.g[u...] = Inf
                tofix = succ(ds, u, avoid_obstacles=true)
                push!(tofix, u)
                for s in tofix
                    if ds.rhs[s...] == c(ds, s, u) + g_old
                        if s != ds.s_goal
                            min_s = Inf
                            for s2 in succ(ds, s, avoid_obstacles=true)
                                min_s = min(min_s, c(ds, s, s2) + ds.g[s2...])
                            end
                            ds.rhs[s...] = min_s
                        end
                    end
                    update_vertex!(ds, s)
                end
            end
        end
        return
    end

    # build a path (list of grid nodes) from s_start to s_goal using greedy follow of min g+cost
    function build_path_from_ds(ds::AgentDS)
        path = Agents.Pathfinding.Path{2,Int64}()
        cur = ds.s_start
        push!(path, cur)
        while cur != ds.s_goal
            succs = succ(ds, cur, avoid_obstacles=true)
            best = nothing; bestval = Inf
            for s in succs
                val = c(ds, cur, s) + ds.g[s...]
                if val < bestval
                    bestval = val
                    best = s
                end
            end
            if best === nothing || best == cur
                break
            end
            push!(path, best)
            cur = best
            if length(path) > ds.rows*ds.cols
                @warn "build_path_from_ds: stuck, aborting"
                break
            end
        end
        return path
    end

    # get next step (one grid cell) from ds
    function get_next_step(ds::AgentDS)
        # greedy as in build_path_from_ds but returns next cell
        cur = ds.s_start
        succs = succ(ds, cur, avoid_obstacles=true)
        best = nothing; bestval = Inf
        for s in succs
            val = c(ds, cur, s) + ds.g[s...]
            if val < bestval
                bestval = val; best = s
            end
        end
        return best
    end

    # handle_map_changes!: vc is a structure describing vertices changed (we accept a vector of (v, edges_and_old))
    function handle_map_changes!(ds::AgentDS, vc)
        # For now expect vc to be vector of tuples: (v::Tuple, edges_and_old::Dict{Tuple,Float64})
        if vc === nothing; return end
        # Update ds.new_edges_and_old_costs to inform rescan (not used in this simple impl)
        ds.new_edges_and_old_costs = vc
        # For simplicity, we'll update ds.sensed_walkmap if entire cell became blocked/unblocked
        for (v, edges_old) in vc
            # decide new occupancy: if any edge now Inf => block the node
            # (caller should have already computed new occupancy map)
            # Here we just toggle based on edges_old info not available; skip
        end
    end

    # ========= Interfacing functions for Agents.jl =========

    # plan_best_route!(agent, dests, pathfinder)  — interface used in your script
    function plan_best_route!(agent, dests, pathfinder::DStarLite)
        # choose first dest for now
        dest = dests[1]

        # IMPORTANT: convert continuous agent.pos (x,y) -> grid (row, col) = (y,x)
        s_start = ( clamp(floor(Int, agent.pos[2]), 1, size(pathfinder.walkmap,1)),
                    clamp(floor(Int, agent.pos[1]), 1, size(pathfinder.walkmap,2)) )

        s_goal  = ( clamp(floor(Int, dest[2]), 1, size(pathfinder.walkmap,1)),
                    clamp(floor(Int, dest[1]), 1, size(pathfinder.walkmap,2)) )

        # create AgentDS and attach to agent
        ds = AgentDS(size(pathfinder.walkmap,1), size(pathfinder.walkmap,2), s_start, s_goal)
        # initialize ds.sensed_walkmap from pathfinder.walkmap
        ds.sensed_walkmap .= pathfinder.walkmap

        # compute initial plan
        compute_shortest_path!(ds)

        # build path and set into pathfinder.agent_paths[agent.id]
        pathfinder.agent_paths[agent.id] = build_path_from_ds(ds)

        # store ds in agent if agent has :dstar field
        try
            agent.dstar = ds
        catch e
            # ignore if no field
        end

        return
    end

    # minimal overload so that Agents.remove_agent! etc. still work if needed
    function Agents.remove_agent!(agent::Agents.AbstractAgent, model::Agents.ABM, pathfinder::DStarLite)
        delete!(pathfinder.agent_paths, agent.id)
        Agents.remove_agent_from_model!(agent, model)
        Agents.remove_agent_from_space!(agent, model)
    end

end # module