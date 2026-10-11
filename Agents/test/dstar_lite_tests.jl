using Agents, Test
using Agents.Pathfinding

function _travel_cost(pf, path)
    cost = 0.0
    for i in 1:(length(path) - 1)
        cost += delta_cost(pf, path[i], path[i + 1])
    end
    return cost
end

@testset "DStarLite" begin
    space = GridSpace((8, 8); periodic = false)
    pf = DStarLite(space; diagonal_movement = true)

    @testset "find_path" begin
        p = collect(find_path(pf, (1, 1), (8, 8)))
        @test !isempty(p)
        @test p[end] == (8, 8)
        @test first(p) != (1, 1)
        @test isnothing(find_path(pf, (1, 1), (0, 1)))
    end

    @testset "full-field planner (Scenario 3 API)" begin
        pl = init_planner(pf, (8, 8))
        @test pl.s_start === nothing
        @test pl.km == 0.0
        @test pl.g[8, 8] == 0.0
        @test pl.g[1, 1] < Inf
        path = extract_path(pl, (1, 1))
        @test path[1] == (1, 1)
        @test path[end] == (8, 8)
        @test pl.g[3, 4] < Inf
    end

    @testset "focused D* Lite matches full-field cost-to-go at start" begin
        full = init_planner(pf, (8, 8))
        focused = init_planner(pf, (8, 8), (1, 1); heuristic = :manhattan)
        @test focused.s_start == (1, 1)
        @test focused.heuristic_kind === :manhattan
        # {10''} may stop with s_start overconsistent, so rhs(s_start) (not
        # g) holds its goal distance (paper, "Optimizations")
        @test focused.rhs[1, 1] ≈ full.g[1, 1]
        path = extract_path(focused, (1, 1))
        @test path[end] == (8, 8)
        @test _travel_cost(pf, path) ≈ full.g[1, 1]
    end

    @testset "km changes only when edge costs change ({37''-39''})" begin
        pl = init_planner(pf, (8, 8), (1, 1); heuristic = :manhattan)
        @test pl.km == 0.0
        update_start!(pl, (2, 1))
        @test pl.s_start == (2, 1)
        @test pl.km == 0.0            # moving alone never touches km
        @test pl.s_last == (1, 1)
        @test extract_path(pl, (2, 1))[end] == (8, 8)
        update_after_cm_change!(pl, Tuple{Int,Int}[])
        @test pl.km ≈ 1.0             # h(s_last, s_start)
        @test pl.s_last == (2, 1)
        @test_throws ArgumentError update_start!(init_planner(pf, (8, 8)), (2, 1))
    end

    @testset "inadmissible heuristics are rejected" begin
        pf_max = DStarLite(space; cost_metric = MaxDistance{2}())
        # a diagonal step costs 1 but Manhattan / Euclidean estimate 2 / √2
        @test_throws ArgumentError init_planner(pf_max, (8, 8), (1, 1); heuristic = :manhattan)
        @test_throws ArgumentError init_planner(pf_max, (8, 8), (1, 1); heuristic = :euclidean)
        @test init_planner(pf_max, (8, 8), (1, 1); heuristic = :delta_cost).rhs[1, 1] ≈ 7
    end

    @testset "cost-map change replans" begin
        pmap = zeros(Int, 8, 8)
        pf_pen = DStarLite(space; cost_metric = AbsolutePenaltyMap(pmap, MaxDistance{2}()))
        pl = init_planner(pf_pen, (8, 8))
        path_before = extract_path(pl, (1, 1))
        @test path_before[end] == (8, 8)
        pmap[4, 4] = 80
        update_after_cm_change!(pl, [(4, 4)])
        path_after = extract_path(pl, (1, 1))
        @test path_after[end] == (8, 8)
        @test (4, 4) ∉ path_after
    end

    @testset "blocked cells" begin
        walk = trues(8, 8)
        walk[4, 1:8] .= false
        walk[4, 8] = true
        pf_wall = DStarLite(space; walkmap = walk, diagonal_movement = false)
        path = extract_path(init_planner(pf_wall, (8, 8)), (1, 1))
        @test path[end] == (8, 8)
        @test all(c -> c[1] != 4 || c[2] == 8, path)
    end

    @testset "periodic heuristic respects wraparound" begin
        pspace = GridSpace((8, 8); periodic = true)
        pf_periodic = DStarLite(
            pspace;
            diagonal_movement = false,
            cost_metric = DirectDistance{2}([1]),
        )
        # (1,1) -> (8,8) is 14 steps direct but 2 steps via wraparound in each dim
        for hk in (:manhattan, :euclidean)
            pl = init_planner(pf_periodic, (8, 8), (1, 1); heuristic = hk)
            @test pl.rhs[1, 1] ≈ 2.0   # g(s_start) may stay ∞ under {10''}
            path = extract_path(pl, (1, 1))
            @test path[end] == (8, 8)
            @test _travel_cost(pf_periodic, path) ≈ 2.0
        end
    end

    @testset "cost increase on the cached focused path" begin
        pmap = zeros(Int, 8, 8)
        pf_pen = DStarLite(space; cost_metric = AbsolutePenaltyMap(pmap, MaxDistance{2}()))
        pl = init_planner(pf_pen, (8, 8), (1, 1))
        @test (4, 4) in extract_path(pl, (1, 1))

        pmap[4, 4] = 80
        update_after_cm_change!(pl, [(4, 4)])
        path = extract_path(pl, (1, 1))
        @test path[1] == (1, 1)
        @test path[end] == (8, 8)
        @test (4, 4) ∉ path
        @test allunique(path)
        @test _travel_cost(pf_pen, path) ≈ init_planner(pf_pen, (8, 8)).g[1, 1]
    end

    @testset "update_after_cm_change! with new_start" begin
        pmap = zeros(Int, 8, 8)
        pf_pen = DStarLite(space; cost_metric = AbsolutePenaltyMap(pmap, MaxDistance{2}()))
        pl = init_planner(pf_pen, (8, 8), (1, 1))
        extract_path(pl, (1, 1))

        pmap[4, 4] = 80
        new_start = (2, 2)
        update_after_cm_change!(pl, [(4, 4)], new_start)
        @test pl.s_start == new_start
        @test pl.s_last == new_start
        @test pl.km ≈ 1.0             # Chebyshev h((1,1), (2,2))

        path = extract_path(pl, new_start)
        @test path[1] == new_start
        @test path[end] == (8, 8)
        @test (4, 4) ∉ path
        @test _travel_cost(pf_pen, path) ≈ init_planner(pf_pen, (8, 8)).g[new_start...]
    end

    @testset "Theorem 8: incremental replanning matches a fresh search" begin
        # The agent follows its path while random penalty changes happen; after
        # each step the greedy path must cost exactly the goal distance
        # computed from scratch on the current costs.
        rng = Xoshiro(2002)
        n = 12
        sp = GridSpace((n, n); periodic = false)
        goal = (n, n)
        for trial in 1:40
            walk = rand(rng, n, n) .> 0.25
            walk[1, 1] = walk[goal...] = true
            pmap = rand(rng, 0:5, n, n)
            base = isodd(trial) ? MaxDistance{2}() : DirectDistance{2}()
            pf_r = DStarLite(sp; walkmap = walk, cost_metric = AbsolutePenaltyMap(pmap, base))
            pl = init_planner(pf_r, goal, (1, 1))
            s = (1, 1)
            for step in 1:60
                d = init_planner(pf_r, goal).g[s...]
                path = extract_path(pl, s)
                if d == Inf
                    @test path == [s]
                    break
                end
                @test path[end] == goal
                @test allunique(path)
                @test _travel_cost(pf_r, path) ≈ d
                s == goal && break
                s = path[2]
                changed = unique([(rand(rng, 1:n), rand(rng, 1:n)) for _ in 1:4])
                for c in changed
                    pmap[c...] = rand(rng, 0:30)
                end
                if isodd(step)
                    update_start!(pl, s)
                    update_after_cm_change!(pl, changed)
                else
                    update_after_cm_change!(pl, changed, s)
                end
            end
        end
    end

    @testset "full-field incremental updates match a fresh search (Scenario 3)" begin
        rng = Xoshiro(480)
        n = 12
        for _ in 1:20
            walk = rand(rng, n, n) .> 0.25
            walk[n, n] = true
            pmap = rand(rng, 0:5, n, n)
            pf_r = DStarLite(GridSpace((n, n); periodic = false);
                walkmap = walk, cost_metric = AbsolutePenaltyMap(pmap, MaxDistance{2}()))
            pl = init_planner(pf_r, (n, n))
            for _ in 1:15
                changed = unique([(rand(rng, 1:n), rand(rng, 1:n)) for _ in 1:6])
                for c in changed
                    pmap[c...] = rand(rng, 0:40)
                end
                update_after_cm_change!(pl, changed)
                @test pl.g == init_planner(pf_r, (n, n)).g
                @test isempty(pl.U)
            end
        end
    end

    @testset "find_path is optimal" begin
        rng = Xoshiro(476)
        for _ in 1:20
            walk = rand(rng, 10, 10) .> 0.25
            walk[1, 1] = walk[10, 10] = true
            pmap = rand(rng, 0:9, 10, 10)
            pf_r = DStarLite(GridSpace((10, 10); periodic = false);
                walkmap = walk, cost_metric = AbsolutePenaltyMap(pmap, MaxDistance{2}()))
            d = init_planner(pf_r, (10, 10)).g[1, 1]
            p = find_path(pf_r, (1, 1), (10, 10))
            if d == Inf
                @test isnothing(p)
            else
                @test _travel_cost(pf_r, [(1, 1); collect(p)]) ≈ d
            end
        end
    end

    @testset "CellHeap keeps min-order under insert / key change / delete" begin
        H = Agents.Pathfinding.CellHeap((20, 20))
        ref = Dict{Tuple{Int,Int},Tuple{Float64,Float64}}()
        rng = Xoshiro(7)
        for _ in 1:2000
            u = (rand(rng, 1:20), rand(rng, 1:20))
            if rand(rng) < 0.3
                delete!(H, u); delete!(ref, u)
            else
                k = (Float64(rand(rng, 0:50)), Float64(rand(rng, 0:50)))
                H[u] = k; ref[u] = k
            end
            @test length(H) == length(ref)
            @test all(haskey(H, v) for v in keys(ref))
        end
        popped = Tuple{Float64,Float64}[]
        while !isempty(H)
            u, k = Agents.Pathfinding.heap_top(H)
            @test ref[u] == k
            @test Agents.Pathfinding.heap_pop!(H) == u
            @test !haskey(H, u)
            push!(popped, k)
        end
        @test issorted(popped)
        @test length(popped) == length(ref)
    end

    @testset "lazy neighbors match walkable_neighbors" begin
        wm = trues(6, 5); wm[2, 2] = false; wm[5, 4] = false
        for periodic in (false, true), diag in (true, false)
            pf = DStarLite((6, 5); periodic, diagonal_movement = diag, walkmap = copy(wm))
            for I in CartesianIndices(wm)
                u = Tuple(I)
                @test collect(Agents.Pathfinding.WalkableNeighbors(pf, u)) ==
                      Agents.Pathfinding.walkable_neighbors(u, pf)
            end
        end
    end
end
