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
        @test focused.g[1, 1] ≈ full.g[1, 1]
        @test focused.rhs[1, 1] ≈ focused.g[1, 1]
        path = extract_path(focused, (1, 1))
        @test path[end] == (8, 8)
        @test _travel_cost(pf, path) ≈ focused.g[1, 1]
    end

    @testset "km updates when start moves" begin
        pl = init_planner(pf, (8, 8), (1, 1); heuristic = :manhattan)
        @test pl.km == 0.0
        update_start!(pl, (2, 1))
        @test pl.s_start == (2, 1)
        @test pl.km ≈ 1.0
        Pathfinding.compute_shortest_path!(pl)
        @test pl.g[2, 1] < Inf
        @test extract_path(pl, (2, 1))[end] == (8, 8)
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
            @test pl.g[1, 1] ≈ 2.0
            path = extract_path(pl, (1, 1))
            @test path[end] == (8, 8)
            @test _travel_cost(pf_periodic, path) ≈ 2.0
        end
    end

    @testset "cost increase invalidates a cached focused path (2-arg form)" begin
        # Regression test: a cost increase on a cell that lies on the agent's
        # already-cached diagonal route must not leave stale g/rhs values on
        # cells the greedy walk in extract_path depends on, even when the
        # caller doesn't pass `new_start` to update_after_cm_change!.
        pmap = zeros(Int, 8, 8)
        pf_pen = DStarLite(space; cost_metric = AbsolutePenaltyMap(pmap, MaxDistance{2}()))
        pl = init_planner(pf_pen, (8, 8), (1, 1); heuristic = :manhattan)
        extract_path(pl, (1, 1))

        pmap[4, 4] = 80
        update_after_cm_change!(pl, [(4, 4)])
        path = extract_path(pl, (1, 1))
        @test path[1] == (1, 1)
        @test path[end] == (8, 8)
        @test (4, 4) ∉ path
        @test allunique(path)
        @test _travel_cost(pf_pen, path) ≈ pl.g[1, 1]
    end

    @testset "update_after_cm_change! with new_start avoids stale-start replan" begin
        pmap = zeros(Int, 8, 8)
        pf_pen = DStarLite(space; cost_metric = AbsolutePenaltyMap(pmap, MaxDistance{2}()))
        pl = init_planner(pf_pen, (8, 8), (1, 1); heuristic = :manhattan)
        extract_path(pl, (1, 1))

        pmap[4, 4] = 80
        new_start = (2, 1)
        update_after_cm_change!(pl, [(4, 4)], new_start)
        # start (and km) should already be updated by update_after_cm_change!
        @test pl.s_start == new_start

        path = extract_path(pl, new_start)
        @test path[1] == new_start
        @test path[end] == (8, 8)
        @test (4, 4) ∉ path
        @test _travel_cost(pf_pen, path) ≈ pl.g[new_start...]
    end
end
