using Agents, Test
using Agents.Pathfinding

function _lpa_travel_cost(pf, path)
    cost = 0.0
    for i in 1:(length(path) - 1)
        cost += delta_cost(pf, path[i], path[i + 1])
    end
    return cost
end

# Οι Invariants του paper: (1) rhs(s) = 0 για την s_start, αλλιώς min_{s'∈pred(s)} g(s') + c(s',s) (εξ. (2)),
# και rhs(s) = g(p(s)) + c(p(s),s)· (2) η U περιέχει ΑΚΡΙΒΩΣ τις τοπικά ασυνεπείς κορυφές, με κλειδί CalculateKey.
function _lpa_invariants_hold(pl)
    PF = Agents.Pathfinding
    for I in CartesianIndices(pl.g)
        s = Tuple(I)
        if s == pl.start
            pl.rhs[I] == 0.0 || return false
        else
            PF._argmin_pred(pl, s)[2] == pl.rhs[I] || return false          # Invariant 1
            if pl.p[I] != PF.LPA_NULL
                pl.g[pl.p[I]...] + PF._cost(pl, pl.p[I], s) == pl.rhs[I] || return false   # rhs(s) = g(p(s)) + c(p(s),s)
            else
                pl.rhs[I] == Inf || return false
            end
        end
        haskey(pl.U, s) == (pl.g[I] != pl.rhs[I]) || return false           # Invariant 2
    end
    return true
end

@testset "LPAStar" begin
    space = GridSpace((8, 8); periodic = false)
    pf = LPAStar(space; diagonal_movement = true)

    @testset "find_path" begin
        p = collect(find_path(pf, (1, 1), (8, 8)))
        @test !isempty(p)
        @test p[end] == (8, 8)
        @test first(p) != (1, 1)
        @test isnothing(find_path(pf, (1, 1), (0, 1)))
        @test isempty(find_path(pf, (3, 3), (3, 3)))          # αρχή = στόχος: κενή διαδρομή
        @test_throws ArgumentError init_planner(pf, (0, 1), (8, 8))
    end

    @testset "planner initialization (Figure 7, {02'}-{07'})" begin
        pl = init_planner(pf, (1, 1), (8, 8))
        @test pl isa LPAStarPlanner
        @test _lpa_invariants_hold(pl)
        @test pl.g[1, 1] == 0.0                               # g(s_start) = rhs(s_start) = 0
        @test pl.rhs[1, 1] == 0.0
        @test pl.rhs[8, 8] ≈ 7 * 14                             # 7 διαγώνια βήματα με DirectDistance [10, 14]
        path = extract_path(pl)
        @test path[1] == (1, 1)
        @test path[end] == (8, 8)
        @test _lpa_travel_cost(pf, path) ≈ pl.rhs[8, 8]
        @test allunique(path)
    end

    @testset "inadmissible heuristics are rejected" begin
        pf_max = LPAStar(space; cost_metric = MaxDistance{2}())
        @test_throws ArgumentError init_planner(pf_max, (1, 1), (8, 8); heuristic = :manhattan)
        @test_throws ArgumentError init_planner(pf_max, (1, 1), (8, 8); heuristic = :euclidean)
        @test init_planner(pf_max, (1, 1), (8, 8); heuristic = :delta_cost).rhs[8, 8] ≈ 7
        @test init_planner(pf_max, (1, 1), (8, 8); heuristic = :none).rhs[8, 8] ≈ 7
    end

    @testset "blocked cells and unreachable goals" begin
        walk = trues(8, 8)
        walk[4, 1:7] .= false
        pf_wall = LPAStar(space; walkmap = walk, diagonal_movement = false)
        path = extract_path(init_planner(pf_wall, (1, 1), (8, 1)))
        @test path[end] == (8, 1)
        @test all(c -> c[1] != 4 || c[2] == 8, path)          # περνά από το άνοιγμα (4, 8)

        walk[4, 8] = false                                    # τώρα ο τοίχος είναι κλειστός
        pf_closed = LPAStar(space; walkmap = walk, diagonal_movement = false)
        pl = init_planner(pf_closed, (1, 1), (8, 1))
        @test pl.rhs[8, 1] == Inf
        @test isnothing(extract_path(pl))
        @test isnothing(find_path(pf_closed, (1, 1), (8, 1)))
    end

    @testset "periodic heuristic respects wraparound" begin
        pf_periodic = LPAStar(
            GridSpace((8, 8); periodic = true);
            diagonal_movement = false,
            cost_metric = DirectDistance{2}([1]),
        )
        # (1,1) -> (8,8) is 14 steps direct but 2 steps via wraparound in each dim
        for hk in (:manhattan, :euclidean, :delta_cost, :none)
            pl = init_planner(pf_periodic, (1, 1), (8, 8); heuristic = hk)
            @test pl.rhs[8, 8] ≈ 2.0
            @test _lpa_travel_cost(pf_periodic, extract_path(pl)) ≈ 2.0
        end
    end

    @testset "cost increase on the cached path" begin
        pmap = zeros(Int, 8, 8)
        pf_pen = LPAStar(space; cost_metric = AbsolutePenaltyMap(pmap, MaxDistance{2}()))
        pl = init_planner(pf_pen, (1, 1), (8, 8))
        @test (4, 4) in extract_path(pl)

        pmap[4, 4] = 80
        update_after_cm_change!(pl, [(4, 4)])
        path = extract_path(pl)
        @test path[1] == (1, 1)
        @test path[end] == (8, 8)
        @test (4, 4) ∉ path
        @test allunique(path)
        @test _lpa_travel_cost(pf_pen, path) ≈ init_planner(pf_pen, (1, 1), (8, 8)).rhs[8, 8]

        pmap[4, 4] = 0                                        # ο δρόμος ξαναφθηναίνει (overconsistent)
        update_after_cm_change!(pl, [(4, 4)])
        @test pl.rhs[8, 8] ≈ init_planner(pf_pen, (1, 1), (8, 8)).rhs[8, 8]
        @test (4, 4) in extract_path(pl)
    end

    @testset "blocking and unblocking cells" begin
        walk = trues(8, 8)
        pf_w = LPAStar(space; walkmap = walk, diagonal_movement = false, cost_metric = MaxDistance{2}())
        pl = init_planner(pf_w, (1, 1), (8, 8))
        @test pl.rhs[8, 8] ≈ 14
        walk[1:7, 4] .= false                                 # τοίχος με άνοιγμα μόνο στο (8, 4)
        update_after_cm_change!(pl, [(i, 4) for i in 1:7])
        @test pl.rhs[8, 8] ≈ init_planner(pf_w, (1, 1), (8, 8)).rhs[8, 8]
        @test all(c -> c[2] != 4 || c[1] == 8, extract_path(pl))
        walk[1:8, 4] .= false                                 # κλείνει και το άνοιγμα
        update_after_cm_change!(pl, [(8, 4)])
        @test pl.rhs[8, 8] == Inf
        @test isnothing(extract_path(pl))
        walk[1:8, 4] .= true                                  # ξεμπλοκάρισμα όλων
        update_after_cm_change!(pl, [(i, 4) for i in 1:8])
        @test pl.rhs[8, 8] ≈ 14
    end

    @testset "incremental replanning matches a fresh search" begin
        # Τυχαίες αλλαγές penalties και εμποδίων· μετά από κάθε αλλαγή η απόσταση rhs(s_goal) και η διαδρομή
        # του αυξητικού planner πρέπει να ισούνται με αναζήτηση από την αρχή, ΚΑΙ με το D* Lite, και να
        # ισχύουν οι Invariants 1-2 του paper σε όλα τα cells.
        rng = Xoshiro(2004)
        n = 12
        sp = GridSpace((n, n); periodic = false)
        for trial in 1:40
            walk = rand(rng, n, n) .> 0.2
            walk[1, 1] = walk[n, n] = true
            pmap = rand(rng, 0:5, n, n)
            base = isodd(trial) ? MaxDistance{2}() : DirectDistance{2}()
            heur = isodd(trial) ? :delta_cost : :none
            pf_r = LPAStar(sp; walkmap = walk, cost_metric = AbsolutePenaltyMap(pmap, base))
            pf_d = DStarLite(sp; walkmap = walk, cost_metric = AbsolutePenaltyMap(pmap, base))
            pl = init_planner(pf_r, (1, 1), (n, n); heuristic = heur)
            for _ in 1:25
                changed = unique([(rand(rng, 1:n), rand(rng, 1:n)) for _ in 1:5])
                for c in changed
                    if rand(rng) < 0.3 && c != (1, 1) && c != (n, n)
                        walk[c...] = !walk[c...]              # αλλαγή βατότητας
                    else
                        pmap[c...] = rand(rng, 0:30)          # αλλαγή penalty
                    end
                end
                update_after_cm_change!(pl, changed)
                fresh = init_planner(pf_r, (1, 1), (n, n); heuristic = heur)
                dstar = init_planner(pf_d, (n, n)).g[1, 1]    # D* Lite full-field cost-to-go
                @test _lpa_invariants_hold(pl)
                @test pl.rhs[n, n] == fresh.rhs[n, n]
                @test pl.rhs[n, n] ≈ dstar
                path = extract_path(pl)
                if pl.rhs[n, n] == Inf
                    @test isnothing(path)
                else
                    @test path[1] == (1, 1) && path[end] == (n, n)
                    @test allunique(path)
                    @test _lpa_travel_cost(pf_r, path) ≈ pl.rhs[n, n]
                end
            end
        end
    end

    @testset "find_path is optimal" begin
        rng = Xoshiro(2005)
        for _ in 1:20
            walk = rand(rng, 10, 10) .> 0.25
            walk[1, 1] = walk[10, 10] = true
            pmap = rand(rng, 0:9, 10, 10)
            sp = GridSpace((10, 10); periodic = false)
            pf_r = LPAStar(sp; walkmap = walk, cost_metric = AbsolutePenaltyMap(pmap, MaxDistance{2}()))
            d = init_planner(DStarLite(sp; walkmap = walk, cost_metric = AbsolutePenaltyMap(pmap, MaxDistance{2}())), (10, 10)).g[1, 1]
            p = find_path(pf_r, (1, 1), (10, 10))
            if d == Inf
                @test isnothing(p)
            else
                @test _lpa_travel_cost(pf_r, [(1, 1); collect(p)]) ≈ d
            end
        end
    end

    @testset "Agents API (GridSpace)" begin
        gspace = GridSpace((5, 5); periodic = false)
        pathfinder = LPAStar(gspace)
        model = StandardABM(Agent3, gspace; properties = (pf = pathfinder,), warn_deprecation = false)
        a = add_agent!((5, 2), model, 654.5)
        @test is_stationary(a, model.pf)

        plan_route!(a, (1, 3), model.pf)
        @test !is_stationary(a, model.pf)
        @test length(model.pf.agent_paths) == 1
        while !is_stationary(a, model.pf)
            move_along_route!(a, model, model.pf)
        end
        @test a.pos == (1, 3)

        @test plan_best_route!(a, [(5, 1), (1, 1), (3, 3)], model.pf) == (1, 1)
        remove_agent!(a, model, model.pf)
        @test length(model.pf.agent_paths) == 0
        @test isnothing(penaltymap(model.pf))
        @test all(model.pf.walkmap[random_walkable(model, model.pf)...] for _ in 1:10)
    end

    @testset "Agents API (ContinuousSpace)" begin
        cspace = ContinuousSpace((5., 5.))
        pathfinder = LPAStar(cspace; walkmap = trues(10, 10))
        @test pathfinder isa LPAStar{2,true,true,Float64}
        model = StandardABM(Agent6, cspace; properties = (pf = pathfinder,), warn_deprecation = false)
        a = add_agent!(SVector(0., 0.), model, SVector(0., 0.), 0.)
        @test is_stationary(a, model.pf)

        plan_route!(a, SVector(4., 4.), model.pf)
        @test !is_stationary(a, model.pf)
        move_along_route!(a, model, model.pf, 0.35355)
        @test all(isapprox.(a.pos, SVector(4.75, 4.75); atol = 0.0001))   # periodic: μικρότερη διαδρομή μέσω wraparound

        # ο LPA* και ο A* δίνουν την ίδια τελική θέση
        move_along_route!(a, model, model.pf, 20.)
        @test all(isapprox.(a.pos, SVector(4., 4.); atol = 0.0001))
    end

    @testset "SocialForce destinations: LPAStar and FlowField" begin
        cspace = ContinuousSpace((10., 10.); periodic = false)
        wm = trues(10, 10)
        wm[5, 1:6] .= false                                    # τοίχος με άνοιγμα στο πάνω μέρος
        dest = SVector(8.5, 1.5)
        for pf in (LPAStar(cspace; walkmap = wm), FlowField(cspace; walkmap = wm))
            model = StandardABM(Agent6, cspace; properties = (pf = pf,), warn_deprecation = false)
            sf = SocialForce(pf)
            a = add_agent!(SVector(1.5, 1.5), model, SVector(0., 0.), 0.)
            plan_route!(a, dest, pf)
            e = desired_direction(a, model, pf, sf)
            @test norm(e) ≈ 1                                  # μοναδιαία επιθυμητή κατεύθυνση (eq 1)
            @test e[2] > 0                                     # ο τοίχος εμποδίζει την ευθεία: πρώτα προς τα πάνω
            # το social_force_step! φέρνει τον agent πιο κοντά στον στόχο, μέσα από το άνοιγμα
            d0 = norm(a.pos - dest)
            for _ in 1:400
                social_force_step!(model, sf, pf, 0.05)
            end
            @test norm(a.pos - dest) < d0
        end
    end
end
