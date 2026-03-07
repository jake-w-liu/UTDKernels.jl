using Test
using UTDKernels

@testset "KP regime classification" begin
    w = Wedge(2π)

    @testset "Lit/shadow classification follows sign of g_j" begin
        # All g_j > 0 => all terms lit, independent of kL scaling.
        ang_lit = RayAngles(0.2, 0.4)
        res_small = wedge_transition_args(w, ang_lit, 0.01, 0.01)
        res_large = wedge_transition_args(w, ang_lit, 100.0, 100.0)
        @test all(>(0), res_small.gj)
        @test res_small.regime == (:lit, :lit, :lit, :lit)
        @test res_large.regime == (:lit, :lit, :lit, :lit)

        # Mixed-sign g_j should map to mixed lit/shadow terms.
        ang_mixed = RayAngles(3.2, 0.2)
        res_mixed = wedge_transition_args(w, ang_mixed, 1.0, 1.0)
        @test res_mixed.regime == (:lit, :lit, :shadow, :shadow)
    end

    @testset "Transition detection at boundary" begin
        phip = 0.2
        phi = π + phip  # half-plane ISB for β⁻ terms
        res = wedge_transition_args(w, RayAngles(phi, phip), 1.0, 1.0)
        @test res.regime[1] == :transition
        @test res.regime[2] == :transition
        @test res.regime[3] == :shadow
        @test res.regime[4] == :shadow
    end
end
