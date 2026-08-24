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

        # The default tolerance and pole target follow the active angle type.
        w32 = Wedge(Float32(1.5π))
        phip32 = Float32(0.3)
        phi32 = Float32(π) + phip32
        terms32 = UTDKernels.kp_four_terms(phi32, phip32, wedge_n(w32))
        res32 = wedge_transition_args(w32, RayAngles(phi32, phip32), 2.0f0, 3.0f0)
        @test terms32.psi[2] == 0.0f0
        @test UTDKernels._kp_transition_detuning(2, terms32, wedge_n(w32)) == 0.0f0
        @test res32.regime[2] == :transition
    end
end
