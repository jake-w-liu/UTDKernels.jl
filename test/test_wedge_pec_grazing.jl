using Test
using ForwardDiff
using UTDKernels

# Independent oracles:
#   four-term pec_wedge_DsDh at well-conditioned h
#   exact Ds(0)=0
#   linear Taylor remainder O(h^3) at h=1e-16
#   ForwardDiff of F_utd
#   n-face PEC mirror of pec_wedge_DsDh
# Tolerances are a few ULPs of the weaker evaluation, or the known order.

function _rel(a, b; floor=1e-300)
    return abs(a - b) / max(abs(b), floor)
end

const W = Wedge(1.5π)
const PHI = 1.7
const K = 100.0
const L = 1.0

@testset "four-term API is unchanged at well-conditioned offset" begin
    ang = RayAngles(PHI, 1e-2)
    four = pec_wedge_DsDh(W, ang, K, L)
    graz = pec_wedge_DsDh_grazing(W, ang, K, L; order=8)
    @test _rel(graz[1], four[1]) < 5e-13
    @test _rel(graz[2], four[2]) < 5e-13
end

@testset "exact grazing parity" begin
    ds, dh = pec_wedge_DsDh_grazing(W, RayAngles(PHI, 0.0), K, L)
    @test ds == 0
    G = two_term_kernel(PHI, W, K, L)
    C = UTDKernels.pec_wedge_prefactor(K, wedge_n(W))
    @test _rel(dh, 2 * C * G) < 2e-15
end

@testset "continuation retains tiny-h soft coefficient" begin
    h = 1e-16
    ang = RayAngles(PHI, h)
    ref = pec_wedge_Ds_linear(W, ang, K, L)
    graz = pec_wedge_DsDh_grazing(W, ang, K, L; order=8)[1]
    four = pec_wedge_DsDh(W, ang, K, L)[1]
    # Linear remainder is O(h^3); at h=1e-16 that is far below binary64.
    @test _rel(graz, ref) < 1e-12
    @test _rel(four, ref) > 0.9
    @test four == 0 || _rel(four, ref) > 0.9
end

@testset "F' matches ForwardDiff and rejects the conjugate DE" begin
    for x in (1e-4, 0.1, 1.0, 10.0, 100.0)
        @test _rel(F_utd_prime(x), ForwardDiff.derivative(F_utd, x)) < 5e-12
    end
    x = 1.0
    wrong = (-im + 0.5 / x) * F_utd(x) + im
    @test _rel(F_utd_prime(x), wrong) > 0.1
end

@testset "G' matches a central difference of G" begin
    δ = 1e-7
    fd = (
        -two_term_kernel(PHI + 2δ, W, K, L) +
        8 * two_term_kernel(PHI + δ, W, K, L) -
        8 * two_term_kernel(PHI - δ, W, K, L) +
        two_term_kernel(PHI - 2δ, W, K, L)
    ) / (12δ)
    @test _rel(two_term_kernel_derivative(PHI, W, K, L), fd) < 1e-8
end

@testset "n-face uses the PEC mirror of the o-face continuation" begin
    h = 1e-2
    ang_n = RayAngles(PHI, W.alpha - h)
    ang_m = RayAngles(W.alpha - PHI, h)
    four_n = pec_wedge_DsDh(W, ang_n, K, L)
    four_m = pec_wedge_DsDh(W, ang_m, K, L)
    @test _rel(four_n[1], four_m[1]) < 5e-13
    @test _rel(four_n[2], four_m[2]) < 5e-13
    graz_n = pec_wedge_DsDh_grazing(W, ang_n, K, L; face=:n, order=8)
    graz_m = pec_wedge_DsDh_grazing(W, ang_m, K, L; face=:o, order=8)
    @test _rel(graz_n[1], graz_m[1]) < 5e-13
    @test _rel(graz_n[2], graz_m[2]) < 5e-13
    @test grazing_local_angles(W, ang_n).face === :n
end

@testset "certificate rejects face exit, branch change, and hidden transition" begin
    bad = Wedge(1.5π)
    @test_throws GrazingDomainError pec_wedge_DsDh_grazing(
        bad, RayAngles(0.02, 0.03), K, L,
    )
    four = pec_wedge_DsDh(bad, RayAngles(0.02, 0.03), K, L)
    fb = pec_wedge_DsDh_grazing(bad, RayAngles(0.02, 0.03), K, L; on_fail=:four_term)
    @test fb == four

    geom_w = Wedge(1.2π)
    phi = 0.5 * geom_w.alpha
    found = nothing
    for h in range(0.05, min(phi, geom_w.alpha - phi) * 0.95; length=200)
        r = grazing_interval_report(geom_w, RayAngles(phi, h), 20.0, 1.0;
            transition_margin=0.0, x_margin=0.0)
        if occursin("branch changes", r.reason)
            found = h
            break
        end
    end
    @test found !== nothing
    @test_throws GrazingDomainError pec_wedge_DsDh_grazing(
        geom_w, RayAngles(phi, found), 20.0, 1.0;
        transition_margin=0.0, x_margin=0.0,
    )

    wt = Wedge(1.5π)
    r = grazing_interval_report(wt, RayAngles(π + 1.0e-3, 1.0e-2), 50.0, 1.0;
        transition_margin=0.0, x_margin=1.0e-14)
    @test !r.valid
    @test r.min_transition_argument == 0.0
end

@testset "unequal L, impedance, and interior wedge are refused" begin
    ang = RayAngles(PHI, 1e-2)
    @test_throws ArgumentError pec_wedge_DsDh_grazing(W, ang, K, 1.0, 2.0, 3.0)
    iw = ImpedanceWedge(W.alpha, WedgeFaceMaterial(4.0 + 0.0im))
    @test_throws ArgumentError pec_wedge_DsDh_grazing(iw, ang, K, L)
    interior = Wedge(0.7π)
    @test_throws GrazingDomainError pec_wedge_DsDh_grazing(
        interior, RayAngles(0.3π, 1e-3), K, L,
    )
    @test pec_wedge_DsDh_grazing(
        interior, RayAngles(0.3π, 1e-3), K, L; on_fail=:four_term,
    ) == pec_wedge_DsDh(interior, RayAngles(0.3π, 1e-3), K, L)
    @test_throws GrazingDomainError pec_wedge_DsDh_grazing(W, ang, K, Inf)
end

@testset "G' report is populated; tiny G' does not switch to four-term" begin
    r = grazing_interval_report(W, RayAngles(PHI, 1e-3), K, L)
    @test r.valid
    @test r.gprime_abs > 0
    @test r.degenerate_odd == false
    h = 1e-16
    graz = pec_wedge_DsDh_grazing(W, RayAngles(PHI, h), K, L)[1]
    @test graz != 0
end

@testset "AD through continuation vs finite difference; four-term AD remains" begin
    f_h(h) = abs(pec_wedge_DsDh_grazing(W, RayAngles(PHI, h), K, L; order=8)[1])
    h0 = 1e-3
    ad = ForwardDiff.derivative(f_h, h0)
    δ = 1e-8
    fd = (f_h(h0 + δ) - f_h(h0 - δ)) / (2δ)
    @test isfinite(ad)
    @test _rel(ad, fd) < 1e-5

    f_phi(phi) = abs(pec_wedge_DsDh(W, RayAngles(phi, 1e-2), K, L)[1])
    ad4 = ForwardDiff.derivative(f_phi, PHI)
    fd4 = (f_phi(PHI + 1e-7) - f_phi(PHI - 1e-7)) / 2e-7
    @test isfinite(ad4)
    @test _rel(ad4, fd4) < 1e-5
end
