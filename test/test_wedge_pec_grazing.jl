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

@testset "large-x transition series preserves Dual values and partials" begin
    x = 1e8
    xd = ForwardDiff.Dual(x, 1.0)
    dual_value(z) = complex(ForwardDiff.value(real(z)), ForwardDiff.value(imag(z)))
    dual_partial(z) = complex(ForwardDiff.partials(real(z))[1], ForwardDiff.partials(imag(z))[1])

    fm1d = F_utd_minus_one(xd)
    fpd = F_utd_prime(xd)
    @test _rel(dual_value(fm1d), F_utd_minus_one(x)) < 5e-15
    @test _rel(dual_value(fpd), F_utd_prime(x)) < 5e-15
    @test _rel(dual_partial(fm1d), F_utd_prime(x)) < 5e-15

    invx = inv(x)
    fpp = sum(
        m * (m + 1) * c * invx^(m + 2)
        for (m, c) in enumerate(UTDKernels.ASYM_F_COEFFS)
    )
    @test _rel(dual_partial(fpd), fpp) < 5e-15
end

@testset "Gauss-Legendre remainder constant" begin
    @test UTDKernels.gauss_legendre_error_constant(1) == 1 / 3
    @test UTDKernels.gauss_legendre_error_constant(2) == 1 / 135
    @test_throws ArgumentError UTDKernels.gauss_legendre_error_constant(0)
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

    # Exact n-face seam: wrap_angle(α, α)=0 must not be treated as o-face h=0.
    ang_seam = RayAngles(PHI, W.alpha)
    loc_seam = grazing_local_angles(W, ang_seam)
    @test loc_seam.face === :n
    @test loc_seam.h == 0
    four_seam = pec_wedge_DsDh(W, ang_seam, K, L)
    graz_seam = pec_wedge_DsDh_grazing(W, ang_seam, K, L)
    @test graz_seam[1] == 0
    @test _rel(graz_seam[2], four_seam[2]) < 5e-13
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

@testset "certificate validates wavenumber and safety parameters" begin
    ang = RayAngles(1.0, 1e-3)

    for k_bad in (Inf, NaN, 0.0, -1.0)
        report = grazing_interval_report(W, ang, k_bad, L)
        @test !report.valid
        @test occursin("finite positive wavenumber", report.reason)
        @test_throws DomainError pec_wedge_DsDh_grazing(W, ang, k_bad, L)
        @test_throws DomainError wedge_DsDh(W, ang, k_bad, L)
    end

    margin_cases = (
        ((transition_margin=NaN,), "transition_margin"),
        ((transition_margin=-1.0,), "transition_margin"),
        ((x_margin=NaN,), "x_margin"),
        ((x_margin=-1.0,), "x_margin"),
        ((branch_margin=NaN,), "branch_margin"),
        ((branch_margin=-1.0,), "branch_margin"),
        ((gprime_reltol=NaN,), "gprime_reltol"),
        ((gprime_reltol=-1.0,), "gprime_reltol"),
    )
    for (kwargs, name) in margin_cases
        report = grazing_interval_report(W, ang, K, L; kwargs...)
        @test !report.valid
        @test occursin(name, report.reason)
    end

    for kwargs in (
        (transition_margin=NaN,), (transition_margin=-1.0,),
        (x_margin=NaN,), (x_margin=-1.0,),
        (branch_margin=NaN,), (branch_margin=-1.0,),
    )
        @test_throws DomainError pec_wedge_DsDh_grazing(W, ang, K, L; kwargs...)
        @test_throws DomainError wedge_DsDh(W, ang, K, L; kwargs...)
    end

    # Zero margins disable only the positive safety buffer; they must still
    # reject an exact cotangent pole and zero transition argument.
    pole = grazing_interval_report(
        W,
        RayAngles(π + 1e-3, 1e-2),
        50.0,
        1.0;
        transition_margin=0.0,
        x_margin=0.0,
    )
    @test !pole.valid
    @test pole.min_abs_sin_psi == 0.0
    @test pole.min_transition_argument == 0.0
    @test occursin("cotangent pole", pole.reason)

    @test_throws ArgumentError pec_wedge_DsDh_grazing(W, ang, K, L; order=0)
    @test_throws ArgumentError wedge_DsDh(W, ang, K, L; order=0)
    for switch in (-1.0, NaN, Inf)
        @test_throws DomainError wedge_DsDh(W, ang, K, L; grazing_switch=switch)
    end
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

@testset "complex k or L is refused cleanly, not by an internal MethodError" begin
    ang = RayAngles(PHI, 1e-2)
    # The four-term baseline handles complex media; the certified continuation
    # does not. Every complex-typed request (including a zero-imaginary Complex)
    # must yield a typed refusal and honour on_fail, never a raw MethodError.
    for (k, Lc) in (
        (Complex(K, 0.0), L),      # complex-typed k, zero imaginary
        (K + 1.0im, L),            # complex k, nonzero imaginary
        (K, Complex(L, 0.0)),      # complex-typed L, zero imaginary
        (K, L + 0.2im),            # complex L, nonzero imaginary
    )
        r = grazing_interval_report(W, ang, k, Lc)
        @test r.valid == false
        @test_throws GrazingDomainError two_term_kernel_derivative(PHI, W, k, Lc)
        @test_throws GrazingDomainError pec_wedge_Ds_linear(W, ang, k, Lc)
        @test_throws GrazingDomainError pec_wedge_DsDh_grazing(W, ang, k, Lc)
        @test pec_wedge_DsDh_grazing(W, ang, k, Lc; on_fail=:four_term) ==
              pec_wedge_DsDh(W, ang, k, Lc)
        # check_domain=false must still route through on_fail, not crash the
        # real-argument soft integrand.
        @test_throws GrazingDomainError pec_wedge_DsDh_grazing(
            W, ang, k, Lc; check_domain=false,
        )
        @test pec_wedge_DsDh_grazing(
            W, ang, k, Lc; check_domain=false, on_fail=:four_term,
        ) == pec_wedge_DsDh(W, ang, k, Lc)
    end
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

@testset "wedge_DsDh auto-routes to the most accurate method (exterior)" begin
    # Far from grazing: returns the four-term result unchanged.
    ang_far = RayAngles(PHI, 0.5)
    @test wedge_DsDh(W, ang_far, K, L) == pec_wedge_DsDh(W, ang_far, K, L)
    # Well-conditioned near grazing: matches four-term.
    ang_wc = RayAngles(PHI, 1e-2)
    a = wedge_DsDh(W, ang_wc, K, L)
    b = pec_wedge_DsDh(W, ang_wc, K, L)
    @test _rel(a[1], b[1]) < 5e-13
    @test _rel(a[2], b[2]) < 5e-13
    # Tiny grazing offset: keeps the soft coefficient (vs the O(h^3) linear
    # reference) while the four-term loses it to cancellation.
    h = 1e-16
    ref = pec_wedge_Ds_linear(W, RayAngles(PHI, h), K, L)
    auto = wedge_DsDh(W, RayAngles(PHI, h), K, L)[1]
    four = pec_wedge_DsDh(W, RayAngles(PHI, h), K, L)[1]
    @test _rel(auto, ref) < 1e-10
    @test four == 0 || _rel(four, ref) > 0.9
end

@testset "wedge_DsDh keeps precision on interior wedges" begin
    Wi = Wedge(0.7π)
    phi = 0.35π
    a = wedge_DsDh(Wi, RayAngles(phi, 1e-2), K, L)
    b = pec_wedge_DsDh(Wi, RayAngles(phi, 1e-2), K, L)
    @test _rel(a[1], b[1]) < 5e-12
    @test _rel(a[2], b[2]) < 5e-12
    h = 1e-16
    ref = pec_wedge_Ds_linear(Wi, RayAngles(phi, h), K, L)
    auto = wedge_DsDh(Wi, RayAngles(phi, h), K, L)[1]
    four = pec_wedge_DsDh(Wi, RayAngles(phi, h), K, L)[1]
    @test _rel(auto, ref) < 1e-10
    @test four == 0 || _rel(four, ref) > 0.9
end

@testset "wedge_DsDh keeps precision in the plane-wave limit L = Inf" begin
    a = wedge_DsDh(W, RayAngles(PHI, 1e-2), K, Inf)
    b = pec_wedge_DsDh(W, RayAngles(PHI, 1e-2), K, Inf)
    @test _rel(a[1], b[1]) < 5e-12
    @test _rel(a[2], b[2]) < 5e-12
    h = 1e-16
    ref = pec_wedge_Ds_linear(W, RayAngles(PHI, h), K, Inf)
    auto = wedge_DsDh(W, RayAngles(PHI, h), K, Inf)[1]
    four = pec_wedge_DsDh(W, RayAngles(PHI, h), K, Inf)[1]
    @test _rel(auto, ref) < 1e-9
    @test four == 0 || _rel(four, ref) > 0.9
end

@testset "wedge_DsDh dispatches non-continuation regimes to the right evaluator" begin
    ang = RayAngles(PHI, 1e-3)
    # Unequal distances: the three-distance four-term form (no grazing cancellation).
    @test wedge_DsDh(W, ang, K, 1.0, 2.5, 0.7) == pec_wedge_DsDh(W, ang, K, 1.0, 2.5, 0.7)
    # Impedance wedge: the impedance evaluator.
    iw = ImpedanceWedge(W.alpha, WedgeFaceMaterial(4.0 + 0.0im))
    @test wedge_DsDh(iw, ang, K, L) == impedance_wedge_DsDh(iw, ang, K, L)
    # Complex media: the complex-capable four-term (never a crash).
    @test wedge_DsDh(W, ang, K + 1.0im, L) == pec_wedge_DsDh(W, ang, K + 1.0im, L)
end

@testset "wedge_DsDh is differentiable (finite and far-field)" begin
    for Lval in (L, Inf)
        f(h) = abs(wedge_DsDh(W, RayAngles(PHI, h), K, Lval)[1])
        h0 = 1e-3
        ad = ForwardDiff.derivative(f, h0)
        fd = (f(h0 + 1e-8) - f(h0 - 1e-8)) / 2e-8
        @test isfinite(ad)
        @test _rel(ad, fd) < 1e-5
    end
end

@testset "pec_wedge_DsDh_grazing opt-in interior and far-field continuation" begin
    # Interior wedges are refused by default but available via allow_interior.
    Wi = Wedge(0.7π)
    ang_i = RayAngles(0.35π, 1e-2)
    @test_throws GrazingDomainError pec_wedge_DsDh_grazing(Wi, ang_i, K, L)
    gi = pec_wedge_DsDh_grazing(Wi, ang_i, K, L; allow_interior=true)
    fi = pec_wedge_DsDh(Wi, ang_i, K, L)
    @test _rel(gi[1], fi[1]) < 5e-12
    @test _rel(gi[2], fi[2]) < 5e-12
    # L = Inf is refused by default but available via allow_infinite_L.
    ang_f = RayAngles(PHI, 1e-2)
    @test_throws GrazingDomainError pec_wedge_DsDh_grazing(W, ang_f, K, Inf)
    gf = pec_wedge_DsDh_grazing(W, ang_f, K, Inf; allow_infinite_L=true)
    ff = pec_wedge_DsDh(W, ang_f, K, Inf)
    @test _rel(gf[1], ff[1]) < 5e-12
    @test _rel(gf[2], ff[2]) < 5e-12
end

@testset "plane-wave limit uses the exact closed form, accurate near poles" begin
    # Ds_inf = C sum_sigma sin(sigma h/n) / [sin psi_sigma(phi-h) sin psi_sigma(phi+h)],
    # exact and cancellation-free; no quadrature and no wide pole fallback.
    # Tiny h: the closed form matches the O(h^3) linear limit while four-term collapses.
    h = 1e-16
    cf = wedge_DsDh(W, RayAngles(PHI, h), K, Inf)[1]
    ref = pec_wedge_Ds_linear(W, RayAngles(PHI, h), K, Inf)
    four = pec_wedge_DsDh(W, RayAngles(PHI, h), K, Inf)[1]
    @test _rel(cf, ref) < 1e-9
    @test four == 0 || _rel(four, ref) > 0.9
    # Near a cotangent pole (3e-3 away), where the finite-L quadrature would be
    # under-resolved, the closed form stays accurate at tiny h.
    n = 0.7
    Wp = Wedge(n * π)
    phi = (2n * π - π) - 3e-3   # psi_+ = pi pole at phi = 2 n pi - pi
    hn = 1e-10
    near = wedge_DsDh(Wp, RayAngles(phi, hn), K, Inf)[1]
    refn = pec_wedge_Ds_linear(Wp, RayAngles(phi, hn), K, Inf)
    @test isfinite(near) && near != 0
    @test _rel(near, refn) < 1e-6
    # An exact cotangent pole inside the interval is refused, not silently wrong.
    @test_throws GrazingDomainError pec_wedge_DsDh_grazing(
        Wp, RayAngles(2n * π - π, 1e-3), K, Inf; allow_infinite_L=true, transition_margin=1e-2,
    )
end
