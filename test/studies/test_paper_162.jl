module Paper162CanonicalTests
import UTDKernels
using Test
using Random
import UTDKernels.FaceEdge
const FaceEdgeUTD = UTDKernels.FaceEdge
import UTDKernels
using UTDKernels.FaceEdge: BoundaryGeometry, transition_F, transition_derivatives
using UTDKernels.FaceEdge: edge_residual_direct, edge_residual_first_order
using UTDKernels.FaceEdge: coefficient_four_term, normalized_raw_score
using UTDKernels.FaceEdge: geometric_defect_score, condition_number, angle_factor
using UTDKernels.FaceEdge: edge_residual_hp, transition_F_hp
using UTDKernels.FaceEdge: nint_kp, sinc_rad, BranchDomainError, kL, wedge_n, EIP4
using UTDKernels.FaceEdge: wedge_go_terms, wedge_pair_terms, wedge_coefficients
using UTDKernels.FaceEdge: wedge_total_field, wedge_pairs_hp
using UTDKernels.FaceEdge: incident_shadow_active
using UTDKernels.FaceEdge: INCIDENT_KLA_GUARD, INCIDENT_FIELD_TOL
using UTDKernels.FaceEdge: cylinder_scattered, cylinder_total, cylinder_radial_derivative
using UTDKernels.FaceEdge: regular_polygon, warped_polygon, vertex_wedge
using UTDKernels.FaceEdge: vertex_defect, vertex_probe_L, vertex_lit, vertex_visible
using UTDKernels.FaceEdge: polygon_go, polygon_total, edge_scores, select_modes
using SpecialFunctions: erfcx
using Statistics: median

central_wedge(g::BoundaryGeometry) = UTDKernels.Wedge(pi * (1 + g.eps))
central_split(g::BoundaryGeometry) = UTDKernels.pec_wedge_face_edge(
    central_wedge(g), g.Delta, g.k, g.L, g.eps,
)
edge_residual_compensated(g::BoundaryGeometry) = central_split(g).edge
face_transition(g::BoundaryGeometry) = UTDKernels.pec_wedge_face_edge(
    central_wedge(g), 0.0, g.k, g.L, g.eps,
).face
function edge_residual_symmetric(g::BoundaryGeometry)
    iszero(g.Delta) || throw(ArgumentError("symmetric formula requires Delta = 0"))
    return central_split(g).edge
end
function coefficients_decomposed(g::BoundaryGeometry)
    split = central_split(g)
    return split.Ds, split.Dh
end
normalized_intrinsic_score(g::BoundaryGeometry) = UTDKernels.pec_wedge_intrinsic_score(
    central_wedge(g), g.Delta, g.k, g.L, g.eps,
)

relerr(got, ref; floor=1e-300) = abs(got - ref) / max(abs(ref), floor)

const PI = π

@testset "FaceEdgeUTD" begin

    @testset "transition F vs high-precision and Python/scipy.wofz" begin
        # Python scipy.wofz(1j*z) ≡ erfcx(z); F(1) from face_edge_utd.py
        # VERIFIED: python transition_F(1) = 0.8095254817474091+0.2321993900552647j
        F1 = transition_F(1.0)
        @test relerr(F1, 0.8095254817474091 + 0.2321993900552647im) < 2e-15

        @test transition_F(0.0) == 0.0 + 0.0im
        @test transition_F(Inf) == 1.0 + 0.0im
        @test_throws DomainError transition_F(-1.0)

        for x in (1e-10, 1e-6, 0.01, 1.0, 20.0, 1e3, 1e6)
            ref = Complex{Float64}(transition_F_hp(x; dps=80))
            got = transition_F(x)
            @test relerr(got, ref) < 2e-12
        end

        # Faddeeva identity used by the Python kernel: wofz(i z) = erfcx(z).
        # VERIFIED by matching python transition_F at the fixtures above; a
        # conjugate-argument form must not reproduce F.
        z = EIP4 * sqrt(1.0)
        F_wrong_arg = sqrt(π) * EIP4 * erfcx(conj(z))
        @test relerr(transition_F(1.0), F_wrong_arg) > 0.1
        # Wrong-convention probe: conjugate DE must not match F'.
        f, fp, _, _ = transition_derivatives(1.0)
        wrong = (-im + 0.5) * f + im
        @test relerr(fp, wrong) > 0.1
    end

    @testset "transition derivatives vs high-precision F" begin
        for x in (0.02, 0.5, 5.0, 100.0)
            _, f1, f2, f3 = transition_derivatives(x)
            # Independent F' oracle: 4th-order CD of BigFloat F (not the ODE).
            xb = BigFloat(x)
            h = BigFloat("1e-8") * xb
            Fp = (transition_F_hp(xb - 2h) - 8 * transition_F_hp(xb - h) +
                  8 * transition_F_hp(xb + h) - transition_F_hp(xb + 2h)) / (12h)
            # The BigFloat CD is accurate to ~1e-32 here; the binary64 ODE value
            # loses ~u|F|/|F'| to cancellation (1.5e-12 measured at x=100), so
            # 2e-9 is a loose bound. A sign or convention error in F' is O(1).
            @test relerr(f1, Complex{Float64}(Fp)) < 2e-9
            # The ODE is stable only below the parent's asymptotic switch.
            # At larger x independent high-precision differences validate the
            # derivatives without requiring a cancellation-prone evaluation.
            if x < 35
                gcoef = im + 0.5 / x
                @test relerr(f2, gcoef * f1 - 0.5 * transition_F(x) / x^2) < 1e-12
                @test relerr(f3, gcoef * f2 - f1 / x^2 + transition_F(x) / x^3) < 1e-12
            else
                hp = offset -> transition_F_hp(xb + offset*h; dps=100)
                second = (-hp(2)+16hp(1)-30hp(0)+16hp(-1)-hp(-2))/(12h^2)
                third = (hp(-3)-8hp(-2)+13hp(-1)-13hp(1)+8hp(2)-hp(3))/(8h^3)
                @test relerr(f2,second) < 2e-9
                @test relerr(f3,third) < 2e-8
            end
        end
        # Hardcoded Python transition_derivatives(1)
        f, fp, fpp, fppp = transition_derivatives(1.0)
        @test relerr(f, 0.8095254817474091 + 0.23219939005526471im) < 2e-15
        @test relerr(fp, 0.17256335081843985 - 0.07437482322495848im) < 2e-14
        @test relerr(fpp, -0.24410624223952615 + 0.019276244178328253im) < 2e-14
        @test relerr(fppp, 0.495632765630878 + 0.07210609312986116im) < 2e-13
    end

    @testset "geometry and branch contract" begin
        g = BoundaryGeometry(0.01, 0.5π, 20.0, 1.0)
        @test wedge_n(g) ≈ 1.01
        @test kL(g) ≈ 20.0
        @test_throws ArgumentError BoundaryGeometry(-1e-9, 0.0, 1.0, 1.0)
        @test_throws ArgumentError BoundaryGeometry(0.0, π, 1.0, 1.0)
        @test_throws ArgumentError BoundaryGeometry(0.0, 0.0, 0.0, 1.0)
        @test_throws ArgumentError BoundaryGeometry(0.0, 0.0, 1e-200, 1e-200)
        @test_throws ArgumentError BoundaryGeometry(0.0, 0.0, 1e200, 1e200)
        @test_throws BranchDomainError edge_residual_compensated(BoundaryGeometry(0.2, 0.1π, 10.0, 1.0))
        @test_throws BranchDomainError edge_residual_direct(BoundaryGeometry(0.2, 0.1π, 10.0, 1.0))
        @test nint_kp(0.5) == 1
        @test nint_kp(-0.5) == -1
        @test nint_kp(2.0) == 2
        @test abs(sinc_rad(0.0) - 1.0) < 1e-16
        @test relerr(sinc_rad(1e-5), sin(1e-5) / 1e-5) < 1e-14
    end

    @testset "decomposition identities and four-term" begin
        rng = MersenneTwister(20260814)
        for _ in 1:250
            epsv = 10^(rand(rng) * (log10(0.2) - (-7)) + (-7))
            dfrac = rand(rng) * (0.94 - min(0.88, epsv + 0.04)) + min(0.88, epsv + 0.04)
            kLv = 10^(rand(rng) * 6 - 3)
            L = 10^(rand(rng) * 2 - 1)
            g = BoundaryGeometry(epsv, dfrac * PI, kLv / L, L)
            ds, dh = coefficients_decomposed(g)
            e = edge_residual_compensated(g)
            f = face_transition(g)
            @test abs((dh + ds) / 2 - e) < 3e-15
            @test abs((dh - ds) / 2 - f) < 3e-15
            ds0 = coefficient_four_term(g, :soft)
            dh0 = coefficient_four_term(g, :hard)
            @test relerr(ds, ds0; floor=1e-12) < 3e-8
            @test relerr(dh, dh0; floor=1e-12) < 3e-8
        end
    end

    @testset "face coplanar limit and symmetric formula" begin
        for kLv in (1e-3, 1.0, 1e3)
            @test face_transition(BoundaryGeometry(0.0, 0.4π, kLv, 1.0)) == 1.0 + 0.0im
        end
        @test face_transition(BoundaryGeometry(0.0, 0.4π, 1.0, 1.0)) == 1.0 + 0.0im
        # Symmetric closed form against the independent BigFloat incident pair
        # (both production names route to the same central call, so comparing
        # them with each other would be tautological). The production path has
        # no cancellation: it agrees with an independent quadrature oracle to
        # <= 5e-16 relative here, so 1e-14 (~45 ulp) leaves margin for erfcx
        # differences while still rejecting the Delta>0 branch formula used at
        # Delta=0 (>= 7e-11 off) and a binary64 cot(pi/(2n)) at eps <= 1e-6.
        for epsv in (1e-12, 1e-6, 0.01, 0.3)
            g = BoundaryGeometry(epsv, 0.0, 50.0, 1.2)
            ref = Complex{Float64}(edge_residual_hp(g; dps=80))
            @test relerr(FaceEdgeUTD.edge_residual_symmetric(g), ref) < 1e-14
        end
        gs = BoundaryGeometry(1e-6, 0.0, 50.0, 1.2)
        for delta in (nextfloat(0.0), eps(Float64), 16eps(Float64))
            @test_throws ArgumentError edge_residual_symmetric(
                BoundaryGeometry(1e-6, delta, 50.0, 1.2),
            )
        end
        # Independent 120-digit symmetric residual. The older Python fixture
        # preceded the retained-defect cofunction repair and was inaccurate by
        # 7.38e-11 relative at this small wedge defect.
        @test relerr(
            edge_residual_symmetric(gs),
            -6.292335834947062e-8 + 6.240128101465637e-8im,
        ) < 2e-14
    end

    @testset "Python kernel fixtures" begin
        g = BoundaryGeometry(1e-12, 0.8π, 100.0, 1.0)
        @test relerr(edge_residual_compensated(g), -4.931717647159758e-13 + 4.2642976963138624e-13im) < 2e-14
        @test relerr(face_transition(g), 0.9999999999822753 - 1.7724538508561682e-11im) < 2e-14
        @test relerr(normalized_intrinsic_score(g), 1.6342384719769234e-11) < 2e-14
        @test relerr(normalized_raw_score(g), 25.066282745878073) < 2e-14
        ds, dh = coefficients_decomposed(g)
        @test relerr(ds, -0.9999999999827685 + 1.8150968278193067e-11im) < 2e-14
        @test relerr(dh, 0.9999999999817821 - 1.7298108738930297e-11im) < 2e-14

        g2 = BoundaryGeometry(1e-8, 0.5π, 20.0, 1.0)
        @test relerr(edge_residual_compensated(g2), -2.0682164625107207e-9 + 1.873568346121566e-9im) < 2e-14

        g3 = BoundaryGeometry(0.01, 0.5π, 10.0, 1.0)
        @test relerr(coefficient_four_term(g3, :soft), -0.9469100729382122 + 0.053702088693178336im) < 2e-14
        @test relerr(coefficient_four_term(g3, :hard), 0.9411099506539375 - 0.04887990186827505im) < 2e-14
    end

    @testset "high-precision residual and cancellation" begin
        precision_probe = edge_residual_hp(
            BoundaryGeometry(1e-10, 0.5π, 100.0, 1.0); dps=80,
        )
        @test precision_probe isa Complex{BigFloat}
        @test precision(real(precision_probe)) >= ceil(Int, 80log2(10))
        @test_throws ArgumentError transition_F_hp(1.0; dps=0)
        @test_throws DomainError transition_F_hp(-1.0; dps=80)

        ref80 = edge_residual_hp(
            BoundaryGeometry(1e-10, 0.5π, 100.0, 1.0); dps=80,
        )
        ref100 = edge_residual_hp(
            BoundaryGeometry(1e-10, 0.5π, 100.0, 1.0); dps=100,
        )
        @test abs(ref80 - ref100) / abs(ref100) < BigFloat("1e-70")

        for d in (0.2, 0.5, 0.8)
            for e in (1e-8, 1e-10, 1e-12, 1e-14)
                g = BoundaryGeometry(e, d * PI, 100.0, 1.0)
                ref = edge_residual_hp(g; dps=80)
                @test relerr(edge_residual_compensated(g), ref) < 2e-11
            end
        end
        g = BoundaryGeometry(1e-12, 0.8π, 100.0, 1.0)
        ref = edge_residual_hp(g; dps=80)
        en = relerr(edge_residual_direct(g), ref)
        ec = relerr(edge_residual_compensated(g), ref)
        @test en > 1e-5
        @test ec < 2e-11
        @test en / ec > 1e5

        # Near Delta=pi, x0 can be tiny or underflow to zero while x0+dx
        # remains representable. This must use the nonanalytic square-root branch,
        # not derivatives at x0.
        for kval in (1e-10, 1e-300)
            near_endpoint = BoundaryGeometry(
                1e-6, prevfloat(Float64(π)), kval, 1.0,
            )
            got = edge_residual_compensated(near_endpoint)
            hp = edge_residual_hp(near_endpoint; dps=100)
            @test isfinite(got)
            @test relerr(got, hp) < 2e-8
        end
    end

    @testset "first-order remainder slope" begin
        es = [1e-2, 5e-3, 2.5e-3, 1.25e-3]
        errs = Float64[]
        for e in es
            g = BoundaryGeometry(e, 0.5π, 20.0, 1.0)
            push!(errs, abs(edge_residual_compensated(g) - edge_residual_first_order(g)))
        end
        slopes = [log(errs[j] / errs[j + 1]) / log(2) for j in 1:3]
        @test minimum(slopes) > 1.85
    end

    @testset "polygon decay and angle factor" begin
        vals = [normalized_intrinsic_score(BoundaryGeometry(2 / N, 0.0, 50.0, 1.0)) for N in (256, 512, 1024, 2048)]
        # Python: 0.012082165266092685 at N=256 and 0.006087961316361315 at N=512
        @test relerr(vals[1], 0.012082165266092685) < 2e-14
        @test relerr(vals[2], 0.006087961316361315) < 2e-14
        for j in 1:3
            @test abs(vals[j] / vals[j + 1] - 2) < 0.02
        end
        @test angle_factor(2.0) ≈ 0.5
        @test_throws DomainError angle_factor(NaN)
        @test_throws DomainError angle_factor(Inf)
        ns = range(1.000001, 2.0; length=80)
        gs = angle_factor.(ns)
        @test all(gs[i + 1] > gs[i] for i in 1:length(gs)-1)
        # stable form: the supplied defect must not be recovered by
        # subtracting rounded angles near pi/2
        e = 1e-15
        @test angle_factor(1 + e, e) ≈ π * e / 2 rtol = 1e-10
        @test abs(angle_factor(1 + e, e) - cot(π / (2(1 + e)))) <
              0.2 * π * e
    end

    @testset "condition-number identity" begin
        rows_eps = 10 .^ range(-15, -6; length=12)
        kappas = Float64[]
        for e in rows_eps
            g = BoundaryGeometry(e, 0.5π, 100.0, 1.0)
            push!(kappas, condition_number(g))
        end
        lx = log10.(rows_eps)
        ly = log10.(kappas)
        xbar = sum(lx) / length(lx)
        ybar = sum(ly) / length(ly)
        slope = sum((lx .- xbar) .* (ly .- ybar)) / sum((lx .- xbar) .^ 2)
        @test abs(slope + 1) < 0.02
        # Direct error should track u κ, compensated should not.
        g = BoundaryGeometry(1e-12, 0.5π, 100.0, 1.0)
        ref = edge_residual_hp(g; dps=80)
        @test relerr(edge_residual_direct(g), ref) > 1e-5
        @test relerr(edge_residual_compensated(g), ref) < 2e-11
    end

    @testset "soft/hard coplanar pairing" begin
        g = BoundaryGeometry(0.0, 0.0, 50.0, 2.25)
        ds, dh = coefficients_decomposed(g)
        @test ds == -sqrt(2.25) + 0.0im
        @test dh == sqrt(2.25) + 0.0im
        @test edge_residual_symmetric(g) == 0.0 + 0.0im
        @test edge_residual_compensated(g) == 0.0 + 0.0im
    end

    @testset "released adapter delegates to UTDKernels" begin
        g = BoundaryGeometry(1e-12, 0.8pi, 100.0, 1.0)
        split = central_split(g)
        @test FaceEdgeUTD.edge_residual_compensated(g) == split.edge
        @test FaceEdgeUTD.face_transition(g) == split.face
        @test FaceEdgeUTD.coefficients_decomposed(g) == (split.Ds, split.Dh)
        @test FaceEdgeUTD.normalized_intrinsic_score(g) ==
              UTDKernels.pec_wedge_intrinsic_score(
                  central_wedge(g), g.Delta, g.k, g.L, g.eps,
              )
    end

    @testset "pruning counterexample regression" begin
        # Independent 90-digit reference values (see paper text): a small
        # score does NOT imply a small deleted field at the slice midpoint.
        e = 1e-6
        n = 1 + e
        g = BoundaryGeometry(e, 0.0, 1.0, 100.0)
        split = central_split(g)
        @test abs(normalized_intrinsic_score(g) - 1.5707686471678075e-6) <
              1e-18
        @test abs(split.face / 10 - (0.9999822754602631 - 1.7724045034637091e-5im)) < 1e-14
        @test abs(abs(split.face / 10) - 0.9999822756173368) < 1e-14
        @test abs(split.Ds / 10 - (-0.9999823198814579 + 1.7768244693869998e-5im)) < 1e-14
        @test abs(split.Dh / 10 - (0.9999822310390683 - 1.7679845375404184e-5im)) < 1e-14
        # Both GO reflections are absent at the midpoint (negative specular
        # coordinate on each face continuation); only the incident ray exists.
        phim = phipm = n * pi / 2
        for pol in (:soft, :hard)
            ui, ur0, urn = wedge_go_terms(phim, phipm, n, 1.0, 100.0, pol)
            @test ur0 == 0.0 + 0.0im
            @test urn == 0.0 + 0.0im
            @test abs(ui) == 1.0
            uf = wedge_total_field(phim, phipm, n, 1.0, 100.0, pol; mode=:full)
            ud = wedge_total_field(phim, phipm, n, 1.0, 100.0, pol;
                                   mode=:delete)
            up = wedge_total_field(phim, phipm, n, 1.0, 100.0, pol;
                                   mode=:preserve)
            # Naive deletion loses the order-one reflection-transition field;
            # retaining the reflection pair bounds the loss by the residual.
            @test abs(uf - ud) > 0.99
            @test abs(uf - up) < 1e-6
        end
        # The same bound holds across the exterior for this geometry.
        for phi in range(1e-4, n * pi - 1e-4, length=800)
            uf = wedge_total_field(phi, phipm, n, 1.0, 100.0, :hard; mode=:full)
            up = wedge_total_field(phi, phipm, n, 1.0, 100.0, :hard;
                                   mode=:preserve)
            @test abs(uf - up) < 2e-6
        end
    end

    @testset "total-field continuity across boundaries" begin
        # The unpruned GO+UTD total field must be continuous through both
        # reflection boundaries and the incident-shadow boundary: paired
        # one-sided values differ by O(offset), not O(1).
        n2 = 1.01
        phip2 = n2 * pi / 2
        bounds = (pi - phip2, (2n2 - 1) * pi - phip2)
        for b in bounds, pol in (:soft, :hard)
            off = 1e-6
            um = wedge_total_field(b - off, phip2, n2, 1.0, 100.0, pol;
                                   mode=:full)
            up = wedge_total_field(b + off, phip2, n2, 1.0, 100.0, pol;
                                   mode=:full)
            @test abs(up - um) < 2e-5
        end
        # Incident shadow boundary at phi = phip + pi (reachable for
        # phip < pi*eps; here eps = 0.5, phip = 0.2pi).
        for pol in (:soft, :hard)
            off = 1e-6
            um = wedge_total_field(1.2pi - off, 0.2pi, 1.5, 1.0, 100.0, pol;
                                   mode=:full)
            up = wedge_total_field(1.2pi + off, 0.2pi, 1.5, 1.0, 100.0, pol;
                                   mode=:full)
            @test abs(up - um) < 2e-5
        end
        # On-slice identity: general evaluator matches the four-term slice
        # reference, and the reflection pair collapses to the face term.
        rng2 = MersenneTwister(4242)
        for _ in 1:40
            e = 10^(rand(rng2) * 3 - 4)
            dfrac = rand(rng2) * 0.8 + 0.1
            e >= dfrac && continue
            kLv = 10^(rand(rng2) * 4 - 2)
            L = 10^(rand(rng2) * 2 - 1)
            g = BoundaryGeometry(e, dfrac * PI, kLv / L, L)
            nn = 1 + e
            phi = (nn * PI + g.Delta) / 2
            phip = (nn * PI - g.Delta) / 2
            c = wedge_coefficients(phi, phip, nn, g.k, L)
            @test abs(c.Ds - coefficient_four_term(g, :soft)) /
                  abs(coefficient_four_term(g, :soft)) < 5e-10
            @test abs(c.Dh - coefficient_four_term(g, :hard)) /
                  abs(coefficient_four_term(g, :hard)) < 5e-10
            @test abs(c.Df - face_transition(g)) /
                  max(abs(face_transition(g)), 1e-14) < 1e-10
        end
    end

    @testset "cylinder series boundary conditions" begin
        # The PEC eigenfunction series must annul the total field (soft) or
        # its radial derivative (hard) on rho = a.
        k = 2pi
        a = 1.0
        beta = 0.3
        for _ in 1:30
            phi = 2pi * rand()
            @test abs(cylinder_total(k, a, a, phi, beta, :soft)) < 5e-12
            @test abs(cylinder_radial_derivative(k, a, a, phi, beta, :hard)) /
                  k < 5e-12
        end
    end

    @testset "polygon geometry and GO" begin
        sq = regular_polygon(4, 1.0)
        w = vertex_wedge(sq, 1)
        @test abs(w.n - 1.5) < 1e-15
        @test abs(vertex_defect(sq, 1) - 0.5) < 1e-15
        # Observer visibility uses atan(dy, dx), and is invariant under a
        # common rigid motion of polygon and observer. These points stay
        # away from the wedge faces, so no boundary convention is involved.
        probes = ((1, (-2.0, 0.2), false), (1, (0.2, -2.0), true),
                  (2, (-2.0, 0.2), true), (2, (0.2, -2.0), false),
                  (3, (2.0, 0.2), false), (3, (0.2, 2.0), true),
                  (4, (2.0, 0.2), true), (4, (0.2, 2.0), false))
        for theta in (0.0, 0.31, -1.2), shift in ((0.0, 0.0), (3.2, -0.7))
            move(q) = (cos(theta) * q[1] - sin(theta) * q[2] + shift[1],
                       sin(theta) * q[1] + cos(theta) * q[2] + shift[2])
            moved = FaceEdgeUTD.PolygonScatterer(move.(sq.verts))
            for (j, point, expected) in probes
                @test vertex_visible(moved, j, move(point)) == expected
            end
        end
        # Warped polygon defect: eps_j = sin(pi*(j-1)/N)*sin(pi/N) for
        # 2 <= j <= N, and eps_1 = 2 sin^2(pi/(2N)) at the wrap vertex.
        N = 64
        wp = warped_polygon(N, 1.0)
        @test abs(vertex_defect(wp, 1) - 2sin(pi / (2N))^2) < 1e-12
        for j in (2, 8, N ÷ 2, N - 1)
            @test abs(vertex_defect(wp, j) - sin(pi * (j - 1) / N) *
                      sin(pi / N)) < 1e-12
        end
        # Convexity: all vertex wedges stay in (1, 2).
        for j in 1:N
            @test 1 < vertex_wedge(wp, j).n < 2
        end
        # Lit/visible vertex counts for a regular polygon under one-sided
        # illumination match the lit semicircle plus grazing vertices.
        k = 2pi
        beta = pi
        nlit = count(vertex_lit(regular_polygon(128, 1.0), j, beta)
                     for j in 1:128)
        @test nlit in 64:66
        # A point in the shadowed half-plane is blocked; a lit point is not.
        p128 = regular_polygon(128, 1.0)
        @test FaceEdgeUTD.incident_blocked(p128, (5.0, 0.0), (-1.0, 0.0))
        @test !FaceEdgeUTD.incident_blocked(p128, (-5.0, 0.0), (-1.0, 0.0))
        # Geometric comparator equals sin^2(pi/N) (stable form).
        gN = BoundaryGeometry(2 / 512, 0.0, 50.0, 1.0)
        @test geometric_defect_score(gN) ≈ sin(pi / 512)^2
        @test geometric_defect_score(gN) != (1 - cos(pi * gN.eps)) / 2 ||
              abs(geometric_defect_score(gN) - (1 - cos(pi * gN.eps)) / 2) /
              geometric_defect_score(gN) < 1e-14
    end

    @testset "multi-edge preserve mechanism" begin
        # On the warped polygon the flattest seams carry a reflection-pair
        # field of order unity; deleting them entirely loses that field,
        # while :preserve keeps the transition.
        k = 2pi
        beta = pi
        wp = warped_polygon(128, 1.0)
        jflat = argmin(vertex_defect(wp, j) for j in 1:128)
        w = vertex_wedge(wp, jflat)
        phip = FaceEdgeUTD._local_angle(beta, w.a0)
        # Probe the vertex's own gap center: phi such that S = n*pi.
        phi_mid = w.n * pi - phip
        s = 50.0
        for pol in (:soft, :hard)
            uf = wedge_total_field(phi_mid, phip, w.n, k, s, pol; mode=:full)
            ud = wedge_total_field(phi_mid, phip, w.n, k, s, pol;
                                   mode=:delete)
            up = wedge_total_field(phi_mid, phip, w.n, k, s, pol;
                                   mode=:preserve)
            @test abs(uf - ud) > 0.9
            @test abs(uf - up) < 1e-4
        end
        # Scores: flattest vertex must score far below the steepest.
        scores_i = edge_scores(wp, k, :intrinsic)
        scores_a = edge_scores(wp, k, :angle)
        @test scores_i[jflat] < 0.01 * maximum(scores_i)
        @test argmin(scores_a) == jflat
        modes = select_modes(scores_i, median(scores_i))
        @test modes[jflat] == :preserve
    end

    @testset "incident-shadow counterexample (review geometry)" begin
        # Independent reproduction, 90 digits: eps=1e-6, n=1+eps,
        # phip=pi*eps/2, phi=pi+3*pi*eps/4, k=1, L=s=100. The symmetric
        # midpoint score is ~1.57e-6 while the incident pair that
        # reflection-only preservation drops carries ~0.99998 of unit
        # incident field: |Di|/sqrt(L) = 0.9999822756166478.
        e = BigFloat("1e-6")
        nb = 1 + e
        pib = BigFloat(π)
        phipb = pib * e / 2
        phib = pib + 3pib * e / 4
        Di, Dr = wedge_pairs_hp(phib, phipb, nb, BigFloat(1),
                                BigFloat(100); dps=90)
        @test abs(abs(Di) / sqrt(BigFloat(100)) -
                  BigFloat("0.9999822756166478")) < BigFloat("1e-15")
        # Float64 input angles lose ~1e-10 here (phi is adjacent to pi).
        inc64, _ = wedge_pair_terms(
            Float64(pi + 3pi * 1e-6 / 4), Float64(pi * 1e-6 / 2),
            1 + 1e-6, 1.0, 100.0,
        )
        Di64 = -cis(-pi / 4) * inc64 / (2(1 + 1e-6) * sqrt(2pi))
        @test abs(abs(Di64) / 10 - 0.9999822756166478) < 5e-9

        # One-sided incident-boundary limits, hard polarization:
        # delta = pi + t*pi*eps with t = -/+1e-25. The complete field is
        # continuous (< 2e-34); reflection-only preservation jumps by one.
        h = BigFloat("1e-25")
        function hard_fields_hp(t)
            delta = pib + t * pib * e
            Di2, Dr2 = wedge_pairs_hp(phipb + delta, phipb, nb,
                                      BigFloat(1), BigFloat(100); dps=90)
            go = t < 0 ? cis(BigFloat(100) * cos(delta)) :
                         zero(Complex{BigFloat})
            spread = cis(-BigFloat(100)) / sqrt(BigFloat(100))
            return go + (Di2 + Dr2) * spread, go + Dr2 * spread
        end
        full_l, keep_l = hard_fields_hp(-h)
        full_r, keep_r = hard_fields_hp(h)
        @test abs(full_r - full_l) < BigFloat("2e-34")
        @test abs(abs(keep_r - keep_l) - 1) < BigFloat("1e-10")
        # Hard complete/preserved magnitudes at the counterexample point.
        @test abs(abs(full_r) - BigFloat("1.9999645512")) <
              BigFloat("1e-8")
        @test abs(abs(keep_r) - BigFloat("0.9999822756")) <
              BigFloat("1e-8")
    end

    @testset "incident-shadow safeguard behavior" begin
        e = 1e-6
        n = 1 + e
        phip = pi * e / 2
        phi_cex = pi + 3pi * e / 4
        # The reviewer geometry: guard must fire (kLa_min ~ 4e-9 and
        # |Di|/sqrt(s) ~ 1), so safeguarded preserve equals full while
        # reflection-only preservation loses ~1 unit of field, both pols.
        c = wedge_coefficients(phi_cex, phip, n, 1.0, 100.0)
        @test incident_shadow_active(phi_cex - phip, n, 1.0, 100.0, c.De)
        for pol in (:soft, :hard)
            uf = wedge_total_field(phi_cex, phip, n, 1.0, 100.0, pol;
                                   mode=:full)
            up = wedge_total_field(phi_cex, phip, n, 1.0, 100.0, pol;
                                   mode=:preserve)
            ur = wedge_total_field(phi_cex, phip, n, 1.0, 100.0, pol;
                                   mode=:preserve_reflection)
            ud = wedge_total_field(phi_cex, phip, n, 1.0, 100.0, pol;
                                   mode=:delete)
            @test abs(uf - up) < 1e-14
            # Reflection-only preservation drops Di ~ sqrt(L): hard loses
            # ~1 of its 2.0 field; soft injects ~1 into a ~3e-10 field.
            @test abs(uf - ur) > 0.99
            # Deletion error equals the dropped coefficient, by definition.
            @test abs(uf - ud) ≈
                  abs(pol === :soft ? c.Ds : c.Dh) / 10
        end
        # One-sided incident-boundary limits, both polarizations:
        # phi = phip + pi +/- h stays inside the exterior wedge for
        # h < pi*eps/2. Full and safeguarded preserve stay continuous;
        # reflection-only preservation jumps by ~1.
        for pol in (:soft, :hard), h in (1e-6, 1e-7)
            um_f = wedge_total_field(phip + pi - h, phip, n, 1.0, 100.0,
                                     pol; mode=:full)
            up_f = wedge_total_field(phip + pi + h, phip, n, 1.0, 100.0,
                                     pol; mode=:full)
            um_p = wedge_total_field(phip + pi - h, phip, n, 1.0, 100.0,
                                     pol; mode=:preserve)
            up_p = wedge_total_field(phip + pi + h, phip, n, 1.0, 100.0,
                                     pol; mode=:preserve)
            um_r = wedge_total_field(phip + pi - h, phip, n, 1.0, 100.0,
                                     pol; mode=:preserve_reflection)
            up_r = wedge_total_field(phip + pi + h, phip, n, 1.0, 100.0,
                                     pol; mode=:preserve_reflection)
            @test abs(up_f - um_f) < 200h + 1e-12
            @test abs(up_p - um_p) ≈ abs(up_f - um_f) atol = 1e-12
            @test abs(up_r - um_r) > 0.9
        end
        # Reflection-boundary one-sided limits are continuous in every
        # preserve mode: the reflection pair is always retained. Use the
        # midpoint-source geometry where both boundaries lie inside the
        # exterior wedge.
        n2 = 1.01
        phip2 = n2 * pi / 2
        for pol in (:soft, :hard), b in (pi - phip2, (2n2 - 1) * pi - phip2)
            h = 1e-6
            for m in (:preserve, :preserve_reflection)
                um = wedge_total_field(b - h, phip2, n2, 1.0, 100.0, pol;
                                       mode=m)
                up = wedge_total_field(b + h, phip2, n2, 1.0, 100.0, pol;
                                       mode=m)
                @test abs(up - um) < 5e-3
            end
        end
        # Declared-domain check: wherever the guard is inactive in this
        # dense exterior sweep, the omitted incident field obeys the
        # declared tolerance; wherever it is active the pair is retained.
        for pol in (:soft, :hard)
            for phi in range(1e-4, n * pi - 1e-4, length=2000)
                cc = wedge_coefficients(phi, phip, n, 1.0, 100.0)
                active = incident_shadow_active(phi - phip, n, 1.0,
                                                100.0, cc.De)
                if !active
                    @test abs(cc.De) / 10 <= INCIDENT_FIELD_TOL
                    uf = wedge_total_field(phi, phip, n, 1.0, 100.0, pol;
                                           mode=:full)
                    up = wedge_total_field(phi, phip, n, 1.0, 100.0, pol;
                                           mode=:preserve)
                    @test abs(uf - up) <= INCIDENT_FIELD_TOL * (1 + 1e-9)
                end
            end
        end
        # Grazing incidence: source bearing adjacent to a face leaves the
        # incident boundary inside the wedge; guard stays active across it.
        for pol in (:soft, :hard)
            for phi in range(phip + pi - 0.2, phip + pi - 1e-6,
                             length=40)
                uf = wedge_total_field(phi, phip, n, 1.0, 100.0, pol;
                                       mode=:full)
                up = wedge_total_field(phi, phip, n, 1.0, 100.0, pol;
                                       mode=:preserve)
                @test abs(uf - up) <= max(INCIDENT_FIELD_TOL, 1e-12) *
                                     (1 + 1e-9)
            end
        end
    end

    @testset "switching-threshold and branch-boundary coverage" begin
        # Both sides of the smooth small-argument crossover x0 = 16
        # (sqrt_x0 = 4): x0 = 2*kL*cos^2(Delta/2).
        for x0 in (15.9, 16.1)
            d = 2acos(sqrt(x0 / 200))
            g = BoundaryGeometry(1e-4, d, 100.0, 1.0)
            ref = edge_residual_hp(g; dps=80)
            @test relerr(edge_residual_compensated(g), ref) < 2e-10
        end
        # Both sides of the |da|/|a0| = 1e-3 direct-difference gate:
        # ratio ~ 2*pi*eps*tan(Delta/2); at Delta = pi/2 the crossover is
        # eps ~ 1.6e-4.
        for e in (1e-4, 2e-4)
            g = BoundaryGeometry(e, 0.5PI, 100.0, 1.0)
            ref = edge_residual_hp(g; dps=80)
            @test relerr(edge_residual_compensated(g), ref) < 2e-10
        end
        # Endpoint branch: |t0|,|t1| <= 0.01 with x1 <= 16 (kL small) and
        # the general branch (kL large), Delta adjacent to pi.
        for kLv in (100.0, 1e5)
            g = BoundaryGeometry(1e-4, PI - 5e-3, kLv, 1.0)
            ref = edge_residual_hp(g; dps=80)
            @test relerr(edge_residual_compensated(g), ref) < 2e-8
        end
        # Nearest-pole branch boundary Delta = pi*eps: the branch requires
        # Delta > pi*eps; just above, the compensated residual matches the
        # unrestricted four-term half-sum; at/below, it throws.
        for e in (1e-3, 1e-4)
            above = BoundaryGeometry(e, e * PI * 1.001, 50.0, 1.0)
            ds, dh = coefficients_decomposed(above)
            @test relerr(edge_residual_compensated(above),
                         (ds + dh) / 2; floor=1e-14) < 5e-10
            below = BoundaryGeometry(e, e * PI * 0.999, 50.0, 1.0)
            @test_throws BranchDomainError edge_residual_compensated(below)
        end
        # Transition-function thresholds: F'-ODE/asymptotic at x = 35 and
        # the F-1 asymptotic at x = 60, checked against high precision.
        for x in (34.9, 35.1, 59.9, 60.1)
            ref = Complex{Float64}(transition_F_hp(x; dps=80))
            @test relerr(transition_F(x), ref) < 2e-12
            xb = BigFloat(x)
            hbig = BigFloat("1e-10") * xb
            Fp = (transition_F_hp(xb - 2hbig) -
                  8 * transition_F_hp(xb - hbig) +
                  8 * transition_F_hp(xb + hbig) -
                  transition_F_hp(xb + 2hbig)) / (12hbig)
            _, f1, _, _ = transition_derivatives(x)
            @test relerr(f1, Complex{Float64}(Fp)) < 2e-8
        end
    end
end

end
