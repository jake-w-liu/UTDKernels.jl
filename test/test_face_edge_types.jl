using Test, UTDKernels, ForwardDiff

struct FaceEdgeTypeFirst end
struct FaceEdgeTypeInner end
struct FaceEdgeTypeOuter end

function _typed_face_edge(epsilon, delta, k, L, explicit)
    p = oftype(epsilon, pi)
    wedge = Wedge((one(epsilon) + epsilon) * p)
    return explicit ? pec_wedge_face_edge(wedge, delta, k, L, epsilon) :
                      pec_wedge_face_edge(wedge, delta, k, L)
end

_field_value(z) = complex(ForwardDiff.value(real(z)), ForwardDiff.value(imag(z)))
_field_partial(z, i) = complex(ForwardDiff.partials(real(z))[i], ForwardDiff.partials(imag(z))[i])

function _face_edge_fields_hp(e, d, k, L)
    e, d, k, L = BigFloat.((e,d,k,L))
    edge = face_edge_incident_hp(e, d, k, L; digits=110)
    n = 1+e
    angle = BigFloat(pi)*n/2
    # The independent unrearranged reflection pair depends only on phi+phip.
    _, face = UTDKernels.FaceEdge.wedge_pairs_hp(angle, angle, n, k, L; dps=110)
    return (; Ds=edge-face, Dh=edge+face, edge, face)
end

@testset "face-edge promoted outputs and flat-boundary germs" begin
    @testset "all compensated routes preserve the promoted scalar type" begin
        for T in (Float32, Float64), explicit in (false, true)
            p = T(pi)
            # Small-argument, Gaussian, derivative-Taylor, direct, asymptotic,
            # endpoint-adjacent and symmetric routes share the same contract.
            for (epsilon, delta, k, L) in (
                (T(0.01), p/2, T(0.2), one(T)),
                (T(0.01), p/2, T(20), one(T)),
                (T(1e-5), p/2, T(200), one(T)),
                (T(0.05), p/2, T(200), one(T)),
                (T(0.01), p/2, T(1e5), one(T)),
                (T(1e-5), prevfloat(p), T(20), one(T)),
                (T(0.01), zero(T), T(20), one(T)),
                (zero(T), p/2, T(20), one(T)),
            )
                fields = _typed_face_edge(epsilon, delta, k, L, explicit)
                @test all(z -> z isa Complex{T}, values(fields))
                @test all(isfinite, values(fields))
                @test fields.Ds == fields.edge - fields.face
                @test fields.Dh == fields.edge + fields.face
            end
        end
        # A genuinely mixed input promotes the output, rather than narrowing
        # every result to the storage type of any one argument.
        for explicit in (false, true)
            fields = _typed_face_edge(0.01, 0.5f0*Float32(pi), 20f0, 1f0, explicit)
            @test all(z -> z isa ComplexF64, values(fields))
        end
    end

    @testset "live seeds and nested tags retain exact range homogeneity" begin
        for T in (Float32, Float64), explicit in (false, true), k in T.((0.2, 20, 200))
            e, d, L = T(0.01), T(pi)/2, T(1.1)
            baseline = _typed_face_edge(e, d, k, L, explicit)
            # k -> k*c, L -> L/c leaves every transition argument unchanged;
            # all four physical coefficients are multiplied by c^(-1/2).
            actual(t) = _typed_face_edge(e+zero(t), d+zero(t),
                                         k*(one(t)+t), L/(one(t)+t), explicit)
            first = actual(ForwardDiff.Dual{FaceEdgeTypeFirst}(zero(T), T(0.25), T(-0.75)))
            inner = ForwardDiff.Dual{FaceEdgeTypeInner}(zero(T), T(0.3))
            outer = ForwardDiff.Dual{FaceEdgeTypeOuter}(inner,
                ForwardDiff.Dual{FaceEdgeTypeInner}(T(0.7), T(0.11)))
            nested = actual(outer)
            @test all(z -> z isa Complex{ForwardDiff.Dual{FaceEdgeTypeFirst,T,2}}, values(first))
            @test all(z -> z isa Complex{ForwardDiff.Dual{FaceEdgeTypeOuter,
                ForwardDiff.Dual{FaceEdgeTypeInner,T,1},1}}, values(nested))
            # Float32 budgets cover the rounded trigonometric pair; the same
            # invariant is checked more tightly in binary64.
            tol = T === Float32 ? 128eps(T) : 5e-11
            for name in keys(baseline)
                value = getproperty(baseline, name)
                f, n = getproperty(first, name), getproperty(nested, name)
                @test _field_value(f) ≈ value rtol=tol
                for (i, seed) in enumerate((T(0.25), T(-0.75)))
                    @test _field_partial(f, i) ≈ -value*seed/2 rtol=tol
                end
                @test _field_partial(_field_value(n), 1) ≈ -value*T(0.3)/2 rtol=tol
                @test _field_value(_field_partial(n, 1)) ≈ -value*T(0.7)/2 rtol=tol
                expected = value*(3T(0.3)*T(0.7)/4-T(0.11)/2)
                @test _field_partial(_field_partial(n, 1), 1) ≈ expected rtol=tol
            end
        end
    end

    @testset "scaled transition derivatives share the real transition owner" begin
        for T in (Float32, Float64), (a, k) in ((T(0.5), T(20)), (one(T), T(200)), (T(2), T(200)))
            first_ref, second_ref = setprecision(BigFloat, 512) do
                ab, kb = BigFloat(a), BigFloat(k)
                x = kb*ab
                f = face_edge_F_hp(x; digits=110)
                recurrence = im+inv(2x)
                first = recurrence*f-im
                second = recurrence*first-f/(2x*x)
                (kb*ab*first/sqrt(kb), (kb*ab)^2*second/sqrt(kb))
            end
            for project in (real, imag)
                b(L) = project(UTDKernels._face_edge_B(a, k, L, T(pi)))
                tol = T === Float32 ? 128eps(T) : 5e-11
                @test ForwardDiff.derivative(b, one(T)) ≈ Float64(project(first_ref)) rtol=tol
                @test ForwardDiff.derivative(L -> ForwardDiff.derivative(b, L), one(T)) ≈
                    Float64(project(second_ref)) rtol=tol
            end
        end
    end

    @testset "zero-defect first, second and purely mixed coefficients" begin
        # The independent unrearranged cotangent-transition reference is
        # already loaded by test_wedge_face_edge.jl. Its one-sided stencil uses
        # 100 decimal digits, with h small enough that O(h) is below binary64.
        edge_first, edge_second = setprecision(BigFloat, 512) do
            h = big"1e-25"
            a = face_edge_incident_hp(h, BigFloat(pi)/2, big"20", big"1"; digits=110)
            b = face_edge_incident_hp(2h, BigFloat(pi)/2, big"20", big"1"; digits=110)
            ((4a-b)/(2h), (b-2a)/(h*h))
        end
        for T in (Float32, Float64), explicit in (false, true)
            # The public path uses the represented, type-local pi for its
            # phase. Retain that constant in the analytic reference: replacing
            # it by mathematical pi biases the small real second coefficient
            # by about 9e-6 in Float32 through its large imaginary partner.
            face_first, face_second = setprecision(BigFloat, 512) do
                p = BigFloat(T(pi))
                phase = cis(p/4)
                (-sqrt(big"40"/BigFloat(pi))*p*phase,
                 p*p*(20phase*phase-BigFloat(1)/4))
            end
            references = (; edge=(edge_first, edge_second),
                           face=(face_first, face_second),
                           Ds=(edge_first-face_first, edge_second-face_second),
                           Dh=(edge_first+face_first, edge_second+face_second))
            fields(e) = _typed_face_edge(e, oftype(e, pi)/2, oftype(e, 20), one(e), explicit)
            tol = T === Float32 ? 64eps(T) : 2e-11
            for name in keys(references), project in (real, imag)
                fun(e) = project(getproperty(fields(e), name))
                d1 = ForwardDiff.derivative(fun, zero(T))
                d2 = ForwardDiff.derivative(e -> ForwardDiff.derivative(fun, e), zero(T))
                mixed = ForwardDiff.derivative(u -> ForwardDiff.derivative(
                    v -> fun(u*v), zero(T)), zero(T))
                @test d1 isa T
                @test d2 isa T
                @test mixed isa T
                expected_first, expected_second = getproperty(references, name)
                @test d1 ≈ T(project(expected_first)) rtol=tol
                @test d2 ≈ T(project(expected_second)) rtol=tol
                @test mixed ≈ T(project(expected_first)) rtol=tol
            end
        end
    end

    @testset "endpoint angular derivatives use the smooth sine coordinate" begin
        epsilon, delta = 1e-6, prevfloat(Float64(pi))
        alpha = (1+epsilon)*pi
        for explicit in (false, true)
            expected_first, expected_second = setprecision(BigFloat, 512) do
                e = explicit ? BigFloat(epsilon) :
                    (BigFloat(alpha)-BigFloat(Float64(pi)))/BigFloat(Float64(pi))
                d, h = BigFloat(delta), BigFloat(2)^(-80)
                # This entire stencil lies inside delta < pi; a generic
                # binary-scale difference step would cross the pole boundary.
                f(t) = face_edge_incident_hp(e, d+t, big"1", big"1"; digits=110)
                fm2, fm, f0, fp, fp2 = (f(j*h) for j in (-2,-1,0,1,2))
                ((fm2-8*fm+8*fp-fp2)/(12h),
                 (-fp2+16*fp-30*f0+16*fm-fm2)/(12h*h))
            end
            fields(d) = explicit ? pec_wedge_face_edge(Wedge(alpha), d, 1.0, 1.0, epsilon) :
                                   pec_wedge_face_edge(Wedge(alpha), d, 1.0, 1.0)
            for project in (real, imag), name in (:edge, :Ds, :Dh)
                fun(d) = project(getproperty(fields(d), name))
                @test ForwardDiff.derivative(fun, delta) ≈ Float64(project(expected_first)) rtol=2e-8
                @test ForwardDiff.derivative(d -> ForwardDiff.derivative(fun, d), delta) ≈
                    Float64(project(expected_second)) rtol=2e-8
            end
            for project in (real, imag)
                face(d) = project(fields(d).face)
                @test iszero(ForwardDiff.derivative(face, delta))
                @test iszero(ForwardDiff.derivative(d -> ForwardDiff.derivative(face, d), delta))
            end
        end
    end

    @testset "large face-transition derivatives" begin
        for T in (Float32,Float64), k in T.((1e5,1e8))
            e, d = T(0.01), T(pi)/2
            first_ref, second_ref = setprecision(BigFloat, 512) do
                h = BigFloat(2)^(-35)
                f(t) = _face_edge_fields_hp(e,d,k,one(BigFloat)+t).face
                fm2,fm,f0,fp,fp2 = (f(j*h) for j in (-2,-1,0,1,2))
                ((fm2-8*fm+8*fp-fp2)/(12h),
                 (-fp2+16*fp-30*f0+16*fm-fm2)/(12h*h))
            end
            fields(L) = _typed_face_edge(e,d,k,L,true)
            for project in (real,imag)
                face(L) = project(fields(L).face)
                tol = T === Float32 ? 64eps(T) : 5e-11
                @test ForwardDiff.derivative(face,one(T)) ≈ Float64(project(first_ref)) rtol=tol
                @test ForwardDiff.derivative(L -> ForwardDiff.derivative(face,L),one(T)) ≈
                    Float64(project(second_ref)) rtol=tol
            end
        end
    end

    @testset "original-factor germs precede encoded product overflow" begin
        for T in (Float32,Float64), e in T.((0.01,1e-5))
            k,L,seed = T === Float32 ? (T(1e20),T(1e-10),T(1e20)) :
                                      (T(1e200),T(1e-190),T(1e110))
            d = T(pi)/2
            reference = setprecision(BigFloat, 2300) do
                h = BigFloat(2)^(-45)*BigFloat(L)/BigFloat(seed)
                f(t) = _face_edge_fields_hp(e,d,k,BigFloat(L)+BigFloat(seed)*t)
                fm2,fm,fp,fp2 = (f(j*h) for j in (-2,-1,1,2))
                map((a,b,c,q) -> (a-8*b+8*c-q)/(12h),fm2,fm,fp,fp2)
            end
            l = ForwardDiff.Dual{FaceEdgeTypeFirst}(L,seed)
            fields = _typed_face_edge(e,d,k,l,true)
            tol = T === Float32 ? 64eps(T) : 2e-11
            for name in keys(fields)
                field = getproperty(fields,name)
                @test field isa Complex{ForwardDiff.Dual{FaceEdgeTypeFirst,T,1}}
                @test _field_partial(field,1) ≈ getproperty(reference,name) rtol=tol
            end
            @test _field_partial(fields.Ds,1) ==
                _field_partial(fields.edge,1)-_field_partial(fields.face,1)
            @test _field_partial(fields.Dh,1) ==
                _field_partial(fields.edge,1)+_field_partial(fields.face,1)
        end
    end
end
