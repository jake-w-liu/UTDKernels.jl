using Test, UTDKernels, QuadGK, ForwardDiff

# Independent contour quadrature: erfcx(z) = 2/sqrt(pi) integral_0^Inf
# exp(-t^2-2zt) dt. Scaling t resolves large positive-real z without using
# the production erfcx implementation or its large-argument expansion.
function boundary_contour_oracle(delta, n, k, L)
    root = ComplexF64(sqrt(complex(big(k)*big(L))))
    side = delta < 0 ? -1 : 1
    z = (1+im)*root*(side*sin(n*delta))
    scale = 1+abs(z)
    integral = first(quadgk(t -> exp(-(t/scale)^2-2(z/scale)*t),
                           0.0, Inf; rtol=2e-13, atol=1e-14))/scale
    geometric = iszero(delta) ? n : cos(delta)*sin(n*delta)/sin(delta)
    return side*2(1+im)*geometric*integral*root
end

@testset "Exact and subnormal wedge-transition detunings" begin
    for n in (0.5, 1.0, 1.5, 2.0), kL in (1e-12, 1.0, 1e8, 1e16, 1e24)
        for delta in (0.0, 1e-200, -1e-200, 1e-12, -1e-12, 1e-9, -1e-9)
            actual = UTDKernels._cot_F_regularized(delta, 2sin(n*delta)^2,
                kL, 1.0; n, detuning=delta)
            expected = boundary_contour_oracle(delta, n, kL, 1.0)
            @test actual ≈ expected rtol=2e-11 atol=1e-25
        end
    end
    for (k,L) in ((1e250,1e100), (1e-250,1e-100), (1e250,1e-250)), n in (0.5,2.0)
        actual = UTDKernels._cot_F_regularized(0.0,0.0,k,L;n,detuning=0.0)
        expected = ComplexF64(n*sqrt(2big(pi)*big(k)*big(L))*exp(im*big(pi)/4))
        @test actual ≈ expected rtol=4e-15 atol=0
    end
    for delta in (-1e-9,1e-9), n in (0.5,2.0)
        # sqrt(kL) is representable although kL itself overflows; in this
        # regime the relative Fresnel correction is below binary precision.
        actual=UTDKernels._cot_F_regularized(delta,2sin(n*delta)^2,
            1e308,1e308;n,detuning=delta)
        @test actual ≈ cot(delta) rtol=4e-15 atol=0
    end
    for delta in (-2e-8,-1e-8,1e-8,2e-8), n in (0.5,1.5,2.0), kL in (0.1,1e16)
        # Cross the local-arithmetic routing threshold on both sides.
        actual=UTDKernels._cot_F_regularized(delta,2sin(n*delta)^2,
            kL,1.0;n,detuning=delta)
        @test actual ≈ boundary_contour_oracle(delta,n,kL,1.0) rtol=2e-11 atol=1e-25
    end
end

@testset "One-sided boundary derivatives preserve the analytic limit" begin
    for n in (0.5,1.5,2.0), k in (0.2,4.0,1e6,3.0-0.4im), L in (0.3,2.0)
        field(delta) = UTDKernels._cot_F_regularized(delta,2sin(n*delta)^2,
            k,L;n,detuning=delta)
        value = n*sqrt(2pi*k*L)*cispi(0.25)
        first_derivative = -4im*n^2*k*L
        second_derivative = value*(4im*n^2*k*L-(n^2+2)/3)
        @test field(0.0) ≈ value rtol=3e-15
        for component in (real,imag)
            @test ForwardDiff.derivative(t -> component(field(t)),0.0) ≈
                  component(first_derivative) rtol=3e-13 atol=3e-13*abs(first_derivative)
            @test ForwardDiff.derivative(t -> ForwardDiff.derivative(
                s -> component(field(s)),t),0.0) ≈ component(second_derivative) rtol=3e-12 atol=3e-12*abs(second_derivative)
        end
    end
end

@testset "Public PEC boundary retains the half incident field at high frequency" begin
    wedge=Wedge(3pi/2); angles=RayAngles(5pi/4,pi/4)
    # The single pole contributes exactly -sqrt(L)/2; the remaining three
    # nonsingular terms are independently evaluated by contour quadrature.
    n=1.5
    terms=UTDKernels.kp_four_terms(angles.phi,angles.phip,n)
    for k in (1.0,1e8,1e16,1e24), L in (0.2,1.0,3.0)
        values=map(1:4) do j
            detuning=UTDKernels._kp_transition_detuning(j,terms,n)
            if iszero(detuning)
                -sqrt(L)/2
            else
                x=k*L*terms.aj[j]
                z=(1+im)*sqrt(x/2)
                scale=1+abs(z)
                integral=first(quadgk(t -> exp(-(t/scale)^2-2(z/scale)*t),
                    0.0,Inf;rtol=2e-13,atol=1e-14))/scale
                F=2sqrt(x)*cispi(0.25)*integral
                -cispi(-0.25)/(2n*sqrt(2pi*k))*cot(terms.psi[j])*F
            end
        end
        expected_soft=values[1]+values[2]-values[3]-values[4]
        expected_hard=sum(values)
        actual_soft,actual_hard=pec_wedge_DsDh(wedge,angles,k,L)
        @test actual_soft ≈ expected_soft rtol=2e-11 atol=2e-13
        @test actual_hard ≈ expected_hard rtol=2e-11 atol=2e-13
    end
end
