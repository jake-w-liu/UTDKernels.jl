using ForwardDiff
using LinearAlgebra
using Random

include("support/hessian_metric_oracle.jl")

const HESSIAN_ALLOC_H2 = [2.0 0.4; 0.4 1.3]
const HESSIAN_ALLOC_G2 = [0.8, -0.5]
const HESSIAN_ALLOC_H3 = [2.0 0.2 0.1; 0.2 1.4 -0.1; 0.1 -0.1 3.0]
const HESSIAN_ALLOC_G3 = [0.8, -0.5, 0.3]
const HESSIAN_ALLOC_H4 = Matrix(Diagonal([1.0, 2.0, 3.0, 4.0]))
const HESSIAN_ALLOC_G4 = [0.3, -0.4, 0.5, 0.6]

@noinline _hessian_q2_allocation_probe() =
    hessian_metric_q2(HESSIAN_ALLOC_H2, HESSIAN_ALLOC_G2)
@noinline _hessian_L2_allocation_probe() =
    hessian_effective_L(HESSIAN_ALLOC_H2, HESSIAN_ALLOC_G2)
@noinline _hessian_coordinate_allocation_probe() =
    hessian_transition_coordinate(0.2, HESSIAN_ALLOC_H2, HESSIAN_ALLOC_G2)
@noinline _hessian_argument_allocation_probe() =
    hessian_transition_argument(10.0, 0.2, HESSIAN_ALLOC_H2, HESSIAN_ALLOC_G2)
@noinline _hessian_L3_allocation_probe() =
    hessian_effective_L(HESSIAN_ALLOC_H3, HESSIAN_ALLOC_G3)
@noinline _hessian_directional_allocation_probe() =
    directional_effective_L(0.4, 4.0, 0.7)
@noinline _hessian_L4_allocation_probe() =
    hessian_effective_L(HESSIAN_ALLOC_H4, HESSIAN_ALLOC_G4)

@testset "Coordinate-invariant Hessian metric" begin
    @testset "isotropic and principal-direction limits" begin
        radius = 3.7
        H = radius * I(2)
        g = [cos(0.8), sin(0.8)]
        @test hessian_metric_q2(H, g) ≈ inv(radius) atol=2e-16
        @test hessian_effective_L(H, g) ≈ radius atol=2e-15
        @test hessian_effective_L(H, 2 .* g) ≈ radius / 4 atol=2e-15

        R1, R2 = 0.4, 4.0
        @test directional_effective_L(R1, R2, 0.0) ≈ R1 atol=1e-15
        @test directional_effective_L(R1, R2, pi / 2) ≈ R2 atol=1e-13
        @test directional_effective_L(R1, R2, pi / 4) ≈ 8 / 11 atol=1e-15
        @test directional_effective_L(2.5, 2.5, 0.7) == 2.5
        angles = range(-pi, pi; length=501)
        values = directional_effective_L.(R1, R2, angles)
        @test minimum(values) >= R1 - 2e-15
        @test maximum(values) <= R2 + 2e-12
    end

    @testset "coordinate and pole-equation invariance" begin
        rng = MersenneTwister(0x2026_0167)
        for dimension in (2, 3), _ in 1:80
            H = _hessian_oracle_random_spd(rng, dimension; condition_max=1e3)
            g = randn(rng, dimension)
            left, _ = qr(randn(rng, dimension, dimension))
            right, _ = qr(randn(rng, dimension, dimension))
            A = Matrix(left) * Diagonal(exp.(4 .* rand(rng, dimension) .- 2)) *
                Matrix(right)'
            transformed_H, transformed_g = _hessian_oracle_transform(H, g, A)
            q2 = hessian_metric_q2(H, g)
            q2_transformed = hessian_metric_q2(transformed_H, transformed_g)
            @test q2_transformed ≈ q2 rtol=3e-11
            delta = randn(rng)
            coordinate = hessian_transition_coordinate(delta, H, g)
            transformed_coordinate = hessian_transition_coordinate(
                delta, transformed_H, transformed_g,
            )
            @test transformed_coordinate ≈ coordinate rtol=3e-11 atol=1e-13
        end

        H = [2.0 0.3; 0.3 5.0]
        g = [0.4, -0.7]
        delta = 0.23
        reference_coordinate = hessian_transition_coordinate(delta, H, g)
        reference_argument = hessian_transition_argument(9.0, delta, H, g)
        for scale in (1e-300, 1e-100, 1e100, 1e300)
            scaled_coordinate = hessian_transition_coordinate(
                scale * delta, H, scale .* g,
            )
            scaled_argument = hessian_transition_argument(
                9.0, scale * delta, H, scale .* g,
            )
            @test scaled_coordinate ≈ reference_coordinate rtol=4e-15
            @test scaled_argument ≈ reference_argument rtol=8e-15
        end

        transform = Diagonal([1e-8, 1.0])
        transformed_H, transformed_g = _hessian_oracle_transform(H, g, transform)
        @test hessian_metric_q2(transformed_H, transformed_g) ≈
              hessian_metric_q2(H, g) rtol=5e-15

        reversed = hessian_transition_coordinate(-delta, H, -g)
        @test reversed ≈ -reference_coordinate rtol=3e-15
        @test hessian_transition_coordinate(delta, H, -g) ≈
              reference_coordinate rtol=3e-15
        @test hessian_transition_argument(9.0, -delta, H, -g) ≈
              reference_argument rtol=5e-15
    end

    @testset "safe inverse, Schur, and two-dimensional identities" begin
        @test hessian_metric_q2(reshape([4.0], 1, 1), [2.0]) == 1.0
        @test hessian_effective_L(reshape([4.0], 1, 1), [2.0]) == 1.0

        H = [2.0 0.4; 0.4 1.3]
        g = [0.8, -0.5]
        q2 = hessian_metric_q2(H, g)
        @test q2 ≈ dot(g, inv(H) * g) rtol=3e-15
        @test abs(q2 - dot(g, H * g)) / q2 > 0.5

        schur_H = [2.0 0.9; 0.9 1.0]
        schur = 2.0 - 0.9^2
        @test hessian_effective_L(schur_H, [1.0, 0.0]) ≈ schur atol=2e-15

        beta = 0.8
        H2 = [2.0 0.35; 0.35 1.4]
        normal = [cos(beta), sin(beta)]
        determinant_formula = det(H2) /
            (H2[2, 2] * cos(beta)^2 - H2[1, 2] * sin(2beta) +
             H2[1, 1] * sin(beta)^2)
        @test hessian_effective_L(H2, normal) ≈ determinant_formula rtol=4e-15

        H4 = [4.0 0.2 0.1 0.0;
              0.2 3.0 -0.1 0.1;
              0.1 -0.1 2.0 0.3;
              0.0 0.1 0.3 1.5]
        g4 = [0.4, -0.3, 0.8, 0.2]
        @test hessian_metric_q2(H4, g4) ≈ dot(g4, inv(H4) * g4) rtol=5e-15
        @test hessian_metric_q2(H4, g4) * hessian_effective_L(H4, g4) ≈
              1.0 rtol=5e-15

        nearly_symmetric = [2.0 nextfloat(0.4); 0.4 1.3]
        @test hessian_metric_q2(nearly_symmetric, g) ≈ q2 rtol=5e-15
    end

    @testset "signed coordinate and transition argument" begin
        H = Diagonal([2.0, 8.0])
        g = [1.0, 0.0]
        coordinate = hessian_transition_coordinate(0.2, H, g)
        argument = hessian_transition_argument(10.0, 0.2, H, g)
        @test coordinate ≈ 0.2sqrt(2) atol=1e-15
        @test argument ≈ 0.4 atol=1e-15
        @test argument == hessian_transition_argument(10.0, -0.2, H, g)
        @test F_utd(argument) ≈
              0.630872283432779 + 0.2725598353263281im rtol=5e-15

        @test hessian_transition_argument(1e300, 1e-150, I(2), [1.0, 0.0]) ≈
              0.5 rtol=5e-15
        @test hessian_transition_argument(1e-300, 1e150, I(2), [1.0, 0.0]) ≈
              0.5 rtol=5e-15
    end

    @testset "extreme Hessian scales" begin
        smallest = nextfloat(0.0)
        Hsubnormal = Diagonal([smallest, 1.0])
        @test hessian_metric_q2(Hsubnormal, [1.0, 0.0]) == Inf
        @test hessian_effective_L(Hsubnormal, [1.0, 0.0]) == smallest
        @test hessian_transition_coordinate(1.0, Hsubnormal, [1.0, 0.0]) ==
              sqrt(smallest)

        disparate = Diagonal([1e-300, 1e300])
        @test hessian_effective_L(disparate, [1.0, 0.0]) ≈ 1e-300 rtol=5e-15
        @test hessian_effective_L(disparate, [0.0, 1.0]) ≈ 1e300 rtol=5e-15
        @test hessian_transition_coordinate(1.0, disparate, [1.0, 0.0]) ≈
              1e-150 rtol=5e-15

        for scale in (1e-300, 1e300)
            H = Diagonal([scale, scale])
            @test hessian_effective_L(H, [1.0, 0.0]) ≈ scale rtol=5e-15
        end
    end

    @testset "independent multidimensional Gaussian reduction" begin
        cases = (
            ([2.0 0.4; 0.4 1.3], [0.8, -0.5], 0.6 + 1.4im, 64, 2e-11),
            ([2.0 0.2 0.1; 0.2 1.4 -0.1; 0.1 -0.1 3.0],
             [0.8, -0.5, 0.3], -0.5 + 1.7im, 42, 1e-10),
        )
        for (H, g, zeta, order, tolerance) in cases
            gaussian = _hessian_oracle_gh(H, g, zeta; order=order)
            faddeeva = UTDKernels._faddeeva_w(zeta)
            @test abs(gaussian - faddeeva) / abs(faddeeva) < tolerance
        end
        zeta = 0.2 + 1.3im
        @test _hessian_oracle_faddeeva_quadgk(zeta) ≈
              UTDKernels._faddeeva_w(zeta) rtol=1e-12
    end

    @testset "types, automatic differentiation, and domains" begin
        H32 = Float32[2 0.4; 0.4 1.3]
        g32 = Float32[0.8, -0.5]
        @test hessian_metric_q2(H32, g32) isa Float32
        @test hessian_effective_L(H32, g32) isa Float32
        @test hessian_transition_coordinate(0.2f0, H32, g32) isa Float32
        @test hessian_transition_argument(10f0, 0.2f0, H32, g32) isa Float32
        @test directional_effective_L(0.4f0, 4f0, 0.7f0) isa Float32
        @test hessian_effective_L(H32, g32) ≈
              Float32(hessian_effective_L(Float64.(H32), Float64.(g32))) rtol=5e-6

        setprecision(BigFloat, 256) do
            Hbig = BigFloat[2 0.4; 0.4 1.3]
            gbig = BigFloat[0.8, -0.5]
            @test hessian_effective_L(Hbig, gbig) isa BigFloat
            @test hessian_metric_q2(Hbig, gbig) * hessian_effective_L(Hbig, gbig) ≈
                  one(BigFloat) rtol=big"1e-70"
            @test directional_effective_L(big"0.4", big"4", big"0.7") isa BigFloat
        end

        derivative = ForwardDiff.derivative(2.0) do diagonal
            hessian_effective_L([diagonal 0.4; 0.4 1.3], [0.8, -0.5])
        end
        step = 1e-5
        centered = (
            hessian_effective_L([2.0 + step 0.4; 0.4 1.3], [0.8, -0.5]) -
            hessian_effective_L([2.0 - step 0.4; 0.4 1.3], [0.8, -0.5])
        ) / (2step)
        @test derivative ≈ centered rtol=2e-9 atol=2e-11

        directional_derivative = ForwardDiff.derivative(
            angle -> directional_effective_L(0.4, 4.0, angle), 0.7,
        )
        directional_centered = (
            directional_effective_L(0.4, 4.0, 0.7 + step) -
            directional_effective_L(0.4, 4.0, 0.7 - step)
        ) / (2step)
        @test directional_derivative ≈ directional_centered rtol=2e-9

        @test_throws DimensionMismatch hessian_metric_q2(ones(2, 3), ones(2))
        @test_throws DimensionMismatch hessian_metric_q2(I(2), ones(3))
        @test_throws DomainError hessian_metric_q2([1.0 0.0; 0.0 -1.0], ones(2))
        @test_throws DomainError hessian_metric_q2([1.0 2.0; 2.0 1.0], ones(2))
        @test_throws DomainError hessian_metric_q2([1.0 0.0; 0.0 0.0], ones(2))
        @test_throws DomainError hessian_metric_q2(I(2), zeros(2))
        @test_throws DomainError hessian_metric_q2([1.0 0.1; 0.0 1.0], ones(2))
        @test_throws DomainError hessian_metric_q2([1.0 NaN; NaN 1.0], ones(2))
        @test_throws DomainError hessian_metric_q2(I(2), [1.0, Inf])
        @test_throws ArgumentError hessian_metric_q2(zeros(0, 0), Float64[])
        @test_throws ArgumentError hessian_metric_q2(
            Matrix{Real}([1.0 0.0; 0.0 1.0]), Real[1.0, 0.0],
        )
        @test_throws DomainError hessian_transition_coordinate(NaN, I(2), ones(2))
        @test_throws DomainError hessian_transition_argument(0.0, 0.1, I(2), ones(2))
        @test_throws DomainError directional_effective_L(0.0, 1.0, 0.2)
        @test_throws DomainError directional_effective_L(1.0, 2.0, NaN)
    end

    @testset "allocation bounds" begin
        _hessian_q2_allocation_probe()
        _hessian_L2_allocation_probe()
        _hessian_coordinate_allocation_probe()
        _hessian_argument_allocation_probe()
        _hessian_L3_allocation_probe()
        _hessian_directional_allocation_probe()
        _hessian_L4_allocation_probe()
        @test @allocated(_hessian_q2_allocation_probe()) == 0
        @test @allocated(_hessian_L2_allocation_probe()) == 0
        @test @allocated(_hessian_coordinate_allocation_probe()) == 0
        @test @allocated(_hessian_argument_allocation_probe()) == 0
        @test @allocated(_hessian_L3_allocation_probe()) == 0
        @test @allocated(_hessian_directional_allocation_probe()) == 0
        @test @allocated(_hessian_L4_allocation_probe()) <= 1024
    end
end
