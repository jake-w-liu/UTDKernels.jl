using ForwardDiff
using Random
using SpecialFunctions: erfcx

include("support/multipole_transition_oracle.jl")

_multipole_relative_error(value, reference; floor=1e-300) =
    abs(value - reference) / max(abs(reference), floor)

function _multipole_regular_nodes(center, radius, order, phase=0.17)
    return [center + radius * cis(phase + 2pi * index / order)
            for index in 0:(order - 1)]
end

const MULTIPOLE_SEPARATED_NODES = ComplexF64[
    -0.6 + 0.8im,
    0.2 + 1.1im,
    0.9 + 0.7im,
]
const MULTIPOLE_CLUSTERED_NODES = ComplexF64[
    0.4 + 0.8im + 1e-4 * cis(0.17 + 2pi * index / 3) for index in 0:2
]
const MULTIPOLE_REPEATED_NODES = fill(0.4 + 0.8im, 4)

@noinline _multipole_single_allocation_probe() =
    multipole_transition(view(MULTIPOLE_SEPARATED_NODES, 1:1))
@noinline _multipole_direct_allocation_probe() =
    faddeeva_divided_difference(MULTIPOLE_SEPARATED_NODES)
@noinline _multipole_auto_direct_allocation_probe() =
    multipole_transition(MULTIPOLE_SEPARATED_NODES)
@noinline _multipole_repeated_allocation_probe() =
    multipole_transition(MULTIPOLE_REPEATED_NODES)
@noinline _multipole_cluster_allocation_probe() =
    multipole_transition(MULTIPOLE_CLUSTERED_NODES)
@noinline _multipole_derivative_allocation_probe() =
    UTDKernels._faddeeva_scaled_derivatives(0.4 + 0.8im, 31)

@testset "Clustered-pole multipole transition" begin
    @testset "single-node Faddeeva identity and diagnostics" begin
        frozen = (
            (0.2 + 0.5im, 0.6015132279083472 + 0.10078197394037552im),
            (-1.1 + 0.7im, 0.30781552738461526 - 0.2868151163702649im),
            (2.0 + 1.5im, 0.15041543887103978 + 0.17037114276247706im),
        )
        for (node, reference) in frozen
            value, info = multipole_transition([node]; return_info=true)
            @test value == UTDKernels._faddeeva_w(node)
            @test _multipole_relative_error(value, reference) < 2e-15
            @test value ≈ erfcx(-im * node) rtol=1e-15
            @test info isa MultipoleEvaluationInfo
            @test info.method === :single
            @test info.order == 1
            @test info.center == node
            @test iszero(info.cluster_radius)
            @test info.direct_condition_estimate == 1
            @test info.terms_used == 1
        end
    end

    @testset "scaled Faddeeva derivative recurrence" begin
        for node in (0.3 + 0.8im, -0.7 + 1.2im)
            scaled = UTDKernels._faddeeva_scaled_derivatives(node, 12)
            @test scaled[1] == UTDKernels._faddeeva_w(node)
            @test scaled[2] ≈ -2node * scaled[1] + 2im / sqrt(pi) rtol=2e-15
            for order in 1:11
                expected = (-2node * scaled[order + 1] - 2scaled[order]) /
                           (order + 1)
                @test scaled[order + 2] ≈ expected rtol=2e-15 atol=2e-16
            end
        end
        @test_throws DomainError UTDKernels._faddeeva_scaled_derivatives(0.2im, -1)
    end

    @testset "direct divided difference and cancellation estimate" begin
        value, condition = faddeeva_divided_difference(
            MULTIPOLE_SEPARATED_NODES; return_condition=true,
        )
        reference = ComplexF64(_multipole_oracle_divided_difference(
            MULTIPOLE_SEPARATED_NODES,
        ))
        @test _multipole_relative_error(value, reference) < 2e-14
        @test _multipole_relative_error(
            value, -0.15659371487015955 - 0.04355440919326803im,
        ) < 5e-14
        @test isfinite(condition) && condition >= 1

        pair = MULTIPOLE_SEPARATED_NODES[1:2]
        explicit_pair = (UTDKernels._faddeeva_w(pair[1]) -
                         UTDKernels._faddeeva_w(pair[2])) / (pair[1] - pair[2])
        @test faddeeva_divided_difference(pair) ≈ explicit_pair rtol=3e-15
        @test_throws ArgumentError faddeeva_divided_difference(
            ComplexF64[0.4 + 0.8im, 0.4 + 0.8im],
        )
    end

    @testset "confluent and merging clusters" begin
        center = 0.4 + 0.8im
        for order in 1:7
            repeated = fill(center, order)
            value, info = multipole_transition(repeated; return_info=true)
            contour = _multipole_oracle_contour(repeated)
            @test _multipole_relative_error(value, contour) < 4e-12
            @test info.method === (order == 1 ? :single : :cluster)
            @test info.terms_used == 1
        end

        for order in 2:7, radius in (1e-2, 1e-5, 1e-9)
            nodes = _multipole_regular_nodes(center, radius, order)
            value = multipole_transition(nodes)
            reference = ComplexF64(_multipole_oracle_divided_difference(nodes))
            @test _multipole_relative_error(value, reference) < 1e-12
        end

        partial = ComplexF64[center, center, center + 0.15 + 0.04im]
        partial_value = multipole_transition(partial)
        partial_contour = _multipole_oracle_contour(partial)
        @test _multipole_relative_error(partial_value, partial_contour) < 4e-12
        @test multipole_transition(fill(center, 3); max_terms=2) ==
              multipole_transition(fill(center, 3))

        # A regular pair has alternating structural zero corrections. One small
        # term cannot certify convergence because a later even term is nonzero.
        nodes = _multipole_regular_nodes(center, 0.0520055, 2)
        value, info = multipole_transition(nodes; return_info=true)
        reference = ComplexF64(_multipole_oracle_divided_difference(nodes))
        @test info.method === :cluster
        @test info.terms_used >= 10
        @test _multipole_relative_error(value, reference) < 3e-15

        # The analytic divided difference is entire even though a physical
        # contour crossing requires caller-owned residue bookkeeping.
        for lower_center in (0.2 - 0.4im, -0.5 - 0.7im)
            lower_nodes = ComplexF64[
                lower_center - 0.03,
                lower_center + 0.02 + 0.01im,
                lower_center + 0.01 - 0.02im,
            ]
            lower_reference = ComplexF64(
                _multipole_oracle_divided_difference(lower_nodes),
            )
            @test _multipole_relative_error(
                multipole_transition(lower_nodes), lower_reference,
            ) < 2e-12
        end
    end

    @testset "permutation invariance" begin
        rng = MersenneTwister(31)
        nodes = ComplexF64[-0.2 + 0.9im, 0.15 + 0.72im,
                           0.38 + 1.01im, 0.61 + 0.86im]
        reference = multipole_transition(nodes)
        for _ in 1:20
            value = multipole_transition(nodes[randperm(rng, length(nodes))])
            @test _multipole_relative_error(value, reference) < 3e-13
        end

        clustered = ComplexF64[
            0.4 + 0.8im + 1e-3 * offset for offset in
            (-0.8 + 0.1im, -0.2 - 0.7im, 0.1 + 0.9im,
             0.35 - 0.25im, 0.55 - 0.05im)
        ]
        clustered_reference, clustered_info = multipole_transition(
            clustered; return_info=true,
        )
        @test clustered_info.method === :cluster
        for _ in 1:20
            value = multipole_transition(clustered[randperm(rng, length(clustered))])
            @test _multipole_relative_error(value, clustered_reference) < 3e-13
        end
    end

    @testset "radius and cancellation selection boundaries" begin
        nodes = ComplexF64[-0.4 + 0.8im, 0.6 + 0.8im]
        _, condition = faddeeva_divided_difference(nodes; return_condition=true)
        center = sum(nodes) / length(nodes)
        radius = maximum(abs, nodes .- center)
        radius_boundary = radius / max(1.0, abs(center))

        _, radius_equal = multipole_transition(
            nodes;
            cluster_radius_threshold=radius_boundary,
            cancellation_threshold=nextfloat(condition),
            return_info=true,
        )
        _, radius_below = multipole_transition(
            nodes;
            cluster_radius_threshold=prevfloat(radius_boundary),
            cancellation_threshold=nextfloat(condition),
            return_info=true,
        )
        @test radius_equal.method === :cluster
        @test radius_below.method === :direct

        _, cancellation_equal = multipole_transition(
            nodes;
            cluster_radius_threshold=nextfloat(0.0),
            cancellation_threshold=condition,
            return_info=true,
        )
        _, cancellation_above = multipole_transition(
            nodes;
            cluster_radius_threshold=nextfloat(0.0),
            cancellation_threshold=nextfloat(condition),
            return_info=true,
        )
        @test cancellation_equal.method === :cluster
        @test cancellation_above.method === :direct
    end

    @testset "rational Gaussian contour identity" begin
        nodes = ComplexF64[-0.55 + 0.9im, 0.1 + 1.15im,
                           0.75 + 0.8im, 1.05 + 1.3im]
        coefficients = ComplexF64[0.7 - 0.2im, -0.3 + 0.1im, 0.05 + 0.08im]
        value = _multipole_oracle_newton_integral(nodes, coefficients)
        contour = _multipole_oracle_contour(nodes; numerator=coefficients)
        @test _multipole_relative_error(value, contour) < 2e-12
    end

    @testset "types, AD, domains, and nonconvergence" begin
        nodes32 = ComplexF32[-0.3f0 + 0.8f0im, 0.4f0 + 0.9f0im]
        @test faddeeva_divided_difference(nodes32) isa ComplexF32
        @test multipole_transition(nodes32) isa ComplexF32
        @test eltype(UTDKernels._faddeeva_scaled_derivatives(nodes32[1], 4)) ===
              ComplexF32
        close32 = ComplexF32[
            0.4f0 + 0.8f0im + 1f-3 * cis(Float32(0.17 + 2pi * index / 3))
            for index in 0:2
        ]
        value32, info32 = multipole_transition(close32; return_info=true)
        reference32 = multipole_transition(ComplexF64.(close32))
        @test value32 isa ComplexF32
        @test info32.method === :cluster
        @test _multipole_relative_error(value32, reference32) < 2e-5

        derivative = ForwardDiff.derivative(0.0) do shift
            nodes = [complex(0.4 + shift - 1e-4, 0.8),
                     complex(0.4 + shift + 1e-4, 0.8)]
            real(multipole_transition(nodes; cluster_radius_threshold=1.0))
        end
        step = 1e-6
        shifted_value(shift) = real(multipole_transition(
            ComplexF64[0.4 + shift - 1e-4 + 0.8im,
                       0.4 + shift + 1e-4 + 0.8im];
            cluster_radius_threshold=1.0,
        ))
        centered = (shifted_value(step) - shifted_value(-step)) / (2step)
        @test derivative ≈ centered rtol=2e-8 atol=2e-10

        coalescent_derivative = ForwardDiff.derivative(0.0) do displacement
            real(multipole_transition([
                complex(0.4 + displacement, 0.8),
                complex(0.4, 0.8),
            ]))
        end
        coalescent_value(displacement) = real(multipole_transition(ComplexF64[
            0.4 + displacement + 0.8im,
            0.4 + 0.8im,
        ]))
        coalescent_centered = (
            coalescent_value(step) - coalescent_value(-step)
        ) / (2step)
        @test coalescent_derivative ≈ coalescent_centered rtol=2e-8 atol=2e-10

        @test_throws ArgumentError multipole_transition(ComplexF64[])
        @test_throws ArgumentError multipole_transition(fill(0.4 + 0.8im, 8))
        @test_throws DomainError multipole_transition(ComplexF64[complex(NaN, 0.0)])
        @test_throws ArgumentError multipole_transition(Number[0.4 + 0.8im])
        @test_throws ArgumentError multipole_transition(Complex{BigFloat}[
            complex(big"0.4", big"0.8"),
        ])
        @test_throws DomainError multipole_transition(
            MULTIPOLE_CLUSTERED_NODES; cluster_radius_threshold=0.0,
        )
        @test_throws DomainError multipole_transition(
            MULTIPOLE_CLUSTERED_NODES; cancellation_threshold=Inf,
        )
        @test_throws DomainError multipole_transition(
            MULTIPOLE_CLUSTERED_NODES; relative_tolerance=0.0,
        )
        @test_throws DomainError multipole_transition(
            MULTIPOLE_CLUSTERED_NODES; max_terms=129,
        )
        @test_throws DomainError multipole_transition(
            MULTIPOLE_CLUSTERED_NODES; max_terms=1,
        )
        @test_throws ArgumentError multipole_transition(
            _multipole_regular_nodes(0.4 + 0.8im, 0.05, 3);
            cluster_radius_threshold=1.0,
            relative_tolerance=eps(Float64)^2,
            max_terms=8,
        )
    end

    @testset "allocation bounds" begin
        _multipole_single_allocation_probe()
        _multipole_direct_allocation_probe()
        _multipole_auto_direct_allocation_probe()
        _multipole_repeated_allocation_probe()
        _multipole_cluster_allocation_probe()
        _multipole_derivative_allocation_probe()
        @test @allocated(_multipole_single_allocation_probe()) == 0
        @test @allocated(_multipole_direct_allocation_probe()) == 0
        @test @allocated(_multipole_auto_direct_allocation_probe()) == 0
        @test @allocated(_multipole_repeated_allocation_probe()) == 0
        @test @allocated(_multipole_cluster_allocation_probe()) <= 1024
        @test @allocated(_multipole_derivative_allocation_probe()) <= 640
    end
end
