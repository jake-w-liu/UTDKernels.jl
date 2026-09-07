using ForwardDiff
using FastGaussQuadrature: gausshermite
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
const MULTIPOLE_GAUSS_HERMITE_RULE = gausshermite(256)

function _multipole_gauss_hermite_repeated_oracle(node, order)
    abscissae, weights = MULTIPOLE_GAUSS_HERMITE_RULE
    return -im / pi * sum(weights ./ ((abscissae .- node) .^ order))
end

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
            value, info = multipole_transition_with_info([node])
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

    @testset "stable intermediate- and large-argument derivatives" begin
        for node in (16.0 + 1.0im, 32.0 + 1.0im), order in 1:7
            reference = _multipole_gauss_hermite_repeated_oracle(node, order)
            value = multipole_transition(fill(node, order))
            @test _multipole_relative_error(value, reference) < 5e-13
        end

        intermediate = 3.0 + 3.0im
        for order in 2:7
            value = multipole_transition(fill(intermediate, order))
            reference = _multipole_oracle_contour(fill(intermediate, order))
            @test _multipole_relative_error(value, reference) < 5e-13
        end

        reference = 3.612878758843347e-12 + 1.6191433223230267e-11im
        for T in (Float16, Float32, Float64)
            node = complex(T(32), T(1))
            @test multipole_transition(fill(node, 7)) ≈ Complex{T}(reference)
        end

        for node in (3.0 + 3.0im, 32.0 + 1.0im), order in 2:6
            real_derivative = ForwardDiff.derivative(0.0) do shift
                real(multipole_transition(fill(node + shift, order)))
            end
            imag_derivative = ForwardDiff.derivative(0.0) do shift
                imag(multipole_transition(fill(node + shift, order)))
            end
            next_value = order * multipole_transition(fill(node, order + 1))
            @test real_derivative ≈ real(next_value) rtol=2e-13 atol=1e-18
            @test imag_derivative ≈ imag(next_value) rtol=2e-13 atol=1e-18
            if order <= 5
                second_derivative = ForwardDiff.derivative(0.0) do outer
                    ForwardDiff.derivative(outer) do inner
                        real(multipole_transition(fill(node + inner, order)))
                    end
                end
                second_reference = order * (order + 1) *
                                   multipole_transition(fill(node, order + 2))
                @test second_derivative ≈ real(second_reference) rtol=3e-13 atol=1e-18
            end
        end

        for node in (-16.0 + 1.0im, 16.0 - 1.0im, 1.0 - 16.0im,
                     -1.0 - 16.0im, -16.0 + 0.0im, 16.0 + 0.0im),
            derivative_order in 0:6
            reference = ComplexF64(
                _multipole_oracle_scaled_derivative(node, derivative_order),
            )
            value = UTDKernels._faddeeva_scaled_derivative(
                node, derivative_order,
            )
            @test _multipole_relative_error(value, reference) < 5e-13
        end

        for boundary in (1.5, 12.0), derivative_order in 0:6
            below = UTDKernels._faddeeva_scaled_derivative(
                complex(prevfloat(boundary), 0.0), derivative_order,
            )
            at = UTDKernels._faddeeva_scaled_derivative(
                complex(boundary, 0.0), derivative_order,
            )
            above = UTDKernels._faddeeva_scaled_derivative(
                complex(nextfloat(boundary), 0.0), derivative_order,
            )
            @test below ≈ at rtol=2e-13 atol=2e-15
            @test above ≈ at rtol=2e-13 atol=2e-15
        end

        initial_precision = precision(BigFloat)
        low_ambient = setprecision(BigFloat, 128) do
            multipole_transition(fill(3.0 + 3.0im, 7))
        end
        high_ambient = setprecision(BigFloat, 512) do
            multipole_transition(fill(3.0 + 3.0im, 7))
        end
        @test low_ambient == high_ambient
        @test precision(BigFloat) == initial_precision

        expected_threaded = [
            multipole_transition(fill(3.0 + 3.0im, 7)),
            multipole_transition(fill(32.0 + 1.0im, 7)),
        ]
        tasks = map(1:64) do index
            Threads.@spawn multipole_transition(fill(
                isodd(index) ? 3.0 + 3.0im : 32.0 + 1.0im,
                7,
            ))
        end
        @test all(
            index -> fetch(tasks[index]) ==
                     expected_threaded[isodd(index) ? 1 : 2],
            eachindex(tasks),
        )
    end

    @testset "direct divided difference and cancellation estimate" begin
        value, condition = faddeeva_divided_difference_with_condition(
            MULTIPOLE_SEPARATED_NODES,
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
            value, info = multipole_transition_with_info(repeated)
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
        value, info = multipole_transition_with_info(
            nodes; cluster_radius_threshold=0.075,
        )
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
        clustered_reference, clustered_info =
            multipole_transition_with_info(clustered)
        @test clustered_info.method === :cluster
        for _ in 1:20
            value = multipole_transition(clustered[randperm(rng, length(clustered))])
            @test _multipole_relative_error(value, clustered_reference) < 3e-13
        end
    end

    @testset "radius and cancellation selection boundaries" begin
        nodes = ComplexF64[-0.4 + 0.8im, 0.6 + 0.8im]
        _, condition = faddeeva_divided_difference_with_condition(nodes)
        center = sum(nodes) / length(nodes)
        radius = maximum(abs, nodes .- center)
        radius_boundary = radius / max(1.0, abs(center))

        _, radius_equal = multipole_transition_with_info(
            nodes;
            cluster_radius_threshold=radius_boundary,
            cancellation_threshold=nextfloat(condition),
        )
        _, radius_below = multipole_transition_with_info(
            nodes;
            cluster_radius_threshold=prevfloat(radius_boundary),
            cancellation_threshold=nextfloat(condition),
        )
        @test radius_equal.method === :cluster
        @test radius_below.method === :direct

        _, cancellation_equal = multipole_transition_with_info(
            nodes;
            cancellation_threshold=condition,
        )
        _, cancellation_above = multipole_transition_with_info(
            nodes;
            cancellation_threshold=nextfloat(condition),
        )
        @test cancellation_equal.method === :cluster
        @test cancellation_above.method === :direct

        large_center_pair = ComplexF64[31.0 + 1.0im, 33.0 + 1.0im]
        direct_pair = faddeeva_divided_difference(large_center_pair)
        automatic_pair, automatic_info = multipole_transition_with_info(
            large_center_pair,
        )
        clustered_pair, clustered_info = multipole_transition_with_info(
            large_center_pair;
            cancellation_threshold=2.0,
            cluster_radius_threshold=1.0,
        )
        @test automatic_info.method === :direct
        @test automatic_pair == direct_pair
        @test clustered_info.method === :cluster
        @test clustered_pair ≈ direct_pair rtol=2e-13

        difficult32 = ComplexF32[
            1.9190488 + 4.2873015im,
            3.37497 + 1.4908031im,
            1.7197671 + 4.134997im,
            2.5417588 + 2.2994096im,
            4.660606 + 2.6533122im,
            3.069683 + 3.1570365im,
            4.781017 + 2.7132313im,
        ]
        difficult_reference = ComplexF64(
            _multipole_oracle_divided_difference(ComplexF64.(difficult32)),
        )
        difficult_value, difficult_info =
            multipole_transition_with_info(difficult32)
        @test difficult_info.method === :cluster
        @test _multipole_relative_error(
            difficult_value, difficult_reference,
        ) < 5e-5
        direct32, direct32_info = multipole_transition_with_info(
            difficult32; cancellation_threshold=4000.0,
        )
        @test direct32_info.method === :direct
        @test _multipole_relative_error(direct32, difficult_reference) < 2e-4
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
        @test (@inferred faddeeva_divided_difference(nodes32)) isa ComplexF32
        @test (@inferred faddeeva_divided_difference_with_condition(
            nodes32,
        )) isa Tuple{ComplexF32,Float32}
        @test (@inferred multipole_transition(nodes32)) isa ComplexF32
        @test (@inferred multipole_transition_with_info(
            nodes32,
        )) isa Tuple{ComplexF32,MultipoleEvaluationInfo{ComplexF32,Float32}}
        for T in (Float16, Float64)
            typed_nodes = Complex{T}[complex(T(-0.3), T(0.8)),
                                     complex(T(0.4), T(0.9))]
            @test (@inferred faddeeva_divided_difference(typed_nodes)) isa Complex{T}
            @test (@inferred faddeeva_divided_difference_with_condition(
                typed_nodes,
            )) isa Tuple{Complex{T},T}
            @test (@inferred multipole_transition(typed_nodes)) isa Complex{T}
            @test (@inferred multipole_transition_with_info(
                typed_nodes,
            )) isa Tuple{Complex{T},MultipoleEvaluationInfo{Complex{T},T}}
        end

        dual_real = ForwardDiff.Dual(0.4, 1.0)
        dual_imag = ForwardDiff.Dual(0.8, 0.0)
        dual_node = complex(dual_real, dual_imag)
        dual_nodes = [dual_node - 0.1, dual_node + 0.1]
        dual_type = typeof(dual_node)
        @test (@inferred faddeeva_divided_difference(dual_nodes)) isa dual_type
        @test (@inferred faddeeva_divided_difference_with_condition(
            dual_nodes,
        )) isa Tuple{dual_type,Float64}
        @test (@inferred multipole_transition(dual_nodes)) isa dual_type
        @test (@inferred multipole_transition_with_info(
            dual_nodes,
        )) isa Tuple{dual_type,MultipoleEvaluationInfo{dual_type,Float64}}
        @test eltype(UTDKernels._faddeeva_scaled_derivatives(nodes32[1], 4)) ===
              ComplexF32
        close32 = ComplexF32[
            0.4f0 + 0.8f0im + 1f-3 * cis(Float32(0.17 + 2pi * index / 3))
            for index in 0:2
        ]
        value32, info32 = multipole_transition_with_info(close32)
        reference32 = multipole_transition(ComplexF64.(close32))
        @test value32 isa ComplexF32
        @test info32.method === :cluster
        @test _multipole_relative_error(value32, reference32) < 2e-5

        derivative = ForwardDiff.derivative(0.0) do shift
            nodes = [complex(0.4 + shift - 1e-4, 0.8),
                     complex(0.4 + shift + 1e-4, 0.8)]
            real(multipole_transition(nodes; cancellation_threshold=1.0))
        end
        step = 1e-6
        shifted_value(shift) = real(multipole_transition(
            ComplexF64[0.4 + shift - 1e-4 + 0.8im,
                       0.4 + shift + 1e-4 + 0.8im];
            cancellation_threshold=1.0,
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
            cancellation_threshold=1.0,
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
