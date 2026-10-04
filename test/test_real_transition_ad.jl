using Test, UTDKernels, ForwardDiff
isdefined(@__MODULE__, :passive_transition_oracle) ||
    include(joinpath(@__DIR__, "support", "passive_transition_oracle.jl"))

real_axis_derivative(f, x) = complex(
    ForwardDiff.derivative(q -> real(f(q)), x),
    ForwardDiff.derivative(q -> imag(f(q)), x))

@testset "Real transition derivatives retain small large-argument sensitivities" begin
    for x in (1e-16, 1e-6, 0.1, 0.35, 1.0, 20.0, 34.9, 35.0, 35.1, 47.9, 48.0, 48.1,
              1e2, 1e3, 1e4, 1e6, 1e8, 1e12, 1e16)
        reference = passive_transition_oracle(x)
        @test F_utd(x) ≈ reference[1] rtol=2e-14 atol=0
        @test real_axis_derivative(F_utd, x) ≈ reference[3] rtol=2e-10 atol=0
        @test real_axis_derivative(q -> real_axis_derivative(F_utd,q), x) ≈
              reference[4] rtol=2e-9 atol=0
    end
    @test F_utd(1.0f0) isa ComplexF64
    @test F_utd(0.0) == 0
    @test F_utd(Inf) == 1
    @test_throws ArgumentError F_utd(big"1.0")
end

@testset "Real wedge distance derivatives against independent precision" begin
    for n in (1.5, 2.0), L in (1.0, 748.8103857590023, 840.6652885618325,
                              890.735463861044, 1000.0, 1e4), channel in (1,2)
        phi, phip, k = pi/3, pi/4, 10.0
        wedge = Wedge(n*pi)
        magnitude(q) = abs(pec_wedge_DsDh(wedge,RayAngles(phi,phip),k,q)[channel])
        reference = setprecision(BigFloat, 384) do
            q = BigFloat(L)
            h = q*big"1e-20"
            value(t) = abs(passive_wedge_oracle(BigFloat(n)*BigFloat(pi),
                BigFloat(phi),BigFloat(phip),k,t;digits=105)[channel])
            (value(q+h)-value(q-h))/(2h)
        end
        @test ForwardDiff.derivative(magnitude,L) ≈ reference rtol=2e-8 atol=1e-29
    end
end

@testset "Local wedge representation shares the large-transition derivatives" begin
    for delta in (-1e-8, 1e-8), k0 in (1e20, 1e24, 1e28),
        k in (k0, complex(k0), k0*(1-0.1im))
        n, L = 1.5, 1.3
        a = 2sin(n*delta)^2
        product(q) = UTDKernels._cot_F_regularized(delta,a,k,q;n,detuning=delta)
        reference = setprecision(BigFloat, 384) do
            d, nb, kb, lb = BigFloat(delta), BigFloat(n), Complex{BigFloat}(k), BigFloat(L)
            factor = kb*2sin(nb*d)^2
            cot(d)*passive_transition_oracle(factor*lb;digits=105)[3]*factor
        end
        @test real_axis_derivative(product,L) ≈ reference rtol=2e-10 atol=0
    end
end

@testset "Transition AD propagates independent parameter directions and Hessians" begin
    point = [4.0, 25.0]
    reference = passive_transition_oracle(100.0)
    for component in (real, imag)
        field(q) = component(F_utd(q[1]*q[2]))
        first, second = component(reference[3]), component(reference[4])
        expected_gradient = first .* reverse(point)
        expected_hessian = [second*point[2]^2 second*prod(point)+first;
                            second*prod(point)+first second*point[1]^2]
        @test ForwardDiff.gradient(field, point) ≈ expected_gradient rtol=2e-10 atol=0
        @test ForwardDiff.hessian(field, point) ≈ expected_hessian rtol=2e-9 atol=0
    end
end
