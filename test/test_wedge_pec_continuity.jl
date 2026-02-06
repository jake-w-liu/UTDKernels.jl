using Test
using UTDKernels

# Helper: check if φ is near a shadow/reflection boundary by testing
# whether any cotangent argument ψ_j is close to a multiple of π.
function _near_boundary(phi, phip, n; margin=0.05)
    terms = UTDKernels.kp_four_terms(phi, phip, n)
    for j in 1:4
        psi = terms.psi[j]
        dist = abs(psi - round(psi / π) * π)
        dist < margin && return true
    end
    return false
end

@testset "PEC wedge continuity across shadow boundaries" begin

    @testset "Half-plane (α=2π) finiteness and smoothness" begin
        w = Wedge(2π)
        n = wedge_n(w)  # n = 2
        k = 10.0
        L = 1.0
        phip = π/4

        # Sweep φ across the incident shadow boundary at φ = π + φ' = 5π/4
        isb = π + phip
        phis = range(isb - 0.5, isb + 0.5, length=201)

        Ds_vals = ComplexF64[]
        Dh_vals = ComplexF64[]

        for phi in phis
            ang = RayAngles(phi, phip)
            Ds, Dh = pec_wedge_DsDh(w, ang, k, L)
            push!(Ds_vals, Ds)
            push!(Dh_vals, Dh)
        end

        # All values must be finite (no NaN/Inf from cot·F singularity)
        @test all(isfinite.(abs.(Ds_vals)))
        @test all(isfinite.(abs.(Dh_vals)))

        # D has a compensating discontinuity at the shadow boundary
        # (cancels the GO field jump).  Check smoothness only AWAY from it.
        smooth_idx = [i for i in 1:length(phis)
                      if !_near_boundary(phis[i], phip, n; margin=0.03)]

        for (label, vals) in [("Ds", Ds_vals), ("Dh", Dh_vals)]
            for i in 1:length(smooth_idx)-1
                i1, i2 = smooth_idx[i], smooth_idx[i+1]
                if i2 == i1 + 1  # only adjacent sweep points
                    jump = abs(vals[i2] - vals[i1])
                    @test jump < 1e-2
                end
            end
        end
    end

    @testset "90° wedge (α=3π/2) finiteness and smoothness" begin
        w = Wedge(3π/2)
        n = wedge_n(w)
        k = 20.0
        L = 0.5
        phip = π/6

        phis = range(0.1, w.alpha - 0.1, length=301)

        Ds_vals = ComplexF64[]
        Dh_vals = ComplexF64[]

        for phi in phis
            ang = RayAngles(phi, phip)
            Ds, Dh = pec_wedge_DsDh(w, ang, k, L)
            push!(Ds_vals, Ds)
            push!(Dh_vals, Dh)
        end

        # All values must be finite
        @test all(isfinite.(abs.(Ds_vals)))
        @test all(isfinite.(abs.(Dh_vals)))

        # Smoothness away from boundaries
        smooth_idx = [i for i in 1:length(phis)
                      if !_near_boundary(phis[i], phip, n; margin=0.05)]

        for (label, vals) in [("Ds", Ds_vals), ("Dh", Dh_vals)]
            for i in 1:length(smooth_idx)-1
                i1, i2 = smooth_idx[i], smooth_idx[i+1]
                if i2 == i1 + 1
                    jump = abs(vals[i2] - vals[i1])
                    @test jump < 0.1
                end
            end
        end
    end
end
