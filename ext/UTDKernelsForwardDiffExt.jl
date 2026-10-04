"""
Package extension: ForwardDiff support for UTDKernels.

Provides a stable large-argument derivative rule for `F_utd(::Dual)` and forward-mode
AD rules for `erfcx(::Complex{Dual})` and `erfc(::Complex{Dual})` so transition
values, passive derivatives, bivariate
switches, and wedge coefficients can be differentiated through by ForwardDiff.

The complex derivative of erfcx is:
    d/dz erfcx(z) = 2z·erfcx(z) − 2/√π
"""
module UTDKernelsForwardDiffExt

import UTDKernels
import SpecialFunctions
import ForwardDiff: Dual, Tag, value, partials, Partials

# Differentiate F itself: the erfcx product's large leading terms cancel in
# its derivative. F_utd_prime evaluates that derivative with its stable
# inverse-power series while the primal value keeps the fast erfcx evaluator.
function UTDKernels.F_utd(x::Dual{T,V,N}) where {T,V,N}
    primal = UTDKernels._primal_value(x)
    if UTDKernels._passive_supported_type(x) && isfinite(primal) &&
       primal >= UTDKernels.MIN_F_PRIME_ASYMPTOTIC_THRESHOLD
        x_value = value(x)
        f_value = UTDKernels.F_utd(x_value)
        derivative = UTDKernels.F_utd_prime(x_value + zero(Float64))
        seeds = partials(x)
        return Complex(
            Dual{T}(real(f_value), Partials(ntuple(j -> real(derivative)*seeds[j], Val(N)))),
            Dual{T}(imag(f_value), Partials(ntuple(j -> imag(derivative)*seeds[j], Val(N)))),
        )
    end
    return UTDKernels._F_utd_erfcx(x)
end

function SpecialFunctions.erfcx(z::Complex{Dual{T,V,N}}) where {T,V,N}
    # Primal evaluation
    z_val = Complex(value(real(z)), value(imag(z)))
    f_val = SpecialFunctions.erfcx(z_val)

    # d/dz erfcx(z) = 2z·erfcx(z) − 2/√π
    scalar = real(z_val)
    two = oftype(scalar, 2)
    pi_value = oftype(scalar, π)
    df_dz = two * z_val * f_val - two / sqrt(pi_value)

    # Propagate partials via complex chain rule:
    #   df = df_dz · dz,  where dz = d(Re z) + i·d(Im z)
    p_re = partials(real(z))
    p_im = partials(imag(z))
    dr = ntuple(i -> real(df_dz) * p_re[i] - imag(df_dz) * p_im[i], Val(N))
    di = ntuple(i -> imag(df_dz) * p_re[i] + real(df_dz) * p_im[i], Val(N))

    return Complex(
        Dual{T}(real(f_val), Partials(dr)),
        Dual{T}(imag(f_val), Partials(di)),
    )
end

function SpecialFunctions.erfc(z::Complex{Dual{T,V,N}}) where {T,V,N}
    z_val = Complex(value(real(z)), value(imag(z)))
    f_val = SpecialFunctions.erfc(z_val)
    scalar = real(z_val)
    two = oftype(scalar, 2)
    pi_value = oftype(scalar, π)
    df_dz = -two / sqrt(pi_value) * exp(-z_val * z_val)

    p_re = partials(real(z))
    p_im = partials(imag(z))
    dr = ntuple(i -> real(df_dz) * p_re[i] - imag(df_dz) * p_im[i], Val(N))
    di = ntuple(i -> imag(df_dz) * p_re[i] + real(df_dz) * p_im[i], Val(N))
    return Complex(
        Dual{T}(real(f_val), Partials(dr)),
        Dual{T}(imag(f_val), Partials(di)),
    )
end

end # module
