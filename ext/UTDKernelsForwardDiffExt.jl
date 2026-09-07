"""
Package extension: ForwardDiff support for UTDKernels.

Provides a forward-mode AD rule for `erfcx(::Complex{Dual})` so transition
values, passive derivatives, and wedge coefficients can be differentiated
through by ForwardDiff.

The complex derivative of erfcx is:
    d/dz erfcx(z) = 2z·erfcx(z) − 2/√π
"""
module UTDKernelsForwardDiffExt

import SpecialFunctions
import ForwardDiff: Dual, Tag, value, partials, Partials

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

end # module
