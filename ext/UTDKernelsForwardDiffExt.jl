"""
Package extension: ForwardDiff support for UTDKernels.

Provides a forward-mode AD rule for `erfcx(::Complex{Dual})` so that
`F_utd` and `pec_wedge_DsDh` can be differentiated through by ForwardDiff.

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
    df_dz = 2 * z_val * f_val - 2 / sqrt(π)

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
