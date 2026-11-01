"""
Branch-aware square-root utilities.

Mathematical kernels use the principal branch. Material-wave quantities use
the passive limiting-absorption value on the negative-real branch cut.
"""

"""
    safe_sqrt(x::Number)

Principal-branch square root. For real negative x, returns i√|x|.
This is the standard `sqrt` behavior in Julia, but we wrap it to make
the branch policy explicit.
"""
function safe_sqrt(x::Real)
    primal = _primal_value(x)
    if primal > zero(primal)
        return complex(sqrt(x))
    elseif primal < zero(primal)
        # Convert before negation so typemin(Int) cannot wrap. Real square
        # roots also avoid spurious ComplexDual range failures away from zero.
        working = float(x)
        return complex(zero(working), sqrt(-working))
    end
    # Preserve the principal branch-point behavior, signed zeros and NaNs.
    return sqrt(complex(x))
end
safe_sqrt(x::Complex) = sqrt(x)

"""
    radiation_sqrt(x::Number)

Square root for material-wave longitudinal quantities under the package's
`exp(+iωt)` and `exp(-ik_z z)` conventions. Away from the negative-real branch
cut this is the principal square root. At an exactly negative-real argument it
returns the passive lower-bank limit, `-im * sqrt(abs(x))`, which is continuous
with passive material inputs whose imaginary part approaches zero from below.

This helper is intentionally separate from [`safe_sqrt`](@ref): transition
functions and other mathematical kernels still require the principal branch.
"""
@inline function radiation_sqrt(x::Number)
    root = safe_sqrt(x)
    real_x = _primal_value(real(x))
    imag_x = _primal_value(imag(x))
    root_imag = _primal_value(imag(root))
    if iszero(imag_x) && real_x < zero(real_x) && root_imag > zero(root_imag)
        # Julia's principal sqrt chooses the upper-bank value (+i√|x|) when the
        # imaginary part is +0. The passive exp(+iωt) limit approaches the cut
        # from Im(x)<0 and therefore requires the opposite sign.
        return -root
    end
    return root
end
