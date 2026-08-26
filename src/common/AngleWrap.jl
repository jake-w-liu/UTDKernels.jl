"""
Angle normalization utilities.
"""

"""
    wrap_angle(phi, alpha)

Wrap angle `phi` into [0, alpha).

Ordinary real inputs use Julia's robust `mod`, including when `abs(phi/alpha)`
is too large for the algebraic expression `phi - fld(phi, alpha)*alpha` to be
accurate or finite. For forward-mode AD numbers, the same primal `mod` value is
combined with the local tangent of `phi - q*alpha`, where
`q = fld(primal(phi), primal(alpha))`. This keeps the value identical to `mod`
while retaining a finite unit derivative with respect to `phi` at an exact
periodic seam.
"""
function wrap_angle(phi::Real, alpha::Real)
    isfinite(phi) || throw(DomainError(phi, "angle phi must be finite"))
    alpha_primal = _primal_value(alpha)
    isfinite(alpha_primal) && alpha_primal > zero(alpha_primal) ||
        throw(DomainError(alpha, "wrap interval alpha must be finite and positive"))

    # Base.mod is both more accurate and more overflow-resistant than forming
    # phi - fld(phi, alpha)*alpha for ordinary floating-point inputs.
    if !(hasproperty(phi, :value) || hasproperty(alpha, :value))
        return mod(phi, alpha)
    end

    # Reconstruct an AD value from the robust primal remainder and the
    # piecewise-linear tangent. `zero(x) + primal(x)` preserves the AD scalar
    # type while giving it zero partials.
    phi0 = _primal_value(phi)
    alpha0 = _primal_value(alpha)
    remainder = mod(phi0, alpha0)
    wrapped = zero(phi) + zero(alpha) + remainder
    wrapped += phi - (zero(phi) + phi0)

    alpha_tangent = alpha - (zero(alpha) + alpha0)
    if !iszero(alpha_tangent)
        q = fld(phi0, alpha0)
        isfinite(q) || throw(DomainError(
            (phi, alpha),
            "angle-to-period ratio is too large for a finite derivative with respect to alpha",
        ))
        wrapped -= q * alpha_tangent
    end
    return wrapped
end
