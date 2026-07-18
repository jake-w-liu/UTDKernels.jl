"""
UTD transition function F(x) — Kouyoumjian–Pathak (1974), eq. (12).

Convention: exp(+iωt), outgoing waves ~ exp(−iks).

    F(x) = √(πx) · e^{+iπ/4} · erfcx(e^{+iπ/4}·√x)      (erfcx form)

This equals the KP transition function F_KP(x) directly (standard j = +i).

Properties:
  - F(x) → 1 as Re(x) → +∞  (GTD recovery)
  - F(x) → 0 as x → 0         (transition region)
"""

using SpecialFunctions: erfc, erfcx

"""
    F_utd(x::Number) -> ComplexF64

Evaluate the UTD transition function at `x` (real or complex).
Uses the erfcx representation for numerical stability.
"""
function F_utd(x::Number)
    if x isa Real && isinf(x) && x > zero(x)
        return one(Complex(x))
    end
    isfinite(real(x)) && isfinite(imag(x)) ||
        throw(DomainError(x, "transition argument x must be finite or +Inf on the real axis"))

    # Handle x ≈ 0: F(0) = 0 (limit). The leading-order branch below is exact to
    # relative order O(x), so a √eps floor on the dimensionless detour parameter |x|
    # both avoids the degenerate erfcx evaluation and keeps the truncation error at
    # machine precision. AD-safe: floor derived from the primal real float type.
    F_utd_zero_floor = sqrt(eps(float(real(typeof(x)))))
    if abs(x) < F_utd_zero_floor
        # Leading-order: F(x) ≈ √(πx) · e^{+iπ/4} for small x
        return sqrt(π) * safe_sqrt(x) * exp(+im * π/4)
    end

    sqrtx = safe_sqrt(x)
    z = exp(+im * π/4) * sqrtx   # argument of erfcx

    return sqrt(π) * sqrtx * exp(+im * π/4) * erfcx(z)
end
