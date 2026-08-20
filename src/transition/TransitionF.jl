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
    F_utd(x::Number) -> Complex

Evaluate the UTD transition function at `x` (real or complex).
Uses the erfcx representation for numerical stability.

Throws `ArgumentError` when `SpecialFunctions.erfcx` has no method for the
scaled complex type derived from `x`. This currently includes finite
`BigFloat` inputs because complex-`BigFloat` erfcx is unavailable.
"""
function F_utd(x::Number)
    if x isa Real && isinf(x) && x > zero(x)
        return one(Complex(x))
    end
    isfinite(real(x)) && isfinite(imag(x)) ||
        throw(DomainError(x, "transition argument x must be finite or +Inf on the real axis"))

    # The erfcx form is exact across the whole domain, including x → 0:
    # √(πx)·e^{+iπ/4}·erfcx(e^{+iπ/4}√x) → √0·erfcx(0) = 0 to machine precision for
    # arbitrarily small |x| (erfcx(0)=1 is non-degenerate). No small-argument
    # surrogate is used. The dropped-erfcx leading-order form √(πx)·e^{+iπ/4} has
    # relative error (2/√π)√x = O(√x) — ≈1.4e-4 at |x|=√eps, ~11 orders above
    # machine precision — so a √eps floor injected that truncation error plus a
    # ~1.4e-4 jump discontinuity for no benefit. AD note: the surrogate was not
    # Dual protection — erfcx(::Complex{Dual}) is supplied by the ForwardDiff
    # package extension, so this form differentiates exactly for all x>0, and
    # exactly at x=0 both forms share the identical safe_sqrt(x) branch-point
    # derivative, so removing the surrogate does not change AD behavior there.
    sqrtx = safe_sqrt(x)
    z = exp(+im * π/4) * sqrtx   # argument of erfcx
    scaled_erfcx = try
        erfcx(z)
    catch err
        err isa MethodError || rethrow()
        throw(ArgumentError(
            "F_utd does not support $(typeof(x)): erfcx is unavailable for $(typeof(z))",
        ))
    end

    return sqrt(π) * sqrtx * exp(+im * π/4) * scaled_erfcx
end
