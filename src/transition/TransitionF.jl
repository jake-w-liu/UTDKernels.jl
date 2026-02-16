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
    # Handle x ≈ 0: F(0) = 0 (limit)
    if abs(x) < 1e-30
        # Leading-order: F(x) ≈ √(πx) · e^{+iπ/4} for small x
        return sqrt(π * Complex(x)) * exp(+im * π/4)
    end

    sqrtx = safe_sqrt(x)
    z = exp(+im * π/4) * sqrtx   # argument of erfcx

    return sqrt(π * Complex(x)) * exp(+im * π/4) * erfcx(z)
end
