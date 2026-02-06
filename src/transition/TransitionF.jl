"""
UTD transition function F(x) for the exp(+iωt) convention.

Reference definition (corrected Kouyoumjian–Pathak form):

    F(x) = √(πx) · exp(-i(π/4 + x)) · erfc(exp(-iπ/4)·√x)

Properties:
  - F(x) → 1 as Re(x) → +∞  (GTD recovery)
  - F(x) → 0 as x → 0         (transition region)
  - F(x) = -2i√x · e^{-ix} · ∫_{√x}^∞ e^{it²}dt  (integral form)
"""

using SpecialFunctions: erfc, erfcx

"""
    F_utd(x::Number) -> ComplexF64

Evaluate the UTD transition function at `x` (real or complex).
Uses the erfc representation for numerical stability.
"""
function F_utd(x::Number)
    # Handle x ≈ 0: F(0) = 0 (limit)
    if abs(x) < 1e-30
        # Leading-order: F(x) ≈ √(πx) · e^{-iπ/4} for small x
        return sqrt(π * Complex(x)) * exp(-im * π/4)
    end

    sqrtx = safe_sqrt(x)
    z = exp(-im * π/4) * sqrtx   # argument of erfc

    # For large |z| with Re(z) > 0, use erfcx to avoid underflow:
    #   erfc(z) = erfcx(z) · exp(-z²)
    #   F(x) = √(πx) · e^{-iπ/4} · e^{-ix} · erfcx(z) · exp(-z²)
    # Note: z² = exp(-iπ/2)·x = -ix, so exp(-z²) = exp(ix).
    # Therefore: F(x) = √(πx) · e^{-iπ/4} · erfcx(z)

    # Since z² = -ix:
    #   e^{-i(π/4+x)} · erfc(z) = e^{-iπ/4} · e^{-ix} · e^{-z²} · erfcx(z)
    #                             = e^{-iπ/4} · e^{-ix} · e^{ix} · erfcx(z)
    #                             = e^{-iπ/4} · erfcx(z)
    # So: F(x) = √(πx) · e^{-iπ/4} · erfcx(z)

    return sqrt(π * Complex(x)) * exp(-im * π/4) * erfcx(z)
end
