# [Distributional Shadow-Boundary Sensitivity](@id shadow_sensitivity_tutorial)

## Canonical switch and derivative

For a real signed coordinate ``q`` with its shadow boundary at zero, the
supported `exp(+i*omega*t)` convention gives

```math
W(q)=\frac{1}{2}\operatorname{erfc}(-e^{i\pi/4}q),
\qquad
K(q)=W'(q)=\frac{e^{i\pi/4}}{\sqrt\pi}e^{-iq^2}.
```

`shadow_switch(q)` evaluates ``W`` and
`shadow_sensitivity_kernel(q)` evaluates its exact derivative. The real-axis
identities ``W(q)+W(-q)=1`` and ``K(q)=K(-q)`` follow immediately.
Large coordinates in the certified near-real sector
``|\operatorname{Im}q|\leq|\operatorname{Re}q|/8`` use an inverse-power
complementary-error-function tail on the shadow side and symmetry on the
illuminated side. The oscillatory quadratic phase is reduced in an
exponent-aware bounded-precision workspace from the exact stored binary
coordinate before conversion, so a rounded ``q^2`` cannot change the phase by
an order-one angle.

```@example shadow-sensitivity
using UTDKernels

q = 0.4
value = shadow_switch(q)
kernel = shadow_sensitivity_kernel(q)
@assert shadow_switch(q) + shadow_switch(-q) ≈ 1
(value, kernel)
```

For a positive parameter ``\kappa``, the two-argument kernel evaluates

```math
K_\kappa(s)=\sqrt\kappa K(\sqrt\kappa s).
```

Its magnitude is ``\sqrt{\kappa/\pi}`` at every real ``s``. Concentration is
distributional and comes from quadratic-phase cancellation; the kernel does
not acquire a pointwise decaying envelope.

## Exact Fourier multiplier and finite hierarchy

With ``\widehat f(\xi)=\int f(s)e^{-i\xi s}\,ds``, the exact multiplier is

```math
\widehat K_\kappa(\xi)=\exp\left(\frac{i\xi^2}{4\kappa}\right).
```

`shadow_sensitivity_multiplier(xi, kappa)` evaluates this unit-modulus
response. Set `terms=M` with ``1\leq M\leq256`` to retain ``M`` terms of

```math
\sum_{m=0}^{M-1}\frac{1}{m!}
\left(\frac{i\xi^2}{4\kappa}\right)^m.
```

The exact multiplier forms ``\xi^2/(4\kappa)`` from the original stored
inputs in the same bounded phase workspace when direct reduction is unsafe.
A truncated polynomial whose terms require substantial cancellation is summed
in precision selected from the phase and expected bit loss; a request above
the 8192-bit resource ceiling fails instead of returning an uncertified sum.

```@example shadow-sensitivity
xi = 1.7
kappa = 64.0
exact = shadow_sensitivity_multiplier(xi, kappa)
approximation = shadow_sensitivity_multiplier(xi, kappa; terms=3)
@assert abs(exact) ≈ 1
(exact, approximation)
```

For spectrum supported on ``[-\Omega,\Omega]``, the corresponding action
remainder satisfies

```math
|\langle R_{M,\kappa},f\rangle|
\leq\frac{\Omega^{2M}}{(4\kappa)^M M!}
\frac{\|\widehat f\|_1}{2\pi}.
```

The production certificate evaluates this positive expression with an
outward-rounded scaled recurrence. An exactly zero bandwidth or mass gives
zero. A positive bound below the active numeric range rounds upward to the
smallest positive subnormal rather than becoming a false zero.

## Nonlinear-coordinate pullback

Suppose ``g(s_0)=0`` and write
``a=g'(s_0)>0``, ``b=g''(s_0)``, ``c=g'''(s_0)``, and
``d=g^{(4)}(s_0)``. Given local probe derivatives ``f_0,\ldots,f_4``, call

```@example shadow-sensitivity
pullback = shadow_sensitivity_pullback(
    1.0, 0.2, -2.0, 0.3, -12.0,
    1.0, 0.6, 0.0, 0.0, 64.0,
)
pullback
```

The returned named tuple contains the leading action, the complete
``\kappa^{-1}`` correction, and the complete ``\kappa^{-2}`` correction. The
first correction uses

```math
(f\circ g^{-1})''(0)=\frac{f_2}{a^2}-\frac{bf_1}{a^3}.
```

Thus, a width-only rule based on ``a`` misses the curvature term proportional
to ``b f_1`` whenever the probe is asymmetric.
The implementation uses the normalized map derivatives ``b/a``, ``c/a``, and
``d/a`` together with ``1/(\kappa a^2)``. Consequently it preserves the
coordinate-rescaling symmetry even when reciprocal values of ``a`` and
``\kappa`` are individually extreme. Cancellation tests cover both correction
brackets and both outer corrected sums. An ill-conditioned result is recomputed
as the complete `(leading, first, second)` tuple from the original inputs in
successively wider bounded workspaces, and is accepted only after the requested
binary result stabilizes.

## Types, bounds, and scope

Binary32 and binary64 inputs preserve their promoted output type. Binary16 and
binary32 ordinary phase calculations use a binary64 workspace before
conversion. Large phases, near-real switch tails, cancellation-prone finite
series, differentiated kernels, remainder bounds, and extreme pullback
balances use bounded wider workspaces. Differentiated expressions are
converted only after their complete component tree is assembled. Finiteness
checks recurse through ForwardDiff partials, and supported ForwardDiff
coordinates retain their derivatives. Ordinary scalar canonical,
scaled, multiplier, pullback, bound, and half-plane compatibility paths
allocate zero memory after compilation; the exceptional certified recovery
paths allocate their bounded workspaces.

The distributional result applies to a phase-demodulated real simple-pole
branch and to smooth or bandwidth-controlled test quantities. It does not
differentiate the appearance or disappearance of a discrete ray path, select
complex pole sheets, include carrier/spreading/polarization derivatives, or
bound total-field modeling error. Complex-coordinate values are numerical
analytic continuations only; the real-axis distributional claim does not
extend automatically to lossy or complex-ray configurations.
