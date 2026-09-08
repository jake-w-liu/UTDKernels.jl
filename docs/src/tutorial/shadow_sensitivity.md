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

The production certificate evaluates this expression in the log domain. An
exactly zero bandwidth or mass gives zero. A positive bound below the active
numeric range rounds upward to the smallest positive subnormal rather than
becoming a false zero.

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

## Types, bounds, and scope

Binary32 and binary64 inputs preserve their promoted output type. Binary16 and
binary32 phase calculations use a binary64 workspace before conversion, which
keeps low-precision quadratic phases from accumulating avoidable rounding.
Supported ForwardDiff coordinates retain their derivatives. Scalar canonical,
scaled, multiplier, pullback, bound, and half-plane compatibility paths
allocate zero memory after compilation.

The distributional result applies to a phase-demodulated real simple-pole
branch and to smooth or bandwidth-controlled test quantities. It does not
differentiate the appearance or disappearance of a discrete ray path, select
complex pole sheets, include carrier/spreading/polarization derivatives, or
bound total-field modeling error. Complex-coordinate values are numerical
analytic continuations only; the real-axis distributional claim does not
extend automatically to lossy or complex-ray configurations.
