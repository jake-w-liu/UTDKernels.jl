# [Continuous-Order Saddle--Endpoint Transitions](@id continuous_order_tutorial)

## Canonical family

For a positive endpoint order ``\mu`` and finite canonical coordinate
``\zeta``, define

```math
\mathcal B_\mu(\zeta)=\frac{2}{\Gamma(\mu/2)}
\int_0^\infty u^{\mu-1}\exp(-u^2+2\zeta u)\,du.
```

`continuous_order_transition(mu, zeta)` evaluates this family with
``\mathcal B_\mu(0)=1``. The order is continuous: noninteger values are
evaluated directly rather than interpolated between integer cases.

```@example continuous-order
using UTDKernels

orders = (0.5, 1.0, 1.5, 2.0)
values = [continuous_order_transition(mu, 0.4 + 0.3im) for mu in orders]
@assert all(isfinite, values)
values
```

At exact unit order,

```math
\mathcal B_1(\zeta)=\exp(\zeta^2)\operatorname{erfc}(-\zeta)
=w(-i\zeta).
```

The implementation reuses the package's Faddeeva backend for this member. It
therefore has the same branch convention as `F_utd` and the multipole and
null-uniform families. For ``X\geq0``, the conventional transition satisfies

```math
F(X)=\sqrt{\pi X}\,e^{i\pi/4}
\mathcal B_1(-e^{i\pi/4}\sqrt X).
```

## Certified half-line quadrature

On the first transformed interval, the substitution ``u=x^{1/\mu}`` removes
the integrable singularity that occurs when ``0<\mu<1``. A second exact scale
tracks the shrinking endpoint layer for large negative ``\Re\zeta``. The
normalized integrand is evaluated in the log domain, and its tail is split at
the analytic mode.

Each transformed piece uses an embedded Gauss--Kronrod rule. The rescaled
piecewise error estimates are added and must meet the requested mixed
tolerance; an uncertified value raises `DomainError`. `maxevals` is bounded to
`63:10_000_000` for each transformed piece. The default relative tolerance is
`2e-13` for binary64 and at least 128 machine epsilons for lower precision.

Unscaled positive saddle coordinates eventually exceed the active numeric
range. Such calls fail rather than return `Inf` or a partial quadrature.

## Scaled saddle branch

For a finite real saddle coordinate,
`scaled_continuous_order_transition(mu, zeta)` evaluates

```math
e^{-\zeta^2}\mathcal B_\mu(\zeta)
```

with the exponential shift inside the log-domain integrand. It does not first
construct the overflowing unscaled value. Unit order reduces exactly to
``\operatorname{erfc}(-\zeta)``.

```@example continuous-order
scaled = scaled_continuous_order_transition(1.0, 30.0)
@assert scaled == 2.0
scaled
```

## Analytic-amplitude moments

For local endpoint order ``\nu``, positive ``k`` and quadratic curvature
``h``, signed displacement ``\tau``, and ascending amplitude coefficient
``a_m``, `continuous_order_moment(nu, m, k, h, tau, a_m)` evaluates

```math
J_m=a_m e^{-i\pi(\nu+m)/4}\frac{\Gamma[(\nu+m)/2]}{2}
\left(\frac{2}{kh}\right)^{(\nu+m)/2}
\mathcal B_{\nu+m}(\zeta),
\qquad
\zeta=e^{i\pi/4}\tau\sqrt{\frac{k}{2h}}.
```

The coefficient-vector overload evaluates ``\sum_m J_m``. It validates
coefficients in place, uses compensated scalar summation, and creates no
coefficient or term array. `order` may truncate the available coefficients;
moment indices are bounded to 0 through 64.

```@example continuous-order
coefficients = ComplexF64[1.0, -0.35 + 0.2im, 0.12]
hierarchy = continuous_order_moment(
    0.8, coefficients, 50.0, 1.3, 0.15,
)
@assert isfinite(hierarchy)
hierarchy
```

## Types and differentiation

Binary32 and binary64 calls preserve their promoted complex type; binary16 is
widened to binary32. ForwardDiff coordinates propagate through a fixed
quadrature route, and both the unscaled and scaled coordinate derivatives are
covered by independent identities. Endpoint-order differentiation is a
separate logarithmic-moment calculation; automatic differentiation of
``\mu``, ``k``, ``h``, or ``\tau`` is rejected instead of silently converting
to binary64. BigFloat is reserved for independent caller or test oracles.

## Scope

This API supplies a coalescence-normalized scalar family that recovers the
ordinary UTD transition at unit order, plus physical moments after a local
quadratic saddle--endpoint reduction. It does not claim
that parabolic-cylinder saddle--endpoint asymptotics are new. The caller still
owns the geometry reduction, endpoint exponent, vector coefficients,
polarization transport, exterior ray phases, nonstationary remainder, branch
cuts crossed by contour rotation, and matching to other canonical mechanisms.
