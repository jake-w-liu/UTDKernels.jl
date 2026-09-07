# [Bivariate Fresnel Transitions](@id bivariate_transition)

## Why correlation matters

Two canonical boundaries with normalized mixed curvature ``\rho`` do not, in
general, switch independently. The product of two one-boundary Fresnel factors
is exact only at ``\rho=0``. UTDKernels evaluates the coupled factor as

```math
T_2(\xi,\eta;\rho)
=T_1(\xi)T_1(\eta)
+\int_0^\rho
\frac{\exp\!\left[-\frac{i}{2(1-r^2)}
(\xi^2-2r\xi\eta+\eta^2)\right]}
{2\pi\sqrt{1-r^2}}\,d r.
```

The substitution ``r=\sin\theta`` removes the algebraic endpoint factor and
turns the correction into a finite integral from zero to
``\arcsin\rho``.

```@example bivariate
using UTDKernels

value = bivariate_fresnel_transition(0.3, -0.7, 0.4)
fixed = bivariate_fresnel_transition(0.3, -0.7, 0.4; order=96)
@assert value ≈ fixed rtol=3e-12

weights = bivariate_mechanism_weights(0.3, -0.7, 0.4)
@assert sum(weights) ≈ 1
round.(weights, digits=6)
```

Coordinates must be finite. The real correlation must satisfy
``|\rho|<1``. At exactly zero correlation, the evaluator returns the product
without quadrature.

## Adaptive and fixed evaluation

The default uses QuadGK. Its estimate must satisfy
`max(atol, rtol*abs(correction))`; otherwise the call raises `DomainError`.
The defaults are `rtol=2e-13` and `atol=2e-13`.

An explicit positive `order` selects a fixed Gauss--Legendre rule. Orders are
bounded to `1:2048`. The shared cache is locked during lookup and first
construction, stores only two ``O(n)`` vectors per requested order, and never
returns those mutable vectors from a public API. The grazing continuation keeps
its narrower existing `1:256` order contract even though both families share
the cache.

Fixed quadrature is useful for deterministic bulk evaluation and supports
ForwardDiff on a smooth branch. Adaptive quadrature is the certified default,
especially as ``|\rho|`` approaches one. A fixed order should be checked
against the adaptive result for the intended coordinate and correlation range.

## Four mechanism weights

`bivariate_mechanism_weights` returns `(W00, W10, W01, W11)`. Algebraically,

```math
\sum_{a,b\in\{0,1\}}W_{ab}=1,
\qquad W_{10}+W_{11}=T_1(\xi),
\qquad W_{01}+W_{11}=T_1(\eta).
```

These are complex canonical switching weights, not probabilities. Their
partition and marginal identities are the useful invariants.

## Mapping a quadratic phase

For

```math
\frac{k}{2}\left(a u^2+2buv+c v^2\right),
```

`bivariate_transition_hessian(a,b,c,u,v;k)` returns

```math
\rho=-\frac{b}{\sqrt{ac}},\qquad
\xi=\sqrt{ka(1-\rho^2)}\,u,\qquad
\eta=\sqrt{kc(1-\rho^2)}\,v.
```

The Hessian must be positive definite, with finite `a`, `b`, `c`, `u`, `v`,
and finite positive `k`. Near rank one, the normalized determinant is computed
from power-of-two-scaled products and fused-product residuals. This avoids both
dimensional overflow and an allocating arbitrary-precision fallback while
retaining the low product bits visible in ``1-\rho^2``.

## Scope

This API supplies a canonical correlation-aware factor. It does not include
electromagnetic amplitudes, polarization transport, visibility, multiple-edge
path topology, or a complete double-diffraction coefficient. At
``|\rho|=1`` the quadratic form loses rank and the outer formula is singular;
the API rejects that boundary instead of assigning an inner-layer limit.
