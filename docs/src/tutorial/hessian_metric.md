# [Coordinate-Invariant Hessian Metrics](@id hessian_metric_tutorial)

## Dual metric and effective distance

Let ``H`` be a real symmetric positive-definite optical-path Hessian and let
``g`` be the nonzero covector normal to a locally linear pole equation. The
coordinate-invariant scalar is

```math
q^2=g^{\mathsf T}H^{-1}g,
\qquad
L_H=\frac{1}{q^2}.
```

`hessian_metric_q2(H, g)` evaluates ``q^2`` by a Cholesky solve; it never forms
`inv(H)`. `hessian_effective_L(H, g)` forms the inverse dual-norm length before
squaring, which preserves representable subnormal and large distances even
when ``q^2`` itself overflows or underflows.

```@example hessian-metric
using UTDKernels

H = [2.0 0.4; 0.4 1.3]
g = [0.8, -0.5]

q2 = hessian_metric_q2(H, g)
L = hessian_effective_L(H, g)
@assert q2 * L ≈ 1
(q2, L)
```

The inputs must use one-based indexing and concrete real element types. `H`
must be finite, square, nonempty, symmetric to input precision, and positive
definite. `g` must have the same dimension and contain a finite nonzero
covector. Invalid matrices are rejected rather than symmetrized or
regularized beyond their input-roundoff tolerance.

## Signed coordinate and transition argument

For a finite signed pole offset ``\delta`` and a finite positive wavenumber
``k``, the canonical quantities are

```math
\chi=\frac{\delta}{q},
\qquad
X=\frac{k\chi^2}{2}.
```

Use `hessian_transition_coordinate(delta, H, g)` for ``\chi`` and
`hessian_transition_argument(k, delta, H, g)` for ``X``. The coordinate keeps
the sign of ``\delta``. Changing only the sign of ``g`` does not change it;
changing both ``\delta`` and ``g`` reverses it while preserving ``X``. A
positive rescaling of the complete pole equation also leaves both quantities
unchanged.

```@example hessian-metric
delta = 0.2
k = 10.0
chi = hessian_transition_coordinate(delta, H, g)
X = hessian_transition_argument(k, delta, H, g)
transition = F_utd(X)
@assert X >= 0 && isfinite(transition)
(chi, X, transition)
```

`hessian_transition_argument` returns the argument only. The existing
`F_utd` implementation remains the single owner of the UTD transition and its
real-axis limiting behavior.

## Principal-radius projection

For two positive principal effective distances, the unit covector at angle
``\beta`` has

```math
L(\beta)=\left(
\frac{\cos^2\beta}{R_1}+\frac{\sin^2\beta}{R_2}
\right)^{-1}.
```

`directional_effective_L(R1, R2, beta)` evaluates this harmonic projection in
a ratio-ordered form that avoids forming ``R_1R_2``. It recovers ``R_1`` and
``R_2`` on their principal axes and the ordinary isotropic distance when the
radii are equal. For example, ``R_1=0.4``, ``R_2=4``, and ``\beta=\pi/4`` give
``L=8/11`` rather than the arithmetic radius ``2.2``.

## Types, memory, and differentiation

Float32 inputs remain Float32, and supported BigFloat inputs retain their
precision. ForwardDiff values propagate through the scalar one-, two-, and
three-dimensional Cholesky formulas; positive-definiteness and dimension
checks remain discrete domain decisions.

After compilation, the metric, distance, coordinate, argument, and
directional functions allocate no memory for dimensions one through three.
Higher dimensions use one ``n\times n`` Cholesky workspace and one length-``n``
solve vector; memory is ``O(n^2)`` and no inverse is formed.

## Scope

The metric supplies a scalar canonical coordinate for a caller-provided local
Hessian and pole covector. It does not derive those inputs from geometry,
choose a pole sheet, or supply residues, spreading, polarization, caustic
uniformization, or complete diffraction amplitudes. Loss of positive
definiteness indicates that this nondegenerate quadratic model is outside its
domain.
