# [Curvature-Measure Seam Limits](@id curvature_measure)

## Intrinsic seam coefficient

A convex faceted approximation replaces smooth normal rotation by positive
exterior turning angles ``\delta_j`` at its seams. On the symmetric PEC
reflection boundary, `intrinsic_seam_coefficient(delta, k, L)` evaluates the
intrinsic component from the central face--edge decomposition. It does not
reconstruct the coefficient by subtracting complete soft and hard fields.

For small turning angle,

```math
D_e(\delta;k,L)=A(k,L)\,\delta+O(\delta^2),
```

and the exact normalized local bias is

```math
\beta(\delta)=\frac{D_e}{A\delta}
=\frac{2\tan[\delta/(2n)]}{n\delta},
\qquad n=1+\frac{\delta}{\pi}.
```

The implementation evaluates the small-angle series before subtraction loses
``\beta-1``. It accepts finite ``0<\delta\leq\pi`` and finite positive `k` and
`L`.

## Supplied turning measures

For amplitudes ``q_j``, the normalized raw and debiased sums are

```math
S_{\rm raw}=\sum_j\beta(\delta_j)\,\delta_j q_j,
\qquad
S_{\rm debiased}=\sum_j\delta_j q_j.
```

```@example curvature-measure
using UTDKernels

N = 32
turning = fill(2pi / N, N)
theta = 2pi .* (0:N-1) ./ N
amplitudes = @. exp(2im * theta - 12im * cos(theta - 0.2))

raw = curvature_measure_sum(turning, amplitudes)
debiased = debiased_curvature_measure_sum(turning, amplitudes)
@assert isfinite(raw) && isfinite(debiased)
round.((raw, debiased), digits=6)
```

The two arrays must have identical axes and concrete numeric element types.
Every turn must lie in ``(0,\pi)``. By default, compensated summation must close
the positive exterior measure to ``2\pi``; `require_closed=false` permits an
explicitly verified open arc. Negative/CW turning data are rejected instead of
being silently reoriented. The closure tolerance includes one unit in the last
place at the stored ``2\pi`` endpoint, so ordinary `Float16` angle
quantization is accepted without admitting visibly open partitions. Both
scalar accumulation paths allocate no memory after compilation for built-in
fixed-precision floating inputs. They do not reorder or copy caller arrays.
All-`Float16` turning data are accumulated in `Float32`. When an input contains
`BigFloat` values, evaluation uses the highest precision stored in those values
rather than the ambient default precision. All `BigFloat` turning angles in
one call must share a stored precision.

## Circular harmonic reference

`curvature_continuum_harmonic(m, z, phi)` evaluates

```math
\int_0^{2\pi}e^{im\theta}e^{-iz\cos(\theta-\phi)}\,d\theta
=2\pi(-i)^mJ_m(z)e^{im\phi}.
```

This closed form is useful for testing polygonal aliasing and convergence of a
turning measure. General geometry generation, ellipse sampling, alias-series
truncation, and application-specific smooth amplitudes remain caller or paper
responsibilities. The harmonic follows the same `Float16` widening and stored
`BigFloat` precision policy as the supplied-measure sums. Fixed-precision
orders must satisfy `-typemax(Cint) < m < typemax(Cint)`, and `BigFloat`
orders must satisfy `typemin(Clong) < m < typemax(Clong)`. These open bounds
keep both neighboring Bessel orders used by automatic differentiation inside
the numerical backend's integer range.

## Scope

The debiased sum is a seam-consistency diagnostic for a prescribed local
amplitude. It is not a complete faceted-object field, an error bound for edge
removal, or a creeping-wave replacement. It does not supply visibility,
multiple diffraction, geodesic transport, caustics, polarization coupling, or
surface-wave physics. Those mechanisms require their own asymptotic models and
global assembly.
