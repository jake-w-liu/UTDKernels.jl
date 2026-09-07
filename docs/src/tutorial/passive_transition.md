# [Passive-Sheet Complex Transitions](@id passive_transition)

## Physical sector

UTDKernels uses the ``\exp(+i\omega t)`` convention and outgoing propagation
``\exp(-ikr)``. A passive homogeneous wavenumber therefore has
``\operatorname{Re}k>0`` and ``\operatorname{Im}k\leq0``. Multiplication by a
nonnegative real geometrical factor maps the transition argument into

```math
\operatorname{Re}x\geq0,\qquad \operatorname{Im}x\leq0.
```

`is_passive_transition_argument` checks this sector with a scale-aware
64-ulp allowance on the axes. It returns `false` for nonfinite values and does
not clamp, reflect, or conjugate its input.

```@example passive-transition
using UTDKernels

k = passive_wavenumber(20.0, 0.1) # 20 - 2im
x = 1.3k
@assert is_passive_transition_argument(x)

F = F_utd(x)
Fm1 = F_utd_minus_one(x)
Fp = F_utd_prime(x)
Fpp = F_utd_second(x)
round.((F, Fm1, Fp, Fpp), digits=6)
```

`passive_wavenumber(k0, attenuation)` requires finite `k0>0` and
`attenuation>=0` and returns `k0*(1-im*attenuation)`.

## Branch and representation policy

All transition mappings use the principal square root. The passive
material-wave helper `radiation_sqrt` is intentionally not used here: its
lower-bank value on the negative-real cut serves a different physical
quantity.

One internal regime selector evaluates ``F``, ``F-1``, ``F'``, and ``F''``
together:

- a convergent series in ``\sqrt{x}`` handles small ``|x|``;
- scaled complementary-error-function evaluation handles intermediate
  arguments; and
- a differentiated inverse-power series handles large ``|x|``.

The residual switches to its inverse-power form before direct subtraction
loses `F-1`. The large-argument derivatives likewise avoid subtracting two
order-one terms in the transition differential equation. These complex-sector
switches are separate from the established positive-real thresholds of
`F_utd_minus_one(::Real)` and `F_utd_prime(::Real)`; those methods retain their
existing behavior.

`F_utd_minus_one(::Complex)`, `F_utd_prime(::Complex)`, and
`F_utd_second(::Complex)` are passive-only. A finite wrong-sheet request raises
`PassiveSheetError`; a zero derivative request raises `DomainError`.
Float32, Float64, and ForwardDiff values based on those types are supported.
Finite complex-BigFloat scaled erfc is unavailable from the package dependency
and is rejected explicitly.

## General complex continuation

`F_utd` predates the passive derivative API and remains the general
principal-root analytic-continuation evaluator. Nonpassive complex arguments
continue to use that API. The passive-only restriction applies to the
complex residual and derivative methods, not to `F_utd` itself.

For wedge coefficients, a passive complex `k` with finite positive real `L`
uses the shared passive transition backend away from a cotangent pole. The
existing regularized cotangent--transition product still handles coincident
shadow/reflection limits. A complex effective distance retains the general
analytic-continuation path; this extension does not reinterpret it as a
physical passive-sheet distance.

## Limiting absorption

Lossless positive-real values are the boundary limit of the passive sector:

```math
\lim_{\eta\to0^+}F\!\left[x(1-i\eta)\right]=F(x),\qquad x>0.
```

This limit is continuous, but it does not change the package's phasor
convention or authorize the opposite square-root sign. Selecting the negative
principal root adds an exponentially growing complementary solution in the
loss direction.
