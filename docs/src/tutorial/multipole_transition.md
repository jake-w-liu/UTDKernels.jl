# [Clustered-Pole Multipole Transitions](@id multipole_transition_tutorial)

## Canonical function

For finite canonical nodes ``z_1,\ldots,z_p``, the multipole transition is the
Faddeeva divided difference

```math
\mathcal W_p(z_1,\ldots,z_p)
=w[z_1,\ldots,z_p]
=\sum_{j=1}^{p}\frac{w(z_j)}{\prod_{\ell\ne j}(z_j-z_\ell)},
\qquad w(z)=\operatorname{erfcx}(-iz).
```

The direct formula is well suited to distinct separated nodes. Its individual
terms grow and cancel when nodes merge, although the analytic divided
difference remains finite. At complete coalescence,

```math
\mathcal W_p(z,\ldots,z)=\frac{w^{(p-1)}(z)}{(p-1)!}.
```

`faddeeva_divided_difference(nodes)` evaluates the direct representation and
therefore requires distinct nodes. Its `return_condition=true` form also
returns ``\sum_j|T_j|/|\sum_jT_j|``, which estimates cancellation in the
barycentric sum.

## Automatic confluent evaluation

`multipole_transition(nodes)` calculates the arithmetic center ``m`` and uses
the centered expansion

```math
\mathcal W_p=\sum_{q=0}^{\infty}
\frac{w^{(p-1+q)}(m)}{(p-1+q)!}
h_q(z_1-m,\ldots,z_p-m),
```

when the centered radius or direct cancellation estimate makes the
barycentric basis unsuitable. The scaled derivative recurrence forms
``w^{(n)}/n!`` without integer factorials. Complete homogeneous coefficients
use one bounded work vector, and exact repeated nodes need no workspace.

```@example multipole-transition
using UTDKernels

separated = ComplexF64[-0.6 + 0.8im, 0.2 + 1.1im, 0.9 + 0.7im]
direct, cancellation = faddeeva_divided_difference(
    separated; return_condition=true,
)
automatic, info = multipole_transition(separated; return_info=true)

@assert info.method === :direct
@assert automatic == direct
(automatic, cancellation, info.terms_used)
```

`MultipoleEvaluationInfo.method` is `:single`, `:direct`, or `:cluster`. The
remaining fields record the node count, center, centered radius, direct
cancellation estimate, and retained cluster terms. Equality at a selection
boundary takes the safer cluster representation.

## Contract and numerical limits

The public functions accept nonempty vectors with a concrete numeric element
type and finite nodes. Orders 1 through 7 are supported; this bound matches the
validated coalescence, confluent-recurrence, and contour evidence. The automatic
series uses at most 48 terms by default and accepts a caller bound from 2
through 128. Failure to obtain the required run of small terms raises an error
instead of returning an uncertified truncation.

Float32 inputs remain ComplexF32. ForwardDiff values propagate through a
fixed selected representation; method selection itself is discrete at its
documented boundaries. Complex BigFloat evaluation is rejected because the
production SpecialFunctions backend does not supply complex-BigFloat `erfcx`;
callers needing a high-precision oracle should use an independent entire
series or contour calculation.

After compilation, single-node, distinct direct, automatic-direct, and exact
repeated-node calls allocate no memory. A nonrepeated cluster call allocates
one vector proportional to `max_terms`; no node copy, term array, derivative
array, or quadratic workspace is formed.

## Physical scope

This API evaluates only the scalar Gaussian saddle interaction with a finite
pole cluster. The caller must supply the canonical nodes and determine the
physical numerator, residues, pole sheets, and boundary conditions. Crossing
a contour requires explicit residue bookkeeping. The canonical factor does
not perform geometry-specific mechanism matching or provide a complete
multipole diffraction coefficient.
