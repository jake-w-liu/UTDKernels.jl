# [Finite-Edge Endpoint Transitions](@id finite_edge)

An infinite straight-edge ray assumes that the axial propagation stationary
point lies on the edge. A finite segment introduces another transition: the
stationary point can cross either endpoint as the source, observer, or segment
moves. Discarding the ray with a binary segment test creates a discontinuity,
although the finite line integral is continuous.

UTDKernels evaluates the smooth scalar or componentwise integral

```math
I(a,b)=\int_a^b A(s)\exp[-i k\Phi(s)]\,d s,
\qquad
\Phi(s)=R_s(s)+R_p(s),
```

with the package's ``\exp(+i\omega t)`` convention. The formulation assumes a
straight edge, one nondegenerate propagation stationary point, and an amplitude
that is smooth on the stationary endpoint scale.

## Geometry and exact phase coordinate

Construct the source and observer geometry from their positive transverse
ranges and axial coordinates:

```julia
geometry = FiniteEdgeGeometry(
    1.0,   # source transverse range
    1.4,   # observer transverse range
   -0.6,   # source axial coordinate
    0.9,   # observer axial coordinate
)
```

`finite_edge_phase_data(geometry)` returns the stationary coordinate ``s_0``,
stationary phase ``\Phi_0``, and phase derivatives through fourth order. The
signed coordinate returned by `finite_edge_phase_coordinate` satisfies

```math
\Phi(s)=\Phi_0+t(s)^2.
```

Near ``s_0``, the implementation uses a rationalized distance identity instead
of subtracting two nearly equal phase values. Consequently, the coordinate and
its derivative remain finite at endpoint crossing and invariant under a common
translation of all axial coordinates.

## Local amplitude data

Supply the physical amplitude value and its first two axial derivatives at
``s_0``:

```julia
amplitude = FiniteEdgeAmplitude(A0, A1, A2)
```

The amplitude coefficients may be real or complex but must be finite. A vector
or dyadic amplitude is evaluated componentwise with the same geometry and
Fresnel moments.

## Endpoint-uniform evaluation

For physical endpoints `a` and `b`, evaluate the quadratic approximation with

```julia
value = finite_edge_epm(a, b, k, geometry, amplitude; order=2)
```

`order=0`, `1`, or `2` retains the corresponding transformed-amplitude terms.
The evaluator uses closed moments

```math
J_n=\int_{t_a}^{t_b} t^n\exp(-i k t^2)\,d t,
\qquad n=0,1,2.
```

Reversing the endpoints changes the sign of the result. Equal endpoints return
zero. As both scaled endpoints recede, the order-zero term approaches
`finite_edge_stationary_phase`.

## Endpoint and scene derivatives

The derivative with respect to a moving upper endpoint is

```julia
dI_db = finite_edge_endpoint_derivative(b, k, geometry, amplitude)
```

Negate this value for a moving lower endpoint. The endpoint derivative retains
its finite stationary limit. The complete evaluator is ForwardDiff-compatible,
so a caller can differentiate source/observer geometry while rebuilding local
amplitude data from the same differentiable parameters.

When derivatives of the transformed amplitude, stationary phase, and endpoint
coordinates are already available, `finite_edge_parameter_derivative`
assembles the same derivative without differentiating through the evaluator.
This interface keeps geometry-specific differentiation outside the canonical
moment kernel.

## Scope and routing

This API is a local smooth-propagation kernel. It does not supply a complete
incremental diffraction coefficient or a full electromagnetic field model.
Do not polynomially expand a singular shadow-boundary factor across the
stationary endpoint layer. Keep that factor in its appropriate canonical
uniformization or route the contribution to a quadrature that resolves it.

Curved edges, vertices, multiple stationary points, and electrically remote
tails require additional canonical or nonstationary treatment.
