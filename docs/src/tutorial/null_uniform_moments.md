# [Null-Uniform Faddeeva Moments](@id null_uniform_moments_tutorial)

## Canonical moment hierarchy

For a finite canonical pole coordinate ``z``, define

```math
W_m(z)=\frac{i}{\pi}\int_{-\infty}^{\infty}
\frac{t^m e^{-t^2}}{z-t}\,d t,
\qquad \operatorname{Im}z>0
```

on its defining contour. The analytic moments satisfy

```math
W_0(z)=w(z),
\qquad
W_{m+1}(z)=zW_m(z)-\frac{i}{\pi}\mu_m,
```

where odd Gaussian moments vanish and
``\mu_{2q}=\Gamma(q+1/2)``. `faddeeva_moments(z, max_order)` returns
``W_0,\ldots,W_{\mathrm{max\_order}}`` and reuses the same central Faddeeva
function as the multipole transition.

```@example null-moments
using UTDKernels

z = 0.7 + 0.4im
moments = faddeeva_moments(z, 3)
@assert moments[1] == faddeeva_divided_difference([z])
@assert moments[2] ≈ z * moments[1] - im / sqrt(pi)
moments
```

## Certified numerical routing

The forward recurrence carries a propagated roundoff estimate and is
preferred at `abs(z) <= 10`. Each large-argument moment is instead evaluated
from its own least-term inverse-power series,

```math
W_m(z)\sim\frac{i}{\pi}\sum_{q\geq0}\frac{\mu_{m+q}}{z^{q+1}}.
```

The lower-half-plane continuation adds ``2z^m e^{-z^2}``. Selection is
order-aware: a branch is accepted only when its estimate, with a safety
factor, meets the requested relative tolerance. If neither binary branch is
certified, a bounded internal recovery selects at least 128 bits from the
order and expected cancellation loss, up to a 1024-bit ceiling, then evaluates
the shared Faddeeva identity and recurrence before converting to the requested
binary type.

Float32 asymptotic terms are assembled in a Float64 workspace before the
finished moments are rounded, so a representable inverse-power term is not
rejected merely because its unscaled Gaussian moment overflows Float32. A
lower-half-plane continuation whose exponential has already vanished is
short-circuited before its polynomial factor is formed.

Orders 0 through 64 and series bounds 1 through 256 are supported. The default
series bound is 80. The `switch` and `rtol` keywords are nondifferentiated
finite real controls; automatic-differentiation inputs for these routing
parameters are rejected. A certified moderate four-moment call allocates 240 bytes
for one complex result and one real error vector; an algebraic or directly
representable large call allocates only its 144-byte result vector. A balanced
lower-half-plane continuation may use a bounded wide term workspace. A demonstrated
hard order-10 recovery uses about 638 kB; it is a rare accuracy path, is
bounded by the public order limit, and is smaller than the migrated source
path. Complex BigFloat is reserved for independent caller/test oracles rather
than exposed as a production return type.

## Analytic amplitude coefficients

For ascending coefficients ``a_m`` and positive ``k``,
`null_uniform_transition(k, z, coefficients)` evaluates

```math
I(k,z)=\sum_m a_m k^{-m/2}W_m(z).
```

```@example null-moments
coefficients = ComplexF64[0.08, 1.0, 0.35, -0.15, 0.08]
value = null_uniform_transition(250.0, 0.9 + 0.45im, coefficients)
@assert isfinite(value)
value
```

The ordinary path advances one inverse-square-root scale, uses compensated
summation, and skips exact-zero terms before scale multiplication. A zero,
subnormal, or non-finite iterated scale or completed term is reconstructed
from the stored `k` and integer order in a bounded wide workspace. If a moment
or shifted basis loses range, or if the final hierarchy is ill-conditioned,
the complete hierarchy is recomputed from the stored inputs in successively
bounded 256-, 512-, and 1024-bit workspaces. Recovery is accepted only when
two workspaces round to the same requested binary result. It does not copy the
coefficient array. An optional `order` limits the retained coefficient order.
Empty coefficients return zero after the common scalar, control, and explicit
order checks have run.

## Moving higher-order zeros

If the regular amplitude is

```math
a(t)=\left(t+\frac{\Lambda}{\sqrt{k}}\right)^r
\sum_{q\geq0}b_qt^q,
```

then `shifted_null_transition(k, z, Lambda, r, b)` evaluates the corresponding
binomial basis

```math
U_{r,q}(\Lambda,z)=\sum_{j=0}^r
\binom{r}{j}\Lambda^{r-j}W_{q+j}(z).
```

All required moments are computed once. Each ``U_{r,q}`` uses Horner's rule,
so no temporary power array or per-coefficient moment vector is created.
For ``\Lambda=0``, the leading field is proportional to
``k^{-r/2}W_r(z)``.

## Types and differentiation

Float32 and Float64 inputs return the corresponding complex type. ForwardDiff
inputs use bounded analytic wide recomputation for the complete requested
moment vector or physical hierarchy, including derivative components, and
convert only the completed result back to the promoted Dual type. This avoids
certifying a primal recurrence while a differentiated component has already
lost accuracy.

## Scope

These APIs provide scalar canonical moments for coefficients already expanded
in a smooth saddle coordinate. Callers remain responsible for extracting
vector or scalar amplitude coefficients, transporting polarization bases,
choosing pole sheets, restoring spreading and phase factors, and matching
physical diffraction mechanisms. A null caused by cancellation among
different rays is not an amplitude zero within one canonical integral.
