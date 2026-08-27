# [Numerical Methods](@id numerical)

This chapter describes five numerical challenges that arise when evaluating UTD diffraction coefficients in IEEE 754 floating-point arithmetic, and the solutions implemented in UTDKernels.jl.

## Challenge 1: Overflow in the transition function

### The problem

The erfc representation of the transition function is

```math
F(x) = \sqrt{\pi x}\;e^{+i(\pi/4 + x)}\;\operatorname{erfc}\!\bigl(e^{+i\pi/4}\sqrt{x}\bigr).
```

For large real ``x``, let ``z = e^{+i\pi/4}\sqrt{x}``. Then ``|z| = \sqrt{x}`` grows without bound, and the large-``z`` asymptotic of ``\operatorname{erfc}`` is

```math
\operatorname{erfc}(z) \sim \frac{e^{-z^2}}{\sqrt{\pi}\,z}
= \frac{e^{-ix}}{\sqrt{\pi}\,e^{+i\pi/4}\sqrt{x}}
\qquad (x \to +\infty,\; z=e^{+i\pi/4}\sqrt{x}).
```

So for this argument ray, ``\operatorname{erfc}(z)`` does **not** decay like ``e^{-x}`` in magnitude; the leading magnitude is ``O(x^{-1/2})``.
The numerical difficulty in the direct erfc form is instead a **conditioning/cancellation** issue:
``e^{+ix}`` in the prefactor cancels the ``e^{-ix}`` hidden inside ``\operatorname{erfc}(z)``.
At large ``x``, this cancellation can lose relative accuracy, while the true result should approach ``F(x)\to 1``.

In IEEE 754 double precision, this manifests as loss of relative accuracy in the direct representation:

```math
\underbrace{e^{+ix}}_{\text{rapid phase}} \times
\underbrace{\operatorname{erfc}(z)}_{\text{contains }e^{-ix}\text{ factor}}
\;\;\Rightarrow\;\; \text{cancellation-sensitive evaluation}.
```

### The solution: erfcx

The **scaled complementary error function**

```math
\operatorname{erfcx}(z) = e^{z^2}\operatorname{erfc}(z)
```

absorbs the exponential decay into the scaling factor. As shown in [The UTD Transition Function](@ref transition), the erfc and erfcx forms are related by the identity ``z^2 = +ix``, which leads to the cancellation

```math
e^{+i(\pi/4+x)} \cdot \operatorname{erfc}(z) = e^{+i\pi/4} \cdot \operatorname{erfcx}(z),
```

giving the numerically stable form:

```math
F(x) = \sqrt{\pi x}\;e^{+i\pi/4}\;\operatorname{erfcx}\!\bigl(e^{+i\pi/4}\sqrt{x}\bigr).
```

The function ``\operatorname{erfcx}(z)`` is bounded for ``\operatorname{Re}(z) \ge 0``:

```math
|\operatorname{erfcx}(z)| \le \max\!\left(1,\;\frac{1}{\sqrt{\pi}\,\operatorname{Re}(z)}\right).
```

No exponentially large or small intermediate values appear in the implemented formula. In the validated operating regimes used by the package tests and examples, the result tracks reference values to near machine precision.

```@example numerical
using UTDKernels
# erfcx form works for arbitrarily large x
for x in [1e2, 1e4, 1e8, 1e12]
    println("F($x) = $(F_utd(x)),  |F| = $(abs(F_utd(x)))")
end
```

## Challenge 2: The cotangent--transition-function singularity

### The problem

The diffraction coefficient contains the product ``\cot(\psi_j) \cdot F(X_j)``. At every shadow or reflection boundary:

- ``\sin(\psi_j) \to 0``, so ``\cot(\psi_j) = \cos(\psi_j)/\sin(\psi_j) \to \pm\infty``.
- ``a_j \to 0``, so ``X_j = kLa_j \to 0``, and ``F(X_j) \to 0``.

The product ``\cot(\psi_j) \cdot F(X_j)`` has a **finite limit** (this is the entire point of UTD), but the naive floating-point evaluation produces

```math
\underbrace{\cot(\psi_j)}_{\to \pm\infty} \times \underbrace{F(X_j)}_{\to 0} = \texttt{NaN}.
```

### Analysis of the limit

Near a boundary, let the angular deviation from the exact boundary be ``\delta`` (small). From the [KP Coefficients](@ref kp) chapter, both ``\sin(\psi_j)`` and ``g_j = \cos((2n\pi N_j - \beta_j)/2)`` vanish **linearly** in ``\delta``:

```math
\sin(\psi_j) \approx (-1)^m \cdot \frac{\delta}{2n}, \qquad g_j \approx \frac{\delta}{2},
```

where ``m`` is the boundary integer. Therefore:

```math
a_j = 2g_j^2 \approx \frac{\delta^2}{2}, \qquad X_j = kLa_j \approx \frac{kL\delta^2}{2}, \qquad \sqrt{X_j} \approx \sqrt{\frac{kL}{2}}\,|\delta|.
```

The ratio that appears in the regularised form is

```math
\frac{|g_j|}{|\sin(\psi_j)|} = \frac{|\delta|/2}{|\delta|/(2n)} = n.
```

This shows that the linear parts cancel, and the ratio converges to the wedge parameter ``n``.

### The regularised form

Starting from the erfcx representation of ``F``:

```math
\cot(\psi_j)\,F(X_j) = \frac{\cos(\psi_j)}{\sin(\psi_j)} \cdot \sqrt{\pi X_j}\;e^{+i\pi/4}\;\operatorname{erfcx}(z_j),
```

where ``z_j = e^{+i\pi/4}\sqrt{X_j}``. Rearranging:

```math
\boxed{\cot(\psi_j)\,F(X_j) = \cos(\psi_j) \cdot \frac{\sqrt{\pi X_j}}{\sin(\psi_j)} \cdot e^{+i\pi/4} \cdot \operatorname{erfcx}(z_j).}
```

In this form:
- ``\cos(\psi_j) \to (-1)^m`` (bounded, does not vanish),
- ``\operatorname{erfcx}(z_j) \to \operatorname{erfcx}(0) = 1`` (bounded),
- ``e^{+i\pi/4}`` is a constant phase,
- the ratio ``\sqrt{\pi X_j}/\sin(\psi_j)`` is the critical factor.

### The ratio ``\sqrt{\pi X_j}/\sin(\psi_j)``

Using the linear approximations:

```math
\frac{\sqrt{\pi X_j}}{\sin(\psi_j)} \approx \frac{\sqrt{\pi \cdot kL \cdot \delta^2/2}}{(-1)^m \cdot \delta/(2n)} = (-1)^m \cdot \frac{2n\sqrt{\pi kL/2}}{1} = (-1)^m \cdot n\sqrt{2\pi kL}.
```

This is finite and nonzero. The implementation evaluates this ratio directly when ``|\sin(\psi_j)| < \sqrt{\varepsilon_{\text{mach}}} \approx 1.49 \times 10^{-8}``, avoiding the ``\infty \cdot 0`` cancellation.

### Exact boundary handling

At the **exact** boundary (``\delta = 0`` in floating point), both ``\sin(\psi_j)`` and ``a_j`` are exactly zero. The one-sided limits are

```math
\lim_{\delta \to 0^\pm} \cot(\psi_j)\,F(X_j) = \pm\cos(m\pi) \cdot n\sqrt{2\pi kL}\;e^{+i\pi/4},
```

which are finite but have opposite signs from the two sides.
The implementation uses two explicit policies:

1. **Finite ``L``:** evaluate a one-sided surrogate at angular offset ``\delta\psi = \varepsilon_{\text{tol}}``, computing ``\cot(\psi + \delta\psi)\,F(kL \cdot 2n^2\delta\psi^2)`` with matched ``a = 2n^2\delta\psi^2``. The offset sign is chosen from the angular detuning (lit-side default if detuning is numerically zero).
2. **``L = \infty``:** use the exact far-field ``F \to 1`` branch; at exact transition samples, return the symmetric midpoint value (zero for the singular term) to keep coefficients finite and consistent with the far-field limit path.

### Implementation thresholds

The regularisation logic uses the machine-precision-derived threshold ``\varepsilon_{\text{tol}} = \sqrt{\varepsilon_{\text{mach}}} \approx 1.49 \times 10^{-8}`` (for Float64):

1. ``|\sin(\psi_j)| > \varepsilon_{\text{tol}}``: use the regularised ratio form ``\cos(\psi) \cdot [\sqrt{\pi X}/\sin(\psi)] \cdot e^{+i\pi/4} \cdot \operatorname{erfcx}(z)``.
2. ``|\sin(\psi_j)| \le \varepsilon_{\text{tol}}`` **and** ``|a_j| \le \varepsilon_{\text{tol}}`` with finite ``L``: exact boundary, evaluate one-sided surrogate at offset ``\delta\psi = \varepsilon_{\text{tol}}``.
3. Exact-boundary samples with ``L = \infty``: return midpoint value for the singular term (zero), otherwise use ``\cot(\psi_j)``.
4. The same ``\varepsilon_{\text{tol}}`` threshold is used for both ``\sin(\psi_j)`` and ``a_j``, derived from ``\sqrt{\texttt{eps(Float64)}}``.

Note: all paths use the regularised ratio form internally; the threshold only selects whether surrogate angles are needed.

```@example numerical
# Demonstrate: regularised implementation gives finite values at the ISB
w = Wedge(2π)
phip = π/4
isb = π + phip  # 225° = incident shadow boundary

for offset in [0.1, 1e-3, 1e-6, 1e-10, 0.0, -1e-10, -1e-6, -1e-3, -0.1]
    phi = isb + offset
    Ds, _ = pec_wedge_DsDh(w, RayAngles(phi, phip), 10.0, 1.0)
    println("φ = ISB + $(lpad(offset, 8)):  |Ds| = $(round(abs(Ds), digits=6)),  finite = $(isfinite(abs(Ds)))")
end
```

## Challenge 3: Branch-cut consistency

### The problem

The transition function contains ``\sqrt{x}`` in two places:

1. The prefactor ``\sqrt{\pi x}``,
2. The erfcx argument ``z = e^{+i\pi/4}\sqrt{x}``.

If these two square roots are evaluated on different branches, the result
acquires spurious sign flips. Material propagation requires an additional
choice: under the package's ``\exp(+i\omega t)`` convention, a passive lossy
medium has ``\operatorname{Im}\varepsilon_r<0`` and a decaying factor
``\exp(-ik_z z)`` requires ``\operatorname{Im}k_z\leq0``.

### The solution: separate mathematical and material roots

Mathematical kernels use `safe_sqrt`, which applies the principal branch:

```julia
safe_sqrt(x::Real) = sqrt(complex(x))
safe_sqrt(x::Complex) = sqrt(x)
```

```math
\sqrt{z} : \quad \arg(z) \in (-\pi, \pi], \quad \operatorname{Re}(\sqrt{z}) \ge 0.
```

The explicit conversion `Complex(x)` ensures that real negative inputs return a
complex value instead of raising `DomainError`. Transition-function roots use
this policy consistently.

Fresnel longitudinal roots and Maliuzhinets impedance roots instead use
`radiation_sqrt`. It equals the principal root away from the negative-real cut.
At ``z=-a`` with ``a>0``, it selects the passive limiting-absorption value

```math
\sqrt{-a-i0}=-i\sqrt{a}.
```

This makes an exactly lossless negative-real material continuous with the
package's passive lossy convention. It does not change positive-real materials,
ordinary passive lossy materials, or mathematical transition-function roots.

### Why this matters for AD

Forward-mode AD (ForwardDiff.jl) propagates dual numbers through the selected
sheet. Derivatives are meaningful within a smooth sheet, but not across its
branch point or a discrete sheet change. The implementation therefore uses
primal values only for the sheet decision while retaining dual arithmetic in
the selected root.

For typical PEC UTD inputs, transition arguments are positive real and stay
away from the principal-branch cut. Material derivatives are valid away from
the branch point and agree with finite differences along the selected sheet.

## Challenge 4: Grazing-incidence seam aliasing

### The problem

The KP terms are periodic in the wedge angle interval, so `phi` and `phi + m*alpha` are physically equivalent after wrapping. However, at grazing incidence (``\phi' \approx 0`` or ``\phi' \approx \alpha``), the angular difference ``\beta^- = \phi - \phi'`` can map two equivalent directions to opposite sides of the interval seam (``0 \leftrightarrow \alpha``), causing artificial jumps in intermediate diffraction terms.

### The solution: use a differentiable face-local angle at grazing

The kernel computes effective angles with a grazing-specific rule:

1. Wrap `phi` and `phip` into `[0,\alpha)` using `wrap_angle`.
2. If `phip` is within `DEFAULT_TRANSITION_TOL` of a face, express it as a
   signed local offset from that face. At the o-face this is `phip`; at the
   n-face the mirrored coordinates are `alpha - phi` and `alpha - phip`.
   The source-angle value is zero at exact grazing, while its ForwardDiff
   partial is retained with the correct local sign.

This removes the value ambiguity without replacing the physical one-sided
source-angle sensitivity by zero. Standard ``[0, \alpha)`` wrapping is
essential: the sign of ``\sin(\psi_2)`` must flip as ``\phi`` crosses ``\pi``
to produce the compensating discontinuity in the diffraction coefficient. A
centered wrap ``(-\alpha/2, \alpha/2]`` would place its branch cut at the ISB
for the half-plane (``\alpha = 2\pi``), destroying this compensation and
breaking total-field continuity.

For ordinary real values, `wrap_angle` delegates the primal remainder to
Julia's `mod`. This remains finite and accurate even when `abs(phi/alpha)` is
too large for the algebraic expression `phi - fld(phi, alpha)*alpha`. Its AD
path combines the same primal remainder with the local periodic-translation
derivative, including at an exact seam.

## Challenge 5: Soft-coefficient cancellation at face grazing

### The problem

For a PEC wedge with a source offset ``h`` from a face, the soft four-term
pairing can be written as

```math
D_s = C\,[G(\phi-h)-G(\phi+h)].
```

The physical coefficient is ``O(h)``. When ``h`` approaches machine precision,
the two ``O(1)`` kernel values coalesce and their subtraction can lose every
significant digit even though each term is evaluated accurately.

### Domain-certified, quadrature-checked continuation

On an interval that does not cross a KP integer branch, cotangent pole, or
vanishing transition argument, the package uses the identity

```math
D_s = -C h\int_{-1}^{1}G'(\phi+h\xi)\,d\xi.
```

`grazing_interval_report` checks those conditions analytically. Its `valid`
field is a domain certificate; it does not claim that a fixed quadrature order
is accurate. For finite `L`, the production evaluator refines the
Gauss–Legendre order and returns only after the highest-order estimate agrees
with two lower-order checks at the requested tolerance. The recommended
`wedge_DsDh` entry point attempts the continuation below its default
`grazing_switch=1e-2` rad and falls back to the four-term expression when the
certificate is not valid or the quadrature does not converge. For `L=Inf`, it
uses an exact cancellation-free closed form instead of quadrature.

```@example numerical
w = Wedge(1.5π)
ang = RayAngles(1.7, 1e-16)
k = 100.0
L = 1.0

report = grazing_interval_report(w, ang, k, L)
println("certified = $(report.valid), face = $(report.face)")

Ds_auto, Dh_auto = wedge_DsDh(w, ang, k, L)
Ds_direct, Dh_direct = pec_wedge_DsDh_grazing(w, ang, k, L; order=8)
Ds_linear = pec_wedge_Ds_linear(w, ang, k, L)
Ds_four, _ = pec_wedge_DsDh(w, ang, k, L)

println("router agrees with direct continuation: $(Ds_auto == Ds_direct && Dh_auto == Dh_direct)")
println("relative error vs leading term (router): $(abs(Ds_auto-Ds_linear)/abs(Ds_linear))")
println("relative error vs leading term (four-term): $(abs(Ds_four-Ds_linear)/abs(Ds_linear))")
```

The starting Gauss–Legendre order defaults to 8. Refinement uses
`quadrature_rtol=1e-11`, `quadrature_atol=0`, and `max_order=256`; the starting
order must be in `1:256` and cannot exceed `max_order`. Use
`pec_wedge_DsDh_grazing(...; on_fail=:four_term)` for an explicit fallback, or
inspect the report's margin and `reason` fields before requesting the direct
continuation.
