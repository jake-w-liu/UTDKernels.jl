# [Numerical Methods](@id numerical)

This chapter describes the three numerical challenges that arise when evaluating UTD diffraction coefficients in IEEE 754 floating-point arithmetic, and the solutions implemented in UTDKernels.jl.

## Challenge 1: Overflow in the transition function

### The problem

The erfc representation of the transition function is

```math
F(x) = \sqrt{\pi x}\;e^{-i(\pi/4 + x)}\;\operatorname{erfc}\!\bigl(e^{-i\pi/4}\sqrt{x}\bigr).
```

For large real ``x``, let ``z = e^{-i\pi/4}\sqrt{x}``. Then ``|z| = \sqrt{x}`` grows without bound, and ``\operatorname{erfc}(z)`` decays super-exponentially:

```math
|\operatorname{erfc}(z)| \sim \frac{e^{-|z|^2}}{\sqrt{\pi}\,|z|} = \frac{e^{-x}}{\sqrt{\pi x}} \qquad (x \to +\infty).
```

Meanwhile, ``|e^{-i(\pi/4+x)}| = 1`` for real ``x`` (pure phase), but the product ``e^{-ix} \cdot \operatorname{erfc}(z)`` encounters the problem that ``\operatorname{erfc}(z)`` underflows to zero in Float64 for ``x \gtrsim 30`` while the overall result ``F(x)`` should be approximately 1.

In IEEE 754 double precision, this manifests as:

```math
\underbrace{e^{-ix}}_{\text{magnitude 1}} \times \underbrace{\operatorname{erfc}(z)}_{\text{underflows to 0}} = \texttt{0.0} \neq F(x) \approx 1.
```

### The solution: erfcx

The **scaled complementary error function**

```math
\operatorname{erfcx}(z) = e^{z^2}\operatorname{erfc}(z)
```

absorbs the exponential decay into the scaling factor. As shown in [The UTD Transition Function](@ref transition), the erfc and erfcx forms are related by the identity ``z^2 = -ix``, which leads to the cancellation

```math
e^{-i(\pi/4+x)} \cdot \operatorname{erfc}(z) = e^{-i\pi/4} \cdot \operatorname{erfcx}(z),
```

giving the numerically stable form:

```math
F(x) = \sqrt{\pi x}\;e^{-i\pi/4}\;\operatorname{erfcx}\!\bigl(e^{-i\pi/4}\sqrt{x}\bigr).
```

The function ``\operatorname{erfcx}(z)`` is bounded for ``\operatorname{Re}(z) \ge 0``:

```math
|\operatorname{erfcx}(z)| \le \max\!\left(1,\;\frac{1}{\sqrt{\pi}\,\operatorname{Re}(z)}\right).
```

No exponentially large or small intermediate values appear, and the result is accurate to machine precision for all ``x``.

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
\cot(\psi_j)\,F(X_j) = \frac{\cos(\psi_j)}{\sin(\psi_j)} \cdot \sqrt{\pi X_j}\;e^{-i\pi/4}\;\operatorname{erfcx}(z_j),
```

where ``z_j = e^{-i\pi/4}\sqrt{X_j}``. Rearranging:

```math
\boxed{\cot(\psi_j)\,F(X_j) = \cos(\psi_j) \cdot \frac{\sqrt{\pi X_j}}{\sin(\psi_j)} \cdot e^{-i\pi/4} \cdot \operatorname{erfcx}(z_j).}
```

In this form:
- ``\cos(\psi_j) \to (-1)^m`` (bounded, does not vanish),
- ``\operatorname{erfcx}(z_j) \to \operatorname{erfcx}(0) = 1`` (bounded),
- ``e^{-i\pi/4}`` is a constant phase,
- the ratio ``\sqrt{\pi X_j}/\sin(\psi_j)`` is the critical factor.

### The ratio ``\sqrt{\pi X_j}/\sin(\psi_j)``

Using the linear approximations:

```math
\frac{\sqrt{\pi X_j}}{\sin(\psi_j)} \approx \frac{\sqrt{\pi \cdot kL \cdot \delta^2/2}}{(-1)^m \cdot \delta/(2n)} = (-1)^m \cdot \frac{2n\sqrt{\pi kL/2}}{1} = (-1)^m \cdot n\sqrt{2\pi kL}.
```

This is finite and nonzero. The implementation evaluates this ratio directly when ``|\sin(\psi_j)| < 10^{-10}``, avoiding the ``\infty \cdot 0`` cancellation.

### Exact boundary handling

At the **exact** boundary (``\delta = 0`` in floating point), both ``\sin(\psi_j)`` and ``a_j`` are exactly zero. The one-sided limits are

```math
\lim_{\delta \to 0^\pm} \cot(\psi_j)\,F(X_j) = \pm\cos(m\pi) \cdot n\sqrt{2\pi kL}\;e^{-i\pi/4},
```

which are finite but have opposite signs from the two sides. The implementation returns **zero** at this measure-zero set of angles, which is the symmetric midpoint of the one-sided limits. This convention does not affect the physical total field (GO + diffracted), which is continuous.

### Implementation thresholds

The regularisation logic uses two thresholds:

1. ``|\sin(\psi_j)| > \epsilon_{\text{tol}} = 10^{-10}``: use the direct formula ``\cot(\psi) \cdot F(X)``.
2. ``|\sin(\psi_j)| \le 10^{-10}`` **and** ``|a_j| < 10^{-28}``: exact boundary, return zero.
3. ``|\sin(\psi_j)| \le 10^{-10}`` **and** ``|a_j| \ge 10^{-28}``: use the regularised ratio form.

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
2. The erfcx argument ``z = e^{-i\pi/4}\sqrt{x}``.

If these two square roots are evaluated on **different branches**, the result acquires spurious sign flips. Worse, automatic differentiation (AD) breaks down at branch cuts because the derivative is undefined there.

### The solution: `safe_sqrt`

Every square root in UTDKernels.jl is evaluated via a single wrapper function:

```julia
safe_sqrt(x::Number) = sqrt(Complex(x))
safe_sqrt(x::Complex) = sqrt(x)
```

This enforces the **principal branch**:

```math
\sqrt{z} : \quad \arg(z) \in (-\pi, \pi], \quad \operatorname{Re}(\sqrt{z}) \ge 0.
```

The explicit conversion `Complex(x)` ensures that even real negative inputs are handled correctly: ``\sqrt{-1} = i`` rather than throwing a `DomainError`.

### Why this matters for AD

Forward-mode AD (ForwardDiff.jl) computes derivatives by propagating dual numbers through the computation graph. If the computation crosses a branch cut, the derivative is discontinuous and the AD result is meaningless. By enforcing a single, documented branch for all square roots, we ensure that:

1. The function is smooth everywhere except on the branch cut (the negative real axis for ``x``).
2. For the typical UTD use case (real positive ``k``, ``L``, and real angles), the arguments to ``\sqrt{\cdot}`` are either positive real or have positive real part, staying safely away from the branch cut.
3. Gradients computed by AD are well-defined and agree with finite differences.
