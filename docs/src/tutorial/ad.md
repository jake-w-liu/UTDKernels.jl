# [Automatic Differentiation](@id ad)

This chapter derives the custom derivative rule for ``\operatorname{erfcx}`` that enables forward-mode automatic differentiation (AD) through the UTD diffraction-coefficient pipeline at smooth operating points, explains the complex chain rule used by the ForwardDiff extension, and demonstrates gradient computation.

## Why AD for UTD?

Many modern applications of UTD require **gradients** of diffracted fields with respect to physical parameters:

- **Antenna placement optimisation**: gradient of received power with respect to antenna position.
- **Inverse scattering**: fitting diffraction models to measured data via gradient descent.
- **Sensitivity analysis**: computing ``\partial D_{s/h}/\partial\phi``, ``\partial D_{s/h}/\partial k``, or ``\partial D_{s/h}/\partial L``.

Without AD, practitioners must either:
1. **Hand-derive adjoint equations** --- error-prone, maintenance-intensive, and specific to each problem.
2. **Use finite differences** --- slow (requires ``2N`` function evaluations for ``N`` parameters) and inaccurate (truncation vs. round-off trade-off).

Forward-mode AD computes machine-precision derivatives at smooth points, at a cost of roughly ``3\text{--}4\times`` a single function evaluation.

## The missing piece: erfcx for complex Dual numbers

The Julia package ForwardDiff.jl implements forward-mode AD using **dual numbers**: ``x = a + b\varepsilon`` where ``\varepsilon^2 = 0``. For a function ``f``, evaluating ``f(a + b\varepsilon)`` yields ``f(a) + f'(a)\,b\varepsilon``, giving the derivative ``f'(a)`` automatically.

However, ForwardDiff does not natively support ``\operatorname{erfcx}`` for `Complex{Dual}` arguments. Since the entire UTD pipeline flows through ``F_{\text{utd}}(x) \to \operatorname{erfcx}(z)``, a custom derivative rule is needed.

## Derivation of the erfcx derivative

We derive ``\frac{d}{dz}\operatorname{erfcx}(z)`` from first principles.

### Step 1: Start from the definition

```math
\operatorname{erfcx}(z) = e^{z^2}\,\operatorname{erfc}(z).
```

### Step 2: Apply the product rule

```math
\frac{d}{dz}\operatorname{erfcx}(z) = \frac{d}{dz}\!\left[e^{z^2}\right] \cdot \operatorname{erfc}(z) + e^{z^2} \cdot \frac{d}{dz}\!\left[\operatorname{erfc}(z)\right].
```

Compute each factor:

```math
\frac{d}{dz}\!\left[e^{z^2}\right] = 2z\,e^{z^2}.
```

### Step 3: The erfc derivative

The complementary error function is defined as

```math
\operatorname{erfc}(z) = \frac{2}{\sqrt{\pi}} \int_z^{\infty} e^{-t^2}\,dt.
```

Differentiating with respect to the lower limit:

```math
\frac{d}{dz}\operatorname{erfc}(z) = -\frac{2}{\sqrt{\pi}}\,e^{-z^2}.
```

### Step 4: Substitute and simplify

```math
\frac{d}{dz}\operatorname{erfcx}(z) = 2z\,e^{z^2}\,\operatorname{erfc}(z) + e^{z^2}\!\left(-\frac{2}{\sqrt{\pi}}\,e^{-z^2}\right).
```

The first term is ``2z\,\operatorname{erfcx}(z)``. The second term simplifies:

```math
e^{z^2} \cdot e^{-z^2} = 1.
```

### Step 5: Final result

```math
\boxed{\frac{d}{dz}\operatorname{erfcx}(z) = 2z\,\operatorname{erfcx}(z) - \frac{2}{\sqrt{\pi}}.}
```

This is a clean, closed-form expression that involves only ``z`` and ``\operatorname{erfcx}(z)`` itself --- no additional special functions are needed.

## Complex chain rule for ForwardDiff

ForwardDiff represents a complex dual number as `Complex{Dual{T,V,N}}`, where the real and imaginary parts are each dual numbers carrying their own partials.

For a holomorphic function ``f(z)`` with ``z = x + iy`` and ``f = u + iv``, the complex derivative ``df/dz`` relates to the real partial derivatives via the **Cauchy--Riemann equations**. The chain rule for propagating partials is:

```math
\begin{aligned}
d(\operatorname{Re} f) &= \operatorname{Re}\!\left(\frac{df}{dz}\right) dx - \operatorname{Im}\!\left(\frac{df}{dz}\right) dy, \\[4pt]
d(\operatorname{Im} f) &= \operatorname{Im}\!\left(\frac{df}{dz}\right) dx + \operatorname{Re}\!\left(\frac{df}{dz}\right) dy.
\end{aligned}
```

**Derivation**: Let ``df/dz = u_x + iv_x`` where ``u_x = \partial u/\partial x`` and ``v_x = \partial v/\partial x``. By the Cauchy--Riemann equations, ``u_y = -v_x`` and ``v_y = u_x``. Then:

```math
du = u_x\,dx + u_y\,dy = u_x\,dx - v_x\,dy = \operatorname{Re}(df/dz)\,dx - \operatorname{Im}(df/dz)\,dy,
```

```math
dv = v_x\,dx + v_y\,dy = v_x\,dx + u_x\,dy = \operatorname{Im}(df/dz)\,dx + \operatorname{Re}(df/dz)\,dy.
```

In the implementation, for each partial index ``i``:

```julia
dr[i] = real(df_dz) * p_re[i] - imag(df_dz) * p_im[i]
di[i] = imag(df_dz) * p_re[i] + real(df_dz) * p_im[i]
```

where `p_re[i]` and `p_im[i]` are the ``i``-th partials of the real and imaginary parts of ``z``.

## Package extension mechanism

The AD rule is provided as a **Julia 1.9+ package extension** (`UTDKernelsForwardDiffExt`). This means:

- ForwardDiff is a **weak dependency**: it is not loaded unless the user explicitly imports it.
- The base package (UTDKernels) has **zero AD overhead** when ForwardDiff is not used.
- When the user writes `using ForwardDiff`, the extension is automatically activated.

## End-to-end gradient example

```@example ad
using UTDKernels, ForwardDiff

w = Wedge(2π)
k = 10.0; L = 1.0; phip = π/4

# Define a scalar function: |Ds| as a function of observation angle
f(phi) = abs(pec_wedge_DsDh(w, RayAngles(phi, phip), k, L)[1])

# Compute the gradient via AD
phi_test = π/2
grad_ad = ForwardDiff.derivative(f, phi_test)

# Compare with centred finite differences
h = 1e-7
grad_fd = (f(phi_test + h) - f(phi_test - h)) / (2h)

println("AD gradient:  $grad_ad")
println("FD gradient:  $grad_fd")
println("Relative error: $(abs(grad_ad - grad_fd) / abs(grad_ad))")
```

```@example ad
# Gradient with respect to wavenumber k
g(k_val) = abs(pec_wedge_DsDh(w, RayAngles(π/3, phip), k_val, L)[1])

grad_k_ad = ForwardDiff.derivative(g, 10.0)
h_k = 10.0 * 1e-7
grad_k_fd = (g(10.0 + h_k) - g(10.0 - h_k)) / (2h_k)

println("∂|Ds|/∂k (AD): $grad_k_ad")
println("∂|Ds|/∂k (FD): $grad_k_fd")
println("Relative error: $(abs(grad_k_ad - grad_k_fd) / abs(grad_k_ad))")
```

```@example ad
# Gradient with respect to effective distance L
g_L(L_val) = abs(pec_wedge_DsDh(w, RayAngles(π/3, phip), k, L_val)[1])

grad_L_ad = ForwardDiff.derivative(g_L, 1.0)
h_L = 1e-7
grad_L_fd = (g_L(1.0 + h_L) - g_L(1.0 - h_L)) / (2h_L)

println("∂|Ds|/∂L (AD): $grad_L_ad")
println("∂|Ds|/∂L (FD): $grad_L_fd")
println("Relative error: $(abs(grad_L_ad - grad_L_fd) / abs(grad_L_ad))")
```

The AD gradients agree with finite differences to ``O(10^{-8})`` or better on these test points, confirming differentiability of the implemented computational graph in smooth regions. As with any piecewise/asymptotic UTD formulation, exact transition-boundary points and branch-cut seams remain non-smooth sets where derivatives are not uniquely defined.
