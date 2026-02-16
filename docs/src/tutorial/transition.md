# [The UTD Transition Function](@id transition)

The transition function ``F(x)`` is the mathematical heart of the uniform theory of diffraction. It interpolates smoothly between the shadow region (``x \ll 1``, where diffraction dominates) and the lit region (``x \gg 1``, where geometrical optics dominates). This chapter derives ``F(x)`` in full detail, starting from the Fresnel integral and proceeding through three equivalent representations.

## Physical motivation

In the geometrical theory of diffraction (GTD), the diffracted field is proportional to ``\cot(\psi_j)`` --- a function that diverges at shadow and reflection boundaries. GTD therefore fails at precisely the locations where the diffracted field is most important.

UTD fixes this by multiplying each cotangent term by a transition function ``F(X_j)``, where ``X_j = k L a_j`` measures the **optical path difference** between the direct and diffracted rays in units of the wavelength. When the observer is:

- **Far from a boundary** (``X_j \gg 1``): ``F(X_j) \to 1``, recovering the GTD result.
- **Near a boundary** (``X_j \ll 1``): ``F(X_j) \to 0``, taming the cotangent singularity.

The smooth transition between these limits is what makes the total field (GO + diffracted) continuous.

## Definition: the Fresnel integral form

For the ``\exp(+i\omega t)`` convention, the UTD transition function is defined by

```math
F(x) \equiv 2i\sqrt{x}\,e^{+ix} \int_{\sqrt{x}}^{\infty} e^{-it^2}\,dt,
```

where ``\sqrt{\cdot}`` is evaluated on the **principal branch** (branch cut along the negative real axis, ``\arg(x) \in (-\pi, \pi]``, ``\operatorname{Re}(\sqrt{x}) \ge 0``), and the integral is defined by analytic continuation for complex ``x``.

The integral ``\int_{\sqrt{x}}^{\infty} e^{-it^2}\,dt`` is an **upper incomplete Fresnel integral**. For real positive ``x``, this integral is well-defined and the integrand oscillates with decreasing amplitude.

## Step-by-step derivation: Fresnel integral to erfc

We now transform the Fresnel integral into the complementary error function. This requires a careful change of variables.

### Step 1: Substitution ``t = e^{-i\pi/4}\,\tau``

Define the substitution

```math
t = e^{-i\pi/4}\,\tau, \qquad dt = e^{-i\pi/4}\,d\tau.
```

Compute ``-it^2``:

```math
-it^2 = -i \cdot \bigl(e^{-i\pi/4}\bigr)^2 \cdot \tau^2 = -i \cdot e^{-i\pi/2} \cdot \tau^2 = -i \cdot (-i) \cdot \tau^2 = -\tau^2.
```

So ``e^{-it^2} = e^{-\tau^2}``, converting the oscillatory Fresnel integrand into a decaying Gaussian.

### Step 2: Transform the integration limits

When ``t = \sqrt{x}``:

```math
\tau = e^{+i\pi/4}\,\sqrt{x}.
```

When ``t \to \infty``: ``\tau \to \infty`` along a ray at angle ``+\pi/4`` in the complex plane.

### Step 3: Apply the substitution

```math
\int_{\sqrt{x}}^{\infty} e^{-it^2}\,dt
= e^{-i\pi/4} \int_{e^{+i\pi/4}\sqrt{x}}^{\infty} e^{-\tau^2}\,d\tau.
```

### Step 4: Relate to erfc

The complementary error function is defined as

```math
\operatorname{erfc}(z) = \frac{2}{\sqrt{\pi}} \int_{z}^{\infty} e^{-\tau^2}\,d\tau,
```

so

```math
\int_{z}^{\infty} e^{-\tau^2}\,d\tau = \frac{\sqrt{\pi}}{2}\,\operatorname{erfc}(z).
```

With ``z = e^{+i\pi/4}\sqrt{x}``:

```math
\int_{\sqrt{x}}^{\infty} e^{-it^2}\,dt = e^{-i\pi/4} \cdot \frac{\sqrt{\pi}}{2}\,\operatorname{erfc}\!\bigl(e^{+i\pi/4}\sqrt{x}\bigr).
```

### Step 5: Substitute back into ``F(x)``

```math
F(x) = 2i\sqrt{x}\,e^{+ix} \cdot e^{-i\pi/4} \cdot \frac{\sqrt{\pi}}{2}\,\operatorname{erfc}\!\bigl(e^{+i\pi/4}\sqrt{x}\bigr).
```

Simplify the prefactor:

```math
2i\sqrt{x} \cdot \frac{\sqrt{\pi}}{2} = i\sqrt{\pi x}.
```

So:

```math
F(x) = i\sqrt{\pi x}\,e^{+ix}\,e^{-i\pi/4}\,\operatorname{erfc}\!\bigl(e^{+i\pi/4}\sqrt{x}\bigr).
```

### Step 6: Simplify ``i \cdot e^{-i\pi/4}``

Express ``i`` in exponential form:

```math
i = e^{+i\pi/2}.
```

Therefore:

```math
i \cdot e^{-i\pi/4} = e^{+i\pi/2} \cdot e^{-i\pi/4} = e^{+i\pi/2 - i\pi/4} = e^{+i\pi/4}.
```

### Step 7: The erfc form

Combining Steps 5 and 6:

```math
\boxed{F(x) = \sqrt{\pi x}\;e^{+i(\pi/4 + x)}\;\operatorname{erfc}\!\bigl(e^{+i\pi/4}\sqrt{x}\bigr).}
```

This is exact and valid for all complex ``x`` on the principal branch.

## Step-by-step derivation: erfc to erfcx

The erfc form is mathematically clean but numerically dangerous: for large real ``x``, ``\operatorname{erfc}(z)`` underflows to zero in Float64 while the prefactor ``\sqrt{\pi x}`` remains finite, so the product collapses to zero instead of the correct value ``F(x) \approx 1``. The **scaled complementary error function** ``\operatorname{erfcx}`` absorbs this cancellation.

### Step 1: Definition of erfcx

```math
\operatorname{erfcx}(z) \equiv e^{z^2}\,\operatorname{erfc}(z).
```

Equivalently:

```math
\operatorname{erfc}(z) = \operatorname{erfcx}(z) \cdot e^{-z^2}.
```

The key property is that ``\operatorname{erfcx}(z)`` is **bounded** for all ``z`` with ``\operatorname{Re}(z) \ge 0``.

### Step 2: Compute ``z^2``

With ``z = e^{+i\pi/4}\sqrt{x}``:

```math
z^2 = \bigl(e^{+i\pi/4}\bigr)^2 \cdot x = e^{+i\pi/2} \cdot x = ix.
```

### Step 3: Compute ``e^{-z^2}``

```math
e^{-z^2} = e^{-(ix)} = e^{-ix}.
```

### Step 4: Substitute into the erfc form

Starting from the erfc representation:

```math
F(x) = \sqrt{\pi x}\;e^{+i(\pi/4 + x)}\;\operatorname{erfc}(z).
```

Replace ``\operatorname{erfc}(z) = \operatorname{erfcx}(z) \cdot e^{-z^2} = \operatorname{erfcx}(z) \cdot e^{-ix}``:

```math
F(x) = \sqrt{\pi x}\;e^{+i(\pi/4 + x)} \cdot \operatorname{erfcx}(z) \cdot e^{-ix}.
```

### Step 5: Cancel the exponentials

```math
e^{+i(\pi/4 + x)} \cdot e^{-ix} = e^{+i\pi/4} \cdot e^{+ix} \cdot e^{-ix} = e^{+i\pi/4}.
```

### Step 6: The erfcx form

```math
\boxed{F(x) = \sqrt{\pi x}\;e^{+i\pi/4}\;\operatorname{erfcx}\!\bigl(e^{+i\pi/4}\sqrt{x}\bigr).}
```

This is the form used in the implementation. No exponentially growing or decaying factors remain --- the entire function is expressed in terms of bounded quantities.

## Limiting behaviour

### ``F(x) \to 0`` as ``x \to 0``

From the erfcx form, using ``\operatorname{erfcx}(0) = \operatorname{erfc}(0) = 1``:

```math
\lim_{x \to 0} F(x) = \sqrt{\pi \cdot 0}\;e^{+i\pi/4} \cdot 1 = 0.
```

More precisely, for small ``x``:

```math
F(x) \approx \sqrt{\pi x}\;e^{+i\pi/4} \qquad (x \to 0),
```

showing that ``F`` vanishes as ``\sqrt{x}`` near the origin.

### ``F(x) \to 1`` as ``x \to +\infty`` (real ``x``)

For large ``|z|`` with ``\operatorname{Re}(z) > 0``, the asymptotic expansion of erfcx is

```math
\operatorname{erfcx}(z) \sim \frac{1}{\sqrt{\pi}\,z} \qquad (|z| \to \infty).
```

With ``z = e^{+i\pi/4}\sqrt{x}``:

```math
\operatorname{erfcx}(z) \sim \frac{1}{\sqrt{\pi}\,e^{+i\pi/4}\,\sqrt{x}}.
```

Substituting:

```math
F(x) \sim \sqrt{\pi x}\;e^{+i\pi/4} \cdot \frac{1}{\sqrt{\pi}\,e^{+i\pi/4}\,\sqrt{x}} = \frac{\sqrt{\pi x}}{\sqrt{\pi}\,\sqrt{x}} = 1.
```

This is the **GTD recovery**: far from any shadow boundary, the transition function equals unity and UTD reduces to GTD.

### Phase behaviour

For small real ``x``, ``\arg F(x) \approx +\pi/4`` (from the ``e^{+i\pi/4}`` prefactor). As ``x \to +\infty``, ``\arg F(x) \to 0`` (since ``F \to 1``).

The following figure summarises both the magnitude and phase behaviour of ``F(x)`` across the full range from the transition region to the GTD limit.

```@example transition
using UTDKernels, CairoMakie

xs = 10 .^ range(-3, 4, length = 500)
Fvals = F_utd.(xs)

fig = Figure(size = (750, 320))

# Left panel: |F(x)|
ax1 = Axis(fig[1, 1], xlabel = "x", ylabel = "|F(x)|",
    title = "Magnitude", xscale = log10,
    xticks = ([1e-3, 1e-1, 10, 1e3], ["10⁻³", "10⁻¹", "10", "10³"]))
lines!(ax1, xs, abs.(Fvals), color = :royalblue, linewidth = 2)
hlines!(ax1, [1.0], color = :gray60, linestyle = :dash, linewidth = 1)
text!(ax1, 1e3, 0.92, text = "GTD limit |F| = 1", fontsize = 11, color = :gray50)
# √(πx) envelope
lines!(ax1, xs[1:200], sqrt.(π .* xs[1:200]), color = :red3, linewidth = 1, linestyle = :dot)
text!(ax1, 0.03, 0.22, text = "√(πx)", fontsize = 11, color = :red3)
ylims!(ax1, 0, 1.15)

# Right panel: arg(F(x))
ax2 = Axis(fig[1, 2], xlabel = "x", ylabel = "arg F(x)  [rad]",
    title = "Phase", xscale = log10,
    xticks = ([1e-3, 1e-1, 10, 1e3], ["10⁻³", "10⁻¹", "10", "10³"]),
    yticks = ([0, π/8, π/4], ["0", "+π/8", "+π/4"]))
lines!(ax2, xs, angle.(Fvals), color = :royalblue, linewidth = 2)
hlines!(ax2, [π/4], color = :gray60, linestyle = :dash, linewidth = 1)
hlines!(ax2, [0.0], color = :gray60, linestyle = :dash, linewidth = 1)
text!(ax2, 1e-2, 0.68, text = "+π/4", fontsize = 11, color = :gray50)
ylims!(ax2, -0.15, 1.0)

fig # hide
```

## Numerical demonstration

```@example transition
using UTDKernels

# F(x) at representative values
for x in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1e4]
    F = F_utd(x)
    println("F($x) = $(round(real(F), sigdigits=5)) + $(round(imag(F), sigdigits=5))i,  " *
            "|F| = $(round(abs(F), sigdigits=5))")
end
```

```@example transition
# Verify the limiting behaviours
println("F(0):    |F| = $(abs(F_utd(0.0)))")
println("F(1e8):  |F| = $(abs(F_utd(1e8)))")
println("F(1e12): |F| = $(abs(F_utd(1e12)))")
```

```@example transition
# Verify the erfc identity (both representations agree to machine precision)
using SpecialFunctions: erfc, erfcx
x = 5.0
sqx = sqrt(Complex(x))
z = exp(+im*π/4) * sqx
F_erfc  = sqrt(π * Complex(x)) * exp(+im*(π/4 + x)) * erfc(z)
F_erfcx = F_utd(x)
println("erfc form:  $F_erfc")
println("erfcx form: $F_erfcx")
println("difference: $(abs(F_erfc - F_erfcx))")
```

## Connection to other representations

Different authors define the UTD transition function with different sign conventions. The two forms arise from two equally valid orientations of the steepest descent contour through the saddle point, which produce Fresnel integrands ``e^{+i\tau^2}`` and ``e^{-i\tau^2}`` respectively.

The representation used in **this package** is the standard Kouyoumjian--Pathak (KP) form with ``\operatorname{Im}(F(x)) > 0`` for real positive ``x``:

```math
F(x) = \sqrt{\pi x}\;e^{+i\pi/4}\;\operatorname{erfcx}\!\bigl(e^{+i\pi/4}\sqrt{x}\bigr).
```

The complex conjugate form ``F^*(x)`` appears in some references, with ``\operatorname{Im}(F^*) < 0``:

```math
F^*(x) = \sqrt{\pi x}\;e^{-i\pi/4}\;\operatorname{erfcx}\!\bigl(e^{-i\pi/4}\sqrt{x}\bigr).
```

Both forms share identical magnitude ``|F| = |F^*|`` and the same limiting behaviour (``F \to 1`` as ``x \to +\infty``). They produce the **same total physical field** ``u^{\text{GO}} + u^d`` because the conjugation is consistently absorbed throughout the diffraction coefficient prefactor. See [KP Coefficients](@ref kp) for the full derivation and convention comparison.
