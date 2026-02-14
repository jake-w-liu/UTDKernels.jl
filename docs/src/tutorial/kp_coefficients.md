# [Kouyoumjian--Pathak Diffraction Coefficients](@id kp)

This chapter derives the Kouyoumjian--Pathak (KP) diffraction coefficients for PEC wedge diffraction from first principles, following the treatment of Balanis (Ch. 13) and the original KP paper [10]. The derivation proceeds in five stages: exact modal solution → high-frequency asymptotics → contour integral → steepest descent → UTD four-term structure. Every intermediate step is presented, with careful attention to the ``\exp(+i\omega t)`` sign convention used throughout this package.

## The canonical problem

Consider a two-dimensional PEC wedge with exterior angle ``\alpha = n\pi`` (wedge material fills the region ``\phi \in (\alpha, 2\pi)``). A line source of current ``I`` is placed at ``(\rho', \phi')`` inside the free-space region ``0 < \phi < \alpha``. The observation point is at ``(\rho, \phi)``.

The total scalar field ``u(\rho, \phi)`` satisfies the Helmholtz equation ``(\nabla^2 + k^2)u = 0`` with:
- **Dirichlet** (soft) BC: ``u = 0`` on both wedge faces, or
- **Neumann** (hard) BC: ``\partial u/\partial n = 0`` on both wedge faces.

We write the field as ``u = G``, where ``G`` is the Green's function for the wedge.

## Stage 1: Exact modal solution

Using separation of variables in cylindrical coordinates, the Green's function is expanded as an infinite series of Bessel functions and angular eigenfunctions. For ``\rho \le \rho'``, the exact solution is (Balanis Eq. 13-38):

```math
G = \frac{1}{n} \sum_{m=0}^{\infty} \varepsilon_m\, J_{m/n}(k\rho)\, H_{m/n}^{(2)}(k\rho') \left[\cos\frac{m}{n}(\phi - \phi') \pm \cos\frac{m}{n}(\phi + \phi')\right],
```

where:
- ``J_{m/n}`` is the Bessel function of the first kind of order ``m/n``,
- ``H_{m/n}^{(2)}`` is the Hankel function of the second kind (outgoing waves in the ``\exp(+i\omega t)`` convention),
- ``\varepsilon_m`` is the Neumann factor: ``\varepsilon_0 = 1``, ``\varepsilon_m = 2`` for ``m \ge 1``,
- the ``+`` sign is for hard (Neumann) and the ``-`` sign is for soft (Dirichlet) polarisation.

For ``\rho \ge \rho'``, the roles of ``\rho`` and ``\rho'`` are interchanged.

!!! note "Boundary condition verification"
    For **soft** polarisation (``-`` sign), at ``\phi = 0``: ``\cos\frac{m}{n}(0 - \phi') - \cos\frac{m}{n}(0 + \phi') = 0``. At ``\phi = \alpha = n\pi``: ``\cos(m\pi - \frac{m}{n}\phi') - \cos(m\pi + \frac{m}{n}\phi') = 2\sin(m\pi)\sin(\frac{m}{n}\phi') = 0``. Both faces give ``G = 0``. ✓

This series converges well for small ``k\rho``, but converges extremely slowly for large ``k\rho`` (many terms needed for high-frequency fields), motivating the asymptotic approach.

## Stage 2: High-frequency asymptotic expansion

When the source is far from the edge (``k\rho' \gg 1``), we replace the Hankel function by its large-argument asymptotic form:

```math
H_{m/n}^{(2)}(k\rho') \;\xrightarrow{k\rho' \to \infty}\; \sqrt{\frac{2}{\pi k\rho'}}\; e^{-i[k\rho' - \pi/4 - (m/n)(\pi/2)]}.
```

Substituting into the modal series and pulling out common factors:

```math
G \approx \sqrt{\frac{2}{\pi k\rho'}}\; e^{-i(k\rho' - \pi/4)}\; \mathcal{F}(k\rho),
```

where we define the **normalised total field function**

```math
\mathcal{F}(k\rho) = \frac{1}{n} \sum_{m=0}^{\infty} \varepsilon_m\, J_{m/n}(k\rho)\, e^{i(m/n)(\pi/2)} \left[\cos\frac{m}{n}(\phi - \phi') \pm \cos\frac{m}{n}(\phi + \phi')\right].
```

The function ``\mathcal{F}(k\rho)`` contains the full angular dependence and is the object we will evaluate asymptotically.

## Stage 3: Contour integral representation

To apply asymptotic methods, we must convert the slowly converging series into a contour integral. This is accomplished by:

1. **Expressing the cosine terms in complex exponential form:**

```math
\cos\!\left(\frac{m}{n}\,\xi^{\mp}\right) = \frac{1}{2}\left[e^{i(m/n)\xi^{\mp}} + e^{-i(m/n)\xi^{\mp}}\right],
```

where ``\xi^- = \phi - \phi'`` and ``\xi^+ = \phi + \phi'``.

2. **Replacing the Bessel functions by their contour integral representation** (Sommerfeld integral):

```math
J_{m/n}(k\rho) = \frac{1}{2\pi} \int_C e^{ik\rho\cos z + i(m/n)(z - \pi/2)}\,dz,
```

where ``C`` is the Sommerfeld contour in the complex ``z``-plane.

3. **Interchanging the order of summation and integration** and evaluating the geometric series over ``m``, which produces cotangent functions via the identity

```math
\frac{1}{1 - e^{-i(\xi + z)/n}} = \frac{1}{2} - \frac{i}{2}\cot\!\left(\frac{\xi + z}{2n}\right).
```

The result is a pair of contour integrals:

```math
\mathcal{F}(k\rho) = \frac{1}{4\pi in} \int_{C'-C} \cot\!\left(\frac{\phi - \phi' + z}{2n}\right) e^{ik\rho\cos z}\,dz \;\pm\; \frac{1}{4\pi in} \int_{C'-C} \cot\!\left(\frac{\phi + \phi' + z}{2n}\right) e^{ik\rho\cos z}\,dz.
```

This is the key result: the infinite Bessel series has been transformed into two contour integrals of the form ``\int H(z)\,e^{ik\rho\,h(z)}\,dz``, where ``h(z) = \cos z``. These can now be evaluated by the method of steepest descent for large ``k\rho``.

## Stage 4: Steepest descent decomposition

### Saddle points and steepest descent paths

The phase function is ``h(z) = \cos z``, with saddle points at ``z_s = \pm\pi`` (where ``h'(z) = -\sin z = 0``). The steepest descent paths (SDPs) through these saddle points are curves along which ``\operatorname{Re}[ik\rho\cos z]`` decreases most rapidly from its saddle-point value, while the phase ``\operatorname{Im}[ik\rho\cos z]`` remains constant (constant-phase contours satisfying ``\cos x \cosh y = \text{const}``).


### Contour deformation

The key idea is to deform the original Sommerfeld contours ``C`` and ``C'`` onto the steepest descent paths through the saddle points at ``z = \pm\pi``. The closed contour ``C_T = C' + \text{SDP}_{+\pi} - C + \text{SDP}_{-\pi}`` is shown in the figure below.

```@example kp
using CairoMakie

fig = Figure(size = (700, 450))
ax = Axis(fig[1, 1],
    xlabel = "Re z", ylabel = "Im z",
    title = "Contour deformation in the complex z-plane",
    xticks = ([-3π/2, -π, -π/2, 0, π/2, π, 3π/2],
              ["-3π/2", "-π", "-π/2", "0", "π/2", "π", "3π/2"]),
    yticks = ([-2, -1, 0, 1, 2], ["-2", "-1", "0", "1", "2"]))

# --- Shaded convergent (permitted) regions ---
# |e^{ikρ cos z}| = e^{kρ sin(x)sinh(y)}, converges where sin(x)sinh(y) < 0
# Upper half (sinh > 0): sin(x) < 0 → x ∈ (-π,0) ∪ (π,2π)
# Lower half (sinh < 0): sin(x) > 0 → x ∈ (-2π,-π) ∪ (0,π)
ymax = 2.8; xmin = -5.5; xmax = 5.5
shade = (:green, 0.08)
for (x1, x2) in [(xmin, -2π), (-π, 0.0), (π, 2π)]
    poly!(ax, Point2f[(x1, 0), (x2, 0), (x2, ymax), (x1, ymax)],
        color = shade, strokewidth = 0)
end
for (x1, x2) in [(-2π, -π), (0.0, π), (2π, xmax)]
    poly!(ax, Point2f[(x1, -ymax), (x2, -ymax), (x2, 0), (x1, 0)],
        color = shade, strokewidth = 0)
end
text!(ax, -π/4, 2.2, text = "convergent", fontsize = 10, color = (:green4, 0.6),
    align = (:center, :center), rotation = π/2)

# --- Steepest descent paths (curved) ---
# cos(x)·cosh(y) = -1 → x = x_s ± arccos(1/cosh(y))
y_sdp = collect(range(0, 2.5, length = 100))
Δx = [acos(1 / cosh(y)) for y in y_sdp]

# SDP through +π: (≈π/2, -∞) → (π,0) → (≈3π/2, +∞)  [upward]
sdp_px = vcat((π .- reverse(Δx)), (π .+ Δx)[2:end])
sdp_py = vcat(.-reverse(y_sdp), y_sdp[2:end])
lines!(ax, sdp_px, sdp_py, color = :green4, linewidth = 2.5, label = "SDP±π")

# SDP through -π: (≈-π/2, +∞) → (-π,0) → (≈-3π/2, -∞)  [downward]
sdp_mx = vcat(reverse(-π .+ Δx), (-π .- Δx)[2:end])
sdp_my = vcat(reverse(y_sdp), .-y_sdp[2:end])
lines!(ax, sdp_mx, sdp_my, color = :green4, linewidth = 2.5)

# Tangent-following arrows along each SDP
for y0 in [1.5, -1.5]
    δ = acos(1 / cosh(abs(y0)))
    s = 1 / cosh(abs(y0))           # |dx/dy| = sech(y)
    len = sqrt(s^2 + 1)
    dx, dy = 0.3s / len, 0.3 / len  # unit tangent scaled to 0.3
    # SDP_{+π}: lower-left → upper-right (arrows point up-right)
    x_p = y0 > 0 ? π + δ : π - δ
    arrows!(ax, [x_p], [y0], [dx], [dy],
        color = :green4, linewidth = 2, arrowsize = 10)
    # SDP_{-π}: upper-right → lower-left (arrows point down-left)
    x_m = y0 > 0 ? -π + δ : -π - δ
    arrows!(ax, [x_m], [y0], [-dx], [-dy],
        color = :green4, linewidth = 2, arrowsize = 10)
end

# --- Sommerfeld contour C (upper half-plane, arms at -π/2 and 3π/2) ---
# C: (-π/2, +∞) ↓ to (-π/2, 0) → right to (3π/2, 0) ↑ to (3π/2, +∞)
lines!(ax, fill(-π/2, 40), collect(range(2.5, 0, length = 40)),
    color = :royalblue, linewidth = 2, label = "C")
lines!(ax, collect(range(-π/2, 3π/2, length = 50)), fill(0.04, 50),
    color = :royalblue, linewidth = 2)
lines!(ax, fill(3π/2, 40), collect(range(0, 2.5, length = 40)),
    color = :royalblue, linewidth = 2)
arrows!(ax, [-π/2], [1.5], [0.0], [-0.3],
    color = :royalblue, linewidth = 2, arrowsize = 12)
arrows!(ax, [π/2], [0.04], [0.3], [0.0],
    color = :royalblue, linewidth = 2, arrowsize = 12)
arrows!(ax, [3π/2], [0.3], [0.0], [0.3],
    color = :royalblue, linewidth = 2, arrowsize = 12)

# --- Sommerfeld contour C' (lower half-plane, arms at -3π/2 and π/2) ---
# C': (-3π/2, -∞) ↑ to (-3π/2, 0) → right to (π/2, 0) ↓ to (π/2, -∞)
lines!(ax, fill(-3π/2, 40), collect(range(-2.5, 0, length = 40)),
    color = :red3, linewidth = 2, label = "C'")
lines!(ax, collect(range(-3π/2, π/2, length = 50)), fill(-0.04, 50),
    color = :red3, linewidth = 2)
lines!(ax, fill(π/2, 40), collect(range(0, -2.5, length = 40)),
    color = :red3, linewidth = 2)
arrows!(ax, [-3π/2], [-1.5], [0.0], [0.3],
    color = :red3, linewidth = 2, arrowsize = 12)
arrows!(ax, [-π/2], [-0.04], [0.3], [0.0],
    color = :red3, linewidth = 2, arrowsize = 12)
arrows!(ax, [π/2], [-0.3], [0.0], [-0.3],
    color = :red3, linewidth = 2, arrowsize = 12)

# --- Saddle points ---
scatter!(ax, [π, -π], [0.0, 0.0], color = :black, markersize = 12, marker = :circle)
text!(ax, π + 0.15, 0.3, text = "+π", fontsize = 13, font = :bold)
text!(ax, -π + 0.15, 0.3, text = "−π", fontsize = 13, font = :bold)

# --- Pole locations (example: half-plane n=2, φ−φ'=π/4) ---
# Poles of cot((β+z)/(2n)) at z = -β + 2nπN
beta = π/4; n_w = 2.0
for N in -1:1
    zp = -beta + 2π * n_w * N
    if -2.2π < zp < 2.2π
        scatter!(ax, [zp], [0.0], color = :orange, markersize = 14, marker = :xcross)
    end
end
text!(ax, -beta + 0.15, -0.35, text = "poles (×)", fontsize = 11, color = :orange)

# --- Legend ---
axislegend(ax, position = :lt, framevisible = true, labelsize = 12)

xlims!(ax, -5.5, 5.5)
ylims!(ax, -2.8, 2.8)

fig # hide
```

Using this closed contour, we can write:

```math
\frac{1}{4\pi in}\int_{C'-C} H_1(z)\,e^{ik\rho\,h(z)}\,dz = \frac{1}{4\pi in}\oint_{C_T} H_1(z)\,e^{ik\rho\,h(z)}\,dz - \frac{1}{4\pi in}\int_{\text{SDP}_{+\pi}} H_1(z)\,e^{ik\rho\,h(z)}\,dz - \frac{1}{4\pi in}\int_{\text{SDP}_{-\pi}} H_1(z)\,e^{ik\rho\,h(z)}\,dz,
```

where ``H_1(z) = \cot\!\left(\frac{\phi - \phi' + z}{2n}\right)``.

The three terms on the right have distinct physical interpretations:
1. **Closed contour integral** ``\oint_{C_T}`` → evaluated by **residue calculus** → gives the **geometrical optics (GO) field**.
2. **SDP integrals** → evaluated by **steepest descent** → give the **diffracted field**.

### Residue contributions: the GO field

The integrand ``\frac{N(z)}{D(z)} = \frac{\cos[(\phi-\phi'+z)/(2n)]\,e^{ik\rho\cos z}}{4\pi in\sin[(\phi-\phi'+z)/(2n)]}`` has simple poles where

```math
\frac{\phi - \phi' + z}{2n} = N\pi, \qquad N = 0, \pm 1, \pm 2, \ldots
```

i.e., at ``z_p = -(\phi - \phi') + 2\pi nN``. Only poles enclosed by ``C_T`` (i.e., within ``-\pi \le \operatorname{Re}(z_p) \le +\pi``) contribute.

Computing the residues (using ``\operatorname{Res} = N(z_p)/D'(z_p)``):

```math
\operatorname{Res}(z = z_p) = \frac{1}{2\pi i}\, e^{ik\rho\cos[-(\phi-\phi') + 2\pi nN]}.
```

Summing over enclosed poles and using the unit step function ``U`` to track which poles are inside the contour:

```math
\frac{1}{4\pi in}\oint_{C_T} H_1(z)\,e^{ik\rho\,h(z)}\,dz = e^{ik\rho\cos(\phi - \phi')}\,U[\pi - |\phi - \phi'|].
```

This is precisely the **incident GO field**: it equals ``e^{ik\rho\cos(\phi-\phi')}`` when ``|\phi - \phi'| < \pi`` (observer in the lit region) and zero otherwise (observer in the shadow). The step function implements the ISB.

An identical analysis for the ``H_2(z) = \cot[(\phi+\phi'+z)/(2n)]`` integral yields the **reflected GO field**, with the RSB implemented by a similar step function.

### Saddle-point contributions: the diffracted field

The SDP integrals give the diffracted field. The evaluation depends on whether the poles of the cotangent are **near** or **far from** the saddle points at ``z_s = \pm\pi``.

**Case 1: Poles far from saddle points (conventional steepest descent).**
When the observer is away from shadow boundaries, the cotangent varies slowly near the saddle point. The standard steepest descent formula gives, for the SDP through ``z_s = +\pi``:

```math
\frac{1}{4\pi in}\int_{\text{SDP}_{+\pi}} H_1(z)\,e^{ik\rho\,h(z)}\,dz \approx \frac{e^{-i\pi/4}}{2n\sqrt{2\pi k}}\,\cot\!\left[\frac{\pi + (\phi - \phi')}{2n}\right] \frac{e^{-ik\rho}}{\sqrt{\rho}},
```

and similarly for ``\text{SDP}_{-\pi}`` with ``+`` replaced by ``-`` in the cotangent argument. Combining the two SDPs for ``H_1``:

```math
\left.\mathcal{F}_D^{(1)}\right|_{\text{SDPs}} \approx \frac{e^{-i\pi/4}}{2n\sqrt{2\pi k}} \left[\cot\!\left(\frac{\pi + (\phi-\phi')}{2n}\right) + \cot\!\left(\frac{\pi - (\phi-\phi')}{2n}\right)\right] \frac{e^{-ik\rho}}{\sqrt{\rho}}.
```

This is the **incident diffracted field**. It exists everywhere (not just in shadow) and is proportional to ``e^{-ik\rho}/\sqrt{\rho}`` — the characteristic cylindrical-wave spreading of edge diffraction.

Similarly, the ``H_2`` integral gives the **reflected diffracted field** with ``\phi - \phi'`` replaced by ``\phi + \phi'``.

**Case 2: Poles near saddle points (Pauli--Clemmow modified steepest descent).**
At or near shadow boundaries, a pole of the cotangent approaches a saddle point, and the conventional steepest descent breaks down (the cotangent varies rapidly). The **Pauli--Clemmow modified steepest descent** method handles this by splitting the contribution into a singular part (that cancels the GO discontinuity) and a smooth remainder involving a **Fresnel integral**.

This is the crucial step that distinguishes UTD from GTD: it introduces the **transition function** ``F``.

## Stage 5: The transition function from Pauli--Clemmow steepest descent

### Derivation of ``F``

Near the ISB, let ``\phi - \phi' = \pi - \varepsilon`` where ``\varepsilon`` is small. The singular cotangent term behaves as ``\cot(\varepsilon/(2n)) \approx 2n/\varepsilon``. The Pauli--Clemmow method evaluates the SDP integral by expressing it as:

```math
\frac{1}{4\pi in}\int_{\text{SDP}_{+\pi}} H_1(z)\,e^{ik\rho\,h_1(z)}\,dz = \frac{e^{-i\pi/4}}{2n\sqrt{2\pi k}}\,\cot\!\left[\frac{\pi + (\phi-\phi')}{2n}\right] \cdot F\!\left[k\rho\,g^+(\phi-\phi')\right] \cdot \frac{e^{-ik\rho}}{\sqrt{\rho}},
```

where ``F`` is the UTD transition function that **smoothly interpolates** between:
- ``F \to 1`` when ``g^+ \gg 0`` (far from boundary → recovers conventional steepest descent result),
- ``F \to 0`` when ``g^+ \to 0`` (at boundary → tames the cotangent singularity).

The function ``g^{\pm}`` measures the separation between the pole and the saddle point:

```math
g^{\pm}(\phi \mp \phi') = 1 + \cos[(\phi \mp \phi') - 2\pi n N^{\pm}],
```

where ``N^{\pm}`` is the integer that most closely satisfies the boundary condition ``2\pi n N^{\pm} - (\phi \mp \phi') = \pm\pi``.

### Transition function: Balanis vs. our convention

The Pauli--Clemmow method produces the transition function in the form of a Fresnel integral. In the Balanis/KP convention:

```math
F_{\text{KP}}(X) = 2i\sqrt{X}\,e^{iX} \int_{\sqrt{X}}^{\infty} e^{-i\tau^2}\,d\tau.
```

This is equivalent to (derived in the [Transition Function](@ref transition) chapter):

```math
F_{\text{KP}}(X) = \sqrt{\pi X}\;e^{+i\pi/4}\;\operatorname{erfcx}\!\bigl(e^{+i\pi/4}\sqrt{X}\bigr).
```

The transition function used in **this package** is the complex conjugate of the KP form:

```math
\boxed{F(x) = \sqrt{\pi x}\;e^{-i\pi/4}\;\operatorname{erfcx}\!\bigl(e^{-i\pi/4}\sqrt{x}\bigr) = [F_{\text{KP}}(x)]^*.}
```

Both forms share identical key properties:
- ``|F_{\text{KP}}(x)| = |F(x)|`` for all ``x``,
- ``F_{\text{KP}}(x) \to 1`` and ``F(x) \to 1`` as ``x \to +\infty`` (GTD recovery),
- ``F_{\text{KP}}(0) = F(0) = 0`` (singularity taming).

They differ only in phase for small ``x``: ``\operatorname{Im}(F_{\text{KP}}) > 0`` vs. ``\operatorname{Im}(F) < 0``. The choice of branch determines the sign of ``e^{\mp i\pi/4}`` in the erfcx argument. Both conventions produce the **same total field** ``u^{\text{GO}} + u^d`` because the complex conjugation is consistently absorbed throughout the diffraction coefficient.

!!! note "Why two conventions exist"
    The two forms arise from the two possible orientations of the steepest descent contour through the saddle point, which produce Fresnel integrands ``e^{-i\tau^2}`` and ``e^{+i\tau^2}`` respectively. Balanis and the original KP paper (1974) use the ``e^{-i\tau^2}`` parametrisation, producing ``F_{\text{KP}}`` (with ``e^{+i\pi/4}``). This package uses the ``e^{+i\tau^2}`` parametrisation, producing our ``F`` (with ``e^{-i\pi/4}``). Both are equally valid and produce identical total fields. The full derivation of ``F`` from first principles is given in [The UTD Transition Function](@ref transition).

## The Keller (GTD) diffraction coefficient

Before presenting the full UTD result, we note the simpler **GTD** coefficient that the steepest descent method produces when the transition function is replaced by its far-field limit ``F \to 1``.

### Incident diffraction coefficient

Collecting the two SDP contributions from the ``\xi^- = \phi - \phi'`` integral:

```math
D^i(\phi - \phi', n) = -\frac{e^{-i\pi/4}}{2n\sqrt{2\pi k}} \left\{\cot\!\left(\frac{\pi + (\phi - \phi')}{2n}\right) + \cot\!\left(\frac{\pi - (\phi - \phi')}{2n}\right)\right\}.
```

### Reflected diffraction coefficient

Similarly from the ``\xi^+ = \phi + \phi'`` integral:

```math
D^r(\phi + \phi', n) = -\frac{e^{-i\pi/4}}{2n\sqrt{2\pi k}} \left\{\cot\!\left(\frac{\pi + (\phi + \phi')}{2n}\right) + \cot\!\left(\frac{\pi - (\phi + \phi')}{2n}\right)\right\}.
```

### Soft and hard GTD coefficients

The soft and hard polarisation coefficients are (Balanis Eq. 13-70c,d):

```math
\begin{aligned}
D_s^{\text{GTD}} &= D^i - D^r, \\[4pt]
D_h^{\text{GTD}} &= D^i + D^r.
\end{aligned}
```

The ``-`` sign for soft polarisation comes from the PEC reflection coefficient ``R = -1`` (soft/Dirichlet), while ``+`` comes from ``R = +1`` (hard/Neumann).

Written explicitly:

```math
D_s^{\text{GTD}} = -\frac{e^{-i\pi/4}}{2n\sqrt{2\pi k}} \left[\cot\!\left(\frac{\pi + (\phi-\phi')}{2n}\right) + \cot\!\left(\frac{\pi - (\phi-\phi')}{2n}\right) - \cot\!\left(\frac{\pi + (\phi+\phi')}{2n}\right) - \cot\!\left(\frac{\pi - (\phi+\phi')}{2n}\right)\right].
```

```math
D_h^{\text{GTD}} = -\frac{e^{-i\pi/4}}{2n\sqrt{2\pi k}} \left[\cot\!\left(\frac{\pi + (\phi-\phi')}{2n}\right) + \cot\!\left(\frac{\pi - (\phi-\phi')}{2n}\right) + \cot\!\left(\frac{\pi + (\phi+\phi')}{2n}\right) + \cot\!\left(\frac{\pi - (\phi+\phi')}{2n}\right)\right].
```

These are Keller's diffraction coefficients. They are valid everywhere **except** at shadow and reflection boundaries, where the cotangent functions diverge.

## The full UTD four-term diffraction coefficient

### From GTD to UTD

The UTD coefficient is obtained by replacing each bare cotangent ``\cot(\psi_j)`` with the product ``\cot(\psi_j)\,F(X_j)``, where ``F(X_j)`` is the transition function evaluated at an argument ``X_j`` that measures proximity to the corresponding boundary:

```math
\boxed{D_{s/h}(\phi, \phi'; k, L) = C(k, n) \sum_{j=1}^{4} \sigma_j^{s/h}\,\cot(\psi_j)\,F(X_j),}
```

where the four terms correspond to:

| Term ``j`` | Angular variable | Cotangent argument ``\psi_j`` | Physical origin |
|:-----------|:-----------------|:------------------------------|:----------------|
| 1 | ``\beta^- = \phi - \phi'`` | ``(\pi + \beta^-)/(2n)`` | ISB, SDP at ``+\pi`` |
| 2 | ``\beta^- = \phi - \phi'`` | ``(\pi - \beta^-)/(2n)`` | ISB, SDP at ``-\pi`` |
| 3 | ``\beta^+ = \phi + \phi'`` | ``(\pi + \beta^+)/(2n)`` | RSB, SDP at ``+\pi`` |
| 4 | ``\beta^+ = \phi + \phi'`` | ``(\pi - \beta^+)/(2n)`` | RSB, SDP at ``-\pi`` |

We now define each ingredient precisely.

## Angle wrapping

For numerical robustness, both the observation azimuth ``\phi`` and the incident azimuth ``\phi'`` are normalised to the range ``[0, \alpha)`` before any computation:

```math
\operatorname{wrap}_\alpha(\varphi) \equiv \varphi - \alpha\left\lfloor \frac{\varphi}{\alpha} \right\rfloor,
```

where ``\lfloor \cdot \rfloor`` is the floor function. This is simply the modular arithmetic operation ``\varphi \bmod \alpha``.

```@example kp
using UTDKernels
# Wrapping examples
println("wrap(π/2, 2π) = $(wrap_angle(π/2, 2π))")        # π/2 (no change)
println("wrap(3π, 2π)  = $(wrap_angle(3π, 2π))")          # π
println("wrap(-π/4, 2π) = $(wrap_angle(-π/4, 2π))")       # 7π/4
```

## Angular differences

The four KP terms are built from two fundamental angular differences:

```math
\beta^- \equiv \phi - \phi', \qquad \beta^+ \equiv \phi + \phi'.
```

- ``\beta^-`` governs the **incident shadow boundary** (ISB): terms 1 and 2.
- ``\beta^+`` governs the **reflection shadow boundaries** (RSB): terms 3 and 4.

## The four cotangent arguments ``\psi_j``

Each KP term has a cotangent factor ``\cot(\psi_j)`` where

```math
\psi_j = \frac{\pi + (-1)^{j+1}\,\beta_j}{2n},
```

with ``\beta_1 = \beta_2 = \beta^-`` and ``\beta_3 = \beta_4 = \beta^+``. Written explicitly:

```math
\begin{aligned}
\psi_1 &= \frac{\pi + \beta^-}{2n}, \\[4pt]
\psi_2 &= \frac{\pi - \beta^-}{2n}, \\[4pt]
\psi_3 &= \frac{\pi + \beta^+}{2n}, \\[4pt]
\psi_4 &= \frac{\pi - \beta^+}{2n}.
\end{aligned}
```

### Tracing the origin of ``\psi_j``

Each ``\psi_j`` arises from one specific combination of:
- which angular variable (``\xi^- = \phi - \phi'`` or ``\xi^+ = \phi + \phi'``), and
- which saddle point (``z_s = +\pi`` or ``z_s = -\pi``).

Term 1 comes from ``\cot[(\xi^- + z)/(2n)]`` evaluated at ``z_s = +\pi``: ``\psi_1 = (\pi + \xi^-)/(2n)``.
Term 2 comes from ``\cot[(\xi^- + z)/(2n)]`` evaluated at ``z_s = -\pi``: ``\psi_2 = (-\pi + \xi^-)/(2n)``. By the identity ``\cot(-x) = -\cot(x)`` and the ``-`` sign from the SDP orientation, this is equivalent to using ``\psi_2 = (\pi - \xi^-)/(2n)`` with a ``+`` sign.

### Physical meaning of the cotangent poles

The cotangent ``\cot(\psi_j)`` has poles (diverges to ``\pm\infty``) when ``\psi_j = m\pi`` for integer ``m``. This occurs when

```math
\frac{\pi \pm \beta_j}{2n} = m\pi \quad \Longrightarrow \quad \beta_j = 2mn\pi \mp \pi.
```

These are precisely the angular locations of the **shadow and reflection boundaries**. For example, for the half-plane (``n=2``) with ``\phi' = \pi/4``:

- Term 2: ``\psi_2 = m\pi`` when ``\beta^- = \phi - \phi' = \pi``, i.e., ``\phi = \pi + \pi/4 = 225°``. This is the ISB.
- Term 4: ``\psi_4 = m\pi`` when ``\beta^+ = \phi + \phi' = \pi``, i.e., ``\phi = \pi - \pi/4 = 135°``. This is the RSB from the ``\phi=0`` face.

The four terms together capture all shadow and reflection boundaries for the given wedge geometry.

## Boundary-tracking integers ``N_j``

For each term, an integer ``N_j`` identifies which boundary is nearest. It is computed by rounding:

```math
\begin{aligned}
N_1 &= \operatorname{round}\!\left(\frac{\beta^- + \pi}{2n\pi}\right), &\quad
N_2 &= \operatorname{round}\!\left(\frac{\beta^- - \pi}{2n\pi}\right), \\[4pt]
N_3 &= \operatorname{round}\!\left(\frac{\beta^+ + \pi}{2n\pi}\right), &\quad
N_4 &= \operatorname{round}\!\left(\frac{\beta^+ - \pi}{2n\pi}\right).
\end{aligned}
```

**Origin from the steepest descent**: ``N_j`` is the integer ``m`` for which the pole ``z_p = -\beta_j + 2\pi n m`` is closest to the saddle point. This is equivalent to rounding ``(\beta_j \pm \pi)/(2n\pi)`` to the nearest integer. At the exact boundary, the rounding is exact: ``N_j = m``.

The values ``N^{\pm}`` correspond to those in Balanis Eqs. (13-66e,f): ``2\pi nN^+ - (\phi \mp \phi') = +\pi`` and ``2\pi nN^- - (\phi \mp \phi') = -\pi``.

## Distance parameters ``a_j``

The distance parameter for each term measures how far the observation angle is from the nearest boundary:

```math
a_j = 2\cos^2\!\left(\frac{2n\pi N_j - \beta_j}{2}\right), \qquad j = 1, \ldots, 4.
```

In Balanis Eq. (13-66c,d) the quantity ``g^{\pm} = 1 + \cos[(\phi \mp \phi') - 2\pi n N^{\pm}]`` is defined. By the half-angle identity ``1 + \cos\theta = 2\cos^2(\theta/2)``, Balanis's ``g^{\pm}`` is identical to our ``a_j``. (Note: the signed square-root form ``g_j = \cos(\theta/2)`` satisfying ``a_j = 2g_j^2`` is introduced [below](@ref signed-distance).)

### Why ``a_j`` vanishes at boundaries

At a shadow/reflection boundary, ``\psi_j = m\pi`` and ``N_j = m``, so

```math
2n\pi N_j - \beta_j = 2mn\pi - \beta_j.
```

From the boundary condition ``\beta_j = 2mn\pi \mp \pi``:

```math
2mn\pi - \beta_j = 2mn\pi - (2mn\pi \mp \pi) = \pm\pi.
```

Therefore:

```math
a_j = 2\cos^2\!\left(\frac{\pm\pi}{2}\right) = 2\cos^2\!\left(\frac{\pi}{2}\right) = 2 \cdot 0 = 0.
```

Away from boundaries, ``a_j > 0``.

### Quadratic vanishing

Near a boundary, let the angular deviation from the exact boundary be ``\delta`` (small). Then

```math
\frac{2n\pi N_j - \beta_j}{2} \approx \frac{\pi}{2} - \frac{\delta}{2},
```

so

```math
a_j = 2\cos^2\!\left(\frac{\pi}{2} - \frac{\delta}{2}\right) = 2\sin^2\!\left(\frac{\delta}{2}\right) \approx \frac{\delta^2}{2}.
```

This **quadratic** vanishing is what ensures the product ``\cot(\psi_j) \cdot F(X_j)`` has a finite limit (see [Numerical Methods](@ref numerical)).

## [Signed distance ``g_j``](@id signed-distance)

It is sometimes convenient to define the signed quantity

```math
g_j \equiv \cos\!\left(\frac{2n\pi N_j - \beta_j}{2}\right),
```

so that ``a_j = 2g_j^2``. Proximity to a transition boundary is detected by ``|g_j| \approx 0``.

## Transition-function arguments ``X_j``

```math
X_j \equiv k\,L\,a_j, \qquad j = 1, \ldots, 4.
```

Since ``a_j \ge 0`` (for real arguments), we have ``X_j \ge 0``. Equivalently, ``X_j = 2kL\,g_j^2``.

**Physical meaning**: ``X_j`` is proportional to the product of ``kL`` (measuring frequency × distance from the edge) and ``a_j`` (measuring angular distance from the nearest boundary). When ``X_j`` is small, the observer is in the transition region for term ``j``, and ``F(X_j) \ll 1``; when ``X_j`` is large, ``F(X_j) \approx 1``.

**Connection to Balanis**: the argument ``k\rho\,g^{\pm}`` in Balanis Eq. (13-68e,f) becomes ``kL\,a_j = kL \cdot 2(g_j)^2`` in our notation, with ``\rho \to L`` (the effective distance parameter accounting for wavefront curvature).

## Common prefactor ``C(k,n)``

```math
C(k, n) \equiv -\frac{e^{-i\pi/4}}{2n\sqrt{2\pi k}}.
```

This prefactor arises from asymptotic matching of the UTD solution to the exact canonical (Sommerfeld) solution. It contains:
- the phase ``e^{-i\pi/4}`` from the Fresnel integral asymptotics (specifically, ``\sqrt{2\pi/(k\rho)} \cdot e^{-i\pi/4}`` from the saddle-point evaluation),
- the factor ``1/(2n)`` from the wedge periodicity (the ``1/n`` from the Green's function, combined with the ``1/2`` from the contour integral),
- the factor ``1/\sqrt{2\pi k}`` from the high-frequency normalisation.

**Tracing ``C`` through the derivation**: From the SDP evaluation (Balanis Eq. 13-60a), each saddle-point contribution carries a factor ``\frac{e^{-i\pi/4}}{2n\sqrt{2\pi k}} \cdot \frac{1}{\sqrt{\rho}}``. The ``1/\sqrt{\rho}`` is part of the spreading factor ``A(s,s') = 1/\sqrt{\rho}``, leaving ``C(k,n)`` as the prefactor of the diffraction coefficient proper.

## PEC sign factors ``\sigma_j``

The sign factors enforce the PEC boundary conditions on both wedge faces:

```math
\begin{aligned}
\text{Soft (Dirichlet):} \quad &\sigma_1^s = +1,\;\; \sigma_2^s = +1,\;\; \sigma_3^s = -1,\;\; \sigma_4^s = -1, \\[4pt]
\text{Hard (Neumann):} \quad &\sigma_1^h = +1,\;\; \sigma_2^h = +1,\;\; \sigma_3^h = +1,\;\; \sigma_4^h = +1.
\end{aligned}
```

### Physical meaning

- **Terms 1 and 2** (``\beta^-`` terms): Both have ``\sigma = +1`` for both polarisations. These terms produce the incident shadow boundary behaviour.
- **Terms 3 and 4** (``\beta^+`` terms): For **soft** polarisation, ``\sigma_3 = \sigma_4 = -1``. For **hard**, ``\sigma_3 = \sigma_4 = +1``.

### Tracing the sign factors through the derivation

The soft coefficient is ``D_s = D^i - D^r`` (Balanis Eq. 13-70c), and the hard coefficient is ``D_h = D^i + D^r`` (Eq. 13-70d). Since ``D^i`` involves the ``\beta^-`` cotangents (terms 1,2) and ``D^r`` involves the ``\beta^+`` cotangents (terms 3,4):

- ``D_s = D^i - D^r``: terms 1,2 get ``+1``, terms 3,4 get ``-1`` → ``\sigma^s = (+1, +1, -1, -1)``.
- ``D_h = D^i + D^r``: all terms get ``+1`` → ``\sigma^h = (+1, +1, +1, +1)``.

The ``-`` sign on ``D^r`` for soft polarisation comes from the PEC reflection coefficient ``R = -1`` (Dirichlet: the reflected field has opposite sign to maintain ``E_z = 0`` on the surface). For hard polarisation, ``R = +1`` (Neumann: the reflected field has the same sign to maintain ``\partial H_z/\partial n = 0``).

## Summary: the complete coefficient

Combining all ingredients:

```math
\boxed{D_{s/h}(\phi, \phi'; k, L) = -\frac{e^{-i\pi/4}}{2n\sqrt{2\pi k}} \sum_{j=1}^{4} \sigma_j^{s/h}\,\cot(\psi_j)\,F(X_j).}
```

```@example kp
# Compute diffraction coefficients for a half-plane
w = Wedge(2π)
ang = RayAngles(π/2, π/4)   # φ = 90°, φ' = 45°
k = 10.0
L = 1.0

Ds, Dh = pec_wedge_DsDh(w, ang, k, L)
println("|Ds| = $(round(abs(Ds), digits=6))")
println("|Dh| = $(round(abs(Dh), digits=6))")
```

```@example kp
# Inspect individual KP terms
inspect_kp_terms(w, ang, k, L)
```

## The GTD limit

When ``kL \to \infty`` (high frequency, far from the edge), all four ``X_j \to \infty`` (provided the observer is away from boundaries, so ``a_j > 0``). Since ``F(X_j) \to 1``:

```math
D_{s/h}^{\text{GTD}}(\phi, \phi'; k) = C(k, n) \sum_{j=1}^{4} \sigma_j^{s/h}\,\cot(\psi_j).
```

This is the **Keller GTD coefficient**, which was the pre-UTD result. The convergence rate is ``O(1/kL)``, reflecting the leading-order correction in the asymptotic expansion ``F(x) \approx 1 - 1/(2x) + \cdots`` for large ``x``.

```@example kp
# Demonstrate GTD convergence
phi_test = π/2
phip = π/4

for kL in [1e1, 1e3, 1e5]
    Ds_utd, _ = pec_wedge_DsDh(w, RayAngles(phi_test, phip), 1.0, kL)
    Ds_utd_norm = Ds_utd * sqrt(1.0)  # normalise out √k dependence

    # GTD reference (F = 1)
    n = wedge_n(w)
    terms = UTDKernels.kp_four_terms(phi_test, phip, n)
    C = -exp(-im*π/4) / (2n * sqrt(2π * 1.0))
    Ds_gtd = C * sum(UTDKernels.PEC_SIGMA_SOFT[j] * cot(terms.psi[j]) for j in 1:4)

    err = abs(Ds_utd - Ds_gtd) / abs(Ds_gtd)
    println("kL = $kL:  relative error = $(round(err, sigdigits=3))")
end
```

The following figure shows the soft diffraction coefficient ``|D_s|`` as a function of observation angle ``\phi`` for the half-plane. The UTD coefficient (solid) is finite everywhere, while the GTD coefficient (dashed, obtained by setting ``F = 1``) diverges at the shadow and reflection boundaries. This is the key practical advantage of UTD over GTD.

```@example kp
using CairoMakie

w = Wedge(2π)
phip = π/4; k = 1.0; L = 1.0
n = wedge_n(w)
isb_deg = rad2deg(π + phip)
rsb_deg = rad2deg(π - phip)

phis = range(0.01, 2π - 0.01, length = 800)

Ds_utd_vals = Float64[]
Ds_gtd_vals = Float64[]

for phi in phis
    Ds, _ = pec_wedge_DsDh(w, RayAngles(phi, phip), k, L)
    push!(Ds_utd_vals, abs(Ds))

    phi_w  = wrap_angle(phi, 2π)
    phip_w = wrap_angle(phip, 2π)
    terms = UTDKernels.kp_four_terms(phi_w, phip_w, n)
    C = -exp(-im*π/4) / (2n * sqrt(2π * k))
    Ds_gtd = C * sum(UTDKernels.PEC_SIGMA_SOFT[j] * cot(terms.psi[j]) for j in 1:4)
    push!(Ds_gtd_vals, min(abs(Ds_gtd), 3.0))   # clamp for display
end

fig = Figure(size = (600, 350))
ax = Axis(fig[1, 1], xlabel = "φ  [deg]", ylabel = "|Ds|",
    title = "UTD vs GTD diffraction coefficient  (half-plane, φ' = 45°, kL = 1)",
    xticks = [0, 45, 90, 135, 180, 225, 270, 315, 360])

lines!(ax, rad2deg.(phis), Ds_utd_vals, color = :royalblue, linewidth = 2, label = "UTD")
lines!(ax, rad2deg.(phis), Ds_gtd_vals, color = :red3, linewidth = 1.5,
    linestyle = :dash, label = "GTD (F = 1)")

vlines!(ax, [isb_deg], color = :gray50, linestyle = :dash, linewidth = 1)
vlines!(ax, [rsb_deg], color = :gray50, linestyle = :dashdot, linewidth = 1)
text!(ax, isb_deg + 3, 2.7, text = "ISB", fontsize = 11, color = :gray50)
text!(ax, rsb_deg + 3, 2.7, text = "RSB", fontsize = 11, color = :gray50)

ylims!(ax, 0, 3.0)
axislegend(ax, position = :rt, framevisible = true, labelsize = 12)

fig # hide
```

## Reciprocity

An important symmetry of the KP coefficients is **reciprocity**:

```math
D_{s/h}(\phi, \phi') = D_{s/h}(\phi', \phi).
```

**Proof**: Swapping ``\phi \leftrightarrow \phi'`` sends ``\beta^- \to -\beta^-`` while ``\beta^+ \to \beta^+`` (unchanged). Under ``\beta^- \to -\beta^-``:

```math
\psi_1 = \frac{\pi + \beta^-}{2n} \;\longrightarrow\; \frac{\pi - \beta^-}{2n} = \psi_2,
```

and similarly ``\psi_2 \to \psi_1``. So terms 1 and 2 **swap**. Since ``\sigma_1 = \sigma_2`` for both soft and hard, the sum of terms 1 and 2 is invariant.

The same argument applies to ``N_1 \leftrightarrow N_2`` and ``a_1 \leftrightarrow a_2`` (and hence ``X_1 \leftrightarrow X_2``). Terms 3 and 4 are unchanged because ``\beta^+`` is symmetric in ``\phi`` and ``\phi'``.

Therefore the full sum is unchanged, proving reciprocity. ``\square``

```@example kp
# Numerical verification of reciprocity
w = Wedge(3π/2)
Ds_fwd, Dh_fwd = pec_wedge_DsDh(w, RayAngles(π/4, π/3), 10.0, 1.0)
Ds_rev, Dh_rev = pec_wedge_DsDh(w, RayAngles(π/3, π/4), 10.0, 1.0)
println("|Ds_fwd - Ds_rev| = $(abs(Ds_fwd - Ds_rev))")
println("|Dh_fwd - Dh_rev| = $(abs(Dh_fwd - Dh_rev))")
```

## Shadow-boundary continuity

The most important property of UTD is that the **total field** ``u^{\text{GO}} + u^d`` is continuous across every shadow boundary, despite the step discontinuity in ``u^{\text{GO}}``.

### How the cancellation works

At the ISB (say ``\phi - \phi' = \pi``), let ``\varepsilon`` denote the small angular deviation from the exact boundary:
1. The GO field ``u^{\text{GO}}`` drops by ``-\frac{1}{2}e^{-ik\rho}`` (the incident field switches off).
2. The singular cotangent term ``\cot(\psi_j)`` diverges as ``\pm 2n/\varepsilon``.
3. The transition function ``F(X_j)`` vanishes as ``\sqrt{\pi k\rho/2}\;e^{-i\pi/4}\,|\varepsilon|`` (since ``X_j = k\rho\,a_j \approx k\rho\,\varepsilon^2/2``).
4. The product ``\cot(\psi_j) \cdot F(X_j)`` tends to a finite limit ``\pm n\sqrt{2\pi k\rho}\;e^{-i\pi/4}`` (the ``\pm`` gives opposite signs from the two sides of the boundary; cf. Balanis Eq. 13-82 after adjusting for our conjugate convention).
5. Combined with the prefactor ``C(k,n)`` and ``e^{-ik\rho}/\sqrt{\rho}``, the diffracted field jumps by exactly ``+\frac{1}{2}e^{-ik\rho}``, cancelling the GO discontinuity.

This is the fundamental achievement of UTD over GTD: the transition function ``F`` was specifically designed (via the Pauli--Clemmow method) to produce this exact cancellation.
