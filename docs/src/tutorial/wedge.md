# [Wedge Geometry and Geometrical Optics](@id wedge)

This chapter defines the canonical PEC wedge geometry, identifies the shadow and reflection boundaries, introduces the ray-geometric quantities, and sets up the UTD diffracted-field representation.

## Canonical wedge geometry

Consider a perfectly electrically conducting (PEC) straight wedge whose edge lies along the ``z``-axis. We use cylindrical coordinates ``(\rho, \phi, z)`` centred on the edge. The two wedge faces are the half-planes

```math
\phi = 0 \qquad \text{and} \qquad \phi = \alpha,
```

where ``\alpha \in (0, 2\pi]`` is the **exterior (free-space) angle** --- the angular extent of the region accessible to electromagnetic fields. The wedge material fills the complement, from ``\phi = \alpha`` to ``\phi = 2\pi`` (when ``\alpha < 2\pi``).

```@example wedge_params
using UTDKernels
# Half-plane: α = 2π (no wedge material, just a thin sheet)
w_hp = Wedge(2π)
# Right-angle wedge: α = 3π/2 (90° interior solid angle)
w_ra = Wedge(3π/2)
# Flat plane: α = π (no edge, no diffraction)
w_fp = Wedge(π)
nothing # hide
```

The figure below shows the half-plane geometry (``\alpha = 2\pi``) with the incident, reflected, and diffracted rays, together with the shadow and reflection boundaries.

```@example wedge_params
using CairoMakie

fig = Figure(size = (600, 550))
ax = Axis(fig[1, 1], aspect = DataAspect(),
    xlabel = "x / ρ", ylabel = "y / ρ",
    title = "Half-plane wedge geometry  (α = 2π, n = 2)")

# Parameters
phip = π/4          # incident direction
R = 1.0             # normalised distance
isb = π + phip      # ISB angle
rsb = π - phip      # RSB angle

# --- Wedge face (thick line along positive x-axis) ---
lines!(ax, [0.0, 1.4], [0.0, 0.0], color = :black, linewidth = 4)
text!(ax, 1.35, -0.08, text = "PEC face (φ = 0)", fontsize = 11, align = (:right, :top))

# --- Edge marker ---
scatter!(ax, [0.0], [0.0], color = :black, markersize = 10)
text!(ax, 0.07, -0.07, text = "edge", fontsize = 11)

# --- Incident ray ---
x_i = [R * cos(phip), 0.0]
y_i = [R * sin(phip), 0.0]
lines!(ax, x_i, y_i, color = :royalblue, linewidth = 2)
arrows!(ax, [0.5cos(phip)], [0.5sin(phip)],
    [-0.15cos(phip)], [-0.15sin(phip)], color = :royalblue, linewidth = 2, arrowsize = 12)
text!(ax, 0.75cos(phip) + 0.05, 0.75sin(phip) + 0.08,
    text = "incident (φ')", fontsize = 12, color = :royalblue)

# --- Reflected ray (specular off φ=0 face: π - φ') ---
phi_r = π - phip     # Snell's law: angle of incidence = angle of reflection
x_r = [0.0, 0.85 * cos(phi_r)]
y_r = [0.0, 0.85 * sin(phi_r)]
lines!(ax, x_r, y_r, color = :red3, linewidth = 2)
arrows!(ax, [0.3cos(phi_r)], [0.3sin(phi_r)],
    [0.15cos(phi_r)], [0.15sin(phi_r)], color = :red3, linewidth = 2, arrowsize = 12)
text!(ax, 0.6cos(phi_r) + 0.15, 0.6sin(phi_r) + 0.08,
    text = "reflected", fontsize = 12, color = :red3)

# --- Diffracted ray (example, going into shadow) ---
phi_d = 1.5π   # example diffracted direction
x_d = [0.0, 0.9cos(phi_d)]
y_d = [0.0, 0.9sin(phi_d)]
lines!(ax, x_d, y_d, color = :green4, linewidth = 2, linestyle = :dot)
arrows!(ax, [0.35cos(phi_d)], [0.35sin(phi_d)],
    [0.15cos(phi_d)], [0.15sin(phi_d)], color = :green4, linewidth = 2, arrowsize = 12)
text!(ax, 0.65cos(phi_d) - 0.12, 0.65sin(phi_d),
    text = "diffracted", fontsize = 12, color = :green4)

# --- ISB (dashed) ---
lines!(ax, [0.0, 1.3cos(isb)], [0.0, 1.3sin(isb)],
    color = :gray40, linewidth = 1.5, linestyle = :dash)
text!(ax, 1.15cos(isb) - 0.15, 1.15sin(isb) - 0.06,
    text = "ISB (π + φ')", fontsize = 11, color = :gray40)

# --- RSB (dashed) ---
lines!(ax, [0.0, 1.3cos(rsb)], [0.0, 1.3sin(rsb)],
    color = :gray40, linewidth = 1.5, linestyle = :dashdot)
text!(ax, 1.15cos(rsb) + 0.05, 1.15sin(rsb) + 0.06,
    text = "RSB (π − φ')", fontsize = 11, color = :gray40)

# --- Region labels (arcs with text) ---
# Lit region arc
theta_lit = range(rsb + 0.1, isb - 0.1, length = 50)
lines!(ax, 1.2 .* cos.(theta_lit), 1.2 .* sin.(theta_lit),
    color = (:royalblue, 0.3), linewidth = 8)
mid_lit = (rsb + isb) / 2
text!(ax, 1.32cos(mid_lit), 1.32sin(mid_lit),
    text = "incident only", fontsize = 10, color = :royalblue, align = (:center, :center))

# Reflection region arc
theta_refl = range(0.05, rsb - 0.1, length = 30)
lines!(ax, 1.2 .* cos.(theta_refl), 1.2 .* sin.(theta_refl),
    color = (:red3, 0.3), linewidth = 8)
mid_refl = rsb / 2
text!(ax, 1.35cos(mid_refl), 1.35sin(mid_refl),
    text = "incident +\nreflected", fontsize = 10, color = :red3, align = (:center, :center))

# Shadow region arc
theta_shad = range(isb + 0.1, 2π - 0.05, length = 30)
lines!(ax, 1.2 .* cos.(theta_shad), 1.2 .* sin.(theta_shad),
    color = (:gray50, 0.3), linewidth = 8)
mid_shad = (isb + 2π) / 2
text!(ax, 1.35cos(mid_shad), 1.35sin(mid_shad),
    text = "shadow", fontsize = 10, color = :gray50, align = (:center, :center))

# --- φ' angle arc ---
theta_arc = range(0, phip, length = 30)
lines!(ax, 0.25 .* cos.(theta_arc), 0.25 .* sin.(theta_arc), color = :royalblue, linewidth = 1)
text!(ax, 0.30cos(phip/2), 0.30sin(phip/2) - 0.02,
    text = "φ'", fontsize = 13, color = :royalblue)

xlims!(ax, -1.55, 1.55)
ylims!(ax, -1.55, 1.55)

fig # hide
```

### Wedge parameters

Two derived parameters appear throughout the KP formulation:

```math
n \equiv \frac{\alpha}{\pi}, \qquad \nu \equiv \frac{\pi}{\alpha} = \frac{1}{n}.
```

The parameter ``n`` gives the exterior angle in units of ``\pi``. Special cases:

| Geometry | ``\alpha`` | ``n`` | ``\nu`` |
|:---------|:-----------|:------|:--------|
| Half-plane | ``2\pi`` | 2 | 1/2 |
| Right-angle wedge (90° interior) | ``3\pi/2`` | 3/2 | 2/3 |
| 60° interior wedge | ``5\pi/3`` | 5/3 | 3/5 |
| Flat plane (no edge) | ``\pi`` | 1 | 1 |

```@example wedge_params
println("Half-plane:  n = $(wedge_n(w_hp)),  ν = $(wedge_nu(w_hp))")
println("Right-angle: n = $(wedge_n(w_ra)),  ν = $(wedge_nu(w_ra))")
```

### PEC boundary condition

On each wedge face, the tangential electric field vanishes:

```math
\hat{\mathbf{n}} \times \mathbf{E} = \mathbf{0} \qquad \text{on } \phi = 0 \text{ and } \phi = \alpha,
```

where ``\hat{\mathbf{n}}`` is the outward unit normal to the face. In the scalar picture:

- **Soft** (Dirichlet): ``u(\rho, 0) = 0`` and ``u(\rho, \alpha) = 0``.
- **Hard** (Neumann): ``\partial u / \partial \phi\big|_{\phi=0} = 0`` and ``\partial u / \partial \phi\big|_{\phi=\alpha} = 0``.

## Incident, reflected, and diffracted fields

Let a plane wave be incident at azimuth angle ``\phi' \in (0, \alpha)`` from the source direction. The incident field at the observation point ``(\rho, \phi)`` is

```math
u^i(\rho, \phi) = e^{+ik\rho\cos(\phi - \phi')}.
```

!!! note "Angle handling in the implementation"
    Public APIs accept any real `phi`/`phip`; internally, angles are wrapped into ``[0, \alpha)`` with `wrap_angle`.
    Near grazing incidence (``\phi' \approx 0`` or ``\phi' \approx \alpha``), the kernel collapses ``\phi'`` to zero while keeping ``\phi`` in the standard ``[0, \alpha)`` range. This avoids the branch-seam ambiguity at ``0 \leftrightarrow \alpha`` while preserving the ISB compensating discontinuity in ``\sin(\psi_2)``.

When the incident wave strikes a wedge face, a reflected wave is produced. The geometrical-optics (GO) field is

```math
u^{\text{GO}} = u^i + u^r,
```

where ``u^r`` exists only in the region illuminated by specular reflection.

The GO field has **step discontinuities** at two types of boundaries:

### Incident shadow boundary (ISB)

The edge blocks the direct ray for observation angles beyond

```math
\phi_{\text{ISB}} = \pi + \phi'.
```

- For ``\phi < \phi_{\text{ISB}}``: the observer is **lit** (receives the incident field).
- For ``\phi > \phi_{\text{ISB}}``: the observer is in **shadow** (no incident field).

### Reflection shadow boundary (RSB)

The specular reflection from the ``\phi = 0`` face is visible only for

```math
\phi < \phi_{\text{RSB}} = \pi - \phi'.
```

The reflected field from this face has the form ``R \cdot e^{+ik\rho\cos(\phi + \phi')}``, where the reflection coefficient is ``R = -1`` for soft polarisation and ``R = +1`` for hard polarisation (PEC boundary condition).

!!! note "General wedges"
    For a general wedge with ``\alpha < 2\pi``, there is also a reflection from the ``\phi = \alpha`` face. The KP four-term structure (covered in [Kouyoumjian--Pathak Coefficients](@ref kp)) automatically accounts for all shadow and reflection boundaries through the four cotangent arguments.

### Total field via UTD

The uniform theory of diffraction (UTD) supplies a **diffracted field** ``u^d`` that compensates for the GO discontinuities, yielding a continuous total field:

```math
u(\mathbf{r}) \approx u^{\text{GO}}(\mathbf{r}) + u^{d}(\mathbf{r}).
```

The key achievement of UTD (over the earlier geometrical theory of diffraction, GTD) is that this total field is continuous across every shadow and reflection boundary.

## Ray geometry

Let ``Q`` denote the diffraction point on the wedge edge and ``P`` the observation point. The relevant distances are:

- ``s = \|\mathbf{r}_P - \mathbf{r}_Q\|``: distance from the edge to the observer along the diffracted ray.
- ``s'``: distance from the source to the edge along the incident ray. For plane-wave incidence, ``s' \to \infty``.

The **effective distance parameter** that governs the width of the transition region is

```math
L \equiv \frac{s\,s'}{s + s'}.
```

This is the harmonic mean of ``s`` and ``s'`` (up to a factor of 2). For plane-wave incidence, ``L \to s`` as ``s' \to \infty``.

```@example wedge_params
# Effective distance examples
d1 = Distances(1.0, 1.0)
println("s=1, s'=1:   L = $(effective_L(d1))")   # 0.5

d2 = Distances(2.0, Inf)
println("s=2, s'=Inf: L = $(effective_L(d2))")    # 2.0 (plane wave)

d3 = Distances(1.0, 3.0)
println("s=1, s'=3:   L = $(effective_L(d3))")    # 0.75
```

**Physical interpretation**: ``L`` measures the effective distance from the edge in a way that accounts for the curvature of the incident wavefront. When the source is close to the edge (``s'`` small), the transition region is narrower than for a plane wave.

## UTD diffracted field

The UTD diffracted field is expressed as

```math
\mathbf{E}^d(P) = \overline{\mathbf{D}}_{\text{UTD}}(\phi, \phi'; k, L) \cdot \mathbf{E}^i(Q) \cdot A(s, s') \cdot e^{-iks},
```

where:
- ``\overline{\mathbf{D}}_{\text{UTD}}`` is the **diffraction dyadic**,
- ``A(s,s')`` is the **spreading factor** (accounts for geometrical divergence),
- ``e^{-iks}`` is the **outgoing phase factor** (``\exp(+i\omega t)`` convention: outgoing waves carry ``e^{-iks}``).

### Spreading factor

For a straight edge in free space, the spreading factor is

```math
A(s, s') \equiv \sqrt{\frac{s'}{s\,(s + s')}}.
```

**Derivation**: The diffracted field from a straight edge spreads cylindrically (like a line source) with amplitude ``\sim 1/\sqrt{s}``. The factor ``\sqrt{s'/(s+s')}`` accounts for the incident wavefront curvature. In the plane-wave limit:

```math
\lim_{s' \to \infty} A(s, s') = \lim_{s' \to \infty} \sqrt{\frac{s'}{s(s + s')}} = \lim_{s' \to \infty} \sqrt{\frac{1}{s(1 + s/s')}} = \frac{1}{\sqrt{s}}.
```

```@example wedge_params
println("A(1, Inf) = $(spreading_factor(1.0, Inf))")   # 1.0
println("A(4, Inf) = $(spreading_factor(4.0, Inf))")   # 0.5
println("A(1, 1)   = $(spreading_factor(1.0, 1.0))")   # √(1/2)
```

### Diffraction dyadic

In the ray-fixed transverse polarisation basis ``(\hat{\mathbf{u}}_s, \hat{\mathbf{u}}_h)``, the diffraction dyadic is diagonal:

```math
\overline{\mathbf{D}}_{\text{UTD}} = D_s\,\hat{\mathbf{u}}_s\hat{\mathbf{u}}_s + D_h\,\hat{\mathbf{u}}_h\hat{\mathbf{u}}_h,
```

where ``D_s`` and ``D_h`` are the **scalar diffraction coefficients** for soft (Dirichlet) and hard (Neumann) polarisations, respectively. The computation of these scalar coefficients is the core task of this library and is detailed in the next two chapters.

The function [`pec_wedge_apply_sh`](@ref) applies this dyadic to incident field components:

```@example wedge_params
w = Wedge(2π)
ang = RayAngles(π/2, π/4)
k = 10.0; L = 1.0
Ds, Dh = pec_wedge_DsDh(w, ang, k, L)

# Apply to unit incident fields
Es_i = 1.0 + 0.0im   # unit soft component
Eh_i = 1.0 + 0.0im   # unit hard component
s = 2.0; sp = Inf     # plane wave, observer at distance 2

Es_d, Eh_d = pec_wedge_apply_sh(Ds, Dh, Es_i, Eh_i, k, s, sp)
println("|Es_d| = $(round(abs(Es_d), digits=6))")
println("|Eh_d| = $(round(abs(Eh_d), digits=6))")
```
