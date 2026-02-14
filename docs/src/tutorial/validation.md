# [Validation](@id validation)

This chapter validates UTDKernels.jl against the exact Sommerfeld half-plane solution, demonstrates GTD convergence, verifies reciprocity, and checks shadow-boundary continuity.

## The exact Sommerfeld half-plane solution

The PEC half-plane (``\alpha = 2\pi``, ``n = 2``) is the only wedge geometry that admits a closed-form exact solution. This makes it the ideal validation target.

### Pauli--Clemmow representation

Following Pauli (1938) and Clemmow (1966), the exact total field under unit-amplitude plane-wave illumination is built from the function

```math
V(\rho, \psi) = \frac{1}{2}\,e^{-ik\rho\cos\psi}\;\operatorname{erfc}\!\bigl(-\xi\,e^{-i\pi/4}\bigr),
```

where

```math
\xi = \sqrt{2k\rho}\,\cos\!\left(\frac{\psi}{2}\right).
```

The total field for each polarisation is:

```math
\begin{aligned}
u_s(\rho, \phi) &= V(\rho,\, \phi - \phi') - V(\rho,\, \phi + \phi') \qquad &\text{(soft / Dirichlet)}, \\[4pt]
u_h(\rho, \phi) &= V(\rho,\, \phi - \phi') + V(\rho,\, \phi + \phi') \qquad &\text{(hard / Neumann)}.
\end{aligned}
```

### Verification of boundary conditions

The function ``V(\rho, \psi)`` is **even** in ``\psi``:

```math
V(\rho, -\psi) = \frac{1}{2}\,e^{-ik\rho\cos(-\psi)}\;\operatorname{erfc}\!\bigl(-\sqrt{2k\rho}\cos(-\psi/2)\,e^{-i\pi/4}\bigr) = V(\rho, \psi),
```

since ``\cos`` is even.

**Dirichlet condition** (soft, ``u_s = 0`` at ``\phi = 0``):

```math
u_s(\rho, 0) = V(\rho, -\phi') - V(\rho, \phi') = V(\rho, \phi') - V(\rho, \phi') = 0. \quad\checkmark
```

**Neumann condition** (hard, ``\partial u_h/\partial\phi = 0`` at ``\phi = 0``):

Since ``V(\rho, \psi)`` is even in ``\psi``, its derivative ``V'(\rho, \psi) = \partial V/\partial\psi`` is **odd**: ``V'(\rho, -\psi) = -V'(\rho, \psi)``. Therefore:

```math
\frac{\partial u_h}{\partial\phi}\bigg|_{\phi=0} = V'(\rho, -\phi') + V'(\rho, \phi') = -V'(\rho, \phi') + V'(\rho, \phi') = 0. \quad\checkmark
```

### UTD total field for the half-plane

The UTD approximation to the total field is:

```math
u_{s/h}^{\text{UTD}}(\rho, \phi) = u_{s/h}^{\text{GO}}(\phi) + D_{s/h}(\phi, \phi'; k, \rho) \cdot \frac{e^{-ik\rho}}{\sqrt{\rho}},
```

where for the half-plane with plane-wave incidence, ``L = \rho`` and the spreading factor is ``A = 1/\sqrt{\rho}``.

The GO field for the half-plane with source at ``\phi' = \pi/4`` is:

```math
u_{s/h}^{\text{GO}}(\phi) = \begin{cases}
e^{-ik\rho\cos(\phi - \phi')} + R\,e^{-ik\rho\cos(\phi + \phi')} & \phi < \pi - \phi' \;\text{(both incident and reflected)}, \\[4pt]
e^{-ik\rho\cos(\phi - \phi')} & \pi - \phi' < \phi < \pi + \phi' \;\text{(incident only)}, \\[4pt]
0 & \phi > \pi + \phi' \;\text{(shadow)},
\end{cases}
```

with ``R = -1`` for soft and ``R = +1`` for hard.

### Numerical comparison

```@example validation
using UTDKernels
using SpecialFunctions: erfc

# Sommerfeld exact solution
function sommerfeld_V(k, rho, psi)
    xi = sqrt(2k * rho) * cos(psi / 2)
    z  = -xi * exp(-im * π/4)
    exp(-im * k * rho * cos(psi)) * erfc(z) / 2
end

function sommerfeld_exact(k, rho, phi, phip; pol=:soft)
    V1 = sommerfeld_V(k, rho, phi - phip)
    V2 = sommerfeld_V(k, rho, phi + phip)
    pol == :soft ? V1 - V2 : V1 + V2
end

# UTD total field
function utd_total(k, rho, phi, phip; pol=:soft)
    u_i    = exp(-im * k * rho * cos(phi - phip))
    u_r    = exp(-im * k * rho * cos(phi + phip))
    R      = pol == :soft ? -1 : +1
    isb    = π + phip
    rsb    = π - phip

    u_GO = zero(ComplexF64)
    phi < isb && (u_GO += u_i)
    phi < rsb && (u_GO += R * u_r)

    w   = Wedge(2π)
    Ds, Dh = pec_wedge_DsDh(w, RayAngles(phi, phip), k, rho)
    D   = pol == :soft ? Ds : Dh
    u_d = D * exp(-im * k * rho) / sqrt(rho)

    u_GO + u_d
end

# Compare at kρ = 50 (high-frequency regime)
k = 1.0; krho = 50.0; rho = krho / k; phip = π/4
println("Comparison at kρ = $krho:")
println("  φ (deg)  |exact|     |UTD|       rel. error (soft)")
for phi_deg in [30, 60, 90, 120, 200, 250, 300]
    phi = deg2rad(phi_deg)
    exact = sommerfeld_exact(k, rho, phi, phip; pol=:soft)
    utd   = utd_total(k, rho, phi, phip; pol=:soft)
    err   = abs(utd - exact) / max(abs(exact), 1e-30)
    println("  $(lpad(phi_deg, 3))°     $(round(abs(exact), digits=4))    $(round(abs(utd), digits=4))     $(round(err, sigdigits=2))")
end
```

The agreement is best seen in a plot. The following figure overlays the Sommerfeld exact solution and the UTD approximation for both polarisations across the full azimuthal range, including the shadow and reflection boundaries.

```@example validation
using CairoMakie

k = 1.0; krho = 50.0; rho = krho / k; phip = π/4
phis = range(0.01, 2π - 0.01, length = 600)
isb_deg = rad2deg(π + phip)
rsb_deg = rad2deg(π - phip)

fig = Figure(size = (700, 380))

for (col, pol, label) in [(1, :soft, "Soft (Dirichlet)"), (2, :hard, "Hard (Neumann)")]
    ax = Axis(fig[1, col], xlabel = "φ  [deg]", ylabel = "|u|",
        title = label, xticks = [0, 45, 90, 135, 180, 225, 270, 315, 360])

    exact_vals = [abs(sommerfeld_exact(k, rho, phi, phip; pol)) for phi in phis]
    utd_vals   = [abs(utd_total(k, rho, phi, phip; pol)) for phi in phis]

    lines!(ax, rad2deg.(phis), exact_vals, color = :black, linewidth = 2, label = "Sommerfeld")
    lines!(ax, rad2deg.(phis), utd_vals, color = :red3, linewidth = 1.5,
        linestyle = :dash, label = "UTD")

    vlines!(ax, [isb_deg], color = :gray50, linestyle = :dash, linewidth = 1)
    vlines!(ax, [rsb_deg], color = :gray50, linestyle = :dashdot, linewidth = 1)
    text!(ax, isb_deg + 3, 1.85, text = "ISB", fontsize = 10, color = :gray50)
    text!(ax, rsb_deg + 3, 1.85, text = "RSB", fontsize = 10, color = :gray50)

    ylims!(ax, 0, 2.1)
    col == 1 && axislegend(ax, position = :rt, framevisible = true, labelsize = 11)
end

fig # hide
```

## Shadow-boundary continuity

A key property of UTD is that the total field is **continuous** across shadow boundaries, even though the GO field has step discontinuities. The diffracted field has a compensating singularity that exactly cancels the GO jump.

```@example validation
# Sweep through the ISB at φ = 225° and verify finiteness
w = Wedge(2π); phip = π/4; k = 10.0; L = 1.0
isb = π + phip

println("Diffraction coefficients near the ISB (φ = $(round(rad2deg(isb), digits=1))°):")
for offset in [-5.0, -1.0, -0.1, -0.01, 0.0, 0.01, 0.1, 1.0, 5.0]
    phi = isb + deg2rad(offset)
    Ds, Dh = pec_wedge_DsDh(w, RayAngles(phi, phip), k, L)
    println("  ISB + $(lpad(offset, 6))°:  |Ds| = $(round(abs(Ds), digits=5)),  |Dh| = $(round(abs(Dh), digits=5)),  finite = $(isfinite(abs(Ds)))")
end
```

## GTD convergence

As ``kL \to \infty``, the UTD coefficient converges to the GTD coefficient at a rate ``O(1/kL)``. This follows from the asymptotic expansion of the transition function:

```math
F(x) = 1 - \frac{1}{2x} + O\!\left(\frac{1}{x^2}\right) \qquad (x \to +\infty).
```

Therefore:

```math
\left|\frac{D^{\text{UTD}} - D^{\text{GTD}}}{D^{\text{GTD}}}\right| = O\!\left(\frac{1}{kL}\right).
```

On a log-log plot, this gives a line with slope ``-1``.

```@example validation
# GTD convergence test
w = Wedge(2π)
n = wedge_n(w)
phi = π/2; phip = π/4
ang = RayAngles(phi, phip)

# GTD reference (F = 1)
phi_w  = wrap_angle(phi, 2π)
phip_w = wrap_angle(phip, 2π)
terms  = UTDKernels.kp_four_terms(phi_w, phip_w, n)

println("GTD convergence (half-plane, φ=90°, φ'=45°):")
println("  kL          relative error")
for p in 1:5
    kL = 10.0^p
    Ds, _ = pec_wedge_DsDh(w, ang, 1.0, kL)
    C = -exp(-im*π/4) / (2n * sqrt(2π))
    Ds_gtd = C * sum(UTDKernels.PEC_SIGMA_SOFT[j] * cot(terms.psi[j]) for j in 1:4)
    err = abs(Ds - Ds_gtd) / abs(Ds_gtd)
    println("  $(rpad(kL, 10))  $(round(err, sigdigits=3))")
end
```

## Reciprocity verification

The diffraction coefficient satisfies ``D_{s/h}(\phi, \phi') = D_{s/h}(\phi', \phi)``, as proved analytically in [KP Coefficients](@ref kp). Here we verify numerically:

```@example validation
println("Reciprocity verification (k=10, L=1):")
println("  Wedge        φ      φ'     |Ds(φ,φ')-Ds(φ',φ)|   |Dh(φ,φ')-Dh(φ',φ)|")
for (alpha, label) in [(2π, "half-plane"), (3π/2, "90° wedge"), (5π/4, "135° wedge")]
    local w = Wedge(alpha)
    for (phi, phip) in [(π/4, π/3), (π/6, π/2)]
        if phi >= alpha || phip >= alpha
            continue
        end
        Ds_f, Dh_f = pec_wedge_DsDh(w, RayAngles(phi, phip), 10.0, 1.0)
        Ds_r, Dh_r = pec_wedge_DsDh(w, RayAngles(phip, phi), 10.0, 1.0)
        println("  $(rpad(label, 12)) $(round(rad2deg(phi)))°  " *
                "$(round(rad2deg(phip)))°   $(abs(Ds_f - Ds_r))    $(abs(Dh_f - Dh_r))")
    end
end
```

## Flat-plane degenerate case

For ``\alpha = \pi`` (flat conducting plane, no edge), the diffraction coefficients should vanish identically. The four KP cotangent terms cancel analytically when ``n = 1``:

```@example validation
w_flat = Wedge(π)
Ds, Dh = pec_wedge_DsDh(w_flat, RayAngles(π/4, π/3), 10.0, 1.0)
println("Flat plane (α = π):")
println("  |Ds| = $(abs(Ds))")
println("  |Dh| = $(abs(Dh))")
println("  (Both should be ≈ 0, at machine epsilon level)")
```
