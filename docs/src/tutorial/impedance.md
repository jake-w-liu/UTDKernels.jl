# [Impedance Wedge Diffraction](@id impedance)

This chapter extends the PEC diffraction theory of the preceding chapters to wedges with **impedance boundary conditions**. The four-term Kouyoumjian--Pathak (KP) structure is retained, with face-specific Fresnel reflection coefficients on the reflection terms and Fresnel-product weights on the incident terms. This is the **Holm (2000) heuristic**, which provides a practical, ForwardDiff-compatible impedance-wedge diffraction coefficient.

## Physical motivation

Real-world diffracting edges --- building corners, terrain ridges, vehicle panels --- are not perfectly conducting. Their surfaces have finite permittivity and possibly conductivity, which affects the reflected fields. The PEC UTD coefficients assume ``R_s = -1`` and ``R_h = +1`` for the reflection terms, but for an imperfect conductor these become angle- and material-dependent Fresnel coefficients.

The challenge is to incorporate these material effects into the UTD framework without losing the numerical robustness (overflow-free ``\operatorname{erfcx}`` form, regularised ``\cot\!\cdot\!F``, branch safety) established for the PEC case.

## The impedance boundary condition

The Leontovich (impedance) boundary condition relates the tangential electric and magnetic fields at the surface:

```math
\hat{\mathbf{n}} \times \mathbf{E} = Z_s\,(\hat{\mathbf{n}} \times \hat{\mathbf{n}} \times \mathbf{H}),
```

where ``Z_s = \eta Z_0`` is the surface impedance, ``\eta = Z_s/Z_0`` is the normalised impedance, and ``\hat{\mathbf{n}}`` is the outward surface normal. For a homogeneous dielectric half-space with relative permittivity ``\varepsilon_r`` (and ``\mu_r = 1``):

```math
\eta = \frac{1}{\sqrt{\varepsilon_r}}.
```

This approximation is valid when the material is electrically thick (``|\varepsilon_r| \gg 1``) or when the radius of curvature is large compared to the wavelength.

## Fresnel reflection coefficients

For a plane wave at **grazing angle** ``\psi`` (measured from the surface, not the normal; ``\psi = \pi/2 - \theta_i``), the Fresnel reflection coefficients are:

```math
\begin{aligned}
R_{\text{TE}}(\psi, \varepsilon_r) &= \frac{\sin\psi - \sqrt{\varepsilon_r - \cos^2\psi}}{\sin\psi + \sqrt{\varepsilon_r - \cos^2\psi}}, \\[6pt]
R_{\text{TM}}(\psi, \varepsilon_r) &= \frac{\varepsilon_r\sin\psi - \sqrt{\varepsilon_r - \cos^2\psi}}{\varepsilon_r\sin\psi + \sqrt{\varepsilon_r - \cos^2\psi}}.
\end{aligned}
```

### PEC and PMC limits

As ``|\varepsilon_r| \to \infty`` (PEC):

```math
R_{\text{TE}} \to -1, \qquad R_{\text{TM}} \to +1.
```

These are exactly the PEC sign factors ``\sigma_s = (-1, -1)`` and ``\sigma_h = (+1, +1)`` on the reflection terms, confirming that the impedance formulation reduces to PEC.

As ``\varepsilon_r \to 1`` (free space, no interface):

```math
R_{\text{TE}} \to 0, \qquad R_{\text{TM}} \to 0.
```

No reflection, as expected for a matched boundary.

### Brewster angle

Setting ``R_{\text{TM}} = 0`` gives the Brewster grazing angle
``\psi_B = \sin^{-1}(1/\sqrt{1+\varepsilon_r})`` (equivalently,
``\psi_B = \tan^{-1}(1/\sqrt{\varepsilon_r})``). At this angle, the TM
(hard) reflection vanishes --- a physical effect that the impedance UTD
correctly captures.

This differs from the impedance-BC approximation in Balanis Ch. 14
(``\phi_B = \sin^{-1}(\eta)`` for hard polarization), because the formulas
above use the exact dielectric Fresnel coefficient in terms of
``\varepsilon_r``. For high-contrast media (small ``\eta \approx
1/\sqrt{\varepsilon_r}``), both expressions become close.

### Implementation

The package provides `fresnel_te` and `fresnel_tm`:

```@example impedance
using UTDKernels

# Concrete example: ε_r = 4 (lossless dielectric)
psi = π/4  # 45° grazing angle
eps_r = 4.0

R_te = fresnel_te(psi, eps_r)
R_tm = fresnel_tm(psi, eps_r)

println("R_TE(ψ=45°, ε_r=4) = $(round(R_te, sigdigits=5))")
println("R_TM(ψ=45°, ε_r=4) = $(round(R_tm, sigdigits=5))")
```

```@example impedance
# PEC limit
println("PEC limit (ε_r = 10²⁶):")
println("  R_TE = $(fresnel_te(π/4, 1e26))")
println("  R_TM = $(fresnel_tm(π/4, 1e26))")
```

```@example impedance
# Complex permittivity (lossy material)
eps_r_lossy = 4.0 - 0.5im  # exp(+iωt) convention: loss → negative imaginary
R_te_lossy = fresnel_te(π/4, eps_r_lossy)
R_tm_lossy = fresnel_tm(π/4, eps_r_lossy)
println("Lossy (ε_r = 4 - 0.5i):")
println("  R_TE = $(round(R_te_lossy, sigdigits=5))")
println("  R_TM = $(round(R_tm_lossy, sigdigits=5))")
```

## Material specification

Face materials are described by the `WedgeFaceMaterial` type, which stores the effective complex relative permittivity. Two constructors are available:

```@example impedance
# From complex permittivity directly
mat1 = WedgeFaceMaterial(4.0 + 0.0im)

# From real permittivity + conductivity + frequency
# ε_r_eff = ε_r - i σ/(ω ε₀)  [exp(+iωt) convention]
mat2 = WedgeFaceMaterial(4.0, 0.01, 1e9)  # ε_r=4, σ=0.01 S/m, f=1 GHz
println("mat2.eps_r = $(mat2.eps_r)")
```

An `ImpedanceWedge` combines the exterior angle with materials on each face:

```@example impedance
# Same material on both faces
iw_sym = ImpedanceWedge(1.5π, WedgeFaceMaterial(4.0 + 0.0im))

# Different materials on each face
face_o = WedgeFaceMaterial(4.0 + 0.0im)   # o-face (φ = 0)
face_n = WedgeFaceMaterial(25.0 + 0.0im)  # n-face (φ = α)
iw_asym = ImpedanceWedge(1.5π, face_o, face_n)
```

## The Holm (2000) heuristic

The Holm heuristic weights both the incident and reflection terms with Fresnel coefficients evaluated at face-specific angles that depend on the incident and diffracted rays. The structure is:

```math
\begin{aligned}
D_s &= G C(k,n) \bigl[\, W_{\text{TE},n}\,c_1 + W_{\text{TE},0}\,c_2 + R_{\text{TE},n}\,c_3 + R_{\text{TE},0}\,c_4 \,\bigr], \\[4pt]
D_h &= G C(k,n) \bigl[\, W_{\text{TM},n}\,c_1 + W_{\text{TM},0}\,c_2 + R_{\text{TM},n}\,c_3 + R_{\text{TM},0}\,c_4 \,\bigr],
\end{aligned}
```

where:
- ``c_j = \cot(\psi_j) \cdot F(k L a_j)`` are the regularised KP terms (identical to PEC),
- ``C(k,n) = -e^{-i\pi/4}/(2n\sqrt{2\pi k})`` is the universal prefactor,
- ``R_{\text{TE/TM},0}`` and ``R_{\text{TE/TM},n}`` are Fresnel coefficients at the 0-face and n-face,
- Holm's grazing angles are ``\theta_o=\min(\phi',\phi)`` and ``\theta_n=\min(\alpha-\phi',\alpha-\phi)``.
- For either polarization, the incident weights are fixed throughout the wedge: ``W_n=R_0R_n`` and ``W_0=1``.
- ``G=1/2`` at exact face-grazing incidence and ``G=1`` otherwise.

### Why this works

Term 3 accounts for diffraction of the field reflected from the n-face, and term 4 accounts for the corresponding contribution from the 0-face. Holm's product weights couple these face reflections to the two incident-shadow terms. In the PEC limit away from exact grazing incidence, the product weights approach unity and the four-term expression recovers the PEC coefficient.

### Key properties

1. **PEC reduction**: At fixed non-grazing incidence, the Holm coefficient approaches the PEC UTD as ``|\varepsilon_r| \to \infty``.
2. **Shadow-boundary continuity**: The regularised ``\cot\!\cdot\!F`` product ensures finite values at shadow boundaries, independent of the face material.
3. **ForwardDiff compatibility**: The coefficient supports ForwardDiff at smooth angular points, enabling gradient-based optimisation away from branch seams.
4. **Heuristic nature**: The formula is not derived from the exact impedance-wedge solution. It is an engineering approximation whose accuracy improves as ``|\varepsilon_r|`` increases.

## Computing impedance-wedge coefficients

### Basic usage

```@example impedance
using UTDKernels

# 270° wedge with ε_r = 10 on both faces
mat = WedgeFaceMaterial(10.0 + 0.0im)
iw = ImpedanceWedge(1.5π, mat)
ang = RayAngles(π/2, π/4)
k = 10.0; L = 1.0

Ds, Dh = impedance_wedge_DsDh(iw, ang, k, L)
println("|Ds| = $(round(abs(Ds), sigdigits=5))")
println("|Dh| = $(round(abs(Dh), sigdigits=5))")
```

### Comparison with PEC

```@example impedance
# PEC coefficients for the same geometry
w = Wedge(1.5π)
Ds_pec, Dh_pec = pec_wedge_DsDh(w, ang, k, L)

println("PEC:       |Ds| = $(round(abs(Ds_pec), sigdigits=5)),  |Dh| = $(round(abs(Dh_pec), sigdigits=5))")
println("ε_r = 10:  |Ds| = $(round(abs(Ds), sigdigits=5)),  |Dh| = $(round(abs(Dh), sigdigits=5))")
```

### PEC convergence

```@example impedance
# Show D_h converging to PEC as ε_r increases
Ds_pec, Dh_pec = pec_wedge_DsDh(w, ang, k, L)

println("PEC convergence of |D_h|:")
println("  ε_r        |D_h|       rel. error vs PEC")
for eps_r in [4.0, 10.0, 100.0, 1e4, 1e8]
    mat_test = WedgeFaceMaterial(Complex(eps_r))
    iw_test = ImpedanceWedge(1.5π, mat_test)
    _, Dh_test = impedance_wedge_DsDh(iw_test, ang, k, L)
    err = abs(abs(Dh_test) - abs(Dh_pec)) / abs(Dh_pec)
    println("  $(rpad(eps_r, 10))  $(round(abs(Dh_test), sigdigits=5))    $(round(err, sigdigits=2))")
end
```

### Asymmetric faces

```@example impedance
# Different materials on each face
face_o = WedgeFaceMaterial(4.0 + 0.0im)   # concrete-like
face_n = WedgeFaceMaterial(25.0 + 0.0im)  # high-permittivity

iw_asym = ImpedanceWedge(1.5π, face_o, face_n)
Ds_asym, Dh_asym = impedance_wedge_DsDh(iw_asym, ang, k, L)
println("Asymmetric (ε_o=4, ε_n=25): |Ds| = $(round(abs(Ds_asym), sigdigits=5)),  |Dh| = $(round(abs(Dh_asym), sigdigits=5))")
```

### Separate transition distances

For curved wedge faces or non-plane-wave incidence, the transition distances for the incident and reflected terms may differ:

```@example impedance
# Three-distance API: Li (incident), Lro (o-face reflection), Lrn (n-face reflection)
Li = 2.0; Lro = 1.5; Lrn = 1.0

Ds_3L, Dh_3L = impedance_wedge_DsDh(iw, ang, k, Li, Lro, Lrn)
println("Three-distance: |Ds| = $(round(abs(Ds_3L), sigdigits=5)),  |Dh| = $(round(abs(Dh_3L), sigdigits=5))")
```

The single-distance API also supports finite nonzero complex ``L`` for
analytic-continuation use cases (for example, complex-source-beam
formulations). In standard geometric optics interpretation, ``L`` is real and
positive (or ``L=\infty`` for far-field ``F\to 1`` studies).

## AD through impedance wedge coefficients

The entire impedance-wedge pipeline is ForwardDiff-compatible:

```@example impedance
using ForwardDiff

# Gradient of |D_h| with respect to observation angle
f(phi) = abs(impedance_wedge_DsDh(iw, RayAngles(phi, π/4), k, L)[2])

phi_test = π/2
grad_ad = ForwardDiff.derivative(f, phi_test)

h = 1e-7
grad_fd = (f(phi_test + h) - f(phi_test - h)) / (2h)

println("∂|D_h|/∂φ (AD): $(round(grad_ad, sigdigits=6))")
println("∂|D_h|/∂φ (FD): $(round(grad_fd, sigdigits=6))")
println("Relative error:  $(round(abs(grad_ad - grad_fd) / abs(grad_ad), sigdigits=2))")
```

## Limitations of the Holm heuristic

The Holm formulation is a **heuristic** --- it is not derived from the exact impedance-wedge solution. Known limitations include:

1. **Accuracy degrades at low ``\varepsilon_r``**: For materials close to free space (``\varepsilon_r \lesssim 4``), the heuristic error can exceed 10% compared to the exact Maliuzhinets solution. The error decreases monotonically with increasing ``|\varepsilon_r|``.

2. **No surface wave terms**: The exact impedance-wedge solution includes surface wave contributions that are absent from the Holm heuristic. These can be significant for reactive (highly lossy) surfaces.

3. **Not reciprocal**: Holm's original fixed incident weights do not in general preserve the coefficient when source and observation directions are exchanged. Later reciprocal modifications use different angle and weight prescriptions and are distinct models.

4. **Normal incidence only**: Like the PEC UTD, the 2D formulation assumes normal incidence on the edge. Extension to oblique incidence requires additional considerations.

For validation against the exact solution, see [Maliuzhinets Exact Solution](@ref maliuzhinets).
