# [Maliuzhinets Exact Solution](@id maliuzhinets)

This chapter presents the exact impedance-wedge diffraction coefficients via the Maliuzhinets spectral function method. This serves as the validation reference for the Holm heuristic of the [previous chapter](@ref impedance).

## Why an exact solution?

The Holm heuristic is fast, differentiable, and numerically robust --- but it is an approximation. To quantify its error, we need a reference solution that is **exact** for the impedance boundary condition. The Maliuzhinets method provides this for the canonical case of normal plane-wave incidence on a wedge with uniform (but possibly different) surface impedances on each face.

The exact solution was first derived by Maliuzhinets (1958) and has been extensively studied by Griesser and Balanis (1989) and others. The implementation here follows the spectral function approach of Kotelnikov et al. (2013).

## The Maliuzhinets function

The Maliuzhinets function ``\psi_\Phi(w)`` is a special function of the complex variable ``w`` parametrised by the half exterior angle ``\Phi = \alpha/2``. It is defined by the integral

```math
\psi_\Phi(w) = \exp\!\left[-\frac{1}{2}\int_0^{\infty} \frac{\cosh(w\eta) - 1}{\eta\,\cosh(\pi\eta/2)\,\sinh(2\Phi\eta)}\,d\eta\right].
```

### Key properties

1. **Normalisation**: ``\psi_\Phi(0) = 1``.
2. **Even symmetry**: ``\psi_\Phi(-w) = \psi_\Phi(w)``.
3. **Functional relation**: ``\displaystyle\frac{\psi_\Phi(w + 2\Phi)}{\psi_\Phi(w - 2\Phi)} = \cot\!\left(\frac{w}{2} + \frac{\pi}{4}\right)``.
4. **Convergence strip**: The integral converges for ``|\operatorname{Re}(w)| < \pi/2 + 2\Phi``.

The functional relation (property 3) is the key to efficient evaluation: any argument outside the convergence strip can be reduced into it by repeated application of the relation, accumulating cotangent prefactors.

### Numerical evaluation

The implementation uses adaptive Gauss--Kronrod quadrature (via QuadGK.jl) within the convergence strip, with three numerical refinements:

1. **Removable singularity**: At ``\eta = 0``, L'Hôpital gives the integrand limit ``w^2/(4\Phi)``.
2. **Large-``\eta`` asymptotics**: For ``2\Phi\eta > 30``, a simplified exponential form avoids overflow in ``\sinh``.
3. **Strip extension**: For arguments outside the strip, the functional relation is applied backwards (subtracting ``4\Phi`` from the argument) until the argument lies within the strip.

```@example maliuzhinets
using UTDKernels

# Evaluate ψ_Φ at several points for a 270° wedge
Phi = 3π/4  # half of α = 1.5π

println("ψ_Φ(0) = $(psi_Phi(0.0, Phi))  (should be 1)")
println("ψ_Φ(0.5) = $(psi_Phi(0.5, Phi))")
println("ψ_Φ(1.0 + 2.0i) = $(psi_Phi(1.0 + 2.0im, Phi))")
```

```@example maliuzhinets
# Verify even symmetry
w = 0.5 + 1.0im
println("ψ_Φ(w)  = $(psi_Phi(w, Phi))")
println("ψ_Φ(-w) = $(psi_Phi(-w, Phi))")
println("Difference: $(abs(psi_Phi(w, Phi) - psi_Phi(-w, Phi)))")
```

```@example maliuzhinets
# Verify functional relation: ψ(w+2Φ)/ψ(w-2Φ) = cot(w/2 + π/4)
w = 0.3
lhs = psi_Phi(w + 2Phi, Phi) / psi_Phi(w - 2Phi, Phi)
rhs = cot(w/2 + π/4)
println("LHS = $lhs")
println("RHS = $rhs")
println("Relative error: $(abs(lhs - rhs) / abs(rhs))")
```

## The auxiliary Maliuzhinets function

The auxiliary function ``\Psi_0(z)`` is a product of four Maliuzhinets functions that encodes the impedance boundary conditions on both faces:

```math
\Psi_0(z) = \psi_\Phi(z + \Phi + \tfrac{\pi}{2} - \chi_+)\;\psi_\Phi(z + \Phi - \tfrac{\pi}{2} + \chi_+)\;\psi_\Phi(z - \Phi + \tfrac{\pi}{2} - \chi_-)\;\psi_\Phi(z - \Phi - \tfrac{\pi}{2} + \chi_-),
```

where ``\chi_+`` and ``\chi_-`` are the **impedance angles** on the ``+`` and ``-`` faces (at ``+\Phi`` and ``-\Phi`` in the centred coordinate system).

## Impedance angles

The impedance angles relate the surface impedance to the Maliuzhinets formulation. For a face with normalised impedance ``\eta = 1/\sqrt{\varepsilon_r}``:

| Polarisation | Impedance angle | PEC limit (``\varepsilon_r \to \infty``) |
|:---:|:---:|:---:|
| TE (hard, ``H_z``) | ``\sin\chi_{\text{TE}} = 1/\sqrt{\varepsilon_r}`` | ``\chi \to 0`` (Neumann) |
| TM (soft, ``E_z``) | ``\sin\chi_{\text{TM}} = \sqrt{\varepsilon_r}`` | ``\chi \to \pi/2 + i\infty`` (Dirichlet) |

These are equivalent to the Brewster angle definitions in Balanis Ch. 14 (Eqs. 14-6, 14-7) and Griesser (1988, Eq. 4.2-2).

## The spectral function approach

The diffraction coefficient is extracted from the spectral function ``s(u)`` via a saddle-point evaluation of the Sommerfeld integral. Following Kotelnikov et al. (2013):

```math
D = C(k) \bigl[\,s(\theta + \pi) - s(\theta - \pi)\,\bigr],
```

where ``\theta = \phi - \Phi`` is the centred observation angle, and:

```math
s(u) = \frac{\nu\cos(\nu\theta_0)}{\sin(\nu u) - \sin(\nu\theta_0)} \cdot \frac{\Psi_0(u)}{\Psi_0(\theta_0)},
```

with ``\nu = \pi/(2\Phi)`` and ``\theta_0 = \phi' - \Phi`` the centred source angle. The prefactor is the universal saddle-point constant:

```math
C(k) = \frac{-e^{-i\pi/4}}{\sqrt{2\pi k}}.
```

### Angle convention mapping

The Maliuzhinets formulation uses centred coordinates ``\theta \in (-\Phi, \Phi)`` with faces at ``\pm\Phi``, while UTDKernels uses ``\phi \in (0, \alpha)`` with faces at ``0`` and ``\alpha``. The mapping is:

```math
\theta = \phi - \frac{\alpha}{2}, \qquad \theta_0 = \phi' - \frac{\alpha}{2}.
```

The face assignment is: ``+\Phi`` face = n-face (``\phi = \alpha``), ``-\Phi`` face = o-face (``\phi = 0``).

### Practical domain constraints (implemented)

The current `maliuzhinets_DsDh` implementation enforces:

- ``\pi < \alpha < 2\pi`` (exterior wedge)
- ``k > 0``
- ``0 \le \phi \le \alpha`` and ``0 \le \phi' \le \alpha``

Inputs outside this domain raise a `DomainError`.

## Computing exact coefficients

```@example maliuzhinets
using UTDKernels

# 270° wedge with ε_r = 10 on both faces
alpha = 1.5π
eps_r = 10.0
k = 2π
phi = 0.4 * alpha
phip = 0.3 * alpha

Ds, Dh = maliuzhinets_DsDh(alpha, eps_r, eps_r, phi, phip, k)
println("Exact (Maliuzhinets):")
println("  |Ds| = $(round(abs(Ds), sigdigits=5))")
println("  |Dh| = $(round(abs(Dh), sigdigits=5))")
```

```@example maliuzhinets
# Compare with the Holm heuristic
mat = WedgeFaceMaterial(Complex(eps_r))
iw = ImpedanceWedge(alpha, mat)
ang = RayAngles(phi, phip)
L = 1.0  # effective distance (Maliuzhinets is distance-independent at this level)

Ds_holm, Dh_holm = impedance_wedge_DsDh(iw, ang, k, L)
println("Holm heuristic (L=$L):")
println("  |Ds| = $(round(abs(Ds_holm), sigdigits=5))")
println("  |Dh| = $(round(abs(Dh_holm), sigdigits=5))")
```

Note that the Maliuzhinets solution gives the bare diffraction coefficient without the transition function ``F(X)``, while the Holm formula includes it. For comparison purposes (away from shadow boundaries where ``F \approx 1``), the magnitudes are directly comparable.

## Validation: PEC recovery

As ``\varepsilon_r \to \infty``, the impedance angles approach their PEC limits and the Maliuzhinets coefficients must match the PEC UTD.

```@example maliuzhinets
alpha = 1.5π; k = 2π; phi = 0.4alpha; phip = 0.3alpha

# PEC reference
w = Wedge(alpha)
_, Dh_pec = pec_wedge_DsDh(w, RayAngles(phi, phip), k, Inf)

# Maliuzhinets at very large ε_r
_, Dh_mal = maliuzhinets_DsDh(alpha, 1e26, 1e26, phi, phip, k)

println("PEC recovery test:")
println("  |D_h| PEC (KP):        $(abs(Dh_pec))")
println("  |D_h| Maliuzhinets:    $(abs(Dh_mal))")
println("  Relative error:        $(abs(abs(Dh_mal) - abs(Dh_pec)) / abs(Dh_pec))")
```

The ``D_h`` coefficient converges to PEC as ``O(1/\sqrt{|\varepsilon_r|})`` --- a property of the impedance angle ``\chi_{\text{TE}} = \sin^{-1}(1/\sqrt{\varepsilon_r})``.

## Validation: reciprocity

The exact solution satisfies reciprocity: ``D(\phi, \phi') = D(\phi', \phi)`` when both faces have the same impedance.

```@example maliuzhinets
alpha = 1.5π; k = 2π
phi1 = 0.3alpha; phi2 = 0.6alpha

for eps_r in [4.0, 25.0, 100.0]
    Ds_fwd, Dh_fwd = maliuzhinets_DsDh(alpha, eps_r, eps_r, phi1, phi2, k)
    Ds_rev, Dh_rev = maliuzhinets_DsDh(alpha, eps_r, eps_r, phi2, phi1, k)
    println("ε_r = $eps_r:")
    println("  |D_h(φ₁,φ₂) - D_h(φ₂,φ₁)| = $(abs(Dh_fwd - Dh_rev))")
end
```

## Asymmetric face impedances

The Maliuzhinets solution supports different permittivities on each face:

```@example maliuzhinets
alpha = 1.5π; k = 2π; phi = 0.4alpha; phip = 0.3alpha

# Symmetric
Ds_sym, Dh_sym = maliuzhinets_DsDh(alpha, 10.0, 10.0, phi, phip, k)

# Asymmetric
Ds_asym, Dh_asym = maliuzhinets_DsDh(alpha, 4.0, 25.0, phi, phip, k)

println("Symmetric  (ε_o = ε_n = 10): |D_h| = $(round(abs(Dh_sym), sigdigits=5))")
println("Asymmetric (ε_o=4, ε_n=25):  |D_h| = $(round(abs(Dh_asym), sigdigits=5))")
```

For asymmetric faces, the wedge flip symmetry holds: ``D(\phi, \phi'; \varepsilon_o, \varepsilon_n) = D(\alpha - \phi, \alpha - \phi'; \varepsilon_n, \varepsilon_o)``.

```@example maliuzhinets
# Verify wedge flip symmetry
Ds_flip, Dh_flip = maliuzhinets_DsDh(alpha, 25.0, 4.0, alpha - phi, alpha - phip, k)
println("Wedge flip symmetry:")
println("  D_h original = $Dh_asym")
println("  D_h flipped  = $Dh_flip")
println("  Match: $(isapprox(Dh_asym, Dh_flip, rtol=1e-8))")
```

## Holm heuristic error vs permittivity

The Holm heuristic error decreases monotonically with ``|\varepsilon_r|``. Typical errors at a representative observation angle for a 270° exterior wedge:

```@example maliuzhinets
alpha = 1.5π; k = 2π; phi = 0.6alpha; phip = 0.3alpha

println("Holm heuristic error vs exact Maliuzhinets:")
println("  ε_r      |D_h| exact    |D_h| Holm (Keller)    rel. error")
for eps_r in [4.0, 10.0, 25.0, 100.0, 500.0]
    # Exact
    _, Dh_exact = maliuzhinets_DsDh(alpha, eps_r, eps_r, phi, phip, k)

    # Holm (Keller-type, no transition function → set L = Inf for F → 1)
    mat = WedgeFaceMaterial(Complex(eps_r))
    iw = ImpedanceWedge(alpha, mat)
    _, Dh_holm = impedance_wedge_DsDh(iw, RayAngles(phi, phip), k, Inf)

    err = abs(abs(Dh_exact) - abs(Dh_holm)) / abs(Dh_exact)
    println("  $(rpad(eps_r, 8))  $(round(abs(Dh_exact), sigdigits=5))       $(round(abs(Dh_holm), sigdigits=5))              $(round(100err, sigdigits=2))%")
end
```

## Performance considerations

The Maliuzhinets solution involves adaptive quadrature (24 evaluations of ``\psi_\Phi`` per call to `maliuzhinets_DsDh` in the current implementation: 12 per polarization, each requiring numerical integration). Typical timings:

- `impedance_wedge_DsDh`: microseconds (closed-form Fresnel + ``\operatorname{erfcx}``)
- `maliuzhinets_DsDh`: milliseconds (adaptive quadrature)

The exact solution is intended as a **validation reference**, not for production ray-tracing. For high-performance applications, use `impedance_wedge_DsDh`.
