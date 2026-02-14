# [Maxwell's Equations and the Helmholtz Equation](@id maxwell)

This chapter establishes the electromagnetic framework that underlies the uniform theory of diffraction. We fix the time-harmonic convention, derive the frequency-domain Maxwell equations, and reduce them to the scalar Helmholtz equation that governs two-dimensional wedge diffraction.

## Time-harmonic convention

Throughout this package and its documentation, all electromagnetic fields are represented as complex phasors with time dependence ``\exp(+i\omega t)``. A physical, real-valued field ``\mathcal{E}(\mathbf{r},t)`` is recovered from its phasor ``\mathbf{E}(\mathbf{r})`` by

```math
\mathcal{E}(\mathbf{r}, t) = \operatorname{Re}\!\bigl[\mathbf{E}(\mathbf{r})\,e^{+i\omega t}\bigr].
```

!!! note "Convention choice"
    The ``\exp(+i\omega t)`` convention is standard in electrical engineering and antenna theory (Harrington, Balanis). The physics literature often uses ``\exp(-i\omega t)``. The two conventions lead to complex-conjugate phasors and opposite signs in the phase factors. UTDKernels.jl currently supports only ``\exp(+i\omega t)``; the constant `EXP_IWT` encodes this choice.

Under this convention, the time derivative of a phasor quantity ``\mathbf{A}(\mathbf{r})\,e^{+i\omega t}`` produces a factor of ``+i\omega``:

```math
\frac{\partial}{\partial t}\bigl[\mathbf{A}\,e^{+i\omega t}\bigr] = +i\omega\,\mathbf{A}\,e^{+i\omega t}.
```

## Medium parameters

Consider a homogeneous, isotropic, lossless background medium with permittivity ``\epsilon`` and permeability ``\mu``. The **wavenumber** and **intrinsic impedance** are

```math
k \equiv \omega\sqrt{\mu\epsilon}, \qquad \eta \equiv \sqrt{\frac{\mu}{\epsilon}}.
```

For free space, ``\epsilon = \epsilon_0 \approx 8.854 \times 10^{-12}\,\text{F/m}`` and ``\mu = \mu_0 = 4\pi \times 10^{-7}\,\text{H/m}``, giving ``k = \omega/c`` with ``c = 1/\sqrt{\mu_0\epsilon_0}``.

## Frequency-domain Maxwell's equations

Starting from the time-domain Maxwell curl equations (source-free region):

```math
\nabla \times \mathcal{E} = -\frac{\partial \mathcal{B}}{\partial t}, \qquad
\nabla \times \mathcal{H} = \frac{\partial \mathcal{D}}{\partial t},
```

and substituting the phasor representation with ``\partial/\partial t \to +i\omega``, the constitutive relations ``\mathcal{B} = \mu\mathcal{H}`` and ``\mathcal{D} = \epsilon\mathcal{E}``, and cancelling the common ``e^{+i\omega t}`` factor, we obtain:

**Faraday's law:**

```math
\nabla \times \mathbf{E} = -i\omega\mu\,\mathbf{H}.
```

*Derivation*: ``\nabla \times \mathbf{E}\,e^{+i\omega t} = -\partial(\mu\mathbf{H}\,e^{+i\omega t})/\partial t = -\mu\mathbf{H}\cdot(+i\omega)\,e^{+i\omega t}``. Dividing both sides by ``e^{+i\omega t}`` gives the result.

**Ampère's law:**

```math
\nabla \times \mathbf{H} = +i\omega\epsilon\,\mathbf{E}.
```

*Derivation*: ``\nabla \times \mathbf{H}\,e^{+i\omega t} = \partial(\epsilon\mathbf{E}\,e^{+i\omega t})/\partial t = \epsilon\mathbf{E}\cdot(+i\omega)\,e^{+i\omega t}``. Dividing by ``e^{+i\omega t}`` gives the result. Note the **positive** sign on the right-hand side, which is characteristic of the ``\exp(+i\omega t)`` convention.

## Vector Helmholtz equation

Eliminating ``\mathbf{H}`` from the two curl equations yields the **vector Helmholtz equation**. Taking the curl of Faraday's law:

```math
\nabla \times (\nabla \times \mathbf{E}) = -i\omega\mu\,(\nabla \times \mathbf{H}).
```

Substituting Ampère's law for ``\nabla \times \mathbf{H}``:

```math
\nabla \times (\nabla \times \mathbf{E}) = -i\omega\mu\,(+i\omega\epsilon\,\mathbf{E}) = -i^2\omega^2\mu\epsilon\,\mathbf{E} = +\omega^2\mu\epsilon\,\mathbf{E} = k^2\,\mathbf{E}.
```

Rearranging:

```math
\nabla \times \nabla \times \mathbf{E} - k^2\,\mathbf{E} = \mathbf{0}, \qquad \mathbf{r} \in \mathbb{R}^3 \setminus \Gamma,
```

where ``\Gamma`` denotes the scattering boundary (the wedge surface).

## Scalar Helmholtz equation for 2D problems

For a wedge that is uniform along the ``z``-axis and with fields that do not vary in ``z``, the vector problem decouples into two independent scalar problems. In cylindrical coordinates ``(\rho, \phi, z)`` about the wedge edge:

- **Soft polarisation** (also called TM``_z`` or E-polarisation): The electric field is ``\mathbf{E} = E_z(\rho,\phi)\,\hat{z}``. The scalar field ``u = E_z`` satisfies the **Dirichlet** boundary condition ``u = 0`` on the PEC surface.

- **Hard polarisation** (also called TE``_z`` or H-polarisation): The magnetic field is ``\mathbf{H} = H_z(\rho,\phi)\,\hat{z}``. The scalar field ``u = H_z`` satisfies the **Neumann** boundary condition ``\partial u/\partial n = 0`` on the PEC surface.

### Why soft = Dirichlet and hard = Neumann

These boundary conditions follow directly from the PEC condition ``\hat{\mathbf{n}} \times \mathbf{E} = \mathbf{0}`` (tangential electric field vanishes on a perfect conductor). The two cases arise because the ``z``-directed field component has a different electromagnetic role in each polarisation.

**Soft (TM``_z``): ``E_z = 0`` on the surface (Dirichlet).**
When the electric field is purely ``z``-directed, ``\mathbf{E} = E_z\,\hat{z}``, the component ``E_z`` is tangential to any surface whose normal ``\hat{\mathbf{n}}`` lies in the ``(\rho, \phi)`` plane (which is the case for the wedge faces). The PEC boundary condition requires this tangential component to vanish:

```math
\hat{\mathbf{n}} \times \mathbf{E} = \mathbf{0} \quad\Longrightarrow\quad E_z\,(\hat{\mathbf{n}} \times \hat{z}) = \mathbf{0} \quad\Longrightarrow\quad E_z = 0 \;\text{ on the surface}.
```

Since ``\hat{\mathbf{n}} \times \hat{z} \neq \mathbf{0}`` (the surface normal is perpendicular to ``\hat{z}``), the only way to satisfy this is ``E_z = 0``. This is a **Dirichlet** condition on the scalar field ``u = E_z``.

The name "soft" comes from acoustics: a pressure-release (soft) boundary forces the pressure (the scalar field) to zero, exactly like the Dirichlet condition here.

**Hard (TE``_z``): ``\partial H_z / \partial n = 0`` on the surface (Neumann).**
When the magnetic field is purely ``z``-directed, ``\mathbf{H} = H_z\,\hat{z}``, the PEC condition ``\hat{\mathbf{n}} \times \mathbf{E} = \mathbf{0}`` does **not** directly constrain ``H_z``, because ``H_z`` is a magnetic field component. Instead, we must use Faraday's law to relate the electric field to ``H_z``.

From ``\nabla \times \mathbf{E} = -i\omega\mu\,\mathbf{H} = -i\omega\mu\,H_z\,\hat{z}``, the ``z``-component of the curl gives a constraint on the transverse electric field components ``E_\rho`` and ``E_\phi``. More directly, the transverse electric field components are obtained from ``\mathbf{H} = H_z\,\hat{z}`` via Ampère's law. In cylindrical coordinates, the relevant component is:

```math
E_\phi = \frac{i\omega\mu}{k^2}\,\frac{\partial H_z}{\partial\rho}, \qquad E_\rho = -\frac{i\omega\mu}{k^2}\,\frac{1}{\rho}\frac{\partial H_z}{\partial\phi}.
```

On the wedge face (say ``\phi = 0``), the outward normal is ``\hat{\mathbf{n}} = \hat{\phi}`` (or ``-\hat{\phi}``). The tangential electric field component in the ``z``-direction is zero (there is no ``E_z`` in this polarisation), but the tangential component along ``\hat{\rho}`` must also vanish:

```math
\hat{\mathbf{n}} \times \mathbf{E}\big|_{\text{surface}} = \mathbf{0} \quad\Longrightarrow\quad E_\rho\big|_{\phi=0} = 0 \quad\Longrightarrow\quad \frac{\partial H_z}{\partial\phi}\bigg|_{\phi=0} = 0.
```

Since ``\partial/\partial\phi`` is proportional to ``\partial/\partial n`` (the normal derivative) on the wedge face, this is a **Neumann** condition on the scalar field ``u = H_z``.

The name "hard" comes from acoustics: a rigid (hard) boundary forces the normal velocity (proportional to the normal derivative of pressure) to zero, matching the Neumann condition.

!!! note "Summary of the physical logic"
    - **Soft/TM``_z``**: ``\mathbf{E} \parallel \hat{z}`` → ``E_z`` is tangential to the wedge → PEC kills it → **Dirichlet** ``(u = 0)``.
    - **Hard/TE``_z``**: ``\mathbf{H} \parallel \hat{z}`` → electric field is transverse → PEC kills tangential ``\mathbf{E}`` → via Faraday/Ampère this constrains the normal derivative of ``H_z`` → **Neumann** ``(\partial u/\partial n = 0)``.

In both cases, the scalar field ``u(\rho,\phi)`` satisfies the **scalar Helmholtz equation**:

```math
\bigl(\nabla^2 + k^2\bigr)\,u = 0,
```

where ``\nabla^2`` is the two-dimensional Laplacian in cylindrical coordinates:

```math
\nabla^2 u = \frac{1}{\rho}\frac{\partial}{\partial\rho}\!\left(\rho\frac{\partial u}{\partial\rho}\right) + \frac{1}{\rho^2}\frac{\partial^2 u}{\partial\phi^2}.
```

## Radiation condition

The scattered (diffracted) field must satisfy the **Sommerfeld radiation condition** to ensure that energy propagates outward. For the ``\exp(+i\omega t)`` convention, outgoing cylindrical waves have the spatial dependence ``e^{-ik\rho}/\sqrt{\rho}`` as ``\rho \to \infty``. Formally:

```math
\lim_{\rho\to\infty} \sqrt{\rho}\left(\frac{\partial u^{\text{sc}}}{\partial\rho} + ik\,u^{\text{sc}}\right) = 0.
```

This selects outgoing waves and excludes incoming waves.

## Summary of sign conventions

| Quantity | ``\exp(+i\omega t)`` convention |
|:---------|:-------------------------------|
| Time dependence | ``e^{+i\omega t}`` |
| Outgoing wave | ``e^{-ikr}`` |
| Faraday's law | ``\nabla \times \mathbf{E} = -i\omega\mu\mathbf{H}`` |
| Ampère's law | ``\nabla \times \mathbf{H} = +i\omega\epsilon\mathbf{E}`` |
| Green's function (2D) | ``\sim e^{-ik\rho}/\sqrt{\rho}`` |

These conventions are locked in by `PhasorConvention(+1)` (aliased as `EXP_IWT`) and are assumed throughout the library.
