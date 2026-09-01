# [Reflection-Boundary Face--Edge Decomposition](@id reflection_boundary)

## Purpose

On a PEC reflection boundary, the KP coefficient contains two repeated
reflection terms and two incident terms. The repeated terms remain finite as
an exterior wedge becomes coplanar. They describe the face transition and
must not be mistaken for persistent structural edge strength.

`pec_wedge_face_edge` separates these contributions without modifying the
complete coefficient:

```math
D_h = D_e + D_f, \qquad D_s = D_e - D_f,
```

where ``D_e`` is returned as `edge` and ``D_f`` as `face`. The routine forms
the incident residual directly. Computing `(Dh + Ds)/2` from already assembled
coefficients performs another subtraction of quantities dominated by the face
term and can lose the small residual.

## Boundary geometry and domain

The caller supplies the exterior wedge and angular difference ``\Delta``. The
corresponding boundary angles are

```math
\phi=\frac{\alpha+\Delta}{2},\qquad
\phi'=\frac{\alpha-\Delta}{2},\qquad \phi+\phi'=\alpha.
```

The evaluator does not project arbitrary `RayAngles` onto this boundary. Its
verified domain is:

- ``\pi\leq\alpha\leq2\pi``;
- ``0\leq\Delta<\pi``;
- finite real ``k>0`` and ``L>0``; and
- either ``\Delta=0`` or ``\Delta>\alpha-\pi``.

The last inequality fixes the nearest-pole branch used by the compensated
incident pair. A request outside this cell raises `FaceEdgeDomainError`; use
the complete [`wedge_DsDh`](@ref) evaluator instead.

```@example reflection-boundary
using UTDKernels

epsilon = 0.01
wedge = Wedge((1 + epsilon) * pi)
delta = 0.5pi
split = pec_wedge_face_edge(wedge, delta, 20.0, 1.0, epsilon)

@assert split.Dh == split.edge + split.face
@assert split.Ds == split.edge - split.face
round(pec_wedge_intrinsic_score(wedge, delta, 20.0, 1.0, epsilon), digits=6)
```

The optional fifth positional argument carries
``\epsilon=\alpha/\pi-1`` separately. Use it when the defect is known before
`wedge.alpha` is rounded and is close enough to machine precision that it
cannot be reconstructed accurately from the stored angle. The method checks
that the supplied defect rounds back to the wedge angle. Ordinary geometries
use the four-argument form.

## Stable decomposition

Let ``n=\alpha/\pi`` and ``\epsilon=n-1``. For ``\Delta>0``, the two incident
terms are rearranged as

```math
S=(\cot\psi_1+\cot\psi_2)B(a_0)
  +\cot\psi_1\,[B(a_1)-B(a_0)],
\qquad B(a)=\frac{F(kLa)}{\sqrt{k}}.
```

Small angular and transition differences are evaluated before large terms are
combined. Near ``\Delta=\pi``, the implementation differences the smooth
regularized products as one quantity; it does not invoke an allocating
arbitrary-precision fallback. At ``\Delta=0``, it uses the symmetric closed
form. The face term uses its cancellation-free sinc ratio and satisfies

```math
\lim_{\alpha\to\pi^+}D_f=\sqrt{L},\qquad
\lim_{\alpha\to\pi^+}D_e=0.
```

Float32, Float64, and ForwardDiff inputs are supported on smooth branches.
Finite complex-BigFloat `erfcx` is not available from the package dependency,
so such requests fail explicitly with `ArgumentError`.

## Intrinsic score and limits of use

The local dimensionless score is

```math
\eta_e=\sqrt{2\pi k}\,|D_e|.
```

It can rank a geometrically selected PEC candidate edge when the diagnostic
geometry and distance convention are held fixed. It is not proof that deleting
an edge preserves the field of a complete object. Visibility, path birth and
death, phase accumulation, polarization transport, multiple interactions,
material faces, curved edges, and vertices remain outside this scalar local
diagnostic. A retained path must still use the complete coefficient.
