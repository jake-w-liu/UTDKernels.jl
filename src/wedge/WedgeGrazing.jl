"""
Face-grazing continuation of the standard PEC four-term coefficient.

`pec_wedge_DsDh` is unchanged. This file adds a certified alternative path
that replaces the soft pairing G(φ−h)−G(φ+h) by an integral of G'.
"""

"""
Raised when a requested grazing interval leaves the certified branch-local domain.
"""
struct GrazingDomainError <: Exception
    msg::String
end

Base.showerror(io::IO, e::GrazingDomainError) = print(io, "GrazingDomainError: ", e.msg)

"""
Certificate and diagnostics for one face-grazing continuation request.
"""
struct GrazingIntervalReport
    valid::Bool
    face::Symbol
    h::Float64
    min_abs_sin_psi::Float64
    min_transition_argument::Float64
    signatures::Tuple{Tuple{Int,Int},Tuple{Int,Int}}
    min_branch_distance::Float64
    gprime_abs::Float64
    degenerate_odd::Bool
    reason::String
end

const _GL_CACHE = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()

# Report fields are Float64 diagnostics. Dual/other numbers contribute only
# their primal value so AD can pass through the continuation itself.
_primal_float(x::Integer) = Float64(x)
_primal_float(x::AbstractFloat) = Float64(x)
function _primal_float(x)
    hasproperty(x, :value) && return _primal_float(getproperty(x, :value))
    return Float64(real(x))
end

function gauss_legendre_nodes(order::Int)
    order >= 1 || throw(ArgumentError("Gauss–Legendre order must be positive"))
    got = get(_GL_CACHE, order, nothing)
    got !== nothing && return got
    if order == 1
        nodes, weights = ([0.0], [2.0])
    else
        β = [k / sqrt(4k^2 - 1) for k in 1:(order - 1)]
        T = LinearAlgebra.SymTridiagonal(zeros(order), β)
        F = LinearAlgebra.eigen(T)
        nodes = F.values
        weights = 2 .* abs2.(F.vectors[1, :])
    end
    _GL_CACHE[order] = (nodes, weights)
    return nodes, weights
end

function _phi_in_wedge(wedge::Wedge, ang::RayAngles)
    alpha = wedge.alpha
    phi = wrap_angle(ang.phi, alpha)
    if phi <= DEFAULT_TRANSITION_TOL && _primal_iszero(ang.phi - alpha)
        phi = phi + alpha
    end
    return phi
end

function _phip_from_faces(wedge::Wedge, ang::RayAngles)
    alpha = wedge.alpha
    phip = wrap_angle(ang.phip, alpha)
    # wrap_angle(α, α) = 0 aliases exact n-face incidence onto the o-face.
    # Recover the n-face offset from the unwrapped input, matching
    # `_effective_angles_for_kp`.
    near_n_raw = _primal_iszero(ang.phip - alpha)
    if phip <= DEFAULT_TRANSITION_TOL && near_n_raw
        return (alpha, zero(phip))
    end
    return (phip, alpha - phip)
end

"""
    grazing_local_angles(wedge, ang; face=:auto)

Map a request onto an o-face-measured pair `(φ_loc, h)` and the grazed face.
PEC n-face incidence is mirrored to the o-face of the same wedge.
"""
function grazing_local_angles(wedge::Wedge, ang::RayAngles; face::Symbol=:auto)
    face in (:auto, :o, :n) || throw(ArgumentError("face must be :auto, :o, or :n"))
    alpha = wedge.alpha
    phi = _phi_in_wedge(wedge, ang)
    h_o, h_n = _phip_from_faces(wedge, ang)
    chosen = face === :auto ? (h_n < h_o ? :n : :o) : face
    if chosen === :n
        return (face=:n, phi=alpha - phi, h=h_n)
    end
    return (face=:o, phi=phi, h=h_o)
end

function _term_N(beta::Real, sigma::Int, n::Real)
    return kp_Nj(beta, sigma, n)
end

function _branch_signature(beta::Real, n::Real)
    return (_term_N(beta, +1, n), _term_N(beta, -1, n))
end

"""
    two_term_kernel(beta, wedge, k, L)

G(β) = Σ_σ cot(ψ_σ) F(k L a_σ) on the current nearest-integer branch.
"""
function two_term_kernel(beta::Real, wedge::Wedge, k::Number, L::Number)
    n = wedge_n(wedge)
    function one_term(sigma::Int)
        psi = cotangent_arg(beta, sigma, n)
        N = kp_Nj(beta, sigma, n)
        a = kp_aj(beta, N, n)
        x = k * L * a
        sψ = sin(psi)
        if !(real(x) > 0) || sψ == 0
            throw(GrazingDomainError(
                "the branch-local kernel is at a separate UTD transition; use pec_wedge_DsDh",
            ))
        end
        return (cos(psi) / sψ) * F_utd(x)
    end
    return one_term(+1) + one_term(-1)
end

"""
    two_term_kernel_derivative(beta, wedge, k, L)

dG/dβ on a fixed nearest-integer branch. Uses u' = −1 and a' = sin(u).
The certified derivative requires real `k` and `L`; use `pec_wedge_DsDh` for
complex media.
"""
function two_term_kernel_derivative(beta::Real, wedge::Wedge, k::Number, L::Number)
    if !(k isa Real && L isa Real)
        throw(GrazingDomainError(
            "kernel derivative requires real k and L; use pec_wedge_DsDh for complex media",
        ))
    end
    _validate_wavenumber(k)
    _validate_effective_L(L)

    n = wedge_n(wedge)
    function one_term(sigma::Int)
        psi = cotangent_arg(beta, sigma, n)
        N = kp_Nj(beta, sigma, n)
        a = kp_aj(beta, N, n)
        u = 2 * n * π * N - beta
        x = k * L * a
        sψ = sin(psi)
        if !(real(x) > 0) || sψ == 0
            throw(GrazingDomainError(
                "kernel derivative is singular/non-smooth at a separate UTD transition",
            ))
        end
        dψ = sigma / (2 * n)
        if isinf(real(x))
            # Far-field limit L → ∞: F(x) → 1 and cot(ψ)·F'(x)·(kL)·a' → 0
            # because F'(x) ~ O(x^{-2}), so F'(kLa)·kL ~ 1/(kL·a²) → 0. Only the
            # −ψ'/sin²ψ term survives, giving the exact plane-wave dG_∞/dβ.
            return complex(-dψ / (sψ * sψ))
        end
        F = F_utd(x)
        Fp = F_utd_prime(x)
        cotψ = cos(psi) / sψ
        da = sin(u)
        return -dψ * F / (sψ * sψ) + cotψ * Fp * (k * L) * da
    end
    return one_term(+1) + one_term(-1)
end

function _contains_lattice_point(lo::Real, hi::Real, offset::Real, period::Real)
    if lo > hi
        lo, hi = hi, lo
    end
    m_lo = ceil((lo - offset) / period)
    m_hi = floor((hi - offset) / period)
    return m_lo <= m_hi
end

function _distance_to_half_integer(q::Real)
    return abs((q - floor(q)) - 0.5)
end

function _minimum_term_margins(beta_lo, beta_hi, sigma::Int, n::Real, N::Int, kL)
    T = typeof(float(real(n)))
    πT = T(π)
    psi_lo = (πT + sigma * beta_lo) / (2n)
    psi_hi = (πT + sigma * beta_hi) / (2n)
    min_sin = if _contains_lattice_point(psi_lo, psi_hi, zero(T), πT)
        zero(T)
    else
        min(abs(sin(psi_lo)), abs(sin(psi_hi)))
    end

    u_lo = 2n * πT * N - beta_hi
    u_hi = 2n * πT * N - beta_lo
    min_a = if _contains_lattice_point(u_lo, u_hi, πT, 2πT)
        zero(T)
    else
        min(2 * cos(u_lo / 2)^2, 2 * cos(u_hi / 2)^2)
    end

    scale = 2n * πT
    q_lo = (beta_lo + sigma * πT) / scale
    q_hi = (beta_hi + sigma * πT) / scale
    branch_distance = if _contains_lattice_point(q_lo, q_hi, T(0.5), one(T))
        zero(T)
    else
        scale * min(_distance_to_half_integer(q_lo), _distance_to_half_integer(q_hi))
    end
    return min_sin, kL * min_a, branch_distance
end

function _empty_report(face::Symbol, h, reason::String)
    return GrazingIntervalReport(
        false, face, _primal_float(h), 0.0, 0.0, ((0, 0), (0, 0)), 0.0, 0.0, false, reason,
    )
end

@inline _valid_grazing_margin(value::Real) = !isnan(value) && value >= zero(value)

@inline function _validate_grazing_margin(value::Real, name::AbstractString)
    _valid_grazing_margin(value) ||
        throw(DomainError(value, "$name must be nonnegative and not NaN"))
    return value
end

"""
    grazing_interval_report(wedge, ang, k, L; kwargs...)

Certify that the local interval [φ−h, φ+h] stays inside (0, α), keeps both KP
integers fixed, and avoids cotangent poles and vanishing transition arguments.
Invalid wavenumbers or safety margins return a report with `valid == false`.
"""
function grazing_interval_report(
    wedge::Wedge,
    ang::RayAngles,
    k::Number,
    L::Number;
    face::Symbol=:auto,
    transition_margin::Real=1.0e-3,
    x_margin::Real=1.0e-8,
    branch_margin::Real=64 * eps(Float64),
    gprime_reltol::Real=1.0e-12,
    allow_interior::Bool=false,
    allow_infinite_L::Bool=false,
)
    loc = grazing_local_angles(wedge, ang; face)
    return _interval_report_local(
        wedge, loc.phi, loc.h, loc.face, k, L;
        transition_margin, x_margin, branch_margin, gprime_reltol,
        allow_interior, allow_infinite_L,
    )
end

function _interval_report_local(
    wedge::Wedge,
    phi,
    h,
    face::Symbol,
    k,
    L;
    transition_margin::Real=1.0e-3,
    x_margin::Real=1.0e-8,
    branch_margin::Real=64 * eps(Float64),
    gprime_reltol::Real=1.0e-12,
    include_gprime::Bool=true,
    allow_interior::Bool=false,
    allow_infinite_L::Bool=false,
)
    alpha = wedge.alpha
    if !(h >= 0 && isfinite(h))
        return _empty_report(face, 0, "h must be finite and nonnegative")
    end
    # The certificate margins (|sin ψ|, kL·a, branch distance) are real
    # comparisons and the soft path integrates F' (defined for real argument).
    # A complex-typed k or L cannot be ordered against the real margins, so a
    # complex request is refused here instead of crashing on `min`/`<`. The
    # complex-capable four-term baseline remains available via `pec_wedge_DsDh`.
    if !(k isa Real && L isa Real)
        return _empty_report(
            face, h,
            "continuation certificate requires real k and L; use pec_wedge_DsDh for complex media",
        )
    end
    if !(isfinite(k) && k > zero(k))
        return _empty_report(face, h, "continuation requires a finite positive wavenumber k")
    end
    for (value, name) in (
        (transition_margin, "transition_margin"),
        (x_margin, "x_margin"),
        (branch_margin, "branch_margin"),
        (gprime_reltol, "gprime_reltol"),
    )
        _valid_grazing_margin(value) ||
            return _empty_report(face, h, "$name must be nonnegative and not NaN")
    end
    # The paper method is stated for exterior wedges π < α ≤ 2π. The identity
    # Ds = C[G(φ−h) − G(φ+h)] = −C h ∫ G'(φ+hξ)dξ holds for any 0 < α ≤ 2π, and
    # the pole/branch/transition margins below certify correctness independent of
    # α, so the robust dispatcher enables interior wedges via `allow_interior`.
    alpha_lower_ok = allow_interior ? (alpha > 0) : (alpha > π)
    if !(alpha_lower_ok && alpha <= 2π)
        return _empty_report(
            face, h,
            allow_interior ? "continuation requires 0 < α ≤ 2π" :
                             "continuation requires an exterior wedge π < α ≤ 2π",
        )
    end
    # Finite positive L is the default; the far-field limit L = +Inf is a valid
    # continuation (F → 1, and the F' term k L a' → 0) enabled via
    # `allow_infinite_L`. A non-positive or NaN L is always refused.
    L_ok = allow_infinite_L ? (L > 0 && !isnan(L)) : (L > 0 && isfinite(L))
    if !L_ok
        return _empty_report(face, h, allow_infinite_L ?
            "continuation requires a positive common L (finite or +Inf)" :
            "continuation requires a finite positive common L")
    end
    if phi - h <= 0 || phi + h >= alpha
        return _empty_report(face, h, "interval leaves the exterior angular region")
    end
    n = wedge_n(wedge)
    beta_lo = phi - h
    beta_hi = phi + h
    # In the plane-wave limit the soft coefficient reduces to an F-free closed
    # form that does not use the KP integer N or the distance parameter a, so the
    # branch and transition-argument tests do not apply; only the cotangent-pole
    # exclusion and the in-domain interval remain relevant.
    farfield = isinf(L)
    sig_lo = _branch_signature(beta_lo, n)
    sig_hi = _branch_signature(beta_hi, n)
    signatures = (sig_lo, sig_hi)
    if !farfield && sig_lo != sig_hi
        return GrazingIntervalReport(
            false, face, _primal_float(h), 0.0, 0.0, signatures, 0.0, 0.0, false,
            "KP integer branch changes inside interval",
        )
    end

    kL = k * L
    min_sin = Inf
    min_x = Inf
    min_branch = Inf
    for (idx, sigma) in enumerate((+1, -1))
        smin, xmin, bmin = _minimum_term_margins(beta_lo, beta_hi, sigma, n, sig_lo[idx], kL)
        min_sin = min(min_sin, smin)
        if !farfield
            min_x = min(min_x, xmin)
            min_branch = min(min_branch, bmin)
        end
    end
    if !farfield && min_branch <= branch_margin
        return GrazingIntervalReport(
            false, face, _primal_float(h), _primal_float(min_sin), _primal_float(min_x),
            signatures, _primal_float(min_branch), 0.0, false,
            "KP integer branch boundary is too close",
        )
    end
    if min_sin <= transition_margin
        return GrazingIntervalReport(
            false, face, _primal_float(h), _primal_float(min_sin), _primal_float(min_x),
            signatures, _primal_float(min_branch), 0.0, false,
            "cotangent pole is too close",
        )
    end
    if min_x <= x_margin
        return GrazingIntervalReport(
            false, face, _primal_float(h), _primal_float(min_sin), _primal_float(min_x),
            signatures, _primal_float(min_branch), 0.0, false,
            "transition argument is too small",
        )
    end

    # G'(φ) and G(φ) are diagnostic-only (the `gprime_abs`/`degenerate_odd`
    # fields). The production evaluator computes the soft integral from the
    # quadrature nodes and never reads them, so it requests `include_gprime=false`
    # to skip one derivative and one kernel evaluation per certified call.
    if !include_gprime
        return GrazingIntervalReport(
            true, face, _primal_float(h), _primal_float(min_sin), _primal_float(min_x),
            signatures, _primal_float(min_branch), NaN, false, "",
        )
    end
    gprime = two_term_kernel_derivative(phi, wedge, k, L)
    G0 = two_term_kernel(phi, wedge, k, L)
    gabs = abs(gprime)
    # Use a unit reference when both G and G' are near an exact symmetry zero.
    # Scaling only by |G| (or eps) classifies roundoff-sized derivative noise as
    # nondegenerate precisely when the coefficient-relative metric is undefined.
    degenerate = gabs <= gprime_reltol * max(one(gabs), abs(G0))
    return GrazingIntervalReport(
        true, face, _primal_float(h), _primal_float(min_sin), _primal_float(min_x),
        signatures, _primal_float(min_branch), _primal_float(gabs), degenerate, "",
    )
end

function _soft_continuation(wedge::Wedge, phi, h, k, L, order::Int)
    C = pec_wedge_prefactor(k, wedge_n(wedge))
    h == 0 && return zero(C)
    nodes, weights = gauss_legendre_nodes(order)
    # Seed the accumulator from the first quadrature node. This evaluates the
    # integrand exactly `order` times (no throwaway warm-up call at φ) and gives
    # `acc` the integrand's element type from the start, so a Dual argument does
    # not force a ComplexF64→Complex{Dual} widening on the first `+=`.
    acc = weights[1] * two_term_kernel_derivative(phi + h * nodes[1], wedge, k, L)
    for i in 2:length(nodes)
        acc += weights[i] * two_term_kernel_derivative(phi + h * nodes[i], wedge, k, L)
    end
    return -C * h * acc
end

"""
    _farfield_pec_DsDh(wedge, phi, h, k)

Exact plane-wave (L → ∞) soft and hard coefficients on an o-face-measured pair
`(phi, h)`. With F ≡ 1 the paired cotangent difference collapses in closed form,
    Ds_∞ = C Σ_σ sin(σ h / n) / [sin ψ_σ(φ−h) · sin ψ_σ(φ+h)],
so no quadrature is needed and the only small factor, sin(σ h / n), is formed
directly rather than by subtracting two coalescing cotangents. The expression
stays accurate up to a genuine cotangent pole (a shadow or reflection boundary),
where it is refused. The hard coefficient is the subtraction-free sum
`C [G_∞(φ−h) + G_∞(φ+h)]` and has the same pole exclusion.
"""
function _farfield_pec_DsDh(wedge::Wedge, phi, h, k)
    n = wedge_n(wedge)
    C = pec_wedge_prefactor(k, n)
    acc_ds = zero(C)
    acc_g = zero(C)
    for sigma in (+1, -1)
        psim = (π + sigma * (phi - h)) / (2n)
        psip = (π + sigma * (phi + h)) / (2n)
        sm = sin(psim)
        sp = sin(psip)
        if sm == 0 || sp == 0
            throw(GrazingDomainError(
                "plane-wave kernel is at a cotangent pole; use pec_wedge_DsDh",
            ))
        end
        acc_ds += sin(sigma * h / n) / (sm * sp)
        acc_g += cos(psim) / sm + cos(psip) / sp
    end
    return (C * acc_ds, C * acc_g)
end

function _grazing_fail(wedge, ang, k, L, report, on_fail::Symbol)
    on_fail in (:error, :four_term) ||
        throw(ArgumentError("on_fail must be :error or :four_term"))
    if on_fail === :four_term
        return pec_wedge_DsDh(wedge, ang, k, L)
    end
    throw(GrazingDomainError(report.reason))
end

"""
    pec_wedge_DsDh_grazing(wedge, ang, k, L; kwargs...)

Cancellation-free PEC evaluation of the standard KP pairing.

- Soft: Ds = −C h ∫_{-1}^{1} G'(φ + h ξ) dξ after the interval certificate.
- Hard: Dh = C [G(φ−h) + G(φ+h)], which has no odd-pair cancellation.
- `pec_wedge_DsDh` is left unchanged for the four-term comparison and for AD
  through the original pairing.

Keyword `on_fail` is `:error` (default) or `:four_term`. Impedance wedges and
unequal transition distances are refused. `L = Inf` is refused by default and
enabled through `allow_infinite_L=true`, which uses the exact far-field closed
form. A small G'(φ) is reported on `grazing_interval_report` but does not block
the integral.
"""
function pec_wedge_DsDh_grazing(
    wedge::Wedge,
    ang::RayAngles,
    k::Number,
    L::Number;
    order::Int=8,
    on_fail::Symbol=:error,
    check_domain::Bool=true,
    face::Symbol=:auto,
    transition_margin::Real=1.0e-3,
    x_margin::Real=1.0e-8,
    branch_margin::Real=64 * eps(Float64),
    allow_interior::Bool=false,
    allow_infinite_L::Bool=false,
    convention::PhasorConvention=EXP_IWT,
)
    convention.sgn == +1 || error("Only exp(+iωt) convention is supported")
    on_fail in (:error, :four_term) ||
        throw(ArgumentError("on_fail must be :error or :four_term"))
    order >= 1 || throw(ArgumentError("Gauss–Legendre order must be positive"))
    _validate_grazing_margin(transition_margin, "transition_margin")
    _validate_grazing_margin(x_margin, "x_margin")
    _validate_grazing_margin(branch_margin, "branch_margin")
    _validate_wavenumber(k)
    _validate_effective_L(L)
    if isinf(L) && !allow_infinite_L
        report = _empty_report(:o, 0, "continuation requires a finite positive common L")
        return _grazing_fail(wedge, ang, k, L, report, on_fail)
    end
    # A complex k or L has no certified continuation (the soft integrand F' is
    # real-argument and the margins are real). Route it through `on_fail` before
    # any certificate work so `:four_term` reaches the complex-capable baseline
    # and `:error` reports a typed domain error, even when `check_domain=false`.
    if !(k isa Real && L isa Real)
        report = _empty_report(
            :o, 0,
            "continuation requires real k and L; use pec_wedge_DsDh for complex media",
        )
        return _grazing_fail(wedge, ang, k, L, report, on_fail)
    end

    loc = grazing_local_angles(wedge, ang; face)
    report = _interval_report_local(
        wedge, loc.phi, loc.h, loc.face, k, L;
        transition_margin, x_margin, branch_margin, include_gprime=false,
        allow_interior, allow_infinite_L,
    )
    if check_domain && !report.valid
        return _grazing_fail(wedge, ang, k, L, report, on_fail)
    end

    if isinf(L)
        # Plane-wave limit: exact cancellation-free closed form (no quadrature).
        return _farfield_pec_DsDh(wedge, loc.phi, loc.h, k)
    end
    C = pec_wedge_prefactor(k, wedge_n(wedge))
    if loc.h == 0
        return (zero(C), 2 * C * two_term_kernel(loc.phi, wedge, k, L))
    end
    Ds = _soft_continuation(wedge, loc.phi, loc.h, k, L, order)
    gm = two_term_kernel(loc.phi - loc.h, wedge, k, L)
    gp = two_term_kernel(loc.phi + loc.h, wedge, k, L)
    return (Ds, C * (gm + gp))
end

function pec_wedge_DsDh_grazing(
    wedge::Wedge,
    ang::RayAngles,
    k::Number,
    Li::Real,
    Lro::Real,
    Lrn::Real;
    kwargs...,
)
    throw(ArgumentError(
        "Face-grazing continuation requires a common distance parameter L. " *
        "Unequal Li, Lro, Lrn can make the paired kernels disagree at h = 0. " *
        "Use pec_wedge_DsDh(wedge, ang, k, Li, Lro, Lrn) for the four-term evaluation.",
    ))
end

function pec_wedge_DsDh_grazing(iw::ImpedanceWedge, args...; kwargs...)
    throw(ArgumentError(
        "Face-grazing continuation is defined for the PEC pairing. " *
        "Impedance wedges weight reflection terms by Fresnel coefficients, " *
        "so the two-term kernels need not cancel. Use impedance_wedge_DsDh.",
    ))
end

"""
    pec_wedge_Ds_linear(wedge, ang, k, L; face=:auto)

Leading soft Taylor term −2 C h G'(φ). Comparison formula, not a replacement
for `pec_wedge_DsDh` or `pec_wedge_DsDh_grazing`.
"""
function pec_wedge_Ds_linear(wedge::Wedge, ang::RayAngles, k::Number, L::Number; face::Symbol=:auto)
    loc = grazing_local_angles(wedge, ang; face)
    C = pec_wedge_prefactor(k, wedge_n(wedge))
    return -2 * C * loc.h * two_term_kernel_derivative(loc.phi, wedge, k, L)
end

"""
    wedge_DsDh(wedge, ang, k, L; grazing_switch=1e-2, kwargs...) -> (Ds, Dh)

Robust wedge diffraction coefficients using the most accurate available method
for the request. This is the recommended entry point; the caller does not need
to know whether the geometry is near face grazing.

Routing:

- PEC `Wedge`, common real `L` (finite or `+Inf`): near face grazing the soft
  coefficient of the four-term Kouyoumjian–Pathak form loses significance to
  cancellation, so the certified cancellation-free continuation
  ([`pec_wedge_DsDh_grazing`](@ref)) is used; away from grazing, or whenever the
  interval certificate cannot be met, the four-term [`pec_wedge_DsDh`](@ref) is used.
  Interior and exterior wedges, both faces, and the plane-wave limit `L = Inf`
  are handled.
- PEC `Wedge` with complex `k`/`L`: the complex-capable four-term
  [`pec_wedge_DsDh`](@ref). The real-argument continuation is unavailable, so a
  common complex distance can retain the grazing-cancellation risk.
- PEC `Wedge` with three genuinely unequal distances `Li, Lro, Lrn`: the
  four-term [`pec_wedge_DsDh`](@ref). If all three distances are equal and the
  standard PEC signs are used, this overload delegates to the common-distance
  router so it does not reintroduce the grazing cancellation.
- `ImpedanceWedge`: [`impedance_wedge_DsDh`](@ref); Fresnel-weighted reflection terms
  do not cancel at grazing.

`grazing_switch` (default `1e-2` rad) is the face offset below which the
continuation is attempted. Above it the four-term value is returned unchanged;
the two agree to ~1e-13 there, so the transition is numerically seamless. The
returned `(Ds, Dh)` are the soft and hard scalar coefficients.
"""
function wedge_DsDh(
    wedge::Wedge,
    ang::RayAngles,
    k::Number,
    L::Number;
    grazing_switch::Real=1.0e-2,
    order::Int=8,
    transition_margin::Real=1.0e-3,
    x_margin::Real=1.0e-8,
    branch_margin::Real=64 * eps(Float64),
    face::Symbol=:auto,
    convention::PhasorConvention=EXP_IWT,
)
    convention.sgn == +1 || error("Only exp(+iωt) convention is supported")
    isfinite(grazing_switch) && grazing_switch >= zero(grazing_switch) ||
        throw(DomainError(grazing_switch, "grazing_switch must be finite and nonnegative"))
    order >= 1 || throw(ArgumentError("Gauss–Legendre order must be positive"))
    _validate_grazing_margin(transition_margin, "transition_margin")
    _validate_grazing_margin(x_margin, "x_margin")
    _validate_grazing_margin(branch_margin, "branch_margin")
    # The cancellation-free continuation applies to a real, positive common L
    # (finite or the +Inf plane-wave limit). Everything else is well conditioned
    # in the four-term form.
    if k isa Real && L isa Real && L > 0 && !isnan(L)
        loc = grazing_local_angles(wedge, ang; face)
        if loc.h < grazing_switch
            # Finite L uses the certified derivative continuation; the plane-wave
            # limit L = Inf uses the exact closed form, which stays accurate up to
            # a genuine cotangent pole, so both share the same pole margin.
            return pec_wedge_DsDh_grazing(
                wedge, ang, k, L;
                order, on_fail=:four_term, face,
                transition_margin, x_margin, branch_margin,
                allow_interior=true, allow_infinite_L=true, convention,
            )
        end
    end
    return pec_wedge_DsDh(wedge, ang, k, L; convention)
end

"""
    wedge_DsDh(wedge, ang, k, Li, Lro, Lrn; kwargs...) -> (Ds, Dh)

Three-distance PEC form. Genuinely unequal incident-shadow and reflection
distances do not satisfy the common-kernel continuation identity, so the
four-term [`pec_wedge_DsDh`](@ref) is used. When all three distances are equal
and `Rs=-1`, `Rh=+1`, this method delegates to the common-distance router and
preserves its cancellation-free grazing behavior.
"""
function wedge_DsDh(
    wedge::Wedge,
    ang::RayAngles,
    k::Number,
    Li::Real,
    Lro::Real,
    Lrn::Real;
    Rs::Number=-1,
    Rh::Number=+1,
    convention::PhasorConvention=EXP_IWT,
)
    if Li == Lro == Lrn && Rs == -1 && Rh == +1
        return wedge_DsDh(wedge, ang, k, Li; convention)
    end
    return pec_wedge_DsDh(wedge, ang, k, Li, Lro, Lrn; Rs, Rh, convention)
end

"""
    wedge_DsDh(iw::ImpedanceWedge, ang, k, L...; convention=EXP_IWT) -> (Ds, Dh)

Impedance wedge coefficients. Fresnel weighting changes the PEC odd pairing, so
the PEC continuation identity does not apply and [`impedance_wedge_DsDh`](@ref)
is used directly.
"""
function wedge_DsDh(
    iw::ImpedanceWedge,
    ang::RayAngles,
    k::Number,
    L::Number;
    convention::PhasorConvention=EXP_IWT,
)
    return impedance_wedge_DsDh(iw, ang, k, L; convention)
end

function wedge_DsDh(
    iw::ImpedanceWedge,
    ang::RayAngles,
    k::Number,
    Li::Real,
    Lro::Real,
    Lrn::Real;
    convention::PhasorConvention=EXP_IWT,
)
    return impedance_wedge_DsDh(iw, ang, k, Li, Lro, Lrn; convention)
end
