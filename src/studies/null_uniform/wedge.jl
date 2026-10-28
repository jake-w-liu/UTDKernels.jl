# Sommerfeld half-plane spectral model and its canonical reduction.
#
# Geometry: PEC wedge with interior angle (2 - N) pi and exterior field
# region 0 < phi < N pi.  The edge is along z; the transverse wave scale
# is the dimensionless product Krho = k * rho (normal incidence,
# e^{+i omega t} convention, outgoing field ~ exp(-i Krho)).  The paper's
# electromagnetic example is the half-plane, N = 2.
#
# Scalar eigenfunction solution (Dirichlet :soft for E_z, Neumann :hard
# for H_z):
#   u(rho,phi) = (4/N) sum_{m>=1} i^{nu_m} J_{nu_m}(Krho) s(nu_m phi)
#                s(nu_m phi_i)  +  (2/N) J_0(Krho)   [Neumann only]
# with nu_m = m/N, s = sin (soft) or cos (hard).
#
# Sommerfeld spectral kernel (period 2N pi in alpha = phi + sigma):
#   f(alpha) = (1/2N)[cot((alpha+pi-phi_i)/(2N))
#                   ± cot((alpha+pi+phi_i)/(2N))]
# (+ Neumann/hard, - Dirichlet/soft in the second cotangent).
# Poles: alpha_p = -pi -/+ phi_i + 2N pi m.  Residues are +1 on the
# first cotangent family and +/-1 on the second (+ Neumann, - Dirichlet);
# for the Neumann kernel both families have residue +1.
# Hard-kernel zeros: alpha_z = -pi + m N pi.
#
# Diffracted field on the Sommerfeld contour (verified against
# eigenfunction - GO to ~1e-14 for N = 2):
#   u^d = s(-2 pi) - s(0),
#   s(sigma0) = (1/2 pi i) ∫_{SDP(sigma0)} f(phi + sigma)
#               e^{-i Krho cos sigma} d sigma,
#   SDP(sigma0):  sigma(b) = sigma0 + sign(b) acos(sech b) + i b,
#                 d sigma/db = sech(b) + i,  b in (-inf, inf).
# Along this contour Re(-i cos sigma) = -sinh(|b|) tanh(|b|), so the
# integrand decays doubly exponentially and the phase at the saddle is
# exp(-i Krho), the outgoing cylindrical factor.
#
# Local canonical reduction about each saddle sigma0 in 2 pi Z.
#   u = c sin((sigma - sigma0)/2),   c = e^{-i pi/4} sqrt(2 Krho),
#   sigma(u) = sigma0 + 2 asin(u/c),   du/d sigma = (c/2) cos((sigma-sigma0)/2),
# gives e^{-i Krho cos sigma} = e^{-i Krho} e^{-u^2} and the mapped SDP is
# a contour through u = 0 along arg u = -pi/4.  Continuing the contour to
# the real u-axis fixes the pole sheets, and
#   s(sigma0) = -(e^{-i Krho}/2) I_c,
#   I_c = (i/pi) ∫ \tilde a(u) e^{-u^2} / prod_j (z_j - u) du,
#   \tilde a(u) = f(phi + sigma(u)) prod_j (z_j - u) / J(u),
#   z_j = c sin((sigma_{p,j} - sigma0)/2).
# For N = 2 every 4 pi image of the kernel poles sigma_p = -phi - pi
# +/- phi_i + 4 pi m maps onto the same two points z_1, z_2, so the
# canonical integral is exactly a two-pole canonical form.  The hard
# kernel zero sigma_z = pi - phi places an exact spectral amplitude zero
# that reaches sigma0 at phi = pi: the mapped null coordinate
# Lambda = -u(sigma_z) vanishes there for both saddles.

"""
    wedge_kernel(alpha, phi_i, N, polarization)

Sommerfeld spectral kernel for the PEC wedge. `polarization` is `:soft`
(Dirichlet, E_z) or `:hard` (Neumann, H_z).
"""
function wedge_kernel(alpha::Number, phi_i::Real, N::Real,
                      polarization::Symbol)
    polarization in (:soft, :hard) ||
        throw(ArgumentError("polarization must be :soft or :hard"))
    isfinite(N) && N > 0 || throw(ArgumentError("wedge index N must be positive and finite"))
    _as_complex(alpha, "spectral angle")
    isfinite(phi_i) || throw(ArgumentError("source angle must be finite"))
    A = (alpha + pi - phi_i) / (2N)
    B = (alpha + pi + phi_i) / (2N)
    s = polarization === :soft ? -1 : 1
    return (cot(A) + s * cot(B)) / (2N)
end

"""
    wedge_kernel_zeros(N) -> zeros of the hard kernel (mod 2Nπ)

Zeros of the Neumann kernel numerator sin((α+π)/N): α = -π + m Nπ.  The
soft kernel has no corresponding moving zero.
"""
function wedge_kernel_zeros(N::Real)
    isfinite(N) && N > 0 || throw(ArgumentError("wedge index N must be positive and finite"))
    zeros = [-pi + m * N * pi for m in 0:4]
    all(UTDKernels._number_isfinite, zeros) || throw(DomainError(
        zeros, "kernel zero locations exceed the active numeric range"))
    return zeros
end

"""
    wedge_kernel_poles(φ_i, N) -> kernel poles in the alpha-plane
"""
function wedge_kernel_poles(phi_i::Real, N::Real)
    isfinite(N) && N > 0 && isfinite(phi_i) || throw(ArgumentError(
        "source angle must be finite and wedge index positive and finite"))
    poles = typeof(complex(float(phi_i + N)))[]
    for m in -2:3, s in (-1, 1)
        push!(poles, -pi + s * phi_i + 2 * N * pi * m)
    end
    all(UTDKernels._number_isfinite, poles) || throw(DomainError(
        poles, "kernel pole locations exceed the active numeric range"))
    return poles
end

"""
    wedge_eigenfunction(Kρ, φ, φ_i, N, polarization; rtol=1e-13)

Converged eigenfunction-series solution of the wedge problem.
"""
function wedge_eigenfunction(Krho::Real, phi::Real, phi_i::Real, N::Real,
                             polarization::Symbol; rtol::Real=1e-13,
                             mmax::Integer=200_000)
    polarization in (:soft, :hard) ||
        throw(ArgumentError("polarization must be :soft or :hard"))
    all(isfinite,(Krho,phi,phi_i,N)) && Krho>0 && N>0 || throw(ArgumentError("finite angles and positive finite Krho,N are required"))
    isfinite(rtol) && rtol>0 || throw(ArgumentError("rtol must be positive and finite"))
    mmax>=1 || throw(ArgumentError("mmax must be positive"))
    soft = polarization === :soft
    acc = complex(0.0)
    soft || (acc += (2.0 / N) * besselj(0.0, Krho))
    A = 4.0 / N
    smallrun = 0
    converged=false
    for m in 1:mmax
        nu = m / N
        ang = soft ? sin(nu * phi) * sin(nu * phi_i) :
                     cos(nu * phi) * cos(nu * phi_i)
        term = A * (im^nu) * besselj(nu, Krho) * ang
        acc += term
        if abs(term) <= rtol * max(abs(acc), 1e-300)
            smallrun += 1
            if smallrun>=8
                converged=true
                break
            end
        else
            smallrun = 0
        end
    end
    converged || throw(ArgumentError("wedge eigenfunction series did not converge within mmax=$mmax"))
    return acc
end

"""
    wedge_go(Kρ, φ, φ_i, N, polarization; half_weight=false)

Geometrical-optics part of the wedge field for a source at φ_i with faces
at φ = 0 and φ = Nπ.  Covers the incident wave and single reflections off
lit faces.  With `half_weight=false` each constituent is counted only
strictly inside its lit sector (the boundary value itself is not
assigned).  With `half_weight=true` the symmetric half-residue boundary
convention is applied: a constituent exactly on its transition boundary
contributes one half of its wave, paired with the symmetric
principal-value convention of [`moment_int_plane`](@ref).
"""
function wedge_go(Krho::Real, phi::Real, phi_i::Real, N::Real,
                  polarization::Symbol; half_weight::Bool=false)
    polarization in (:soft,:hard) || throw(ArgumentError("polarization must be :soft or :hard"))
    all(isfinite,(Krho,phi,phi_i,N)) && Krho>0 && N>0 || throw(ArgumentError("finite angles and positive finite Krho,N are required"))
    soft = polarization === :soft
    s = soft ? -1 : 1
    u = complex(0.0)
    weight(lit, on_boundary) = lit ? 1.0 :
        (half_weight && on_boundary ? 0.5 : 0.0)
    tol = 1e-12
    # incident wave: blocked beyond the shadow boundary phi = phi_i + pi.
    if phi_i + pi >= N * pi
        w = weight(0 < phi < N * pi, abs(phi - N * pi) < tol)
    else
        w = weight(0 < phi < phi_i + pi, abs(phi - phi_i - pi) < tol)
    end
    u += w * exp(im * Krho * cos(phi - phi_i))
    # face-0 reflection: incident rays reach the upper face for
    # 0 < phi_i < pi; reflected rays fill 0 < phi < pi - phi_i.
    if 0 < phi_i < pi
        w = weight(0 < phi < pi - phi_i, abs(phi - pi + phi_i) < tol)
        u += s * w * exp(im * Krho * cos(phi + phi_i))
    end
    # face-Nπ reflection: reached when phi_i > (N-1) pi; reflected rays
    # fill the sector between the reflection boundary and the face.
    if phi_i > (N - 1) * pi
        rb = 2 * N * pi - phi_i - pi
        w = weight(rb < phi < N * pi, abs(phi - rb) < tol)
        u += s * w * exp(im * Krho * cos(phi - 2 * N * pi + phi_i))
    end
    return u
end

"""
    wedge_diffracted(Kρ, φ, φ_i, N, polarization; half_weight=false) -> u^d

Diffracted reference field `u - u^GO` from the converged eigenfunction
series and the geometrical-optics constituents.  Off the shadow and
reflection boundaries this is a converged reference; exactly on a
boundary the two-sided prescription differs by half a residue and the
boundary value is not assigned by this function.  `half_weight` selects
the symmetric half-residue GO convention of [`wedge_go`](@ref).
"""
function wedge_diffracted(Krho::Real, phi::Real, phi_i::Real, N::Real,
                          polarization::Symbol; half_weight::Bool=false,
                          kw...)
    return wedge_eigenfunction(Krho, phi, phi_i, N, polarization; kw...) -
           wedge_go(Krho, phi, phi_i, N, polarization;
                    half_weight=half_weight)
end

"""
    wedge_sdp_quadrature(Kρ, φ, φ_i, N, polarization; saddle, B, rtol)

Converged quadrature of one Sommerfeld saddle contribution

    s(σ0) = (1/2πi) ∫_{SDP(σ0)} f(φ + σ) e^{-i Kρ cos σ} dσ,

on the exact steepest-descent contour through σ0.  Only saddles with
cos σ0 = +1 (σ0 ∈ 2πℤ: the outgoing-sheet saddles 0 and -2π of the
half-plane decomposition) are supported.  The contour is
σ(b) = σ0 + sign(b) acos(sech b) + i b, for which cos(Re σ) cosh(Im σ) = 1
and the integrand decays like exp(-Kρ sinh|b| tanh|b|).  A kernel pole
exactly at σ0 places a singularity on the contour; that is the physical
transition point and is handled by the canonical model, not by this
quadrature.
"""
function wedge_sdp_quadrature(Krho::Real, phi::Real, phi_i::Real, N::Real,
                              polarization::Symbol; saddle::Real=0.0,
                              B::Real=7.0, rtol::Real=1e-13)
    _positive_real(Krho, "Krho")
    B > 0 || throw(ArgumentError("B must be positive"))
    abs(cos(saddle) - 1) < 1e-9 || throw(ArgumentError(
        "only saddles with cos(saddle) = 1 (saddle in 2πℤ) are supported"))
    integrand = b -> begin
        sig = saddle + sign(b) * acos(sech(b)) + im * b
        wedge_kernel(phi + sig, phi_i, N, polarization) *
            exp(-im * Krho * cos(sig)) * (sech(b) + im)
    end
    val, err = quadgk(integrand, -B, 0.0, B; rtol=rtol, maxevals=10^7)
    err > 1e-9 * max(1.0, abs(val)) &&
        @warn "wedge_sdp_quadrature loose tolerance" err val
    return val / (2pi * im)
end

"""
    wedge_diffracted_sdp(Kρ, φ, φ_i, N, polarization; kw...) -> u^d

Two-saddle Sommerfeld decomposition `s(-2π) - s(0)` of the diffracted
field.  For the half-plane this equals `wedge_diffracted` to quadrature
accuracy at every interior angle; on an exact shadow or reflection
boundary the two-sided eigenfunction/GO reference and the saddle sum
differ by the boundary half-residue convention.
"""
function wedge_diffracted_sdp(Krho::Real, phi::Real, phi_i::Real, N::Real,
                              polarization::Symbol; kw...)
    return wedge_sdp_quadrature(Krho, phi, phi_i, N, polarization;
                                saddle=-2pi, kw...) -
           wedge_sdp_quadrature(Krho, phi, phi_i, N, polarization;
                                saddle=0.0, kw...)
end

# ---- canonical reduction -----------------------------------------------

"""
    wedge_phase_map(Kρ; saddle=0) -> (u_of, sigma_of, jac, c, J0)

Exact local phase map about an outgoing-sheet saddle σ0 ∈ 2πℤ:

    u(σ) = c sin((σ - σ0)/2),   c = e^{-iπ/4} √(2Kρ),
    σ(u) = σ0 + 2 asin(u/c)        (branch through σ0),
    J(σ) = du/dσ = (c/2) cos((σ - σ0)/2),

for which e^{-iKρ cos σ} = e^{-iKρ} e^{-u²}.  `J0 = c/2` is the Jacobian
at the saddle.  The map is 4π-periodic in σ and conformal for
|σ - σ0| < π, equivalently |u| < |c|.
"""
function wedge_phase_map(Krho::Real; saddle::Real=zero(Krho))
    Krho, saddle = promote(_positive_real(Krho, "Krho"), float(saddle))
    abs(cos(saddle) - 1) < 1e-9 || throw(ArgumentError(
        "only saddles with cos(saddle) = 1 (saddle in 2πℤ) are supported"))
    c = conj(UTDKernels._shadow_quarter_phase(Krho)) * sqrt(Krho) * sqrt(oftype(float(Krho), 2))
    u_of = sigma -> c * sin((sigma - saddle) / 2)
    sigma_of = u -> saddle + 2 * asin(u / c)
    jac = sigma -> (c / 2) * cos((sigma - saddle) / 2)
    return (u_of=u_of, sigma_of=sigma_of, jac=jac, c=c, J0=c / 2,
            saddle=saddle)
end

"""
    wedge_canonical_model(Kρ, φ, φ_i, N, polarization; saddle, ncoef)

Canonical data for one local saddle contribution of the half-plane
(N = 2) diffracted field.  The kernel poles
σ_p = -φ - π ± φ_i + 4πm all alias under the 4π-periodic phase map onto
exactly two points z_1, z_2; extracting them gives

    s(σ0) = prefactor · I_c,
    I_c = (i/π) ∫ ã(u) e^{-u²} / [(z_1 - u)(z_2 - u)] du,

with `prefactor = -e^{-iKρ}/2` and
`ã(u) = f(φ+σ(u)) (z_1-u)(z_2-u) / J(u)` analytic on |u| < |c|.

Returns a NamedTuple with `poles` (mapped z_j), `zeros` (mapped kernel
zeros, hard polarization only), `lambda` (mapped null coordinate
-u(σ_z) of the σ-plane kernel zero nearest the saddle, the
branch-consistent record of the paper's null coordinate; `NaN` for soft
polarization, whose kernel has no such zero),
`amplitude_u`, `coefficients` (Taylor coefficients
c_m = ã^{(m)}(0)/m!), `pole_limits` (analytic removable limits of ã at
each z_j, see below), `prefactor`, and `maps`.

The pole sheets of the mapped real-axis integral are the analytic
continuations selected by [`moment_int_plane`](@ref): the upper-plane
value for Im z_j > 0 and its Stokes continuation for Im z_j < 0.

`pole_limits[j]` is the removable-singularity value of ã at z_j.  When
the principal-sheet preimage σ(z_j) of a mapped pole is itself a kernel
pole σ_p (which happens on the saddle lift the pole actually crosses),
the kernel residue Res_j multiplies the surviving regular factor and

    lim_{u→z_j} ã(u) = Res_j (z_j - z_{3-j});

when σ(z_j) is a regular point the factor (z_j - u) makes the limit
zero.  Mapped poles whose σ-plane preimage lies on a different
4π-periodic lift of the same algebraic z value therefore have a zero
limit even though a physical pole crossing may project onto them.
"""
function wedge_canonical_model(Krho::Real, phi::Real, phi_i::Real, N::Real,
                               polarization::Symbol; saddle::Real=zero(Krho),
                               ncoef::Integer=14)
    Krho, phi, phi_i, saddle = promote(_positive_real(Krho, "Krho"), float(phi), float(phi_i), float(saddle))
    N == 2 || throw(ArgumentError(
        "the canonical pole-pair model is implemented for the half-plane N = 2"))
    polarization in (:soft, :hard) ||
        throw(ArgumentError("polarization must be :soft or :hard"))
    ncoef >= 1 || throw(ArgumentError("ncoef must be positive"))
    maps = wedge_phase_map(Krho; saddle=saddle)
    u_of, sigma_of, jac = maps.u_of, maps.sigma_of, maps.jac
    # All kernel poles with their cotangent-family tag (s=+1 selects the
    # residue-+1 family, s=-1 the signed family); their 4π images alias
    # onto the same mapped values.
    raw = [(-phi - pi + s * phi_i + 4pi * m, s)
           for m in -2:2 for s in (-1, 1)]
    C = typeof(maps.c + phi + phi_i)
    # The 4pi images map to the same point analytically. Evaluate one
    # representative of each cotangent family, so an absolute distance
    # threshold cannot merge small-scale poles or split large-scale aliases.
    zs = C[u_of(-phi - pi + sign * phi_i) for sign in (-1, 1)]
    iszero(zs[2] - zs[1]) && throw(ArgumentError(
        "the half-plane canonical model requires two distinct mapped poles"))
    sort!(zs; by=abs)
    # Rationalizing cos(v)-cos(h+y), with t=sin(y)=u/c,
    # cancels the extracted two-pole polynomial analytically. The remaining
    # inverse square root has exact binomial Taylor coefficients, avoiding
    # numerical Cauchy differentiation and retaining parameter tangents.
    h = (phi + saddle + pi) / 2
    v = phi_i / 2
    a, sh, ch = cos(v), sin(h), cos(h)
    c = maps.c
    atilde = function (u)
        t = u / c
        w = sqrt(one(t) - t*t)
        return polarization === :soft ?
            c * sin(v) * (a + sh*t + ch*w) / w :
            c * (a*sh + t + ch*(a*t + sh)/w)
    end
    coeffs = Vector{C}(undef, ncoef + 1)
    q = one(real(c))
    inverse_c = inv(c)
    inverse_power = one(c)
    for m in 0:ncoef
        m >= 2 && (inverse_power *= inverse_c)
        coefficient = if m == 0
            polarization === :soft ? c*sin(v)*(a+ch) : c*sh*(a+ch)
        elseif iseven(m)
            q *= (m-1) / m
            polarization === :soft ? sin(v)*a*q*inverse_power : sh*ch*q*inverse_power
        elseif m == 1
            polarization === :soft ? sin(v)*sh : one(a)+a*ch
        else
            polarization === :soft ? sin(v)*sh*q*inverse_power : a*ch*q*inverse_power
        end
        UTDKernels._number_isfinite(coefficient) || throw(DomainError(
            coefficient, "half-plane amplitude coefficient exceeds the active numeric range"))
        coeffs[m+1] = coefficient
    end
    # Analytic removable limits of ã at the mapped poles.  For each z_j,
    # σ(z_j) on the saddle's principal asin sheet is checked against the
    # σ-plane kernel poles.  A match means the factor (z_j - u) cancels
    # the kernel pole and the limit is Res_j (z_j - z_{3-j}), where Res_j
    # is +1 for the s=+1 family and ±1 for the s=-1 family (+ hard,
    # - soft).  No match means ã vanishes at z_j.
    z_other(j) = zs[3 - j]
    branch_tolerance = max(1e-9, 32eps(typeof(UTDKernels._primal_value(real(c)))))
    pole_limits = map(enumerate(zs)) do (j, zj)
        sig = sigma_of(zj)
        lim = zero(C)
        for (p, s) in raw
            if abs(sig - p) < branch_tolerance * max(1.0, abs(sig))
                res = s == 1 ? one(C) :
                      (polarization === :hard ? one(C) :
                       -one(C))
                lim = res * (zj - z_other(j))
                break
            end
        end
        lim
    end
    # Mapped null coordinate: the hard-kernel zeros α_z = -π + 2πm sit at
    # σ_z = α_z - φ = -π - φ + 2πm in the σ plane.  The branch-consistent
    # record is the image of the σ-plane zero nearest the saddle, not the
    # mapped image of smallest |u|: distinct lifts alias onto ±u.
    zeros_u = C[]
    lambda = oftype(maps.c + phi + phi_i, NaN)
    if polarization === :hard
        sigma_zs = [-pi - phi + 2pi * m for m in -2:2]
        # A 2pi zero-family shift changes the mapped sign exactly. Avoid
        # a scale-dependent numerical deduplication of its periodic images.
        sigma_z = sigma_zs[argmin(abs.(sigma_zs .- saddle))]
        zero_base = convert(C, u_of(sigma_z))
        push!(zeros_u, zero_base)
        iszero(zero_base) || push!(zeros_u, -zero_base)
        lambda = -zero_base
    end
    prefactor = -exp(-im * Krho) / 2
    return (saddle=saddle, poles=zs, zeros=zeros_u, lambda=lambda,
            amplitude_u=atilde, coefficients=coeffs,
            pole_limits=pole_limits,
            prefactor=prefactor, maps=maps)
end

"""
    moment_int_plane(z, m; pv_tol=nothing)

Fixed-real-contour moments C_0..C_m(z): the defining upper-half-plane
value `W_m(z)` for `Im z > 0` and its continuation
`W_m(z) - 2 z^m e^{-z^2}` for `Im z < 0` (the crossed-pole residue moves
to the geometrical-optics part).  With `pv_tol` set, a pole inside
`|Im z| <= pv_tol` of the real axis is evaluated by the symmetric
principal-value convention `C_m(z) = W_m(z) - z^m e^{-z^2}`, the
half-residue boundary value paired with half-weight geometrical-optics
constituents.
"""
function moment_int_plane(z::Number, m::Integer;
                          pv_tol::Union{Nothing,Real}=nothing)
    pv_tol===nothing || (isfinite(pv_tol) && pv_tol>=0) || throw(ArgumentError("pv_tol must be finite and nonnegative"))
    W = faddeeva_moments(z, m)
    zz = _as_complex(z)
    if pv_tol !== nothing && abs(imag(zz)) <= float(pv_tol)
        e = exp(-zz * zz)
        return [W[j + 1] - e * zz^j for j in 0:m]
    elseif imag(zz) < 0
        e = 2 * exp(-zz * zz)
        return [W[j + 1] - e * zz^j for j in 0:m]
    end
    return collect(W)
end

function _validate_canonical_poles(zs)
    length(zs) in (1,2) || throw(ArgumentError("only one or two poles supported"))
    all(z -> z isa Number && isfinite(real(z)) && isfinite(imag(z)),zs) || throw(ArgumentError("mapped poles must be finite numbers"))
    length(zs)==1 || abs(zs[2]-zs[1])>1e-12 || throw(ArgumentError("mapped pole pair is degenerate"))
    return nothing
end

"""
    canonical_pole_pair_value(zs, coeffs_u, order; pv_tol=nothing)

Evaluate `I_c ≈ Σ_{m<=order} c_m [C_m(z_1) - C_m(z_2)] / (z_2 - z_1)` for
a two-pole extraction, or the single-pole form when `zs` has one entry.
`pv_tol` selects the symmetric principal-value convention of
[`moment_int_plane`](@ref) for a mapped pole on the real axis.
"""
function canonical_pole_pair_value(zs::AbstractVector,
                                   coeffs_u::AbstractVector,
                                   order::Integer;
                                   pv_tol::Union{Nothing,Real}=nothing)
    _validate_canonical_poles(zs)
    all(v -> v isa Number && UTDKernels._number_isfinite(v), coeffs_u) || throw(ArgumentError("amplitude coefficients must be finite numbers"))
    isempty(coeffs_u) && throw(ArgumentError("amplitude coefficients must not be empty"))
    order >= 0 || throw(ArgumentError("order must be nonnegative"))
    M = min(order, length(coeffs_u) - 1)
    if length(zs) == 1
        W = moment_int_plane(zs[1], M; pv_tol=pv_tol)
        return sum(coeffs_u[m + 1] * W[m + 1] for m in 0:M)
    end
    length(zs) == 2 ||
        throw(ArgumentError("only one or two poles supported"))
    dz = zs[2] - zs[1]
    abs(dz) > 1e-12 ||
        throw(ArgumentError("mapped pole pair is degenerate"))
    W1 = moment_int_plane(zs[1], M; pv_tol=pv_tol)
    W2 = moment_int_plane(zs[2], M; pv_tol=pv_tol)
    return sum(coeffs_u[m + 1] * (W1[m + 1] - W2[m + 1]) / dz for m in 0:M)
end

"""
    canonical_pole_extraction(zs, a_fn; pole_values=nothing,
                              coeffs=nothing, pv_tol=nothing)

Classical pole extraction plus the first regular saddle term applied to
each factor of the pole-pair canonical integral: for each z_j,

    L_j = ã(z_j) C_0(z_j) - i [ã(z_j) - ã(0)] / (√π z_j),

combined by the same partial fraction.  This uses the full regular
amplitude ã (more information than the two Taylor coefficients c_0,
c_1).  `pole_values[j]` may supply the analytic removable-singularity
value of ã at z_j (e.g. `wedge_canonical_model(...).pole_limits`) in
place of direct evaluation (required if that evaluation is non-finite);
`coeffs` may supply the Taylor coefficients
of ã for the ã(0) and ã'(0) values in the divided difference.  `pv_tol`
selects the symmetric principal-value boundary convention of
[`moment_int_plane`](@ref).
"""
# A regular amplitude is finite at the pole. A closure that forms 0*Inf
# must supply its analytic removable value explicitly; a symmetric sample
# cannot distinguish a removable zero from a genuine additional pole.
function _pole_limit_value(a_fn,zj::Number)
    value=a_fn(zj)
    isfinite(real(value)) && isfinite(imag(value)) || throw(DomainError(zj,
        "amplitude is non-finite at the pole; provide analytic pole_values"))
    return value
end

function canonical_pole_extraction(zs::AbstractVector, a_fn;
                                   pole_values=nothing,
                                   coeffs=nothing,
                                   pv_tol::Union{Nothing,Real}=nothing)
    _validate_canonical_poles(zs)
    coeffs===nothing || length(coeffs)>=2 || throw(ArgumentError("pole extraction needs at least two amplitude coefficients"))
    if pole_values !== nothing && length(pole_values) != length(zs)
        throw(ArgumentError("pole_values must match the mapped poles"))
    end
    pole_value_at(j, zz) = pole_values === nothing ?
        _pole_limit_value(a_fn, zz) : _as_complex(pole_values[j])
    a_at_zero() = coeffs === nothing ?
        _pole_limit_value(a_fn, zero(complex(first(zs)))) : _as_complex(coeffs[1])
    slope_at_zero() = coeffs === nothing ?
        (a_fn(im * 1e-8) - a_fn(-im * 1e-8)) / (2im * 1e-8) :
        _as_complex(coeffs[2])
    single(j) = begin
        zz = _as_complex(zs[j])
        az = pole_value_at(j, zz)
        a0 = a_at_zero()
        B = abs(zz) < 1e-10 ? slope_at_zero() : (az - a0) / zz
        return az * moment_int_plane(zz, 0; pv_tol=pv_tol)[1] -
               im * B / sqrt(pi)
    end
    if length(zs) == 1
        return single(1)
    end
    length(zs) == 2 || throw(ArgumentError("only one or two poles supported"))
    return (single(1) - single(2)) / (zs[2] - zs[1])
end

"""
    canonical_slope_separated(zs, coeffs_u)

Separated-pole slope baseline for the pole-pair integral: the frozen
term c_0 W_0(z_j) plus the nonuniform slope form
c_1 i / (2√π z_j²) on each pole, combined by partial fractions.
Singular as either z_j -> 0.
"""
function canonical_slope_separated(zs::AbstractVector,
                                   coeffs_u::AbstractVector)
    _validate_canonical_poles(zs)
    all(v -> v isa Number && UTDKernels._number_isfinite(v), coeffs_u) || throw(ArgumentError("amplitude coefficients must be finite numbers"))
    length(coeffs_u) >= 2 ||
        throw(ArgumentError("need at least two amplitude coefficients"))
    single(zj) = begin
        iszero(zj) &&
            throw(ArgumentError("separated slope is singular at z=0"))
        return coeffs_u[1] * moment_int_plane(zj, 0)[1] +
               coeffs_u[2] * im / (2 * sqrt(pi) * zj^2)
    end
    if length(zs) == 1
        return single(zs[1])
    end
    length(zs) == 2 || throw(ArgumentError("only one or two poles supported"))
    return (single(zs[1]) - single(zs[2])) / (zs[2] - zs[1])
end

"""
    canonical_pair_quadrature(zs, atilde; T=8.0, rtol=1e-12)

Independent adaptive quadrature of the pole-pair canonical integral on
the real axis.  The mapped poles z_j are complex, so the integrand is
regular on ℝ except in the degenerate limits z_j -> ℝ (pole on the
saddle), which this function rejects.  Requires |u| < |c| on the
integration window because `atilde` is evaluated through `σ(u)`.
"""
function canonical_pair_quadrature(zs::AbstractVector, atilde;
                                   T::Real=8.0, rtol::Real=1e-12)
    _validate_canonical_poles(zs)
    _quadrature_tolerances(rtol, 0.0)
    for zj in zs
        abs(imag(zj)) > 1e-10 || throw(DomainError(zj,
            "mapped pole lies on the real axis (saddle--pole coincidence); " *
            "use the canonical moment evaluation instead"))
    end
    isfinite(T) && T > 0 || throw(ArgumentError("T must be positive and finite"))
    # e^{-u^2} makes the tails negligible well before T = 8.
    breaks = sort!(collect(Float64, clamp.(real.(zs), -T, T)))
    val, err = quadgk(u -> _as_complex(atilde(u), "amplitude value") * exp(-u * u) /
                          prod(zj - u for zj in zs),
                      -T, breaks..., T; rtol=rtol, maxevals=10^7)
    err > 1e-9 * max(1.0, abs(val)) &&
        @warn "canonical_pair_quadrature loose tolerance" err val
    return im / pi * val
end

"""
    wedge_saddle_field(model, order; kind=:moment, pv_tol=nothing)

Local saddle field `s(σ0) = prefactor · I_c` from a
[`wedge_canonical_model`](@ref).  `kind` selects the approximation:
`:moment` truncates the canonical moment hierarchy at `order`;
`:pole_extraction` is the classical full-amplitude comparator, evaluated
with the model's analytic `pole_limits`; `:slope_separated` is the
nonuniform separated-slope comparator; `:quadrature` is an independent
real-axis reference (order ignored).  `pv_tol` selects the symmetric
principal-value boundary convention of [`moment_int_plane`](@ref) for
moment-based kinds.
"""
function wedge_saddle_field(model, order::Integer=1;
                            kind::Symbol=:moment,
                            pv_tol::Union{Nothing,Real}=nothing)
    if kind === :moment
        return model.prefactor *
               canonical_pole_pair_value(model.poles, model.coefficients,
                                         order; pv_tol=pv_tol)
    elseif kind === :pole_extraction
        return model.prefactor *
               canonical_pole_extraction(model.poles, model.amplitude_u;
                                         pole_values=model.pole_limits,
                                         coeffs=model.coefficients,
                                         pv_tol=pv_tol)
    elseif kind === :slope_separated
        return model.prefactor *
               canonical_slope_separated(model.poles, model.coefficients)
    elseif kind === :quadrature
        return model.prefactor *
               canonical_pair_quadrature(model.poles, model.amplitude_u)
    end
    throw(ArgumentError("kind must be :moment, :pole_extraction, " *
                        ":slope_separated, or :quadrature"))
end

"""
    wedge_diffracted_model(Kρ, φ, φ_i, N, polarization; order=1, kind,
                           pv_tol)

Diffracted field `s(-2π) - s(0)` assembled from the two canonical saddle
models under one approximation `kind` (see [`wedge_saddle_field`](@ref)).
`pv_tol` selects the symmetric principal-value boundary convention for
moment-based kinds at a mapped pole on the real axis.
"""
function wedge_diffracted_model(Krho::Real, phi::Real, phi_i::Real,
                                N::Real, polarization::Symbol;
                                order::Integer=1, kind::Symbol=:moment,
                                ncoef::Integer=14,
                                pv_tol::Union{Nothing,Real}=nothing)
    work = promote(float(Krho), float(phi), float(phi_i))[1]
    period = 2UTDKernels._typed_pi(work)
    m2 = wedge_canonical_model(Krho, phi, phi_i, N, polarization;
                               saddle=-period, ncoef=ncoef)
    m0 = wedge_canonical_model(Krho, phi, phi_i, N, polarization;
                               saddle=zero(period), ncoef=ncoef)
    return wedge_saddle_field(m2, order; kind=kind, pv_tol=pv_tol) -
           wedge_saddle_field(m0, order; kind=kind, pv_tol=pv_tol)
end

"""
    wedge_term_scale(model, order)

Scale normalization of a saddle canonical model: the sum of retained
canonical term magnitudes `Σ_m |c_m (W_m(z_1) - W_m(z_2))/(z_2 - z_1)|
|prefactor|`, consistent with the paper's scale-normalized error.
"""
function wedge_term_scale(model, order::Integer;
                          pv_tol::Union{Nothing,Real}=nothing)
    dz = model.poles[2] - model.poles[1]
    M = min(order, length(model.coefficients) - 1)
    W1 = moment_int_plane(model.poles[1], M; pv_tol=pv_tol)
    W2 = moment_int_plane(model.poles[2], M; pv_tol=pv_tol)
    s = sum(abs(model.coefficients[m + 1] * (W1[m + 1] - W2[m + 1]) / dz)
            for m in 0:M)
    return s * abs(model.prefactor)
end
