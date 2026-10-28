# Complete Sommerfeld half-plane field and receiver-weighted observables.
#
# The two-branch plane-wave solution for a PEC screen used in the
# manuscript's physical benchmark is
#
#   u_c(x, kappa) = exp(+i kappa cos x) W(sqrt(2 kappa) cos(x/2)),
#   V(s, p, kappa) = u_c(s - p) - u_c(s + p),
#
# where p is the incidence-angle parameter and s = pi - psi is the
# observation coordinate used by the half-plane map in the manuscript. The
# positive carrier exponent is required: with W(q) = erfc(-e^{i pi/4} q)/2
# the residual (Delta + k^2)[exp(-i k r cos x) W(q)] = -2 i k q C W'(q)/r
# does not vanish, whereas the positive carrier solves the Helmholtz
# equation away from the edge and the branch cut. In the lit region
# u_c(s - p) ~ exp(+i kappa cos(s - p)), so the incident-wave carrier is
# u_inc = exp(+i kappa cos(s - p)) and the demodulated field is
# D = V / u_inc = exp(-i kappa cos(s - p)) V.
#
# The receiver window is frozen at the reference incidence angle p0:
# w0(s) = B(s - p0 - pi) does not depend on p. The misfit
#
#   J_2(p; p0) = (1/2) ∫ w0(s) |D(s, p) - 1|^2 ds
#              = (1/2) ∫ w0(s) |V(s, p) - u_inc(s, p)|^2 ds
#
# is a real scattered-field-defect energy; its sensitivity at p = p0 is
#
#   J_2'(p0) = Re ∫ w0(s) conj(D - 1) D_p(s, p0) ds.
#
# The demodulated derivative D_p has a transition (shadow-boundary) part
# -L_kappa built from the sensitivity kernel and a smooth remainder that is
# evaluated by direct quadrature. All formulas were checked against
# independent finite differences of the stored field.

# The parent owns phase-accurate real tails and bounded complex/AD recovery.
_branch_switch(q::Real) = canonical_switch(q)

"""Incident-side cosine Sommerfeld branch `u_c(x, kappa)`."""
function sommerfeld_cosine_branch(x::Real, kappa::Real)
    _validate_finite(x, "x")
    kap = _validate_positive(kappa, "kappa")
    root = sqrt(kap)
    root2 = sqrt(oftype(root, 2))
    q = (root2 * cos(x / 2)) * root
    return UTDKernels._cis_product(kap, cos(x)) * _branch_switch(q)
end

"""
    sommerfeld_cosine_branch_with_derivatives(x, kappa)

Value and analytic partial derivatives `(value, dx, dkappa)` of the same
Sommerfeld branch, for transverse field reconstruction in a 3D edge frame.
"""
function sommerfeld_cosine_branch_with_derivatives(x::Real, kappa::Real)
    _validate_finite(x, "x")
    kap = _validate_positive(kappa, "kappa")
    root = sqrt(kap)
    root2 = sqrt(oftype(root, 2))
    q = (root2 * cos(x / 2)) * root
    carrier = UTDKernels._cis_product(kap, cos(x))
    switch = _branch_switch(q)
    kernel = canonical_kernel(q)
    value = carrier * switch
    dx = carrier * (-im * kap * sin(x) * switch - (root / root2) * sin(x / 2) * kernel)
    dkappa = carrier * (im * cos(x) * switch + (cos(x / 2) / root2) * (kernel / root))
    return (; value, dx, dkappa)
end

function sommerfeld_cosine_branch(x::AbstractArray, kappa::Real)
    _validate_positive(kappa, "kappa")
    return sommerfeld_cosine_branch.(x, Ref(kappa))
end

"""Complete two-branch half-plane field `V(s, p, kappa)`."""
function sommerfeld_field(s::Real, p::Real, kappa::Real; polarization::Symbol=:soft)
    _validate_finite(s, "s")
    _validate_finite(p, "p")
    kap = _validate_positive(kappa, "kappa")
    polarization in (:soft, :hard) || throw(ArgumentError("polarization must be :soft or :hard"))
    sign = polarization === :soft ? -1 : 1
    return sommerfeld_cosine_branch(s - p, kap) + sign *
           sommerfeld_cosine_branch(s + p, kap)
end

"""
    sommerfeld_field_with_derivatives(s, p, kappa; polarization=:soft)

Complete scalar half-plane field and analytic partials `(value, ds, dp,
dkappa)`, using the same branch owner for both PEC polarizations.
"""
function sommerfeld_field_with_derivatives(s::Real, p::Real, kappa::Real;
                                           polarization::Symbol=:soft)
    polarization in (:soft,:hard) || throw(ArgumentError("polarization must be :soft or :hard"))
    minus = sommerfeld_cosine_branch_with_derivatives(s-p,kappa)
    plus = sommerfeld_cosine_branch_with_derivatives(s+p,kappa)
    sign = polarization === :soft ? -1 : 1
    return (; value=minus.value+sign*plus.value,
              ds=minus.dx+sign*plus.dx,
              dp=-minus.dx+sign*plus.dx,
              dkappa=minus.dkappa+sign*plus.dkappa)
end

"""Incident-wave carrier `u_inc(s, p, kappa) = exp(+i kappa cos(s - p))`."""
function sommerfeld_incident(s::Real, p::Real, kappa::Real)
    _validate_finite(s, "s")
    _validate_finite(p, "p")
    kap = _validate_positive(kappa, "kappa")
    return UTDKernels._cis_product(kap, cos(float(s) - float(p)))
end

"""Demodulated complete field `D = exp(-i kappa cos(s - p)) V = V / u_inc`."""
function sommerfeld_demodulated(s::Real, p::Real, kappa::Real)
    return sommerfeld_field(s, p, kappa) * conj(sommerfeld_incident(s, p, kappa))
end

"""
    sommerfeld_demodulated_dp(s, p, kappa)

Derivative `dD/dp` of the demodulated complete field
`D(s, p) = W(sqrt(kappa) g_-(s)) - osc(s) W(sqrt(kappa) g_+(s))` with
`osc = exp(-2 i kappa sin s sin p)`, including the p-dependence of the
demodulating carrier. The transition part is `-L_kappa` with
`L_kappa(s) = sqrt(kappa) v_-(s) K(sqrt(kappa) g_-(s))`,
`g_-(s) = sqrt(2) cos((s - p)/2)`, `v_-(s) = -g_-'(s)`, evaluated on the
minus branch; the remainder is the smooth `osc`-modulated plus-branch
term. Verified against finite differences of `sommerfeld_demodulated`.
"""
function sommerfeld_demodulated_dp(s::Real, p::Real, kappa::Real)
    _validate_finite(s, "s")
    _validate_finite(p, "p")
    sf, pf, kap = promote(float(s), float(p), _validate_positive(kappa, "kappa"))
    root2 = sqrt(oftype(kap, 2))
    gm = root2 * cos((sf - pf) / 2)
    gp = root2 * cos((sf + pf) / 2)
    vm = sin((sf - pf) / 2) / root2
    vp = sin((sf + pf) / 2) / root2
    Lkm = -sqrt(kap) * vm * canonical_kernel(sqrt(kap) * gm)
    Lkp = -sqrt(kap) * vp * canonical_kernel(sqrt(kap) * gp)
    Wplus = _branch_switch(sqrt(kap) * gp)
    osc = UTDKernels._cis_product(kap, -2sin(sf)*sin(pf))
    regular = -im * (kap * ((2sin(sf)*cos(pf))*Wplus)) + Lkp
    return _finite_result(-Lkm - osc * regular, "demodulated Sommerfeld derivative")
end

"""Smooth C-infinity receiver window of radius `radius` about `center`."""
receiver_window(s::Real, center::Real, radius::Real) =
    bump_asymmetric(float(s) - float(center); radius=radius, alpha=0.0)

@inline _receiver_isfinite(value::Number) = UTDKernels._number_isfinite(value)
@inline _receiver_isfinite(value::AbstractArray{<:Number}) = all(UTDKernels._number_isfinite,value)

function _receiver_mean_update(average::Number,value::Number,ratio::Real)
    difference=value-average
    result=average+ratio*difference
    UTDKernels._number_isfinite(difference) && UTDKernels._number_isfinite(result) && return result
    C=typeof(average+value+zero(ratio))
    return UTDKernels._shadow_stable_wide_value(C,4096,"receiver weighted mean") do
        aw,vw,rw=UTDKernels._faddeeva_widen_bigfloat.((average,value,ratio))
        aw+rw*(vw-aw)
    end
end

function _receiver_mean_update(average::AbstractArray{<:Number},value::AbstractArray{<:Number},ratio::Real)
    axes(average)==axes(value) || throw(DimensionMismatch("receiver callback shapes must agree"))
    return map((a,v)->_receiver_mean_update(a,v,ratio),average,value)
end

"""
    receiver_quadrature(fun, lo, hi; npanels, order=12, max_panels=10_000_000)

Composite Gauss--Legendre action for a pure scalar or array-valued receiver
integrand. The accumulator retains the sample's element type and shape,
including ForwardDiff tangents. Panels are fixed while differentiating.
"""
function receiver_quadrature(integrand, lo::Real, hi::Real;
                              npanels::Integer, order::Integer=12,
                              max_panels::Integer=DEFAULT_MAX_PANELS)
    isfinite(lo) && isfinite(hi) && lo < hi || throw(ArgumentError("receiver interval requires finite lo < hi"))
    1 <= npanels <= max_panels || throw(ArgumentError("receiver panel count must lie in 1:max_panels"))
    nodes,weights=UTDKernels._cached_gauss_legendre_rule(order)
    sample=integrand(lo/2+hi/2)
    _receiver_isfinite(sample) || throw(DomainError(sample,"receiver integrand must be finite"))
    total=zero(sample)
    for panel in 1:npanels
        fraction_left=(panel-1)/npanels
        fraction_right=panel/npanels
        left=(1-fraction_left)*lo+fraction_left*hi
        right=panel==npanels ? hi : (1-fraction_right)*lo+fraction_right*hi
        mid=left/2+right/2
        halfwidth=right/2-left/2
        average=sample
        weight_total=zero(first(weights))
        for j in eachindex(nodes)
            value=integrand(mid+halfwidth*nodes[j])
            _receiver_isfinite(value) || throw(DomainError(value,"receiver integrand must be finite"))
            next_weight=weight_total+weights[j]
            average=_receiver_mean_update(average,value,weights[j]/next_weight)
            weight_total=next_weight
        end
        # Accumulate the weighted mean before applying a tiny interval width:
        # scaling every node separately can discard a representable integral.
        width=right-left
        contribution=isfinite(width) ? (width*average)*(weight_total/2) :
            (halfwidth*average)*weight_total
        total+=contribution
    end
    _receiver_isfinite(total) || throw(DomainError(total,"receiver integral exceeds the active numeric range"))
    return total
end

_gl_panels(integrand,lo::Real,hi::Real,npanels::Integer,order::Integer) =
    receiver_quadrature(integrand,lo,hi;npanels=npanels,order=order)

function _check_window(center::Real, radius::Real)
    _validate_finite(center, "center")
    R = _validate_positive(radius, "radius")
    return R
end

# ---- remainder (non-transition) part of D_p ----
# Integrand of the smooth remainder multiplied by an optional factor.
function _remainder_integral(kappa, p, center, radius, factor, npanels)
    kap = _validate_positive(kappa, "kappa")
    pf = float(p)
    lo = float(center) - radius
    hi = float(center) + radius
    gp(s) = sqrt(2.0) * cos((s + pf) / 2)
    vp(s) = sin((s + pf) / 2) / sqrt(2.0)
    return _gl_panels(lo, hi, npanels, 12) do s
        Lkp = -sqrt(kap) * vp(s) * canonical_kernel(sqrt(kap) * gp(s))
        osc = UTDKernels._cis_product(kap, -2sin(s)*sin(pf))
        inner = -im * (kap * ((2sin(s)*cos(pf)) *
                _branch_switch(sqrt(kap) * gp(s)))) + Lkp
        base = receiver_window(s, center, radius) * (-osc) * inner
        factor === nothing ? base : base * factor(s)
    end
end

"""
    default_receiver_panels(kappa)

Panel count of the canonical receiver policy, for positive finite `kappa`.
Explicit receiver quadrature still enforces its declared `max_panels` limit.
"""
function default_receiver_panels(kappa::Real)
    kap=_validate_positive(kappa,"kappa")
    estimate=20sqrt(kap)
    estimate<typemax(Int)÷20 || throw(ArgumentError("receiver panel count exceeds the representable Int range"))
    return max(1600,ceil(Int,estimate)*20)
end

"""
    field_sensitivity_exact(kappa, p, center, radius; npanels=...)

Exact receiver-weighted sensitivity `S_1 = ∫ w(s) D_p(s, p) ds` of the
demodulated complete field, evaluated by adaptive composite
Gauss--Legendre quadrature on the window `[center-radius, center+radius]`.
"""
function field_sensitivity_exact(kappa::Real, p::Real, center::Real,
                                 radius::Real;
                                 npanels::Integer=default_receiver_panels(kappa))
    kap = _validate_positive(kappa, "kappa")
    R = _check_window(center, radius)
    npanels >= 8 || throw(ArgumentError("npanels must be at least 8"))
    return _gl_panels(float(center) - R, float(center) + R, npanels, 12) do s
        receiver_window(s, center, R) * sommerfeld_demodulated_dp(s, p, kap)
    end
end

"""
    misfit_sensitivity_exact(kappa, p, center, radius; npanels=...)

Exact real observable `J_2'(p) = Re ∫ w conj(D - 1) D_p ds`, the
sensitivity of the receiver-weighted scattered-field misfit
`(1/2) ∫ w |V - u_inc|^2 ds`, evaluated by composite Gauss--Legendre
quadrature.
"""
function misfit_sensitivity_exact(kappa::Real, p::Real, center::Real,
                                  radius::Real;
                                  npanels::Integer=default_receiver_panels(kappa))
    kap = _validate_positive(kappa, "kappa")
    R = _check_window(center, radius)
    npanels >= 8 || throw(ArgumentError("npanels must be at least 8"))
    value = _gl_panels(float(center) - R, float(center) + R, npanels, 12) do s
        receiver_window(s, center, R) *
        conj(sommerfeld_demodulated(s, p, kap) - 1) *
        sommerfeld_demodulated_dp(s, p, kap)
    end
    return real(value)
end

"""
    field_sensitivity_remainder / misfit_sensitivity_remainder

Direct quadrature of the smooth remainder part of the two observables
(the `osc`-modulated second branch of `D_p`). The transition action plus
this remainder equals the exact observable.
"""
function field_sensitivity_remainder(kappa::Real, p::Real, center::Real,
                                     radius::Real;
                                     npanels::Integer=default_receiver_panels(kappa))
    R = _check_window(center, radius)
    npanels >= 8 || throw(ArgumentError("npanels must be at least 8"))
    return _remainder_integral(kappa, p, center, R, nothing, npanels)
end

function misfit_sensitivity_remainder(kappa::Real, p::Real, center::Real,
                                      radius::Real;
                                      npanels::Integer=default_receiver_panels(kappa))
    kap = _validate_positive(kappa, "kappa")
    R = _check_window(center, radius)
    npanels >= 8 || throw(ArgumentError("npanels must be at least 8"))
    return _remainder_integral(kappa, p, center, R,
                               s -> conj(sommerfeld_demodulated(s, p, kap) - 1),
                               npanels)
end

# ---- canonical transition coordinate for the minus branch ----
# Shadow boundary at s* = p + pi with g_-(s) = sqrt(2) cos((s - p)/2):
# y = g_-(s), h(y) = p + 2 acos(y / sqrt(2)), h'(0) = -sqrt(2) (decreasing),
# so <L_kappa, F> = -<K_kappa, F o h> and the transition part of S_1 or J_2'
# equals +<K_kappa, F o h>.
function sommerfeld_inverse_map(y::Real, p::Real)
    _validate_finite(y, "y")
    _validate_finite(p, "p")
    yf, pf = promote(float(y), float(p))
    normalized = yf / sqrt(oftype(yf, 2))
    abs(normalized) <= 1 || throw(DomainError(y,
        "Sommerfeld inverse coordinate requires abs(y) <= sqrt(2)"))
    return pf + 2acos(normalized)
end

function _transition_action(kappa, p, center, radius, kernel_density,
                            testfun, npanels)
    kap = _validate_positive(kappa, "kappa")
    R = _check_window(center, radius)
    npanels >= 8 || throw(ArgumentError("npanels must be at least 8"))
    _validate_finite(p, "p")
    pf = float(p)
    # Integrate the signed coordinate Jacobian on the actual receiver.
    # A single acos inverse reflects windows outside its principal branch
    # and can silently omit their transition contribution.
    return _gl_panels(float(center)-R, float(center)+R, npanels, 12) do s
        angle = (s-pf)/2
        g = sqrt(2.0)*cos(angle)
        velocity = sin(angle)/sqrt(2.0)
        testfun(s)*sqrt(kap)*velocity*kernel_density(sqrt(kap)*g)
    end
end

"""
    field_transition_action(kappa, p, center, radius; kernel=:canonical)

Signed transition action on the actual receiver window. On the decreasing
coordinate branch this equals `<K_kappa, w o h>`. The same signed Jacobian
is used with the real Gaussian `rho(y) = exp(-y^2)/sqrt(pi)` when
`kernel=:gaussian`.
"""
function field_transition_action(kappa::Real, p::Real, center::Real,
                                 radius::Real;
                                 kernel::Symbol=:canonical,
                                 npanels::Integer=default_receiver_panels(kappa))
    density = kernel === :canonical ? canonical_kernel :
              kernel === :gaussian ? (y -> exp(-y^2) / sqrt(pi)) :
              throw(ArgumentError("kernel must be :canonical or :gaussian"))
    return _transition_action(kappa, p, center, radius, density,
                              s -> receiver_window(s, center, radius), npanels)
end

"""
    misfit_transition_action(kappa, p, center, radius; kernel=:canonical)

Signed transition action on the complete receiver test
`F_2(s) = w(s) conj(D(s, p) - 1)`, or the real-Gaussian counterpart.
On a single decreasing coordinate branch this equals `<K_kappa, F_2 o h>`.
"""
function misfit_transition_action(kappa::Real, p::Real, center::Real,
                                  radius::Real;
                                  kernel::Symbol=:canonical,
                                  npanels::Integer=default_receiver_panels(kappa))
    kap = _validate_positive(kappa, "kappa")
    density = kernel === :canonical ? canonical_kernel :
              kernel === :gaussian ? (y -> exp(-y^2) / sqrt(pi)) :
              throw(ArgumentError("kernel must be :canonical or :gaussian"))
    return _transition_action(kappa, p, center, radius, density,
                              s -> receiver_window(s, center, radius) *
                                   conj(sommerfeld_demodulated(s, p, kap) - 1),
                              npanels)
end

# ---- local jets of the effective tests at the transition point ----
# The transition point of the minus branch is sstar = p + pi, where
# g_-(sstar) = 0. The receiver window center may differ; the Dirac
# truncation is always evaluated at the boundary point.

"""Transition (shadow-boundary) coordinate of the minus branch."""
sommerfeld_boundary(p::Real) = float(p) + π

# Analytic derivatives of the receiver window w(s) = exp(1 - 1/(1-x^2)),
# x = (s - center)/radius, at an arbitrary evaluation point. The exponent
# phi(x) = -(1-x^2)^{-1} has series derivatives
# phi(x) = -(1-x^2)^(-1); its exact rational Taylor recurrence is
# evaluated below. Window derivatives follow from Faa di Bruno.
function _bump_exponent_jets(x::Real, rmax::Integer)
    abs(x) < 1 || return nothing
    # Taylor coefficients of 1/(1-(x+h)^2) follow an exact three-term
    # recurrence. Unlike a truncated geometric series this remains accurate
    # close to the compact window boundary.
    inverse = [inv(1 - x*x)]
    for n in 1:rmax
        previous2 = n == 1 ? zero(x) : inverse[n-1]
        push!(inverse, (2x*inverse[n] + previous2) / (1-x*x))
    end
    return [-prod(one(x) * j for j in 1:n; init=one(x))*inverse[n+1] for n in 0:rmax]
end

"""
    bump_jets(s_eval, center, radius, rmax)

Derivatives `w^{(r)}(s_eval)`, `r = 0..rmax`, of the smooth compact bump
`w(s) = exp(1 - 1/(1 - ((s-center)/radius)^2))`. Zeros outside the
support; Faa di Bruno evaluation inside.
"""
function bump_jets(s_eval::Real, center::Real, radius::Real, rmax::Integer)
    R = _validate_positive(radius, "radius")
    rmax >= 0 || throw(ArgumentError("rmax must be nonnegative"))
    all(isfinite, (s_eval, center)) || throw(ArgumentError("window coordinates must be finite"))
    x = (float(s_eval) - float(center)) / R
    abs(x) >= 1 && return zeros(typeof(x), rmax + 1)
    phid = _bump_exponent_jets(x, rmax)
    w0 = exp(1 + phid[1])
    out = [w0]
    for r in 1:rmax
        # (e^phi)^{(r)} = e^phi * sum_j B_{r,j}(phi')
        val = 0.0
        for j in 1:r
            val += _bell_partial(r, j, view(phid, 2:length(phid)))
        end
        push!(out, w0 * val)
    end
    result = out ./ R .^ (0:rmax)
    all(UTDKernels._number_isfinite, result) || throw(DomainError(result,"window jets exceed the active numeric range"))
    return result
end

# Partial (exponential) Bell polynomial B_{n,j}(x_1, ..., x_{n-j+1}).
function _bell_partial(n::Integer, j::Integer, x)
    n == j == 0 && return 1.0
    (n <= 0 || j <= 0 || n < j) && return 0.0
    total = 0.0
    for i in 0:(n - j)
        total += binomial(n - 1, i) * x[i + 1] * _bell_partial(n - 1 - i, j - 1, x)
    end
    return total
end

# Switch jet W^{(j)}(0): W'(q) = K(q), K^{(n)}(q) has even part at 0.
function _switch_jet(j::Integer)
    j == 0 && return 0.5 + 0im
    isodd(j) || return 0.0 + 0im
    half = div(j - 1, 2)
    return exp(im * π / 4) / sqrt(π) * factorial(j - 1) * (-im)^half /
           factorial(half)
end

# Jets W^{(j)}(q), j = 0..rmax, at an arbitrary real argument.  W' = K and
# K' = -2 i q K, so K^{(n+1)} = -2i (q K^{(n)} + n K^{(n-1)}): a two-term
# recurrence seeded by _branch_switch and canonical_kernel.  At q = 0 the
# values coincide with _switch_jet.
function _switch_jets_at(rmax::Integer, q::Real)
    out = [_branch_switch(q)]
    rmax >= 1 || return out
    knm1 = canonical_kernel(q)
    push!(out, knm1)
    rmax >= 2 || return out
    kn = -2im * q * knm1
    push!(out, kn)
    n = 1
    while length(out) <= rmax
        knp1 = -2im * (q * kn + n * knm1)
        push!(out, knp1)
        knm1, kn = kn, knp1
        n += 1
    end
    return out
end

"""
    sommerfeld_branch_jets(rmax, kappa, branch_sign, p, sstar)

Derivatives `d^r/ds^r W(sqrt(kappa) g_b(s))` at `sstar`, where
`g_b(s) = sqrt(2) cos((s + b p)/2)` and `branch_sign` is `-1` (transition
branch) or `+1` (remainder branch). Uses Faa di Bruno with partial Bell
polynomials and the switch jets at `sqrt(kappa) g_b(sstar)` (the plus
branch is not expanded at zero); verified against high-order finite
differences.
"""
function sommerfeld_branch_jets(rmax::Integer, kappa::Real,
                                branch_sign::Integer, p::Real, sstar::Real)
    rmax >= 0 || throw(ArgumentError("rmax must be nonnegative"))
    kap = _validate_positive(kappa, "kappa")
    abs(branch_sign) == 1 ||
        throw(ArgumentError("branch_sign must be +1 or -1"))
    q0 = sqrt(kap) * sqrt(2.0) *
         cos((float(sstar) + branch_sign * float(p)) / 2)
    wj = _switch_jets_at(rmax, float(q0))
    gv = [sqrt(2.0) * (0.5)^n *
          cos((float(sstar) + branch_sign * float(p)) / 2 + n * π / 2)
          for n in 1:max(rmax, 1)]
    out = [wj[1]]
    for r in 1:rmax
        val = 0.0 + 0im
        for j in 1:r
            val += wj[j + 1] * kap^(j / 2) * _bell_partial(r, j, gv)
        end
        push!(out, val)
    end
    return out
end

# Jets of osc(s) = exp(-2 i kappa sin s sin p) at sstar (Faa di Bruno).
function _osc_jets(rmax, kappa, p, sstar)
    a = -im * (kappa * (2sin(float(p))))
    x = [a * sin(float(sstar) + n * π / 2) for n in 1:max(rmax, 1)]
    osc0 = UTDKernels._cis_product(kappa, -2sin(float(sstar))*sin(float(p)))
    out = [osc0]
    for r in 1:rmax
        val = 0.0 + 0im
        for j in 1:r
            val += _bell_partial(r, j, x)
        end
        push!(out, osc0 * val)
    end
    return out
end

# Receiver-window jets at an arbitrary evaluation point.
_window_jets(s_eval, center, radius) =
    bump_jets(s_eval, center, radius, 4)

"""
    misfit_local_jets(kappa, p, center, radius)

Derivatives `F_2^{(r)}(sstar)`, `r = 0..4`, of the complete effective
test `F_2(s) = w(s) conj(D(s) - 1)` with
`D(s) = W_c(s) - osc(s) W_+(s)`, evaluated at the minus-branch boundary
point `sstar = p + pi`. The receiver window is centered at `center`.
Computed by exact Leibniz/Faa di Bruno algebra. These jets carry the
kappa-dependent frequencies that make a fixed truncation nonuniform.
"""
function misfit_local_jets(kappa::Real, p::Real, center::Real, radius::Real)
    kap = _validate_positive(kappa, "kappa")
    R = _check_window(center, radius)
    _validate_finite(p,"p")
    sstar = sommerfeld_boundary(p)
    pf = float(p)
    wd = _window_jets(sstar, center, R)
    C=typeof(complex(first(wd)+zero(kap)+zero(pf)))
    # Outside the compact receiver support the effective test is identically
    # zero, even when separated field jets would exceed the numeric range.
    all(UTDKernels._scalar_exact_zero,wd) && return fill(zero(C),length(wd))
    # At zero bearing the two Sommerfeld branches coincide and osc(s)==1,
    # hence D(s)==0 exactly. Preserve -window jets before separately growing
    # kappa powers can create Inf-Inf. A seeded p tape retains the general path.
    UTDKernels._scalar_exact_zero(pf) && return [-complex(v+zero(kap)+zero(pf)) for v in wd]
    Wm = sommerfeld_branch_jets(4, kap, -1, pf, sstar)
    Wp = sommerfeld_branch_jets(4, kap, +1, pf, sstar)
    os = _osc_jets(4, kap, pf, sstar)
    # D^{(m)} = W_c^{(m)} - (osc W_+)^{(m)}
    Dd = typeof(first(Wm) + first(Wp) + first(os))[]
    for m in 0:4
        conv = 0.0 + 0im
        for l in 0:m
            conv += binomial(m, l) * os[m - l + 1] * Wp[l + 1]
        end
        push!(Dd, Wm[m + 1] - conv)
    end
    Fd = typeof(first(Dd) * first(wd))[]
    for r in 0:4
        val = 0.0 + 0im
        for j in 0:r
            m = r - j
            conjterm = m == 0 ? conj(Dd[1] - 1) : conj(Dd[m + 1])
            val += binomial(r, j) * wd[j + 1] * conjterm
        end
        push!(Fd, val)
    end
    return [_finite_result(value,"misfit test jet") for value in Fd]
end

"""
    window_composed_jets(p, center, radius)

Derivatives `(w o h)^{(0)},(w o h)''',(w o h)''''` of the receiver window
composed with the inverse canonical map at `y = 0`. With
`h(y) = p + 2 acos(y/sqrt(2))`, `h'(0) = -sqrt(2)` and `h'''(0) = -1/sqrt(2)`,
so `(w o h)'' = 2 w''(sstar)` and `(w o h)'''' = 4 w''''(sstar) + 4 w''(sstar)`
with `sstar = p + pi`.
"""
function window_composed_jets(p::Real, center::Real, radius::Real)
    sstar = sommerfeld_boundary(p)
    wd = _window_jets(sstar, center, radius)
    return (f0=wd[1], d2=2 * wd[3], d4=4 * wd[5] + 4 * wd[3])
end

# Truncated canonical action <K, F o h> = sum_m (-i)^m (Foh)^{(2m)}(0)/(4^m m! k^m)
function _truncated_canonical(f0, d2, d4, kappa, order_M)
    1 <= order_M <= 3 || throw(ArgumentError("local jets support order_M in 1:3"))
    result = complex(f0)
    order_M >= 2 && (result += (-im / 4) * (d2 / kappa))
    order_M >= 3 && (result -= ((d4 / kappa) / kappa) / 32)
    return _finite_result(result, "truncated canonical action")
end

"""
    field_truncated_action(kappa, p, center, radius, order_M; remainder)

Truncated transition action for the demodulated field sensitivity: the
`order_M`-th truncation of `<K_kappa, w o h>` plus the smooth remainder
when `remainder` is supplied. `order_M = 1` is the Dirac limit.
"""
function field_truncated_action(kappa::Real, p::Real, center::Real,
                                radius::Real, order_M::Integer;
                                remainder::Number=0.0)
    kap = _validate_positive(kappa, "kappa")
    _validate_finite(remainder, "remainder")
    jets = window_composed_jets(p, center, radius)
    return _truncated_canonical(jets.f0, jets.d2, jets.d4, kap, order_M) +
           remainder
end

"""
    misfit_truncated_action(kappa, p, center, radius, order_M; remainder)

Truncated transition action for the misfit sensitivity: the `order_M`-th
truncation of `<K_kappa, F_2 o h>` using the complete local jets of `F_2`
at the boundary point `sstar = p + pi`, plus the smooth remainder when
`remainder` is supplied.
"""
function misfit_truncated_action(kappa::Real, p::Real, center::Real,
                                 radius::Real, order_M::Integer;
                                 remainder::Number=0.0)
    kap = _validate_positive(kappa, "kappa")
    _validate_finite(remainder, "remainder")
    Fd = misfit_local_jets(kap, p, center, radius)
    Foh2 = 2 * Fd[3]
    Foh4 = 4 * Fd[5] + 4 * Fd[3]
    return real(_truncated_canonical(Fd[1], Foh2, Foh4, kap, order_M) +
                remainder)
end

# ---- carrier-restoration counterexample -------------------------------
# On the composed canonical test F(y) = f(h(y)) exp(i kappa y^2) exp(-i kappa)
# the order-M truncation yields exp(-i kappa) * sum_{m<M} binom(2m,m)/4^m,
# i.e. the O(1) constants 1, 3/2, 15/8, while the exact carrier-multiplied
# action is exp(i pi/4 - i kappa) sqrt(kappa/pi) * integral f(h(y)) dy.

"""Exact carrier-multiplied transition action for a unit canonical window
`f(h(y)) = 1` on `|y| <= yhalf`: `exp(i pi/4 - i kappa) sqrt(kappa/pi) * 2
yhalf`, proportional to `sqrt(kappa)`."""
function carrier_action_exact(kappa::Real, yhalf::Real)
    kap = _validate_positive(kappa, "kappa")
    yh = _validate_positive(yhalf, "yhalf")
    kap, yh = promote(kap, yh)
    # Keep the quarter-cycle phase separate: pi/4-kappa loses that phase
    # for large kappa. Take the square root before division near zero.
    phase = UTDKernels._shadow_quarter_phase(kap) * exp(-im * kap)
    scale = sqrt(kap) * (2 / sqrt(UTDKernels._typed_pi(kap)))
    if !any(UTDKernels._number_contains_ad, (kap, yh))
        mantissa, exponent = frexp(scale)
        scaled = UTDKernels._scaled_field_product(phase, yh, one(phase), mantissa, exponent)
        scaled === nothing || return _finite_result(scaled, "carrier action")
    end
    return _finite_result(phase * (scale * yh), "carrier action")
end

"""Order-`order_M` truncation of the carrier-restored canonical action:
`exp(-i kappa) * sum_{m<order_M} binomial(2m, m) / 4^m` (1, 3/2, 15/8),
independent of kappa."""
function carrier_action_truncated(kappa::Real, order_M::Integer)
    kap = _validate_positive(kappa, "kappa")
    1 <= order_M <= typemax(Int) || throw(ArgumentError("order_M must be a positive representable Int"))
    # binomial(2m,m)/4^m has the ratio (2m-1)/(2m). The recurrence
    # avoids native-integer binomial overflow while each weight is <= 1.
    weight = one(kap)
    total = weight
    for m in 1:(Int(order_M) - 1)
        weight *= 1 - inv(2oftype(weight, m))
        total += weight
    end
    return exp(-im * kap) * total
end
