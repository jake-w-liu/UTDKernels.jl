"""
Endpoint-uniform evaluation of a smooth finite straight-edge propagation
integral.

The scalar/componentwise model is

```
I = ∫ₐᵇ A(s) exp(-im*k*Φ(s)) ds,
Φ(s) = R_s(s) + R_p(s),
```

under the package's `exp(+im*ω*t)` phasor convention. The transformed local
amplitude is retained through quadratic order. Singular diffraction
coefficients, curved edges, vertices, and multiple stationary points require a
different canonical model.
"""

using SpecialFunctions: erfcx

const _FINITE_EDGE_CANCELLATION_THRESHOLD = 1.0e-5

@inline function _finite_edge_validate_coordinate(value::Real, name::AbstractString)
    primal = _primal_value(value)
    isfinite(primal) || throw(ArgumentError("$name must be finite"))
    return value
end

@inline function _finite_edge_validate_positive(value::Real, name::AbstractString)
    primal = _primal_value(value)
    isfinite(primal) && primal > zero(primal) ||
        throw(ArgumentError("$name must be finite and positive"))
    return value
end

@inline function _finite_edge_validate_wavenumber(k::Real)
    primal = _primal_value(k)
    isfinite(primal) && primal > zero(primal) ||
        throw(ArgumentError("k must be finite and positive"))
    return k
end

@inline function _finite_edge_checked(value::Number, context::AbstractString)
    _number_isfinite(value) || throw(DomainError(value, "$context is non-finite"))
    return value
end

@inline function _finite_edge_pi(value::Real)
    primal_type = typeof(float(_primal_value(value)))
    return zero(value) + convert(primal_type, π)
end

"""
    FiniteEdgeGeometry(rho_s, rho_p, z_s, z_p)

Source and observer geometry relative to a straight edge axis. `rho_s` and
`rho_p` are positive transverse ranges; `z_s` and `z_p` are finite axial
coordinates. All quantities have units of length.
"""
struct FiniteEdgeGeometry{T<:Real}
    rho_s::T
    rho_p::T
    z_s::T
    z_p::T

    function FiniteEdgeGeometry(
        rho_s::Real,
        rho_p::Real,
        z_s::Real,
        z_p::Real,
    )
        _finite_edge_validate_positive(rho_s, "source transverse range rho_s")
        _finite_edge_validate_positive(rho_p, "observer transverse range rho_p")
        _finite_edge_validate_coordinate(z_s, "source axial coordinate z_s")
        _finite_edge_validate_coordinate(z_p, "observer axial coordinate z_p")
        promoted = promote(float(rho_s), float(rho_p), float(z_s), float(z_p))
        T = typeof(promoted[1])
        return new{T}(promoted...)
    end
end

"""Local derivatives of the exact propagation phase at its stationary point."""
struct FiniteEdgePhaseData{T<:Real}
    q::T
    c::T
    s0::T
    phi0::T
    phi2::T
    phi3::T
    phi4::T
end

"""
    FiniteEdgeAmplitude(A0, A1, A2)

Value and first two axial derivatives of the smooth physical amplitude at the
propagation stationary point.
"""
struct FiniteEdgeAmplitude{T<:Number}
    A0::T
    A1::T
    A2::T

    function FiniteEdgeAmplitude(A0::Number, A1::Number, A2::Number)
        _number_isfinite(A0) || throw(ArgumentError("finite-edge amplitude A0 must be finite"))
        _number_isfinite(A1) || throw(ArgumentError("finite-edge amplitude A1 must be finite"))
        _number_isfinite(A2) || throw(ArgumentError("finite-edge amplitude A2 must be finite"))
        promoted = promote(A0, A1, A2)
        T = typeof(promoted[1])
        return new{T}(promoted...)
    end
end

"""Inverse-phase coefficients and quadratic transformed-amplitude data."""
struct FiniteEdgeTransformData{T<:Real,C<:Number}
    alpha::T
    beta::T
    gamma::T
    G0::C
    G1::C
    G2::C
end

@inline function _finite_edge_length_scale(geometry::FiniteEdgeGeometry)
    return max(
        abs(_primal_value(geometry.rho_s)),
        abs(_primal_value(geometry.rho_p)),
        abs(_primal_value(geometry.z_p - geometry.z_s)),
    )
end

"""
    finite_edge_distances(s, geometry) -> (R_s, R_p)

Return the source-to-edge and edge-to-observer ranges at axial coordinate `s`.
"""
function finite_edge_distances(s::Real, geometry::FiniteEdgeGeometry)
    _finite_edge_validate_coordinate(s, "edge coordinate s")
    rs = hypot(geometry.rho_s, s - geometry.z_s)
    rp = hypot(geometry.rho_p, s - geometry.z_p)
    _finite_edge_checked(rs, "source-to-edge range")
    _finite_edge_checked(rp, "edge-to-observer range")
    return rs, rp
end

function finite_edge_distances(s::AbstractArray{<:Real}, geometry::FiniteEdgeGeometry)
    all(value -> isfinite(_primal_value(value)), s) ||
        throw(DomainError(s, "edge coordinates s must be finite"))
    rs = hypot.(geometry.rho_s, s .- geometry.z_s)
    rp = hypot.(geometry.rho_p, s .- geometry.z_p)
    all(_number_isfinite, rs) || throw(DomainError(s, "source-to-edge ranges are non-finite"))
    all(_number_isfinite, rp) || throw(DomainError(s, "edge-to-observer ranges are non-finite"))
    return rs, rp
end

"""Exact propagation phase `Φ(s) = R_s(s) + R_p(s)`."""
function finite_edge_phase(s, geometry::FiniteEdgeGeometry)
    rs, rp = finite_edge_distances(s, geometry)
    value = rs .+ rp
    if value isa Number
        return _finite_edge_checked(value, "finite-edge phase")
    end
    all(_number_isfinite, value) || throw(DomainError(s, "finite-edge phase is non-finite"))
    return value
end

"""Axial derivative `dΦ/ds` of the exact propagation phase."""
function finite_edge_phase_derivative(s::Real, geometry::FiniteEdgeGeometry)
    rs, rp = finite_edge_distances(s, geometry)
    value = (s - geometry.z_s) / rs + (s - geometry.z_p) / rp
    return _finite_edge_checked(value, "finite-edge phase derivative")
end

function finite_edge_phase_derivative(
    s::AbstractArray{<:Real},
    geometry::FiniteEdgeGeometry,
)
    rs, rp = finite_edge_distances(s, geometry)
    value = (s .- geometry.z_s) ./ rs .+ (s .- geometry.z_p) ./ rp
    all(_number_isfinite, value) ||
        throw(DomainError(s, "finite-edge phase derivatives are non-finite"))
    return value
end

"""Partial derivative of `Φ(s)` with respect to `z_p` at fixed `s`."""
function finite_edge_phase_observer_derivative(
    s::Real,
    geometry::FiniteEdgeGeometry,
)
    _, rp = finite_edge_distances(s, geometry)
    value = (geometry.z_p - s) / rp
    return _finite_edge_checked(value, "observer-coordinate phase derivative")
end

function finite_edge_phase_observer_derivative(
    s::AbstractArray{<:Real},
    geometry::FiniteEdgeGeometry,
)
    _, rp = finite_edge_distances(s, geometry)
    value = (geometry.z_p .- s) ./ rp
    all(_number_isfinite, value) ||
        throw(DomainError(s, "observer-coordinate phase derivatives are non-finite"))
    return value
end

"""
    finite_edge_phase_data(geometry) -> FiniteEdgePhaseData

Return the stationary coordinate, stationary phase, and phase derivatives
through fourth order.
"""
function finite_edge_phase_data(geometry::FiniteEdgeGeometry{T}) where {T}
    rho_s = geometry.rho_s
    rho_p = geometry.rho_p
    z_s = geometry.z_s
    z_p = geometry.z_p
    rho_sum = rho_s + rho_p
    _finite_edge_checked(rho_sum, "sum of finite-edge transverse ranges")
    dz = z_p - z_s
    _finite_edge_checked(dz, "finite-edge axial separation")
    q = dz / rho_sum
    c = hypot(one(T), q)
    s0 = z_s + (rho_s / rho_sum) * dz
    phi0 = rho_sum * c
    phi2 = (inv(rho_s) + inv(rho_p)) * c^(-3)
    phi3 = 3 * q * (rho_p^(-2) - rho_s^(-2)) * c^(-5)
    phi4 = 3 * (4 * q * q - one(T)) *
           (rho_s^(-3) + rho_p^(-3)) * c^(-7)
    _finite_edge_checked(q, "finite-edge phase datum q")
    _finite_edge_checked(c, "finite-edge phase datum c")
    _finite_edge_checked(s0, "finite-edge phase datum s0")
    _finite_edge_checked(phi0, "finite-edge phase datum phi0")
    _finite_edge_checked(phi2, "finite-edge phase datum phi2")
    _finite_edge_checked(phi3, "finite-edge phase datum phi3")
    _finite_edge_checked(phi4, "finite-edge phase datum phi4")
    _primal_value(phi2) > zero(_primal_value(phi2)) ||
        throw(DomainError(phi2, "finite-edge stationary point must be nondegenerate"))
    return FiniteEdgePhaseData{T}(q, c, s0, phi0, phi2, phi3, phi4)
end

@inline function _finite_edge_inverse_phase_coefficients(data::FiniteEdgePhaseData)
    phi2 = data.phi2
    alpha = sqrt(2 / phi2)
    beta = -data.phi3 / (3 * phi2^2)
    gamma = sqrt(2 * one(phi2)) *
            (5 * data.phi3^2 - 3 * phi2 * data.phi4) /
            (36 * phi2^3 * sqrt(phi2))
    return alpha, beta, gamma
end

"""
    finite_edge_transform_data(data, amplitude) -> FiniteEdgeTransformData

Transform physical amplitude derivatives to the exact quadratic phase
coordinate.
"""
function finite_edge_transform_data(
    data::FiniteEdgePhaseData,
    amplitude::FiniteEdgeAmplitude,
)
    alpha, beta, gamma = _finite_edge_inverse_phase_coefficients(data)
    G0 = amplitude.A0 * alpha
    G1 = amplitude.A1 * alpha^2 + 2 * amplitude.A0 * beta
    G2 = (one(alpha) / 2) * amplitude.A2 * alpha^3 +
         3 * amplitude.A1 * alpha * beta + 3 * amplitude.A0 * gamma
    _finite_edge_checked(G0, "finite-edge transformed amplitude G0")
    _finite_edge_checked(G1, "finite-edge transformed amplitude G1")
    _finite_edge_checked(G2, "finite-edge transformed amplitude G2")
    return FiniteEdgeTransformData(alpha, beta, gamma, G0, G1, G2)
end

function _finite_edge_phase_increment_factor(
    s::Real,
    geometry::FiniteEdgeGeometry,
    data::FiniteEdgePhaseData,
)
    x = s - data.s0
    rs, rp = finite_edge_distances(s, geometry)
    rs0, rp0 = finite_edge_distances(data.s0, geometry)
    ys = data.s0 - geometry.z_s
    yp = data.s0 - geometry.z_p
    ds = rs + rs0
    dp = rp + rp0
    return inv(ds) - ys * (2ys + x) / (rs0 * ds^2) +
           inv(dp) - yp * (2yp + x) / (rp0 * dp^2)
end

function _finite_edge_phase_slope_factor(
    s::Real,
    geometry::FiniteEdgeGeometry,
    data::FiniteEdgePhaseData,
)
    x = s - data.s0
    rs, rp = finite_edge_distances(s, geometry)
    rs0, rp0 = finite_edge_distances(data.s0, geometry)
    ys = data.s0 - geometry.z_s
    yp = data.s0 - geometry.z_p
    ds = rs + rs0
    dp = rp + rp0
    return (rs0 - ys * (2ys + x) / ds) / (rs * rs0) +
           (rp0 - yp * (2yp + x) / dp) / (rp * rp0)
end

@inline function _finite_edge_phase_coordinate(
    s::Real,
    geometry::FiniteEdgeGeometry,
    data::FiniteEdgePhaseData,
    threshold_primal::Real,
)
    x = s - data.s0
    if abs(_primal_value(x)) <= threshold_primal * _finite_edge_length_scale(geometry)
        iszero(x) && return zero(x)
        factor = _finite_edge_phase_increment_factor(s, geometry, data)
        factor_primal = _primal_value(factor)
        factor_primal > zero(factor_primal) || throw(DomainError(
            (s, geometry),
            "rationalized phase-increment factor must be positive away from s0",
        ))
        return x * sqrt(factor)
    end

    delta = finite_edge_phase(s, geometry) - data.phi0
    delta_eval = _primal_value(delta) < zero(_primal_value(delta)) ? zero(delta) : delta
    root = sqrt(delta_eval)
    return _primal_value(x) < zero(_primal_value(x)) ? -root : root
end

"""
    finite_edge_phase_coordinate(s, geometry; cancellation_threshold=1e-5)

Return signed `t` satisfying `Φ(s)=Φ₀+t²`. Near the stationary point, an
exact rationalized identity avoids subtracting nearly equal phases.
"""
function finite_edge_phase_coordinate(
    s::Real,
    geometry::FiniteEdgeGeometry;
    cancellation_threshold::Real=_FINITE_EDGE_CANCELLATION_THRESHOLD,
)
    _finite_edge_validate_coordinate(s, "edge coordinate s")
    threshold_primal = _primal_value(cancellation_threshold)
    isfinite(threshold_primal) && threshold_primal >= zero(threshold_primal) ||
        throw(ArgumentError("cancellation_threshold must be finite and nonnegative"))
    data = finite_edge_phase_data(geometry)
    return _finite_edge_phase_coordinate(s, geometry, data, threshold_primal)
end

function finite_edge_phase_coordinate(
    s::AbstractArray{<:Real},
    geometry::FiniteEdgeGeometry;
    cancellation_threshold::Real=_FINITE_EDGE_CANCELLATION_THRESHOLD,
)
    all(value -> isfinite(_primal_value(value)), s) ||
        throw(ArgumentError("edge coordinates s must be finite"))
    threshold_primal = _primal_value(cancellation_threshold)
    isfinite(threshold_primal) && threshold_primal >= zero(threshold_primal) ||
        throw(ArgumentError("cancellation_threshold must be finite and nonnegative"))
    data = finite_edge_phase_data(geometry)
    return map(
        value -> _finite_edge_phase_coordinate(
            value,
            geometry,
            data,
            threshold_primal,
        ),
        s,
    )
end

@inline function _finite_edge_coordinate_derivative(
    s::Real,
    geometry::FiniteEdgeGeometry,
    data::FiniteEdgePhaseData,
)
    x = s - data.s0
    if abs(_primal_value(x)) <=
       _FINITE_EDGE_CANCELLATION_THRESHOLD * _finite_edge_length_scale(geometry)
        iszero(x) && return sqrt(data.phi2 / 2)
        factor = _finite_edge_phase_increment_factor(s, geometry, data)
        factor_primal = _primal_value(factor)
        factor_primal > zero(factor_primal) || throw(DomainError(
            (s, geometry),
            "rationalized phase-increment factor must be positive away from s0",
        ))
        slope = _finite_edge_phase_slope_factor(s, geometry, data)
        return slope / (2 * sqrt(factor))
    end
    t = _finite_edge_phase_coordinate(
        s,
        geometry,
        data,
        _FINITE_EDGE_CANCELLATION_THRESHOLD,
    )
    iszero(t) && throw(DomainError((s, geometry), "phase coordinate derivative is undefined"))
    return finite_edge_phase_derivative(s, geometry) / (2 * t)
end


"""Derivative `dt/ds` of the exact finite-edge phase coordinate."""
function finite_edge_coordinate_derivative(s::Real, geometry::FiniteEdgeGeometry)
    _finite_edge_validate_coordinate(s, "edge coordinate s")
    data = finite_edge_phase_data(geometry)
    return _finite_edge_coordinate_derivative(s, geometry, data)
end

"""
    finite_edge_fresnel_cs(z) -> (C, S)

Return the standard Fresnel integrals
`C(z)=∫₀ᶻ cos(πt²/2) dt` and `S(z)=∫₀ᶻ sin(πt²/2) dt`.
"""
function finite_edge_fresnel_cs(z::Real)
    _finite_edge_validate_coordinate(z, "Fresnel coordinate z")
    zf = float(z)
    primal = _primal_value(zf)
    if iszero(primal)
        # The local first-order jet is C(z)=z+O(z^5), S(z)=O(z^3).
        return zf, zero(zf)
    end
    sign_value = primal < zero(primal) ? -one(zf) : one(zf)
    az = sign_value * zf
    pi_value = _finite_edge_pi(zf)
    one_value = one(zf)
    alpha = complex(one_value, -one_value) * sqrt(pi_value) / 2
    argument = alpha * az
    scaled = try
        erfcx(argument)
    catch err
        err isa MethodError || rethrow()
        throw(ArgumentError(
            "finite_edge_fresnel_cs does not support $(typeof(z)): " *
            "erfcx is unavailable for $(typeof(argument))",
        ))
    end
    prefactor = complex(one_value, one_value) / 2
    value = prefactor *
            (one_value - exp(im * pi_value * az^2 / 2) * scaled)
    _finite_edge_checked(value, "finite-edge Fresnel integral")
    return sign_value * real(value), sign_value * imag(value)
end

"""
    finite_edge_fresnel_moments(t_a, t_b, k) -> (J0, J1, J2)

Return `J_n=∫[t_a,t_b] t^n exp(-im*k*t²) dt` for `n=0,1,2`.
"""
function finite_edge_fresnel_moments(t_a::Real, t_b::Real, k::Real)
    _finite_edge_validate_coordinate(t_a, "lower phase coordinate t_a")
    _finite_edge_validate_coordinate(t_b, "upper phase coordinate t_b")
    _finite_edge_validate_wavenumber(k)
    pi_value = _finite_edge_pi(k)
    scale = sqrt(2 * k / pi_value)
    u_a = scale * t_a
    u_b = scale * t_b
    C_a, S_a = finite_edge_fresnel_cs(u_a)
    C_b, S_b = finite_edge_fresnel_cs(u_b)
    J0 = sqrt(pi_value / (2 * k)) * ((C_b - C_a) - im * (S_b - S_a))
    e_a = exp(-im * k * t_a^2)
    e_b = exp(-im * k * t_b^2)
    J1 = (e_a - e_b) / (2im * k)
    J2 = (J0 - t_b * e_b + t_a * e_a) / (2im * k)
    _finite_edge_checked(J0, "finite-edge Fresnel moment J0")
    _finite_edge_checked(J1, "finite-edge Fresnel moment J1")
    _finite_edge_checked(J2, "finite-edge Fresnel moment J2")
    return J0, J1, J2
end

"""
    finite_edge_epm(a, b, k, geometry, amplitude; order=2)

Evaluate the endpoint-uniform moment approximation on the physical interval
`[a,b]`. `amplitude` contains local physical derivatives at the propagation
stationary point. `order` is 0, 1, or 2.
"""
function finite_edge_epm(
    a::Real,
    b::Real,
    k::Real,
    geometry::FiniteEdgeGeometry,
    amplitude::FiniteEdgeAmplitude;
    order::Integer=2,
)
    _finite_edge_validate_coordinate(a, "lower endpoint a")
    _finite_edge_validate_coordinate(b, "upper endpoint b")
    _finite_edge_validate_wavenumber(k)
    order in (0, 1, 2) || throw(ArgumentError("order must be 0, 1, or 2"))
    if _primal_value(b) < _primal_value(a)
        return -finite_edge_epm(b, a, k, geometry, amplitude; order)
    end
    data = finite_edge_phase_data(geometry)
    transformed = finite_edge_transform_data(data, amplitude)
    t_a = _finite_edge_phase_coordinate(
        a,
        geometry,
        data,
        _FINITE_EDGE_CANCELLATION_THRESHOLD,
    )
    t_b = _finite_edge_phase_coordinate(
        b,
        geometry,
        data,
        _FINITE_EDGE_CANCELLATION_THRESHOLD,
    )
    moments = finite_edge_fresnel_moments(t_a, t_b, k)
    coefficients = (transformed.G0, transformed.G1, transformed.G2)
    acc = zero(coefficients[1] * moments[1])
    @inbounds for n in 0:order
        acc += coefficients[n + 1] * moments[n + 1]
    end
    result = exp(-im * k * data.phi0) * acc
    return _finite_edge_checked(result, "finite-edge EPM value")
end

"""
    finite_edge_stationary_phase(k, geometry, A0)

Return the leading infinite-edge stationary-phase value for amplitude `A0`.
"""
function finite_edge_stationary_phase(
    k::Real,
    geometry::FiniteEdgeGeometry,
    A0::Number,
)
    _finite_edge_validate_wavenumber(k)
    _number_isfinite(A0) ||
        throw(ArgumentError("stationary finite-edge amplitude A0 must be finite"))
    data = finite_edge_phase_data(geometry)
    pi_value = _finite_edge_pi(k)
    result = A0 * exp(-im * k * data.phi0 - im * pi_value / 4) *
             sqrt(2 * pi_value / (k * data.phi2))
    return _finite_edge_checked(result, "finite-edge stationary-phase value")
end

"""
    finite_edge_endpoint_derivative(endpoint, k, geometry, amplitude)

Return the EPM2 Leibniz derivative with respect to a moving upper endpoint.
For a moving lower endpoint, negate the returned value.
"""
function finite_edge_endpoint_derivative(
    endpoint::Real,
    k::Real,
    geometry::FiniteEdgeGeometry,
    amplitude::FiniteEdgeAmplitude,
)
    _finite_edge_validate_coordinate(endpoint, "finite-edge endpoint")
    _finite_edge_validate_wavenumber(k)
    data = finite_edge_phase_data(geometry)
    transformed = finite_edge_transform_data(data, amplitude)
    t = _finite_edge_phase_coordinate(
        endpoint,
        geometry,
        data,
        _FINITE_EDGE_CANCELLATION_THRESHOLD,
    )
    polynomial = transformed.G0 + transformed.G1 * t + transformed.G2 * t^2
    result = exp(-im * k * data.phi0) * polynomial * exp(-im * k * t^2) *
             _finite_edge_coordinate_derivative(endpoint, geometry, data)
    return _finite_edge_checked(result, "finite-edge endpoint derivative")
end

"""
    finite_edge_parameter_derivative(
        a, b, k, geometry, amplitude,
        transformed_derivatives, phase0_derivative,
        lower_coordinate_derivative, upper_coordinate_derivative;
        order=2,
    )

Differentiate the endpoint-uniform approximation with respect to a smooth
scene parameter at fixed `k`. `transformed_derivatives` contains
`(dG0,dG1,dG2)`, `phase0_derivative` is `dΦ0`, and the final two arguments are
`dt_a` and `dt_b` for the physical endpoints. The caller may obtain these
quantities analytically, by automatic differentiation, or from a separately
validated local fit.
"""
function finite_edge_parameter_derivative(
    a::Real,
    b::Real,
    k::Real,
    geometry::FiniteEdgeGeometry,
    amplitude::FiniteEdgeAmplitude,
    transformed_derivatives::NTuple{3,<:Number},
    phase0_derivative::Real,
    lower_coordinate_derivative::Real,
    upper_coordinate_derivative::Real;
    order::Integer=2,
)
    _finite_edge_validate_coordinate(a, "lower endpoint a")
    _finite_edge_validate_coordinate(b, "upper endpoint b")
    _finite_edge_validate_wavenumber(k)
    _finite_edge_validate_coordinate(
        phase0_derivative,
        "stationary-phase parameter derivative",
    )
    _finite_edge_validate_coordinate(
        lower_coordinate_derivative,
        "lower endpoint-coordinate derivative",
    )
    _finite_edge_validate_coordinate(
        upper_coordinate_derivative,
        "upper endpoint-coordinate derivative",
    )
    all(_number_isfinite, transformed_derivatives) || throw(ArgumentError(
        "transformed-amplitude parameter derivatives must be finite",
    ))
    order in (0, 1, 2) || throw(ArgumentError("order must be 0, 1, or 2"))
    if _primal_value(b) < _primal_value(a)
        return -finite_edge_parameter_derivative(
            b,
            a,
            k,
            geometry,
            amplitude,
            transformed_derivatives,
            phase0_derivative,
            upper_coordinate_derivative,
            lower_coordinate_derivative;
            order,
        )
    end

    data = finite_edge_phase_data(geometry)
    transformed = finite_edge_transform_data(data, amplitude)
    t_a = _finite_edge_phase_coordinate(
        a,
        geometry,
        data,
        _FINITE_EDGE_CANCELLATION_THRESHOLD,
    )
    t_b = _finite_edge_phase_coordinate(
        b,
        geometry,
        data,
        _FINITE_EDGE_CANCELLATION_THRESHOLD,
    )
    moments = finite_edge_fresnel_moments(t_a, t_b, k)
    coefficients = (transformed.G0, transformed.G1, transformed.G2)

    M = zero(coefficients[1] * moments[1])
    dM = zero(transformed_derivatives[1] * moments[1])
    endpoint_sum = zero(M * upper_coordinate_derivative)
    @inbounds for n in 0:order
        coefficient = coefficients[n + 1]
        M += coefficient * moments[n + 1]
        dM += transformed_derivatives[n + 1] * moments[n + 1]
        endpoint_sum += coefficient * (
            t_b^n * exp(-im * k * t_b^2) * upper_coordinate_derivative -
            t_a^n * exp(-im * k * t_a^2) * lower_coordinate_derivative
        )
    end

    result = exp(-im * k * data.phi0) *
             (dM - im * k * phase0_derivative * M + endpoint_sum)
    return _finite_edge_checked(result, "finite-edge parameter derivative")
end
