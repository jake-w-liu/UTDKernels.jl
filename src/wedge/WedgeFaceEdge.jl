"""
Reflection-boundary face--edge decomposition of the PEC KP coefficient.

The formulas in this file are restricted to `phi + phip = wedge.alpha`. They
reuse the package transition function and convention, but form the incident
pair before adding the repeated reflection terms so a small intrinsic residual
is not recovered by subtracting complete soft/hard coefficients.
"""

"""
    FaceEdgeDomainError

Error raised when a requested reflection-boundary face--edge decomposition is
outside its verified exterior-wedge or nearest-pole branch.
"""
struct FaceEdgeDomainError <: Exception
    message::String
end

Base.showerror(io::IO, error::FaceEdgeDomainError) =
    print(io, "FaceEdgeDomainError: ", error.message)

# Low parts of mathematical pi after the nearest Float32/Float64 value. They
# matter only when `delta` is adjacent to the typed endpoint: `pi_hi-delta`
# then has the same size as the discarded low part. The domain comparison
# still uses the type-local endpoint, so Float32(pi)/Float64(pi) is rejected.
@inline _face_edge_pi_tail(::Type{Float32}) = -8.742277657347586f-8
@inline _face_edge_pi_tail(::Type{Float64}) = 1.2246467991473532e-16
@inline _face_edge_pi_tail(::Type{T}) where {T<:AbstractFloat} = zero(T)

@inline function _face_edge_float_type(value::Real)
    return typeof(float(_primal_value(value)))
end

@inline function _face_edge_pi_parts(value::Real)
    T = _face_edge_float_type(value)
    return T(pi), _face_edge_pi_tail(T)
end

@inline function _face_edge_sinc(value::Real)
    primal = abs(_primal_value(value))
    if primal < oftype(primal, 0.01)
        square = value * value
        return one(value) + square * (
            -one(value) / 6 + square * (
                one(value) / 120 + square * (
                    -one(value) / 5040 + square / 362880
                )
            )
        )
    end
    return sin(value) / value
end

@inline function _face_edge_erfcx(z)
    try
        return erfcx(z)
    catch error
        error isa MethodError || rethrow()
        throw(ArgumentError(
            "pec_wedge_face_edge does not support $(typeof(real(z))): " *
            "erfcx is unavailable for $(typeof(z))",
        ))
    end
end

@inline function _face_edge_phase(value::Real, pi_hi::Real)
    return cis((zero(value) + pi_hi) / 4)
end

@inline function _face_edge_B(a::Real, k::Real, L::Real, pi_hi::Real)
    a_primal = _primal_value(a)
    a_primal >= zero(a_primal) ||
        throw(DomainError(a, "face-edge transition parameter must be nonnegative"))
    phase = _face_edge_phase(a + k + L, pi_hi)
    sqrt_x = _scaled_sqrt_product(k, L, a)
    if isinf(_primal_value(sqrt_x))
        # F(kLa) -> 1 while B(a)=F(kLa)/sqrt(k).
        return complex(inv(sqrt(k)))
    end
    root_La = sqrt(L) * sqrt(a)
    value = sqrt(zero(a) + pi_hi) * root_La * phase *
            _face_edge_erfcx(phase * sqrt_x)
    _number_isfinite(value) || throw(DomainError(
        (a, k, L),
        "face-edge scaled transition value is non-finite",
    ))
    return value
end

# erfcx(z+dz)-erfcx(z). The differential recurrence
# E^(m+1)=2z E^m+2m E^(m-1) avoids subtracting nearby function values. The
# eighth-order local series is used only for |dz|<=1/8 and |z|<=4.
@inline function _face_edge_erfcx_difference(z, dz, value)
    dz_primal = abs(_primal_value(abs(dz)))
    z_primal = abs(_primal_value(abs(z)))
    if dz_primal > oftype(dz_primal, 0.125) || z_primal > oftype(z_primal, 4)
        return _face_edge_erfcx(z + dz) - value
    end

    inverse_sqrt_pi = inv(sqrt(zero(real(z)) + _face_edge_pi_parts(real(z))[1]))
    previous = value
    derivative = 2z * value - 2inverse_sqrt_pi
    power = dz
    factorial = one(real(z))
    difference = derivative * power
    for order in 2:8
        next_derivative = 2z * derivative + 2(order - 1) * previous
        previous = derivative
        derivative = next_derivative
        power *= dz
        factorial *= order
        difference += derivative * power / factorial
    end
    return difference
end

# Stable B(a1)-B(a0). At small transition arguments it differentiates the
# analytic y*erfcx(c*sqrt(kL)*y) form in y=sqrt(a), which removes the singular
# real-axis F derivatives at x=0. At moderate/large x it uses the shared F
# derivative bundle, including its asymptotic derivative branch.
function _face_edge_B_difference(
    a0::Real,
    a1::Real,
    delta_a::Real,
    k::Real,
    L::Real,
    b0,
    pi_hi::Real,
)
    a0_primal = _primal_value(a0)
    if iszero(a0_primal)
        return _face_edge_B(a1, k, L, pi_hi) - b0
    end

    sqrt_x0 = _scaled_sqrt_product(k, L, a0)
    if _primal_value(sqrt_x0) <= oftype(_primal_value(sqrt_x0), 4)
        y0 = sqrt(a0)
        y1 = sqrt(a1)
        delta_y = delta_a / (y0 + y1)
        relative_y = delta_y / y0
        phase = _face_edge_phase(a0 + k + L, pi_hi)
        z0 = phase * sqrt_x0
        delta_z = z0 * relative_y
        e0 = _face_edge_erfcx(z0)
        delta_e = _face_edge_erfcx_difference(z0, delta_z, e0)
        difference = sqrt(zero(a0) + pi_hi) * sqrt(L) * phase *
                     (delta_y * e0 + y1 * delta_e)
        _number_isfinite(difference) && return difference
    end

    if abs(_primal_value(delta_a)) > oftype(a0_primal, 1e-3) * abs(a0_primal)
        return _face_edge_B(a1, k, L, pi_hi) - b0
    end

    x0 = (k * L) * a0
    delta_x = (k * L) * delta_a
    x1 = x0 + delta_x
    if isfinite(_primal_value(x0)) && isfinite(_primal_value(delta_x)) &&
       _primal_value(x0) > zero(_primal_value(x0)) &&
       _primal_value(x1) >= zero(_primal_value(x1))
        _, first, second, third = _F_utd_derivatives3(x0)
        # Associate each Taylor term from its (possibly tiny) derivative. Forming
        # `delta_x^2` first can overflow even when the final product is zero or
        # representable, turning a valid asymptotic difference into `0*Inf=NaN`.
        first_term = first * delta_x
        second_term = (second * delta_x) * delta_x / 2
        third_term = ((third * delta_x) * delta_x) * delta_x / 6
        delta_f = first_term + second_term + third_term
        difference = delta_f / sqrt(k)
        _number_isfinite(difference) && return difference
    end

    return _face_edge_B(a1, k, L, pi_hi) - b0
end

@inline function _face_edge_Q(value::Real, n::Real)
    primal = abs(_primal_value(value))
    if primal <= oftype(primal, 0.01)
        inverse_n = inv(n)
        inverse_n2 = inverse_n * inverse_n
        q2 = -n / 6 - inverse_n / 3
        q4 = n / 120 + inverse_n / 18 - inverse_n * inverse_n2 / 45
        q6 = -n / 5040 - inverse_n / 360 + inverse_n * inverse_n2 / 270 -
             2inverse_n * inverse_n2^2 / 945
        q8 = n / 362880 + inverse_n / 15120 - inverse_n * inverse_n2 / 5400 +
             inverse_n * inverse_n2^2 / 2835 - inverse_n * inverse_n2^3 / 4725
        square = value * value
        return n + square * (q2 + square * (q4 + square * (q6 + square * q8)))
    end
    return sin(value) * cot(value / n)
end

@inline function _face_edge_Q_difference(
    t0::Real,
    t1::Real,
    delta_t::Real,
    n::Real,
)
    largest = max(abs(_primal_value(t0)), abs(_primal_value(t1)))
    if largest <= oftype(largest, 0.01)
        inverse_n = inv(n)
        inverse_n2 = inverse_n * inverse_n
        q2 = -n / 6 - inverse_n / 3
        q4 = n / 120 + inverse_n / 18 - inverse_n * inverse_n2 / 45
        q6 = -n / 5040 - inverse_n / 360 + inverse_n * inverse_n2 / 270 -
             2inverse_n * inverse_n2^2 / 945
        q8 = n / 362880 + inverse_n / 15120 - inverse_n * inverse_n2 / 5400 +
             inverse_n * inverse_n2^2 / 2835 - inverse_n * inverse_n2^3 / 4725
        p0 = t0 * t0
        p1 = t1 * t1
        divided = q2 + q4 * (p0 + p1) +
                  q6 * (p0^2 + p0 * p1 + p1^2) +
                  q8 * (p0^3 + p0^2 * p1 + p0 * p1^2 + p1^3)
        # `t1` may be much larger than `delta_t`; subtracting its rounded square
        # would discard the defect carried separately by `delta_t`.
        return (-delta_t * (t0 + t1)) * divided
    end
    return _face_edge_Q(t0, n) - _face_edge_Q(t1, n)
end

function _face_edge_endpoint_incident(
    n::Real,
    eta::Real,
    q::Real,
    k::Real,
    L::Real,
    pi_hi::Real,
)
    t0 = q / 2
    t1 = t0 + eta
    a0 = 2sin(t0)^2
    a1 = 2sin(t1)^2
    sqrt_x0 = _scaled_sqrt_product(k, L, a0)
    sqrt_x1 = _scaled_sqrt_product(k, L, a1)
    phase = _face_edge_phase(n + eta + q + k + L, pi_hi)
    z0 = phase * sqrt_x0
    sin_t0 = sin(t0)
    sin_difference = 2cos(t0 + eta / 2) * sin(eta / 2)
    delta_sqrt_x = iszero(_primal_value(sin_t0)) ?
                   sqrt_x1 : sqrt_x0 * (sin_difference / sin_t0)
    e0 = _face_edge_erfcx(z0)
    delta_e = _face_edge_erfcx_difference(z0, phase * delta_sqrt_x, e0)
    q1 = _face_edge_Q(t1, n)
    q_difference = _face_edge_Q_difference(t0, t1, eta, n)
    incident = sqrt(2 * (zero(n) + pi_hi)) * sqrt(L) * phase *
               (q_difference * e0 - q1 * delta_e)
    _number_isfinite(incident) || throw(DomainError(
        (eta, q, k, L),
        "endpoint-adjacent incident pair is non-finite",
    ))
    return incident
end

@inline function _face_edge_face(n::Real, eta::Real, k::Real, L::Real, pi_hi::Real)
    iszero(_primal_value(eta)) && return complex(sqrt(L))
    half_eta = eta / 2
    local_angle = half_eta / n
    a_reflection = 2sin(half_eta)^2
    sqrt_x = _scaled_sqrt_product(k, L, a_reflection)
    phase = _face_edge_phase(n + eta + k + L, pi_hi)
    geometry_factor = cos(local_angle) * _face_edge_sinc(half_eta) /
                      _face_edge_sinc(local_angle)

    if isinf(_primal_value(sqrt_x))
        # Equivalent raw reflection-pair form with F=1.
        return -conj(phase) * cot(-local_angle) /
               (n * sqrt(2 * (zero(n) + pi_hi)) * sqrt(k))
    end

    face = sqrt(L) * geometry_factor * _face_edge_erfcx(phase * sqrt_x)
    _number_isfinite(face) || throw(DomainError(
        (n, eta, k, L),
        "face-transition component is non-finite",
    ))
    return face
end

function _face_edge_incident(
    n::Real,
    epsilon::Real,
    delta::Real,
    eta::Real,
    q::Real,
    k::Real,
    L::Real,
    pi_hi::Real,
)
    if iszero(_primal_value(epsilon))
        phase = _face_edge_phase(n + delta + k + L, pi_hi)
        return zero(phase)
    end

    phase = _face_edge_phase(n + delta + k + L, pi_hi)
    scale = -conj(phase) / (2n * sqrt(2 * (zero(n) + pi_hi)))
    if iszero(_primal_value(delta))
        cotangent = cot((zero(n) + pi_hi) / (2n))
        incident = 2cotangent * _face_edge_B(2one(n), k, L, pi_hi)
        edge = scale * incident
        _number_isfinite(edge) || throw(DomainError(
            (n, delta, k, L),
            "symmetric intrinsic edge component is non-finite",
        ))
        return edge
    end

    t0 = q / 2
    t1 = t0 + eta
    a0 = 2sin(t0)^2
    a1 = 2sin(t1)^2
    sqrt_x1 = _scaled_sqrt_product(k, L, a1)
    # When both local endpoint angles are small, evaluate their smooth
    # regularized products as one difference. Restrict this route to a small
    # transition argument; elsewhere the general compensated incident-pair
    # identity is better conditioned.
    local_endpoint = max(abs(_primal_value(t0)), abs(_primal_value(t1))) <= 0.01
    incident = if local_endpoint &&
                  _primal_value(sqrt_x1) <= oftype(_primal_value(sqrt_x1), 4)
        _face_edge_endpoint_incident(n, eta, q, k, L, pi_hi)
    else
        u0 = t0 / n
        u1 = t1 / n
        delta_a = 2sin(eta) * sin(eta + q)
        b0 = _face_edge_B(a0, k, L, pi_hi)
        delta_b = _face_edge_B_difference(a0, a1, delta_a, k, L, b0, pi_hi)
        cotangent_sum = sin(eta / n) / (sin(u0) * sin(u1))
        cotangent_sum * b0 - cot(u1) * delta_b
    end
    edge = scale * incident
    _number_isfinite(edge) || throw(DomainError(
        (n, delta, k, L),
        "intrinsic edge component is non-finite",
    ))
    return edge
end

@inline function _face_edge_validate_numeric(delta::Real, k::Real, L::Real, pi_hi::Real)
    delta_primal = _primal_value(delta)
    pi_primal = _primal_value(pi_hi)
    (isfinite(delta_primal) && zero(delta_primal) <= delta_primal < pi_primal) ||
        throw(FaceEdgeDomainError("delta must satisfy 0 <= delta < pi"))
    _validate_wavenumber(k)
    L_primal = _primal_value(L)
    (isfinite(L_primal) && L_primal > zero(L_primal)) ||
        throw(DomainError(L, "face-edge distance L must be finite and positive"))
    return nothing
end

function _face_edge_parameters(
    wedge::Wedge,
    delta::Real,
    k::Real,
    L::Real,
)
    alpha, delta_value, k_value, L_value = promote(
        float(wedge.alpha), float(delta), float(k), float(L),
    )
    pi_hi, pi_lo = _face_edge_pi_parts(alpha)
    alpha_primal = _primal_value(alpha)
    alpha_primal >= pi_hi || throw(FaceEdgeDomainError(
        "face-edge decomposition requires pi <= wedge.alpha <= 2pi",
    ))
    _face_edge_validate_numeric(delta_value, k_value, L_value, pi_hi)
    typed_pi = zero(alpha) + pi_hi
    n = alpha / typed_pi
    # With no separate precision carrier, the stored wedge angle is the
    # authority. Sterbenz subtraction preserves every representable bit of its
    # angular defect and makes the branch test exactly `delta > alpha-pi`.
    eta = alpha - typed_pi
    epsilon = eta / typed_pi
    if !iszero(_primal_value(delta_value)) &&
       _primal_value(delta_value) <= _primal_value(eta)
        throw(FaceEdgeDomainError(
            "for delta > 0, the verified nearest-pole branch requires " *
            "delta > wedge.alpha - pi",
        ))
    end
    q = ((zero(delta_value) + pi_hi) - delta_value) + (zero(delta_value) + pi_lo)
    return n, epsilon, delta_value, eta, q, k_value, L_value, pi_hi
end

function _face_edge_parameters(
    wedge::Wedge,
    delta::Real,
    k::Real,
    L::Real,
    epsilon::Real,
)
    alpha, delta_value, k_value, L_value, epsilon_value = promote(
        float(wedge.alpha), float(delta), float(k), float(L), float(epsilon),
    )
    pi_hi, pi_lo = _face_edge_pi_parts(alpha)
    alpha_primal = _primal_value(alpha)
    alpha_primal >= pi_hi || throw(FaceEdgeDomainError(
        "face-edge decomposition requires pi <= wedge.alpha <= 2pi",
    ))
    _face_edge_validate_numeric(delta_value, k_value, L_value, pi_hi)
    epsilon_primal = _primal_value(epsilon_value)
    (isfinite(epsilon_primal) && epsilon_primal >= zero(epsilon_primal)) ||
        throw(FaceEdgeDomainError("epsilon must be finite and nonnegative"))
    n = one(epsilon_value) + epsilon_value
    expected_alpha = (zero(n) + pi_hi) * n
    expected_primal = _primal_value(expected_alpha)
    base_type = _face_edge_float_type(expected_alpha)
    tolerance = 4eps(base_type) * max(abs(expected_primal), one(expected_primal))
    abs(alpha_primal - expected_primal) <= tolerance || throw(FaceEdgeDomainError(
        "epsilon is inconsistent with wedge.alpha at the active precision",
    ))
    eta = (zero(epsilon_value) + pi_hi) * epsilon_value +
          (zero(epsilon_value) + pi_lo) * epsilon_value
    if !iszero(_primal_value(delta_value)) &&
       _primal_value(delta_value) <= _primal_value(eta)
        throw(FaceEdgeDomainError(
            "for delta > 0, the verified nearest-pole branch requires " *
            "delta > pi*epsilon",
        ))
    end
    q = ((zero(delta_value) + pi_hi) - delta_value) + (zero(delta_value) + pi_lo)
    return n, epsilon_value, delta_value, eta, q, k_value, L_value, pi_hi
end

function _face_edge_result(parameters)
    n, epsilon, delta, eta, q, k, L, pi_hi = parameters
    edge = _face_edge_incident(n, epsilon, delta, eta, q, k, L, pi_hi)
    face = _face_edge_face(n, eta, k, L, pi_hi)
    Ds = edge - face
    Dh = edge + face
    _number_isfinite(Ds) && _number_isfinite(Dh) || throw(DomainError(
        (Ds, Dh),
        "face-edge decomposition produced a non-finite coefficient",
    ))
    return (; Ds, Dh, edge, face)
end

"""
    pec_wedge_face_edge(wedge, delta, k, L; convention=EXP_IWT)
    pec_wedge_face_edge(wedge, delta, k, L, epsilon; convention=EXP_IWT)

Decompose the PEC coefficient on the reflection boundary into intrinsic
`edge` and face-transition `face` components. The boundary angles are
`phi=(wedge.alpha+delta)/2` and `phip=(wedge.alpha-delta)/2`; arbitrary ray
angles are not projected onto this boundary.

The verified domain is `pi <= wedge.alpha <= 2pi`, `0 <= delta < pi`, finite
real `k>0` and `L>0`, and either `delta==0` or
`delta>wedge.alpha-pi`. The result is a named tuple
`(; Ds, Dh, edge, face)` with `Dh=edge+face` and `Ds=edge-face`.

The five-argument overload accepts the dimensionless defect
`epsilon=wedge.alpha/pi-1` as a precision carrier. It is useful when a defect
near machine precision was known before `wedge.alpha` was rounded; the value is
rejected unless it rounds back to the supplied wedge angle.
"""
function pec_wedge_face_edge(
    wedge::Wedge,
    delta::Real,
    k::Real,
    L::Real;
    convention::PhasorConvention=EXP_IWT,
)
    convention.sgn == +1 || throw(ArgumentError("only exp(+i*omega*t) is supported"))
    return _face_edge_result(_face_edge_parameters(wedge, delta, k, L))
end

function pec_wedge_face_edge(
    wedge::Wedge,
    delta::Real,
    k::Real,
    L::Real,
    epsilon::Real;
    convention::PhasorConvention=EXP_IWT,
)
    convention.sgn == +1 || throw(ArgumentError("only exp(+i*omega*t) is supported"))
    return _face_edge_result(_face_edge_parameters(wedge, delta, k, L, epsilon))
end

"""
    pec_wedge_intrinsic_score(wedge, delta, k, L)
    pec_wedge_intrinsic_score(wedge, delta, k, L, epsilon)

Return the dimensionless local candidate-edge score
`sqrt(2pi*k)*abs(edge)` from [`pec_wedge_face_edge`](@ref). This is a local
diagnostic; it is not a bound on the field error caused by removing an edge
from a complete object.
"""
function pec_wedge_intrinsic_score(
    wedge::Wedge,
    delta::Real,
    k::Real,
    L::Real;
    convention::PhasorConvention=EXP_IWT,
)
    convention.sgn == +1 || throw(ArgumentError("only exp(+i*omega*t) is supported"))
    parameters = _face_edge_parameters(wedge, delta, k, L)
    n, epsilon, delta_value, eta, q, k_value, L_value, pi_hi = parameters
    edge = _face_edge_incident(
        n, epsilon, delta_value, eta, q, k_value, L_value, pi_hi,
    )
    return sqrt(2 * (zero(k_value) + pi_hi)) * sqrt(k_value) * abs(edge)
end

function pec_wedge_intrinsic_score(
    wedge::Wedge,
    delta::Real,
    k::Real,
    L::Real,
    epsilon::Real;
    convention::PhasorConvention=EXP_IWT,
)
    convention.sgn == +1 || throw(ArgumentError("only exp(+i*omega*t) is supported"))
    parameters = _face_edge_parameters(wedge, delta, k, L, epsilon)
    n, epsilon_value, delta_value, eta, q, k_value, L_value, pi_hi = parameters
    edge = _face_edge_incident(
        n, epsilon_value, delta_value, eta, q, k_value, L_value, pi_hi,
    )
    return sqrt(2 * (zero(k_value) + pi_hi)) * sqrt(k_value) * abs(edge)
end
