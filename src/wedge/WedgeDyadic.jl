"""
Dyadic application in the ray-fixed soft/hard basis.
"""

"""
    spreading_factor(s, sp)

UTD spreading factor A(s,s') = √(s'/(s(s+s'))) for a straight edge.
For plane-wave incidence (sp=Inf), returns 1/√s.
"""
function spreading_factor(s::Real, sp::Real)
    _validate_distance(s, "observer distance s")
    _validate_distance(sp, "source distance sp")
    isinf(s) && return zero(promote(s, sp)[1])
    if isinf(sp)
        return 1 / sqrt(s)
    end
    if sp >= s
        return inv(sqrt(s) * sqrt(1 + s / sp))
    end
    # When sp/s underflows, sqrt(sp)/s remains representable over a much wider
    # range than forming A^2 first.
    return (sqrt(sp) / s) / sqrt(1 + sp / s)
end

@inline _ldexp_number(z::Real, exponent::Int) = ldexp(z, exponent)
@inline _ldexp_number(z::Complex, exponent::Int) =
    complex(ldexp(real(z), exponent), ldexp(imag(z), exponent))
@inline _max_component(z::Number) = max(abs(real(z)), abs(imag(z)))

function _spreading_binary_scale(s::T, sp::T) where {T<:AbstractFloat}
    isinf(sp) && return frexp(inv(sqrt(s)))
    if sp >= s
        return frexp(inv(sqrt(s) * sqrt(one(T) + s / sp)))
    end

    msp, esp = frexp(sqrt(sp))
    ms, es = frexp(s)
    mc, ec = frexp(inv(sqrt(one(T) + sp / s)))
    mantissa, correction = frexp((msp / ms) * mc)
    return mantissa, esp - es + ec + correction
end

function _scaled_field_product(
    x::Number,
    y::Number,
    phase::Number,
    scale_mantissa::Real,
    scale_exponent::Int,
)
    T = promote_type(
        typeof(float(real(x))), typeof(float(imag(x))),
        typeof(float(real(y))), typeof(float(imag(y))),
        typeof(float(real(phase))), typeof(float(imag(phase))),
        typeof(float(scale_mantissa)),
    )
    T in (Float16, Float32, Float64) || return nothing

    xc = Complex{T}(x)
    yc = Complex{T}(y)
    pc = Complex{T}(phase)
    (iszero(xc) || iszero(yc)) && return zero(xc * yc * pc)

    _, ex = frexp(_max_component(xc))
    _, ey = frexp(_max_component(yc))
    xn = _ldexp_number(xc, -ex)
    yn = _ldexp_number(yc, -ey)

    normalized = xn * yn * pc * T(scale_mantissa)
    _, en = frexp(_max_component(normalized))
    normalized = _ldexp_number(normalized, -en)
    return _ldexp_number(normalized, ex + ey + scale_exponent + en)
end

"""
    pec_wedge_apply_sh(Ds, Dh, Es_i, Eh_i, k, s, sp; convention=EXP_IWT)

Apply the PEC wedge diffraction dyadic in the soft/hard basis:

    E_s^d = Ds · E_s^i · A(s,s') · exp(-iks)
    E_h^d = Dh · E_h^i · A(s,s') · exp(-iks)

Returns (Es_d, Eh_d).
"""
function pec_wedge_apply_sh(
    Ds::Number, Dh::Number,
    Es_i::Number, Eh_i::Number,
    k::Number, s::Real, sp::Real;
    convention::PhasorConvention = EXP_IWT,
)
    convention.sgn == +1 || error("Only exp(+iωt) convention is supported")
    _validate_finite_number(Ds, "soft diffraction coefficient Ds")
    _validate_finite_number(Dh, "hard diffraction coefficient Dh")
    _validate_finite_number(Es_i, "incident soft field Es_i")
    _validate_finite_number(Eh_i, "incident hard field Eh_i")
    _validate_wavenumber(k)
    _validate_distance(s, "observer distance s")
    _validate_distance(sp, "source distance sp")
    # The spreading factor is exactly zero for an infinitely distant observer
    # (s = Inf), where the diffracted amplitude vanishes. Return typed zeros
    # before multiplying the input amplitudes: otherwise two large but finite
    # inputs can overflow first and turn the exact zero result into Inf*0 = NaN.
    if isinf(s)
        A = zero(promote(s, sp)[1])
        propagation_zero = zero(-im * A * k)
        return (
            zero(Ds) * zero(Es_i) * propagation_zero,
            zero(Dh) * zero(Eh_i) * propagation_zero,
        )
    end

    phase = -im * k * s
    _number_isfinite(phase) || throw(DomainError(
        (k, s),
        "propagation phase -im*k*s is non-finite",
    ))
    phase_factor = exp(phase)
    if k isa Real
        T = promote_type(typeof(float(s)), typeof(float(sp)))
        if T in (Float16, Float32, Float64)
            s_eval, sp_eval = promote(float(s), float(sp))
            A_mantissa, A_exponent = _spreading_binary_scale(s_eval, sp_eval)
            Es_d = _scaled_field_product(
                Ds, Es_i, phase_factor, A_mantissa, A_exponent,
            )
            Eh_d = _scaled_field_product(
                Dh, Eh_i, phase_factor, A_mantissa, A_exponent,
            )
            if Es_d !== nothing && Eh_d !== nothing
                return _checked_coefficients(Es_d, Eh_d)
            end
        end
    end

    A = spreading_factor(s, sp)
    factor = A * phase_factor
    _number_isfinite(factor) || throw(DomainError(
        (k, s, sp),
        "propagation factor is non-finite at the requested parameters",
    ))
    return _checked_coefficients(Ds * Es_i * factor, Dh * Eh_i * factor)
end
