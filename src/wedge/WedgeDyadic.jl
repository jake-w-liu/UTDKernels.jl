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
    _validate_wavenumber(k)
    A = spreading_factor(s, sp)
    # The spreading factor is exactly zero for an infinitely distant observer
    # (s = Inf), where the diffracted amplitude vanishes. Guard the propagation
    # phase so exp(-i k s) = NaN at s = Inf does not turn the zero amplitude into
    # NaN; evaluate the phase only for finite s.
    factor = iszero(A) ? zero(A * exp(-im * k * one(s))) : A * exp(-im * k * s)
    return (Ds * Es_i * factor, Dh * Eh_i * factor)
end
