"""
Dyadic application in the ray-fixed soft/hard basis.
"""

"""
    spreading_factor(s, sp)

UTD spreading factor A(s,s') = √(s'/(s(s+s'))) for a straight edge.
For plane-wave incidence (sp=Inf), returns 1/√s.
"""
function spreading_factor(s::Real, sp::Real)
    if isinf(sp)
        return 1 / sqrt(s)
    end
    sqrt(sp / (s * (s + sp)))
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
    phase = exp(-im * k * s)  # outgoing, exp(+iωt) convention
    factor = A * phase
    return (Ds * Es_i * factor, Dh * Eh_i * factor)
end
