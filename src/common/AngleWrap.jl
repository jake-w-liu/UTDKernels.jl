"""
Angle normalization utilities.
"""

"""
    wrap_angle(phi, alpha)

Wrap angle `phi` into [0, alpha) via modular arithmetic.
"""
function wrap_angle(phi::Real, alpha::Real)
    mod(phi, alpha)
end
