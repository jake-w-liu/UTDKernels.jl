"""
Angle normalization utilities.
"""

"""
    wrap_angle(phi, alpha)

Wrap angle `phi` into [0, alpha).

Implemented as `phi - fld(phi, alpha)*alpha` rather than `mod(phi, alpha)`.
This is the defining identity of `mod` (with `fld` the floor division), so the
value agrees with `mod` to machine precision and is exact at multiples of
`alpha`. The reason for the explicit form is forward-mode AD: `mod`'s rule
returns a NaN derivative exactly on a wrap boundary (`phi` a multiple of
`alpha`, e.g. grazing incidence `phi = 0`), whereas the wrap is locally a
translation. With the `fld` form the integer count contributes a zero
derivative, so the wrapped angle differentiates with the correct local slope
(unit in `phi`) and stays differentiable through grazing/boundary aspects.
"""
function wrap_angle(phi::Real, alpha::Real)
    phi - fld(phi, alpha) * alpha
end
