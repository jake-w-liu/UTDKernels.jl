using Printf

function latex_scientific(value::Real, digits::Integer=2)
    v = Float64(value)
    isfinite(v) || throw(ArgumentError("scientific notation requires a finite value"))
    iszero(v) && return "0"
    mantissa, exponent = split(@sprintf("%.*e", digits, v), "e")
    return string(mantissa, "\\times 10^{", parse(Int, exponent), "}")
end

# Both saddle contributions enter the separated comparator, but only the
# -2pi saddle's poles are recorded in these diagnostic tables. An exact
# zero belonging only to the other saddle cannot be represented by their
# singular-exclusion columns.
function _separated_slope_singular(m2, m0)
    recorded_zero = any(iszero, m2.poles)
    if any(iszero, m0.poles) && !recorded_zero
        throw(DomainError(m0.poles,
            "separated-slope zero belongs only to the unrecorded saddle"))
    end
    return recorded_zero
end

# The separated-slope boundary comparator is undefined at an exact mapped
# pole. Only its two boundary-table entries permit this notation.
function _separated_boundary_latex(value::Real)
    return isnan(value) ? "\\text{undefined}" : latex_scientific(value)
end
