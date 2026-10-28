function _panel_count(kappa::Real, span_root::Real,
                      phase_increment::Real, max_panels::Integer)
    # span_root = sqrt(high^2-low^2), evaluated without squaring either
    # endpoint. Compare square roots before forming the panel estimate.
    normalized = (sqrt(kappa) * span_root) / sqrt(phase_increment)
    isfinite(normalized) && normalized <= sqrt(float(max_panels)) ||
        throw(ArgumentError(
            "quadrature needs more than max_panels=$max_panels; narrow the " *
            "interval or increase phase_increment"))
    estimate = normalized^2
    estimate <= max_panels || throw(ArgumentError("quadrature panel estimate exceeds max_panels=$max_panels"))
    return max(8, ceil(Int, estimate))
end

function composite_phase_quadrature(
    amplitude,
    lo::Real,
    hi::Real,
    kappa::Real;
    phase_increment::Real=2.0,
    order::Integer=16,
    max_panels::Integer=DEFAULT_MAX_PANELS,
)
    isfinite(lo) && isfinite(hi) && lo < hi || throw(ArgumentError(
        "quadrature requires finite lo < hi",
    ))
    _validate_positive(kappa, "kappa")
    _validate_positive(phase_increment, "phase_increment")
    1 <= order <= typemax(Int) || throw(ArgumentError("order must be a positive representable Int"))
    8 <= max_panels <= typemax(Int) || throw(ArgumentError("max_panels must be a representable Int of at least 8"))
    nodes, weights = gausslegendre(order)
    sample = _validate_finite(amplitude(lo / 2 + hi / 2), "amplitude value")
    parameter = float(kappa)
    increment = float(phase_increment)

    function side(a::Real, b::Real)
        a == b && return zero(complex(parameter) + sample)
        positive = b > 0
        low = min(abs(a), abs(b))
        high = max(abs(a), abs(b))
        span_root = UTDKernels._scaled_sqrt_product(high-low, high, 1+low/high)
        panels = _panel_count(parameter, span_root, increment, max_panels)
        total = zero(complex(parameter) + sample)
        for panel in 1:panels
            if positive
                left = panel == 1 ? low : hypot(low, span_root*sqrt((panel-1)/panels))
                right = panel == panels ? high : hypot(low, span_root*sqrt(panel/panels))
            else
                left = panel == 1 ? -high : -hypot(low, span_root*sqrt(1-(panel-1)/panels))
                right = panel == panels ? -low : -hypot(low, span_root*sqrt(1-panel/panels))
            end
            midpoint = (left + right) / 2
            halfwidth = (right - left) / 2
            subtotal = zero(complex(parameter) + sample)
            for index in eachindex(nodes)
                y = midpoint + halfwidth * nodes[index]
                value = amplitude(y)
                _validate_finite(value, "amplitude value")
                subtotal += weights[index] *
                            scaled_kernel(y, parameter) * value
            end
            total += halfwidth * subtotal
        end
        return total
    end

    total = zero(complex(parameter) + sample)
    lo < 0 && (total += side(float(lo), min(float(hi), 0.0)))
    hi > 0 && (total += side(max(float(lo), 0.0), float(hi)))
    return _finite_result(total, "composite phase integral")
end
