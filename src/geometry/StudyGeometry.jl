# Geometry shared by the canonical study assemblers. Directions and means
# are invariant under common positive scaling; use that before subtraction
# or summation can overflow, while retaining the original numeric/AD type.
function _study_unit_direction2(a, b)
    length(a) == length(b) == 2 && all(x -> x isa Real && _number_isfinite(x), (a..., b...)) ||
        throw(ArgumentError("edge endpoints must contain two finite real coordinates"))
    ax, ay, bx, by = promote(float(a[1]), float(a[2]), float(b[1]), float(b[2]))
    dx, dy = bx - ax, by - ay
    scale = max(abs(_primal_value(dx)), abs(_primal_value(dy)))
    scale > 0 || throw(ArgumentError("edge endpoints must be distinct"))
    differentiated=any(_number_contains_ad,(dx,dy))
    squared_scale=scale*scale
    native_ad_range=isfinite(squared_scale) && squared_scale>=floatmin(typeof(float(scale)))
    if all(_number_isfinite, (dx,dy)) && (!differentiated || native_ad_range)
        magnitude = hypot(dx,dy)
        if !any(_number_contains_ad, (dx,dy)) && _number_isfinite(magnitude)
            direction = (dx/magnitude,dy/magnitude)
            all(_number_isfinite, direction) && return direction
        end
        x,y = dx/scale,dy/scale
        magnitude = hypot(x,y)
        direction = (x/magnitude,y/magnitude)
        all(_number_isfinite, direction) && return direction
    end
    C = typeof(complex(ax+ay+bx+by))
    wide = _shadow_stable_wide_value(C, 4096, "edge direction") do
        axw,ayw,bxw,byw = _faddeeva_widen_bigfloat.((ax,ay,bx,by))
        xw,yw = bxw-axw,byw-ayw
        magnitude_w = hypot(xw,yw)
        complex(xw/magnitude_w,yw/magnitude_w)
    end
    return real(wide), imag(wide)
end

function _study_centroid2(points::AbstractMatrix{<:Real})
    size(points, 2) == 2 && size(points, 1) > 0 && all(_number_isfinite, points) ||
        throw(ArgumentError("centroid points must have finite real shape (N,2), N>0"))
    scale = maximum(x -> abs(_primal_value(float(x))), points)
    iszero(scale) && (scale = one(scale))
    count = size(points, 1)
    center = ntuple(2) do j
        primal_mean = sum((_primal_value(points[i,j]) / scale) / count for i in axes(points, 1)) * scale
        # Geometry scale must not divide AD seeds: a tiny object can carry a
        # huge common translation derivative. Anchor the nilpotent mean so
        # common translations cancel exactly in subsequent relative vectors.
        first_index = first(axes(points, 1))
        anchor = points[first_index,j] - _primal_value(points[first_index,j])
        nilpotent_mean = anchor + sum(
            ((points[i,j] - _primal_value(points[i,j])) - anchor) / count
            for i in axes(points, 1))
        value = primal_mean + nilpotent_mean
        if !_number_isfinite(value)
            C = typeof(sum(zero(float(points[i,j])) for i in axes(points,1)))
            return _shadow_stable_wide_value(C, 4096, "polygon centroid") do
                sum(_faddeeva_widen_bigfloat(points[i,j]) for i in axes(points, 1)) / count
            end
        end
        value
    end
    all(_number_isfinite, center) || throw(DomainError(
        center, "polygon centroid exceeds the active numeric range"))
    return center
end

# Positive local turns alone also admit multiply wound star polygons.
# A strictly convex CCW boundary must close with one full turning cycle.
function _study_validate_ccw_turns(turns)
    length(turns) >= 3 && all(t -> t isa Real && _number_isfinite(t) && 0 < _primal_value(t) < pi, turns) ||
        throw(ArgumentError("polygon must be strictly convex and counterclockwise"))
    values = float.(_primal_value.(turns))
    R = typeof(first(values))
    total = sum(values)
    period = R(2) * R(pi)
    tolerance = min(period / 4, 64eps(R) * length(values) * R(pi))
    abs(total - period) <= tolerance || throw(ArgumentError(
        "polygon must have exactly one counterclockwise full turn"))
    return nothing
end

function _study_distance2(a, b)
    length(a) == length(b) == 2 && all(x -> x isa Real && _number_isfinite(x), (a..., b...)) ||
        throw(ArgumentError("distance endpoints must contain two finite real coordinates"))
    ax, ay, bx, by = promote(float(a[1]), float(a[2]), float(b[1]), float(b[2]))
    dx, dy = bx-ax, by-ay
    scale = max(abs(_primal_value(dx)), abs(_primal_value(dy)))
    scale > 0 || throw(ArgumentError("distance endpoints must be distinct"))
    differentiated=any(_number_contains_ad,(dx,dy))
    squared_scale=scale*scale
    native_ad_range=isfinite(squared_scale) && squared_scale>=floatmin(typeof(float(scale)))
    if all(_number_isfinite, (dx, dy)) && isfinite(scale) && (!differentiated || native_ad_range)
        # Base hypot is range-safe in value; normalized arguments also keep
        # its nested AD rules away from squares of tiny or huge distances.
        result = any(_number_contains_ad, (dx, dy)) ?
            scale*hypot(dx/scale,dy/scale) : hypot(dx,dy)
        _number_isfinite(result) && return result
    end
    C = typeof(ax+ay+bx+by)
    return _shadow_stable_wide_value(C, 4096, "physical distance") do
        axw,ayw,bxw,byw = _faddeeva_widen_bigfloat.((ax,ay,bx,by))
        hypot(bxw-axw,byw-ayw)
    end
end

# atan on unit arguments avoids the tiny/huge squared denominator in AD
# rules. Keep the original represented primal angle for geometric handoffs.
function _study_angle2(a, b)
    direction = _study_unit_direction2(a, b)
    angle = atan(direction[2], direction[1])
    ax,ay,bx,by = promote(float.(_primal_value.((a...,b...)))...)
    dx,dy = bx-ax,by-ay
    all(isfinite, (dx,dy)) || return angle
    primal_angle = atan(dy,dx)
    return angle + (primal_angle-_primal_value(angle))
end
