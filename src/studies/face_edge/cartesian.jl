# Cartesian products are evaluated in their true units. Predicates, angles,
# and ratios have separate contracts, so an unrepresentable determinant is
# never silently rescaled into a misleading numeric magnitude.
function _cartesian_vectors2(a,b)
    length(a)==length(b)==2 && all(x->x isa Real && UTDKernels._number_isfinite(x),(a...,b...)) ||
        throw(ArgumentError("Cartesian vectors require two finite real components"))
    ax,ay,bx,by=promote(float(a[1]),float(a[2]),float(b[1]),float(b[2]))
    return (ax,ay),(bx,by)
end

function _cartesian_controls(values)
    low=Inf;high=-Inf;depth=0;bits=0
    for value in values
        l,h=UTDKernels._scalar_leaf_range(value)
        low=min(low,l);high=max(high,h)
        depth=max(depth,UTDKernels._shadow_ad_depth(value))
        bits=max(bits,precision(float(UTDKernels._primal_value(value))))
    end
    # A determinant has degree two; nested quotient/angle derivatives
    # increase the product budget. Only a demonstrated exponent or
    # conditioning risk selects the wider path for ordinary binary inputs.
    degree=2(depth+1)
    R=promote_type((typeof(float(UTDKernels._primal_value(v))) for v in values)...)
    risk=R===BigFloat || (isfinite(high) && degree*high>log2(floatmax(R))-16) ||
        (isfinite(low) && degree*low<log2(floatmin(R))+16)
    span=isfinite(low) && isfinite(high) ? max(zero(high),high-low) : 0
    work=max(256,2bits+ceil(Int,2span)+depth+64)
    work<=UTDKernels._SHADOW_MAX_PHASE_WORK_BITS || throw(ArgumentError(
        "Cartesian evaluation exceeds the bounded wide workspace"))
    return risk,work
end

function _cartesian_cross_native(a,b)
    first=a[1]*b[2];second=a[2]*b[1]
    value=first-second
    primal=UTDKernels._primal_value(value)
    R=typeof(float(primal))
    bound=8eps(R)*(abs(UTDKernels._primal_value(first))+abs(UTDKernels._primal_value(second)))
    return value,isfinite(primal) && abs(primal)>bound
end

function _cartesian_wide(evaluate,values,::Type{C}=typeof(sum(zero(value) for value in values))) where {C<:Number}
    _,bits=_cartesian_controls(values)
    return UTDKernels._shadow_stable_wide_value(C,bits,"Cartesian geometry") do
        evaluate(map(UTDKernels._faddeeva_widen_bigfloat,values))
    end
end

"""
    orientation2(a,b)

Sign `-1`, `0`, or `1` of the exact represented Cartesian determinant
`a[1]*b[2]-a[2]*b[1]`. Vector magnitudes are immaterial; this discrete
predicate does not differentiate visibility. Zero vectors give sign zero.
"""
function orientation2(a,b)
    a,b=_cartesian_vectors2(a,b)
    value,certified=_cartesian_cross_native(a,b)
    certified && return Int(sign(UTDKernels._primal_value(value)))
    values=UTDKernels._primal_value.((a...,b...))
    _,bits=_cartesian_controls(values)
    return setprecision(BigFloat,bits) do
        ax,ay,bx,by=BigFloat.(values)
        Int(sign(ax*by-ay*bx))
    end
end

# Internal determinant magnitudes retain their physical units and AD
# coefficients. Public predicates/angles/ratios should use their own owners.
function _cross2(a,b)
    a,b=_cartesian_vectors2(a,b)
    values=(a...,b...)
    value,certified=_cartesian_cross_native(a,b)
    risk,_=_cartesian_controls(values)
    certified && !risk && UTDKernels._number_isfinite(value) && return value
    return _cartesian_wide(values) do x
        x[1]*x[4]-x[2]*x[3]
    end
end

"""
    signed_angle2(a,b)

Oriented angle from nonzero vector `a` to `b`, in `[-pi,pi]`. Products
retain their original Cartesian inputs near a pole; finite vector rescaling
cannot turn an actual side into a zero or an `Inf/Inf` angle. Supported AD
coefficients are retained; wider arithmetic is bounded and range-selected.
"""
function signed_angle2(a,b)
    a,b=_cartesian_vectors2(a,b)
    all(v->max(abs(UTDKernels._primal_value(v[1])),abs(UTDKernels._primal_value(v[2])))>0,(a,b)) ||
        throw(ArgumentError("signed angle requires nonzero vectors"))
    values=(a...,b...)
    determinant,certified=_cartesian_cross_native(a,b)
    product=a[1]*b[1]+a[2]*b[2]
    risk,_=_cartesian_controls(values)
    if certified && !risk && UTDKernels._number_isfinite(determinant) && UTDKernels._number_isfinite(product)
        result=atan(determinant,product)
        UTDKernels._number_isfinite(result) && return result
    end
    return _cartesian_wide(values) do x
        atan(x[1]*x[4]-x[2]*x[3],x[1]*x[3]+x[2]*x[4])
    end
end

"""
    crossing_fraction2(a,b,c,d)

Return `cross(a,b)/(cross(a,b)-cross(c,d))` without forming a ratio of
underflowed or overflowing determinants. Both determinants retain the same
physical units; a zero denominator is rejected. This fraction supports AD.
"""
function crossing_fraction2(a,b,c,d)
    a,b=_cartesian_vectors2(a,b);c,d=_cartesian_vectors2(c,d)
    values=promote(a...,b...,c...,d...)
    a,b,c,d=(values[1],values[2]),(values[3],values[4]),(values[5],values[6]),(values[7],values[8])
    first,first_ok=_cartesian_cross_native(a,b)
    second,second_ok=_cartesian_cross_native(c,d)
    denominator=first-second
    risk,_=_cartesian_controls(values)
    condition=abs(UTDKernels._primal_value(denominator))>
        32eps(typeof(float(UTDKernels._primal_value(denominator))))*
        (abs(UTDKernels._primal_value(first))+abs(UTDKernels._primal_value(second)))
    if first_ok && second_ok && condition && !risk &&
       all(UTDKernels._number_isfinite,(first,second,denominator))
        result=first/denominator
        UTDKernels._number_isfinite(result) && return result
    end
    return _cartesian_wide(values) do x
        first_w=x[1]*x[4]-x[2]*x[3]
        second_w=x[5]*x[8]-x[6]*x[7]
        denominator_w=first_w-second_w
        UTDKernels._primal_iszero(denominator_w) && throw(DomainError(
            denominator_w,"crossing fraction requires distinct determinants"))
        first_w/denominator_w
    end
end

"""Unit direction from endpoint `a` to distinct endpoint `b`, retaining
promoted inputs and supported AD coefficients across coordinate scales."""
unit_direction2(a,b)=UTDKernels._study_unit_direction2(a,b)
