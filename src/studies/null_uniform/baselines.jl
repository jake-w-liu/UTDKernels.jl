# Established-asymptotics comparators for the canonical saddle--pole
# integral, evaluated against the same normalization as the moment
# hierarchy.  These are the baselines required for the matched-order
# numerical comparison: the classical pole-extraction construction
# (Van der Waerden's method, as presented by Temme), its matched-
# information linearization, and the nonuniform separated-pole slope
# evaluation used by slope-diffraction formulas.

# Exact-zero coefficients include every AD seed; pole singularities instead
# depend only on primal coordinates.
_baseline_exactzero(x::Number)=UTDKernels._scalar_exact_zero(real(x)) && UTDKernels._scalar_exact_zero(imag(x))
_baseline_primalzero(x::Number)=iszero(UTDKernels._primal_value(real(x))) && iszero(UTDKernels._primal_value(imag(x)))

function _baseline_work_precision(values; zero_pole=false)
    low=Inf;high=-Inf;stored=0;depth=0
    for value in values,part in (real(value),imag(value))
        l,h=UTDKernels._scalar_leaf_range(part)
        low=min(low,l);high=max(high,h)
        stored=max(stored,precision(float(UTDKernels._primal_value(part))))
        depth=max(depth,UTDKernels._shadow_ad_depth(part))
    end
    span=isfinite(low) && isfinite(high) ? max(zero(high),high-low) : 0
    extent=isfinite(low) && isfinite(high) ? max(abs(low),abs(high)) : 0
    required=max(256,2stored+2span+64depth,2stored+2extent+64depth)
    # Zero-pole differentiation subtracts callback values. This rare limit
    # reserves enough precision for the full binary64 coefficient range.
    zero_pole && (required=max(required,4096))
    required<=UTDKernels._SHADOW_MAX_PHASE_WORK_BITS || throw(ArgumentError(
        "canonical baseline exceeds the bounded wide workspace"))
    return ceil(Int,required)
end

function _baseline_cancellation(value::Number,left::Number,right::Number)
    primal(x)=complex(UTDKernels._primal_value(real(x)),UTDKernels._primal_value(imag(x)))
    v,a,b=primal.((value,left,right))
    scale=max(abs(a),abs(b))
    iszero(scale) && return false
    R=typeof(float(real(v)))
    # Leave roundoff margin before accepting a strongly cancelled pole/saddle
    # difference. Ordinary well-conditioned scalar comparisons stay native.
    return abs(v)<=16sqrt(sqrt(eps(R)))*scale
end

function _baseline_keeps_precision(value::Real,bits)
    if hasproperty(value,:partials)
        return _baseline_keeps_precision(getproperty(value,:value),bits) &&
            all(x->_baseline_keeps_precision(x,bits),getproperty(value,:partials))
    end
    return value isa BigFloat && precision(value)>=bits
end
_baseline_keeps_precision(value::Complex,bits)=
    _baseline_keeps_precision(real(value),bits) && _baseline_keeps_precision(imag(value),bits)

function _baseline_derivative_agrees(a::Real,b::Real,tolerance)
    aa=hasproperty(a,:partials);bb=hasproperty(b,:partials)
    aa==bb || return false
    if aa
        _baseline_derivative_agrees(getproperty(a,:value),getproperty(b,:value),tolerance) || return false
        pa,pb=getproperty(a,:partials),getproperty(b,:partials)
        length(pa)==length(pb) || return false
        return all(_baseline_derivative_agrees(x,y,tolerance) for (x,y) in zip(pa,pb))
    end
    scale=max(abs(a),abs(b))
    return iszero(scale) || abs(a-b)<=tolerance*scale
end
_baseline_derivative_agrees(a::Complex,b::Complex,tolerance)=
    _baseline_derivative_agrees(real(a),real(b),tolerance) && _baseline_derivative_agrees(imag(a),imag(b),tolerance)

function _baseline_zero_divided_difference(a_fn,q,relative_tolerance,project)
    # B(0,q)=integral_0^1 a'(tau*q)dtau. Gauss integration retains the whole
    # nilpotent q jet. Richardson removes the central differences' h^2 term;
    # independent real/imaginary directions detect unresolved local behavior.
    order=max(1,cld(UTDKernels._shadow_ad_depth(q)+1,2))
    nodes,weights=UTDKernels._cached_gauss_legendre_rule(order)
    bits=precision(BigFloat)
    previous_real=nothing;previous_imag=nothing;previous_projected=nothing
    richardson_factor=BigFloat(2)^64
    for refinement in 0:63
        step=BigFloat(2)^(-div(bits,4)-32refinement)
        real_sum=zero(q);imag_sum=zero(q)
        for (node,weight) in zip(nodes,weights)
            point=((BigFloat(node)+1)/2)*q
            samples=map((point+step,point-step,point+im*step,point-im*step)) do x
                value=_as_complex(a_fn(x),"amplitude value")
                _baseline_keeps_precision(value,bits) || throw(ArgumentError(
                    "zero-pole amplitude callbacks must retain widened numeric precision; use a type-generic callback such as one(t)"))
                value
            end
            real_sum+=(BigFloat(weight)/2)*((samples[1]-samples[2])/(2step))
            imag_sum+=(BigFloat(weight)/2)*((samples[3]-samples[4])/(2im*step))
        end
        real_derivative=previous_real===nothing ? real_sum :
            (richardson_factor*real_sum-previous_real)/(richardson_factor-1)
        imag_derivative=previous_imag===nothing ? imag_sum :
            (richardson_factor*imag_sum-previous_imag)/(richardson_factor-1)
        estimate=(real_derivative+imag_derivative)/2
        projected=project(estimate)
        if UTDKernels._number_isfinite(projected) && previous_projected!==nothing &&
           UTDKernels._number_isequal_all(projected,previous_projected) &&
           _baseline_derivative_agrees(real_derivative,imag_derivative,relative_tolerance)
            return estimate
        end
        previous_projected=UTDKernels._number_isfinite(projected) ? projected : nothing
        previous_real=real_sum;previous_imag=imag_sum
    end
    throw(ArgumentError("analytic amplitude derivative did not resolve within the bounded step refinement"))
end

function _classical_pole_wide(::Type{C},k,z,a_fn,a0) where {C<:Number}
    work=_baseline_work_precision((k,z,a0);zero_pole=_baseline_primalzero(z))
    R=UTDKernels._shadow_binary_type(real(k),real(z),imag(z))
    derivative_tolerance=8eps(R)
    return UTDKernels._shadow_stable_wide_value(C,work,"classical pole extraction") do
        kw=UTDKernels._faddeeva_widen_bigfloat(k);zw=UTDKernels._faddeeva_widen_bigfloat(z)
        q=zw/sqrt(kw)
        aq=_as_complex(a_fn(q),"pole amplitude")
        saddle=_as_complex(a_fn(zero(q)),"saddle amplitude")
        numerator=aq-saddle
        w0=UTDKernels._null_wide_faddeeva(zw)
        normalizer=sqrt(BigFloat(pi))*sqrt(kw)
        finish=B->aq*w0-im*B/normalizer
        B=_baseline_primalzero(q) ? _baseline_zero_divided_difference(
            a_fn,q,derivative_tolerance,B->convert(C,finish(B))) :
            _baseline_exactzero(numerator) ? zero(q+numerator) : numerator/q
        finish(B)
    end
end

"""
    classical_pole_term(k, z, a_fn) -> I_cl

Classical pole extraction with the first regular saddle term for the
canonical integral

    I(k,z) = (i/π) ∫ a(u/√k) e^{-u²} / (z - u) du,   Im z > 0.

With q = z/√k and B(t,q) = [a(t) - a(q)]/(t - q), the construction is

    I_cl = a(q) w(z) - i B(0,q) / √(π k),

which uses the full amplitude function (strictly more information than
the two coefficients a_0, a_1).  The continuous derivative limit is used
at q = 0. At that exact limit, callbacks must retain the widened working
numeric type (including AD coefficients); use `t -> one(t)` rather than a
fixed-precision literal callback for a constant amplitude. Unresolved or
precision-erasing zero-pole callbacks raise an explicit error.
"""
function classical_pole_term(k::Real,z::Number,a_fn)
    kf=_positive_real(k,"k");zz=_as_complex(z)
    raw_saddle=a_fn(zero(zz)+complex(zero(kf)))
    raw_saddle isa Number || throw(ArgumentError("saddle amplitude must be numeric"))
    a0=complex(float(raw_saddle))
    C=promote_type(typeof(zz),typeof(complex(kf)),typeof(a0))
    UTDKernels._number_isfinite(a0) || return _classical_pole_wide(C,kf,zz,a_fn,zero(a0))
    if any(UTDKernels._number_contains_ad,(kf,zz,a0)) || _baseline_primalzero(zz)
        return _classical_pole_wide(C,kf,zz,a_fn,a0)
    end
    q=zz/sqrt(kf)
    if !UTDKernels._number_isfinite(q) || _baseline_primalzero(q)
        return _classical_pole_wide(C,kf,zz,a_fn,a0)
    end
    raw_pole=a_fn(q)
    raw_pole isa Number || throw(ArgumentError("pole amplitude must be numeric"))
    aq=complex(float(raw_pole))
    C=promote_type(C,typeof(aq))
    UTDKernels._number_isfinite(aq) || return _classical_pole_wide(C,kf,zz,a_fn,a0)
    UTDKernels._number_contains_ad(aq) && return _classical_pole_wide(C,kf,zz,a_fn,a0)
    numerator=aq-a0
    R=typeof(float(UTDKernels._primal_value(real(q))))
    if abs(numerator)<=32sqrt(eps(R))*max(abs(aq),abs(a0))
        return _classical_pole_wide(C,kf,zz,a_fn,a0)
    end
    regular=(numerator/q)/(sqrt(oftype(kf,pi))*sqrt(kf))
    pole_value=aq*faddeeva_moments(zz,0)[1]
    saddle_value=im*regular
    value=pole_value-saddle_value
    if !all(UTDKernels._number_isfinite,(regular,pole_value,value)) ||
       _baseline_exactzero(regular) || _baseline_cancellation(value,pole_value,saddle_value)
        return _classical_pole_wide(C,kf,zz,a_fn,a0)
    end
    return value
end

"""
    linearized_pole_term(k, z, a0, a1) -> I_lin

Matched-information reduction of [`classical_pole_term`](@ref): the
amplitude is replaced by its saddle tangent `a(t) ≈ a_0 + a_1 t`, so
`a(q) ≈ a_0 + a_1 z/√k` and `B(0,q) ≈ a_1`.  The result is algebraically
identical to the two-moment truncation `a_0 W_0 + a_1 k^{-1/2} W_1`
because `W_1 = z W_0 - i/√π`; it is evaluated through the pole-extraction
formula itself so that the equivalence can be checked numerically.
"""
function linearized_pole_term(k::Real,z::Number,a0::Number,a1::Number)
    kf=_positive_real(k,"k");zz=_as_complex(z)
    a0c=_as_complex(a0,"a0");a1c=_as_complex(a1,"a1")
    C=promote_type(typeof(zz),typeof(complex(kf)),typeof(a0c),typeof(a1c))
    if _baseline_exactzero(a1c)
        constant=convert(C,a0c*faddeeva_moments(zz,0)[1])
        UTDKernels._number_isfinite(constant) && return constant
    end
    if !any(UTDKernels._number_contains_ad,(kf,zz,a0c,a1c))
        q=zz/sqrt(kf)
        regular=a1c/(sqrt(oftype(kf,pi))*sqrt(kf))
        pole=a0c+a1c*q
        pole_value=pole*faddeeva_moments(zz,0)[1]
        saddle_value=im*regular
        value=pole_value-saddle_value
        if all(UTDKernels._number_isfinite,(q,regular,pole,value)) && !_baseline_exactzero(regular) &&
           !_baseline_cancellation(value,pole_value,saddle_value)
            return value
        end
    end
    work=_baseline_work_precision((kf,zz,a0c,a1c))
    return UTDKernels._shadow_stable_wide_value(C,work,"linearized pole extraction") do
        kw,zw,aw,bw=UTDKernels._faddeeva_widen_bigfloat.((kf,zz,a0c,a1c))
        (aw+bw*(zw/sqrt(kw)))*UTDKernels._null_wide_faddeeva(zw)-im*bw/(sqrt(BigFloat(pi))*sqrt(kw))
    end
end

"""
    slope_separated_term(k, z, a0, a1) -> I_sep

Nonuniform separated-pole evaluation of the first two canonical
contributions: the frozen-amplitude term `a_0 W_0` plus the slope term
with `W_1` replaced by its leading separated-pole form
`i / (2√π z²)`.  This is the canonical analog of adding a slope-
diffraction correction to the value term; it uses the same two
coefficients as the two-moment truncation but is nonuniform as
`z -> 0`, where the pole reaches the saddle.
"""
function slope_separated_term(k::Real,z::Number,a0::Number,a1::Number)
    kf=_positive_real(k,"k");zz=_as_complex(z)
    _baseline_primalzero(zz) && throw(ArgumentError("separated slope term is singular at z = 0"))
    a0c=_as_complex(a0,"a0");a1c=_as_complex(a1,"a1")
    C=promote_type(typeof(zz),typeof(complex(kf)),typeof(a0c),typeof(a1c))
    if _baseline_exactzero(a1c)
        constant=convert(C,a0c*faddeeva_moments(zz,0)[1])
        UTDKernels._number_isfinite(constant) && return constant
    end
    if !any(UTDKernels._number_contains_ad,(kf,zz,a0c,a1c))
        scaled=(im*a1c/2)/(sqrt(oftype(kf,pi))*sqrt(kf))
        slope=(scaled/zz)/zz
        value_term=a0c*faddeeva_moments(zz,0)[1]
        value=value_term+slope
        if all(UTDKernels._number_isfinite,(scaled,slope,value)) &&
           !_baseline_exactzero(scaled) && !_baseline_exactzero(slope) &&
           !_baseline_cancellation(value,value_term,-slope)
            return value
        end
    end
    work=_baseline_work_precision((kf,zz,a0c,a1c))
    return UTDKernels._shadow_stable_wide_value(C,work,"separated slope term") do
        kw,zw,aw,bw=UTDKernels._faddeeva_widen_bigfloat.((kf,zz,a0c,a1c))
        aw*UTDKernels._null_wide_faddeeva(zw)+(im*bw/(2sqrt(BigFloat(pi))*sqrt(kw))/zw)/zw
    end
end

"""
    nonpolynomial_amplitude(k, Λ, β) -> t -> a(t)

Exactly factored nonpolynomial moving-null amplitude
`a(t) = (t + Λ/√k) e^{βt}`.  Its zero is exactly at `t = -Λ/√k` for
every finite `k`, and it has infinitely many nonzero Taylor
coefficients, so any finite moment truncation leaves a genuine analytic
remainder.
"""
function nonpolynomial_amplitude(k::Real, Lambda::Number, beta::Real)
    kf = _positive_real(k, "k")
    lam = _as_complex(Lambda, "Lambda")
    UTDKernels._number_isfinite(beta) || throw(ArgumentError("beta must be finite"))
    bf=float(beta)
    return function(t::Number)
        working=t+zero(lam)+zero(kf)+zero(bf)
        kw=kf+zero(real(working));lw=lam+zero(working);bw=bf+zero(real(working))
        offset=_baseline_exactzero(lam) ? zero(working) : lw/sqrt(kw)
        carrier=_baseline_exactzero(bf) ? one(working) : exp(bw*working)
        (working+offset)*carrier
    end
end

"""
    nonpolynomial_closed_form(k, z, Λ, β) -> I

Closed form of the canonical integral for the amplitude
`a(t) = (t + Λ/√k) e^{βt}` obtained by completing the square in the
exponent.  With `s = β/(2√k)`,

    I = e^{s²} [ (Λ/√k + β/(2k)) w(z - s) + k^{-1/2} W_1(z - s) ]
      = (e^{s²}/√k) [ (z + Λ) w(z - s) - i/√π ].

Both displayed forms are algebraically identical through
`W_1 = z w - i/√π`; the first is evaluated with its common scale factored.
The shift `s` is real, so
`Im(z - s) = Im z > 0` keeps the defining sheet.
"""
function nonpolynomial_closed_form(k::Real,z::Number,Lambda::Number,beta::Real)
    kf=_positive_real(k,"k");zz=_as_complex(z);lam=_as_complex(Lambda,"Lambda")
    UTDKernels._number_isfinite(beta) || throw(ArgumentError("beta must be finite"))
    bf=float(beta)
    C=promote_type(typeof(complex(kf)),typeof(zz),typeof(lam),typeof(complex(bf)))
    if !any(UTDKernels._number_contains_ad,(kf,zz,lam,bf))
        shift=bf/(2sqrt(kf));coordinate=zz-shift
        if all(UTDKernels._number_isfinite,(shift,coordinate))
            growth=exp(shift*shift)
            if UTDKernels._number_isfinite(growth)
                moments=faddeeva_moments(coordinate,1)
                value_term=(lam+shift)*moments[1]
                bracket=value_term+moments[2]
                numerator=growth*bracket
                value=numerator/sqrt(kf)
                if all(UTDKernels._number_isfinite,(bracket,numerator,value)) &&
                   !(_baseline_exactzero(numerator) && !_baseline_exactzero(bracket)) &&
                   !_baseline_cancellation(bracket,value_term,-moments[2])
                    return value
                end
            end
        end
    end
    R=UTDKernels._shadow_binary_type(kf,real(zz),imag(zz),bf)
    work=_baseline_work_precision((kf,zz,lam,bf))
    return UTDKernels._shadow_stable_wide_value(C,work,"nonpolynomial canonical field") do
        kw,zw,lw,bw=UTDKernels._faddeeva_widen_bigfloat.((kf,zz,lam,bf))
        shift=bw/(2sqrt(kw))
        moments=UTDKernels._null_wide_moments(zw-shift,1;input_precision=precision(R))
        exp(shift*shift)*(((lw+shift)*moments[1]+moments[2])/sqrt(kw))
    end
end
