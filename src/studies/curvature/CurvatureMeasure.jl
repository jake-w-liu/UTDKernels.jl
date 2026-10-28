"""
Curvature-measure limit for intrinsic UTD seam sums.

All phasors use exp(+i omega t); outgoing propagation is exp(-i k r).
The module implements the scalar canonical quantities used in the paper.
"""
module CurvatureMeasure

using SpecialFunctions
import ..UTDKernels
using ..UTDKernels: F_utd, intrinsic_seam_coefficient, curvature_continuum_harmonic, curvature_measure_sum, debiased_curvature_measure_sum

export transition_F, intrinsic_coefficient, linear_prefactor
export local_bias_factor, local_bias_series
export continuum_harmonic, polygon_measure_sum, polygon_alias_sum
export normalized_seam_sum, physical_seam_sum, debiased_measure_sum
export smooth_amplitude, smooth_continuum, smooth_polygon_sum
export relative_error, fit_log_slope
export ErrorDecomposition, error_decomposition
export nonuniform_turning_partition, nonuniform_measure_sum
export polygon_turning_angles, ellipse_points, ellipse_curvature_density
export ellipse_field_sum, ellipse_continuum_reference

include("MonostaticAssembly.jl")
using .MonostaticAssembly
export MonostaticAssembly

function _finite_phase(z::Real, phi::Real)
    zf = float(z)
    pf = float(phi)
    isfinite(zf) && isfinite(pf) ||
        throw(ArgumentError("z and phi must be finite"))
    return zf, pf
end

function _positive_finite_kL(k::Real, L::Real)
    kf = float(k)
    Lf = float(L)
    isfinite(kf) && isfinite(Lf) && kf > 0.0 && Lf > 0.0 ||
        throw(ArgumentError("k and L must be positive and finite"))
    return kf, Lf
end

"""
    transition_F(x)

Kouyoumjian–Pathak transition function for finite real `x >= 0`:

```
F(x) = sqrt(pi x) exp(i pi/4) erfcx(exp(i pi/4) sqrt(x))
```

The formula is continuous at `x = 0`, where `F(0) = 0`.
"""
function transition_F(x::Real)
    xf = float(x)
    isfinite(xf) && xf >= 0.0 ||
        throw(ArgumentError("transition_F requires finite x >= 0"))
    return F_utd(xf)
end

function transition_F(x::AbstractArray)
    return transition_F.(x)
end

"""
    intrinsic_coefficient(delta, k, L)

Intrinsic near-coplanar reflection-boundary wedge coefficient.

`delta = alpha - pi > 0`, `n = alpha/pi = 1 + delta/pi`.
"""
function intrinsic_coefficient(delta::Real, k::Real, L::Real)
    all(isfinite, (delta, k, L)) && delta > 0 && k > 0 && L > 0 ||
        throw(ArgumentError("delta, k, and L must be positive and finite"))
    delta <= UTDKernels._typed_pi(delta) || throw(ArgumentError("delta must not exceed pi"))
    return intrinsic_seam_coefficient(delta, k, L)
end

"""
    linear_prefactor(k, L)

Coefficient `A` in `D_e(delta; k, L) = A(k, L) delta + O(delta^2)`.
"""
function linear_prefactor(k::Real, L::Real)
    isfinite(k) && isfinite(L) && k > 0 && L > 0 ||
        throw(ArgumentError("k and L must be positive and finite"))
    return UTDKernels._intrinsic_seam_linear_prefactor(k, L)
end

"""
    local_bias_factor(delta)

Exact ratio `D_e / (A delta)`, independent of `k` and `L`.
"""
function local_bias_factor(delta::Real)
    isfinite(delta) && delta > 0 ||
        throw(ArgumentError("delta must be positive and finite"))
    delta <= UTDKernels._typed_pi(delta) || throw(ArgumentError("delta must not exceed pi"))
    return UTDKernels._curvature_local_bias(delta)
end

"""
    local_bias_series(delta)

Second-order small-delta expansion of `local_bias_factor`.
"""
local_bias_series(delta::Real) = UTDKernels._curvature_local_bias_series(delta)

"""
    continuum_harmonic(m, z, phi)

Integral `int_0^{2pi} exp(i m theta) exp[-i z cos(theta-phi)] d theta`.
The phase parameters `z` and `phi` must be finite.
"""
# Python (-1j)**m; Complex{Int} negative powers are not defined in Julia.
neg_im_power(m::Integer) = (-1.0im)^m

function continuum_harmonic(m::Integer, z::Real, phi::Real)
    zf, pf = _finite_phase(z, phi)
    return curvature_continuum_harmonic(m, zf, pf)
end

# Harmonics on a regular N-node turning mesh are exactly periodic modulo N.
# A centered residue avoids losing the integer mode in a large floating phase.
@inline function _discrete_harmonic(N::Integer,m::Integer)
    mode=mod(m,N)
    return Int(mode>N÷2 ? mode-N : mode)
end

"""
    polygon_measure_sum(N, m, z, phi)

Regular-`N` discrete curvature-measure sum with `delta = 2pi/N`.
The phase parameters `z` and `phi` must be finite.
"""
function polygon_measure_sum(N::Integer, m::Integer, z::Real, phi::Real)
    N < 3 && throw(ArgumentError("N must be at least 3"))
    N=Int(N)
    zf, pf = _finite_phase(z, phi)
    delta = 2 * pi / N
    mode=oftype(delta,_discrete_harmonic(N,m))
    acc = zero(complex(zf+pf+delta))
    @inbounds for j in 0:(N - 1)
        theta = 2 * pi * j / N
        acc += exp(im * mode * theta - im * zf * cos(theta - pf))
    end
    return delta * acc
end

"""
    polygon_alias_sum(N, m, z, phi; ell_max=20)

Truncated exact Bessel-alias representation of `polygon_measure_sum`.
"""
function polygon_alias_sum(N::Integer, m::Integer, z::Real, phi::Real; ell_max::Integer=20)
    N >= 3 || throw(ArgumentError("N must be at least 3"))
    ell_max >= 0 || throw(ArgumentError("ell_max must be nonnegative"))
    N=Int(N)
    ell_max=Int(ell_max)
    zf, pf = _finite_phase(z, phi)
    total = 0.0 + 0.0im
    mode=_discrete_harmonic(N,m)
    @inbounds for ell in (-ell_max):ell_max
        order = widemul(ell,N)-mode
        total += neg_im_power(order) * besselj(order, zf) * exp(-im * order * pf)
    end
    return 2 * pi * total
end

"""
    normalized_seam_sum(N, m, z, phi, k, L)

Seam sum divided by `A(k, L)`; equals `beta_N` times the measure sum.
The cancelled normalization still requires positive finite `k` and `L`.
"""
function normalized_seam_sum(N::Integer, m::Integer, z::Real, phi::Real, k::Real, L::Real)
    N >= 3 || throw(ArgumentError("N must be at least 3"))
    _positive_finite_kL(k, L)
    delta = 2 * pi / N
    beta = UTDKernels._curvature_local_bias(delta)
    return beta * polygon_measure_sum(N, m, z, phi)
end

"""
    physical_seam_sum(N, m, z, phi, k, L)

Physical sum of intrinsic coefficients over a regular polygon.
"""
function physical_seam_sum(N::Integer, m::Integer, z::Real, phi::Real, k::Real, L::Real)
    N >= 3 || throw(ArgumentError("N must be at least 3"))
    N=Int(N)
    delta=2*pi/N
    coefficient=intrinsic_seam_coefficient(delta,k,L)
    return coefficient*(polygon_measure_sum(N,m,z,phi)/delta)
end

"""
    debiased_measure_sum(N, m, z, phi, k, L)

Remove the exact local wedge bias from the normalized seam sum.
"""
function debiased_measure_sum(N::Integer, m::Integer, z::Real, phi::Real, k::Real, L::Real)
    delta = 2 * pi / N
    return normalized_seam_sum(N, m, z, phi, k, L) /
           UTDKernels._curvature_local_bias(delta)
end

"""
    smooth_amplitude(theta)

Nonuniform smooth angular amplitude used in the numerical study.
"""
function smooth_amplitude(theta::Real)
    return 1 + 0.35 * cos(2 * theta) - 0.20 * sin(3 * theta) + 0.12 * cos(5 * theta)
end

smooth_amplitude(theta::AbstractArray) = smooth_amplitude.(theta)

"""
    smooth_continuum(z, phi)

Analytical continuum integral for `smooth_amplitude`.
"""
function smooth_continuum(z::Real, phi::Real)
    zf, pf = _finite_phase(z, phi)
    # Fourier coefficients of smooth_amplitude.
    coeffs = Dict{Int,ComplexF64}(
        0 => 1.0 + 0.0im,
        2 => 0.35 / 2,
        -2 => 0.35 / 2,
        3 => 0.20im / 2,
        -3 => -0.20im / 2,
        5 => 0.12 / 2,
        -5 => 0.12 / 2,
    )
    acc = 0.0 + 0.0im
    for (m, c) in coeffs
        acc += c * continuum_harmonic(m, zf, pf)
    end
    return acc
end

function smooth_polygon_sum(N::Integer, z::Real, phi::Real)
    N < 3 && throw(ArgumentError("N must be at least 3"))
    zf, pf = _finite_phase(z, phi)
    delta = 2 * pi / N
    acc = 0.0 + 0.0im
    @inbounds for j in 0:(N - 1)
        theta = 2 * pi * j / N
        acc += smooth_amplitude(theta) * exp(-im * zf * cos(theta - pf))
    end
    return delta * acc
end

relative_error(value::Number, reference::Number; floor::Real=1e-30) =
    abs(value - reference) / max(abs(reference), floor)

function fit_log_slope(x, y)
    xs = collect(float(x))
    ys = collect(float(y))
    length(xs) == length(ys) ||
        throw(ArgumentError("x and y must have the same number of samples"))
    mask = (xs .> 0) .& (ys .> 0) .& isfinite.(xs) .& isfinite.(ys)
    count(mask) < 2 && throw(ArgumentError("at least two positive finite samples are required"))
    lx = log.(xs[mask])
    ly = log.(ys[mask])
    mx = sum(lx) / length(lx)
    my = sum(ly) / length(ly)
    denominator = sum((lx .- mx) .^ 2)
    denominator > 0.0 ||
        throw(ArgumentError("positive finite x samples must contain at least two distinct values"))
    return sum((lx .- mx) .* (ly .- my)) / denominator
end

struct ErrorDecomposition{T<:Number}
    total::T
    local_bias::T
    aliasing::T
    residual::T
end

"""
    error_decomposition(N, m, z, phi, k, L)

Exact decomposition of normalized total error:

```
beta * S_N - I = (beta - 1) I + beta (S_N - I)
```
"""
function error_decomposition(N::Integer, m::Integer, z::Real, phi::Real, k::Real, L::Real)
    _positive_finite_kL(k, L)
    delta = 2 * pi / N
    beta = local_bias_factor(delta)
    S = polygon_measure_sum(N, m, z, phi)
    I_cont = continuum_harmonic(m, z, phi)
    total = beta * S - I_cont
    local_term = (beta - 1) * I_cont
    alias = beta * (S - I_cont)
    return ErrorDecomposition(total, local_term, alias, total - local_term - alias)
end

"""
    nonuniform_turning_partition(N; modulation=0.35)

Positive nonuniform turning-angle cells and centered normal angles.

The weights sum to `2 pi`. The nodes are the centers of the cumulative
turning-angle intervals, corresponding to a centered vertex-normal sample.
"""
function nonuniform_turning_partition(N::Integer; modulation::Real=0.35)
    N < 4 && throw(ArgumentError("N must be at least 4"))
    modulation_f = float(modulation)
    isfinite(modulation_f) ||
        throw(ArgumentError("partition modulation must be finite"))
    raw = Vector{typeof(modulation_f)}(undef, N)
    @inbounds for j in 0:(N - 1)
        raw[j + 1] =
            1 + modulation_f * cos(2 * pi * (j + 0.5) / N) +
            0.12 * sin(6 * pi * (j + 0.5) / N)
    end
    any(<=(0), raw) && throw(ArgumentError("partition modulation produced nonpositive weights"))
    delta = 2 * pi .* raw ./ sum(raw)
    edges = similar(delta, N + 1)
    edges[1] = 0.0
    acc = 0.0
    @inbounds for j in 1:N
        acc += delta[j]
        edges[j + 1] = acc
    end
    theta = edges[1:N] .+ 0.5 .* delta
    return delta, theta
end

"""
    nonuniform_measure_sum(N, z, phi; debias=true)

Centered nonuniform curvature sum for `smooth_amplitude`.
"""
function nonuniform_measure_sum(N::Integer, z::Real, phi::Real; debias::Bool=true)
    zf, pf = _finite_phase(z, phi)
    delta, theta = nonuniform_turning_partition(N)
    amplitudes = Vector{typeof(complex(zf + pf + first(theta)))}(undef, length(theta))
    @inbounds for j in eachindex(delta)
        amplitudes[j] = smooth_amplitude(theta[j]) *
                        exp(-im * zf * cos(theta[j] - pf))
    end
    return debias ?
           debiased_curvature_measure_sum(delta, amplitudes) :
           curvature_measure_sum(delta, amplitudes)
end

"""
    polygon_turning_angles(points)

Positive exterior turning angles of a convex CCW polygon.
`points` has size `(N, 2)`.
"""
function polygon_turning_angles(points::AbstractMatrix{<:Real})
    size(points, 2) == 2 && size(points, 1) >= 3 ||
        throw(ArgumentError("points must have shape (N,2) with N >= 3"))
    all(isfinite, points) || throw(ArgumentError("polygon points must be finite"))
    N = size(points, 1)
    incoming = similar(float.(points))
    outgoing = similar(float.(points))
    @inbounds for i in 1:N
        im1 = i == 1 ? N : i - 1
        ip1 = i == N ? 1 : i + 1
        previous = (points[im1,1], points[im1,2])
        current = (points[i,1], points[i,2])
        following = (points[ip1,1], points[ip1,2])
        tin = UTDKernels._study_unit_direction2(previous, current)
        tout = UTDKernels._study_unit_direction2(current, following)
        incoming[i,1], incoming[i,2] = tin
        outgoing[i,1], outgoing[i,2] = tout
    end
    cross = incoming[:, 1] .* outgoing[:, 2] .- incoming[:, 2] .* outgoing[:, 1]
    dotp = incoming[:, 1] .* outgoing[:, 1] .+ incoming[:, 2] .* outgoing[:, 2]
    delta = atan.(cross, dotp)
    UTDKernels._study_validate_ccw_turns(delta)
    return delta
end

"""
    ellipse_points(N, a, b)

Uniform-parameter vertices of a counterclockwise ellipse.
"""
function ellipse_points(N::Integer, a::Real, b::Real)
    N >= 4 && isfinite(a) && isfinite(b) && a > 0 && b > 0 ||
        throw(ArgumentError("N >= 4 and positive finite semiaxes are required"))
    t = 2 * pi .* (0:(N - 1)) ./ N
    points = hcat(a .* cos.(t), b .* sin.(t))
    return t, points
end

function _ellipse_density_scaled(t::Real, a::Real, b::Real)
    scale = max(a, b)
    an, bn = a / scale, b / scale
    denominator = hypot(an * sin(t), bn * cos(t))
    return ((an * bn) / denominator) / denominator
end

function _ellipse_density_scalar(t::Real, a::Real, b::Real)
    tf, af, bf = promote(float(t), float(a), float(b))
    ratio = min(af, bf) / max(af, bf)
    value = _ellipse_density_scaled(tf, af, bf)
    if iszero(UTDKernels._primal_value(ratio)) || !UTDKernels._number_isfinite(value)
        C = typeof(tf + af + bf)
        return UTDKernels._shadow_stable_wide_value(C, 192, "ellipse curvature density") do
            _ellipse_density_scaled(
                UTDKernels._faddeeva_widen_bigfloat(tf),
                UTDKernels._faddeeva_widen_bigfloat(af),
                UTDKernels._faddeeva_widen_bigfloat(bf),
            )
        end
    end
    return value
end

"""
    ellipse_curvature_density(t, a, b)

`d theta / d t` = curvature times arclength density for an ellipse.
"""
function ellipse_curvature_density(t, a::Real, b::Real)
    isfinite(a) && isfinite(b) && a > 0 && b > 0 ||
        throw(ArgumentError("positive finite semiaxes are required"))
    if t isa Real
        isfinite(t) || throw(ArgumentError("ellipse parameters must be finite"))
        return _ellipse_density_scalar(t, a, b)
    end
    t isa AbstractArray && all(x -> x isa Real && isfinite(x), t) ||
        throw(ArgumentError("ellipse parameters must be finite real scalars or arrays"))
    return _ellipse_density_scalar.(t, Ref(a), Ref(b))
end

function ellipse_field_sum(N::Integer, a::Real, b::Real, k::Real, phi::Real; debias::Bool=true)
    kf = float(k)
    pf = float(phi)
    isfinite(kf) && isfinite(pf) ||
        throw(ArgumentError("k and phi must be finite"))
    t_nodes, points = ellipse_points(N, a, b)
    delta_nodes = polygon_turning_angles(points)
    qhat = (cos(pf), sin(pf))
    amplitudes = Vector{typeof(complex(kf + pf + first(delta_nodes)))}(undef, length(delta_nodes))
    @inbounds for j in eachindex(delta_nodes)
        amp = 1 + 0.25 * cos(3 * t_nodes[j]) - 0.15 * sin(2 * t_nodes[j])
        phase = UTDKernels._cis_product(-kf,points[j,1]*qhat[1]+points[j,2]*qhat[2])
        amplitudes[j] = amp * phase
    end
    field = debias ?
            debiased_curvature_measure_sum(delta_nodes, amplitudes) :
            curvature_measure_sum(delta_nodes, amplitudes)
    return field, maximum(delta_nodes), sum(delta_nodes)
end

"""
Dense Riemann-sum reference matching the source experiment (262144 nodes).
"""
function ellipse_continuum_reference(a::Real, b::Real, k::Real, phi::Real; n_dense::Integer=262144)
    isfinite(a) && isfinite(b) && a > 0 && b > 0 ||
        throw(ArgumentError("ellipse semiaxes must be positive and finite"))
    isfinite(k) && isfinite(phi) ||
        throw(ArgumentError("k and phi must be finite"))
    n_dense >= 3 || throw(ArgumentError("n_dense must be at least 3"))
    t = 2 * pi .* (0:(n_dense - 1)) ./ n_dense
    qhat = (cos(phi), sin(phi))
    dens = ellipse_curvature_density(t, a, b)
    acc = 0.0 + 0.0im
    @inbounds for j in eachindex(t)
        amp = 1 + 0.25 * cos(3 * t[j]) - 0.15 * sin(2 * t[j])
        x = a * cos(t[j])
        y = b * sin(t[j])
        acc += amp * UTDKernels._cis_product(-k,x*qhat[1]+y*qhat[2]) * dens[j]
    end
    return 2 * pi * acc / n_dense
end

end # module
