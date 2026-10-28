#!/usr/bin/env julia
# Generate paper/data CSVs and results_macros.tex from the central production modules.
# Seed 1352026.  All experiment inputs are persisted in the CSVs and in
# paper/data/run_manifest.txt so every reported number can be regenerated.
#
# Test amplitude: the exactly factored moving-null form
#   a(t) = (t + Lambda/sqrt(k)) * (1 + b1 t + b2 t^2 + b3 t^3),
# whose zero is exactly at t = -Lambda/sqrt(k) for every finite k.

include(joinpath(@__DIR__, "..", "..", "common.jl"))
using .PaperReproductionPaths
import UTDKernels, DiffUTD
using TOML
const BACKEND = selected_backend()
const SOURCES = source_snapshot(UTDKernels, DiffUTD)
BACKEND === :diffutd && include("backend_diffutd.jl")
const NUM = BACKEND === :kernels ? UTDKernels.NullUniform : SpatialStudy
using FastGaussQuadrature
using .NUM: DEFAULT_TAIL, DEFAULT_SWITCH
using .NUM: REFERENCE_HERMITE_ORDER
using .NUM: scale_normalized_error, polynomial_amplitude
using .NUM: canonical_quadrature, direct_moment, hp_moment
using .NUM: fitted_log_slope, factored_null_coefficients
using UTDKernels
using Printf
using Pkg
using Random
using Statistics
using TOML

const PAPER = output_directory(168)
const DATA = joinpath(PAPER, "data")
mkpath(DATA)

const RNG = MersenneTwister(1352026)
const BTAIL = DEFAULT_TAIL          # b = (0.35, -0.15, 0.08)
const SEED = 1352026

function csvnum(x::Real)
    v = Float64(x)
    return isnan(v) ? "NaN" : @sprintf("%.17g", v)
end

# `comments` are #-prefixed lines recording complete generator inputs.
function write_csv(path, header::Vector{String}, rows;
                   comments::Vector{String}=String[])
    open(path, "w") do io
        for c in comments
            println(io, "# ", c)
        end
        println(io, join(header, ","))
        for row in rows
            println(io, join(csvnum.(row), ","))
        end
    end
    return path
end

function latex_scientific(value::Real, digits::Integer=2)
    v = Float64(value)
    iszero(v) && return "0"
    mantissa, exponent = split(@sprintf("%.*e", digits, v), "e")
    return string(mantissa, "\\times 10^{", parse(Int, exponent), "}")
end

function term_scale(k, z, coeff)
    W = faddeeva_moments(z, length(coeff) - 1)
    s = 0.0
    @inbounds for m in 0:(length(coeff) - 1)
        s += abs(coeff[m + 1] * k^(-0.5 * m) * W[m + 1])
    end
    return s
end

# All canonical-map comparators for one (k, z, Lambda) sample.
function map_errors(k, z, coeff)
    a = polynomial_amplitude(coeff)
    exact = NUM.hierarchy_value(k, z, coeff)
    scale = term_scale(k, z, coeff)
    one_m = NUM.hierarchy_value(k, z, coeff; order=0)
    two_m = NUM.hierarchy_value(k, z, coeff; order=1)
    cl = NUM.classical_pole_term(k, z, a)
    cl_lin = NUM.linearized_pole_term(k, z, coeff[1], coeff[2])
    sep = imag(z) == 0 ? complex(NaN) :
          NUM.slope_separated_term(k, z, coeff[1], coeff[2])
    return (exact=exact, scale=scale,
            e0=scale_normalized_error(one_m, exact, scale),
            e1=scale_normalized_error(two_m, exact, scale),
            ecl=scale_normalized_error(cl, exact, scale),
            elin=scale_normalized_error(cl_lin, exact, scale),
            esep=scale_normalized_error(sep, exact, scale))
end

function null_layer_map()
    k = 400.0
    y = 0.35
    xs = collect(range(-3.0, 3.0; length=181))
    lambdas = collect(range(-4.0, 4.0; length=161))
    nx = length(xs)
    nl = length(lambdas)
    e0m = Matrix{Float64}(undef, nl, nx)
    e1m = Matrix{Float64}(undef, nl, nx)
    eclm = Matrix{Float64}(undef, nl, nx)
    esepm = Matrix{Float64}(undef, nl, nx)
    elinm = Matrix{Float64}(undef, nl, nx)
    exact_mag = Matrix{Float64}(undef, nl, nx)
    rows = Vector{Vector{Float64}}()
    sizehint!(rows, nx * nl)
    for (ix, x) in enumerate(xs)
        z = complex(x, y)
        for (il, lam) in enumerate(lambdas)
            coeff = factored_null_coefficients(k, lam; b=BTAIL)
            r = map_errors(k, z, coeff)
            e0m[il, ix] = r.e0
            e1m[il, ix] = r.e1
            eclm[il, ix] = r.ecl
            elinm[il, ix] = r.elin
            esepm[il, ix] = r.esep
            exact_mag[il, ix] = abs(r.exact)
            push!(rows, [x, lam, r.e0, r.e1, r.ecl, r.elin, r.esep,
                         abs(r.exact)])
        end
    end
    write_csv(joinpath(DATA, "null_layer_map.csv"),
              ["z_real_nd", "lambda_nd", "one_moment_error_nd",
               "two_moment_error_nd", "pole_extraction_error_nd",
               "linearized_pole_error_nd", "slope_separated_error_nd",
               "exact_abs_nd"], rows;
              comments=[
                  "amplitude a(t)=(t+Lambda/sqrt(k))*(1+b1 t+b2 t^2+b3 t^3) " *
                  "exactly factored",
                  "b1=$(BTAIL[1]) b2=$(BTAIL[2]) b3=$(BTAIL[3])",
                  "k=400.0 im_z=0.35",
                  "errors are scale_normalized_error against the full " *
                  "five-coefficient hierarchy",
                  "pole_extraction uses the full amplitude a(t)",
                  "linearized_pole uses only a0,a1 (identical to two_moment " *
                  "algebraically)",
                  "slope_separated uses a0 W0 + a1 k^-1/2 i/(2 sqrt(pi) z^2) " *
                  "(nonuniform)"])
    ix = argmin(abs.(xs .- 0.8))
    slice = Vector{Vector{Float64}}()
    for (il, lam) in enumerate(lambdas)
        push!(slice, [lam, e0m[il, ix], e1m[il, ix], eclm[il, ix],
                      esepm[il, ix]])
    end
    write_csv(joinpath(DATA, "null_layer_section.csv"),
              ["lambda_nd", "one_moment_error_nd", "two_moment_error_nd",
               "pole_extraction_error_nd", "slope_separated_error_nd"],
              slice;
              comments=["section of null_layer_map.csv at Re z=$(xs[ix])",
                        "im_z=$y k=$k"])
    return Dict(
        "k" => k, "imag_z" => y, "x_at_section" => xs[ix],
        "n_points" => nx * nl,
        "one_moment_p95" => quantile(vec(e0m), 0.95),
        "one_moment_median" => median(vec(e0m)),
        "one_moment_max" => maximum(e0m),
        "two_moment_p95" => quantile(vec(e1m), 0.95),
        "two_moment_median" => median(vec(e1m)),
        "two_moment_max" => maximum(e1m),
        "pole_extraction_p95" => quantile(vec(eclm), 0.95),
        "pole_extraction_median" => median(vec(eclm)),
        "pole_extraction_max" => maximum(eclm),
        "slope_separated_p95" => quantile(vec(esepm), 0.95),
        "slope_separated_median" => median(vec(esepm)),
        "slope_separated_max" => maximum(esepm),
        "linearized_max_dev" => maximum(abs.(elinm .- e1m)),
        "fraction_one_moment_above_0p1" => mean(vec(e0m) .> 0.1),
        "median_ratio_two_to_cl" =>
            median(vec(e1m) ./ max.(vec(eclm), 1e-300)),
    )
end

function convergence_sweep()
    z = 0.8 + 0.35im
    Lambda = 1.2
    ks = 10 .^ range(1.5, 7.0; length=30)
    rows = Vector{Vector{Float64}}()
    errors = [Float64[] for _ in 0:3]
    cl_errors = Float64[]
    for k in ks
        coeff = factored_null_coefficients(k, Lambda; b=BTAIL)
        a = polynomial_amplitude(coeff)
        exact = null_uniform_transition(k, z, coeff)
        scale = term_scale(k, z, coeff)
        row = [k]
        for order in 0:3
            approx = null_uniform_transition(k, z, coeff; order=order)
            err = scale_normalized_error(approx, exact, scale)
            push!(errors[order + 1], err)
            push!(row, err)
        end
        ecl = scale_normalized_error(
            NUM.classical_pole_term(k, z, a), exact, scale)
        push!(cl_errors, ecl)
        push!(row, ecl)
        push!(rows, row)
    end
    write_csv(joinpath(DATA, "convergence.csv"),
              ["k_nd", "error_M0_nd", "error_M1_nd", "error_M2_nd",
               "error_M3_nd", "error_pole_extraction_nd"], rows;
              comments=["z=$(real(z))+$(imag(z))im Lambda=$Lambda",
                        "exactly factored amplitude, b=$BTAIL",
                        "OLS slope fit of log10(error) on log10(k) uses " *
                        "ALL 30 samples"])
    # All-30-sample ordinary least squares on log10-log10 data.
    slopes = Dict("M$m" => fitted_log_slope(ks, errors[m + 1])
                  for m in 0:3)
    slopes["CL"] = fitted_log_slope(ks, cl_errors)
    zero_rows = Vector{Vector{Float64}}()
    zero_slopes = Dict{String,Float64}()
    zero_vals = Dict{Int,Vector{Float64}}()
    for r in 1:3
        coeff = zeros(r + 1)
        coeff[r + 1] = 1.0
        vals = [abs(null_uniform_transition(k, z, coeff)) for k in ks]
        zero_vals[r] = vals
        zero_slopes["order_$r"] = fitted_log_slope(ks, vals)
    end
    for (i, k) in enumerate(ks)
        push!(zero_rows, [k, zero_vals[1][i], zero_vals[2][i], zero_vals[3][i]])
    end
    write_csv(joinpath(DATA, "zero_order_scaling.csv"),
              ["k_nd", "abs_order1_nd", "abs_order2_nd", "abs_order3_nd"],
              zero_rows;
              comments=["monomial amplitudes a_m=delta_{m,r}, z=$z"])
    return Dict("z_real" => real(z), "z_imag" => imag(z), "Lambda" => Lambda,
                "slopes" => slopes, "zero_slopes" => zero_slopes,
                "n_samples" => length(ks))
end

function random_quadrature_validation()
    rows = Vector{Vector{Float64}}()
    full_errors = Float64[]
    standard_errors = Float64[]
    null_errors = Float64[]
    cl_errors = Float64[]
    inputs = Tuple{Float64,Float64,Float64,Float64,Float64,Float64,Float64}[]
    for idx in 0:319
        k = 10.0^(rand(RNG) * (log10(3e4) - log10(30.0)) + log10(30.0))
        z = complex(rand(RNG) * 5.6 - 2.8, rand(RNG) * 0.8 + 0.3)
        Lambda = rand(RNG) * 7.0 - 3.5
        tail = ntuple(_ -> rand(RNG) * 0.7 - 0.35, 3)
        coeff = factored_null_coefficients(k, Lambda; b=tail)
        a = polynomial_amplitude(coeff)
        push!(inputs, (k, real(z), imag(z), Lambda, tail...))
        # Im z >= 0.3 here; see REFERENCE_HERMITE_ORDER for the node count.
        exact = canonical_quadrature(k, z, a; n=REFERENCE_HERMITE_ORDER)
        full = null_uniform_transition(k, z, coeff)
        leading = null_uniform_transition(k, z, coeff; order=0)
        null = null_uniform_transition(k, z, coeff; order=1)
        cl = NUM.classical_pole_term(k, z, a)
        scale = term_scale(k, z, coeff)
        e_full = scale_normalized_error(full, exact, scale)
        e_std = scale_normalized_error(leading, exact, scale)
        e_null = scale_normalized_error(null, exact, scale)
        e_cl = scale_normalized_error(cl, exact, scale)
        push!(full_errors, e_full)
        push!(standard_errors, e_std)
        push!(null_errors, e_null)
        push!(cl_errors, e_cl)
        push!(rows, [idx, k, real(z), imag(z), Lambda, tail[1], tail[2],
                     tail[3], real(exact), imag(exact),
                     e_full, e_std, e_null, e_cl])
    end
    write_csv(joinpath(DATA, "random_quadrature.csv"),
              ["case_nd", "k_nd", "z_real_nd", "z_imag_nd", "lambda_nd",
               "b1_nd", "b2_nd", "b3_nd", "exact_re_nd", "exact_im_nd",
               "full_error_nd", "standard_error_nd", "null_error_nd",
               "pole_extraction_error_nd"], rows;
              comments=["seed=$SEED",
                  "k ~ log-uniform [30,3e4]; Re z ~ U[-2.8,2.8]; " *
                  "Im z ~ U[0.3,1.1]; Lambda ~ U[-3.5,3.5]; " *
                  "b1,b2,b3 ~ U[-0.35,0.35]",
                  "amplitude (t+Lambda/sqrt(k))*(1+b1 t+b2 t^2+b3 t^3)",
                  "Gauss-Hermite reference order n=$REFERENCE_HERMITE_ORDER"])
    return Dict(
        "cases" => length(rows),
        "full_p95" => quantile(full_errors, 0.95),
        "full_max" => maximum(full_errors),
        "standard_p95" => quantile(standard_errors, 0.95),
        "standard_max" => maximum(standard_errors),
        "null_p95" => quantile(null_errors, 0.95),
        "null_max" => maximum(null_errors),
        "cl_p95" => quantile(cl_errors, 0.95),
        "cl_max" => maximum(cl_errors),
        "median_reduction_standard_to_null" =>
            median(standard_errors ./ max.(null_errors, 1e-300)),
        "median_ratio_null_to_cl" =>
            median(null_errors ./ max.(cl_errors, 1e-300)),
        "inputs_recorded" => length(inputs),
    )
end

function direct_integral_validation()
    rows = Vector{Vector{Float64}}()
    errs = Float64[]
    for idx in 0:35
        z = complex(rand(RNG) * 5.0 - 2.5, rand(RNG) * 0.85 + 0.35)
        m = rand(RNG, 0:3)
        val = faddeeva_moments(z, m)[m + 1]
        ref = direct_moment(z, m; rtol=1e-14, atol=1e-14)
        err = abs(val - ref) / max(abs(ref), 1e-15)
        push!(errs, err)
        push!(rows, [idx, m, real(z), imag(z), err])
    end
    write_csv(joinpath(DATA, "direct_integral.csv"),
              ["case_nd", "m_nd", "z_real_nd", "z_imag_nd",
               "relative_error_nd"], rows;
              comments=["seed=$SEED", "m in {0,1,2,3}, QuadGK rtol=atol=1e-14"])
    return Dict("cases" => length(rows), "p95" => quantile(errs, 0.95),
                "max" => maximum(errs))
end

function moment_profile()
    xs = collect(range(-4.0, 4.0; length=401))
    rows = Vector{Vector{Float64}}()
    for x in xs
        W = faddeeva_moments(complex(x, 0.35), 3)
        push!(rows, [x, abs(W[1]), abs(W[2]), abs(W[3]), abs(W[4])])
    end
    write_csv(joinpath(DATA, "moment_profile.csv"),
              ["z_real_nd", "abs_W0_nd", "abs_W1_nd", "abs_W2_nd",
               "abs_W3_nd"], rows;
              comments=["Im z = 0.35"])
    return length(rows)
end

function stability_sweep()
    radii = 10 .^ range(0.3, 12.0; length=150)
    # Multiple complex sheets: three additional upper-half-plane lines
    # plus one lower-half-plane line that exercises the Stokes
    # continuation of the hybrid evaluator against the entire-function
    # high-precision oracle.
    im_parts = (0.1, 0.4, 1.0, 4.0, -0.4)
    rows = Vector{Vector{Float64}}()
    stable_err = Dict(m => Float64[] for m in 1:4)
    recurrence_err = Dict(m => Float64[] for m in 1:4)
    for r in radii, yi in im_parts
        z = complex(r, yi)
        stable = faddeeva_moments(z, 4)
        recurrence = NUM.moment_recurrence(z, 4)
        for m in 1:4
            ref = hp_moment(z, m; precision=384)
            es = abs(stable[m + 1] - ref) / max(abs(ref), 1e-300)
            er = abs(recurrence[m + 1] - ref) / max(abs(ref), 1e-300)
            push!(stable_err[m], es)
            push!(recurrence_err[m], er)
            push!(rows, [r, m, real(z), imag(z), es, er])
        end
    end
    write_csv(joinpath(DATA, "stability.csv"),
              ["radius_nd", "m_nd", "z_real_nd", "z_imag_nd",
               "stable_error_nd", "recurrence_error_nd"], rows;
              comments=["z = radius + y*im, y in $(im_parts) (the -0.4 " *
                        "sheet exercises the lower-half-plane Stokes " *
                        "continuation)",
                        "reference: hp_moment BigFloat recurrence, 384 bits"])
    first_above = Dict{String,Float64}()
    for m in 1:4
        # entries are grouped by radius: each radius contributes
        # length(im_parts) consecutive error samples
        idx = findfirst(>(1e-6), recurrence_err[m])
        first_above[string(m)] = idx === nothing ? radii[end] :
            radii[cld(idx, length(im_parts))]
    end
    return Dict(
        "stable_max" => maximum(maximum(v) for v in values(stable_err)),
        "recurrence_max" => maximum(maximum(v) for v in values(recurrence_err)),
        "recurrence_first_above_1e-6" => first_above,
        "n_radii" => length(radii),
        "n_sheets" => length(im_parts),
        "n_comparisons" => length(rows),
    )
end

# --- nonpolynomial exactly-factored benchmark ---------------------------

function nonpolynomial_benchmark()
    rows = Vector{Vector{Float64}}()
    two_errs = Float64[]
    full_errs = Float64[]
    cl_errs = Float64[]
    closed_vs_quad = Float64[]
    case = 0
    for k in (100.0, 400.0, 1600.0, 10000.0)
        for Lambda in (-2.0, -0.3, 0.3, 2.0)
            for beta in (-1.5, 0.3, 0.7, 1.5)
                case += 1
                z = complex(0.8, 0.35)
                a = NUM.nonpolynomial_amplitude(k, Lambda, beta)
                closed = NUM.nonpolynomial_closed_form(k, z, Lambda, beta)
                quad = canonical_quadrature(k, z, a;
                                            n=REFERENCE_HERMITE_ORDER)
                push!(closed_vs_quad,
                      abs(closed - quad) / max(abs(closed), 1e-300))
                # Taylor coefficients of (t + tau) e^{beta t}:
                # a_0 = tau,  a_m = beta^{m-1}(m + tau*beta)/m!  (m >= 1)
                tau = Lambda / sqrt(k)
                coeff = zeros(ComplexF64, 6)
                coeff[1] = tau
                for m in 1:5
                    coeff[m + 1] = beta^(m - 1) * (m + tau * beta) /
                                   factorial(m)
                end
                scale = term_scale(k, z, coeff)
                full = NUM.hierarchy_value(k, z, coeff)
                two = NUM.hierarchy_value(k, z, coeff; order=1)
                cl = NUM.classical_pole_term(k, z, a)
                e2 = scale_normalized_error(two, closed, scale)
                ef = scale_normalized_error(full, closed, scale)
                ecl = scale_normalized_error(cl, closed, scale)
                push!(two_errs, e2)
                push!(full_errs, ef)
                push!(cl_errs, ecl)
                push!(rows, [case, k, real(z), imag(z), Lambda, beta,
                             real(closed), imag(closed), e2, ef, ecl])
            end
        end
    end
    write_csv(joinpath(DATA, "nonpolynomial_benchmark.csv"),
              ["case_nd", "k_nd", "z_real_nd", "z_imag_nd", "lambda_nd",
               "beta_nd", "closed_re_nd", "closed_im_nd",
               "two_moment_error_nd", "full_error_nd",
               "pole_extraction_error_nd"], rows;
              comments=["a(t)=(t+Lambda/sqrt(k))*exp(beta*t), exact moving " *
                  "zero at -Lambda/sqrt(k)",
                  "closed form via completion of the square",
                  "hierarchy truncated at six Taylor coefficients " *
                  "m=0..5 (M_ref=5 in the term scale)",
                  "independent quadrature check: canonical_quadrature " *
                  "n=$REFERENCE_HERMITE_ORDER"])
    return Dict("cases" => case,
                "closed_vs_quad_max" => maximum(closed_vs_quad),
                "two_max" => maximum(two_errs),
                "two_median" => median(two_errs),
                "full_max" => maximum(full_errs),
                "cl_max" => maximum(cl_errs),
                "cl_median" => median(cl_errs))
end

# --- electromagnetic half-plane benchmark --------------------------------

function wedge_halfplane()
    Krho = 50.0
    N = 2.0
    phi_i = deg2rad(10.0)
    pol = :hard
    phis = collect(range(140.0, 240.0; length=160))
    rows = Vector{Vector{Float64}}()
    scales = Float64[]
    per = Dict{String,Vector{Float64}}(
        n => Float64[] for n in
        ("e0", "e1", "e4", "ecl", "esep"))
    sdp_ref_err = Float64[]
    ud_mags = Float64[]
    lam_abs = Float64[]
    # precompute local scales to fix the floor
    scale_of = Dict{Float64,Float64}()
    for d in phis
        m2 = NUM.wedge_canonical_model(Krho, deg2rad(d), phi_i, N, pol;
                                       saddle=-2pi)
        m0 = NUM.wedge_canonical_model(Krho, deg2rad(d), phi_i, N, pol;
                                       saddle=0.0)
        scale_of[d] = NUM.wedge_term_scale(m2, 4) +
                      NUM.wedge_term_scale(m0, 4)
    end
    smax = maximum(values(scale_of))
    floor_rel = 1e-4
    null_phi = 180.0
    lam_at_pi = complex(NaN)
    field_at_pi = complex(NaN)
    for d in phis
        phi = deg2rad(d)
        ud = NUM.wedge_diffracted(Krho, phi, phi_i, N, pol)
        usdp = NUM.wedge_diffracted_sdp(Krho, phi, phi_i, N, pol)
        push!(sdp_ref_err, abs(usdp - ud))
        scale = max(scale_of[d], floor_rel * smax)
        v0 = NUM.wedge_diffracted_model(Krho, phi, phi_i, N, pol; order=0)
        v1 = NUM.wedge_diffracted_model(Krho, phi, phi_i, N, pol; order=1)
        v4 = NUM.wedge_diffracted_model(Krho, phi, phi_i, N, pol; order=4)
        vcl = NUM.wedge_diffracted_model(Krho, phi, phi_i, N, pol;
                                       kind=:pole_extraction)
        m2 = NUM.wedge_canonical_model(Krho, phi, phi_i, N, pol;
                                       saddle=-2pi)
        # The separated-slope form stays finite on the sampled grid (its
        # smallest mapped pole is |z| ~ 0.016); it is evaluated at every
        # angle.  A genuinely singular evaluation would be recorded as
        # NaN together with the exclusion count in the header.
        esep = try
            abs(NUM.wedge_diffracted_model(Krho, phi, phi_i, N, pol;
                                           kind=:slope_separated) - ud) /
                scale
        catch error
            error isa ArgumentError && occursin("separated slope term is singular at z = 0", sprint(showerror,error)) || rethrow()
            NaN
        end
        push!(per["e0"], abs(v0 - ud) / scale)
        push!(per["e1"], abs(v1 - ud) / scale)
        push!(per["e4"], abs(v4 - ud) / scale)
        push!(per["ecl"], abs(vcl - ud) / scale)
        push!(per["esep"], esep)
        push!(scales, scale)
        push!(ud_mags, abs(ud))
        z1, z2 = m2.poles
        push!(lam_abs, abs(m2.lambda))
        push!(rows, [d, real(ud), imag(ud), scale,
                     per["e0"][end], per["e1"][end], per["e4"][end],
                     per["ecl"][end], esep,
                     real(z1), imag(z1), real(z2), imag(z2),
                     real(m2.lambda), imag(m2.lambda),
                     abs(m2.amplitude_u(-m2.lambda))])
    end
    # exact-null point: evaluate at phi = pi explicitly
    m_pi = NUM.wedge_canonical_model(Krho, pi, phi_i, N, pol; saddle=-2pi)
    lam_at_pi = m_pi.lambda
    field_at_pi = NUM.wedge_diffracted(Krho, pi, phi_i, N, pol)
    model_at_pi = NUM.wedge_diffracted_model(Krho, pi, phi_i, N, pol;
                                           order=1)
    esep_excluded = count(isnan, per["esep"])
    write_csv(joinpath(DATA, "wedge_halfplane.csv"),
              ["phi_deg_nd", "ud_re_nd", "ud_im_nd", "scale_nd",
               "one_moment_error_nd", "two_moment_error_nd",
               "four_moment_error_nd", "pole_extraction_error_nd",
               "slope_separated_error_nd", "z1_re_nd", "z1_im_nd",
               "z2_re_nd", "z2_im_nd", "lambda_re_nd", "lambda_im_nd",
               "amp_at_lambda_nd"], rows;
              comments=[
                  "PEC half-plane N=2, Neumann (hard, H_z) polarization",
                  "Krho=$Krho phi_i_deg=10.0 saddles sigma0=0,-2pi",
                  "reference: wedge_eigenfunction - wedge_go",
                  "scale floor: max(local term scale, $(floor_rel)*max scale)",
                  "two GO poles extracted per saddle",
                  "z, lambda columns refer to the sigma0=-2pi model",
                  "lambda = -u(sigma_z) for the sigma-plane kernel zero " *
                  "nearest the saddle (branch-consistent coordinate)",
                  "amp_at_lambda = |atilde(-lambda)| checks the claimed " *
                  "spectral zero of the regularized amplitude",
                  "two_moment column equals four_moment to ~1e-14: every " *
                  "m>=2 canonical term is shared by the two saddles and " *
                  "cancels in the contour difference",
                  "pole_extraction uses analytic removable limits of the " *
                  "regularized amplitude at the mapped poles",
                  "slope_separated evaluated at all $(length(phis)) angles; " *
                  "$esep_excluded excluded as singular"])
    return Dict(
        "n_angles" => length(phis),
        "sdp_eig_max" => maximum(sdp_ref_err),
        "ud_min" => minimum(ud_mags),
        "ud_max" => maximum(ud_mags),
        "lambda_at_pi" => lam_at_pi,
        "field_at_pi" => field_at_pi,
        "model_at_pi" => model_at_pi,
        "e0_max" => maximum(per["e0"]),
        "e0_median" => median(per["e0"]),
        "e1_max" => maximum(per["e1"]),
        "e4_max" => maximum(per["e4"]),
        "ecl_max" => maximum(per["ecl"]),
        "ecl_median" => median(per["ecl"]),
        "esep_max" => maximum(filter(!isnan, per["esep"])),
        "esep_median" => median(filter(!isnan, per["esep"])),
        "esep_excluded" => esep_excluded,
        "grid_floor" => floor_rel * smax,
        "amp_lambda_max" => maximum(r[16] for r in rows),
    )
end

# Boundary and exact-null diagnostics separate from the sampled sweep.
# At a transition boundary a mapped pole sits on the real canonical
# contour: the symmetric principal-value convention C_m = W_m - z^m
# e^{-z^2} is paired with half-weight geometrical-optics constituents.
# The assembled total field (GO + diffracted) is checked for continuity
# against the converged eigenfunction reference.
function wedge_diagnostics(Krho, N, phi_i, pol, floor_scale)
    entries = [(pi - phi_i, 1.0), (pi, 2.0), (pi + phi_i, 3.0)]
    pv = 1e-9
    rows = Vector{Vector{Float64}}()
    esep_na = 0
    for (phi, tag) in entries
        ue = NUM.wedge_eigenfunction(Krho, phi, phi_i, N, pol)
        go = NUM.wedge_go(Krho, phi, phi_i, N, pol; half_weight=true)
        ud = ue - go
        m2 = NUM.wedge_canonical_model(Krho, phi, phi_i, N, pol;
                                       saddle=-2pi)
        m0 = NUM.wedge_canonical_model(Krho, phi, phi_i, N, pol;
                                       saddle=0.0)
        slocal = NUM.wedge_term_scale(m2, 4; pv_tol=pv) +
                 NUM.wedge_term_scale(m0, 4; pv_tol=pv)
        scale = max(slocal, floor_scale)
        v0 = NUM.wedge_saddle_field(m2, 0; pv_tol=pv) -
             NUM.wedge_saddle_field(m0, 0; pv_tol=pv)
        v1 = NUM.wedge_saddle_field(m2, 1; pv_tol=pv) -
             NUM.wedge_saddle_field(m0, 1; pv_tol=pv)
        vcl = NUM.wedge_saddle_field(m2, 1; kind=:pole_extraction,
                                     pv_tol=pv) -
              NUM.wedge_saddle_field(m0, 1; kind=:pole_extraction,
                                     pv_tol=pv)
        esep = try
            abs(NUM.wedge_saddle_field(m2, 1;
                                       kind=:slope_separated) -
                NUM.wedge_saddle_field(m0, 1;
                                       kind=:slope_separated) - ud) /
                scale
        catch error
            error isa ArgumentError && occursin("separated slope term is singular at z = 0", sprint(showerror,error)) || rethrow()
            esep_na += 1
            NaN
        end
        total = go + v1
        z1, z2 = m2.poles
        push!(rows, [rad2deg(phi), tag, real(ud), imag(ud), scale,
                     abs(v0 - ud) / scale, abs(v1 - ud) / scale,
                     abs(vcl - ud) / scale, esep,
                     abs(total - ue), abs(m2.amplitude_u(-m2.lambda)),
                     real(z1), imag(z1), real(z2), imag(z2),
                     real(m2.lambda), imag(m2.lambda)])
    end
    write_csv(joinpath(DATA, "wedge_diagnostics.csv"),
              ["phi_deg_nd", "boundary_tag_nd", "ud_ref_re_nd",
               "ud_ref_im_nd", "scale_nd", "one_moment_error_nd",
               "two_moment_error_nd", "pole_extraction_error_nd",
               "slope_separated_error_nd", "total_field_residual_nd",
               "amp_at_lambda_nd", "z1_re_nd", "z1_im_nd", "z2_re_nd",
               "z2_im_nd", "lambda_re_nd", "lambda_im_nd"], rows;
              comments=[
                  "separate diagnostics at the exact reflection boundary " *
                  "(tag 1), diffracted-field null (tag 2), and shadow " *
                  "boundary (tag 3); none is a sampled sweep angle",
                  "reference: eigenfunction - GO with the symmetric " *
                  "half-residue GO convention",
                  "moments: symmetric principal-value convention " *
                  "C_m = W_m - z^m e^{-z^2} on the real axis",
                  "scale: max(local term scale, grid floor from the " *
                  "160-angle sweep)",
                  "total_field_residual: |GO_half + two_moment - " *
                  "eigenfunction| (absolute, continuity check)",
                  "amp_at_lambda = |atilde(-lambda)| at the recorded " *
                  "branch-consistent null coordinate",
                  "slope_separated: $esep_na singular exclusions recorded " *
                  "as NaN"])
    return Dict(
        "rows" => rows,
        "total_residual_max" =>
            maximum(r[10] for r in rows),
        "two_moment_max" => maximum(r[7] for r in rows),
        "one_moment_at_null" => rows[2][6],
        "two_moment_at_null" => rows[2][7],
        "sep_at_null" => rows[2][9],
        "refl_one" => rows[1][6],
        "refl_two" => rows[1][7],
        "refl_sep" => rows[1][9],
        "shadow_one" => rows[3][6],
        "shadow_two" => rows[3][7],
        "shadow_sep" => rows[3][9],
        "amp_lambda_max" => maximum(r[11] for r in rows),
        "esep_na" => esep_na,
    )
end

function run_manifest(summary)
    manifest = Dict("backend"=>string(BACKEND), "seed"=>SEED,
                    "sources"=>SOURCES, "reference_hermite_order"=>REFERENCE_HERMITE_ORDER,
                    "csv_count"=>11, "figure_count"=>8)
    open(joinpath(DATA,"run_manifest.toml"),"w") do io
        TOML.print(io,manifest;sorted=true)
    end
    write(joinpath(DATA,"run_manifest.txt"),
          "Canonical 168 reproduction; backend=$(BACKEND); Julia=$(VERSION); " *
          "seed=$(SEED); Hermite order=$(REFERENCE_HERMITE_ORDER).\n" *
          "Full inputs are persisted in CSV headers and rows; source identities in run_manifest.toml.\n")
end

function result_macros(summary)
    nm = summary["null_map"]
    cv = summary["convergence"]
    rv = summary["random_validation"]
    mpv = summary["integral_validation"]
    st = summary["stability"]
    np = summary["nonpoly"]
    wg = summary["wedge"]
    wd = summary["wedge_diagnostics"]
    zs = cv["zero_slopes"]
    sl = cv["slopes"]
    lines = [
        @sprintf("\\newcommand{\\StandardFailureFraction}{%.1f}",
                 100 * nm["fraction_one_moment_above_0p1"]),
        "\\newcommand{\\NullMapStandardP}{\\ensuremath{$(latex_scientific(nm["one_moment_p95"]))}}",
        "\\newcommand{\\NullMapCorrectedP}{\\ensuremath{$(latex_scientific(nm["two_moment_p95"]))}}",
        "\\newcommand{\\NullMapCorrectedMax}{\\ensuremath{$(latex_scientific(nm["two_moment_max"]))}}",
        "\\newcommand{\\NullMapCorrectedMedian}{\\ensuremath{$(latex_scientific(nm["two_moment_median"]))}}",
        "\\newcommand{\\NullMapStandardMedian}{\\ensuremath{$(latex_scientific(nm["one_moment_median"]))}}",
        "\\newcommand{\\NullMapStandardMax}{\\ensuremath{$(latex_scientific(nm["one_moment_max"]))}}",
        "\\newcommand{\\PoleExtMedian}{\\ensuremath{$(latex_scientific(nm["pole_extraction_median"]))}}",
        "\\newcommand{\\PoleExtP}{\\ensuremath{$(latex_scientific(nm["pole_extraction_p95"]))}}",
        "\\newcommand{\\PoleExtMax}{\\ensuremath{$(latex_scientific(nm["pole_extraction_max"]))}}",
        "\\newcommand{\\SlopeSepMedian}{\\ensuremath{$(latex_scientific(nm["slope_separated_median"]))}}",
        "\\newcommand{\\SlopeSepP}{\\ensuremath{$(latex_scientific(nm["slope_separated_p95"]))}}",
        "\\newcommand{\\SlopeSepMax}{\\ensuremath{$(latex_scientific(nm["slope_separated_max"]))}}",
        "\\newcommand{\\LinPoleMaxDev}{\\ensuremath{$(latex_scientific(nm["linearized_max_dev"]))}}",
        @sprintf("\\newcommand{\\TwoToClMedianRatio}{%.1f}",
                 nm["median_ratio_two_to_cl"]),
        @sprintf("\\newcommand{\\RandomCases}{%d}", rv["cases"]),
        "\\newcommand{\\RandomFullP}{\\ensuremath{$(latex_scientific(rv["full_p95"]))}}",
        "\\newcommand{\\RandomFullMax}{\\ensuremath{$(latex_scientific(rv["full_max"]))}}",
        "\\newcommand{\\RandomNullP}{\\ensuremath{$(latex_scientific(rv["null_p95"]))}}",
        "\\newcommand{\\RandomNullMax}{\\ensuremath{$(latex_scientific(rv["null_max"]))}}",
        "\\newcommand{\\RandomClP}{\\ensuremath{$(latex_scientific(rv["cl_p95"]))}}",
        "\\newcommand{\\RandomClMax}{\\ensuremath{$(latex_scientific(rv["cl_max"]))}}",
        @sprintf("\\newcommand{\\MedianReduction}{%.0f}",
                 rv["median_reduction_standard_to_null"]),
        @sprintf("\\newcommand{\\MedianRatioNullToCl}{%.1f}",
                 rv["median_ratio_null_to_cl"]),
        @sprintf("\\newcommand{\\DirectCases}{%d}", mpv["cases"]),
        "\\newcommand{\\DirectP}{\\ensuremath{$(latex_scientific(mpv["p95"]))}}",
        "\\newcommand{\\DirectMax}{\\ensuremath{$(latex_scientific(mpv["max"]))}}",
        @sprintf("\\newcommand{\\ZeroSlopeOne}{\\ensuremath{%.3f}}", zs["order_1"]),
        @sprintf("\\newcommand{\\ZeroSlopeTwo}{\\ensuremath{%.3f}}", zs["order_2"]),
        @sprintf("\\newcommand{\\ZeroSlopeThree}{\\ensuremath{%.3f}}", zs["order_3"]),
        @sprintf("\\newcommand{\\SlopeMOne}{\\ensuremath{%.3f}}", sl["M1"]),
        @sprintf("\\newcommand{\\SlopeMTwo}{\\ensuremath{%.3f}}", sl["M2"]),
        @sprintf("\\newcommand{\\SlopeMThree}{\\ensuremath{%.3f}}", sl["M3"]),
        @sprintf("\\newcommand{\\SlopeCl}{\\ensuremath{%.3f}}", sl["CL"]),
        @sprintf("\\newcommand{\\FitSamples}{%d}", cv["n_samples"]),
        "\\newcommand{\\StableMomentMax}{\\ensuremath{$(latex_scientific(st["stable_max"]))}}",
        "\\newcommand{\\RecurrenceMomentMax}{\\ensuremath{$(latex_scientific(st["recurrence_max"]))}}",
        @sprintf("\\newcommand{\\StableComparisons}{%d}", st["n_comparisons"]),
        @sprintf("\\newcommand{\\NonpolyCases}{%d}", np["cases"]),
        "\\newcommand{\\NonpolyQuadMax}{\\ensuremath{$(latex_scientific(np["closed_vs_quad_max"]))}}",
        "\\newcommand{\\NonpolyTwoMax}{\\ensuremath{$(latex_scientific(np["two_max"]))}}",
        "\\newcommand{\\NonpolyTwoMedian}{\\ensuremath{$(latex_scientific(np["two_median"]))}}",
        "\\newcommand{\\NonpolyFullMax}{\\ensuremath{$(latex_scientific(np["full_max"]))}}",
        "\\newcommand{\\NonpolyClMax}{\\ensuremath{$(latex_scientific(np["cl_max"]))}}",
        "\\newcommand{\\NonpolyClMedian}{\\ensuremath{$(latex_scientific(np["cl_median"]))}}",
        @sprintf("\\newcommand{\\WedgeAngles}{%d}", wg["n_angles"]),
        "\\newcommand{\\WedgeSdpEigMax}{\\ensuremath{$(latex_scientific(wg["sdp_eig_max"]))}}",
        "\\newcommand{\\WedgeOneMax}{\\ensuremath{$(latex_scientific(wg["e0_max"]))}}",
        "\\newcommand{\\WedgeOneMedian}{\\ensuremath{$(latex_scientific(wg["e0_median"]))}}",
        "\\newcommand{\\WedgeTwoMax}{\\ensuremath{$(latex_scientific(wg["e1_max"]))}}",
        "\\newcommand{\\WedgeFourMax}{\\ensuremath{$(latex_scientific(wg["e4_max"]))}}",
        "\\newcommand{\\WedgeClMax}{\\ensuremath{$(latex_scientific(wg["ecl_max"]))}}",
        "\\newcommand{\\WedgeClMedian}{\\ensuremath{$(latex_scientific(wg["ecl_median"]))}}",
        "\\newcommand{\\WedgeSepMax}{\\ensuremath{$(latex_scientific(wg["esep_max"]))}}",
        "\\newcommand{\\WedgeSepMedian}{\\ensuremath{$(latex_scientific(wg["esep_median"]))}}",
        "\\newcommand{\\WedgeNullField}{\\ensuremath{$(latex_scientific(abs(wg["field_at_pi"])))}}",
        "\\newcommand{\\WedgeLambdaAtNull}{\\ensuremath{$(latex_scientific(abs(wg["lambda_at_pi"])))}}",
        "\\newcommand{\\WedgeModelAtNull}{\\ensuremath{$(latex_scientific(abs(wg["model_at_pi"])))}}",
        "\\newcommand{\\WedgeDiagTotalMax}{\\ensuremath{$(latex_scientific(wd["total_residual_max"]))}}",
        "\\newcommand{\\WedgeDiagTwoMax}{\\ensuremath{$(latex_scientific(wd["two_moment_max"]))}}",
        "\\newcommand{\\WedgeDiagOneNull}{\\ensuremath{$(latex_scientific(wd["one_moment_at_null"]))}}",
        "\\newcommand{\\WedgeDiagTwoNull}{\\ensuremath{$(latex_scientific(wd["two_moment_at_null"]))}}",
        "\\newcommand{\\WedgeDiagNullSep}{\\ensuremath{$(latex_scientific(wd["sep_at_null"]))}}",
        "\\newcommand{\\WedgeDiagReflOne}{\\ensuremath{$(latex_scientific(wd["refl_one"]))}}",
        "\\newcommand{\\WedgeDiagReflTwo}{\\ensuremath{$(latex_scientific(wd["refl_two"]))}}",
        "\\newcommand{\\WedgeDiagReflSep}{\\ensuremath{$(latex_scientific(wd["refl_sep"]))}}",
        "\\newcommand{\\WedgeDiagShadowOne}{\\ensuremath{$(latex_scientific(wd["shadow_one"]))}}",
        "\\newcommand{\\WedgeDiagShadowTwo}{\\ensuremath{$(latex_scientific(wd["shadow_two"]))}}",
        "\\newcommand{\\WedgeDiagShadowSep}{\\ensuremath{$(latex_scientific(wd["shadow_sep"]))}}",
        "\\newcommand{\\WedgeAmpLambdaMax}{\\ensuremath{$(latex_scientific(wg["amp_lambda_max"]))}}",
        @sprintf("\\newcommand{\\StableSheets}{%d}", st["n_sheets"]),
        "\\newcommand{\\JuliaVersionNum}{$(VERSION)}",
        @sprintf("\\newcommand{\\ExperimentSeed}{%d}", SEED),
        @sprintf("\\newcommand{\\HermiteOrder}{%d}", REFERENCE_HERMITE_ORDER),
        @sprintf("\\newcommand{\\MomentSwitch}{%d}", Int(DEFAULT_SWITCH)),
    ]
    return join(lines, "\n") * "\n"
end

function main()
    println("null-layer map...")
    nm = null_layer_map()
    println("convergence...")
    cv = convergence_sweep()
    println("random quadrature...")
    rv = random_quadrature_validation()
    println("direct integral...")
    mpv = direct_integral_validation()
    println("moment profile...")
    moment_profile()
    println("stability (BigFloat oracle)...")
    st = stability_sweep()
    println("nonpolynomial benchmark...")
    np = nonpolynomial_benchmark()
    println("half-plane electromagnetic benchmark...")
    wg = wedge_halfplane()
    println("half-plane boundary/null diagnostics...")
    wd = wedge_diagnostics(50.0, 2.0, deg2rad(10.0), :hard,
                           wg["grid_floor"])
    summary = Dict(
        "schema_version" => "2.0",
        "seed" => SEED,
        "null_map" => nm,
        "convergence" => cv,
        "random_validation" => rv,
        "integral_validation" => mpv,
        "stability" => st,
        "nonpoly" => np,
        "wedge" => wg,
        "wedge_diagnostics" => wd,
    )
    macros = result_macros(summary)
    write(joinpath(DATA, "results_macros.tex"), macros)
    write(joinpath(PAPER, "results_macros.tex"), macros)
    run_manifest(summary)
    println("macros and manifest written")
    @printf("map: e0 p95=%.4g  e1 p95=%.4g  cl p95=%.4g  sep p95=%.4g  fail=%.4f\n",
            nm["one_moment_p95"], nm["two_moment_p95"],
            nm["pole_extraction_p95"], nm["slope_separated_p95"],
            nm["fraction_one_moment_above_0p1"])
    @printf("map medians: e0=%.4g e1=%.4g cl=%.4g sep=%.4g ratio2/cl=%.1f\n",
            nm["one_moment_median"], nm["two_moment_median"],
            nm["pole_extraction_median"], nm["slope_separated_median"],
            nm["median_ratio_two_to_cl"])
    @printf("zero slopes = %.4f %.4f %.4f ; M slopes = %.4f %.4f %.4f ; cl %.4f\n",
            cv["zero_slopes"]["order_1"], cv["zero_slopes"]["order_2"],
            cv["zero_slopes"]["order_3"],
            cv["slopes"]["M1"], cv["slopes"]["M2"], cv["slopes"]["M3"],
            cv["slopes"]["CL"])
    @printf("random: full p95=%.4g max=%.4g reduction=%.1f cl p95=%.4g\n",
            rv["full_p95"], rv["full_max"],
            rv["median_reduction_standard_to_null"], rv["cl_p95"])
    @printf("direct p95=%.4g max=%.4g\n", mpv["p95"], mpv["max"])
    @printf("stable max=%.4g recurrence max=%.4g\n",
            st["stable_max"], st["recurrence_max"])
    @printf("nonpoly: quad-vs-closed max=%.4g two max=%.4g full max=%.4g\n",
            np["closed_vs_quad_max"], np["two_max"], np["full_max"])
    @printf("wedge: sdp-eig max=%.4g e0 max=%.4g e1 max=%.4g cl max=%.4g sep max=%.4g null field=%.4g\n",
            wg["sdp_eig_max"], wg["e0_max"], wg["e1_max"], wg["ecl_max"],
            wg["esep_max"], abs(wg["field_at_pi"]))
    return summary
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
