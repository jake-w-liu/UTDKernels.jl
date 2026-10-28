include(joinpath(@__DIR__, "..", "..", "common.jl"))
using .PaperReproductionPaths
import UTDKernels, DiffUTD
using TOML
include("norm_reference.jl")
using .Paper170NormReference: norm_control_record
const BACKEND = selected_backend()
const SOURCES = source_snapshot(UTDKernels, DiffUTD)
BACKEND === :diffutd && include("backend_diffutd.jl")
const Study = BACKEND === :kernels ? UTDKernels.ShadowSensitivity : SpatialStudy
#!/usr/bin/env julia
# Regenerate all numerical evidence used by the manuscript.

using FastGaussQuadrature: gausslegendre
using Printf
using Random
using Statistics
using QuadGK: quadgk
using .Study: bandlimited_action, bump_asymmetric
using .Study: composite_phase_quadrature, curved_coordinate
using .Study: curved_inverse, fitted_log_slope, gaussian_action
using .Study: halfplane_derivatives_reference, real_gaussian_multiplier
using .Study: relative_error, translated_first_correction
using .Study: translated_gaussian_action, translated_objective_action
using .Study: sommerfeld_boundary, sommerfeld_inverse_map
using .Study: field_sensitivity_exact, misfit_sensitivity_exact
using .Study: field_sensitivity_remainder, misfit_sensitivity_remainder
using .Study: field_transition_action, misfit_transition_action
using .Study: field_truncated_action, misfit_truncated_action
using .Study: misfit_local_jets, carrier_action_exact
using .Study: carrier_action_truncated, spectral_moment
using .Study: receiver_window, sommerfeld_demodulated
using UTDKernels: shadow_sensitivity_kernel, shadow_sensitivity_multiplier
using UTDKernels: shadow_sensitivity_pullback, shadow_switch
import UTDKernels

const PAPER = output_directory(170)
const DATA = joinpath(PAPER, "data")
const SEED = 20260824
const RNG = MersenneTwister(SEED)

mkpath(DATA)

function csvnum(value::Real)
    number = Float64(value)
    return isnan(number) ? "NaN" : @sprintf("%.17g", number)
end

function write_csv(path::AbstractString, header::Vector{String}, rows)
    open(path, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join(csvnum.(row), ","))
        end
    end
    return path
end

function latex_scientific(value::Real; digits::Integer=2)
    mantissa, exponent = split(@sprintf("%.*e", digits, Float64(value)), "e")
    return string(mantissa, "\\times 10^{", parse(Int, exponent), "}")
end

function bandlimited_study()
    omega = 6.0
    quadrature_order = 640
    nodes, gauss_weights = gausslegendre(quadrature_order)
    xi = omega .* nodes
    weights = omega .* gauss_weights
    window = max.(0.0, 1 .- (xi ./ omega).^2).^4
    probe_count = 256
    probes = Matrix{ComplexF64}(undef, probe_count, quadrature_order)
    for probe_index in 1:probe_count
        cosine_coefficients = randn(RNG, 5)
        sine_coefficients = randn(RNG, 4)
        spectrum = ComplexF64.(cosine_coefficients[1] .* ones(length(xi)))
        for order in 1:4
            spectrum .+= cosine_coefficients[order + 1] .*
                        cos.(order * π .* xi ./ omega) .+
                        im * sine_coefficients[order] .*
                        sin.(order * π .* xi ./ omega)
        end
        spectrum .*= window
        normalization = sum(weights .* spectrum) / (2π)
        if abs(normalization) < 1e-6
            spectrum .+= window
            normalization = sum(weights .* spectrum) / (2π)
        end
        probes[probe_index, :] .= spectrum ./ normalization
    end

    kappas = Float64[8, 16, 32, 64, 128, 256, 512]
    error_rows = Vector{Vector{Float64}}()
    summary_rows = Vector{Vector{Float64}}()
    bound_rows = Vector{Vector{Float64}}()
    median_errors = Dict(key => Float64[]
                         for key in (:delta, :first, :second, :gaussian))
    minimum_effectivity = Inf
    for kappa in kappas
        errors = Dict(key => Float64[]
                      for key in (:delta, :first, :second, :gaussian))
        for probe_index in 1:probe_count
            spectrum = view(probes, probe_index, :)
            exact = bandlimited_action(spectrum, xi, weights, kappa)
            delta = bandlimited_action(spectrum, xi, weights, kappa; terms=1)
            first = bandlimited_action(spectrum, xi, weights, kappa; terms=2)
            second = bandlimited_action(spectrum, xi, weights, kappa; terms=3)
            gaussian = sum(conj.(spectrum) .* weights .*
                           real_gaussian_multiplier(xi, kappa)) / (2π)
            denominator = max(abs(exact), 1e-10)
            values = (
                delta=abs(delta - exact) / denominator,
                first=abs(first - exact) / denominator,
                second=abs(second - exact) / denominator,
                gaussian=abs(gaussian - exact) / denominator,
            )
            for key in keys(errors)
                push!(errors[key], values[key])
            end
            push!(error_rows, [kappa, probe_index - 1, values.delta,
                               values.first, values.second, values.gaussian])

            spectral_l1 = sum(weights .* abs.(spectrum)) / (2π)
            for order_M in 1:3
                approximation = bandlimited_action(
                    spectrum, xi, weights, kappa; terms=order_M)
                absolute_error = abs(approximation - exact)
                bound = Study.bandlimit_remainder_bound(
                    omega, kappa, order_M, spectral_l1)
                effectivity = bound / max(absolute_error, 1e-300)
                minimum_effectivity = min(minimum_effectivity, effectivity)
                push!(bound_rows, [kappa, order_M, probe_index - 1,
                                   absolute_error, bound, effectivity])
            end
        end
        medians = Dict(key => median(errors[key]) for key in keys(errors))
        p95 = Dict(key => quantile(errors[key], 0.95) for key in keys(errors))
        for key in keys(median_errors)
            push!(median_errors[key], medians[key])
        end
        push!(summary_rows,
              [kappa, medians[:delta], medians[:first], medians[:second],
               medians[:gaussian], p95[:delta], p95[:first], p95[:second],
               p95[:gaussian]])
    end
    write_csv(joinpath(DATA, "bandlimited_probe_errors.csv"),
              ["kappa_nd", "probe_nd", "error_delta_nd", "error_first_nd",
               "error_second_nd", "error_gaussian_nd"], error_rows)
    write_csv(joinpath(DATA, "bandlimited_summary.csv"),
              ["kappa_nd", "median_delta_nd", "median_first_nd",
               "median_second_nd", "median_gaussian_nd", "p95_delta_nd",
               "p95_first_nd", "p95_second_nd", "p95_gaussian_nd"],
              summary_rows)
    write_csv(joinpath(DATA, "bandlimit_bound_effectivity.csv"),
              ["kappa_nd", "order_M_nd", "probe_nd", "absolute_error_nd",
               "bound_nd", "effectivity_nd"], bound_rows)
    slopes = Dict(key => fitted_log_slope(kappas, median_errors[key])
                  for key in keys(median_errors))
    return Dict(
        "probe_count" => probe_count,
        "omega" => omega,
        "kappas" => kappas,
        "slopes" => slopes,
        "minimum_effectivity" => minimum_effectivity,
    )
end

function halfplane_study()
    radius = 1.2
    lo = sqrt(2.0) * sin(-radius / 2)
    hi = sqrt(2.0) * sin(radius / 2)
    kappas = Float64[16, 32, 64, 128, 256, 512, 1024, 2048]
    f0 = 1.0
    f1 = 0.0
    f2 = -2 / radius^2
    f3 = 0.0
    f4 = -12 / radius^4
    a = inv(sqrt(2.0))
    b = 0.0
    c = -inv(4sqrt(2.0))
    d = 0.0
    errors = Dict(key => Float64[] for key in (:leading, :first, :second))
    rows = Vector{Vector{Float64}}()
    amplitude = y -> begin
        s = 2asin(y / sqrt(2.0))
        bump_asymmetric(s; radius=radius, alpha=0.0)
    end
    for kappa in kappas
        exact = composite_phase_quadrature(
            amplitude, lo, hi, kappa; phase_increment=1.5, order=16)
        approximations = BACKEND === :diffutd ?
            Study.coordinate_pullback3d(kappa,s->bump_asymmetric(s;radius=radius,alpha=0.0);halfplane=true) :
            shadow_sensitivity_pullback(f0,f1,f2,f3,f4,a,b,c,d,kappa)
        row = [kappa, real(exact), imag(exact)]
        for (index, key) in enumerate((:leading, :first, :second))
            error = abs(approximations[index] - exact)
            push!(errors[key], error)
            push!(row, error)
        end
        push!(rows, row)
    end
    write_csv(joinpath(DATA, "halfplane_pullback.csv"),
              ["kappa_nd", "exact_real_nd", "exact_imag_nd",
               "error_leading_nd", "error_first_nd", "error_second_nd"], rows)
    fit = (length(kappas) - 3):length(kappas)
    return Dict("slopes" => Dict(key => fitted_log_slope(kappas[fit], errors[key][fit])
                                 for key in keys(errors)))
end

function curved_coordinate_study()
    gamma = 0.6
    radius = 0.8
    alpha = 0.35
    ylo = curved_coordinate(-radius, gamma)
    yhi = curved_coordinate(radius, gamma)
    kappas = Float64[64, 128, 256, 512, 1024, 2048, 4096]
    f0 = 1.0
    f1 = alpha / radius
    f2 = -2 / radius^2
    f3 = -6alpha / radius^3
    f4 = -12 / radius^4
    errors = Dict(key => Float64[]
                  for key in (:leading, :width_only, :full_first, :full_second))
    rows = Vector{Vector{Float64}}()
    amplitude = y -> bump_asymmetric(
        curved_inverse(y, gamma); radius=radius, alpha=alpha)
    for kappa in kappas
        exact = composite_phase_quadrature(
            amplitude, ylo, yhi, kappa; phase_increment=1.5, order=16)
        leading, complete_first, complete_second = BACKEND === :diffutd ?
            Study.coordinate_pullback3d(kappa,s->bump_asymmetric(s;radius=radius,alpha=alpha);gamma=gamma) :
            shadow_sensitivity_pullback(f0,f1,f2,f3,f4,1.0,gamma,0.0,0.0,kappa)
        width_only = 1 - im * f2 / (4kappa)
        approximations = (leading, width_only, complete_first, complete_second)
        row = [kappa, real(exact), imag(exact)]
        for (index, key) in enumerate(
            (:leading, :width_only, :full_first, :full_second))
            error = abs(approximations[index] - exact)
            push!(errors[key], error)
            push!(row, error)
        end
        push!(rows, row)
    end
    write_csv(joinpath(DATA, "curved_coordinate.csv"),
              ["kappa_nd", "exact_real_nd", "exact_imag_nd",
               "error_leading_nd", "error_width_only_nd",
               "error_full_first_nd", "error_full_second_nd"], rows)
    return Dict(
        "slopes" => Dict(key => fitted_log_slope(kappas[end-3:end],
                                                 errors[key][end-3:end])
                         for key in keys(errors)),
    )
end

function derivative_validation()
    kappa_values = Float64[4, 16, 64, 256]
    s_values = range(-2.35, 2.35; length=42)
    rows = Vector{Vector{Float64}}()
    first_abs = Float64[]
    first_rel = Float64[]
    second_abs = Float64[]
    second_rel = Float64[]
    for kappa in kappa_values, s in s_values
        reference_first, reference_second = halfplane_derivatives_reference(
            s, kappa; precision=256, step=1e-5)
        analytic_first,analytic_second = BACKEND === :diffutd ?
            Study.halfplane_branch_derivatives(s,kappa) :
            (Study.halfplane_d1(s,kappa),Study.halfplane_d2(s,kappa))
        error_first = abs(analytic_first - ComplexF64(reference_first))
        error_second = abs(analytic_second - ComplexF64(reference_second))
        relative_first = error_first / max(abs(reference_first), 1e-30)
        relative_second = error_second / max(abs(reference_second), 1e-30)
        push!(first_abs, error_first)
        push!(first_rel, relative_first)
        push!(second_abs, error_second)
        push!(second_rel, relative_second)
        push!(rows, [kappa, s, error_first, relative_first,
                     error_second, relative_second])
    end
    write_csv(joinpath(DATA, "derivative_validation.csv"),
              ["kappa_nd", "s_rad", "d1_abs_error_nd", "d1_rel_error_nd",
               "d2_abs_error_nd", "d2_rel_error_nd"], rows)
    return Dict(
        "comparison_count" => length(first_rel) + length(second_rel),
        "precision_bits" => 256,
        "first_max_abs" => maximum(first_abs),
        "first_max_rel" => maximum(first_rel),
        "second_max_abs" => maximum(second_abs),
        "second_max_rel" => maximum(second_rel),
    )
end

function trapezoidal_uniform(values, step::Real)
    length(values) >= 2 || throw(ArgumentError("at least two values are required"))
    return Float64(step) *
           (values[1] / 2 + sum(@view values[2:end-1]) + values[end] / 2)
end

function sampling_burden()
    sigma = 0.6
    halfwidth = 8sigma
    kappas = Float64[16, 64, 256, 1024]
    panel_counts = [64, 128, 256, 512, 1024, 2048, 4096, 8192,
                    16384, 32768, 65536]
    tolerance = 1e-6
    first_passing = Dict{Int,Int}()
    rows = Vector{Vector{Float64}}()
    for kappa in kappas
        exact = gaussian_action(kappa, sigma)
        for panels in panel_counts
            s = range(-halfwidth, halfwidth; length=panels + 1)
            values = shadow_sensitivity_kernel(s, kappa) .*
                     exp.(-s.^2 ./ (2sigma^2))
            approximation = trapezoidal_uniform(values, step(s))
            absolute_error = abs(approximation - exact)
            push!(rows, [kappa, halfwidth, panels, absolute_error])
            if absolute_error < tolerance && !haskey(first_passing, Int(kappa))
                first_passing[Int(kappa)] = panels
            end
        end
        haskey(first_passing, Int(kappa)) || error(
            "no panel count met the sampling tolerance for kappa=$kappa")
    end
    write_csv(joinpath(DATA, "sampling_burden.csv"),
              ["kappa_nd", "interval_halfwidth_nd", "N_panels_nd",
               "absolute_error_nd"], rows)
    return Dict("first_passing" => first_passing)
end

function translated_objective()
    kappa = 64.0
    sigma = 0.75
    rows = Vector{Vector{Float64}}()
    for nu in range(0.0, 20.0; length=241)
        zeta = nu^2 / (4kappa)
        exact = translated_objective_action(nu, kappa, sigma)
        delta = one(ComplexF64)
        first = translated_first_correction(nu, kappa, sigma)
        gaussian = translated_gaussian_action(nu, kappa, sigma)
        push!(rows, [nu, zeta, real(exact), imag(exact),
                     rad2deg(angle(exact)), rad2deg(angle(first)),
                     rad2deg(angle(gaussian)), relative_error(delta, exact),
                     relative_error(first, exact),
                     relative_error(gaussian, exact)])
    end
    write_csv(joinpath(DATA, "translated_objective.csv"),
              ["nu_nd", "zeta_nd", "exact_real_nd", "exact_imag_nd",
               "exact_phase_deg", "first_phase_deg", "gaussian_phase_deg",
               "error_delta_nd", "error_first_nd", "error_gaussian_nd"], rows)
    target_index = argmin(abs.([row[2] for row in rows] .- 0.5))
    return Dict(
        "phase_at_half" => rows[target_index][5],
        "gaussian_error_at_half" => rows[target_index][10],
    )
end

function profile_data()
    switch_rows = Vector{Vector{Float64}}()
    for s in range(-1.4, 1.4; length=1400)
        w8 = shadow_switch(sqrt(8.0) * s)
        w32 = shadow_switch(sqrt(32.0) * s)
        push!(switch_rows, [s, real(w8), imag(w8), real(w32), imag(w32)])
    end
    write_csv(joinpath(DATA, "canonical_switch_profiles.csv"),
              ["s_nd", "W_kappa8_real_nd", "W_kappa8_imag_nd",
               "W_kappa32_real_nd", "W_kappa32_imag_nd"], switch_rows)

    multiplier_rows = Vector{Vector{Float64}}()
    for xi in range(-8.0, 8.0; length=800)
        push!(multiplier_rows,
              [xi, angle(shadow_sensitivity_multiplier(xi, 8.0)),
               angle(shadow_sensitivity_multiplier(xi, 32.0)),
               angle(shadow_sensitivity_multiplier(xi, 128.0))])
    end
    write_csv(joinpath(DATA, "multiplier_phase.csv"),
              ["xi_nd", "phase_kappa8_rad", "phase_kappa32_rad",
               "phase_kappa128_rad"], multiplier_rows)

    sensitivity_rows = Vector{Vector{Float64}}()
    for s in range(-1.2, 1.2; length=1800)
        d16 = (BACKEND === :diffutd ? first(Study.halfplane_branch_derivatives(s,16.0)) : Study.halfplane_d1(s,16.0)) / sqrt(16.0)
        d64 = (BACKEND === :diffutd ? first(Study.halfplane_branch_derivatives(s,64.0)) : Study.halfplane_d1(s,64.0)) / sqrt(64.0)
        push!(sensitivity_rows,
              [s, real(d16), imag(d16), real(d64), imag(d64)])
    end
    write_csv(joinpath(DATA, "sensitivity_profiles.csv"),
              ["s_rad", "d1_kappa16_real_nd", "d1_kappa16_imag_nd",
               "d1_kappa64_real_nd", "d1_kappa64_imag_nd"],
              sensitivity_rows)
end

# ---------- transformed-test (F-based) remainder-bound study ----------
# The order-selection bound applies to the complete transformed test
# F = f o h (plus amplitude/Jacobian factors), not to the physical test f.
# Rows: case, scale a, exact action (re,im), leading-Dirac error, the bound
# evaluated with the physical-coordinate spectral quantity, the bound
# evaluated with the canonical-coordinate spectral quantity, and the
# resulting violation/hold ratios. Spectral moments come from the package
# routines fourier_samples/spectral_moment.

function transformed_bound_study()
    kappa = 100.0
    rows = Vector{Vector{Float64}}()

    # (a) linear map g = 0.1 s, Gaussian f = exp(-s^2/2): closed-form action.
    #     Transformed test F(y) = exp(-50 y^2) has exact Gaussian action.
    a_lin = 0.1
    sigmaF = a_lin * 1.0          # F(y) = exp(-y^2/(2 sigmaF^2)), sigmaF = a*sigma
    exact_gauss = gaussian_action(kappa, sigmaF)
    err_gauss = abs(1 - exact_gauss)
    # weighted spectral moments: for exp(-s^2/(2 sigma^2)),
    #   int xi^2 |fhat| dxi/(2 pi) = 1/sigma^2,  int |fhat|/(2 pi) = 1.
    bf_gauss = (1.0 / 1.0^2) / (4kappa)                 # sigma = 1 -> 0.0025
    bF_gauss = (1.0 / sigmaF^2) / (4kappa)              # sigmaF = 0.1 -> 0.25
    push!(rows, [1.0, a_lin, real(exact_gauss), imag(exact_gauss),
                 err_gauss, bf_gauss, bF_gauss, err_gauss / bf_gauss,
                 err_gauss / bF_gauss])

    # (b) linear map g = 0.1 s, C^infty compact spectrum exp(-1/(1-xi^2))
    #     on |xi| < 1, normalized to unit action.
    nodes, wts = gausslegendre(512)
    spectrum = [abs(x) < 1 ? exp(-1 / (1 - x^2)) : 0.0 for x in nodes]
    spectrum ./= (sum(wts .* spectrum) / (2π))
    action_order(nterms) = nterms === nothing ?
        bandlimited_action(ComplexF64.(spectrum), nodes, wts, kappa) :
        bandlimited_action(ComplexF64.(spectrum), nodes, wts, kappa;
                           terms=nterms)
    # <K_kappa, F>, F(y) = f(y/a): Fhat(xi) = a fhat(a xi), so the action is
    # bandlimited_action on the rescaled grid nu = a xi with multiplier at
    # kappa; equivalently evaluate the multiplier at xi = nu / a.
    exact_spec = sum(wts .* spectrum .*
                     exp.(im .* (nodes ./ a_lin).^2 ./ (4kappa))) / (2π)
    dirac_spec = sum(wts .* spectrum) / (2π)
    err_spec = abs(dirac_spec - exact_spec)
    # bounds: physical bandwidth 1, canonical bandwidth 1/a, spectral L1
    # invariant under rescaling: ||Fhat||_1 = ||fhat||_1 = 1 (normalized).
    bf_spec = 1^2 * 1.0 / (4kappa)
    bF_spec = (1 / a_lin)^2 * 1.0 / (4kappa)
    push!(rows, [2.0, a_lin, real(exact_spec), imag(exact_spec),
                 err_spec, bf_spec, bF_spec, err_spec / bf_spec,
                 err_spec / bF_spec])

    # (c) nonlinear scale-changing map g(s) = a (s + 0.3 s^2), probe
    #     f = B_{1.2}, kappa = 100, M = 1. Exact action by the package's
    #     phase-adaptive quadrature. Finite-band Fourier moments are numerical
    #     proxies; the independent Sobolev control below includes the whole line.
    radius = 1.2
    gamhat = 0.6
    probe_nl(s) = bump_asymmetric(s; radius=radius, alpha=0.0)
    spectral_cutoff = 6000.0
    mom_f = spectral_moment(probe_nl, -radius, radius;xi_cut=spectral_cutoff)
    norm_record = norm_control_record(radius, gamhat, kappa)
    norm_record["finite_band_physical"] = Dict("moment"=>mom_f.moment,
        "frequency_cutoff"=>spectral_cutoff,
        "error_estimate"=>mom_f.moment_error_estimate,
        "qualification"=>"finite-band grid/Fourier-refinement diagnostic; not a whole-line bound")
    norm_record["nonlinear_cases"] = Dict{String,Any}[]
    nonlinear_rows = Float64[]
    for a_nl in (1.0, 0.5, 0.2)
        g_nl(s) = a_nl * curved_coordinate(s, gamhat)
        h_nl(y) = curved_inverse(y / a_nl, gamhat)
        ylo, yhi = g_nl(-radius), g_nl(radius)
        F(y) = probe_nl(h_nl(y))
        exact_nl = composite_phase_quadrature(
            F, ylo, yhi, kappa; phase_increment=1.2, order=16)
        err_nl = abs(1 - exact_nl)
        mom_F = spectral_moment(F, ylo, yhi;xi_cut=spectral_cutoff)
        bf_nl = mom_f.moment / (4kappa)
        bF_nl = mom_F.moment / (4kappa)
        index = length(norm_record["nonlinear_cases"]) + 1
        physical_control = norm_record["physical_remainder_control"]
        transformed_control = norm_record["transformed_remainder_controls"][index]
        push!(norm_record["nonlinear_cases"], Dict("scale"=>a_nl,
            "action_error"=>err_nl,"finite_band_moment"=>mom_F.moment,
            "frequency_cutoff"=>spectral_cutoff,
            "finite_band_error_estimate"=>mom_F.moment_error_estimate,
            "physical_control_violation_ratio"=>err_nl/physical_control,
            "transformed_control_ratio"=>err_nl/transformed_control))
        err_nl <= transformed_control || error("transformed Sobolev remainder control failed")
        push!(rows, [3.0, a_nl, real(exact_nl), imag(exact_nl), err_nl,
                     bf_nl, bF_nl, err_nl / bf_nl, err_nl / bF_nl])
        push!(nonlinear_rows, a_nl, mom_F.moment,
              mom_F.moment / mom_f.moment, err_nl)
    end
    last(norm_record["nonlinear_cases"])["physical_control_violation_ratio"] > 1 ||
        error("physical-coordinate negative control no longer violates the whole-line control")
    open(joinpath(DATA, "spectral_norm_controls.toml"), "w") do stream
        TOML.print(stream, norm_record; sorted=true)
    end
    write_csv(joinpath(DATA, "transformed_bound.csv"),
              ["case_nd", "scale_a_nd", "exact_real_nd", "exact_imag_nd",
               "dirac_error_nd", "bound_physical_nd", "bound_canonical_nd",
               "violation_ratio_nd", "hold_ratio_nd"], rows)
    write_csv(joinpath(DATA, "transformed_bound_nonlinear.csv"),
              ["scale_a_nd", "moment_canonical_nd", "moment_ratio_nd",
               "dirac_error_nd"],
              collect(Iterators.partition(nonlinear_rows, 4)))
    a02 = findfirst(row -> row[2] == 0.2, rows)
    return Dict(
        "gauss_error" => err_gauss,
        "gauss_violation" => err_gauss / bf_gauss,
        "gauss_bound_F" => bF_gauss,
        "compact_error" => err_spec,
        "compact_violation" => err_spec / bf_spec,
        "nonlinear_error" => rows[end][5],
        "nonlinear_violation" => rows[end][8],
        "nonlinear_hold" => rows[end][9],
        "moment_f" => mom_f.moment,
        "moment_F_a02" => rows[a02][7] * 4kappa,
    )
end

# ---------- carrier-restoration uniformity demonstration ----------------
# The carrier factor C_kappa(s) = exp(-i kappa cos s) turns the canonical
# quadratic phase into a constant on the composed test:
#   K_kappa(y) C_kappa(h(y)) = sqrt(kappa/pi) exp(i pi/4) exp(-i kappa),
# so the exact carrier-multiplied action grows like sqrt(kappa), while the
# order-M truncation of K_kappa acting on the carrier-restored test
# F(y) = f(h(y)) exp(i kappa y^2) exp(-i kappa) yields
# exp(-i kappa) * sum_{m<M} binom(2m,m)/4^m = exp(-i kappa) {1, 3/2, 15/8}.
function carrier_uniformity_study()
    yhalf = 0.5
    rows = Vector{Vector{Float64}}()
    for kappa in (16.0, 64.0, 256.0, 1024.0)
        exact = carrier_action_exact(kappa, yhalf)
        truncs = [carrier_action_truncated(kappa, m) for m in 1:3]
        push!(rows, [kappa, abs(exact),
                     abs.(truncs)...,
                     abs(exact - truncs[1]), abs(exact - truncs[2]),
                     abs(exact - truncs[3])])
    end
    write_csv(joinpath(DATA, "carrier_uniformity.csv"),
              ["kappa_nd", "exact_abs_nd", "trunc_m1_abs_nd",
               "trunc_m2_abs_nd", "trunc_m3_abs_nd",
               "error_m1_nd", "error_m2_nd", "error_m3_nd"], rows)
    return Dict("exact_sqrt_kappa" =>
                [abs(carrier_action_exact(k, yhalf)) / sqrt(k)
                 for k in (16.0, 64.0, 256.0, 1024.0)])
end

# ---------- physical Sommerfeld benchmark -------------------------------
# Complete two-branch half-plane field V = u_c(s-p) - u_c(s+p), incidence
# angle p = 0.8, receiver window radius R = 0.8 centered on the
# minus-branch shadow boundary s* = p + pi. Two observables:
#   S1 = int w D_p ds                      (demodulated field sensitivity)
#   J2' = Re int w conj(D-1) D_p ds        (real scattered-misfit
#         sensitivity = d/dp (1/2) int w |V - u_inc|^2)
# Each is split into the transition action <K_kappa, F o h> (which the
# canonical Dirac hierarchy truncates) and the smooth remainder (direct
# quadrature). Relative errors use denominator max(|exact|, 1e-3);
# absolute errors are reported where |exact| is small.
function sommerfeld_benchmark()
    p = 0.8
    radius = 0.8
    center = sommerfeld_boundary(p)
    kappas = Float64[32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    rows = Vector{Vector{Float64}}()
    for kappa in kappas
        s1 = field_sensitivity_exact(kappa, p, center, radius)
        s1_rem = field_sensitivity_remainder(kappa, p, center, radius)
        s1_tr = field_transition_action(kappa, p, center, radius)
        s1_ga = field_transition_action(kappa, p, center, radius;
                                        kernel=:gaussian)
        s1_approx = (
            canon=s1_tr + s1_rem,
            dirac=field_truncated_action(
                kappa, p, center, radius, 1; remainder=s1_rem),
            first=field_truncated_action(
                kappa, p, center, radius, 2; remainder=s1_rem),
            second=field_truncated_action(
                kappa, p, center, radius, 3; remainder=s1_rem),
            gauss=s1_ga + s1_rem)
        j2 = misfit_sensitivity_exact(kappa, p, center, radius)
        j2_rem = misfit_sensitivity_remainder(kappa, p, center, radius)
        j2_tr = misfit_transition_action(kappa, p, center, radius)
        j2_ga = misfit_transition_action(kappa, p, center, radius;
                                         kernel=:gaussian)
        j2_approx = (
            canon=real(j2_tr) + real(j2_rem),
            dirac=misfit_truncated_action(
                kappa, p, center, radius, 1; remainder=j2_rem),
            first=misfit_truncated_action(
                kappa, p, center, radius, 2; remainder=j2_rem),
            second=misfit_truncated_action(
                kappa, p, center, radius, 3; remainder=j2_rem),
            gauss=real(j2_ga) + real(j2_rem))
        jets = misfit_local_jets(kappa, p, center, radius)
        push!(rows, [kappa,
                     real(s1), imag(s1),
                     abs(s1_approx.canon - s1), abs(s1_approx.dirac - s1),
                     abs(s1_approx.first - s1), abs(s1_approx.second - s1),
                     abs(s1_approx.gauss - s1), abs(s1_rem),
                     j2,
                     abs(j2_approx.canon - j2), abs(j2_approx.dirac - j2),
                     abs(j2_approx.first - j2), abs(j2_approx.second - j2),
                     abs(j2_approx.gauss - j2), abs(j2_rem),
                     abs(jets[3]), abs(jets[5])])
    end
    write_csv(joinpath(DATA, "sommerfeld_benchmark.csv"),
              ["kappa_nd", "s1_real_nd", "s1_imag_nd",
               "s1_err_canon_nd", "s1_err_dirac_nd", "s1_err_first_nd",
               "s1_err_second_nd", "s1_err_gauss_nd", "s1_rem_abs_nd",
               "j2_exact_nd",
               "j2_err_canon_nd", "j2_err_dirac_nd", "j2_err_first_nd",
               "j2_err_second_nd", "j2_err_gauss_nd", "j2_rem_abs_nd",
               "f2_d2_abs_nd", "f2_d4_abs_nd"], rows)

    # receiver-window offset scan at fixed kappa: the sensitivity decays as
    # the window slides off the boundary, giving near-zero gradients where
    # absolute errors are the meaningful measure.
    kappa_scan = 64.0
    scan_rows = Vector{Vector{Float64}}()
    floor_rel = 1e-3
    for offset in range(-1.3, 1.5; length=57)
        c = sommerfeld_boundary(p) + offset
        j2 = misfit_sensitivity_exact(kappa_scan, p, c, radius)
        j2_rem = misfit_sensitivity_remainder(kappa_scan, p, c, radius)
        dirac = misfit_truncated_action(
            kappa_scan, p, c, radius, 1; remainder=j2_rem)
        first = misfit_truncated_action(
            kappa_scan, p, c, radius, 2; remainder=j2_rem)
        abs_d, abs_f = abs(dirac - j2), abs(first - j2)
        denom = max(abs(j2), floor_rel)
        push!(scan_rows, [kappa_scan, p, offset, j2, abs_d, abs_f,
                          abs_d / denom, abs_f / denom])
    end
    write_csv(joinpath(DATA, "sommerfeld_offset_scan.csv"),
              ["kappa_nd", "p_nd", "window_offset_nd", "j2_exact_nd",
               "dirac_abs_err_nd", "first_abs_err_nd",
               "dirac_rel_err_nd", "first_rel_err_nd"], scan_rows)
    j2row = [row for row in rows if row[1] == 64.0][1]

    # finite-difference oracle for the frozen-window misfit objective:
    # differentiate exactly J_2(p; p0) = (1/2) int w0 |D - 1|^2 ds with the
    # receiver window held fixed at the reference incidence p0 = p.
    oracle_step = 2e-6
    misfit_objective(pp) = 0.5 * quadgk(
        s -> receiver_window(s, center, radius) *
             abs2(sommerfeld_demodulated(s, pp, kappa_scan) - 1),
        center - radius, center + radius; rtol=1e-13, atol=1e-13)[1]
    oracle_diff = abs(
        (misfit_objective(p + oracle_step) -
         misfit_objective(p - oracle_step)) / (2oracle_step) - j2row[10])

    fit = (length(kappas) - 3):length(kappas)
    s1_slopes = Dict(
        :delta => fitted_log_slope(kappas[fit], [row[5] for row in rows][fit]),
        :first => fitted_log_slope(kappas[fit], [row[6] for row in rows][fit]),
        :second => fitted_log_slope(kappas[fit], [row[7] for row in rows][fit]),
        :gauss => fitted_log_slope(kappas[fit], [row[8] for row in rows][fit]),
    )
    j2_slopes = Dict(
        :delta => fitted_log_slope(kappas[fit], [row[12] for row in rows][fit]),
        :first => fitted_log_slope(kappas[fit], [row[13] for row in rows][fit]),
        :second => fitted_log_slope(kappas[fit], [row[14] for row in rows][fit]),
        :gauss => fitted_log_slope(kappas[fit], [row[15] for row in rows][fit]),
    )
    return Dict(
        "s1_canon_max" => maximum(row[4] for row in rows),
        "j2_canon_max" => maximum(row[11] for row in rows),
        "j2_dirac_min" => minimum(row[12] for row in rows),
        "j2_second_max" => maximum(row[14] for row in rows),
        "j2_exact_k64" => j2row[10],
        "j2_gauss_max" => maximum(row[15] for row in rows),
        "scan_min_abs_j2" => minimum(abs, [r[4] for r in scan_rows]),
        "s1_slopes" => s1_slopes,
        "j2_slopes" => j2_slopes,
        "oracle_diff" => oracle_diff,
    )
end

function write_macros(summary)
    bandlimited = summary["bandlimited"]
    halfplane = summary["halfplane"]
    curved = summary["curved"]
    derivatives = summary["derivatives"]
    sampling = summary["sampling"]["first_passing"]
    translated = summary["translated"]
    transformed = summary["transformed"]
    sommerfeld = summary["sommerfeld"]
    lines = [
        "\\newcommand{\\ProbeCount}{$(bandlimited["probe_count"])}",
        "\\newcommand{\\DerivativeComparisons}{$(derivatives["comparison_count"])}",
        @sprintf("\\newcommand{\\DeltaSlope}{\\ensuremath{%.3f}}", bandlimited["slopes"][:delta]),
        @sprintf("\\newcommand{\\FirstSlope}{\\ensuremath{%.3f}}", bandlimited["slopes"][:first]),
        @sprintf("\\newcommand{\\SecondSlope}{\\ensuremath{%.3f}}", bandlimited["slopes"][:second]),
        @sprintf("\\newcommand{\\GaussianSlope}{\\ensuremath{%.3f}}", bandlimited["slopes"][:gaussian]),
        @sprintf("\\newcommand{\\HalfLeadingSlope}{\\ensuremath{%.3f}}", halfplane["slopes"][:leading]),
        @sprintf("\\newcommand{\\HalfFirstSlope}{\\ensuremath{%.3f}}", halfplane["slopes"][:first]),
        @sprintf("\\newcommand{\\HalfSecondSlope}{\\ensuremath{%.3f}}", halfplane["slopes"][:second]),
        @sprintf("\\newcommand{\\CurveLeadingSlope}{\\ensuremath{%.3f}}", curved["slopes"][:leading]),
        @sprintf("\\newcommand{\\CurveWidthSlope}{\\ensuremath{%.3f}}", curved["slopes"][:width_only]),
        @sprintf("\\newcommand{\\CurveFullSlope}{\\ensuremath{%.3f}}", curved["slopes"][:full_first]),
        @sprintf("\\newcommand{\\CurveSecondSlope}{\\ensuremath{%.3f}}", curved["slopes"][:full_second]),
        "\\newcommand{\\DOneMaxRel}{\\ensuremath{$(latex_scientific(derivatives["first_max_rel"]))}}",
        "\\newcommand{\\DTwoMaxRel}{\\ensuremath{$(latex_scientific(derivatives["second_max_rel"]))}}",
        "\\newcommand{\\SampleNOne}{$(sampling[16])}",
        "\\newcommand{\\SampleNTwo}{$(sampling[64])}",
        "\\newcommand{\\SampleNThree}{$(sampling[256])}",
        "\\newcommand{\\SampleNFour}{$(sampling[1024])}",
        @sprintf("\\newcommand{\\ObjectivePhaseHalf}{%.1f}", translated["phase_at_half"]),
        @sprintf("\\newcommand{\\ObjectiveGaussianHalf}{%.3f}", translated["gaussian_error_at_half"]),
        @sprintf("\\newcommand{\\MinimumBoundEffectivity}{%.3f}", bandlimited["minimum_effectivity"]),
        @sprintf("\\newcommand{\\CounterexErr}{%.12f}", transformed["gauss_error"]),
        @sprintf("\\newcommand{\\CounterexViol}{%.2f}", transformed["gauss_violation"]),
        @sprintf("\\newcommand{\\CounterexBoundF}{%.2f}", transformed["gauss_bound_F"]),
        @sprintf("\\newcommand{\\CompactSpecErr}{%.7f}", transformed["compact_error"]),
        @sprintf("\\newcommand{\\CompactSpecViol}{%.2f}", transformed["compact_violation"]),
        @sprintf("\\newcommand{\\NonlinMapErr}{%.5f}", transformed["nonlinear_error"]),
        @sprintf("\\newcommand{\\NonlinMapViol}{%.2f}", transformed["nonlinear_violation"]),
        "\\newcommand{\\SommerfeldCanonErr}{\\ensuremath{$(latex_scientific(max(sommerfeld["s1_canon_max"], sommerfeld["j2_canon_max"])))}}",
        "\\newcommand{\\MisfitSecondErr}{\\ensuremath{$(latex_scientific(sommerfeld["j2_second_max"]))}}",
        "\\newcommand{\\MisfitGaussK}{\\ensuremath{$(latex_scientific(sommerfeld["j2_gauss_max"]))}}",
        @sprintf("\\newcommand{\\MisfitExactVal}{%.4f}", sommerfeld["j2_exact_k64"]),
        "\\newcommand{\\MisfitOffsetMin}{\\ensuremath{$(latex_scientific(sommerfeld["scan_min_abs_j2"]; digits=1))}}",
        @sprintf("\\newcommand{\\SOneDeltaSlope}{\\ensuremath{%.2f}}", sommerfeld["s1_slopes"][:delta]),
        @sprintf("\\newcommand{\\SOneFirstSlope}{\\ensuremath{%.2f}}", sommerfeld["s1_slopes"][:first]),
        @sprintf("\\newcommand{\\SOneSecondSlope}{\\ensuremath{%.2f}}", sommerfeld["s1_slopes"][:second]),
        @sprintf("\\newcommand{\\SOneGaussSlope}{\\ensuremath{%.2f}}", sommerfeld["s1_slopes"][:gauss]),
        @sprintf("\\newcommand{\\MisfitDeltaSlope}{\\ensuremath{%.2f}}", sommerfeld["j2_slopes"][:delta]),
        @sprintf("\\newcommand{\\MisfitFirstSlope}{\\ensuremath{%.2f}}", sommerfeld["j2_slopes"][:first]),
        @sprintf("\\newcommand{\\MisfitSecondSlope}{\\ensuremath{%.2f}}", sommerfeld["j2_slopes"][:second]),
        "\\newcommand{\\MisfitOracleErr}{\\ensuremath{$(latex_scientific(sommerfeld["oracle_diff"]; digits=1))}}",
    ]
    contents = join(lines, "\n") * "\n"
    write(joinpath(DATA, "results_macros.tex"), contents)
    write(joinpath(PAPER, "results_macros.tex"), contents)
end

function main()
    println("Generating bandlimited sensitivity data...")
    bandlimited = bandlimited_study()
    println("Generating coordinate-pullback data...")
    halfplane = halfplane_study()
    curved = curved_coordinate_study()
    println("Generating independent derivative data...")
    derivatives = derivative_validation()
    println("Generating sampling and objective data...")
    sampling = sampling_burden()
    translated = translated_objective()
    profile_data()
    println("Generating transformed-test bound data...")
    transformed = transformed_bound_study()
    carrier = carrier_uniformity_study()
    println("Generating Sommerfeld physical benchmark data...")
    sommerfeld = sommerfeld_benchmark()
    summary = Dict(
        "bandlimited" => bandlimited,
        "halfplane" => halfplane,
        "curved" => curved,
        "derivatives" => derivatives,
        "sampling" => sampling,
        "translated" => translated,
        "transformed" => transformed,
        "carrier" => carrier,
        "sommerfeld" => sommerfeld,
    )
    write_macros(summary)
    open(joinpath(DATA,"run_manifest.toml"),"w") do io
        TOML.print(io,Dict("backend"=>string(BACKEND),"seed"=>SEED,"sources"=>SOURCES,
                          "csv_count"=>16,"figure_count"=>8);sorted=true)
    end
    @printf("Bandlimited slopes: %.3f, %.3f, %.3f\n",
            bandlimited["slopes"][:delta], bandlimited["slopes"][:first],
            bandlimited["slopes"][:second])
    @printf("Minimum bound effectivity: %.3f\n",
            bandlimited["minimum_effectivity"])
    @printf("Maximum derivative relative errors: %.3e, %.3e\n",
            derivatives["first_max_rel"], derivatives["second_max_rel"])
    return summary
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
