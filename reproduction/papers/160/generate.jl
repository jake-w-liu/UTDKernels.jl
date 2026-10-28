if !isdefined(@__MODULE__, :PaperReproductionPaths)
    include(joinpath(@__DIR__, "..", "..", "common.jl"))
end
using .PaperReproductionPaths: output_directory, selected_backend

# Regenerates paper/data CSVs for the grazing-continuation study.
# Scalar kernel mathematics is canonical UTDKernels; the selected DiffUTD
# route additionally constructs and projects physical 3D ray dyads.
# Coefficient references use an independent BigFloat implementation of the
# transition function: a convergent erfc series below x = 100 and an optimally
# truncated asymptotic expansion above it (no fixed-order truncation).
# Ordinary references run at >=320 bits (~96 digits before cancellation).
# The independent asymptotic branch carries a rigorous series-truncation
# bound |R_M| <= (1/2)_{M+1}/x^{M+1} < 1e-40 on the asymptotic branch; see
# hp_oracle.jl. A separate nonoscillatory Laplace-contour quadrature checks
# both production transition branches and F' without using their differential
# equation or inverse-power recurrence.
#
# evidence_min_rows: 76
# evidence_density_justification: the canonical log-h sweep uses 151 points
#   (10 intervals per decade from 1e-16 to 1e-1), while the coalescence and
#   regime sweeps use 76 points (5 intervals per decade). This resolves the
#   inverse-offset condition slope and gives the GL4 order fit multiple levels
#   above the binary64 floor; ensembles use >=400 admissible cases per regime;
#   the (h,kL) map is 33x33.

using UTDKernels
using UTDKernels: two_term_kernel, two_term_kernel_derivative, pec_wedge_prefactor,
    pec_wedge_DsDh, pec_wedge_DsDh_grazing, pec_wedge_Ds_linear, wedge_n,
    grazing_interval_report
using SpecialFunctions: erfc
using Statistics
using Random
using QuadGK: quadgk

const PAPER = output_directory(160)
const ROOT = normpath(joinpath(PAPER, ".."))
const DATA = joinpath(PAPER, "data")
const SEED = 20260816
const UNIT_ROUNDOFF = eps(Float64) / 2
const UTD_VERSION = pkgversion(UTDKernels)
mkpath(DATA)

# Shared method wrappers and the independent high-precision oracle
# (Geom, direct/four-term/linear/fixed-order evaluations, kernel_hp,
# soft_hp_terms, cond_from_terms, kernderiv_hp, transition_laplace_hp).
include(joinpath(@__DIR__, "hp_oracle.jl"))


# Regression baseline for the oracle at the exact binary64 canonical inputs
# (alpha = 1.5pi, phi = 1.7, k = 100, L = 1, h = 1e-12, all as Float64 values).
# These 45-digit values come from a separate mpmath implementation of the same
# four-term pairing (erfc-based F, exact inputs) and of its derivative; they are
# not produced by this oracle. Reverting to decimal-string inputs moves the
# coefficient reference by about 1e-16 relative and fails the 1e-30 bound.
# The check additionally pins (i) the independently evaluated mathematical
# condition numbers of the canonical geometry, (ii) the straddling case whose
# two x+ transition arguments bracket the old fixed switch at x = 60, and
# (iii) series/asymptotic overlap and continuity across the current switch.
function check_oracle_regression()
    g = Geom(1.5 * pi, 1.7, 100.0, 1.0)
    ds_ref = setprecision(BigFloat, 256) do
        complex(BigFloat("2.34257560203950476671524691221522608937681889e-14"),
                BigFloat("-2.26677539023163892857360188211704180151439579e-14"))
    end
    gp_ref = setprecision(BigFloat, 256) do
        complex(BigFloat("1.22547928746392248971153453886890245495400534"),
                BigFloat("0.0201528565977458353840865789578476471373535317"))
    end
    ds_err = rel(soft_hp(g, 1e-12; dps=70), ds_ref)
    gp_err = rel(kernderiv_hp(g; dps=60), gp_ref)
    # The working-precision policy is checked against independent pinned
    # values; 1e-30 leaves a wide margin below the
    # 1e-16 effect of a decimal input perturbation.
    ds_err < 1e-30 || error("coefficient oracle regression failed: relative error $ds_err")
    gp_err < 1e-30 || error("derivative oracle regression failed: relative error $gp_err")

    # Independent mathematical condition numbers of the canonical geometry
    # (high-precision numerator and denominator; only the ratio is rounded).
    expected_cond = Dict(
        1e-16 => 1.52987472267e16,
        1.25892541179e-16 => 1.21522268781e16,
        1e-14 => 1.52987472267e14,
        1e-12 => 1.52987472267e12,
    )
    for (h, kappa) in sort(collect(expected_cond))
        got = cond_number(g, h; dps=60)
        abs(got / kappa - 1) < 1e-9 ||
            error("condition-number oracle regression failed at h = $h: got $got, expected $kappa")
    end

    # Straddling case: with alpha = 4.71238898038469, phi = 1.7,
    # k = 53.15169653854231, L = 1, h = 1e-16 (exact binary64 inputs), the two
    # x+ transition arguments are 60 - 3.8e-15 and 60 + 6.7e-15. A reference
    # whose method switches near x = 60 produces a coefficient-reference
    # relative error of about 3.5e-11 here; the precision-controlled oracle
    # must reproduce the independently evaluated kernel difference.
    g_str = Geom(4.71238898038469, 1.7, 53.15169653854231, 1.0)
    diff_ref = setprecision(BigFloat, 256) do
        complex(BigFloat("-2.4480867927249051e-16"),
                BigFloat("-7.5617487833508214e-18"))
    end
    _, gm, gp = soft_hp_terms(g_str, 1e-16; dps=70)
    str_err = rel(gm - gp, diff_ref)
    str_err < 1e-12 || error(
        "straddling oracle regression failed: kernel-difference relative error $str_err",
    )

    # Series/asymptotic overlap and continuity across the current switch at
    # x = _F_ASYM_XMIN. At the 384-bit working precision used for this check
    # (~115 digits), about 72 digits remain at x=100 and 52 at x=144.
    # The branches can therefore be compared throughout [100, 140]; the
    # difference must remain within the certified asymptotic-truncation bound.
    # Continuity is checked against the independent Laplace-contour oracle
    # just below and just above the switch.
    str_err, overlap_err, cont_err = setprecision(BigFloat, 384) do
        ov = 0.0
        for xb in range(BigFloat(100), BigFloat(140); length=9)
            z = cis(BigFloat(pi) / 4) * sqrt(xb)
            f_series = sqrt(BigFloat(pi) * xb) * cis(BigFloat(pi) / 4) *
                       exp(z * z) * _erfc_series_hp(z)
            f_asym = _F_asymptotic_hp(xb)
            ov = max(ov, Float64(abs(f_series - f_asym)))
        end
        ct = 0.0
        for xb in (BigFloat(100) - BigFloat("1e-12"),
                   BigFloat(100) + BigFloat("1e-12"))
            Fref, _ = transition_laplace_hp(xb)
            ct = max(ct, rel(_F_hp_big(xb), Fref))
        end
        return str_err, ov, ct
    end
    overlap_err < 1e-42 ||
        error("oracle overlap failed: |series - asym| reaches $overlap_err")
    cont_err < 1e-42 ||
        error("oracle continuity failed: relative error reaches $cont_err")
    return ds_err, gp_err, str_err
end

# --------------------------------------------------------------------- io helpers
function write_csv(path, header, rows; extra_comments=String[], allow_inf=String[],
                   allow_nan=String[],
                   method="UTDKernels.jl public G/G' kernels + fixed-order GL; pec_wedge_DsDh baseline",
                   reference="independent BigFloat four-term pairing")
    open(path, "w") do io
        println(io, "# generator=reproduction/papers/160/generate.jl")
        println(io, "# method=", method)
        println(io, "# reference=", reference)
        println(io, "# utdkernels_version=", UTD_VERSION)
        println(io, "# backend=", selected_backend())
        isempty(allow_inf) || println(io, "# allow_inf=", join(allow_inf, ","))
        isempty(allow_nan) || println(io, "# allow_nan=", join(allow_nan, ","))
        for c in extra_comments
            println(io, "# ", c)
        end
        println(io, join(header, ","))
        for row in rows
            println(io, join(row, ","))
        end
    end
end

function run_transition_oracle()
    # Include stress points identified by a dense crossover scan, not only the
    # nominal branch endpoint, so the reported maximum covers local DE noise.
    # The upper samples extend through the largest transition arguments used
    # by the evidence sweeps: kL reaches 1e8 and a_sigma <= 2, so x reaches
    # about 2e8.
    xs = (1e-4, 1e-2, 0.3, 1.0, 10.0, 33.3, 34.25, 34.5, 35.0,
          60.0, 80.0, 100.0, 1e3, 1e6, 1e7, 1e8, 2e8)
    rows = Vector{Vector{Float64}}()
    for x in xs
        Fref, Fpref, Fbound, Fpbound = transition_laplace_hp(x)
        Ferr = Float64(abs(Complex{BigFloat}(F_utd(x)) - Fref) / abs(Fref))
        Fperr = Float64(abs(Complex{BigFloat}(F_utd_prime(x)) - Fpref) / abs(Fpref))
        Frelbound = Float64(Fbound / abs(Fref))
        Fprelbound = Float64(Fpbound / abs(Fpref))
        push!(rows, [x, Float64(abs(Fref)), Ferr, Frelbound,
                     Float64(abs(Fpref)), Fperr, Fprelbound])
    end
    maximum(r[3] for r in rows) < 5e-14 || error("F failed the independent Laplace-contour oracle")
    maximum(r[6] for r in rows) < 2e-11 || error("F' failed the independent Laplace-contour oracle")
    maximum(r[4] for r in rows) < 1e-30 || error("F oracle quadrature bound is too large")
    maximum(r[7] for r in rows) < 1e-28 || error("F' oracle quadrature bound is too large")
    write_csv(joinpath(DATA, "transition_oracle.csv"),
        ["x_dimensionless", "F_reference_abs", "F_relative_error", "F_oracle_relative_bound",
         "Fprime_reference_abs", "Fprime_relative_error", "Fprime_oracle_relative_bound"], rows;
        method="UTDKernels.jl F_utd / F_utd_prime",
        reference="independent 256-bit nonoscillatory Laplace-contour quadrature",
        extra_comments=["x_range=1e-4..2e8", "quadrature_rtol=1e-40"])
    return rows, Dict(
        "cases" => length(rows),
        "F_maximum" => maximum(r[3] for r in rows),
        "Fprime_maximum" => maximum(r[6] for r in rows),
        "F_bound_maximum" => maximum(r[4] for r in rows),
        "Fprime_bound_maximum" => maximum(r[7] for r in rows),
    )
end

function log_slope(x, y; lo, hi, floor=0.0)
    mask = (x .>= lo) .& (x .<= hi) .& isfinite.(y) .& (y .> floor)
    xs = log.(x[mask]); ys = log.(y[mask])
    return ([xs ones(length(xs))] \ ys)[1]
end

function sci(x)
    ax = abs(x)
    ax == 0 && return "0"
    expn = floor(Int, log10(ax))
    mant = x / 10.0^expn
    return string(round(mant; sigdigits=3), "\\times 10^{", expn, "}")
end

function require_range(name::AbstractString, value::Real, lower::Real, upper::Real)
    isfinite(value) && lower <= value <= upper || error(
        "$name is $value; expected a value in [$lower, $upper]. " *
        "The generated evidence does not support the manuscript claim.",
    )
    return value
end

function require_at_most(name::AbstractString, value::Real, upper::Real)
    isfinite(value) && value <= upper || error(
        "$name is $value; expected at most $upper. " *
        "The generated evidence does not support the manuscript claim.",
    )
    return value
end

# --------------------------------------------------------------- sampling helper
# sample_geometry (acceptance margins, rejection reasons) lives in
# hp_oracle.jl so the cost study draws from the same documented population.

# ------------------------------------------------------------------- experiments
function run_canonical(g::Geom, name::AbstractString)
    hs = exp10.(range(-16, -1; length=151))
    rows = Vector{Vector{Float64}}()
    for h in hs
        ref, gm, gp = soft_hp_terms(g, h; dps=70)
        kappa = cond_from_terms(gm, gp)
        direct = direct_soft(g, h)
        stable = fixed_order_soft(g, h; order=8)
        push!(rows, [
            h, Float64(abs(ref)),
            rel(direct, ref),
            rel(fourterm_soft(g, h), ref),
            rel(linear_soft(g, h), ref),
            rel(fixed_order_soft(g, h; order=2), ref),
            rel(fixed_order_soft(g, h; order=4), ref),
            rel(stable, ref),
            kappa,
            UNIT_ROUNDOFF * kappa,
            # Complex coefficient storage (re,im pairs) enables independent
            # phase checks without re-running the oracle.
            Float64(real(ref)), Float64(imag(ref)),
            Float64(real(direct)), Float64(imag(direct)),
            Float64(real(stable)), Float64(imag(stable)),
        ])
    end
    h = [r[1] for r in rows]
    i12 = findmin(abs.(h .- 1e-12))[2]
    stats = Dict(
        "condition_slope" => log_slope(h, [r[9] for r in rows]; lo=1e-12, hi=1e-4),
        "linear_slope" => log_slope(h, [r[5] for r in rows]; lo=2e-4, hi=2e-2, floor=1e-15),
        "gl2_slope" => log_slope(h, [r[6] for r in rows]; lo=2e-3, hi=8e-2, floor=3e-14),
        # The 1e-14 cutoff retains five asymptotic levels while excluding the
        # lower-h arithmetic-floor plateau from the eighth-order fit.
        "gl4_slope" => log_slope(h, [r[7] for r in rows]; lo=2e-2, hi=1e-1, floor=1e-14),
        "direct_at_1e_12" => rows[i12][3], "utd_at_1e_12" => rows[i12][4], "gl8_at_1e_12" => rows[i12][8],
        "direct_at_1e_16" => rows[1][3], "utd_at_1e_16" => rows[1][4], "gl8_at_1e_16" => rows[1][8],
        # Absolute-error quantities: reference magnitudes and implied direct
        # absolute errors (relative error times reference magnitude).
        "ref_abs_1e_12" => rows[i12][2], "ref_abs_1e_16" => rows[1][2],
        "direct_abs_1e_12" => rows[i12][3] * rows[i12][2],
        "direct_abs_1e_16" => rows[1][3] * rows[1][2],
    )
    require_range("condition-number slope", stats["condition_slope"], -1.05, -0.95)
    require_range("linear relative-error slope", stats["linear_slope"], 1.8, 2.2)
    require_range("GL2 relative-error slope", stats["gl2_slope"], 3.7, 4.3)
    require_range("GL4 relative-error slope", stats["gl4_slope"], 7.4, 8.6)
    write_csv(joinpath(DATA, "$(name).csv"),
        ["h_rad","reference_abs","direct_rel_error","utdkernels_rel_error",
         "linear_rel_error","gl2_rel_error","gl4_rel_error","gl8_rel_error",
         "condition_number","unit_roundoff_bound",
         "reference_re","reference_im","direct_re","direct_im",
         "stable_re","stable_im"], rows;
        extra_comments=["geometry=alpha=$(g.alpha),phi=$(g.phi),k=$(g.k),L=$(g.L)",
                        "condition_number=high-precision kernel ratio (R1)"],
        allow_inf=["condition_number", "unit_roundoff_bound"])
    return rows, stats
end

function run_coalescence(g::Geom)
    hs = exp10.(range(-16, -1; length=76))
    rows = Vector{Vector{Float64}}()
    for h in hs
        gm = kern(g, g.phi - h); gp = kern(g, g.phi + h)
        push!(rows, [h, abs(gm), abs(gp), abs(gm - gp)])
    end
    write_csv(joinpath(DATA, "coalescence.csv"),
        ["h_rad","abs_G_minus","abs_G_plus","abs_difference"], rows)
end

function run_ensemble(n_cases::Int=800; far_field::Bool=false)
    rng = Xoshiro(far_field ? SEED + 7 : SEED)
    rows = Vector{Vector{Float64}}()
    attempts = 0
    domain_rejections = Dict{String,Int}()
    derivative_rejections = 0
    max_condition = -Inf
    max_condition_geometry = nothing
    while length(rows) < n_cases
        attempts += 1
        attempts > n_cases * 200 && error("failed to obtain enough admissible cases")
        s, status, reason = sample_geometry(rng; far_field)
        if status === :domain
            domain_rejections[reason] = get(domain_rejections, reason, 0) + 1
            continue
        elseif status === :small_derivative
            derivative_rejections += 1
            continue
        end
        g, h, tmargin, xmargin = s
        ref, gm, gp = soft_hp_terms(g, h; dps=60)
        kappa = cond_from_terms(gm, gp)
        if kappa > max_condition
            max_condition = kappa
            max_condition_geometry = (g, h)
        end
        direct = direct_soft(g, h)
        stable = fixed_order_soft(g, h; order=8, transition_margin=tmargin, x_margin=xmargin)
        push!(rows, [
            Float64(length(rows)), g.alpha, g.phi, electrical_distance(g), h,
            kappa,
            rel(direct, ref),
            rel(fourterm_soft(g, h), ref),
            rel(stable, ref),
            rel(linear_soft(g, h), ref),
            Float64(real(ref)), Float64(imag(ref)),
            Float64(real(direct)), Float64(imag(direct)),
            Float64(real(stable)), Float64(imag(stable)),
        ])
    end
    name = far_field ? "ensemble_farfield" : "ensemble"
    direct = [r[7] for r in rows]; utd = [r[8] for r in rows]
    stable = [r[9] for r in rows]; linear = [r[10] for r in rows]
    x = log10.([r[6] for r in rows]); y = log10.(max.(direct, 1e-30))
    mask = (x .> 2) .& (y .> -14) .& (y .< -0.1); nmask = count(mask)
    slope = nmask >= 10 ? ([x[mask] ones(nmask)] \ y[mask])[1] : NaN
    stats = Dict(
        "accepted_cases" => length(rows), "attempts" => attempts,
        "proposed_cases" => attempts,
        "rejected_domain" => sum(values(domain_rejections); init=0),
        "rejected_domain_reasons" => domain_rejections,
        "rejected_small_derivative" => derivative_rejections,
        "n_interior" => count(r -> r[2] < pi, rows), "n_exterior" => count(r -> r[2] > pi, rows),
        "direct_median" => median(direct), "direct_p95" => quantile(direct, 0.95), "direct_max" => maximum(direct),
        "utd_p95" => quantile(utd, 0.95),
        "stable_median" => median(stable), "stable_p95" => quantile(stable, 0.95),
        "stable_p99" => quantile(stable, 0.99), "stable_max" => maximum(stable),
        "linear_p95" => quantile(linear, 0.95),
        "slope" => slope,
        "max_condition" => max_condition,
        "max_condition_geometry" => max_condition_geometry,
    )
    stats["accepted_cases"] == n_cases || error(
        "$name contains $(stats["accepted_cases"]) cases; expected $n_cases.",
    )
    require_at_most("$name stable 95th-percentile error", stats["stable_p95"], 1e-12)
    require_at_most("$name stable maximum error", stats["stable_max"], 1e-11)
    require_range("$name direct 95th-percentile error", stats["direct_p95"], 1e-2, Inf)
    far_field || require_range(
        "$name four-term 95th-percentile error", stats["utd_p95"], 1e-2, Inf,
    )
    rejection_comments = [
        "proposed=$(stats["proposed_cases"]), accepted=$(stats["accepted_cases"]), " *
        "rejected_domain=$(stats["rejected_domain"]), " *
        "rejected_small_derivative=$(stats["rejected_small_derivative"])",
        "domain_rejection_reasons=" *
        join(("$(r)=>$(c)" for (r, c) in sort(collect(domain_rejections))), "; "),
        far_field ? "acceptance: transition_margin=0.01 (pole margin), x_margin=1e-8" :
                    "acceptance: transition_margin=0.02, x_margin=0.001",
        "acceptance: |G'(phi)| >= 1e-4 max(1, |G(phi)|); n in [0.25,1.92]; " *
        "phi/alpha in [0.12,0.88]; " *
        (far_field ? "log10(k) in [0,6]" : "log10(kL) in [-2,6]") *
        "; log10(h) in [-16,-3]",
    ]
    write_csv(joinpath(DATA, "$(name).csv"),
        ["case","alpha_rad","phi_rad","kL","h_rad","condition_number",
         "direct_rel_error","utdkernels_rel_error","stable_rel_error","linear_rel_error",
         "reference_re","reference_im","direct_re","direct_im",
         "stable_re","stable_im"], rows;
        extra_comments=[far_field ? "regime=far_field(L=Inf)" : "regime=finite_L",
                        "alpha_range=0.25pi..1.92pi (interior+exterior)",
                        rejection_comments...],
        allow_inf=far_field ? ["kL", "condition_number"] : ["condition_number"])
    return rows, stats
end

function run_derivative(count::Int=400)
    rng = Xoshiro(SEED + 1)
    rows = Vector{Vector{Float64}}()
    attempts = 0
    domain_rejections = Dict{String,Int}()
    derivative_rejections = 0
    while length(rows) < count
        attempts += 1
        s, status, reason = sample_geometry(rng)
        if status === :domain
            domain_rejections[reason] = get(domain_rejections, reason, 0) + 1
            continue
        elseif status === :small_derivative
            derivative_rejections += 1
            continue
        end
        g, _ = s
        ref = kernderiv_hp(g; dps=60)
        got = kernderiv(g, g.phi)
        push!(rows, [
            Float64(length(rows)), g.alpha, g.phi, electrical_distance(g),
            rel(got, ref),
            Float64(abs(got - ref)),
            Float64(abs(ref)),
        ])
    end
    e = [r[5] for r in rows]
    ea = [r[6] for r in rows]
    stats = Dict(
        "median" => median(e), "p95" => quantile(e, 0.95),
        "maximum" => maximum(e), "cases" => length(rows),
        "abs_median" => median(ea), "abs_p95" => quantile(ea, 0.95),
        "abs_max" => maximum(ea),
        "proposed_cases" => attempts,
        "rejected_domain" => sum(values(domain_rejections); init=0),
        "rejected_domain_reasons" => domain_rejections,
        "rejected_small_derivative" => derivative_rejections,
    )
    stats["cases"] == count || error(
        "derivative ensemble contains $(stats["cases"]) cases; expected $count.",
    )
    require_at_most("derivative maximum relative error", stats["maximum"], 1e-10)
    write_csv(joinpath(DATA, "derivative_verification.csv"),
        ["case","alpha_rad","phi_rad","kL","relative_error","absolute_error",
         "reference_abs"], rows;
        extra_comments=["alpha_range=0.25pi..1.92pi (interior+exterior)",
                        "proposed=$(stats["proposed_cases"]), accepted=$(stats["cases"]), " *
                        "rejected_domain=$(stats["rejected_domain"]), " *
                        "rejected_small_derivative=$(stats["rejected_small_derivative"])"])
    return rows, stats
end

function run_electrical(g0::Geom)
    h = 1e-12
    kLs = exp10.(range(-2, 8; length=101))
    rows = Vector{Vector{Float64}}()
    for kL in kLs
        g = Geom(g0.alpha, g0.phi, kL, 1.0)
        ref, gm, gp = soft_hp_terms(g, h; dps=70)
        push!(rows, [kL, rel(direct_soft(g, h), ref), rel(fixed_order_soft(g, h; order=8, transition_margin=1e-3, x_margin=1e-12), ref), cond_from_terms(gm, gp)])
    end
    stats = Dict(
        "stable_max" => maximum(r[3] for r in rows),
        "direct_median" => median([r[2] for r in rows]),
        "direct_max" => maximum(r[2] for r in rows),
    )
    require_at_most("electrical-sweep GL8 maximum error", stats["stable_max"], 1e-10)
    write_csv(joinpath(DATA, "electrical_sweep.csv"),
        ["kL","direct_rel_error","stable_rel_error","condition_number"], rows)
    return rows, stats
end

function run_error_map(g0::Geom)
    hs = exp10.(range(-16, -4; length=33))
    kLs = exp10.(range(-2, 8; length=33))
    rows = Vector{Vector{Float64}}()
    for kL in kLs, h in hs
        g = Geom(g0.alpha, g0.phi, kL, 1.0)
        ref = soft_hp(g, h; dps=60)
        push!(rows, [kL, h, rel(direct_soft(g, h), ref), rel(fixed_order_soft(g, h; order=8, transition_margin=1e-3, x_margin=1e-12), ref)])
    end
    d = [r[3] for r in rows]; q = [r[4] for r in rows]
    stats = Dict("cases" => length(rows), "direct_max" => maximum(d),
        "frac_1e_6" => mean(d .> 1e-6), "frac_1e_2" => mean(d .> 1e-2),
        "stable_max" => maximum(q), "stable_p99" => quantile(q, 0.99))
    stats["cases"] == 1089 || error(
        "error map contains $(stats["cases"]) cases; expected 1089.",
    )
    require_at_most("error-map GL8 maximum error", stats["stable_max"], 1e-8)
    write_csv(joinpath(DATA, "error_map.csv"), ["kL","h_rad","direct_rel_error","stable_rel_error"], rows)
    return rows, stats
end

# §6.2: production scaled-erfc transition versus the same formula evaluated
# through the unscaled binary64 erfc product, measured against the independent
# Laplace-contour oracle. The unscaled product's relative error grows with x.
# Both formulas are checked at the same positive-real arguments.
function run_scaled_transition()
    xs = (1e2, 1e3, 1e4, 1e6, 1e8, 1e10, 1e12)
    rows = Vector{Vector{Float64}}()
    for x in xs
        Fref, _ = transition_laplace_hp(x)
        z = cis(pi / 4) * sqrt(x)
        # Unscaled binary64 product sqrt(pi x) e^{i pi/4} e^{z^2} erfc(z); the
        # erfc factor is SpecialFunctions.erfc, a different code path from the
        # package's scaled erfcx.
        unscaled = sqrt(pi * x) * cis(pi / 4) * exp(z * z) * erfc(complex(z))
        scaled = F_utd(x)
        push!(rows, [x,
            Float64(abs(Complex{BigFloat}(unscaled) - Fref) / abs(Fref)),
            Float64(abs(Complex{BigFloat}(scaled) - Fref) / abs(Fref))])
    end
    write_csv(joinpath(DATA, "transition_scaled.csv"),
        ["x_dimensionless","unscaled_rel_error","scaled_rel_error"], rows;
        method="binary64 unscaled erfc product vs UTDKernels.jl F_utd",
        reference="independent 256-bit nonoscillatory Laplace-contour quadrature")
    return rows, Dict(
        "unscaled_at_1e2" => rows[1][2], "unscaled_at_1e12" => rows[end][2],
        "scaled_max" => maximum(r[3] for r in rows),
        "unscaled_max" => maximum(r[2] for r in rows),
    )
end

# R4 supporting experiment: per-sigma summands of G' evaluated in high
# precision, each correctly rounded to binary64 and summed, versus the same
# terms summed at reference precision. Reports the summation condition and the
# resulting relative error as beta = pi + delta approaches the sigma=-1
# nearest-integer boundary (one-sided approach; no interval crossing).
function run_derivative_condition()
    prec = _prec(70)
    rows = Vector{Vector{Float64}}()
    g = Geom(1.5 * pi, NaN, 100.0, 1.0)   # beta is supplied directly, not via phi
    for delta in (1e-6, 1e-8, 1e-10, 1e-12)
        terms_hp, sum_hp = setprecision(BigFloat, prec) do
            t = _kernderiv_terms_hp(BigFloat(pi) + BigFloat(delta), g, prec)
            t, sum(t)
        end
        terms64 = complex.(Float64.(real.(terms_hp)), Float64.(imag.(terms_hp)))
        sum64 = sum(terms64)
        cond = Float64(sum(abs.(terms_hp)) / abs(sum_hp))
        push!(rows, [delta, cond, rel(sum64, sum_hp)])
    end
    require_range("derivative summation condition at delta=1e-6", rows[1][2], 1e4, 1e6)
    write_csv(joinpath(DATA, "derivative_condition.csv"),
        ["delta_rad","summation_condition","rounded_sum_rel_error"], rows;
        method="high-precision per-term G' summands + binary64 rounding",
        reference="same terms summed at 70-digit precision",
        extra_comments=["n=1.5,kL=100,beta=pi+delta (one-sided approach)",
                        "formula-level demonstration; outside the acceptance margins"])
    return rows, Dict("max_rel_error" => maximum(r[3] for r in rows))
end

# Derivative-floor fixture: replay the finite-L ensemble's draw sequence
# (identical Xoshiro(SEED) stream) and locate an accepted case whose eight GL
# node arguments all round to the same binary64 value. The GL8 sum then
# reduces to the linear rule, so quadrature truncation is zero by
# construction and the residual coefficient error is produced by the single
# binary64 G' evaluation; a correctly rounded independent G' at that
# argument measures how much of the residual is irreducible rounding.
function run_derivative_floor()
    rng = Xoshiro(SEED)
    nodes8, _ = UTDKernels._cached_gauss_legendre_rule(8)
    rows = Vector{Vector{Float64}}()
    nacc = 0
    while nacc < 800
        s, status, _ = sample_geometry(rng)
        status === :accepted || continue
        g, h, tmargin, xmargin = s
        nacc += 1
        args = [g.phi + h * xi for xi in nodes8]
        all(==(args[1]), args) || continue
        ref = soft_hp(g, h; dps=60)
        e8 = rel(fixed_order_soft(g, h; order=8,
            transition_margin=tmargin, x_margin=xmargin), ref)
        e_lin = rel(linear_soft(g, h), ref)
        # Correctly rounded G' at the single surviving argument (GL weights
        # sum to 2): the floor a perfect derivative evaluation would leave.
        gcr = ComplexF64(kernderiv_hp(g; dps=60))
        e_cr = rel(-prefactor(g) * h * 2 * gcr, ref)
        push!(rows, Float64[
            nacc - 1, g.alpha, g.phi, electrical_distance(g), h,
            e8, e_lin, e_cr,
        ])
    end
    isempty(rows) &&
        error("no collapsed-argument case among the first 800 accepted draws")
    # Report the case whose collapsed-rule residual is furthest above the
    # correct-rounding floor: there the derivative-evaluation error, not
    # quadrature truncation, most clearly sets the limiting accuracy.
    best = argmax([r[6] / r[8] for r in rows])
    br = rows[best]
    require_range("derivative-floor GL8 error", br[6], 1e-16, 1e-10)
    require_at_most("derivative-floor correctly-rounded error", br[8], br[6])
    write_csv(joinpath(DATA, "derivative_floor.csv"),
        ["case", "alpha_rad", "phi_rad", "kL", "h_rad",
         "gl8_rel_error", "linear_rel_error",
         "correctly_rounded_derivative_rel_error"],
        rows;
        extra_comments=[
            "accepted cases whose GL8 node arguments all round to the same " *
            "binary64 phi; GL8 reduces to the linear rule",
            "quadrature truncation is zero; residual error is the binary64 " *
            "G' evaluation error",
            "reported case = maximum gl8/correctly-rounded ratio",
        ])
    return Dict(
        "case" => Int(br[1]), "collapsed_cases" => length(rows),
        "gl8_error" => br[6], "linear_error" => br[7],
        "corrected_error" => br[8],
    )
end

# Raise requested decimal accuracy from 60 to 120 digits (320 to 431 working
# bits). Retain the original cases, both current x = 100 crossovers, and the
# exact geometries attaining the two ensemble maximum condition numbers.
# Keeping the geometry preserves far-field k, absent from the kL = Inf CSV.
function run_precision_convergence(ensemble_stats, farfield_stats)
    cases = [
        ("canonical_h1e-16", CANON, 1e-16),
        ("canonical_h1e-12", CANON, 1e-12),
        ("canonical_h1e-8", CANON, 1e-8),
        ("straddling_x60", Geom(4.71238898038469, 1.7, 53.15169653854231, 1.0), 1e-16),
        ("asymptotic_only_k200", Geom(1.5 * pi, 1.7, 200.0, 1.0), 1e-12),
        # Represented k values straddling the current + and - crossovers.
        ("current_x100_plus", Geom(1.5 * pi, 1.7, 88.58616089757052, 1.0), 1e-16),
        ("current_x100_minus", Geom(1.5 * pi, 1.7, 114.79006830030102, 1.0), 1e-16),
    ]
    for (label, stats) in (("finite_ensemble_max", ensemble_stats),
                           ("farfield_ensemble_max", farfield_stats))
        g, h = stats["max_condition_geometry"]
        push!(cases, (label, g, h))
    end
    rows = Vector{Vector{Float64}}()
    for (name, g, h) in cases
        ref60, gm, gp = soft_hp_terms(g, h; dps=60)
        ref120 = soft_hp(g, h; dps=120)
        push!(rows, [
            g.alpha, g.phi, electrical_distance(g), g.k, g.L,
            h, cond_from_terms(gm, gp),
            Float64(real(ref60)), Float64(imag(ref60)),
            Float64(real(ref120)), Float64(imag(ref120)),
            rel(ref60, ref120),
        ])
        println("  precision convergence $name: rel diff = $(rows[end][end])")
    end
    maxdiff = maximum(last(r) for r in rows)
    # Retain the 20-digit convergence requirement as coverage expands.
    require_at_most(
        "reference precision convergence (dps=60 vs dps=120)",
        maxdiff, 1e-20,
    )
    labels = [c[1] for c in cases]
    open(joinpath(DATA, "precision_convergence.csv"), "w") do io
        println(io, "# generator=reproduction/papers/160/generate.jl")
        println(io, "# method=requested decimal digits 60 to 120; working bits ",
                _prec(60), " to ", _prec(120))
        println(io, "# allow_inf=kL,L_m")
        println(io, "# utdkernels_version=", UTD_VERSION)
        println(io, "# backend=", selected_backend())
        println(io, "# case_names=", join(labels, ";"))
        println(io, join([
            "alpha_rad", "phi_rad", "kL", "k_rad_per_m", "L_m",
            "h_rad", "condition_number",
            "ref_dps60_re", "ref_dps60_im",
            "ref_dps120_re", "ref_dps120_im", "rel_diff_vs_dps120",
        ], ","))
        for r in rows
            println(io, join(r, ","))
        end
    end
    return Dict("cases" => length(rows), "max_rel_diff" => maxdiff)
end

# Three-regime canonical GL8-vs-direct sweep for the regime figure.
function run_regime_sweep(regimes)
    hs = exp10.(range(-16, -1; length=76))
    rows = Vector{Vector{Float64}}()
    for (rid, g, tmargin) in regimes
        for h in hs
            ref = soft_hp(g, h; dps=70)
            push!(rows, [Float64(rid), h,
                rel(direct_soft(g, h), ref),
                rel(fixed_order_soft(g, h; order=8, transition_margin=tmargin, x_margin=1e-8), ref)])
        end
    end
    require_at_most(
        "three-regime continuation maximum error",
        maximum(r[4] for r in rows),
        1e-10,
    )
    write_csv(joinpath(DATA, "regime_sweep.csv"),
        ["regime","h_rad","direct_rel_error","gl8_rel_error"], rows;
        extra_comments=["regime 1=exterior alpha=1.5pi L=1",
                        "regime 2=interior alpha=0.7pi L=1",
                        "regime 3=far-field alpha=1.5pi L=Inf"])
    return rows
end

# ----------------------------------------------------------------------- macros
function write_macros(canon, ens, ff, deriv, elec, emap, oracle, scaled, derivcond, precconv, dfloor)
    path = joinpath(DATA, "results_macros.tex")
    open(path, "w") do io
        p(name, val) = println(io, "\\newcommand{\\", name, "}{", val, "}")
        p("EnsembleCases", ens["accepted_cases"])
        p("EnsembleProposed", ens["proposed_cases"])
        p("EnsembleRejectedDomain", ens["rejected_domain"])
        p("EnsembleRejectedDerivative", ens["rejected_small_derivative"])
        p("EnsembleInterior", ens["n_interior"])
        p("EnsembleExterior", ens["n_exterior"])
        p("FarFieldCases", ff["accepted_cases"])
        p("FarFieldProposed", ff["proposed_cases"])
        p("FarFieldRejectedDomain", ff["rejected_domain"])
        p("FarFieldRejectedDerivative", ff["rejected_small_derivative"])
        p("DerivativeCases", deriv["cases"])
        p("DerivativeProposed", deriv["proposed_cases"])
        p("DerivativeRejectedDomain", deriv["rejected_domain"])
        p("DerivativeRejectedDerivative", deriv["rejected_small_derivative"])
        p("ConditionSlope", round(canon["condition_slope"]; digits=3))
        p("LinearSlope", round(canon["linear_slope"]; digits=3))
        p("GLTwoSlope", round(canon["gl2_slope"]; digits=3))
        p("GLFourSlope", round(canon["gl4_slope"]; digits=3))
        p("DirectPNinetyFive", sci(ens["direct_p95"]))
        p("DirectMax", sci(ens["direct_max"]))
        p("UtdPNinetyFive", sci(ens["utd_p95"]))
        p("StableMedian", sci(ens["stable_median"]))
        p("StablePNinetyFive", sci(ens["stable_p95"]))
        p("StablePNinetyNine", sci(ens["stable_p99"]))
        p("StableMax", sci(ens["stable_max"]))
        p("LinearPNinetyFive", sci(ens["linear_p95"]))
        p("FarFieldStablePNinetyFive", sci(ff["stable_p95"]))
        p("FarFieldStableMax", sci(ff["stable_max"]))
        p("FarFieldDirectPNinetyFive", sci(ff["direct_p95"]))
        p("DerivativeMedian", sci(deriv["median"]))
        p("DerivativePNinetyFive", sci(deriv["p95"]))
        p("DerivativeMax", sci(deriv["maximum"]))
        p("DerivativeAbsMedian", sci(deriv["abs_median"]))
        p("DerivativeAbsMax", sci(deriv["abs_max"]))
        p("TransitionOracleCases", oracle["cases"])
        p("TransitionFOracleMax", sci(oracle["F_maximum"]))
        p("TransitionFPrimeOracleMax", sci(oracle["Fprime_maximum"]))
        p("UnscaledAtHundred", sci(scaled["unscaled_at_1e2"]))
        p("UnscaledAtTera", sci(scaled["unscaled_at_1e12"]))
        p("ScaledMax", sci(scaled["scaled_max"]))
        p("DerivCondMax", sci(derivcond["max_rel_error"]))
        p("ElectricalStableMax", sci(elec["stable_max"]))
        p("ElectricalDirectMedian", sci(elec["direct_median"]))
        p("ElectricalDirectMax", sci(elec["direct_max"]))
        p("DirectAtTwelve", sci(canon["direct_at_1e_12"]))
        p("UtdAtTwelve", sci(canon["utd_at_1e_12"]))
        p("StableAtTwelve", sci(canon["gl8_at_1e_12"]))
        p("DirectAtTiny", sci(canon["direct_at_1e_16"]))
        p("UtdAtTiny", sci(canon["utd_at_1e_16"]))
        p("StableAtTiny", sci(canon["gl8_at_1e_16"]))
        # Absolute-error quantities (R6): reference magnitudes and the implied
        # direct-evaluation absolute errors at h=1e-12 and h=1e-16.
        p("CanonicalAbsTwelve", sci(canon["ref_abs_1e_12"]))
        p("CanonicalAbsTiny", sci(canon["ref_abs_1e_16"]))
        p("DirectAbsTwelve", sci(canon["direct_abs_1e_12"]))
        p("DirectAbsTiny", sci(canon["direct_abs_1e_16"]))
        p("ErrorConditionSlope", round(ens["slope"]; digits=3))
        p("ErrorMapCases", emap["cases"])
        p("ErrorMapDirectFractionMicro", string(round(100 * emap["frac_1e_6"]; digits=1), "\\%"))
        p("ErrorMapDirectFractionPercent", string(round(100 * emap["frac_1e_2"]; digits=1), "\\%"))
        p("ErrorMapStableMax", sci(emap["stable_max"]))
        p("ErrorMapStablePNinetyNine", sci(emap["stable_p99"]))
        p("PrecConvMax", sci(precconv["max_rel_diff"]))
        p("PrecConvCases", precconv["cases"])
        p("DerivFloorCase", dfloor["case"])
        p("DerivFloorStableErr", sci(dfloor["gl8_error"]))
        p("DerivFloorLinearErr", sci(dfloor["linear_error"]))
        p("DerivFloorCorrectedErr", sci(dfloor["corrected_error"]))
    end
    cp(path, joinpath(PAPER, "results_macros.tex"); force=true)
end

# ------------------------------------------------------------------------- driver
const CANON = Geom(1.5 * pi, 1.7, 100.0, 1.0)         # exterior hero geometry
const CANON_INT = Geom(0.7 * pi, 0.35 * pi, 100.0, 1.0)  # interior canonical
const CANON_FF = Geom(1.5 * pi, 1.7, 100.0, Inf)      # far-field canonical
const ELEC_GEOM = Geom(1.57 * pi, 1.82, 1.0, 1.0)

println("oracle regression...");     println("oracle relative errors (coefficient, derivative) = ", check_oracle_regression())
println("canonical (exterior)...");  _, canon = run_canonical(CANON, "canonical_sweep")
println("coalescence...");           run_coalescence(CANON)
println("regime sweep...");          run_regime_sweep([(1, CANON, 1e-3), (2, CANON_INT, 1e-3), (3, CANON_FF, 5e-2)])
println("ensemble (interior+exterior)..."); _, ens = run_ensemble(800)
println("ensemble (far-field)...");  _, ff = run_ensemble(400; far_field=true)
println("derivative...");            _, deriv = run_derivative(400)
println("transition oracle...");     _, oracle = run_transition_oracle()
println("scaled transition...");     _, scaled = run_scaled_transition()
println("derivative condition...");  _, derivcond = run_derivative_condition()
println("derivative floor...");      dfloor = run_derivative_floor()
println("precision convergence..."); pc = run_precision_convergence(ens, ff)
println("electrical...");            _, elec = run_electrical(ELEC_GEOM)
println("error map...");             _, emap = run_error_map(ELEC_GEOM)
write_macros(canon, ens, ff, deriv, elec, emap, oracle, scaled, derivcond, pc, dfloor)
println("canon=", canon)
println("ensemble=", ens)
println("farfield=", ff)
println("deriv=", deriv)
println("oracle=", oracle)
println("scaled=", scaled)
println("derivcond=", derivcond)
println("dfloor=", dfloor)
println("precconv=", pc)
println("electrical=", elec)
println("map=", emap)
println("done; plot with paper/scripts/plotting/plot_paper.jl")
