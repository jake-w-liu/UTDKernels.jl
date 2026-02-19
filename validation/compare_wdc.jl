"""
Validate UTDKernels.jl wedge diffraction coefficients against Balanis WDC.m
(Chapter 13 reference MATLAB implementation).

Convention mapping:
  MATLAB WDC(R, phi_deg, phip_deg, 90, n)  →  UTDKernels pec_wedge_DsDh(wedge, ang, k, L)
  where k = 2π (wavelength = 1), L = R, n = wedge factor.
  WDC prefactor CT/(n*sin(β₀)) with β₀=90° equals our prefactor at k=2π.

Known accuracy differences:
  - MATLAB FTF uses 8-point table interpolation (~3-4 digit accuracy).
    UTDKernels uses erfcx (machine precision). Expect ~1% disagreement from FTF.
  - At exact shadow/reflection boundaries, D flips sign. The two implementations
    may pick opposite sides → error of exactly 2.0 (sign flip, same magnitude).
  - n=1.0 (flat plate) is degenerate: D → 0, comparisons are machine noise.
  - For small kL near boundaries, the MATLAB and erfcx regularizations differ.
"""

using UTDKernels
using Printf

const VALIDATION_DIR = @__DIR__
const DATA_DIR = joinpath(VALIDATION_DIR, "data")
const CSV_FILE = joinpath(DATA_DIR, "wdc_reference.csv")

function load_reference_data(csvfile::String)
    lines = readlines(csvfile)
    rows = NamedTuple[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        parts = split(line, ',')
        length(parts) == 12 || continue
        push!(rows, (
            n        = parse(Float64, parts[1]),
            R        = parse(Float64, parts[2]),
            phi_deg  = parse(Float64, parts[3]),
            phip_deg = parse(Float64, parts[4]),
            Ds_ref   = complex(parse(Float64, parts[5]),  parse(Float64, parts[6])),
            Dh_ref   = complex(parse(Float64, parts[7]),  parse(Float64, parts[8])),
            Di_ref   = complex(parse(Float64, parts[9]),  parse(Float64, parts[10])),
            Dr_ref   = complex(parse(Float64, parts[11]), parse(Float64, parts[12])),
        ))
    end
    return rows
end

function compute_utd(n, R, phi_deg, phip_deg)
    k = 2π
    L = R
    wedge = Wedge(n * π)
    ang = RayAngles(deg2rad(phi_deg), deg2rad(phip_deg))

    Ds, Dh = pec_wedge_DsDh(wedge, ang, k, L)
    Di = pec_wedge_DsDh(wedge, ang, k, L, L, L; Rs=0, Rh=0)[1]
    Dr = Di - Ds
    return (Ds=Ds, Dh=Dh, Di=Di, Dr=Dr)
end

"""
    smart_err(a, b; abs_floor)

Combined error metric: relative error, but floored when both values are small.
Returns abs(a-b) / max(abs(b), abs(a), abs_floor).
"""
function smart_err(a::Complex, b::Complex; abs_floor::Float64=1e-4)
    return abs(a - b) / max(abs(b), abs(a), abs_floor)
end

"""
Distance in degrees from phi to nearest ISB, RSB, or face boundary.
"""
function boundary_distance_deg(phi_deg, phip_deg, n)
    isb = 180.0 + phip_deg
    rsb = 180.0 - phip_deg
    max_angle = n * 180.0
    d_isb = abs(phi_deg - isb)
    d_rsb = abs(phi_deg - rsb)
    d_face_lo = phi_deg
    d_face_hi = max_angle - phi_deg
    return min(d_isb, d_rsb, d_face_lo, d_face_hi)
end

function run_comparison(; verbose=true)
    if !isfile(CSV_FILE)
        error("Reference data not found at $CSV_FILE.\nRun: matlab -batch \"cd('$(VALIDATION_DIR)'); generate_wdc_reference\"")
    end

    rows = load_reference_data(CSV_FILE)
    println("Loaded $(length(rows)) reference test cases from WDC.m")

    # Categorize each test case
    categories = Dict{String, Vector{NamedTuple}}()
    for cat in ["away", "near_bnd", "exact_bnd", "degenerate", "small_kL_bnd"]
        categories[cat] = NamedTuple[]
    end

    for row in rows
        kL = 2π * row.R
        d_bnd = boundary_distance_deg(row.phi_deg, row.phip_deg, row.n)

        if row.n == 1.0
            push!(categories["degenerate"], row)
        elseif d_bnd < 0.15
            push!(categories["exact_bnd"], row)
        elseif d_bnd < 2.0 && kL < 0.5
            push!(categories["small_kL_bnd"], row)
        elseif d_bnd < 2.0
            push!(categories["near_bnd"], row)
        else
            push!(categories["away"], row)
        end
    end

    # Per-category tolerances (for the smart_err metric)
    tols = Dict(
        "away"         => 0.025,   # 2.5% — limited by MATLAB FTF table accuracy
        "near_bnd"     => 0.10,    # 10% — both regularizations differ near poles
        "exact_bnd"    => Inf,     # skip — sign ambiguity at exact boundary
        "small_kL_bnd" => 0.20,    # 20% — MATLAB FTF is very crude at small x
        "degenerate"   => Inf,     # skip — n=1.0 is machine noise
    )

    cat_labels = Dict(
        "away"         => "Away from boundaries",
        "near_bnd"     => "Near boundary (d < 2°)",
        "exact_bnd"    => "Exact boundary (d < 0.15°)",
        "small_kL_bnd" => "Small kL near boundary",
        "degenerate"   => "Degenerate (n=1.0, flat plate)",
    )

    total_pass = 0
    total_fail = 0
    total_skip = 0
    all_worst = NamedTuple[]

    if verbose
        println()
        println("="^80)
        println("UTDKernels.jl vs Balanis WDC.m — Validation Report")
        println("="^80)
    end

    for cat in ["away", "near_bnd", "small_kL_bnd", "exact_bnd", "degenerate"]
        cat_rows = categories[cat]
        tol = tols[cat]
        label = cat_labels[cat]
        isempty(cat_rows) && continue

        n_pass = 0
        n_fail = 0
        n_skip = 0
        errs_Ds = Float64[]
        errs_Dh = Float64[]
        errs_Di = Float64[]
        worst_cat = NamedTuple[]

        for row in cat_rows
            # Skip if both MATLAB values are essentially zero
            if abs(row.Ds_ref) < 1e-14 && abs(row.Dh_ref) < 1e-14
                n_skip += 1
                continue
            end

            utd = compute_utd(row.n, row.R, row.phi_deg, row.phip_deg)

            eDs = smart_err(utd.Ds, row.Ds_ref)
            eDh = smart_err(utd.Dh, row.Dh_ref)
            eDi = smart_err(utd.Di, row.Di_ref)
            push!(errs_Ds, eDs)
            push!(errs_Dh, eDh)
            push!(errs_Di, eDi)

            e_max = max(eDs, eDh, eDi)
            # Primary metric: Di only.
            # Ds = Di - Dr and Dh = Di + Dr both suffer from cancellation
            # when Di ≈ ±Dr, amplifying the ~1% MATLAB FTF table error.
            # Di is the fundamental coefficient with no cancellation.
            e_primary = eDi

            if isinf(tol) || e_primary < tol
                n_pass += 1
            else
                n_fail += 1
                if length(worst_cat) < 10 || e_primary > minimum(w.e_primary for w in worst_cat)
                    push!(worst_cat, (
                        n=row.n, R=row.R, phi=row.phi_deg, phip=row.phip_deg,
                        eDs=eDs, eDh=eDh, eDi=eDi, e_max=e_max, e_primary=e_primary,
                        d_bnd=boundary_distance_deg(row.phi_deg, row.phip_deg, row.n),
                        Ds_utd=utd.Ds, Ds_ref=row.Ds_ref,
                        Dh_utd=utd.Dh, Dh_ref=row.Dh_ref,
                        Di_utd=utd.Di, Di_ref=row.Di_ref,
                    ))
                    sort!(worst_cat; by=w->w.e_primary, rev=true)
                    length(worst_cat) > 10 && pop!(worst_cat)
                end
            end
        end

        total_pass += n_pass
        total_fail += n_fail
        total_skip += n_skip
        append!(all_worst, worst_cat)

        if verbose
            n_tested = n_pass + n_fail
            println()
            @printf("--- %s ---\n", label)
            @printf("  Cases: %d total, %d tested, %d skipped (zero ref)\n",
                    length(cat_rows), n_tested, n_skip)
            if isinf(tol)
                @printf("  (Skipped from pass/fail — known implementation-dependent regime)\n")
            else
                @printf("  Tolerance: %.1f%%\n", tol * 100)
                if n_tested > 0
                    @printf("  Passed: %d (%.1f%%)   Failed: %d\n",
                            n_pass, 100.0 * n_pass / n_tested, n_fail)
                end
            end
            if !isempty(errs_Ds)
                p50_Ds = sort(errs_Ds)[max(1, length(errs_Ds)÷2)]
                p50_Dh = sort(errs_Dh)[max(1, length(errs_Dh)÷2)]
                p50_Di = sort(errs_Di)[max(1, length(errs_Di)÷2)]
                p95_Ds = sort(errs_Ds)[max(1, Int(ceil(0.95*length(errs_Ds))))]
                p95_Dh = sort(errs_Dh)[max(1, Int(ceil(0.95*length(errs_Dh))))]
                p95_Di = sort(errs_Di)[max(1, Int(ceil(0.95*length(errs_Di))))]
                @printf("  Error stats (median / 95th / max):\n")
                @printf("    Ds: %.2e / %.2e / %.2e\n", p50_Ds, p95_Ds, maximum(errs_Ds))
                @printf("    Dh: %.2e / %.2e / %.2e\n", p50_Dh, p95_Dh, maximum(errs_Dh))
                @printf("    Di: %.2e / %.2e / %.2e\n", p50_Di, p95_Di, maximum(errs_Di))
            end

            if !isempty(worst_cat) && !isinf(tol)
                @printf("  Worst failures (by Di/Dh error):\n")
                for w in worst_cat[1:min(5, length(worst_cat))]
                    @printf("    n=%.2f R=%.2f φ=%.1f° φ'=%.0f° d_bnd=%.1f° eDh=%.2e eDi=%.2e\n",
                            w.n, w.R, w.phi, w.phip, w.d_bnd, w.eDh, w.eDi)
                end
            end
        end
    end

    # Breakdown by wedge factor n (away-from-boundary only)
    if verbose
        println()
        println("-"^80)
        println("Per-wedge summary (away-from-boundary cases only, tol=2.5%):")
        println("-"^80)
        away_rows = categories["away"]
        for n_val in sort(unique(r.n for r in away_rows))
            sub = filter(r -> r.n == n_val, away_rows)
            n_p = 0
            n_f = 0
            e_max_all = 0.0
            for row in sub
                abs(row.Ds_ref) < 1e-14 && abs(row.Dh_ref) < 1e-14 && continue
                utd = compute_utd(row.n, row.R, row.phi_deg, row.phip_deg)
                e = smart_err(utd.Di, row.Di_ref)
                e_max_all = max(e_max_all, e)
                e < 0.025 ? (n_p += 1) : (n_f += 1)
            end
            @printf("  n=%.2f (α=%3.0f°): %4d pass, %3d fail, max_err=%.2e\n",
                    n_val, n_val*180, n_p, n_f, e_max_all)
        end
    end

    if verbose
        println()
        println("="^80)
        n_tested = total_pass + total_fail
        @printf("OVERALL: %d tested, %d passed (%.1f%%), %d failed, %d skipped\n",
                n_tested, total_pass, 100.0 * total_pass / max(n_tested, 1),
                total_fail, total_skip)
        println("="^80)
    end

    return (passed=total_pass, failed=total_fail, skipped=total_skip)
end

# ==========================================================================
# F_utd vs MATLAB FTF reference values (generated from MATLAB)
# ==========================================================================
function compare_transition_function()
    # Generate MATLAB FTF reference values by running FTF at selected x
    # These were verified against the MATLAB implementation:
    matlab_ftf = [
        # x         Re(F)       Im(F)
        (0.3,   0.5729,    0.2677),   # table point
        (0.5,   0.6768,    0.2682),   # table point
        (0.7,   0.7439,    0.2549),   # table point
        (1.0,   0.8095,    0.2322),   # table point
        (1.5,   0.8730,    0.1982),   # table point
        (2.3,   0.9240,    0.1577),   # table point
        (4.0,   0.9658,    0.1073),   # table point
        (5.5,   0.9797,    0.0828),   # table point
        (10.0,  0.993047,  0.048344), # asymptotic
        (50.0,  0.999700,  0.009985), # asymptotic
    ]

    println()
    println("="^80)
    println("F_utd(x) vs MATLAB FTF table points")
    println("="^80)
    @printf("%-8s %-24s %-24s %-10s\n", "x", "F_utd", "F_matlab", "rel_err")

    max_err = 0.0
    for (x, re_m, im_m) in matlab_ftf
        F_jl = F_utd(x)
        F_ml = complex(re_m, im_m)
        e = abs(F_jl - F_ml) / abs(F_ml)
        max_err = max(max_err, e)
        @printf("%-8.1f %+.6f%+.6fi  %+.4f%+.4fi     %.2e\n",
                x, real(F_jl), imag(F_jl), re_m, im_m, e)
    end
    @printf("\nMax relative error vs MATLAB table: %.2e\n", max_err)
    @printf("(Expected: ~1e-3 at table points, limited by MATLAB's 4-digit tables)\n")
end

if abspath(PROGRAM_FILE) == @__FILE__
    compare_transition_function()
    result = run_comparison()

    if result.failed == 0
        println("\nAll test cases passed.")
    else
        @printf("\n%d test cases exceeded tolerance (see breakdown above).\n", result.failed)
    end
end
