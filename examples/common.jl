using UTDKernels
using SpecialFunctions: erfcx
using Printf
using PlotlySupply

function save_plot_png(plot, outpath; width = 1150, height = 760)
    PlotlySupply.savefig(plot, outpath; width = width, height = height)
    println("Saved: $outpath")
end

db10(x) = 10 * log10(max(x, 1e-30))

function normalize_db(vals; floor_db = -60.0, power = false)
    vmax = maximum(vals)
    ref = max(vmax, 1e-30)
    # For field amplitudes, dB normalization should be 20*log10(|E|/|E|max),
    # equivalent to power dB below maximum. Set power=true only if `vals`
    # already contain power-like quantities.
    db_scale = power ? 10.0 : 20.0
    out = [db_scale * log10(max(v / ref, 1e-30)) for v in vals]
    return clamp.(out, floor_db, 0.0)
end

function max_rel_err(a, b)
    den = max.(abs.(b), 1e-14)
    maximum(abs.(a .- b) ./ den)
end

function rms_rel_err(a, b)
    den = max.(abs.(b), 1e-14)
    sqrt(sum(abs2.(a .- b) ./ (den .^ 2)) / length(a))
end

sinc_ratio(x) = abs(x) < 1e-12 ? 1.0 : sin(x) / x

function go_halfplane_components(phi, phip, k, rho)
    isb = π + phip
    rsb = π - phip
    inc = phi < isb ? exp(im * k * rho * cos(phi - phip)) : 0.0 + 0.0im
    refl = phi < rsb ? exp(im * k * rho * cos(phi + phip)) : 0.0 + 0.0im
    return (soft = inc - refl, hard = inc + refl, incident = inc, reflected = refl)
end

# ---------------------------------------------------------------------------
# Textbook half-plane (n=2) coefficient reconstruction
# ---------------------------------------------------------------------------

function F_textbook(x::Real)
    if isinf(x)
        return 1.0 + 0.0im
    end
    z = exp(im * π / 4) * sqrt(ComplexF64(x))
    return sqrt(π * ComplexF64(x)) * exp(im * π / 4) * erfcx(z)
end

kp_N(beta, sign_pm, n) = round(Int, (beta + sign_pm * π) / (2 * n * π), RoundNearestTiesAway)
kp_a(beta, N, n) = 2 * cos((2 * n * π * N - beta) / 2)^2
kp_psi(beta, sign_pm, n) = (π + sign_pm * beta) / (2 * n)

function wrap_angle_centered_pi(phi::Real)
    w = mod(phi + π, 2π) - π
    return w == -π ? π : w
end

function cotF_textbook(psi, a, k, L; tol = 1e-12)
    if isinf(L)
        return abs(sin(psi)) < tol ? 0.0 + 0.0im : cot(psi)
    end
    if abs(sin(psi)) < tol && abs(a) < 1e-28
        return 0.0 + 0.0im
    end
    return cot(psi) * F_textbook(k * L * a)
end

function halfplane_textbook_coeffs(phi, phip, k, L; centered = false)
    n = 2.0
    φ = centered ? wrap_angle_centered_pi(phi) : phi
    φp = centered ? wrap_angle_centered_pi(phip) : phip

    βm = φ - φp
    βp = φ + φp

    ψ1 = kp_psi(βm, +1, n)
    ψ2 = kp_psi(βm, -1, n)
    ψ3 = kp_psi(βp, +1, n)
    ψ4 = kp_psi(βp, -1, n)

    N1 = kp_N(βm, +1, n)
    N2 = kp_N(βm, -1, n)
    N3 = kp_N(βp, +1, n)
    N4 = kp_N(βp, -1, n)

    a1 = kp_a(βm, N1, n)
    a2 = kp_a(βm, N2, n)
    a3 = kp_a(βp, N3, n)
    a4 = kp_a(βp, N4, n)

    c1 = cotF_textbook(ψ1, a1, k, L)
    c2 = cotF_textbook(ψ2, a2, k, L)
    c3 = cotF_textbook(ψ3, a3, k, L)
    c4 = cotF_textbook(ψ4, a4, k, L)

    C = -exp(-im * π / 4) / (2 * n * sqrt(2π * k))
    Di = C * (c1 + c2)
    Dr = C * (c3 + c4)

    return (Di = Di, Dr = Dr, Ds = Di - Dr, Dh = Di + Dr)
end

function halfplane_package_coeffs(phi, phip, k, L)
    w = Wedge(2π)
    ang = RayAngles(phi, phip)
    Ds, Dh = pec_wedge_DsDh(w, ang, k, L)
    Di = pec_wedge_DsDh(w, ang, k, L, L, L; Rs = 0, Rh = 0)[1]
    Dr = Di - Ds
    return (Di = Di, Dr = Dr, Ds = Ds, Dh = Dh)
end

function fig_path(filename)
    outdir = joinpath(@__DIR__, "figs")
    mkpath(outdir)
    return joinpath(outdir, filename)
end
