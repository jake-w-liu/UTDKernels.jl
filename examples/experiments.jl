"""Deterministic scalar normalization experiment; no plots or field solvers."""
module NullUniformNormalizationExperiment
using UTDKernels, QuadGK, SHA
const NU = UTDKernels.NullUniform
const ROOT = normpath(joinpath(@__DIR__, ".."))
const OUTPUT = joinpath(ROOT, "validation", "data", "null_uniform_normalization.csv")
const K_GRID = (16.0, 400.0, 1600.0)
const Z_GRID = (0.4 + 0.3im, 0.8 + 0.35im, 1.2 + 0.6im)
const LAMBDA = 1.2
const COARSE = (rtol=1e-8, atol=5e-12)
const TIGHT = (rtol=2e-13, atol=5e-15)
const NATIVE_ULPS = 128
const RESIDUAL_ULPS = 32

require(condition, message) = condition || error(message)

function polynomial_integral(k, z, coefficients, tolerance)
    require(k > 0 && imag(z) > 0, "The independent integral needs k>0 and Im(z)>0")
    # Explicit ascending powers are independent of the production Horner callback.
    integrand(u) = sum(coefficients[m+1] * (u / sqrt(k))^m for m in 0:4) *
                   exp(-u*u) / (z-u)
    raw, estimate = quadgk(integrand, -Inf, -2.0, -1.0, 0.0, 1.0, 2.0, Inf;
        rtol=tolerance.rtol, atol=pi*tolerance.atol)
    value, error_bound = im*raw/pi, estimate/pi
    target = max(tolerance.atol, tolerance.rtol*abs(value))
    require(isfinite(value) && isfinite(error_bound) && error_bound >= 0,
        "Independent quadrature returned a nonfinite value or estimate")
    require(error_bound <= target, "Independent quadrature missed its requested error target")
    return (; value, error_bound, target)
end

function acceptance_budgets(k, z, coefficients)
    h = inv(sqrt(k)); q = z*h; y = imag(z)
    # Integral of |u|^m exp(-u^2), m=0:4; no production moment helper is used.
    absolute_gaussian = (sqrt(pi), 1.0, sqrt(pi)/2, 1.0, 3sqrt(pi)/4)
    integral_scale = sum(abs(coefficients[m+1])*h^m*absolute_gaussian[m+1] for m in 0:4)/(pi*y)
    pole_scale = sum(abs(coefficients[m+1])*abs(q)^m for m in 0:4)/(sqrt(pi)*y)
    saddle_scale = h/sqrt(pi)*sum(abs(coefficients[m+1])*abs(q)^(m-1) for m in 1:4)
    classical_scale = pole_scale+saddle_scale
    remainder_scale = h^3*(abs(coefficients[4])+abs(coefficients[5]*q))/(2sqrt(pi))
    bh = NATIVE_ULPS*eps(Float64)*integral_scale
    bc = NATIVE_ULPS*eps(Float64)*classical_scale
    br = RESIDUAL_ULPS*eps(Float64)*remainder_scale
    subtraction = 8eps(Float64)*(integral_scale+classical_scale+remainder_scale)
    return (; bh, bc, br, subtraction, integral_scale, classical_scale)
end

function run_case(k, z)
    coefficients = NU.factored_null_coefficients(k, LAMBDA)
    amplitude = NU.polynomial_amplitude(coefficients)
    coarse = polynomial_integral(k, z, coefficients, COARSE)
    tight = polynomial_integral(k, z, coefficients, TIGHT)
    b = acceptance_budgets(k, z, coefficients)
    convergence = abs(coarse.value-tight.value)
    require(convergence <= coarse.error_bound+tight.error_bound+2b.bh,
        "Independent quadrature did not converge within its estimates and roundoff allowance")
    hierarchy = NU.hierarchy_value(k, z, coefficients)
    classical = NU.classical_pole_term(k, z, amplitude)
    q = z/sqrt(k)
    remainder = -im*(coefficients[4]+coefficients[5]*q)/(2sqrt(pi)*k^1.5)
    require(all(isfinite, (hierarchy, classical, remainder)), "Native scalar output is nonfinite")
    # The extra bh accounts for arithmetic in the independent integrand itself.
    hierarchy_budget = tight.error_bound+2b.bh
    classical_budget = tight.error_bound+b.bh+b.bc+b.br
    identity_budget = b.bh+b.bc+b.br+b.subtraction
    hierarchy_error = abs(hierarchy-tight.value)
    classical_error = abs(classical-(tight.value-remainder))
    identity_error = abs((hierarchy-classical)-remainder)
    require(hierarchy_error <= hierarchy_budget, "Hierarchy disagrees with the independent polynomial integral")
    require(classical_error <= classical_budget, "Classical pole extraction disagrees with its intended approximation")
    require(identity_error <= identity_budget, "Analytic quartic remainder identity failed")
    if k == 400.0 && z == 0.8+0.35im
        require(identity_error < 1e-17, "Original strict pole-extraction regression failed")
    end
    # The omitted remainder is intentional. These alternatives must fail the
    # same fixed oracle budget; no budget is enlarged after evaluating them.
    common_budget = max(hierarchy_budget, classical_budget, identity_budget)
    sign_error = abs((tight.value-classical)+remainder)
    factor_error = abs((tight.value-classical)-2remainder)
    order_error = abs(NU.hierarchy_value(k, z, coefficients; order=2)-tight.value)
    require(min(sign_error, factor_error, order_error) > 10common_budget,
        "An oracle control cannot distinguish a sign, factor or retained-order mutation")
    return (; k, z, hierarchy, classical, remainder, coarse, tight, convergence,
        hierarchy_error, classical_error, identity_error, hierarchy_budget,
        classical_budget, identity_budget, sign_error, factor_error, order_error)
end

function main()
    require(realpath(pkgdir(UTDKernels)) == realpath(ROOT), "Load this package checkout for the experiment")
    results = [run_case(k,z) for k in K_GRID for z in Z_GRID]
    source_hash = bytes2hex(sha256(read(joinpath(ROOT,"src","studies","null_uniform","baselines.jl"))))
    experiment_hash = bytes2hex(sha256(read(@__FILE__)))
    manifest_hash = bytes2hex(sha256(read(joinpath(ROOT,"Manifest.toml"))))
    header = ["k","z_real","z_imag","lambda","hierarchy_real","hierarchy_imag",
        "classical_real","classical_imag","remainder_real","remainder_imag",
        "coarse_integral_real","coarse_integral_imag","coarse_error_estimate",
        "tight_integral_real","tight_integral_imag","tight_error_estimate",
        "quadrature_difference","hierarchy_error","classical_error","identity_error",
        "hierarchy_budget","classical_budget","identity_budget","sign_mutation_error",
        "factor_mutation_error","order_mutation_error","coarse_rtol","coarse_atol",
        "tight_rtol","tight_atol","source_sha256_text","experiment_sha256_text","manifest_sha256_text"]
    mkpath(dirname(OUTPUT));temporary,io=mktemp(dirname(OUTPUT))
    try
        println(io,join(header,','))
        for r in results
            row = (r.k,real(r.z),imag(r.z),LAMBDA,real(r.hierarchy),imag(r.hierarchy),
                real(r.classical),imag(r.classical),real(r.remainder),imag(r.remainder),
                real(r.coarse.value),imag(r.coarse.value),r.coarse.error_bound,
                real(r.tight.value),imag(r.tight.value),r.tight.error_bound,r.convergence,
                r.hierarchy_error,r.classical_error,r.identity_error,r.hierarchy_budget,
                r.classical_budget,r.identity_budget,r.sign_error,r.factor_error,r.order_error,
                COARSE.rtol,COARSE.atol,TIGHT.rtol,TIGHT.atol,source_hash,experiment_hash,manifest_hash)
            require(length(row)==length(header), "Validation CSV row schema differs from its header")
            println(io,join(row,','))
        end
        close(io);mv(temporary,OUTPUT;force=true)
    finally
        isopen(io) && close(io)
        isfile(temporary) && rm(temporary)
    end
    println("NullUniform normalization experiment passed: ",length(results)," cases, two independent quadrature resolutions, sign/factor/order controls.")
    println("Validation CSV: ",OUTPUT)
    return results
end
end

if abspath(PROGRAM_FILE)==@__FILE__
    NullUniformNormalizationExperiment.main()
end
