# Independent BigFloat scaled-complementary-error-function oracle for the
# passive transition sector. It does not call UTDKernels transition code or
# SpecialFunctions.erfcx.

passive_oracle_bits(digits::Integer) = begin
    digits >= 1 || throw(ArgumentError("digits must be positive"))
    max(256, ceil(Int, digits * log2(10)) + 32)
end

# Exact rational-angle geometry: n=3/2, phi=7pi/10, phi'=3pi/10.
# Three ordinary terms use the independent BigFloat erfcx implementation;
# the fourth uses the analytic positive-side cotangent-transition limit.
function complex_boundary_wedge_oracle(k, L; bits=384)
    setprecision(BigFloat, bits) do
        p = BigFloat(pi)
        n = BigFloat(3) / 2
        kb, lb = Complex{BigFloat}(k), Complex{BigFloat}(L)
        prefactor = -exp(-im*p/4) / (2n*sqrt(2p*kb))
        values = Complex{BigFloat}[]
        for (beta, sign_pm) in ((2p/5, 1), (2p/5, -1), (p, 1))
            psi = (p + sign_pm*beta) / (2n)
            branch = round(Int, (beta + sign_pm*p) / (2n*p))
            a = 2cos((2n*p*branch-beta)/2)^2
            root = sqrt(kb*lb*a)
            phase = exp(im*p/4)
            transition = sqrt(p)*phase*root*_passive_oracle_erfcx(phase*root)
            push!(values, prefactor*cot(psi)*transition)
        end
        push!(values, -sqrt(kb*lb)/(2sqrt(kb)))
        return values[1]+values[2]-values[3]-values[4], sum(values)
    end
end

function _passive_oracle_erf_series(z::Complex{BigFloat})
    total = z
    term = z
    z2 = z * z
    tolerance = 8eps(BigFloat)
    for order in 1:20_000
        term *= -z2 / order
        increment = term / (2order + 1)
        total += increment
        abs(increment) <= tolerance * max(abs(total), eps(BigFloat)) && order > 8 &&
            return 2total / sqrt(BigFloat(pi))
    end
    error("passive oracle erf series did not converge")
end

function _passive_oracle_erfcx_fraction(z::Complex{BigFloat})
    twice_z2 = 2z * z
    tiny = eps(BigFloat)^2
    fraction = one(z)
    upper = fraction
    lower = zero(z)
    # High-precision arguments near the series/fraction crossover need more
    # iterations than large arguments; retain the same convergence tolerance.
    for order in 1:20_000
        coefficient = order / twice_z2
        lower = one(z) + coefficient * lower
        abs(lower) < tiny && (lower = complex(tiny))
        upper = one(z) + coefficient / upper
        abs(upper) < tiny && (upper = complex(tiny))
        lower = inv(lower)
        ratio = upper * lower
        fraction *= ratio
        abs(ratio - 1) <= 16eps(BigFloat) &&
            return inv(sqrt(BigFloat(pi)) * z * fraction)
    end
    error("passive oracle erfcx continued fraction did not converge")
end

function _passive_oracle_erfcx(z::Complex{BigFloat})
    real(z) < 0 && return 2exp(z * z) - _passive_oracle_erfcx(-z)
    abs(z) < BigFloat(5) / 2 &&
        return exp(z * z) * (1 - _passive_oracle_erf_series(z))
    return _passive_oracle_erfcx_fraction(z)
end

function passive_transition_oracle(x::Number; digits::Integer=90)
    setprecision(BigFloat, passive_oracle_bits(digits)) do
        xc = Complex{BigFloat}(BigFloat(real(x)), BigFloat(imag(x)))
        iszero(xc) && return (
            zero(xc), -one(xc), Complex{BigFloat}(Inf), Complex{BigFloat}(Inf),
        )
        root = sqrt(xc)
        phase = cis(BigFloat(pi) / 4)
        value = sqrt(BigFloat(pi)) * root * phase *
                _passive_oracle_erfcx(phase * root)
        recurrence = im + inv(2xc)
        first = recurrence * value - im
        second = recurrence * first - value / (2xc^2)
        return value, value - 1, first, second
    end
end

function passive_wedge_oracle(
    alpha::Real,
    phi::Real,
    phip::Real,
    k::Number,
    L::Real;
    digits::Integer=90,
)
    setprecision(BigFloat, passive_oracle_bits(digits)) do
        pi_big = BigFloat(pi)
        alpha_big = BigFloat(alpha)
        phi_big = BigFloat(phi)
        phip_big = BigFloat(phip)
        k_big = Complex{BigFloat}(BigFloat(real(k)), BigFloat(imag(k)))
        L_big = BigFloat(L)
        n = alpha_big / pi_big
        beta_minus = phi_big - phip_big
        beta_plus = phi_big + phip_big
        soft_signs = (1, 1, -1, -1)
        hard_signs = (1, 1, 1, 1)
        soft = zero(Complex{BigFloat})
        hard = zero(Complex{BigFloat})
        for (index, (beta, sign)) in enumerate((
            (beta_minus, 1), (beta_minus, -1),
            (beta_plus, 1), (beta_plus, -1),
        ))
            psi = (pi_big + sign * beta) / (2n)
            branch = round(Int, (beta + sign * pi_big) / (2n * pi_big))
            a = 2cos((2n * pi_big * branch - beta) / 2)^2
            transition = passive_transition_oracle(k_big * L_big * a; digits)[1]
            term = cot(psi) * transition
            soft += soft_signs[index] * term
            hard += hard_signs[index] * term
        end
        prefactor = -cis(-pi_big / 4) / (2n * sqrt(2pi_big * k_big))
        return prefactor * soft, prefactor * hard
    end
end
