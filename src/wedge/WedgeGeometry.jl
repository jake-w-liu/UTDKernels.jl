"""
Wedge geometry helpers: angular differences, KP integers N_j, and
transition arguments a_j.
"""

"""
    kp_Nj(beta, sign_pm, n)

Compute the KP integer N that most nearly satisfies 2nπN - β = ±π.

  - `beta`: angular difference (φ-φ' or φ+φ')
  - `sign_pm`: +1 for the "+" term, -1 for the "-" term
  - `n`: wedge parameter α/π

Returns: integer N
"""
function kp_Nj(beta::Real, sign_pm::Int, n::Real)
    # Condition: 2nπN - β = sign_pm · π  =>  N = (β + sign_pm·π) / (2nπ)
    round(Int, (beta + sign_pm * π) / (2 * n * π))
end

"""
    kp_aj(beta, N, n)

Kouyoumjian–Pathak distance parameter:
    a = 2cos²((2nπN - β)/2)

This vanishes quadratically at the shadow/reflection boundary where
2nπN - β = ±π.
"""
function kp_aj(beta::Real, N::Int, n::Real)
    arg = (2 * n * π * N - beta) / 2
    2 * cos(arg)^2
end

"""
    cotangent_arg(beta, sign_pm, n)

Cotangent argument ψ = (π ± β) / (2n) for the KP four-term structure.
"""
function cotangent_arg(beta::Real, sign_pm::Int, n::Real)
    (π + sign_pm * beta) / (2 * n)
end

"""
    kp_four_terms(phi, phip, n)

Compute the four KP cotangent arguments ψ_j, the integers N_j,
and the distance parameters a_j.

Returns a NamedTuple with fields:
  - `psi`: NTuple{4,Float64}   (cotangent arguments)
  - `Nj`:  NTuple{4,Int}       (KP integers)
  - `aj`:  NTuple{4,Float64}   (distance parameters)
  - `beta`: NTuple{4,Float64}  (angular differences for each term)
"""
function kp_four_terms(phi::Real, phip::Real, n::Real)
    beta_minus = phi - phip
    beta_plus  = phi + phip

    # Term 1: (π + β⁻)/(2n), N from β⁻ + π condition
    psi1 = cotangent_arg(beta_minus, +1, n)
    N1   = kp_Nj(beta_minus, +1, n)
    a1   = kp_aj(beta_minus, N1, n)

    # Term 2: (π - β⁻)/(2n), N from β⁻ - π condition
    psi2 = cotangent_arg(beta_minus, -1, n)
    N2   = kp_Nj(beta_minus, -1, n)
    a2   = kp_aj(beta_minus, N2, n)

    # Term 3: (π + β⁺)/(2n), N from β⁺ + π condition
    psi3 = cotangent_arg(beta_plus, +1, n)
    N3   = kp_Nj(beta_plus, +1, n)
    a3   = kp_aj(beta_plus, N3, n)

    # Term 4: (π - β⁺)/(2n), N from β⁺ - π condition
    psi4 = cotangent_arg(beta_plus, -1, n)
    N4   = kp_Nj(beta_plus, -1, n)
    a4   = kp_aj(beta_plus, N4, n)

    return (
        psi  = (psi1, psi2, psi3, psi4),
        Nj   = (N1, N2, N3, N4),
        aj   = (a1, a2, a3, a4),
        beta = (beta_minus, beta_minus, beta_plus, beta_plus),
    )
end
