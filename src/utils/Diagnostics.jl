"""
Diagnostic utilities for inspecting intermediate UTD quantities.
"""

"""
    inspect_kp_terms(wedge, ang, k, L)

Print a diagnostic summary of all four KP terms for debugging.
"""
function inspect_kp_terms(wedge::Wedge, ang::RayAngles, k::Number, L::Number)
    k = _validate_wavenumber(k)
    _validate_effective_L(L)
    n = wedge_n(wedge)
    phi, phip, _ = _effective_angles_for_kp(wedge, ang)
    terms = kp_four_terms(phi, phip, n)

    println("Wedge: α=$(wedge.alpha), n=$n, ν=$(wedge_nu(wedge))")
    println("Angles: φ=$phi, φ'=$phip")
    println("k=$k, L=$L")
    println()
    for j in 1:4
        psi = terms.psi[j]
        Nj  = terms.Nj[j]
        aj  = terms.aj[j]
        Xj  = k * L * aj
        Fj  = F_utd(Xj)
        println("Term $j: ψ=$(round(psi, digits=4)), N=$Nj, " *
                "a=$(round(aj, digits=6)), X=$(round(real(Xj), digits=6)), " *
                "F=$(round(Fj, digits=6)), cot(ψ)=$(round(cot(psi), digits=6))")
    end
end
