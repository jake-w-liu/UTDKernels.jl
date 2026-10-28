"""
    wedge_cauchy_coefficients(Krho, phi, phi_i, N, polarization;
                              saddle=0, ncoef=14, points=2048)

Independent Cauchy-trapezoid coefficients of the unrearranged cotangent
spectral kernel after explicit pole extraction. This validation path does not
use the algebraic amplitude or the production binomial coefficients.
"""
function wedge_cauchy_coefficients(Krho::Real, phi::Real, phi_i::Real,
                                    N::Real, polarization::Symbol;
                                    saddle::Real=0.0, ncoef::Integer=14,
                                    points::Integer=2048)
    N == 2 || throw(ArgumentError("the pole-pair reference requires half-plane N=2"))
    ncoef >= 0 || throw(ArgumentError("ncoef must be nonnegative"))
    points > 2ncoef || throw(ArgumentError("Cauchy points must exceed twice ncoef"))
    maps = wedge_phase_map(Krho; saddle=saddle)
    poles = typeof(maps.c + phi + phi_i)[]
    for m in -2:2, sign in (-1,1)
        pole = maps.u_of(-phi-pi+sign*phi_i+4pi*m)
        any(z -> abs((z-pole)/maps.c) < 1e-12, poles) || push!(poles,pole)
    end
    length(poles) == 2 || throw(ArgumentError("the reference requires two distinct poles"))
    r = min(1.2,0.6abs(maps.c))
    values = map(0:points-1) do j
        u = r*cis(2pi*j/points)
        sigma = maps.sigma_of(u)
        wedge_kernel(phi+sigma,phi_i,N,polarization) * prod(z-u for z in poles) / maps.jac(sigma)
    end
    return [sum(values[j+1]*cis(-2pi*m*j/points) for j in 0:points-1)/(points*r^m) for m in 0:ncoef]
end
