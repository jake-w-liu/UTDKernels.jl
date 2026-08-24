module UTDKernels

using LinearAlgebra

# Common
include("common/Types.jl")
include("common/Numerics.jl")
include("common/AngleWrap.jl")
include("common/Branches.jl")

# Transition function
include("transition/TransitionF.jl")
include("transition/TransitionFPrime.jl")

# Fresnel
include("fresnel/Fresnel.jl")

# Wedge
include("wedge/WedgeGeometry.jl")
include("wedge/WedgePEC.jl")
include("wedge/WedgeDyadic.jl")
include("wedge/WedgeImpedance.jl")
include("wedge/WedgeGrazing.jl")
include("wedge/Regimes.jl")

# Maliuzhinets exact solution
include("maliuzhinets/MaliuzhinetsFunction.jl")
include("maliuzhinets/MaliuzhinetsExact.jl")

# Utilities
include("utils/Diagnostics.jl")

# Public API
export PhasorConvention, EXP_IWT
export Wedge, wedge_n, wedge_nu
export RayAngles, Distances, effective_L
export wrap_angle
export F_utd, F_utd_prime, F_utd_minus_one
export pec_wedge_DsDh, pec_wedge_apply_sh
export GrazingDomainError, GrazingIntervalReport
export grazing_local_angles, grazing_interval_report
export two_term_kernel, two_term_kernel_derivative
export pec_wedge_DsDh_grazing, pec_wedge_Ds_linear
export wedge_DsDh
export WedgeFaceMaterial, ImpedanceWedge
export fresnel_te, fresnel_tm
export impedance_wedge_DsDh
export spreading_factor
export wedge_transition_args
export inspect_kp_terms
export psi_Phi, maliuzhinets_DsDh

end # module
