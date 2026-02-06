module UTDKernels

# Common
include("common/Types.jl")
include("common/AngleWrap.jl")
include("common/Branches.jl")
include("common/Numerics.jl")

# Transition function
include("transition/TransitionF.jl")

# Wedge
include("wedge/WedgeGeometry.jl")
include("wedge/WedgePEC.jl")
include("wedge/WedgeDyadic.jl")
include("wedge/Regimes.jl")

# Utilities
include("utils/Diagnostics.jl")

# Public API
export PhasorConvention, EXP_IWT
export Wedge, wedge_n, wedge_nu
export RayAngles, Distances, effective_L
export wrap_angle
export F_utd
export pec_wedge_DsDh, pec_wedge_apply_sh
export spreading_factor
export wedge_transition_args
export inspect_kp_terms

end # module
