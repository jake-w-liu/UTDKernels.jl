using Test

@testset "UTDKernels.jl" begin
    include("test_transition.jl")
    include("test_wedge_pec_continuity.jl")
    include("test_wedge_pec_generalized_L.jl")
    include("test_wedge_pec_infinite_L.jl")
    include("test_wedge_pec_limits.jl")
    include("test_interior_wedge.jl")
    include("test_symmetry.jl")
    include("test_ad.jl")
    include("test_wedge_impedance.jl")
    include("test_reference_values.jl")
    include("test_maliuzhinets.jl")
end
