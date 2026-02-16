using Test

@testset "UTDKernels.jl" begin
    include("test_transition.jl")
    include("test_wedge_pec_continuity.jl")
    include("test_wedge_pec_limits.jl")
    include("test_symmetry.jl")
    include("test_ad.jl")
    include("test_reference_values.jl")
end
