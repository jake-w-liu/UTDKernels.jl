# The complete spatial adapter is a production DiffUTD module, rather than
# a second implementation inside this reproduction tree. Its physical entry
# points assemble HalfPlaneWave geometry and reconstruct full Cartesian E/H.
# Abstract complex canonical inputs and independent oracles remain shared.
const SpatialStudy = DiffUTD.NullUniformStudy
isdefined(DiffUTD,:halfplane_field) && isdefined(DiffUTD,:amplitude_jet) ||
    error("DiffUTD lacks the production 3D null-transition integration")
