# Complete physical sensitivity adapters live in DiffUTD production code.
# They derive source/receiver coordinates and jets from real world curves,
# retain the carrier and complete field, and freeze objective windows.
const SpatialStudy = DiffUTD.ShadowSensitivityStudy
isdefined(DiffUTD,:receiver_field_sensitivity) && isdefined(DiffUTD,:receiver_shadow_pullback) ||
    error("DiffUTD lacks the production 3D shadow-sensitivity integration")
