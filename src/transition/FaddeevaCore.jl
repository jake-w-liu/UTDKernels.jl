"""
Internal scaled-complementary-error-function/Faddeeva identities shared by
transition families. Public APIs keep problem-specific names and domains.
"""

using SpecialFunctions: erfcx

@inline function _faddeeva_erfcx(z::Number)
    try
        return erfcx(z)
    catch error
        error isa MethodError || rethrow()
        throw(ArgumentError(
            "scaled complementary error function is unavailable for $(typeof(z))",
        ))
    end
end

"""Internal Faddeeva function `w(z)=erfcx(-im*z)`."""
@inline _faddeeva_w(z::Number) = _faddeeva_erfcx(-im * z)
