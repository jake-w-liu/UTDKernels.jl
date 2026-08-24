"""
Branch-safe square root and related utilities.

All square roots in the library use this module to enforce a single global
branch convention.
"""

"""
    safe_sqrt(x::Number)

Principal-branch square root. For real negative x, returns i√|x|.
This is the standard `sqrt` behavior in Julia, but we wrap it to make
the branch policy explicit and auditable.
"""
safe_sqrt(x::Real) = sqrt(complex(x))
safe_sqrt(x::Complex) = sqrt(x)
