# Minimum-curvature gridding

RelGravPy now maintains **one** minimum-curvature interpolation algorithm:
the SciPy sparse variational solver exposed through
`relgravpy.utils.min_curvature()`.

For the complete method, numerical assumptions, padding examples, integration
with `fillnan()` and `pad_array()`, and reproduction instructions, see
[Minimum-curvature interpolation and padding](min_curvature_robust.md).

For numerical validation against a published gravity grid, including errors
in mGal for interior gaps and extrapolated borders, see
[SA GRAVITY 2016 benchmark](min_curvature_published_gravity_benchmark.md).

The former development-only alternate finite-difference implementation and
external executable wrapper have been removed. The method name
`"surface"` remains accepted as an alias inside existing gridding helpers,
but invokes the same SciPy solver.
