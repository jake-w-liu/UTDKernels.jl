# Balanis GTD Examples (13-3 to 13-7)

This folder reproduces the canonical edge-diffraction examples from `balanis_gtd.pdf`:

- Example 13-3: Plane wave on PEC half-plane
- Example 13-4: Line source above finite-width strip
- Example 13-5: Monostatic 2D scattering width of strip
- Example 13-6: `λ/4` monopole on finite square ground plane
- Example 13-7: `λ/4` monopole on circular ground plane

Each script compares:

- `UTDKernels` package evaluation
- a direct "textbook equation" reconstruction from the published formulas

Plots are generated with `PlotlySupply` and exported as PNG files in
`examples/figs/`. The generated PNG files are intentionally not tracked.

## Run all examples

```bash
julia --project=examples examples/run_all.jl
```

## Run one example

```bash
julia --project=examples -e 'using Pkg; Pkg.instantiate(); include("examples/example_13_5.jl"); run_example_13_5()'
```

## Outputs

- `examples/figs/example13_3_halfplane_components.png`
- `examples/figs/example13_4_strip_line_source.png`
- `examples/figs/example13_5_monostatic_strip_sw.png`
- `examples/figs/example13_6_monopole_square_ground.png`
- `examples/figs/example13_7_monopole_circular_ground.png`

Notes:

- Example 13-7 uses the textbook two-point curved-edge approximation in the range `10° ≤ θ ≤ 170°`.
- For all examples, the comparison is pattern-level and uses the same phasor convention as `UTDKernels` (`exp(+iωt)`).
- In Examples 13-4/13-6/13-7, the "Textbook Eq." curves are evaluated from standalone textbook equations (no calls to `pec_wedge_DsDh` for those traces), using explicit continuous angle branches to avoid branch-cut aliasing.
