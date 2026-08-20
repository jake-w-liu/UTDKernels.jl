# Validation

Instantiate the dedicated environment once from the package root:

```bash
julia --project=validation -e 'using Pkg; Pkg.instantiate()'
```

Generate the numerical datasets and then render the validation figures:

```bash
julia --project=validation validation/generate_data.jl
julia --project=validation validation/plot_figures.jl
```

The WDC comparison additionally requires `validation/data/wdc_reference.csv`.
Generate it with MATLAB before running the comparison:

```bash
matlab -batch "cd('validation'); generate_wdc_reference"
julia --project=validation validation/compare_wdc.jl
```

Generated CSV and PDF files are ignored by Git. The environment files and
generator sources are tracked.
