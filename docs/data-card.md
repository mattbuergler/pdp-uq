# Data card

## Dataset

**Title:** Dataset for "Uncertainties in Measurements of Bubbly Flows Using
Phase-Detection Probes"

**Permanent identifier:**
[doi:10.3929/ethz-b-000664463](https://doi.org/10.3929/ethz-b-000664463)

**License:** Creative Commons Attribution 4.0 International

**Authors:** Matthias Bürgler, Daniel Valero, Benjamin Hohermuth, Robert M.
Boes, and David F. Vetsch

The immutable source artifact is a 19,617-row, 65-column CSV generated from
synthetic phase-detection-probe simulations. Its expected SHA-256 digest is
`62442b127713e48113a22042a4894aee181ae497e59618437c23ea1ee210c143`.
The acquisition code streams the archive, selects exactly one expected member
without unsafe archive extraction, and verifies its digest before use.

## Why the data exists

The dataset quantifies errors introduced when turbulent bubbly flows are
sampled using dual-tip phase-detection probes and processed with AWCC. Each
simulation includes prescribed ground-truth flow properties, synthetic
probe-response measurements, probe geometry, and processing settings. This
paired structure makes supervised bias correction possible.

## Model-development fields

Seven features are used:

| Prepared feature | Source | Meaning |
| --- | --- | --- |
| `u [m/s]` | `u_x_awcc [m/s]` | AWCC mean streamwise velocity |
| `T_u [-]` | `T_ux_awcc [-]` | AWCC turbulence intensity |
| `c [-]` | `c_real [-]` | air concentration |
| `d_32a [m]` | `d_bx_real [m]` | bubble-size proxy |
| `delta_x [m]` | source column | streamwise tip separation |
| `delta_y [m]` | source column | lateral tip separation |
| `N_p [-]` | source column | particles per AWCC window |

Targets are `u_x_real [m/s]` and `T_ux_real [-]`. The original simulation ID
is retained to make the train/test split stable and auditable.

## Preparation

`pdp-uq data prepare` performs the following deterministic operations:

1. verifies all required columns and numeric types;
2. excludes simulations with 100 or fewer valid AWCC windows;
3. excludes rows with missing required features or targets;
4. renames source fields to the public schema;
5. flags measured features outside the configured application domain;
6. writes a validation report with counts, ranges, and source digest.

The current result contains 19,048 eligible rows:

| Check | Count |
| --- | ---: |
| source rows | 19,617 |
| removed for insufficient AWCC windows | 569 |
| removed for missing required values | 0 |
| retained rows | 19,048 |
| retained but outside measured-input guard | 1,212 |

The default policy flags, rather than silently removes, difficult measured
cases. This preserves the published failure behavior and makes the policy
visible in `configs/model.toml`.

## Splitting and leakage controls

Simulation IDs are shuffled using NumPy's deterministic generator with seed 0.
Eighty percent are frozen as training data and 20% as the final holdout.
Randomized hyperparameter search uses only the training IDs and performs
five-fold shuffled cross-validation with the same recorded seed. Holdout
labels do not influence model selection.

## Representativeness and limitations

The source data is synthetic and was designed around bubbly flows on spillway
and tunnel chutes. The publication sampled most parameters uniformly over
stated ranges, while integral time scale followed a log-normal distribution.
This is not a random sample of all physical installations.

Important exclusions and assumptions include spherical bubbles, no phase slip,
idealized point probes, constant modeled velocity covariance, and omitted
intrusive bubble-probe effects. See the [model card](model-card.md) and the
[paper](Buergler_et_al_2024_Uncertainties.pdf) for the scientific context.

## Reproduction and integrity

```bash
uv sync --locked --all-extras
uv run pdp-uq data fetch
uv run pdp-uq data validate data/simulation_results.csv
uv run dvc repro prepare split
```

The DOI, download endpoint, archive member, expected shape, and digest live in
`artifacts/manifest.json`. DVC records the content identity and downstream
lineage in `dvc.lock`; it is not the canonical publisher of the dataset.

## Unpublished expanded backup

The larger 19,649-row CSV formerly stored in Git is preserved separately under
`data/archive/` with its own checksum, DVC pointer, and provenance manifest.
It is a local unpublished backup and is deliberately excluded from the active
training pipeline until it receives a versioned Research Collection record.
