# Model card

## Model summary

`pdp-uq` contains two independent `RandomForestQuantileRegressor` models:

| Model | Inputs | Target |
| --- | --- | --- |
| Mean velocity | 7 shared features | true streamwise bubble velocity |
| Turbulence intensity | 7 shared features | true streamwise bubble turbulence intensity |

The seven inputs are measured AWCC velocity, derived AWCC turbulence
intensity, air concentration, Sauter mean air-chord diameter, streamwise and
lateral probe-tip separation, and the AWCC particle count per window.

The release configuration uses 100 trees, maximum depth 12, all features
eligible at each split, and random seed 0. Predictions expose quantiles
2.5%, 5%, 10%, 25%, 50%, 75%, 90%, 95%, and 97.5%.

## Intended use

The models estimate and correct intrinsic baseline errors in mean velocity and
turbulence intensity recovered by AWCC from dual-tip phase-detection probes in
turbulent bubbly flows. They also describe conditional predictive uncertainty.

Appropriate uses include:

- analysis of measurements within the documented flow and probe domain;
- sensitivity studies for dual-tip probe geometry;
- reproducible comparison with AWCC estimates;
- research and educational use with explicit uncertainty reporting.

The models are not safety-certified instruments and should not be the sole
basis for infrastructure, operational, or regulatory decisions.

## Evaluation

The deterministic split uses seed 0 and preserves an untouched 20% holdout.
Hyperparameter search, when requested, is confined to five-fold
cross-validation on the 80% training partition.

| Target | Rows | AWCC RMSE | Model RMSE | MAE | R² | 90% coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mean velocity | 3,810 | 10.291 m/s | 1.595 m/s | 0.581 m/s | 0.987 | 92.1% |
| Turbulence intensity | 3,810 | 0.0822 | 0.0355 | 0.0232 | 0.866 | 93.2% |

The empirical coverage exceeds the nominal 90% interval for both targets in
this split, so the intervals are mildly conservative in aggregate. Aggregate
coverage does not guarantee calibration in every part of the feature space.
Quantile crossing was 0% for both models.

Exact machine-readable results, including pinball loss at every predicted
quantile and 50%, 80%, 90%, and 95% interval diagnostics, are generated at
`reports/generated/metrics.json`.

## Domain and out-of-distribution behavior

Inference is limited to:

| Feature | Minimum | Maximum |
| --- | ---: | ---: |
| measured velocity | 1 m/s | 50 m/s |
| measured turbulence intensity | 0.01 | 0.35 |
| air concentration | 0.005 | 0.4 |
| Sauter mean diameter | 0.5 mm | 20 mm |
| streamwise tip separation | 0.5 mm | 10 mm |
| lateral tip separation | 0 mm | 2 mm |
| particles per window | 5 | 20 |

Rows outside these bounds are marked `within_model_domain = false`, receive a
specific `model_domain_warning`, and are assigned `NaN` predictions. This is a
fail-closed software guard, not proof that every in-range combination is
well-supported. Users should inspect local training-data density for critical
applications.

## Scientific limitations

The paper identifies limitations that cannot be removed by software:

- Synthetic bubbles are spherical and follow the instantaneous fluid velocity
  under a no-slip assumption.
- The model corrects toward true simulated bubble velocity statistics, not
  necessarily continuous-phase fluid velocity statistics.
- Probe tips are idealized as points. Flow separation, probe wakes, surface
  tension, bubble deformation, and other bubble-probe interactions are not
  modeled.
- Intrusive velocity bias is not corrected. The paper notes that it may require
  an additional, independently validated correction.
- The simulated signals represent controlled bubbly-flow assumptions.
  Polydispersity, non-spherical bubbles, strongly varying trajectories, and
  additional signal decorrelation may increase real-world uncertainty.
- The paper limits the application to bubbly flows with air concentration at
  or below 0.4 and explicitly warns against use outside its parameter ranges.

## Data and training caveats

The source contains 19,617 simulations. The preparation stage excludes 569
rows with 100 or fewer valid AWCC windows and retains 19,048 rows. It reports
1,212 retained rows whose measured AWCC-derived inputs lie outside the
application guard. These difficult cases are retained to represent AWCC
failure behavior; consequently, the baseline velocity RMSE is sensitive to
rare extreme errors. Report MAE alongside RMSE.

The paper reports a 300-iteration randomized search with five-fold
cross-validation. The default reproducible evaluation uses the fixed released
configuration. Run `dvc repro dvc-tune.yaml` to repeat the expensive search.

## Provenance and security

The raw dataset is tied to DOI
[10.3929/ethz-b-000664463](https://doi.org/10.3929/ethz-b-000664463) and a
SHA-256 digest in `artifacts/manifest.json`. Model digests, package versions,
features, targets, and training-data digest are recorded in
`data/model_metadata.json`; pipeline code and data dependencies are recorded
in `dvc.lock`.

Joblib artifacts use Python pickle semantics. Never load untrusted models.
The inference API verifies the expected SHA-256 digest before deserialization.

## Citation

Bürgler, M., Valero, D., Hohermuth, B., Boes, R. M., and Vetsch, D. F. (2024).
"Uncertainties in measurements of bubbly flows using phase-detection probes."
*International Journal of Multiphase Flow*, 181, 104978.
[doi:10.1016/j.ijmultiphaseflow.2024.104978](https://doi.org/10.1016/j.ijmultiphaseflow.2024.104978).
