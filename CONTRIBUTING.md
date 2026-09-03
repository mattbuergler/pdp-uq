# Contributing

Thank you for improving `pdp-uq`. Changes should preserve both software
quality and the scientific meaning of the published model.

## Set up

```bash
git clone https://github.com/mattbuergler/pdp-uq.git
cd pdp-uq
uv sync --locked --all-extras
```

Create a focused branch and add tests for behavioral changes. Before opening a
pull request, run:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
uv build
```

## Scientific changes

Changes to features, filtering, split logic, hyperparameters, targets,
quantiles, or domain bounds are model changes, not refactors. For these:

- explain the scientific rationale;
- update the configuration and relevant card;
- regenerate DVC outputs and `dvc.lock`;
- report both point and probabilistic holdout metrics;
- do not tune against or repeatedly inspect the final holdout;
- preserve the previous artifact long enough to compare regressions.

Never commit a modified copy of the DOI dataset as if it were the source.
Update `artifacts/manifest.json` only for a new, independently verifiable
dataset release.

## Code changes

Public functions should be typed and documented. Prefer explicit schemas,
deterministic behavior, actionable error messages, and atomic file writes.
Tests must be isolated from network access unless the network behavior itself
is mocked.

## Reporting issues

Include the command, Python version, operating system, minimal input schema,
and full error message. Do not attach confidential or unpublished measurement
data to a public issue.
