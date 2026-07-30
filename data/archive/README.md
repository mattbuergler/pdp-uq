# Unpublished dataset backup

This directory preserves the larger simulation-result artifact that entered
the upstream Git repository in commit
`d788999481db0dfefffe52ea9be204fd0fdaaaae`.

It is an unpublished backup and is **not** an input to the maintained DVC
training pipeline. The active dataset remains the immutable ETH Research
Collection release:

- DOI: <https://doi.org/10.3929/ethz-b-000664463>
- active path: `data/simulation_results.csv`
- shape: 19,617 rows by 65 columns
- SHA-256:
  `62442b127713e48113a22042a4894aee181ae497e59618437c23ea1ee210c143`

## Backup artifact

- path: `data/archive/simulation_results_git_d788999.csv`
- shape: 19,649 rows by 66 columns
- SHA-256:
  `47424517229feaa420d89823b79ab83ac7774fa67196b5b6e27faf4d83a1dcc6`
- DVC pointer: `simulation_results_git_d788999.csv.dvc`

The backup is a strict superset of the released dataset:

- all 19,617 released simulation IDs and all values in the 65 shared columns
  are identical;
- the backup adds 32 simulation IDs;
- the backup adds `u_rms_x_awcc [m/s]`, which is derivable as
  `u_x_awcc [m/s] * T_ux_awcc [-]`;
- 31 of the 32 additional simulations pass the maintained threshold of more
  than 100 valid AWCC windows.

Do not change the active dataset manifest to this file until the expanded
dataset has been published with its own versioned Research Collection record.

The file and DVC cache currently exist only in this local repository. Until a
DVC remote is configured or the expanded dataset is published, this should
not be considered an off-machine disaster-recovery copy.
