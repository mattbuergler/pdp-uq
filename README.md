# pdp-uq

pdp-uq is a regression model for the intrinsic bias correction and uncertainty quantification of mean velocity and turbulence intensity estimations obtained from dual-tip phase-detection probe measurements and AWCC \[[1](#References),[2](#References)\] processing.

The regression model is based on a quantile regression forest model \[[3](#References)\]. The regression model is a python script making use of the Python package [quantile-forest](https://github.com/zillow/quantile-forest) \[[4](#References)\]. 

Further, the model leverages a large of dataset of more than 19,000 simulations of phase-detection probe measurements produced with the [Stochastic Bubble Generator software](link_to_sbg_code). 


# Getting Started

## Prerequisites

The pdp-uq requires the following dependencies:

 * python (>= 3.8) 
 * numpy (>= 1.23) 
 * scikit-learn (>= 1.0) 
 * scipy (>= 1.4)
 * quantile-forest (>= 1.4)

## Installation

```
python -m pip install --user --upgrade pdp-uq
```

# User Guide

## Data Format Requirements

For the application of the model to your data, the data to be in specific format. More specifically, the data should be stored in a CSV-file. Further, the model requires the estimated mean velocities, the root mean quare velocity, the air concentration and the Sauter mean diameter of the air phase obtained from the AWCC processing as input data. The columns containing those values must have the following headings:

|u [m/s]|u_rms [m/s]|c [-]|d_32a [m]|
|-------|-----------|-----|---------|

The Sauter mean diameter diameter can be calculated as $d_{32a} = 1.5\overline{u}c/F$, where $\overline{u}$ is the mean velocity (time-average), $c$ is the air concentration, and $F$ is the bubble frequency.


## Bias Correction and Uncertainty Quantification of Measurements

In order to run the regression model for the bias correction and uncertainty quantification of your phase-detection probe measurements, call the regression_model.py Python script with the necessary command-line arguments. The required command-line arguments are the streamwise tip separation `dx`, the lateral tip separation `dy`, the number of particles per window specified during the AWCC processing `Np` and the absolute or relative path to the file containing your measurements.

```
python3 emulator.py -dx 0.005 -dy 0.001 -Np 10 path/to/my_measurements.csv
```


## Results


'u [m/s]', 'T_u [-]', 'c [-]', 'd_32a [m]', 'delta_x [m]', 'delta_y [m]', 'N_p [-]'

# References

[1] Kramer, M., Valero, D., Chanson, H., & Bung, D. B. (2019). Towards reliable turbulence estimations with phase-detection probes: an adaptive window cross-correlation technique. Experiments in Fluids, 60(1), 2.

[2] Kramer, M., Hohermuth, B., Valero, D., & Felder, S. (2020). Best practices for velocity estimations in highly aerated flows with dual-tip phase-detection probes. International Journal of Multiphase Flow, 126, 103228.

[3] Meinshausen, N., & Ridgeway, G. (2006). Quantile regression forests. Journal of Machine Learning Research, 7(6).

[4] Johnson, R. A. (2024). quantile-forest: A Python Package for Quantile Regression Forests. Journal of Open Source Software, 9(93), 5976.

