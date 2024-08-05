# Intrinsic Bias Correction and Uncertainty Quantification of Phase-Detection Probe Measurements

Intrinsic Bias Correction and Uncertainty Quantification of Phase-Detection Probe Measurements (pdp-ibc-uq) is a regression model for the intrinsic bias correction and uncertainty quantification of mean velocity and turbulence intensity estimations obtained from dual-tip phase-detection probe measurements and AWCC \[[1](#References),[2](#References)\] processing.

The regression model is based on a quantile regression forest model \[[3](#References)\]. The regression model is a python script making use of the Python package [quantile-forest](https://github.com/zillow/quantile-forest) \[[4](#References)\]. 

Further, the model leverages a large of dataset of more than 19,000 simulations of phase-detection probe measurements produced with the [Stochastic Bubble Generator software](https://gitlab.ethz.ch/vaw/public/pdp-sim-tf.git). 


## Getting Started

### Prerequisites

pdp-ibc-uq requires the following dependencies:

- python==3.11.0
- numpy==1.26.4
- pandas==2.2.0
- matplotlib==3.8.3
- quantile-forest==1.3.2
- scikit-learn==1.4.0
- scipy==1.12.0

### Installation

To install pdp-ibc-uq, follow these steps:

1. Clone this repository to your local machine:

    ```bash
    git clone https://gitlab.ethz.ch/vaw/public/pdp-ibc-uq.git
    ```

2. Navigate to the cloned directory:

    ```bash
    cd pdp-ibc-uq
    ```

We recommend running pdp-ibc-uq in a virtual python environment using pipenv. The installation of pipenv is described in the [documentation](docs/user/setup_python_environment.md).

3. Install the required dependencies using pipenv:

    ```bash
    pipenv install
    ```

4. Activate the virtual environment:

    ```bash
    pipenv shell
    ```

5. You're ready to use pdp-ibc-uq!

## Usage

### Data Format Requirements

For the application of the model to your data, the data to be in specific format. More specifically, the data should be stored in a CSV-file. Further, the model requires the estimated mean velocities, the root mean quare velocity, the air concentration and the Sauter mean diameter of the air phase obtained from the AWCC processing as input data. The columns containing those values must have the following headings:

|u [m/s]|u_rms [m/s]|c [-]|d_32a [m]|
|-------|-----------|-----|---------|

The Sauter mean diameter diameter can be calculated as $d_{32a} = 1.5\overline{u}c/F$, where $\overline{u}$ is the mean velocity (time-average), $c$ is the air concentration, and $F$ is the bubble frequency (number of bubbles per unit of time).


### Bias Correction and Uncertainty Quantification of Measurements

In order to run the regression model for the bias correction and uncertainty quantification of your phase-detection probe measurements, call the Python script `regression_model.py` with the necessary command-line arguments. The required command-line arguments are the streamwise tip separation `dx`, the lateral tip separation `dy`, the number of particles per window specified during the AWCC processing `Np` and the absolute or relative path to the file containing your measurements.

```
python3 regression_model.py -dx 0.005 -dy 0.001 -Np 10 path/to/my_measurements.csv
```

The first time the regression model is run, it will train the quantile forest based on a large dataset of measurement errors from more than 19,000 simulations \[[5](#References)\]


### Results

The results will be written to a new file at path/to/my_measurements_uq.csv. The corrected mean velocities and turbulence intensities are written to columns labeled with 'u_corrected_qQUANTILE [m/s]' and 'T_u_corrected_qQUANTILE [-]', where 'QUANTILE' indicates the quantiles (0.025,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.975).


## Application example

This section explains how to apply the regression model to actual dual-tip phase-detection probe measurements. The measurements correspond to an invert-normal profile obtained on a laboratory scale spillway model and are located in [example/pdp_data.csv](example/pdp_data.csv). The file contains estimated mean velocities, the root mean quare velocities, air concentrations and the Sauter mean diameters of the air phase obtained from the AWCC processing the phase-detection probe signal. The dual-tip phase-detection probe used for the measurements has a streamwise tip separation of 0.00452 m and a lateral tip separation of 0.00092 m. Further, the AWCC algorithm was applied with 10 particles per window ($N_p$ = 10).

With the following command, the regression model is applied to the measurements:

```
python3 regression_model.py -dx 0.00452 -dy 0.00092 -Np 10 example/pdp_data.csv
```

The corrected mean velocites and turbulent intensities are written to the file `example/pdp_data_uq.csv`.

The [measurement](example/pdp_data.csv) contain an additional column for the invert-normal distance 'z [m]'. This allows to visuzalize the results with the python script [tools/plot_results.py](tools/plot_results.py): 

```
python3 tools/plot_results.py example/pdp_data_uq.csv
```

This produces the following figure:

![Corrected mean velocitiy and turbulence intensity profile, including median values, the interquartile range (IQR) and the 90% confidence interval (CI).](/docs/application_example/profiles_uq.png?raw=true)
**Figure 1:** Corrected mean velocitiy and turbulence intensity profile, including median values, the interquartile range (IQR) and the 90% confidence interval (CI).


## Support

For support, bug reports, or feature requests, please open an issue in the [issue tracker](https://gitlab.ethz.ch/vaw/multiphade/mpd/-/issues) or contact Matthias Bürgler at <buergler@vaw.baug.ethz.ch>.


## Authors and acknowledgment

This software is developed by Matthias Bürgler in collaboration and under the supervision of Dr. Daniel Valero, Dr. Benjamin Hohermuth, Dr. David F. Vetsch and Prof. Dr. Robert M. Boes. Matthias Bürgler and Dr. Benjamin Hohermuth were supported by the Swiss National Science Foundation (SNSF) [grant number 197208].


## Copyright notice

Copyright (c) 2024 ETH Zurich, Matthias Bürgler, Daniel Valero, Benjamin Hohermuth, David F. Vetsch, Robert M. Boes, D-BAUG, Laboratory of Hydraulics, Hydrology and Glaciology (VAW)

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## References

[1] Kramer, M., Valero, D., Chanson, H., & Bung, D. B. (2019). Towards reliable turbulence estimations with phase-detection probes: an adaptive window cross-correlation technique. Experiments in Fluids, 60(1), 2.

[2] Kramer, M., Hohermuth, B., Valero, D., & Felder, S. (2020). Best practices for velocity estimations in highly aerated flows with dual-tip phase-detection probes. International Journal of Multiphase Flow, 126, 103228.

[3] Meinshausen, N., & Ridgeway, G. (2006). Quantile regression forests. Journal of Machine Learning Research, 7(6).

[4] Johnson, R. A. (2024). quantile-forest: A Python Package for Quantile Regression Forests. Journal of Open Source Software, 9(93), 5976.

[5] Bürgler, M., Valero, D., Hohermuth, B., Boes, R. M., \&  Vetsch, D. F. 2024a. Dataset for "Uncertainties in Measurements of Bubbly Flows Using Phase-Detection Probes". ETH Zurich Research Collection. https://doig.org/10.3929/ethz-b-000664463.

## Citation

If you use this package in academic work, please consider citing our work (tba).