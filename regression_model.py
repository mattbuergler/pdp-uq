#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
import time
import argparse
import numpy as np
import joblib
import pandas as pd 
import requests
from quantile_forest import RandomForestQuantileRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split


"""
    This is a regression model for the intrinsic bias correction
    and uncertainty quantification of mean velocity andturbulence
    intensity estimations obtained from dual-tip phase-detection
    probe measurements and AWCC processing.

    The regression model is based on a quantile regression forest
    model. The regression model is a python script making use of
    the Python package quantile-forest developed and distribute by
    zillow on https://github.com/zillow/quantile-forest. 

    Further, the model leverages a large of dataset of more than
    19,000 simulations of phase-detection probe measurements produced
    with the Phase-Detection Probe Simulator for Turbulent Bubbly 
    Flows (https://gitlab.ethz.ch/vaw/public/pdp-sim-tf.git).     


    Copyright (c) 2024 ETH Zurich, Matthias Bürgler, Daniel Valero, Benjamin Hohermuth,
    Robert M. Boes, David F. Vetsch; Laboratory of Hydraulics, Hydrology
    and Glaciology (VAW); Chair of hydraulic structures

"""

def download_csv(url, save_path):
    try:
        response = requests.get(url)
        response.raise_for_status()

        with open(save_path, 'wb') as file:
            file.write(response.content)

        print(f"CSV file downloaded successfully and saved as: {save_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading CSV file: {e}")

def train_model(target_name):
    # set path to simulation results
    path_to_data = 'data/simulation_results.csv'
    # check if the dataset of errors already exists, otherwise download it:
    if not os.path.exists(path_to_data):
        # Download the dataset from its original source:
        # Bürgler, M., Valero, D., Hohermuth, B., Boes, R. M., \&  Vetsch, D. F. 2024a. 
        # Dataset for "Uncertainties in Measurements of Bubbly Flows Using Phase-Detection
        # Probes". ETH Zurich Research Collection. 
        # https://doig.org/10.3929/ethz-b-000664463.

        dataset_url = "https://example.com/data.csv"

        download_csv(dataset_url, path_to_data)

    # set columns for features and target
    feature_names = ['u_x_awcc','T_ux_awcc','c_real','d_bx_real','delta_x','delta_y','N_p']

    # load and prep data
    df = pd.read_csv(path_to_data, index_col=None)

    df = df[df['n_awcc'] > 100]
    df = df[~df['u_x_awcc'].isna()]
    df = df[~df['T_ux_awcc'].isna()]

    # create array for features and target
    features = df[feature_names].to_numpy()
    target = df[[target_name]].to_numpy()
    target = target.reshape((len(target),))

    # limit max depth to prevent over-fitting
    qrf_model = RandomForestQuantileRegressor(random_state=0, max_depth=12)
    qrf_model.fit(features, target)
    # save
    joblib.dump(qrf_model, f"data/qrf_model_{'_'.join(target_name.split('_', 2)[:2])}.joblib")

def get_valid_range(column_name):
    if column_name == "u [m/s]":
        return  [1.0, 50.0]
    elif column_name == "T_u [-]":
        return  [0.01, 0.35]
    elif column_name == "c [-]":
        return  [0.005, 0.4]
    elif column_name == "d_32a [m]":
        return  [0.0005, 0.02]
    elif column_name == "delta_x [m]":
        return  [0.0005, 0.01]
    elif column_name == "delta_y [m]":
        return  [0.0, 0.002]
    elif column_name == "N_p [-]":
        return  [5, 20]
    else:
        print(f"Invalid column name '{column_name}'.")

def main(dx, dy, Np, path_to_file):
    t0 = time.time()
    # Define the quantiles
    quantiles = [0.025,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.975]
    # Define the expected column names
    expected_data_columns = ["u [m/s]", "u_rms [m/s]", "c [-]", "d_32a [m]"]

    # get the directory
    directory = os.path.dirname(path_to_file)

    # Load the CSV file
    data = pd.read_csv(path_to_file)

    # Check if all expected columns are present in the loaded data
    for col in expected_data_columns:
        if col not in data.columns:
            print(f"Error: Column '{col}' not found in the loaded data.")
            return
    
    # Calculate the turbulence intensity
    data['T_u [-]'] = data['u_rms [m/s]']/data['u [m/s]']


    # Create additional columns with parsed input parameters
    data['delta_x [m]'] = dx
    data['delta_y [m]'] = dy
    data['N_p [-]'] = Np

    # The columns used as features for the random forest
    rf_feature_columns = ['u [m/s]', 'T_u [-]', 'c [-]', 'd_32a [m]', 'delta_x [m]', 'delta_y [m]', 'N_p [-]']

    # Check if trained model already exists, or if model must be trained first
    if os.path.isfile("data/qrf_model_u_x.joblib"):
        # model exists, we can directly apply it
        pass
    else:
        # model does not exists, we must first train it
        print("\nThe regression model for mean velocity bias correction is run for the first time. This will require some time to train the model.")
        t1 = time.time()
        train_model('u_x_real')
        print(f"\nFinished training the model in {(time.time()-t1)/60:.1f} minutes.")

    # Check if trained model already exists, or if model must be trained first
    if os.path.isfile("data/qrf_model_T_ux.joblib"):
        # model exists, we can directly apply it
        pass
    else:
        # model does not exists, we must first train it
        print("\nThe regression model for turbulence intensity bias correction is run for the first time. This will require some time to train the model.")
        t1 = time.time()
        train_model('T_ux_real')
        print(f"\nFinished training the model in {(time.time()-t1)/60:.1f} minutes.")

    print(f"\nApplying the models to data in the file '{path_to_file}'.")
    # Load the trained quantile random forest to predict corrected mean velocities and uncertainties
    qrf_u = joblib.load("data/qrf_model_u_x.joblib")
    pred_u = qrf_u.predict(data[rf_feature_columns].to_numpy(), quantiles=quantiles)

    # Load the trained quantile random forest to predict corrected turbulence intensities and uncertainties
    qrf_T_u = joblib.load("data/qrf_model_T_ux.joblib")
    pred_Tu = qrf_T_u.predict(data[rf_feature_columns].to_numpy(), quantiles=quantiles)

    # Check for each column in data if the values are in the valid range
    out_of_range_sum = np.zeros(len(data))
    for column_name in rf_feature_columns:
        valid_range = get_valid_range(column_name)
        if valid_range:
            # check for out of range rows:
            out_of_range = ~data[column_name].between(valid_range[0], valid_range[1])
            out_of_range_sum = out_of_range_sum + out_of_range
            # Set predictions for out of range data to NaN
            pred_u[out_of_range, :] = np.nan
            pred_Tu[out_of_range, :] = np.nan
    n_out_of_range = np.sum(out_of_range_sum > 0)
    n_total = len(data)
    print(f'\nWarning: {n_out_of_range} out of {n_total} contained values outside the range of application.\nUncertainty was not quantified for those data points.')
    # Add the predictions for all quantiles to the data
    for ii, quantile in enumerate(quantiles):
        data[f'u_corrected_q{quantile} [m/s]'] = pred_u[:,ii]
    for ii, quantile in enumerate(quantiles):
        data[f'T_u_corrected_q{quantile} [-]'] = pred_Tu[:,ii]

    # Save the data to a new file
    path_to_new_file = path_to_file.replace('.csv','_uq.csv')
    print(f"\nSaving the results to a new file: '{path_to_new_file}'.")
    data.to_csv(path_to_new_file, index=False,na_rep='nan')

    print(f'\nFinished in {(time.time()-t0)/60.0:.1f} minutes.')

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process phase-detection probe data from a CSV file")

    # Add arguments
    parser.add_argument("-dx", type=float, help="Streamwise tip separation of the phase-detection probe [m]")
    parser.add_argument("-dy", type=float, help="Transverse tip separation of the phase-detection probe [m]")
    parser.add_argument("-Np", type=int, help="Number of particles per averaging window [-]")
    parser.add_argument("path_to_file", type=str, help="Path to the CSV data file")


    # Parse the arguments
    args = parser.parse_args()
    # Call the main function with parsed arguments
    main(args.dx, args.dy, args.Np, args.path_to_file)
