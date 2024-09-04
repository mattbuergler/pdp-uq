#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import pandas as pd
import requests
import zipfile

from quantile_forest import RandomForestQuantileRegressor
from sklearn.model_selection import train_test_split

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
    Flows (https://gitlab.ethz.ch/vaw/public/pdp-sim.git).


    Copyright (c) 2024 ETH Zurich, Matthias Bürgler, Daniel Valero, Benjamin Hohermuth,
    Robert M. Boes, David F. Vetsch; Laboratory of Hydraulics, Hydrology
    and Glaciology (VAW); Chair of hydraulic structures

"""
def download_and_extract_zip(url, save_path, extract_to='.'):
    try:
        # Step 1: Download the ZIP file
        response = requests.get(url)
        response.raise_for_status()

        # Step 2: Save the ZIP file
        with open(save_path, 'wb') as file:
            file.write(response.content)

        print(f"ZIP file downloaded successfully and saved as: {save_path}")

        # Step 3: Extract the ZIP file
        with zipfile.ZipFile(save_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

        print(f"ZIP file extracted successfully to: {extract_to}")

        # Step 4: Delete the ZIP file after successful extraction
        os.remove(save_path)
        print(f"ZIP file '{save_path}' deleted successfully after extraction.")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the ZIP file: {e}")
    except zipfile.BadZipFile as e:
        print(f"Error: The downloaded file is not a valid ZIP file: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def train_model(target_name, awcc_target_name, n_estimators=100, max_features=1.0, max_depth=None):
    # set path to simulation results
    path_to_data = '../data/simulation_results.csv'
    # check if the dataset of errors already exists, otherwise download it:
    if not os.path.exists(path_to_data):
        # Download the dataset from its original source:
        # Bürgler, M., Valero, D., Hohermuth, B., Boes, R.M., &  Vetsch, D.F. 2024.
        # Dataset for "Uncertainties in Measurements of Bubbly Flows Using Phase-Detection
        # Probes". ETH Zurich.
        # https://doi.org/10.3929/ethz-b-000664463

        dataset_url = "https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/664463/Dataset_Uncertainties_in_Measurements_of_Bubbly_Flows.zip?sequence=2&isAllowed=y"
        path_to_zip = '../data/Dataset_Uncertainties_in_Measurements_of_Bubbly_Flows.zip'
        # download_csv(dataset_url, path_to_data)
        download_and_extract_zip(dataset_url, path_to_zip, extract_to='../data')

    # set columns for features and target
    feature_names = ['id [-]','u_x_awcc [m/s]','T_ux_awcc [-]','c_real [-]','d_bx_real [m]','delta_x [m]','delta_y [m]','N_p [-]']

    # load and prep data
    df = pd.read_csv(path_to_data, index_col=None)

    df = df[df['n_awcc [-]'] > 100]
    df = df[~df['u_x_awcc [m/s]'].isna()]
    df = df[~df['T_ux_awcc [-]'].isna()]
    df = df.sort_values(['id [-]'])
    df = df.set_index('id [-]',drop=False)

    # create array for features and target
    features = df[feature_names].to_numpy()
    target = df[[target_name]].to_numpy()
    target = target.reshape((len(target),))

    feat_train, feat_test, tar_train, tar_test = train_test_split(features, target,test_size=0.2,random_state=0)

    # get simulation ids of training and test data
    train_id = feat_train[:,0].astype('int64')
    test_id = feat_test[:,0].astype('int64')

    feat_train = feat_train[:,1:]
    feat_test = feat_test[:,1:]

    tar_train = tar_train.reshape((len(tar_train),))
    tar_test = tar_test.reshape((len(tar_test),))

    qrf_model = RandomForestQuantileRegressor(random_state=0, n_estimators=n_estimators, max_features=max_features, max_depth=max_depth)

    qrf_model.fit(feat_train, tar_train)

    pred_test = qrf_model.predict(feat_test, quantiles=[0.5])
    pred_train = qrf_model.predict(feat_train, quantiles=[0.5])

    error_pred = pred_test - tar_test
    rel_error_pred = (pred_test - tar_test)/tar_test


    rmse_test = np.sqrt(np.square(pred_test - tar_test).mean(axis=0))
    rmse_train = np.sqrt(np.square(pred_train - tar_train).mean(axis=0))

    pred_awcc_test = np.asarray(df.loc[list(test_id),awcc_target_name])
    rmse_awcc_test = np.sqrt(np.square(pred_awcc_test - tar_test).mean(axis=0))

    pred_awcc_train = np.asarray(df.loc[list(train_id),awcc_target_name])
    rmse_awcc_train = np.sqrt(np.square(pred_awcc_train - tar_train).mean(axis=0))

    return rmse_test, rmse_train, rmse_awcc_test, rmse_awcc_train, qrf_model.n_estimators, qrf_model.max_depth, qrf_model.max_features

def main():

    param_grid = {
        'n_estimators': [3,10, 30, 100, 300, 1000, 3000],
        'max_depth': [3, 10, 30, 50, 70, 90],
        'max_features': [0.1, 0.3, 0.5, 0.7, 1.0]
    }

    target_name = 'u_x_real [m/s]'
    awcc_target_name = 'u_x_awcc [m/s]'
    results = pd.DataFrame(columns=['target_name','rmse_test','rmse_train','rmse_awcc_test','rmse_awcc_train','n_estimators','max_depth','max_features','time'])

    # test effect of number of n_estimators
    parameter = 'n_estimators'
    for parameter_value in param_grid[parameter]:
        print(f"Training the model with '{parameter}' = {parameter_value}.")
        t_start = time.time()
        rmse_test,rmse_train,rmse_awcc_test,rmse_awcc_train,n_estimators,max_depth,max_features = train_model(target_name, awcc_target_name,n_estimators=parameter_value)
        runtime = time.time() - t_start
        print(f"Training finished in {runtime:.0f}s.")
        results.loc[len(results)] = [target_name,rmse_test,rmse_train,rmse_awcc_test,rmse_awcc_train,n_estimators,max_depth,max_features,runtime]
    
    # test effect of number of max_depth
    parameter = 'max_depth'
    for parameter_value in param_grid[parameter]:
        print(f"Training the model with '{parameter}' = {parameter_value}.")
        t_start = time.time()
        rmse_test,rmse_train,rmse_awcc_test,rmse_awcc_train,n_estimators,max_depth,max_features = train_model(target_name, awcc_target_name,max_depth=parameter_value)
        runtime = time.time() - t_start
        print(f"Training finished in {runtime:.0f}s.")
        results.loc[len(results)] = [target_name,rmse_test,rmse_train,rmse_awcc_test,rmse_awcc_train,n_estimators,max_depth,max_features,runtime]

    # test effect of number of max_features
    parameter = 'max_features'
    for parameter_value in param_grid[parameter]:
        print(f"Training the model with '{parameter}' = {parameter_value}.")
        t_start = time.time()
        rmse_test,rmse_train,rmse_awcc_test,rmse_awcc_train,n_estimators,max_depth,max_features = train_model(target_name, awcc_target_name,max_features=parameter_value)
        runtime = time.time() - t_start
        print(f"Training finished in {runtime:.0f}s.")
        results.loc[len(results)] = [target_name,rmse_test,rmse_train,rmse_awcc_test,rmse_awcc_train,n_estimators,max_depth,max_features,runtime]

    target_name = 'T_ux_real [-]'
    awcc_target_name = 'T_ux_awcc [-]'

    # test effect of number of n_estimators
    parameter = 'n_estimators'
    for parameter_value in param_grid[parameter]:
        print(f"Training the model with '{parameter}' = {parameter_value}.")
        t_start = time.time()
        rmse_test,rmse_train,rmse_awcc_test,rmse_awcc_train,n_estimators,max_depth,max_features = train_model(target_name, awcc_target_name,n_estimators=parameter_value)
        runtime = time.time() - t_start
        print(f"Training finished in {runtime:.0f}s.")
        results.loc[len(results)] = [target_name,rmse_test,rmse_train,rmse_awcc_test,rmse_awcc_train,n_estimators,max_depth,max_features,runtime]

    # test effect of number of max_depth
    parameter = 'max_depth'
    for parameter_value in param_grid[parameter]:
        print(f"Training the model with '{parameter}' = {parameter_value}.")
        t_start = time.time()
        rmse_test,rmse_train,rmse_awcc_test,rmse_awcc_train,n_estimators,max_depth,max_features = train_model(target_name, awcc_target_name,max_depth=parameter_value)
        runtime = time.time() - t_start
        print(f"Training finished in {runtime:.0f}s.")
        results.loc[len(results)] = [target_name,rmse_test,rmse_train,rmse_awcc_test,rmse_awcc_train,n_estimators,max_depth,max_features,runtime]

    # test effect of number of max_features
    parameter = 'max_features'
    for parameter_value in param_grid[parameter]:
        print(f"Training the model with '{parameter}' = {parameter_value}.")
        t_start = time.time()
        rmse_test,rmse_train,rmse_awcc_test,rmse_awcc_train,n_estimators,max_depth,max_features = train_model(target_name, awcc_target_name,max_features=parameter_value)
        runtime = time.time() - t_start
        print(f"Training finished in {runtime:.0f}s.")
        results.loc[len(results)] = [target_name,rmse_test,rmse_train,rmse_awcc_test,rmse_awcc_train,n_estimators,max_depth,max_features,runtime]

    results.to_csv(f'../docs/hyperparameter_tuning/hyperparameter_tuning_results.csv')


if __name__ == "__main__":
    main()