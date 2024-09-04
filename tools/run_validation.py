#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import pandas as pd
import requests
import zipfile
import matplotlib
import matplotlib.pyplot as plt
from quantile_forest import RandomForestQuantileRegressor
from sklearn.model_selection import train_test_split

try:
    from plt_config import *
except ImportError:
    print("Error while importing modules")
    raise

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

def train_model(target_name,awcc_target_name):
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

    # limit max depth to prevent over-fitting
    qrf_model = RandomForestQuantileRegressor(random_state=0, max_depth=12)

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

    return tar_train, pred_train, tar_test, pred_test, pred_awcc_test, rmse_test, rmse_train, rmse_awcc_test, rmse_awcc_train


def main():

    t0 = time.time()

    fig, axs = plt.subplots(2, 2, figsize=(6.5,5.5))
    # fig.subplots_adjust(hspace=0.4, wspace=0.4)

    tar_train, pred_train, tar_test, pred_test, pred_awcc_test, rmse_test, rmse_train, rmse_awcc_test, rmse_awcc_train = train_model('u_x_real [m/s]','u_x_awcc [m/s]')
    min_val = 0.0
    max_val = 50.0
    alpha_value = 0.2
    axs[0,0].scatter(tar_train, pred_train, alpha=alpha_value, c='blue',edgecolors='none',label='Model')
    axs[0,0].plot([min_val, max_val], [min_val, max_val], 'k-', label='Perfect pred.')
    axs[0,0].plot([min_val, max_val], [1.1*min_val, 1.1*max_val], 'k--', label='$\pm$ 10% Error')
    axs[0,0].plot([min_val, max_val], [0.9*min_val, 0.9*max_val], 'k--')
    axs[0,0].set_xlabel('$\overline{u}_{x,real}$ [m s$^{-1}$]')
    axs[0,0].set_ylabel('$\overline{u}_{x,pred}$ [m s$^{-1}$]')
    axs[0,0].grid(True)
    axs[0,0].set_xlim([min_val,max_val])
    axs[0,0].set_ylim([min_val,max_val])
    axs[0,0].text(0.05*max_val,0.95*max_val,f'$\mathbf{{Training}}$\n$RMSE_{{pred}}$ = {rmse_train:.2f} ms$^{{1}}$',verticalalignment='top')
    # axs[0,0].legend(loc=4,frameon=True,edgecolor='k',fancybox='False',facecolor='w')
    axs[0,0].annotate('(a)', xy=(0.95, 0.05), xycoords='axes fraction',
                           ha='right', va='bottom')

    axs[0,1].scatter(tar_test, pred_test, alpha=alpha_value, c='blue',edgecolors='none',label='Model')
    axs[0,1].scatter(tar_test, pred_awcc_test, alpha=alpha_value,edgecolors='none', c='red',label='AWCC')
    axs[0,1].plot([min_val, max_val], [min_val, max_val], 'k-', label='Perfect pred.')
    axs[0,1].plot([min_val, max_val], [1.1*min_val, 1.1*max_val], 'k--', label='$\pm$ 10% Error')
    axs[0,1].plot([min_val, max_val], [0.9*min_val, 0.9*max_val], 'k--')
    axs[0,1].set_xlabel('$\overline{u}_{x,real}$ [m s$^{-1}$]')
    axs[0,1].set_ylabel('$\overline{u}_{x,awcc}$ or $\overline{u}_{x,pred}$ [m s$^{-1}$]')
    axs[0,1].set_xlim([min_val,max_val])
    axs[0,1].set_ylim([min_val,max_val])
    axs[0,1].text(0.05*max_val,0.95*max_val,f'$\mathbf{{Testing}}$\n$RMSE_{{pred}}$ = {rmse_test:.2f} ms$^{{1}}$\n$RMSE_{{awcc}}$ = {rmse_awcc_test:.2f} ms$^{{1}}$',verticalalignment='top')
    # axs[0,1].text(0.1*max_val,0.8*max_val,f'')
    axs[0,1].grid(True)
    # axs[0,1].legend(loc=4,frameon=True,edgecolor='k',fancybox='False',facecolor='w')
    axs[0,1].annotate('(b)', xy=(0.95, 0.05), xycoords='axes fraction',
                           ha='right', va='bottom')

    tar_train, pred_train, tar_test, pred_test, pred_awcc_test, rmse_test, rmse_train, rmse_awcc_test, rmse_awcc_train = train_model('T_ux_real [-]','T_ux_awcc [-]')
    min_val = 0.0
    max_val = 0.4
    axs[1,0].scatter(tar_train, pred_train, alpha=alpha_value, c='blue',edgecolors='none',label='Model')
    axs[1,0].plot([min_val, max_val], [min_val, max_val], 'k-', label='Perfect pred.')
    axs[1,0].plot([min_val, max_val], [1.1*min_val, 1.1*max_val], 'k--', label='$\pm$ 10% Error')
    axs[1,0].plot([min_val, max_val], [0.9*min_val, 0.9*max_val], 'k--')
    axs[1,0].set_xlabel('$\mathrm{T}_{u,x,real}$ [-]')
    axs[1,0].set_ylabel('$\mathrm{T}_{u,x,pred}$ [-]')
    axs[1,0].grid(True)
    axs[1,0].set_xlim([min_val,max_val])
    axs[1,0].set_ylim([min_val,0.8])
    axs[1,0].text(0.05*max_val,0.95*0.8,f'$\mathbf{{Training}}$\n$RMSE_{{pred}}$ = {rmse_train:.3f}',verticalalignment='top')
    # axs[1,0].legend(loc=1,frameon=True,edgecolor='k',fancybox='False',facecolor='w')
    axs[1,0].annotate('(c)', xy=(0.95, 0.05), xycoords='axes fraction',
                           ha='right', va='bottom')

    axs[1,1].scatter(tar_test, pred_test, alpha=alpha_value, c='blue',edgecolors='none')
    axs[1,1].scatter(tar_test, pred_awcc_test, alpha=alpha_value, c='red',edgecolors='none')
    axs[1,1].plot([min_val, max_val], [min_val, max_val], 'k-')
    axs[1,1].plot([min_val, max_val], [1.1*min_val, 1.1*max_val], 'k--')
    axs[1,1].plot([min_val, max_val], [0.9*min_val, 0.9*max_val], 'k--')
    axs[1,1].scatter([],[], alpha=0.8, c='blue',label='Model')
    axs[1,1].scatter([],[], alpha=0.8, c='red',label='AWCC')
    axs[1,1].plot([], [], 'k-', label='Perfect pred.')
    axs[1,1].plot([], [], 'k--', label='$\pm$ 10% Error')
    axs[1,1].set_xlabel('$\mathrm{T}_{u,x,real}$ [-]')
    axs[1,1].set_ylabel('$\mathrm{T}_{u,x,awcc}$ or $\mathrm{T}_{u,x,pred}$ [-]')
    axs[1,1].set_xlim([min_val,max_val])
    axs[1,1].set_ylim([min_val,0.8])
    axs[1,1].text(0.05*max_val,0.95*0.8,f'$\mathbf{{Testing}}$\n$RMSE_{{pred}}$ = {rmse_test:.3f}\n$RMSE_{{awcc}}$ = {rmse_awcc_test:.3f}',verticalalignment='top')
    # axs[1,1].text(0.1*max_val,0.8*0.8,f'')
    axs[1,1].grid(True)
    axs[1,1].legend(loc=1,frameon=True,edgecolor='k',fancybox='False',facecolor='w')
    axs[1,1].annotate('(d)', xy=(0.95, 0.05), xycoords='axes fraction',
                           ha='right', va='bottom')
    plt.tight_layout()
    fig.savefig(f"../docs/validation/regression_model_validation.jpg",dpi=1200)


    fig, axs = plt.subplots(1, 2, figsize=(6.5,3))
    # fig.subplots_adjust(hspace=0.4, wspace=0.4)

    tar_train, pred_train, tar_test, pred_test, pred_awcc_test, rmse_test, rmse_train, rmse_awcc_test, rmse_awcc_train = train_model('u_x_real [m/s]','u_x_awcc [m/s]')
    min_val = 0.0
    max_val = 50.0

    axs[0].scatter(tar_test, pred_test, alpha=alpha_value, c='blue',edgecolors='none', label='Model')
    axs[0].scatter(tar_test, pred_awcc_test, alpha=alpha_value, c='red',edgecolors='none',label='AWCC')
    axs[0].plot([min_val, max_val], [min_val, max_val], 'k-', label='Perfect pred.')
    axs[0].plot([min_val, max_val], [1.1*min_val, 1.1*max_val], 'k--', label='$\pm$ 10% Error')
    axs[0].plot([min_val, max_val], [0.9*min_val, 0.9*max_val], 'k--')
    axs[0].set_xlabel('$\overline{u}_{x,real}$ [m s$^{-1}$]')
    axs[0].set_ylabel('$\overline{u}_{x,awcc}$ or $\overline{u}_{x,pred}$ [m s$^{-1}$]')
    axs[0].set_xlim([min_val,max_val])
    axs[0].set_ylim([min_val,max_val])
    axs[0].text(0.05*max_val,0.95*max_val,f'$\mathbf{{Testing}}$\n$RMSE_{{pred}}$ = {rmse_test:.2f} ms$^{{1}}$\n$RMSE_{{awcc}}$ = {rmse_awcc_test:.2f} ms$^{{1}}$',verticalalignment='top')
    # axs[0].text(0.1*max_val,0.8*max_val,f'')
    axs[0].grid(True)
    # axs[0].legend(loc=4,frameon=True,edgecolor='k',fancybox='False',facecolor='w')
    tar_train, pred_train, tar_test, pred_test, pred_awcc_test, rmse_test, rmse_train, rmse_awcc_test, rmse_awcc_train = train_model('T_ux_real [-]','T_ux_awcc [-]')
    min_val = 0.0
    max_val = 0.4
    axs[1].scatter(tar_test, pred_test, alpha=alpha_value, c='blue',edgecolors='none')
    axs[1].scatter(tar_test, pred_awcc_test, alpha=alpha_value, c='red',edgecolors='none')
    axs[1].plot([min_val, max_val], [min_val, max_val], 'k-')
    axs[1].plot([min_val, max_val], [1.1*min_val, 1.1*max_val], 'k--')
    axs[1].plot([min_val, max_val], [0.9*min_val, 0.9*max_val], 'k--')
    axs[1].scatter([],[], alpha=1.0, c='blue',label='Model')
    axs[1].scatter([],[], alpha=1.0, c='red',label='AWCC')
    axs[1].plot([], [], 'k-', label='Perfect pred.')
    axs[1].plot([], [], 'k--', label='$\pm$ 10% Error')
    axs[1].set_xlabel('$\mathrm{T}_{u,x,real}$ [-]')
    axs[1].set_ylabel('$\mathrm{T}_{u,x,awcc}$ or $\mathrm{T}_{u,x,pred}$ [-]')
    axs[1].set_xlim([min_val,max_val])
    axs[1].set_ylim([min_val,0.8])
    axs[1].text(0.05*max_val,0.95*0.8,f'$\mathbf{{Testing}}$\n$RMSE_{{pred}}$ = {rmse_test:.3f}\n$RMSE_{{awcc}}$ = {rmse_awcc_test:.3f}',verticalalignment='top')
    # axs[1].text(0.1*max_val,0.8*0.8,f'')
    axs[1].grid(True)
    axs[1].legend(loc=1,frameon=True,edgecolor='k',fancybox='False',facecolor='w')
    plt.tight_layout()
    fig.savefig(f"../docs/validation/regression_model_validation_graphical_abstract.jpg",dpi=1200)

    print(f'\nFinished in {(time.time()-t0)/60.0:.1f} minutes.')

if __name__ == "__main__":
    main()
