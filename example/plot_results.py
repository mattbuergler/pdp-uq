import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

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

def plot_profiles(data, directory):
    # Set plt parameters
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.sans-serif'] = ['Computer Modern Roman']
    plt.rcParams['font.size'] = 9
    plt.rcParams['mathtext.fontset'] = 'cm'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.4))
    ax1.plot(data['u [m/s]'].to_numpy(),data['z [m]'].to_numpy(),fillstyle='none',linestyle='-',marker='.',color='k',label='AWCC')
    ax1.plot(data['u_corrected_q0.5 [m/s]'].to_numpy(),data['z [m]'].to_numpy(),fillstyle='none',linestyle='--',marker='.',color='b',label='Model median')
    ax1.fill_betweenx(data['z [m]'].to_numpy(),data['u_corrected_q0.25 [m/s]'], data['u_corrected_q0.75 [m/s]'], color='b', alpha=0.3, label='IQR')
    # ax1.fill_betweenx(data['z [m]'].to_numpy(),data['u_corrected_q0.1 [m/s]'], data['u_corrected_q0.9 [m/s]'], color='b', alpha=0.1, label='80%-CI')
    ax1.fill_betweenx(data['z [m]'].to_numpy(),data['u_corrected_q0.05 [m/s]'], data['u_corrected_q0.95 [m/s]'], color='b', alpha=0.1, label='Model 90%-CI')
    # ax1.fill_betweenx(data['z [m]'].to_numpy(),data['u_corrected_q0.025 [m/s]'], data['u_corrected_q0.975 [m/s]'], color='b', alpha=0.1, label='95%-CI')
    ax1.set_xlabel('$\overline{u}_x$ [m/s]')
    ax1.set_ylabel('$z$ [m]')
    ax1.set_xlim([0,1.1*max(np.nanmax(data['u [m/s]'].to_numpy()),np.nanmax(data['u_corrected_q0.9 [m/s]'].to_numpy()))])
    ax1.set_ylim([0,1.1*np.nanmax(data['z [m]'].to_numpy())])
    ax1.text(0.05*ax1.get_xlim()[1],0.05*ax1.get_ylim()[1],'(a)')
    ax2.plot(data['T_u [-]'].to_numpy(),data['z [m]'].to_numpy(),fillstyle='none',linestyle='-',marker='.',color='k',label='AWCC')
    ax2.plot(data['T_u_corrected_q0.5 [-]'].to_numpy(),data['z [m]'].to_numpy(),fillstyle='none',linestyle='--',marker='.',color='b',label='Model median')
    ax2.fill_betweenx(data['z [m]'].to_numpy(),data['T_u_corrected_q0.25 [-]'], data['T_u_corrected_q0.75 [-]'], color='b', alpha=0.3, label='Model IQR')
    # ax2.fill_betweenx(data['z [m]'].to_numpy(),data['T_u_corrected_q0.1 [-]'], data['T_u_corrected_q0.9 [-]'], color='b', alpha=0.1, label='80%-CI')
    ax2.fill_betweenx(data['z [m]'].to_numpy(),data['T_u_corrected_q0.05 [-]'], data['T_u_corrected_q0.95 [-]'], color='b', alpha=0.1, label='Model 90%-CI')
    # ax2.fill_betweenx(data['z [m]'].to_numpy(),data['T_u_corrected_q0.025 [-]'], data['T_u_corrected_q0.975 [-]'], color='b', alpha=0.1, label='95%-CI')
    ax2.set_xlabel('$\mathrm{T}_{u,x}$ [m/s]')
    ax2.set_ylabel('$z$ [m]')
    ax2.set_xlim([0,1.1*max(np.nanmax(data['T_u [-]'].to_numpy()),np.nanmax(data['T_u_corrected_q0.9 [-]'].to_numpy()))])
    ax2.set_ylim([0,1.1*np.nanmax(data['z [m]'].to_numpy())])
    ax2.text(0.05*ax2.get_xlim()[1],0.05*ax2.get_ylim()[1],'(b)')
    # Adjust layout to prevent overlap
    plt.tight_layout()
    ax2.legend()
    ax1.grid(True)
    ax2.grid(True)
    fig.savefig(f'{directory}/profiles_uq.png')
    fig.savefig(f'{directory}/profiles_uq.pdf')
    fig.savefig(f'{directory}/profiles_uq.svg')
    plt.close()

def main(path_to_file):
    # The expected column names
    expected_data_columns = ["z [m]","u [m/s]", "u_rms [m/s]", "c [-]", "d_32a [m]",
                            "u_corrected_q0.025 [m/s]",
                            "u_corrected_q0.05 [m/s]",
                            "u_corrected_q0.1 [m/s]",
                            "u_corrected_q0.25 [m/s]",
                            "u_corrected_q0.5 [m/s]",
                            "u_corrected_q0.75 [m/s]",
                            "u_corrected_q0.9 [m/s]",
                            "u_corrected_q0.95 [m/s]",
                            "u_corrected_q0.975 [m/s]",
                            "T_u_corrected_q0.025 [-]",
                            "T_u_corrected_q0.05 [-]",
                            "T_u_corrected_q0.1 [-]",
                            "T_u_corrected_q0.25 [-]",
                            "T_u_corrected_q0.5 [-]",
                            "T_u_corrected_q0.75 [-]",
                            "T_u_corrected_q0.9 [-]",
                            "T_u_corrected_q0.95 [-]",
                            "T_u_corrected_q0.975 [-]"]

    # get the directory
    path_to_file = os.path.abspath(path_to_file)
    directory = os.path.dirname(path_to_file)

    # Load the CSV file
    data = pd.read_csv(path_to_file)

    # Check if all expected columns are present in the loaded data
    for col in expected_data_columns:
        if col not in data.columns:
            print(f"Error: Column '{col}' not found in the loaded data.")
            return

    plot_profiles(data, directory)


if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process phase-detection probe data from a CSV file")
    parser.add_argument("path_to_file", type=str, help="Path to the CSV file containing the original and corrected data.")

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.path_to_file)
