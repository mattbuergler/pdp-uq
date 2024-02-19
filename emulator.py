import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
import sys
import argparse
import numpy as np
import joblib
import pandas as pd 
import matplotlib.pyplot as plt


def plot_profiles(data, directory):
    # Set plt parameters
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.sans-serif'] = ['Computer Modern Roman']
    plt.rcParams['font.size'] = 9
    plt.rcParams['mathtext.fontset'] = 'cm'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.4))
    ax1.plot(data['u [m/s]'].to_numpy(),data['z [m]'].to_numpy(),fillstyle='none',linestyle='-',marker='o',color='k',label='awcc')
    ax1.plot(data['u_corrected_q0.5 [m/s]'].to_numpy(),data['z [m]'].to_numpy(),fillstyle='none',linestyle='--',marker='o',color='b',label='corr. (50%-tile)')
    ax1.fill_betweenx(data['z [m]'].to_numpy(),data['u_corrected_q0.25 [m/s]'], data['u_corrected_q0.75 [m/s]'], color='b', alpha=0.3, label='25%-/75%-tile')
    ax1.fill_betweenx(data['z [m]'].to_numpy(),data['u_corrected_q0.1 [m/s]'], data['u_corrected_q0.9 [m/s]'], color='b', alpha=0.1, label='10%-/90%-tile')
    ax1.set_xlabel('$\overline{u}_x$ [m/s]')
    ax1.set_ylabel('$z$ [m]')
    ax1.set_xlim([0,1.1*max(np.nanmax(data['u [m/s]'].to_numpy()),np.nanmax(data['u_corrected_q0.9 [m/s]'].to_numpy()))])
    ax1.set_ylim([0,1.1*np.nanmax(data['z [m]'].to_numpy())])
    ax1.text(0.05*ax1.get_xlim()[1],0.05*ax1.get_ylim()[1],'(a)')
    ax2.plot(data['T_u [-]'].to_numpy(),data['z [m]'].to_numpy(),fillstyle='none',linestyle='-',marker='o',color='k',label='awcc')
    ax2.plot(data['T_u_corrected_q0.5 [-]'].to_numpy(),data['z [m]'].to_numpy(),fillstyle='none',linestyle='--',marker='o',color='b',label='corr. (50%-tile)')
    ax2.fill_betweenx(data['z [m]'].to_numpy(),data['T_u_corrected_q0.25 [-]'], data['T_u_corrected_q0.75 [-]'], color='b', alpha=0.3, label='25%-/75%-tile')
    ax2.fill_betweenx(data['z [m]'].to_numpy(),data['T_u_corrected_q0.1 [-]'], data['T_u_corrected_q0.9 [-]'], color='b', alpha=0.1, label='10%-/90%-tile')
    ax2.set_xlabel('$\mathrm{T}_{u,x}$ [m/s]')
    ax2.set_ylabel('$z$ [m]')
    ax2.set_xlim([0,1.1*max(np.nanmax(data['T_u [-]'].to_numpy()),np.nanmax(data['T_u_corrected_q0.9 [-]'].to_numpy()))])
    ax2.set_ylim([0,1.1*np.nanmax(data['z [m]'].to_numpy())])
    ax2.text(0.05*ax2.get_xlim()[1],0.05*ax2.get_ylim()[1],'(b)')
    # Adjust layout to prevent overlap
    plt.tight_layout()
    ax2.legend()
    fig.savefig(f'{directory}/profiles_uq.pdf')
    fig.savefig(f'{directory}/profiles_uq.png')
    fig.savefig(f'{directory}/profiles_uq.svg')

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

def main(dx, dy, Np, plot, path_to_file):
    # Define the quantiles
    quantiles = [0.1,0.25,0.5,0.75,0.9]
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

    # Load the trained quantile random forest to predict corrected mean velocities and uncertainties
    qrf_u = joblib.load("data/random_forest_quantile_u_mean_all.joblib")
    pred_u = qrf_u.predict(data[rf_feature_columns].to_numpy(), quantiles=quantiles)

    # Load the trained quantile random forest to predict corrected turbulence intensities and uncertainties
    qrf_T_u = joblib.load("data/random_forest_quantile_T_u_all.joblib")
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
    print(f'Warning: {n_out_of_range} out of {n_total} contained values outside the range of application.\nUncertainty was not quantified for those data points.')
    # Add the predictions for all quantiles to the data
    for ii, quantile in enumerate(quantiles):
        data[f'u_corrected_q{quantile} [m/s]'] = pred_u[:,ii]
        data[f'T_u_corrected_q{quantile} [-]'] = pred_Tu[:,ii]

    if plot:
        plot_profiles(data, directory)
    # Save the data to a new file
    data.to_csv(path_to_file.replace('.csv','_uq.csv'), index=False)
    print('Finished.')

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process phase-detection probe data from a CSV file")

    # Add arguments
    parser.add_argument("-dx", type=float, help="Streamwise tip separation of the phase-detection probe [m]")
    parser.add_argument("-dy", type=float, help="Transverse tip separation of the phase-detection probe [m]")
    parser.add_argument("-Np", type=int, help="Number of particles per averaging window [-]")
    parser.add_argument('-p', '--plot', action='store_true', help='Plot figures')
    parser.add_argument("path_to_file", type=str, help="Path to the CSV data file")

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.dx, args.dy, args.Np, args.plot, args.path_to_file)
