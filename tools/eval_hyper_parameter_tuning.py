#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import time
import pathlib
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
    Flows (https://gitlab.ethz.ch/vaw/public/pdp-sim.git).


    Copyright (c) 2024 ETH Zurich, Matthias Bürgler, Daniel Valero, Benjamin Hohermuth,
    Robert M. Boes, David F. Vetsch; Laboratory of Hydraulics, Hydrology
    and Glaciology (VAW); Chair of hydraulic structures

"""


Reds = matplotlib.colormaps.get_cmap('Reds')
Blues = matplotlib.colormaps.get_cmap('Blues')

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.sans-serif'] = ['Computer Modern Roman']
plt.rcParams['font.size'] = 9

plt.rcParams['mathtext.fontset'] = 'cm'

plt.rcParams['figure.titlesize'] = 7

plt.rcParams['lines.linewidth'] = 1
plt.rcParams['lines.markersize'] = 2
plt.rcParams['lines.markeredgewidth'] = 0.7

plt.rcParams["legend.frameon"] = False
plt.rcParams['legend.fontsize'] = 6
plt.rcParams["legend.handletextpad"] = 0.2

plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.labelpad'] = 3

plt.rcParams['xtick.major.pad']='4'
plt.rcParams['xtick.direction']='in'

plt.rcParams['ytick.major.pad']='4'
plt.rcParams['ytick.direction']='in'
plt.rcParams["errorbar.capsize"] = 2
plt.rcParams["axes.grid"]=False
plt.rcParams["grid.linestyle"]=":"


def main():

    path = pathlib.Path('../docs/hyperparameter_tuning')
    parameters = ['n_estimators','max_depth','max_features']
    target_names = ['u_x_real [m/s]', 'T_ux_real [-]']
    results = pd.DataFrame(columns=['target_name','rmse_test','rmse_train','rmse_awcc_test','rmse_awcc_train','n_estimators','max_depth','max_features','time'])

    results = pd.read_csv(path / 'hyperparameter_tuning_results.csv')

    max_features_default = 1.0
    n_estimators_default = 100


    target_name = 'u_x_real [m/s]'

    results_tmp1 = results[(results['target_name'] == target_name)]
    default = results_tmp1[(results_tmp1['max_depth'].isna()) & (results_tmp1['max_features'] == max_features_default) & (results_tmp1['n_estimators'] == n_estimators_default)].reset_index(drop=True)
    default = default.iloc[0]

    parameter = 'n_estimators'
    results_tmp2 = results_tmp1[(results_tmp1['max_depth'].isna()) & (results_tmp1['max_features'] == max_features_default)]
    results_tmp2 = results_tmp2.sort_values([parameter])

    x_min_log = 10 ** np.floor(np.log10(np.nanmin(results_tmp2[parameter])))
    x_max_log = 10 ** np.ceil(np.log10(np.nanmax(results_tmp2[parameter])))
    y_min_log = 10 ** np.floor(np.log10(np.nanmin(np.nanmin(results_tmp2[results_tmp2[['rmse_test','rmse_train','rmse_awcc_test','rmse_awcc_train']]>0.0][['rmse_test','rmse_train','rmse_awcc_test','rmse_awcc_train']]))))
    y_max_log = 10 ** np.ceil(np.log10(np.nanmax(np.nanmax(results_tmp2[['rmse_test','rmse_train','rmse_awcc_test','rmse_awcc_train']]))))
    
    f1 = plt.figure(figsize=(3.5,2.5))        
    a1 = f1.gca()
    a1.axhline(default['rmse_train'],color=Reds(1.0),linestyle='--',label='default train')
    a1.axhline(default['rmse_test'],color=Blues(1.0),linestyle='--',label='default test')
    a1.plot(results_tmp2[parameter],results_tmp2['rmse_train'],marker='o',linestyle='-',color=Reds(0.7),label=f'train')
    a1.plot(results_tmp2[parameter],results_tmp2['rmse_test'],marker='o',linestyle='-',color=Blues(0.7),label=f'test')
    a1.axhline(default['rmse_awcc_train'],color=Reds(0.5),linestyle='-.',label='AWCC train')
    a1.axhline(default['rmse_awcc_test'],color=Blues(0.5),linestyle=':',label='AWCC test')
    a1.set_xlabel(f'{parameter} [-]')
    a1.set_ylabel(r'RMSE [m/s]')
    plt.xlim(x_min_log, x_max_log)
    plt.ylim(y_min_log, y_max_log)
    a1.set_xscale('log')
    a1.set_yscale('log')
    a1.legend(labelspacing=0.2,fontsize=6)
    f1.tight_layout()
    a1.grid(which='major', axis='both')
    f1.savefig(path / f'hyperparameter_tuning_{target_name.split()[0]}_{parameter}.jpg',dpi=1200)


    parameter = 'max_depth'
    results_tmp2 = results_tmp1[(results_tmp1['n_estimators'] == n_estimators_default) & (results_tmp1['max_features'] == max_features_default)]
    results_tmp2 = results_tmp2.sort_values([parameter])

    x_min_log = 10 ** np.floor(np.log10(np.nanmin(results_tmp2[parameter])))
    x_max_log = 10 ** np.ceil(np.log10(np.nanmax(results_tmp2[parameter])))
    y_min_log = 10 ** np.floor(np.log10(np.nanmin(np.nanmin(results_tmp2[results_tmp2[['rmse_test','rmse_train','rmse_awcc_test','rmse_awcc_train']]>0.0][['rmse_test','rmse_train','rmse_awcc_test','rmse_awcc_train']]))))
    y_max_log = 10 ** np.ceil(np.log10(np.nanmax(np.nanmax(results_tmp2[['rmse_test','rmse_train','rmse_awcc_test','rmse_awcc_train']]))))
    f1 = plt.figure(figsize=(3.5,2.5))        
    a1 = f1.gca()
    a1.axhline(default['rmse_train'],color=Reds(1.0),linestyle='--',label='default train')
    a1.axhline(default['rmse_test'],color=Blues(1.0),linestyle='--',label='default test')
    a1.plot(results_tmp2[parameter],results_tmp2['rmse_train'],marker='o',linestyle='-',color=Reds(0.7),label=f'RMSE train')
    a1.plot(results_tmp2[parameter],results_tmp2['rmse_test'],marker='o',linestyle='-',color=Blues(0.7),label=f'RMSE test')
    a1.axhline(default['rmse_awcc_train'],color=Reds(0.5),linestyle='-.',label='RMSE AWCC train')
    a1.axhline(default['rmse_awcc_test'],color=Blues(0.5),linestyle='-.',label='RMSE AWCC test')
    a1.set_xlabel(f'{parameter} [-]')
    a1.set_ylabel(r'RMSE [m/s]')
    plt.xlim(x_min_log, x_max_log)
    plt.ylim(y_min_log, y_max_log)
    a1.set_xscale('log')
    a1.set_yscale('log')
    a1.legend(labelspacing=0.2,fontsize=6)
    f1.tight_layout()
    a1.grid(which='major', axis='both')
    f1.savefig(path / f'hyperparameter_tuning_{target_name.split()[0]}_{parameter}.jpg',dpi=1200)

    parameter = 'max_features'
    results_tmp2 = results_tmp1[(results_tmp1['n_estimators'] == n_estimators_default) & (results_tmp1['max_depth'].isna())]
    results_tmp2 = results_tmp2.sort_values([parameter])
    x_min_log = 10 ** np.floor(np.log10(np.nanmin(results_tmp2[parameter])))
    x_max_log = 10 ** np.ceil(np.log10(np.nanmax(results_tmp2[parameter])))
    y_min_log = 10 ** np.floor(np.log10(np.nanmin(np.nanmin(results_tmp2[results_tmp2[['rmse_test','rmse_train','rmse_awcc_test','rmse_awcc_train']]>0.0][['rmse_test','rmse_train','rmse_awcc_test','rmse_awcc_train']]))))
    y_max_log = 10 ** np.ceil(np.log10(np.nanmax(np.nanmax(results_tmp2[['rmse_test','rmse_train','rmse_awcc_test','rmse_awcc_train']]))))
    f1 = plt.figure(figsize=(3.5,2.5))        
    a1 = f1.gca()
    a1.axhline(default['rmse_train'],color=Reds(1.0),linestyle='--',label='default train')
    a1.axhline(default['rmse_test'],color=Blues(1.0),linestyle='--',label='default test')
    a1.plot(results_tmp2[parameter],results_tmp2['rmse_train'],marker='o',linestyle='-',color=Reds(0.7),label=f'RMSE train')
    a1.plot(results_tmp2[parameter],results_tmp2['rmse_test'],marker='o',linestyle='-',color=Blues(0.7),label=f'RMSE test')
    a1.axhline(default['rmse_awcc_train'],color=Reds(0.5),linestyle='-.',label='RMSE AWCC train')
    a1.axhline(default['rmse_awcc_test'],color=Blues(0.5),linestyle='-.',label='RMSE AWCC test')
    a1.set_xlabel(f'{parameter} [-]')
    a1.set_ylabel(r'RMSE [m/s]')
    plt.xlim(x_min_log, x_max_log)
    plt.ylim(y_min_log, y_max_log)
    a1.set_xscale('log')
    a1.set_yscale('log')
    a1.legend(labelspacing=0.2,fontsize=6)
    f1.tight_layout()
    a1.grid(which='major', axis='both')
    f1.savefig(path / f'hyperparameter_tuning_{target_name.split()[0]}_{parameter}.jpg',dpi=1200)

    target_name = 'T_ux_real [-]'

    results_tmp1 = results[(results['target_name'] == target_name)]
    default = results_tmp1[(results_tmp1['max_depth'].isna()) & (results_tmp1['max_features'] == max_features_default) & (results_tmp1['n_estimators'] == n_estimators_default)].reset_index(drop=True)
    default = default.iloc[0]

    parameter = 'n_estimators'
    results_tmp2 = results_tmp1[(results_tmp1['max_depth'].isna()) & (results_tmp1['max_features'] == max_features_default)]
    results_tmp2 = results_tmp2.sort_values([parameter])

    x_min_log = 10 ** np.floor(np.log10(np.nanmin(results_tmp2[parameter])))
    x_max_log = 10 ** np.ceil(np.log10(np.nanmax(results_tmp2[parameter])))
    y_min_log = 10 ** np.floor(np.log10(np.nanmin(np.nanmin(results_tmp2[results_tmp2[['rmse_test','rmse_train','rmse_awcc_test','rmse_awcc_train']]>0.0][['rmse_test','rmse_train','rmse_awcc_test','rmse_awcc_train']]))))
    y_max_log = 10 ** np.ceil(np.log10(np.nanmax(np.nanmax(results_tmp2[['rmse_test','rmse_train','rmse_awcc_test','rmse_awcc_train']]))))
    
    f1 = plt.figure(figsize=(3.5,2.5))        
    a1 = f1.gca()
    a1.axhline(default['rmse_train'],color=Reds(1.0),linestyle='--',label='default train')
    a1.axhline(default['rmse_test'],color=Blues(1.0),linestyle='--',label='default test')
    a1.plot(results_tmp2[parameter],results_tmp2['rmse_train'],marker='o',linestyle='-',color=Reds(0.7),label=f'train')
    a1.plot(results_tmp2[parameter],results_tmp2['rmse_test'],marker='o',linestyle='-',color=Blues(0.7),label=f'test')
    a1.axhline(default['rmse_awcc_train'],color=Reds(0.5),linestyle='-.',label='AWCC train')
    a1.axhline(default['rmse_awcc_test'],color=Blues(0.5),linestyle=':',label='AWCC test')
    a1.set_xlabel(f'{parameter} [-]')
    a1.set_ylabel(r'RMSE [m/s]')
    plt.xlim(x_min_log, x_max_log)
    plt.ylim(y_min_log, y_max_log)
    a1.set_xscale('log')
    a1.set_yscale('log')
    a1.legend(labelspacing=0.2,fontsize=6)
    f1.tight_layout()
    a1.grid(which='major', axis='both')
    f1.savefig(path / f'hyperparameter_tuning_{target_name.split()[0]}_{parameter}.jpg',dpi=1200)


    parameter = 'max_depth'
    results_tmp2 = results_tmp1[(results_tmp1['n_estimators'] == n_estimators_default) & (results_tmp1['max_features'] == max_features_default)]
    results_tmp2 = results_tmp2.sort_values([parameter])

    x_min_log = 10 ** np.floor(np.log10(np.nanmin(results_tmp2[parameter])))
    x_max_log = 10 ** np.ceil(np.log10(np.nanmax(results_tmp2[parameter])))
    y_min_log = 10 ** np.floor(np.log10(np.nanmin(np.nanmin(results_tmp2[results_tmp2[['rmse_test','rmse_train','rmse_awcc_test','rmse_awcc_train']]>0.0][['rmse_test','rmse_train','rmse_awcc_test','rmse_awcc_train']]))))
    y_max_log = 10 ** np.ceil(np.log10(np.nanmax(np.nanmax(results_tmp2[['rmse_test','rmse_train','rmse_awcc_test','rmse_awcc_train']]))))
    f1 = plt.figure(figsize=(3.5,2.5))        
    a1 = f1.gca()
    a1.axhline(default['rmse_train'],color=Reds(1.0),linestyle='--',label='default train')
    a1.axhline(default['rmse_test'],color=Blues(1.0),linestyle='--',label='default test')
    a1.plot(results_tmp2[parameter],results_tmp2['rmse_train'],marker='o',linestyle='-',color=Reds(0.7),label=f'RMSE train')
    a1.plot(results_tmp2[parameter],results_tmp2['rmse_test'],marker='o',linestyle='-',color=Blues(0.7),label=f'RMSE test')
    a1.axhline(default['rmse_awcc_train'],color=Reds(0.5),linestyle='-.',label='RMSE AWCC train')
    a1.axhline(default['rmse_awcc_test'],color=Blues(0.5),linestyle='-.',label='RMSE AWCC test')
    a1.set_xlabel(f'{parameter} [-]')
    a1.set_ylabel(r'RMSE [m/s]')
    plt.xlim(x_min_log, x_max_log)
    plt.ylim(y_min_log, y_max_log)
    a1.set_xscale('log')
    a1.set_yscale('log')
    a1.legend(labelspacing=0.2,fontsize=6)
    f1.tight_layout()
    a1.grid(which='major', axis='both')
    f1.savefig(path / f'hyperparameter_tuning_{target_name.split()[0]}_{parameter}.jpg',dpi=1200)

    parameter = 'max_features'
    results_tmp2 = results_tmp1[(results_tmp1['n_estimators'] == n_estimators_default) & (results_tmp1['max_depth'].isna())]
    results_tmp2 = results_tmp2.sort_values([parameter])
    x_min_log = 10 ** np.floor(np.log10(np.nanmin(results_tmp2[parameter])))
    x_max_log = 10 ** np.ceil(np.log10(np.nanmax(results_tmp2[parameter])))
    y_min_log = 10 ** np.floor(np.log10(np.nanmin(np.nanmin(results_tmp2[results_tmp2[['rmse_test','rmse_train','rmse_awcc_test','rmse_awcc_train']]>0.0][['rmse_test','rmse_train','rmse_awcc_test','rmse_awcc_train']]))))
    y_max_log = 10 ** np.ceil(np.log10(np.nanmax(np.nanmax(results_tmp2[['rmse_test','rmse_train','rmse_awcc_test','rmse_awcc_train']]))))
    f1 = plt.figure(figsize=(3.5,2.5))        
    a1 = f1.gca()
    a1.axhline(default['rmse_train'],color=Reds(1.0),linestyle='--',label='default train')
    a1.axhline(default['rmse_test'],color=Blues(1.0),linestyle='--',label='default test')
    a1.plot(results_tmp2[parameter],results_tmp2['rmse_train'],marker='o',linestyle='-',color=Reds(0.7),label=f'RMSE train')
    a1.plot(results_tmp2[parameter],results_tmp2['rmse_test'],marker='o',linestyle='-',color=Blues(0.7),label=f'RMSE test')
    a1.axhline(default['rmse_awcc_train'],color=Reds(0.5),linestyle='-.',label='RMSE AWCC train')
    a1.axhline(default['rmse_awcc_test'],color=Blues(0.5),linestyle='-.',label='RMSE AWCC test')
    a1.set_xlabel(f'{parameter} [-]')
    a1.set_ylabel(r'RMSE [m/s]')
    plt.xlim(x_min_log, x_max_log)
    plt.ylim(y_min_log, y_max_log)
    a1.set_xscale('log')
    a1.set_yscale('log')
    a1.legend(labelspacing=0.2,fontsize=6)
    f1.tight_layout()
    a1.grid(which='major', axis='both')
    f1.savefig(path / f'hyperparameter_tuning_{target_name.split()[0]}_{parameter}.jpg',dpi=1200)


if __name__ == "__main__":
    main()