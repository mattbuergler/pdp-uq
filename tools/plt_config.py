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
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.sans-serif'] = ['Computer Modern Roman']
plt.rcParams['font.size'] = 9
plt.rcParams['mathtext.fontset'] = 'cm'

plt.rcParams['figure.titlesize'] = 7

plt.rcParams['lines.linewidth'] = 1
plt.rcParams['lines.markersize'] = 2
plt.rcParams['lines.markeredgewidth'] = 0.7

plt.rcParams["legend.frameon"] = False
plt.rcParams['legend.fontsize'] = 8
plt.rcParams["legend.handletextpad"] = 0.2

plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.labelpad'] = 3

plt.rcParams['xtick.major.pad']='4'
plt.rcParams['xtick.direction']='in'

plt.rcParams['ytick.major.pad']='4'
plt.rcParams['ytick.direction']='in'
