# Description: This script contains functions for fitting the decay curve of each pixel in the FLIM data and visualizing the lifetime map.
import matplotlib.pyplot as plt
import numpy as np
from flimlib import GCI_triple_integral_fitting_engine, GCI_Phasor
from math import isnan
from joblib import Parallel, delayed 



def fit_pixel_decay(data, bin_width, i,j):
    photon_count = data[i, j, :]

    # Run the triple integral fitting engine
    fit_result = GCI_triple_integral_fitting_engine(
        period=bin_width,
        photon_count=photon_count,
        compute_fitted=True,  # Set to True for the fitted curve
        compute_residuals=False,
        compute_chisq=False
    ).tau.mean()
    if isnan(fit_result) == True:
        return 0
    else:
        return fit_result
    
def lifetime_fit(data, bin_width):
    # Parallel computation
    result = Parallel(n_jobs=-2)(delayed(fit_pixel_decay)(data, bin_width, i, j) for i in range(data.shape[0]) for j in range(data.shape[1]))
    # Reshape the result to match the original lifetime_map shape
    lifetime_map = np.array(result).reshape(data.shape[0], data.shape[1])
    return lifetime_map

def fit_curve(data, bin_width, i, j):
    # Run the triple integral fitting engine
    fit_result = GCI_triple_integral_fitting_engine(
        period=bin_width,
        photon_count=data[i, j, :],
        compute_fitted=True,
        compute_residuals=False,
        compute_chisq=False
    )
    if isnan(fit_result.tau.mean()) == True:
        print('The fitting result is nan')
        return None
    else:
        fig = plt.figure()
        plt.plot(np.arange(data.shape[-1])*bin_width, fit_result.fitted, label='X: {}, Y: {}, $\\tau$: {}'.format(i, j, fit_result.tau.mean()))
        plt.xlabel('Time {}ns'.format(bin_width))
        plt.ylabel('Photon counts')
        plt.legend()
        return fig

def phasor(data, bin_width, i,j):
    photon_count = data[i, j, :]

    # Run the phasor fitting engine
    fit_result = GCI_Phasor(
        period=bin_width,
        photon_count=photon_count,
        compute_fitted=True,  # Set to True for the fitted curve
        compute_residuals=False,
        compute_chisq=False
    ).tau.mean()
    if isnan(fit_result) == True:
        return 0
    else:
        return fit_result
    
def phasor_map(data, bin_width):
    # Parallel computation
    result = Parallel(n_jobs=-2)(delayed(phasor)(data, bin_width, i, j) for i in range(data.shape[0]) for j in range(data.shape[1]))
    # Reshape the result to match the original lifetime_map shape
    lifetime_map = np.array(result).reshape(data.shape[0], data.shape[1])
    return lifetime_map

def phasor_curve(data, bin_width, i, j):
    # Run the phasor fitting engine
    fit_result = GCI_Phasor(
        period=bin_width,
        photon_count=data[i, j, :],
        compute_fitted=True,
        compute_residuals=False,
        compute_chisq=False
    )
    if isnan(fit_result.tau.mean()) == True:
        print('The fitting result is nan')
        return None
    else:
        fig = plt.figure()
        plt.plot(np.arange(data.shape[-1])*bin_width, fit_result.fitted, label='X: {}, Y: {}, $\\tau$: {}'.format(i, j, fit_result.tau.mean()))
        plt.xlabel('Time {}ns'.format(bin_width))
        plt.ylabel('Photon counts')
        plt.legend()
        return fig

def visu(data, lifetime_map):
    # Visualize the lifetime map
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    im1 = ax[0].imshow(data.sum(axis=-1), cmap='inferno')
    ax[0].set_title('Intensity map')
    fig.colorbar(im1, ax=ax[0], label='#photon counts')
    im2 = ax[1].imshow(lifetime_map, cmap='viridis')
    ax[1].set_title('Lifetime map')
    fig.colorbar(im2, ax=ax[1], label='ns')
    plt.show()
    return fig


