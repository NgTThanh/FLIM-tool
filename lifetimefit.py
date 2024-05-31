# Description: This script contains functions for fitting the decay curve of each pixel in the FLIM data and visualizing the lifetime map.
import matplotlib.pyplot as plt
import numpy as np
from flimlib import GCI_triple_integral_fitting_engine, GCI_Phasor, GCI_marquardt_fitting_engine, GCI_multiexp_tau
from math import isnan
from joblib import Parallel, delayed 

    
def lifetime_fit(data, bin_width):

    # Parallel computation
    result = np.array(Parallel(n_jobs=-2)(delayed(phasor)(data, bin_width, i, j) for i in range(data.shape[0]) for j in range(data.shape[1]))).reshape(data.shape[0], data.shape[1],3)
    tau = result[:,:,0]
    u = result[:,:,1]
    v = result[:,:,2]
    # Reshape the result to match the original lifetime_map shape
    lifetime_map = np.array(tau).reshape(data.shape[0], data.shape[1])
    u_map = np.array(u).reshape(data.shape[0], data.shape[1]).flatten()
    v_map = np.array(v).reshape(data.shape[0], data.shape[1]).flatten()
    return lifetime_map, u_map, v_map

def fit_curve(data, bin_width, i, j):

    photon_count = data[i, j, :]
    # Run the triple integral fitting engine
    estimated_fit_result = GCI_triple_integral_fitting_engine(
        period=bin_width,
        photon_count=photon_count,
        compute_fitted=True,  # Set to True for the fitted curve
        compute_residuals=False,
        compute_chisq=False
    )
    params = np.array([estimated_fit_result.Z, estimated_fit_result.A, 
                       estimated_fit_result.tau])
    
    if isnan(estimated_fit_result.tau) == True:
        estimated_fit_result = GCI_Phasor(
        period=bin_width,
        photon_count=photon_count,
        compute_fitted=True,  # Set to True for the fitted curve
        compute_residuals=False,
        compute_chisq=False
        )
        params = np.array([0, estimated_fit_result.u, estimated_fit_result.tau])

     

    fit_result = GCI_marquardt_fitting_engine(
        period=bin_width,
        photon_count=photon_count,
        param=params,
        fitfunc= GCI_multiexp_tau,
        compute_fitted=True,  # Set to True for the fitted curve
        compute_residuals=False,
        compute_chisq=True
    )

    if isnan(fit_result.param[2]) == True:
        print('The fitting result is nan')
        return None
    else:
        fig = plt.figure()
        plt.plot(np.arange(data.shape[-1])*bin_width, fit_result.fitted, label='X: {}, Y: {}, $\\tau$: {}'.format(i, j, fit_result.param[2]))
        plt.xlabel('Time')
        plt.ylabel('Photon counts')
        plt.legend()
        return fig
    
def phasor(data, bin_width, i, j):
    photon_count = data[i, j, :]
    fit_result = GCI_Phasor(
        period=bin_width,
        photon_count=photon_count,
        compute_fitted=False,  # Set to True for the fitted curve
        compute_residuals=False,
        compute_chisq=False
    )

    if isnan(fit_result.tau.mean()) == True:
        alternative_fit_result = GCI_triple_integral_fitting_engine(
            period=bin_width,
            photon_count=photon_count,
            compute_fitted=False,  # Set to True for the fitted curve
            compute_residuals=False,
            compute_chisq=False
        )
        if isnan(alternative_fit_result.tau) == True:
            t = 0
        else:
            t = alternative_fit_result.tau
    else:
        t = fit_result.tau.mean()

    if isnan(fit_result.u.mean()) == True:
        u = 0
    else:
        u = fit_result.u

    if isnan(fit_result.v.mean()) == True:
        v = 0
    else:
        v = fit_result.v
    mag = np.sqrt(u**2 + v**2)
    u = (u/mag).mean()
    v = (v/mag).mean()
    return t, u, v

def visu(data, bin_width):
    lifetime_map, u_map, v_map = lifetime_fit(data, bin_width)
    # Visualize the lifetime map
    fig= plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(2, 2)
    ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
    im1 = ax[0].imshow(data.sum(axis=-1), cmap='inferno')
    ax[0].set_title('Intensity map')
    fig.colorbar(im1, ax=ax[0], label='#photon counts')
    im2 = ax[1].imshow(lifetime_map, cmap='viridis')
    ax[1].set_title('Lifetime map')
    fig.colorbar(im2, ax=ax[1], label='ns')
    im3 = ax[2].scatter(u_map, v_map, c=lifetime_map.flatten(), cmap='viridis', marker='.')
    t=np.linspace(0, np.pi, 100)
    im3 = plt.plot(0.5 + 0.5 * np.cos(t), 0.5 * np.sin(t), 'r-', label='Universal Circle')
    ax[2].set_xlim(0, 1.1)                             # Adjust x-axis limits
    ax[2].set_ylim(0, 0.6)                             # Adjust y-axis limits
    ax[2].set_xlabel('Y Phasor Coordinate')
    ax[2].yaxis.tick_right()
    ax[2].yaxis.set_label_position("right")
    ax[2].set_ylabel("Your Label")
    ax[2].set_ylabel('X Phasor Coordinate')
    ax[2].set_title('Phasor plot')
    plt.show()
    return fig
