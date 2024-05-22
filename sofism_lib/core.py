import numpy as np
from joblib import Parallel, delayed
import time
from scipy.ndimage import shift
import zarr
import dask.array as da
from dask import delayed as da_delayed,compute as da_compute 

def corr_numpy(v1, v2, tau_range):
    """
    Computes correlation between trace v1 and v2
    Returns array of values for all possible shifts in range(tauRange)
    """
    sig1 = v1 - v1.mean()
    sig2 = v2[:-tau_range] - v2[:-tau_range].mean()
    return np.correlate(sig1, sig2, mode='valid')


def corr_numpy_no_mean(v1, v2, tau_range):
    """
    Computes correlation between trace v1 and v2
    Returns array of values for all possible shifts in range(tauRange)
    """
    sig1 = v1
    sig2 = v2[:-tau_range]
    return np.correlate(sig1, sig2, mode='valid')


def corr_numpy_img(v1, v2, tau_range):
    """
    Computes correlation between trace v1 and v2
    Returns array of values for all possible shifts in range(tauRange)
    """
    sig1 = v1 - v1.mean()
    sig2 = v2[:-tau_range] - v2[:-tau_range].mean()
    return np.correlate(sig1, sig2, mode='valid')[1:tau_range].sum()


def process_single_x_img(data, tau_range, px=0):
    """
    Parameters
    ----------
    data : 4D numpy array
        binned data of dims: detector x y time.
    tau_range : integer
        range of delays [bins] (caution: delay 0 contains cross-talk).
    px : int, optional
        X coordinate to be given in a loop. The default is 0.

    Returns
    -------
    res : 4D array
        dimensions: det1, det2, y coordinate, delay.
    """
    d, _, y, _ = data.shape
    res = np.zeros((d, d, y))
    for i in range(y):
        for j in range(d):
            v1 = data[j, px, i]
            for k in range(d):
                v2 = data[k, px, i]
                res[j, k, i] = corr_numpy_img(v1, v2, tau_range)
    return res



def process_single_x(data, tau_range, px=0, no_mean=False):
    """
    Parameters
    ----------
    data : 4D numpy array
        binned data of dims: detector x y time.
    tau_range : integer
        range of delays [bins] (caution: delay 0 contains cross-talk).
    px : int, optional
        X coordinate to be given in a loop. The default is 0.

    Returns
    -------
    res : 4D array
        dimensions: det1, det2, y coordinate, delay.
    """
    d, _, y, _ = data.shape
    res = np.zeros((d, d, y, tau_range + 1), dtype=np.float32)  # Specify dtype as float32
    for i in range(y):
        for j in range(d):
            v1 = data[j, px, i].astype(np.float32)  # Convert to float32 if not already
            for k in range(d):
                v2 = data[k, px, i].astype(np.float32)  # Convert to float32 if not already
                if no_mean:
                    # Ensure that corr_numpy_no_mean returns float32 array
                    res[j, k, i] = corr_numpy_no_mean(v1, v2, tau_range)
                else:
                    # Ensure that corr_numpy returns float32 array
                    res[j, k, i] = corr_numpy(v1, v2, tau_range)
    return res

def get_corrs(data, tau_range, no_mean):
    """
    Returns 5D array of cross-correlations using Dask
    dimensions: det1, det2, x, y coordinate, delay.
    """
    t1 = time.time()
    d, xs, y, _ = data.shape
    output_zarr_file = 'correlation_results.zarr'

    # Initialize a Zarr array
    zarr_res = zarr.open(output_zarr_file, mode='w', shape=(xs, d, d, y, tau_range + 1), dtype=np.float32)

    # Define a delayed function for processing
    @da_delayed
    def process_and_store(x):
        result = process_single_x(data, tau_range, x, no_mean=no_mean)
        return result

    # Generate delayed tasks
    tasks = [process_and_store(X) for X in range(xs)]

    # Compute all tasks in parallel and store results
    results = da_compute(*tasks)
    for X, result in enumerate(results):
        zarr_res[X] = result

    # Create a Dask array from the Zarr file
    dask_res = da.from_zarr(output_zarr_file)

    # Reorder the axes using Dask
    dask_res = dask_res.rechunk('auto').swapaxes(0, 2)

    t2 = time.time()
    print(f"Correlation computing with Dask took {(t2 - t1):.2f}s")

    return dask_res



def get_corrs_img(data, tau_range):
    """Returns 5D array of cross-correlations
        dimensions: det1, det2, x, y coordinate, delay.
    """
    t1 = time.time()

    _, xs, _, _ = data.shape
    results = Parallel(n_jobs=-2)(delayed(process_single_x_img)(data, tau_range, X) for X in range(xs))
    t2 = time.time()
    print(f"Correlation computing with parallel processing took {(t2 - t1):.2f}s")
    res = np.array(results).swapaxes(0, 1).swapaxes(1, 2)

    return res


def dask_shift(image, shift):
    # Define a custom function for shifting that is compatible with Dask
    def shift_chunk(chunk, block_info=None):
        from scipy.ndimage import shift as nd_shift
        return nd_shift(chunk, shift=shift, mode='nearest')

    return image.map_overlap(shift_chunk, depth={0: 1, 1: 1, 2: 0}, boundary='nearest')

def sofism(data, corr_data, shift_arr, dx=0.05):
    """
    [existing docstring]
    """
    d, x, y, _ = data.shape

    clsm = np.zeros((x, y))
    ism = np.zeros((x, y))

    shifts = np.zeros((d, 2))
    corr_shifts = np.zeros((d, d, 2))
    for i in range(d):
        shifts[i, 0] = (shift_arr[11, 0] - shift_arr[i, 0]) * 0.05 / dx
        shifts[i, 1] = (shift_arr[11, 1] - shift_arr[i, 1]) * 0.05 / dx
    for i in range(d):
        if i == 0:
            pass
        else:
            s1, s2 = shifts[i]
            clsm += data[i, :, :, :].sum(axis=-1)
            ism += shift(data[i, :, :, :].sum(axis=-1), (s1, s2))

    if corr_data is None:
        return {"clsm": clsm, "ism": ism}
    else:
        tau_range = corr_data.shape[-1]
        sofi_clsm = da.zeros((x, y, tau_range))
        sofism_all = da.zeros((x, y, tau_range))
        sofism_xc = da.zeros((x, y, tau_range))

    for i in range(d):
        for j in range(d):
            corr_shifts[i, j, 0] = (shifts[i, 0] + shifts[j, 0]) / 2
            corr_shifts[i, j, 1] = (shifts[i, 1] + shifts[j, 1]) / 2

    for i in range(d):
        for j in range(d):
            if i == 0 or j == 0:
                pass
            else:
                s1, s2 = corr_shifts[i, j]
                shifted_corr_data = dask_shift(corr_data[i, j, :, :, :], (s1, s2, 0))
                sofism_all += shifted_corr_data
                sofi_clsm += corr_data[i, j, :, :, :]
                if i != j:
                    sofism_xc += shifted_corr_data

    res = {"clsm": clsm, "ism": ism, "sofi_clsm": sofi_clsm, "sofism_xc": sofism_xc, "sofism_all": sofism_all}
    return res

