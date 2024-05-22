import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from .params import Parameters
import dask.array as da
import h5py
import gc

def read_single_pixel(mdir, pixel):
    fpath = os.path.join(mdir, 'pix{:02}.txt'.format(pixel))
    data = np.array(pd.read_csv(fpath)).astype(np.int32)
    return data


def get_events(data, event_id):
    events = np.where(data[:, 0] == event_id)
    return events[0]


def split_dwells(data):
    events = get_events(data, event_id=-9)
    return np.split(data, events, axis=0)


def split_lines(data):
    events = get_events(data, event_id=-10)
    return np.split(data, events, axis=0)


def filter_lines(line):
    proper_length = min(len(line[len(line) // 2]), len(line[len(line) // 2 - 1]))
    res = []
    for i in range(proper_length // 2):
        curr_len = len(line[i])
        if curr_len < proper_length:
            continue
        res.append(line[i][curr_len - proper_length:curr_len])
    for i in range(proper_length // 2, len(line), 1):
        curr_len = len(line[i])
        if curr_len < proper_length:
            continue
        res.append(line[i][:proper_length])
    return res


def filter_lines_continuous(line):
    proper_length = min(len(line[len(line) // 2]), len(line[len(line) // 2 - 1]))
    res = []
    for i in range(len(line)):
        curr_len = len(line[i])
        if curr_len < proper_length:
            continue
        if i % 2 == 0:
            res.append(line[i][1:proper_length + 1])
        else:
            res.append(line[i][:proper_length])
    return res


def bin_tags(data, bin_width, dwell_time):
    bw = bin_width * 1e6
    n_bins = int(dwell_time / bin_width)
    curr_tags = np.bincount(np.abs((data // bw)).astype('int32'))
    res = np.zeros(n_bins, dtype='int32')
    res[:curr_tags.shape[0]] = curr_tags[:n_bins]

    return res


def assign_px_time(px_data, clock_tick, first_tag, last_tag):
    non_overflow_mask = np.where(px_data[:, 0] != -56)[0][1:]
    data_corrected = np.array(px_data[non_overflow_mask]) - px_data[0, 1]
    overflow_counter = (non_overflow_mask - np.arange(1, non_overflow_mask.shape[0] + 1, 1))
    data_corrected[:, 1] += overflow_counter * 2 ** 16
    data_corrected[:, 1] = data_corrected[:, 1] * clock_tick
    data_corrected = np.array(data_corrected)
    data_corrected = data_corrected[data_corrected[:, 1] < last_tag]
    data_corrected = data_corrected[data_corrected[:, 1] > first_tag]
    data_corrected[:, 1] -= int(first_tag)

    return data_corrected


def split_bin(px, dwell_time, bin_width, laser_freq, skip_first):
    try:
        x, y = len(px), len(px[0])
    except IndexError:
        print("Something is wrong with SPAD data shape")
        print("Did you disable dummy clock markers?")
        return None
    first_tag = skip_first * 1e6
    last_tag = first_tag + dwell_time * 1e6
    n_bins = int(dwell_time / bin_width)
    tags = np.zeros((x, y, n_bins))
    clock_tick = int(1e3 / laser_freq)
    for i in range(x):
        for j in range(y):
            try:
                curr_px = np.array(px[i][j])
            except IndexError:
                print(f"Error at (i,j)=({i},{j})")
                curr_px = np.zeros((123, 2))
            px_proper = assign_px_time(curr_px, clock_tick, first_tag, last_tag)
            tags[i, j] = bin_tags(px_proper[:, 1], bin_width, dwell_time)
    tags[::2, :] = tags[::2, ::-1]
    return tags


def bin_data_in_interval(data, start, end, num_bins):
    interval_data = data[(data >= start) & (data < end)]
    bins = np.linspace(start, end, num_bins + 1)
    binned_data, _ = np.histogram(interval_data, bins=bins)
    return binned_data

def process_single_line(line, clock_tick, first_tag, last_tag, bin_width, ref_bin_width, dwell_time, n_ref_bins):
    px_proper = assign_px_time(line, clock_tick, first_tag, last_tag)
    num_bins = int(dwell_time / bin_width) if bin_width >= ref_bin_width else n_ref_bins
    tags = np.zeros(num_bins, dtype=np.float32)

    if px_proper.size > 0:
        if bin_width < ref_bin_width:
            # Binning within smaller intervals
            current_bin_width = ref_bin_width * 1e6
            interval_end = first_tag + current_bin_width
            while current_bin_width >= bin_width * 1e6 and interval_end <= last_tag:
                interval_bins = max(1, int(np.ceil(current_bin_width / (bin_width * 1e6))))
                interval_tags = bin_data_in_interval(px_proper[:, 1], first_tag, interval_end, interval_bins)

                if len(interval_tags) < n_ref_bins:
                    interval_tags_padded = np.pad(interval_tags, (0, n_ref_bins - len(interval_tags)), 'constant')
                else:
                    interval_tags_padded = interval_tags[:n_ref_bins]

                tags += interval_tags_padded
                current_bin_width /= 2
                first_tag = interval_end
                interval_end = first_tag + current_bin_width
        else:
            binned_tags = bin_tags(px_proper[:, 1], bin_width, dwell_time)
            tags[:len(binned_tags)] = binned_tags

    return tags


def process_line_group(line_group, clock_tick, first_tag, last_tag, bin_width, ref_bin_width, dwell_time, n_ref_bins):
    # Reduce the number of parallel jobs to manage memory usage
    n_jobs = 4  # Adjust based on your system's memory capacity and CPU cores

    # Parallel processing of each line in the line_group
    tags_list = Parallel(n_jobs=n_jobs)(delayed(process_single_line)(
        line, clock_tick, first_tag, last_tag, bin_width, ref_bin_width, dwell_time, n_ref_bins
    ) for line in line_group)

    return tags_list

def split_unwrap_dwells(meas_file, pixel, bin_width, dwell_time, laser_freq, skip_first):
    ref_bin_width = 0.01  # Reference bin width in ms
    n_ref_bins = int(dwell_time / ref_bin_width)  # Number of bins when bin width is 0.01

    data = read_single_pixel(meas_file, pixel)
    # filter out all events that are not stage triggers, wraparounds or photons
    data = np.array(data[(data[:, 0] > 0) | (data[:, 0] == -9) | (data[:, 0] == -10) | (data[:, 0] == -56)])
    lines = split_lines(data)
    pixels = [split_dwells(line) for line in lines]
    px = filter_lines(pixels)

    clock_tick = int(1e3 / laser_freq)
    first_tag = skip_first * 1e6
    last_tag = first_tag + dwell_time * 1e6
    
    # Use joblib to parallelize the processing of line groups
    tags_list = Parallel(n_jobs=-1)(delayed(process_line_group)(
        line_group, clock_tick, first_tag, last_tag, bin_width, ref_bin_width, dwell_time, n_ref_bins
    ) for line_group in px)
    
    # Convert the list of tags to a 3D numpy array
    max_tags_len = max(len(tags) for line_tags in tags_list for tags in line_tags)
    tags_array = np.zeros((len(tags_list), max(len(group) for group in tags_list), max_tags_len), dtype=np.float32)
    
    for i, line_tags in enumerate(tags_list):
        for j, tags in enumerate(line_tags):
            tags_len = len(tags)
            tags_array[i, j, :tags_len] = tags
    
    # Mirror the tags on even rows
    tags_array[::2, :, :] = tags_array[::2, ::-1, :]

    return tags_array
    
    #tags = split_bin(px, dwell_time, bin_width, laser_freq, skip_first)
    #return tags


def split_dwells_continuous(meas_file, pixel, bin_width, dwell_time, laser_freq, skip_first):
    ref_bin_width = 0.01  # Reference bin width in ms
    n_ref_bins = int(dwell_time / ref_bin_width)  # Number of bins when bin width is 0.01
    
    data = read_single_pixel(meas_file, pixel)
    data = np.array(data[(data[:, 0] > 0) | (data[:, 0] == -9) | (data[:, 0] == -10) | (data[:, 0] == -56)])
    lines = split_lines(data)
    pixels = [split_dwells(line) for line in lines]
    px = filter_lines_continuous(pixels)
    
    clock_tick = int(1e3 / laser_freq)
    first_tag = skip_first * 1e6
    last_tag = first_tag + dwell_time * 1e6
    
    # Use joblib to parallelize the processing of line groups
    tags_list = Parallel(n_jobs=-1)(delayed(process_line_group)(
        line_group, clock_tick, first_tag, last_tag, bin_width, ref_bin_width, dwell_time, n_ref_bins
    ) for line_group in px)

    # Convert the list of tags to a 3D numpy array
    max_tags_len = max(len(tags) for line_tags in tags_list for tags in line_tags)
    tags_array = np.zeros((len(tags_list), max(len(group) for group in tags_list), max_tags_len), dtype=np.float32)
    
    for i, line_tags in enumerate(tags_list):
        for j, tags in enumerate(line_tags):
            tags_len = len(tags)
            tags_array[i, j, :tags_len] = tags
    
    # Mirror the tags on even rows
    tags_array[::2, :, :] = tags_array[::2, ::-1, :]

    return tags_array



def read_tags(meas_file, bin_width, dwell_time, laser_freq, skip_first, num_bins):
    pixels = range(23)
    output_h5_file ='Test.h5'

    # Write data to HDF5 file
    # Create an HDF5 file to store the results
    with h5py.File(output_h5_file, 'w') as h5f:
        # Write data to HDF5 format in chunks
        results = Parallel(n_jobs=-2)(delayed(split_unwrap_dwells)(meas_file, p, bin_width, dwell_time, laser_freq, skip_first) for p in pixels)
        
        for p, result in zip(pixels, results):
            # Store each result in a separate HDF5 dataset
            h5f.create_dataset(f'pixel_{p}', data=result.astype(np.float32), chunks=True)

    return output_h5_file



def read_tags_continuous(meas_file, bin_width, dwell_time, laser_freq, skip_first, num_bins):
    output_h5_file ='Test.h5'
    pixels = range(23)

    # Create an HDF5 file to store the results
    with h5py.File(output_h5_file, 'w') as h5f:
        # Write data to HDF5 format in chunks
        results = Parallel(n_jobs=-2)(delayed(split_dwells_continuous)(meas_file, p, bin_width, dwell_time, laser_freq, skip_first) for p in pixels)
        
        for p, result in zip(pixels, results):
            # Store each result in a separate HDF5 dataset
            h5f.create_dataset(f'pixel_{p}', data=result.astype(np.float32), chunks=True)

    return output_h5_file


def process_measurement_joblib(meas_dir, bin_width, skip_first=0, skip_last=0):
    """
    Parses and bins raw data using parallel CPU processing

    Parameters
    ----------
    meas_dir : string
        path to SPAD23 measurement directory (e.g. Run042)
    bin_width : float
        bin size in ms.
    skip_first : INT
        first skip_first milliseconds of dwell time to skip.
    skip_last : INT
        last skip_last milliseconds of dwell time to skip.

    Returns
    -------
    data : 4D array
        binned data.
    """
    params = Parameters(meas_dir).p
    laser_freq = int(params.loc['lf'].values[0])  # laser frequency in MHz
    dev = params.loc['dev'].values[0]
    dt = int(params.loc['dt'].values[0])
    dwell_time = dt - skip_first - skip_last
    ref_bin_width = 0.01  # Reference bin width in ms
    num_bins = int(dwell_time / ref_bin_width) if bin_width >= ref_bin_width else int(dwell_time / bin_width)
    
    if dev == 'SPADpoints':
        output_h5_file = read_tags(meas_dir, bin_width, dwell_time, laser_freq, skip_first, num_bins)

        with h5py.File(output_h5_file, 'r') as h5f:

            dask_arrays = [da.from_array(h5f[f'pixel_{p}'], chunks=(300, 202, num_bins)) for p in range(23)]

            # Combine Dask arrays and swap axes
            data = da.stack(dask_arrays).swapaxes(1, 2)
            data = data.compute()
            h5f.close()

    elif dev == 'SPADcontinuous':
        # Process for SPADcontinuous
        output_h5_file = read_tags_continuous(meas_dir, bin_width, dwell_time, laser_freq, skip_first, num_bins)
        
        # Using Dask for processing large datasets
        with h5py.File(output_h5_file, 'r') as h5f:

            dask_arrays = [da.from_array(h5f[f'pixel_{p}'], chunks=(300, 202, num_bins)) for p in range(23)]

            # Combine Dask arrays and swap axes
            data = da.stack(dask_arrays).swapaxes(1, 2)
            data = data.compute()
            h5f.close()

    else:
        print("Data format not supported!")
        return -1

    gc.collect()
    print("Tag data shape is {}".format(data.shape))
    return data



