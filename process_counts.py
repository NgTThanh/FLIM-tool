
from .sofism_lib import process_measurement_joblib
import numpy as np
import matplotlib.pyplot as plt
from .sofism_lib.helpers import test_shifts


plt.rcParams['image.cmap'] = 'inferno'
i = 67
#path = f"C:/Users/krupi/Documents/SOFISM/data/2023_05_11/Run{i:>03}"
#shift_path = "C:/Users/krupi/Documents/SOFISM/data/2022_08_14/10nW_tiled/redShiftsCorrect.npy"
def process_counts(path, shift_path):
    shifts = -np.load(shift_path)
    data = process_measurement_joblib(path, 0.01, 5, 5)
    fig = test_shifts(data, shifts)
    return fig

def curr_shifts(shifts, i=0):
    for i in range(8):
        curr_shifts = shifts
        if i == 0:
            pass
        elif i == 1:
            curr_shifts[:, 0] = -shifts[:, 0]
        elif i == 2:
            curr_shifts[:, 1] = -shifts[:, 1]
        elif i == 3:
            curr_shifts[:, 0] = -shifts[:, 0]
            curr_shifts[:, 1] = -shifts[:, 1]
        elif i == 4:
            curr_shifts[:, 0] = shifts[:, 1]
            curr_shifts[:, 1] = shifts[:, 0]
        elif i == 5:
            curr_shifts[:, 0] = -shifts[:, 1]
            curr_shifts[:, 1] = shifts[:, 0]
        elif i == 6:
            curr_shifts[:, 0] = shifts[:, 1]
            curr_shifts[:, 1] = -shifts[:, 0]
        elif i == 7:
            curr_shifts[:, 0] = -shifts[:, 1]
            curr_shifts[:, 1] = -shifts[:, 0]
    return curr_shifts

    
