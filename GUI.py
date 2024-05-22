import tkinter as tk
from tkinter import filedialog
import get_shifts as gs, process_counts as pc
from sofism_lib import process_measurement_joblib
from sofism_lib.helpers import visualize, spad_sum_fig
from sofism_lib.core import get_corrs, sofism
from sofism_lib.params import Parameters
import lifetimefit as lf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import sys
import os

root = tk.Tk()
root.title("FLIM GUI")

class TextRedirector(object):
    def __init__(self, widget):
        self.widget = widget

    def write(self, str):
        self.widget.insert(tk.END, str)
        self.widget.see(tk.END)

    def flush(self):
        pass


def display_plot(fig):
    # Create a new window
    new_window = tk.Toplevel(root)

    # Display the plot in the new window
    canvas = FigureCanvasTkAgg(fig, master=new_window)
    canvas.draw()

    # Add the Navigation Toolbar
    toolbar = NavigationToolbar2Tk(canvas, new_window)
    toolbar.update()

    canvas.get_tk_widget().pack()

def assign_value_x(new_x):
    global x
    x = new_x
    print(f'x is set as {x}')

def assign_value_y(new_y):
    global y
    y = new_y
    print(f'y is set as {y}')

def fit_lifetime(data, bin_width, i, j):
    if i is None or j is None:
        print('Please select a pixel to fit.')
    else:
        if i > data.shape[0] or j > data.shape[1]:
            print('Invalid pixel selection. Please select a pixel within the data range.')     
        else:
            fig = lf.fit_curve(data, bin_width, i, j)
            if fig is not None:
                display_plot(fig)

def fitting():
    global data, lifetime_map, bin_width, x, y
    lifetime_map = lf.lifetime_fit(data, bin_width)
    fig = lf.visu(data, lifetime_map)
    display_plot(fig)

    fitx_label = tk.Label(settings_frame, text="Select x:")
    fitx_entry = tk.Entry(settings_frame)
    fitx_entry.bind("<Return>", lambda event: assign_value_x(int(fitx_entry.get())))

    fitx_label.grid(row=3, column=0)
    fitx_entry.grid(row=3, column=1)

    fity_label = tk.Label(settings_frame, text="Select y:")
    fity_entry = tk.Entry(settings_frame)
    fity_entry.bind("<Return>", lambda event: assign_value_y(int(fity_entry.get())))

    fity_label.grid(row=3, column=2)
    fity_entry.grid(row=3, column=3)
    fit_button = tk.Button(settings_frame, text="Fit", command=lambda: fit_lifetime(data, bin_width, x, y))
    fit_button.grid(row=3, column=4)
    
def phasor_fit_lifetime(data, bin_width, i, j):
    if i is None or j is None:
        print('Please select a pixel to fit.')
    else:
        if i > data.shape[0] or j > data.shape[1]:
            print('Invalid pixel selection. Please select a pixel within the data range.')     
        else:
            fig = lf.phasor_curve(data, bin_width, i, j)
            if fig is not None:
                display_plot(fig)


def phasor():
    global data, lifetime_map, bin_width, x, y
    lifetime_map = lf.phasor_map(data, bin_width)
    fig = lf.visu(data, lifetime_map)
    display_plot(fig)

    fitx_label = tk.Label(settings_frame, text="Select x:")
    fitx_entry = tk.Entry(settings_frame)
    fitx_entry.bind("<Return>", lambda event: assign_value_x(int(fitx_entry.get())))

    fitx_label.grid(row=3, column=0)
    fitx_entry.grid(row=3, column=1)

    fity_label = tk.Label(settings_frame, text="Select y:")
    fity_entry = tk.Entry(settings_frame)
    fity_entry.bind("<Return>", lambda event: assign_value_y(int(fity_entry.get())))

    fity_label.grid(row=3, column=2)
    fity_entry.grid(row=3, column=3)
    fit_button = tk.Button(settings_frame, text="Fit", command=lambda: phasor_fit_lifetime(data, bin_width, x, y))
    fit_button.grid(row=3, column=4)                    

current_path = None
data = None
bin_width=None
shifts=None
shift_array = None
num_bins = None
params = None
name = None
x = None
y = None

# Frame for Path, Browse, Upload Shifts, Load Data, Bin Width
settings_frame = tk.Frame(root)
settings_frame.pack()


path_label = tk.Label(settings_frame, text="Path:")
path_label.grid(row=0, column=0, sticky="w")  # Sticky for alignment

path_entry = tk.Entry(settings_frame, width=40)
path_entry.grid(row=0, column=1)

def load_params(path):
    global params
    params = Parameters(path).p
    print('Parameters loaded.')

def browse_path():
    new_path = filedialog.askdirectory()
    if new_path:
        path_entry.delete(0, tk.END)
        path_entry.insert(0, new_path)
    
    if path_entry.get() is not None:
        print('Shifts calculated for: ', new_path)
        calculate_shifts()
        set_num_bins()
        load_params(new_path)

browse_button = tk.Button(settings_frame, text="Browse", command=browse_path)
browse_button.grid(row=0, column=2)

def update_shifts(shifts, i):
    global shift_array
    shift_array = -np.load(shifts)
    shift_array = pc.curr_shifts(shift_array, i)
    print('Shifts updated as: ', i)

def calculate_shifts():
    global shifts, current_path
    current_path = path_entry.get()
    shifts = gs.get_shifts(current_path)

    # Process counts (assuming 'pc' exists elsewhere)
    fig = pc.process_counts(current_path, shifts) 

    # Display text results 
    shifts_output.delete(1.0, tk.END) 
    display_plot(fig)
   
    # Add label, entry and button for shift selection
    shift_label = tk.Label(settings_frame, text="Select shift (0-8):")
    shift_entry = tk.Entry(settings_frame)
    shift_entry.bind("<Return>", lambda event: update_shifts(shifts, int(shift_entry.get())))

    shift_label.grid(row=1, column=0)
    shift_entry.grid(row=1, column=1)


def upload_shifts():
    global shifts, shift_array
    # Open file dialog to select .npy file
    shifts = filedialog.askopenfilename(filetypes=[("Numpy files", "*.npy")])
    
    if shifts:
        # Load the selected file
        print('Shifts loaded from: ', shifts)
        shift_label = tk.Label(settings_frame, text="Select shift (0-8):")
        shift_entry = tk.Entry(settings_frame)
        shift_entry.bind("<Return>", lambda event: update_shifts(shifts, int(shift_entry.get())))

    
    shift_label.grid(row=1, column=0)
    shift_entry.grid(row=1, column=1)


upload_shifts_button = tk.Button(settings_frame, text="Upload Shifts", command=upload_shifts)
upload_shifts_button.grid(row=0, column=3)

def data_load():
    global data, bin_width_entry, bin_width
    # Open file dialog to select .npy or .txt file
    file_path = filedialog.askopenfilename(filetypes=[("Numpy files", "*.npy"), ("Text files", "*.txt")])
    
    if file_path:
        # Check the file extension
        _, ext = os.path.splitext(file_path)

        if ext == ".npy":
            # Load the selected .npy file
            data = np.load(file_path)
        elif ext == ".txt":
            # Load the selected .txt file and apply transform conditions
            data = np.loadtxt(file_path)
            xy, t = data.shape
            data = data.reshape(np.sqrt(xy).astype(int), np.sqrt(xy).astype(int), t)
        print('Data loaded from: ', file_path)
        print(data.shape)
        analyze()

data_load_button = tk.Button(settings_frame, text="Load Data", command=data_load)
data_load_button.grid(row=0, column=4)

def update_bin_width():
    try:
        # Attempt to convert the input to a float
        new_bin_width = float(bin_width_entry.get()) * 1e-6 
        print("Bin width updated:", bin_width_entry.get(), 'ns')

        # If the conversion was successful, update the global bin_width
        global bin_width
        bin_width = new_bin_width 

    except ValueError:
        print("Invalid input. Please enter a number.")

bin_width_label = tk.Label(settings_frame, text='Set bin width (ns):')
bin_width_label.grid(row=0, column=5)

bin_width_entry = tk.Entry(settings_frame, width=10)
bin_width_entry.grid(row=0, column=6)
bin_width_entry.bind("<Return>", lambda event: update_bin_width())

# Frame for Output and Analysis
output_frame = tk.Frame(root)
output_frame.pack()

output_text = tk.Text(output_frame)
output_text.grid()

# Initialize shifts_output at a higher scope
shifts_output = tk.Text(output_frame)
shifts_output.grid_forget()


def set_num_bins():
    def get_and_close():
        global num_bins
        try:
            new_num_bins = int(num_bins_entry.get())
            num_bins = new_num_bins
            print("Number of Bins is updated to:", num_bins)

        except ValueError:
            print("Invalid input. Please enter an integer.")

    num_bins_label = tk.Label(settings_frame, text="Enter the # bins:")
    num_bins_entry = tk.Entry(settings_frame, width=10)
    num_bins_entry.bind("<Return>", lambda event: get_and_close())

    num_bins_label.grid(row=1,column=2)
    num_bins_entry.grid(row=1, column=3)

def update_data(x, name):
    global data
    data = x
    print('Data is set as ', name)
    if name == 'CLSM' or name == 'ISM':
        fig = spad_sum_fig(data, params)
        display_plot(fig)
    else:
        analyze()

    file_types = [('NPY', '*.npy'), ('TEXT', '*.txt')]
    save_path = filedialog.asksaveasfilename(defaultextension='.npy', filetypes=file_types)

    if save_path: 
        # Get the file extension to determine the save format
        _, file_extension = os.path.splitext(save_path)

        if file_extension == '.npy':
            np.save(save_path, data)
            print('Data saved as .npy:', save_path)
        elif file_extension == '.txt':
            np.savetxt(save_path, data)
            print('Data saved as .txt:', save_path)
        else:
            print('Unsupported file type.')

def measurement():
    global data, bin_width_entry, bin_width, num_bins, shift_array
    path = current_path
    if shifts and num_bins and current_path and bin_width is not None:
        print('Process analysis for: ', path)
        data = process_measurement_joblib(path, bin_width, 5, 5)
        fig, _ = visualize(data, path, params)
        display_plot(fig)
        cor = get_corrs(data, num_bins, no_mean=True)
        res = sofism(data, cor, shift_array, dx=0.05)
        clsm, ism, sofi_clsm, sofism_xc, sofism_all = res.values()
    
        clsm_button = tk.Button(settings_frame, text='CLSM', command=lambda: update_data(clsm, 'CLSM'))
        ism_button = tk.Button(settings_frame, text='ISM', command=lambda: update_data(ism, 'ISM'))
        sofi_clsm_button = tk.Button(settings_frame, text='SOFI_CLSM', command=lambda: update_data(sofi_clsm, 'SOFI-CLSM'))
        sofism_xc_button = tk.Button(settings_frame, text='SOFISM_XC', command=lambda: update_data(sofism_xc, 'SOFISM-XC'))
        sofism_all_button = tk.Button(settings_frame, text='SOFISM_ALL', command=lambda: update_data(sofism_all, 'SOFISM-ALL'))

        clsm_button.grid(row=2, column=0, sticky="n")
        ism_button.grid(row=2, column=1, sticky="n")
        sofi_clsm_button.grid(row=2, column=2, sticky="n") 
        sofism_xc_button.grid(row=2, column=3, sticky="n")
        sofism_all_button.grid(row=2, column=4, sticky="n")

    else:
        print('Please set all parameters before analysis.')
    

# 'Analyze' Buttons 
data_button = tk.Button(settings_frame, text="Analyze", command=measurement)
data_button.grid(row=1, column=5)

def analyze():
    fitting_button = tk.Button(settings_frame, text="Fitting", command=fitting)
    fitting_button.grid(row=3, column=5)

    phasor_button = tk.Button(settings_frame, text="Phasor", command=phasor)
    phasor_button.grid(row=3, column=6)


# Save the original stdout
original_stdout = sys.stdout

# Redirect stdout to output text widget
sys.stdout = TextRedirector(output_text)



try:
    root.mainloop()
finally:
    # Ensure that stdout is set back to its original value when the mainloop is terminated
    sys.stdout = original_stdout
