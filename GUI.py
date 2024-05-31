import tkinter as tk
from tkinter import filedialog
from pakiet_studencki.code import get_shifts as gs, process_counts as pc
from pakiet_studencki.code.sofism_lib import process_measurement_joblib
from pakiet_studencki.code.sofism_lib.helpers import visualize, spad_sum_fig
from pakiet_studencki.code.sofism_lib.core import get_corrs, sofism
from pakiet_studencki.code.sofism_lib.params import Parameters
from pakiet_studencki.code import lifetimefit as lf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import sys
import os


class GUIState:
    def __init__(self):
        self.x = None
        self.y = None
        self.data = None
        self.bin_width = None
        self.shifts = None
        self.shift_array = None
        self.num_bins = None
        self.params = None
        self.name = None
        self.current_path = None
        self._clear_on_exit = False  # Flag to track if clearing is needed

    def __del__(self):
        if self._clear_on_exit:
            self.clear()

    def clear(self):
        # Clear variables
        self.x = None
        self.y = None
        self.data = None
        self.bin_width = None
        self.shifts = None
        self.shift_array = None
        self.num_bins = None
        self.params = None
        self.name = None
        self.current_path = None

    def request_clear_on_exit(self):
        self._clear_on_exit = True  # Signal that variables need clearing
    


state = GUIState()

root = tk.Tk()
root.title("FLIM-tool")

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
    state.x = new_x
    print(f'x is set as {state.x}')

def assign_value_y(new_y):
    state.y = new_y
    print(f'y is set as {state.y}')

    
def fit_decay(state, i, j):
    if i is None or j is None:
        print('Please select a pixel to fit.')
    else:
        if i > state.data.shape[0] or j > state.data.shape[1]:
            print('Invalid pixel selection. Please select a pixel within the data range.')     
        else:
            fig = lf.fit_curve(state.data, state.bin_width, i, j)
            if fig is not None:
                display_plot(fig)


def flim(state):
    fig = lf.visu(state.data, state.bin_width)  # Pass data if needed
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
    fit_button = tk.Button(settings_frame, text="Fit", command=lambda: fit_decay(state, state.x, state.y)) 
    fit_button.grid(row=3, column=4)                  


# Frame for Path, Browse, Upload Shifts, Load Data, Bin Width
settings_frame = tk.Frame(root)
settings_frame.pack()


path_label = tk.Label(settings_frame, text="Path:")
path_label.grid(row=0, column=0, sticky="w")  # Sticky for alignment

path_entry = tk.Entry(settings_frame, width=40)
path_entry.grid(row=0, column=1)

def load_params(path):
    state.params = Parameters(path).p
    print('Parameters loaded.')

def browse_path():
    new_path = filedialog.askdirectory()
    if new_path:
        path_entry.delete(0, tk.END)
        path_entry.insert(0, new_path)
    
    if path_entry.get() is not None:
        print('Shifts calculated for: ', new_path)
        calculate_shifts(state)
        set_num_bins()
        load_params(new_path)

browse_button = tk.Button(settings_frame, text="Browse", command=browse_path)
browse_button.grid(row=0, column=2)

def update_shifts(shifts, i):
    shift_array = -np.load(shifts)
    state.shift_array = pc.curr_shifts(shift_array, i+1)
    print('Shifts updated as: ', i)

def calculate_shifts(state):
    state.current_path = path_entry.get() 
    state.shifts = gs.get_shifts(state.current_path)

    # Process counts (assuming 'pc' exists elsewhere)
    fig = pc.process_counts(state.current_path, state.shifts) 

    # Display text results 
    shifts_output.delete(1.0, tk.END) 
    display_plot(fig)
   
    # Add label, entry and button for shift selection
    shift_label = tk.Label(settings_frame, text="Select shift (0-8):")
    shift_entry = tk.Entry(settings_frame)
    shift_entry.bind("<Return>", lambda event: update_shifts(state.shifts, int(shift_entry.get())))

    shift_label.grid(row=1, column=0)
    shift_entry.grid(row=1, column=1)


def upload_shifts():
    # Open file dialog to select .npy file
    state.shifts = filedialog.askopenfilename(filetypes=[("Numpy files", "*.npy")])
    
    if state.shifts:
        # Load the selected file
        print('Shifts loaded from: ', state.shifts)
        shift_label = tk.Label(settings_frame, text="Select shift (0-8):")
        shift_entry = tk.Entry(settings_frame)
        shift_entry.bind("<Return>", lambda event: update_shifts(state.shifts, int(shift_entry.get())))

    
    shift_label.grid(row=1, column=0)
    shift_entry.grid(row=1, column=1)


upload_shifts_button = tk.Button(settings_frame, text="Upload Shifts", command=upload_shifts)
upload_shifts_button.grid(row=0, column=3)

def data_load():
    # Open file dialog to select .npy or .txt file
    file_path = filedialog.askopenfilename(filetypes=[("Numpy files", "*.npy"), ("Text files", "*.txt")])
    
    if file_path:
        # Check the file extension
        _, ext = os.path.splitext(file_path)

        if ext == ".npy":
            # Load the selected .npy file
            state.data = np.load(file_path)
        elif ext == ".txt":
            # Load the selected .txt file and apply transform conditions
            state.data = np.loadtxt(file_path)
            xy, t = state.data.shape
            state.data = state.data.reshape(np.sqrt(xy).astype(int), np.sqrt(xy).astype(int), t)
        print('Data loaded from: ', file_path)
        print(state.data.shape)
        if state.bin_width is not None:
            analyze(state)
        else:
            print('Please set bin width before upload data.')

data_load_button = tk.Button(settings_frame, text="Load Data", command=data_load)
data_load_button.grid(row=0, column=4)

def update_bin_width():
    try:
        # Attempt to convert the input to a float
        new_bin_width = float(bin_width_entry.get())
        print("Bin width updated:", bin_width_entry.get(), 'ns')

        # If the conversion was successful, update the global bin_width
        state.bin_width = new_bin_width 

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
        try:
            new_num_bins = int(num_bins_entry.get())
            state.num_bins = new_num_bins
            print("Number of Bins is updated to:", state.num_bins)

        except ValueError:
            print("Invalid input. Please enter an integer.")

    num_bins_label = tk.Label(settings_frame, text="Enter the # bins:")
    num_bins_entry = tk.Entry(settings_frame, width=10)
    num_bins_entry.bind("<Return>", lambda event: get_and_close())

    num_bins_label.grid(row=1,column=2)
    num_bins_entry.grid(row=1, column=3)

def update_data(x, name):
    state.data = x
    print('Data is set as ', name)
    if name == 'CLSM' or name == 'ISM':
        fig = spad_sum_fig(state.data, state.params)
        display_plot(fig)
    else:
        analyze(state)

    file_types = [('NPY', '*.npy'), ('TEXT', '*.txt')]
    save_path = filedialog.asksaveasfilename(defaultextension='.npy', filetypes=file_types)

    if save_path: 
        # Get the file extension to determine the save format
        _, file_extension = os.path.splitext(save_path)

        if file_extension == '.npy':
            np.save(save_path, state.data)
            print('Data saved as .npy:', save_path)
        elif file_extension == '.txt':
            np.savetxt(save_path, state.data)
            print('Data saved as .txt:', save_path)
        else:
            print('Unsupported file type.')

def measurement(state):
    path = state.current_path  # Access current_path through state
    if state.shifts and state.num_bins and path and state.bin_width is not None:
        print('Process analysis for: ', path)
        data = process_measurement_joblib(path, state.bin_width*1e-6, 5, 5)  
        fig, _ = visualize(data, path, state.params)
        display_plot(fig)
        cor = get_corrs(data, state.num_bins, no_mean=True)
        res = sofism(data, cor, state.shift_array, dx=0.05)
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
        if state.shifts is None:
            print('Shifts is missing!')
        if state.num_bins is None:
            print('Please set number of bins before analysis.')
        if state.bin_width is None:
            print('Please set bin width before analysis.')
        
    

# 'Analyze' Buttons 
data_button = tk.Button(settings_frame, text="Analyze", command=lambda: measurement(state))
data_button.grid(row=1, column=5)

def analyze(state):
    flim_button = tk.Button(settings_frame, text="FLIM anaysis", command=lambda: flim(state))
    flim_button.grid(row=3, column=5)


# Save the original stdout
original_stdout = sys.stdout

# Redirect stdout to output text widget
sys.stdout = TextRedirector(output_text)

def on_closing():
    state.request_clear_on_exit()
    root.destroy() 

try:
    root.protocol("WM_DELETE_WINDOW", on_closing) # Ensure that stdout is set back to its original value when the mainloop is terminated
    root.mainloop()
finally:
    sys.stdout = original_stdout
