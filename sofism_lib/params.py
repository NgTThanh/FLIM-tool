import os
import pandas as pd


class Parameters:

    def __init__(self, param_dir, **params: dict):
        self.p = pd.DataFrame(params)
        self.dir = param_dir
        self.load_params()

    def __str__(self):
        return f"Parameters for SPAD measurement data in {self.dir} \n {self.p}"

    def save_params(self):
        if self.dir == None:
            print("Save path is empty!")
        else:
            self.p.to_csv(os.path.join(self.dir, "scanParameters.csv"))

    def load_params(self):
        if self.dir == None:
            print("Load path is empty!")
        else:
            self.p = pd.read_csv(os.path.join(self.dir, "scanParameters.csv"),
                                 index_col=0)
