from pathlib import Path
import sys
import os

BASE_DIR = Path(__file__).resolve().parent.parent.parent
print(BASE_DIR)
sys.path.append(str(BASE_DIR))

from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, seq_length, forecasting_steps, flag:str, normalize:bool=True):
        super().__init__()
        self.seq_length = seq_length
        self.forecasting_steps = forecasting_steps
        flag_map = {"train": 0, "val": 1, "test": 2}
        self.flag = flag_map[flag]
        self.normalize = normalize
        self.scalers = {}
        self._load_data()

    def _preprocessing(self):
        pass
    
    def _load_data(self):
        print(f"current working dir {Path.cwd()}")
        
        df = pd.read_pickle(str(BASE_DIR)+"/data/processed/dataset.pkl")

        data_len = len(df.values) - self.seq_length + self.forecasting_steps + 1

        self.subset_map = {0: 0.7, 1: 0.85, 2: 1}
        if self.flag==0:
            borders = [0, int(np.ceil(data_len*0.7))]
        else:
            train_borders, borders = self._get_splits_border(data_len=data_len)


        if self.normalize:
            self.scalers["predictors"] = StandardScaler()
            self.scalers["target"] = StandardScaler()

            if self.flag==0:
                self.target = df["prices"].values[borders[0]:borders[1]]
                df = df.drop(columns=["prices"])
                self.predictors = df.values[borders[0]:borders[1]]

                self.predictors = self.scalers["predictors"].fit_transform(self.predictors)
                self.target =  self.scalers["target"].fit_transform(self.target.reshape(-1, 1))
            else:
                self.target = df["prices"].values[borders[0]:borders[1]]
                self.scalers["target"].fit(df["prices"].values[train_borders[0]:train_borders[1]].reshape(-1, 1))
                self.target = self.scalers["target"].transform(self.target.reshape(-1, 1))

                df = df.drop(columns=["prices"])
                self.predictors = df.values[borders[0]:borders[1]]

                self.scalers["predictors"].fit(df.values[train_borders[0]:train_borders[1]])
                print(f"predictors shape: {self.predictors.shape}")
                self.predictors = self.scalers["predictors"].transform(self.predictors)

        else:
            self.target = df["prices"].values[borders[0]:borders[1]]
            df = df.drop(columns=["prices"])
            self.predictors = df.values[borders[0]:borders[1]]

    def _get_splits_border(self, data_len):
        
        train_borders = [0, int(np.ceil(data_len*0.7))]
        borders = [int(np.ceil(data_len*self.subset_map[self.flag-1])), int(np.ceil(data_len*self.subset_map[self.flag]))]
        return train_borders, borders

    def __getitem__(self, index):
        preds = self.predictors[index:index+self.seq_length]
        target = self.target[index+self.seq_length+self.forecasting_steps] if self.forecasting_steps>1 else self.target[index+self.seq_length]
        return torch.tensor(preds, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

    def __len__(self):
        return len(self.predictors) - self.seq_length
    
    def inverse_transform_target(self, data):
        """Inverse la normalisation pour les valeurs prédites"""
        if self.normalize:
            return self.scalers['target'].inverse_transform(data.reshape(-1, 1)).flatten()
        return data