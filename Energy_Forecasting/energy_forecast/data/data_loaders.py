from pathlib import Path
import sys
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
from data.dataset import CustomDataset
from torch.utils.data import DataLoader

class DataHelper:
    def __init__(self):
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def get_data_loaders(self, seq_length, forecasting_steps, batch_size, 
                        drop_last=True, 
                        train_params=None,
                        val_params=None,
                        test_params=None):

        self.train_set = CustomDataset(seq_length=seq_length, forecasting_steps=forecasting_steps, flag="train")
        self.val_set = CustomDataset(seq_length=seq_length, forecasting_steps=forecasting_steps, flag="val")
        self.test_set = CustomDataset(seq_length=seq_length, forecasting_steps=forecasting_steps, flag="test")

        train_default = {"batch_size": batch_size, "drop_last": drop_last, "shuffle": True}
        val_default = {"batch_size": batch_size, "drop_last": drop_last, "shuffle": False}
        test_default = {"batch_size": batch_size, "drop_last": drop_last, "shuffle": False}

        if train_params is not None:
            train_default.update(train_default)
        if val_params is not None:
            val_default.update(val_params)
        if test_params is not None:
            test_default.update(test_params)

        train_loader = DataLoader(dataset=self.train_set, **train_default)
        val_loader = DataLoader(dataset=self.val_set, **val_default)
        test_loader = DataLoader(dataset=self.test_set, **test_default)

        return {"train": train_loader, "val": val_loader, "test": test_loader}