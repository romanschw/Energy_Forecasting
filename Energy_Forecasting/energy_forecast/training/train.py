from tqdm import tqdm
import time
from training.early_stopping import EarlyStopping
from torch import nn
import torch

class Trainer:
    def __init__(self, model, loss_function, optimizer, device='cpu'):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self. device = device
        self.train_losses = []
        self.best_val_loss = float('inf')

    def fit(self, train_loader, val_loader, num_epochs, patience, verbose=True):

        # Init early stopping
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        for epoch in range(num_epochs):

            epoch_loss = 0.0
            self.model.train()

            epoch_time = time.time()

            for i, (batch_x, batch_y) in enumerate(tqdm(train_loader, desc=f"epoch{epoch+1}/{num_epochs}", unit="batch")):
                # batch_x = batch_x.to_device(device)
                # batch_y = batch_y.to_device(device)
                self.optimizer.zero_grad()

                pred = self.model(batch_x)
                loss = self.loss_function(pred, batch_y)

                # backprop
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_train_loss = epoch_loss/len(train_loader)
            self.train_losses.append(avg_train_loss)

            val_loss = self._evaluate(val_loader)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss

            # Early stopping check
            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if verbose:
                print(f"Epoch: {epoch + 1}/{num_epochs} | Cost time: {time.time() - epoch_time}")
                print(f"Train Loss: {avg_train_loss} | Validation Loss: {val_loss}")
        # Load best model:
        self.model.load_state_dict(torch.load(early_stopping.path))
        
        return self.model, self.best_val_loss
    
    def _evaluate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(val_loader):
                pred = self.model(batch_x)
                loss = self.loss_function(pred, batch_y)
                val_loss += loss.item()

        avg_val_loss = val_loss/len(val_loader)
        return avg_val_loss


