import torch
import numpy as np

class Evaluate:
    def __init__(self, model, loss_function):
        self.model = model
        self.loss_function = loss_function
    
    def evaluate_model(self, test_loader, dataset):

        test_loss = 0.0
        preds_series = []
        targets_series = []
        self.model.eval() #desactivate dropout

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                preds = self.model(batch_x)
                loss = self.loss_function(preds, batch_y)
                test_loss += loss
                preds_series.extend(preds.numpy())
                targets_series.extend(batch_y.numpy())

        # Calculer la perte moyenne
        avg_test_loss = test_loss / len(test_loader)
        
        # Convertir en arrays numpy
        preds_series = np.array(preds_series)
        targets_series = np.array(targets_series)
        
        # Inverser la normalisation si nécessaire
        if dataset.normalize:
            preds_series = dataset.inverse_transform_target(preds_series)
            targets_series = dataset.inverse_transform_target(targets_series)
        
        # Calculer les métriques
        mse = np.mean((preds_series - targets_series) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(preds_series - targets_series))
        
        metrics = {
            'test_loss': avg_test_loss,
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }
        
        return metrics, preds_series, targets_series