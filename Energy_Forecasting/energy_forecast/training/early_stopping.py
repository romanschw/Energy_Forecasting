import torch

class EarlyStopping:
    def __init__(self, patience=7, verbose=True, path:str="checkpoint.pt"):
        self.patience = patience
        self.verbose = verbose
        self.path = path
        self.best_score = None
        self.counter = 0
        self.val_loss_min = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss, model):
        "Dunder methods to use a class instance as a function"
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, val_loss)

        elif score < self.best_score:

            self.counter += 1
            if self.verbose:
                print(f"Validation loss decreased: {self.counter} out of {self.patience} epochs")
            if self.counter==self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, val_loss)
            self.counter = 0

    
    def save_checkpoint(self, model, val_loss):
        if self.verbose:
            print(f"Validation metric decreased: {self.val_loss_min: .6f} ----> {val_loss}")
        # .state_dict(): Return a dictionary containing references to the whole state of the module.
        torch.save(obj=model.state_dict(), f=self.path)
        self.val_loss_min = val_loss
