import matplotlib.pyplot as plt
import morethemes as mt


def plot_results(train_losses, val_losses, pred, target, title="Model Performance"):
    """
    Affiche les résultats d'entraînement et de test
    """
    mt.set_theme("ft", reset_to_default=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 5))
    
    # Tracer les pertes
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Tracer les prédictions vs valeurs réelles
    test_indices = np.arange(len(pred))
    ax2.plot(test_indices, target, label='Actual')
    ax2.plot(test_indices, pred, label='Predicted')
    ax2.set_xlabel('Test Sample Index')
    ax2.set_ylabel('Value')
    ax2.set_title('Predictions vs Actual')
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig