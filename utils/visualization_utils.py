"""Visualization utilities for plots and analysis."""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple


def plot_training_curves(history: Dict[str, List[float]], 
                         figsize: Tuple[int, int] = (12, 4),
                         save_path: Optional[str] = None) -> None:
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        figsize: Figure size
        save_path: Optional path to save figure
        
    Example:
        >>> plot_training_curves(history)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(cm: np.ndarray, class_names: Optional[List[str]] = None,
                         figsize: Tuple[int, int] = (8, 6),
                         save_path: Optional[str] = None) -> None:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix as numpy array
        class_names: Optional list of class names
        figsize: Figure size
        save_path: Optional path to save figure
        
    Example:
        >>> plot_confusion_matrix(cm, class_names=['cat', 'dog'])
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    if class_names is not None:
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    ax.set_title("Confusion Matrix")
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_samples(images: np.ndarray, labels: Optional[np.ndarray] = None,
                predictions: Optional[np.ndarray] = None, 
                n_samples: int = 10, figsize: Tuple[int, int] = (15, 3),
                save_path: Optional[str] = None) -> None:
    """
    Plot sample images with labels and predictions.
    
    Args:
        images: Array of images
        labels: Optional true labels
        predictions: Optional predictions
        n_samples: Number of samples to plot
        figsize: Figure size
        save_path: Optional path to save figure
        
    Example:
        >>> plot_samples(test_images[:10], test_labels[:10], predictions[:10])
    """
    n_samples = min(n_samples, len(images))
    fig, axes = plt.subplots(1, n_samples, figsize=figsize)
    
    if n_samples == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap='gray' if images[i].ndim == 2 else None)
        ax.axis('off')
        
        title = ""
        if labels is not None:
            title = f"True: {labels[i]}"
        if predictions is not None:
            if title:
                title += f"\nPred: {predictions[i]}"
            else:
                title = f"Pred: {predictions[i]}"
        
        if title:
            ax.set_title(title, fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_learning_rate(lr_history: List[float], figsize: Tuple[int, int] = (10, 4),
                       save_path: Optional[str] = None) -> None:
    """
    Plot learning rate schedule.
    
    Args:
        lr_history: List of learning rates over training
        figsize: Figure size
        save_path: Optional path to save figure
        
    Example:
        >>> plot_learning_rate(lr_history)
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(lr_history)
    ax.set_xlabel('Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
