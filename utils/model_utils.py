"""Model creation and management utilities."""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
        
    Example:
        >>> model = SimpleNet()
        >>> print(f"Parameters: {count_parameters(model):,}")
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                   epoch: int, loss: float, path: str, 
                   additional_info: Optional[Dict[str, Any]] = None) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model to save
        optimizer: Optimizer state to save
        epoch: Current epoch number
        loss: Current loss value
        path: Path to save checkpoint
        additional_info: Optional dictionary of additional info to save
        
    Example:
        >>> save_checkpoint(model, optimizer, epoch=10, loss=0.5, path='checkpoints/model.pt')
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, path)


def load_checkpoint(model: nn.Module, optimizer: Optional[torch.optim.Optimizer], 
                   path: str, device: str = 'cpu') -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model to load weights into
        optimizer: Optional optimizer to load state into
        path: Path to checkpoint file
        device: Device to load model to
        
    Returns:
        Dictionary containing checkpoint information
        
    Example:
        >>> checkpoint = load_checkpoint(model, optimizer, 'checkpoints/model.pt')
        >>> print(f"Resuming from epoch {checkpoint['epoch']}")
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def freeze_layers(model: nn.Module, freeze_until: Optional[str] = None) -> None:
    """
    Freeze model layers for transfer learning.
    
    Args:
        model: PyTorch model
        freeze_until: Name of layer to freeze until (None = freeze all)
        
    Example:
        >>> freeze_layers(model, freeze_until='layer3')
    """
    freeze = True
    for name, param in model.named_parameters():
        if freeze_until and freeze_until in name:
            freeze = False
        param.requires_grad = not freeze


def initialize_weights(model: nn.Module, method: str = 'xavier') -> None:
    """
    Initialize model weights.
    
    Args:
        model: PyTorch model
        method: Initialization method ('xavier', 'kaiming', 'normal')
        
    Example:
        >>> initialize_weights(model, method='kaiming')
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if method == 'xavier':
                nn.init.xavier_uniform_(m.weight)
            elif method == 'kaiming':
                nn.init.kaiming_uniform_(m.weight)
            elif method == 'normal':
                nn.init.normal_(m.weight, mean=0, std=0.01)
            
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
