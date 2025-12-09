"""Training loop and evaluation utilities."""

import torch
import torch.nn as nn
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from tqdm import tqdm


def train_epoch(model: nn.Module, dataloader: torch.utils.data.DataLoader,
               criterion: nn.Module, optimizer: torch.optim.Optimizer,
               device: str = 'cpu', desc: str = 'Training') -> Tuple[float, float]:
    """
    Train model for one epoch.
    
    Args:
        model: PyTorch model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        desc: Progress bar description
        
    Returns:
        Tuple of (average_loss, accuracy)
        
    Example:
        >>> loss, acc = train_epoch(model, train_loader, criterion, optimizer, device='cuda')
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=desc)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                         'acc': f'{100.*correct/total:.2f}%'})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def evaluate(model: nn.Module, dataloader: torch.utils.data.DataLoader,
            criterion: nn.Module, device: str = 'cpu', 
            desc: str = 'Evaluating') -> Tuple[float, float]:
    """
    Evaluate model on validation/test set.
    
    Args:
        model: PyTorch model
        dataloader: Validation/test data loader
        criterion: Loss function
        device: Device to evaluate on
        desc: Progress bar description
        
    Returns:
        Tuple of (average_loss, accuracy)
        
    Example:
        >>> loss, acc = evaluate(model, val_loader, criterion, device='cuda')
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=desc)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                             'acc': f'{100.*correct/total:.2f}%'})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def train_model(model: nn.Module, train_loader: torch.utils.data.DataLoader,
               val_loader: torch.utils.data.DataLoader, criterion: nn.Module,
               optimizer: torch.optim.Optimizer, epochs: int,
               device: str = 'cpu', scheduler: Optional[Any] = None,
               save_best: bool = True, checkpoint_path: str = 'best_model.pt') -> Dict[str, List[float]]:
    """
    Complete training loop with validation.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        epochs: Number of epochs to train
        device: Device to train on
        scheduler: Optional learning rate scheduler
        save_best: Whether to save best model
        checkpoint_path: Path to save best model
        
    Returns:
        Dictionary containing training history
        
    Example:
        >>> history = train_model(model, train_loader, val_loader, 
        ...                       criterion, optimizer, epochs=10, device='cuda')
    """
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0
    
    for epoch in range(epochs):
        print(f'\nEpoch {epoch+1}/{epochs}')
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                           optimizer, device, 'Training')
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, 'Validation')
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        if scheduler:
            scheduler.step()
        
        if save_best and val_acc > best_val_acc:
            best_val_acc = val_acc
            from .model_utils import save_checkpoint
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path,
                          {'val_acc': val_acc, 'train_acc': train_acc})
            print(f'Saved best model with validation accuracy: {val_acc:.2f}%')
    
    return history
