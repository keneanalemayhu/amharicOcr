# src/utils/analysis/training_visualization.py

import matplotlib.pyplot as plt # type: ignore
import torch # type: ignore
import seaborn as sns # type: ignore
import numpy as np # type: ignore
from pathlib import Path
from typing import Dict, List
import torch.nn.functional as F # type: ignore
from sklearn.metrics import confusion_matrix # type: ignore
import sys

def plot_training_history(model_path: str = 'models/best_model.pt'):
    """Plot training and validation metrics history."""
    checkpoint = torch.load(model_path)
    
    # Extract metrics
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    train_accuracies = checkpoint['train_accuracies']
    val_accuracies = checkpoint['val_accuracies']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def create_confusion_matrix(model, val_loader, device, idx_to_char, num_classes):
    """Create and plot confusion matrix for most confused characters."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Find most confused pairs
    errors = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j:
                errors[i,j] = cm[i,j]
    
    # Get top confused pairs
    top_k = 20  # Show top 20 confused pairs
    confused_pairs = []
    for _ in range(top_k):
        max_idx = np.unravel_index(errors.argmax(), errors.shape)
        if errors[max_idx] > 0:
            confused_pairs.append((max_idx[0], max_idx[1], errors[max_idx]))
            errors[max_idx] = 0
    
    # Plot confusion statistics
    plt.figure(figsize=(15, 8))
    plt.barh([f"{idx_to_char[true]}->{idx_to_char[pred]}" 
              for true, pred, _ in confused_pairs],
             [count for _, _, count in confused_pairs])
    plt.title('Top Confused Character Pairs')
    plt.xlabel('Number of Confusions')
    plt.ylabel('Character Pairs (True->Predicted)')
    plt.tight_layout()
    plt.savefig('confusion_pairs.png')
    plt.close()

def fine_tune_model(model, train_loader, val_loader, device, num_epochs=10):
    """Fine-tune the model with improved settings."""
    # Freeze early layers
    for name, param in model.named_parameters():
        if 'conv_layers' in name:
            param.requires_grad = False
    
    # Use a smaller learning rate for fine-tuning
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.0001,
        weight_decay=0.02
    )
    
    # Use cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=5,
        T_mult=2
    )
    
    criterion = torch.nn.CrossEntropyLoss()
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * correct / total
        scheduler.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}]:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, '
              f'Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), project_root / 'models' / 'fine_tuned_model.pt')
    
    return model

if __name__ == "__main__":
    # Get project root directory
    project_root = Path(__file__).parents[3]  # Go up 3 levels from current file
    sys.path.append(str(project_root))  # Add project root to Python path
    
    # Set paths relative to project root
    model_path = project_root / 'models' / 'best_model.pt'
    dataset_path = project_root / 'amharic_dataset'
    
    print(f"Looking for model at: {model_path}")
    print(f"Looking for dataset at: {dataset_path}")
    
    # Plot training history
    plot_training_history(str(model_path))
    
    # Import after adding project root to path
    from model import AmharicCNN
    from src.training.data_loader import create_data_loaders
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    train_loader, val_loader, char_to_idx, idx_to_char = create_data_loaders(
        dataset_path=str(dataset_path),
        batch_size=32
    )
    
    # Load model
    model = AmharicCNN(num_classes=len(char_to_idx))
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Create confusion matrix
    create_confusion_matrix(model, val_loader, device, idx_to_char, len(char_to_idx))
    
    # Fine-tune model
    fine_tuned_model = fine_tune_model(model, train_loader, val_loader, device)