import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
from pathlib import Path
import sys
from typing import Tuple, Dict

# Add src to Python path
sys.path.append(str(Path(__file__).parent))

# Import from correct location
from src.training.data_loader import create_data_loaders

class AmharicCNN(nn.Module):
    def __init__(self, num_classes: int):
        super(AmharicCNN, self).__init__()
        
        # Enhanced Convolutional layers for Amharic characters
        self.conv_layers = nn.Sequential(
            # First conv block with increased filters
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Additional conv layer
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Fourth conv block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # Additional conv layer
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Calculate the size of flattened features
        self.flatten_size = 512 * 6 * 6  # Adjusted for 96x96 input
        
        # Enhanced fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.flatten_size, 4096),  # Increased neurons
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 2048),  # Increased neurons
            nn.ReLU(),
            nn.Dropout(0.4),  # Increased dropout rate
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.view(-1, self.flatten_size)
        x = self.fc_layers(x)
        return x

class AmharicOCRTrainer:
    def __init__(
        self, 
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 0.001,
        model_dir: str = 'models'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Add idx_to_char mapping
        self.idx_to_char = train_loader.dataset.idx_to_char
        
        # Enhanced optimizer with weight decay
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01  # L2 regularization
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3
        )
        # Track initial learning rate
        self.current_lr = learning_rate
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        running_loss = 0.0
        last_print = 0
        print_interval = 25  # Print every 25 batches
        
        # Progress tracking
        print("\nTraining Progress:")
        print("=" * 80)
        print(f"{'Batch':^10} | {'Loss':^12} | {'Accuracy':^12} | {'Running Loss':^12} | {'Characters':^20}")
        print("-" * 80)
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass and optimize
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Print detailed progress
            if (batch_idx + 1) % print_interval == 0:
                accuracy = 100. * correct / total
                avg_running_loss = running_loss / (batch_idx + 1)
                
                # Get some example predictions for this batch
                example_indices = range(min(3, len(labels)))  # Show first 3 examples
                examples = []
                for i in example_indices:
                    true_char = self.idx_to_char[labels[i].item()]
                    pred_char = self.idx_to_char[predicted[i].item()]
                    examples.append(f"{true_char}→{pred_char}")
                
                print(f"{batch_idx + 1:^10} | {loss.item():^12.4f} | {accuracy:^12.2f} | {avg_running_loss:^12.4f} | {' '.join(examples):^20}")
                
                last_print = batch_idx
        
        # Print final statistics if we haven't recently printed
        if last_print != len(self.train_loader) - 1:
            accuracy = 100. * correct / total
            avg_running_loss = total_loss / len(self.train_loader)
            print("-" * 80)
            print(f"Final:   | {loss.item():^12.4f} | {accuracy:^12.2f} | {avg_running_loss:^12.4f} |")
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy

    def validate(self) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        confusion = {}  # Track confusions between characters
        
        print("\nValidation Progress:")
        print("=" * 60)
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.val_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Track confusion patterns
                for true, pred in zip(labels, predicted):
                    true_char = self.idx_to_char[true.item()]
                    pred_char = self.idx_to_char[pred.item()]
                    if true_char != pred_char:
                        key = f"{true_char}→{pred_char}"
                        confusion[key] = confusion.get(key, 0) + 1
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        # Print validation results
        print(f"\nValidation Results:")
        print(f"Loss: {avg_loss:.4f}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Correct predictions: {correct}/{total}")
        
        # Print most common confusions
        if confusion:
            print("\nTop 5 Common Confusions:")
            common_confusions = sorted(confusion.items(), key=lambda x: x[1], reverse=True)[:5]
            for chars, count in common_confusions:
                print(f"{chars}: {count} times")
        
        return avg_loss, accuracy

    def train(self, num_epochs: int):
        best_val_accuracy = 0
        patience = 7  # Early stopping patience
        no_improve = 0
        
        print("\nStarting Training:")
        print("=" * 80)
        print(f"Total epochs: {num_epochs}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Initial learning rate: {self.current_lr}")
        print("=" * 80)
        
        for epoch in range(num_epochs):
            print(f'\nEpoch [{epoch+1}/{num_epochs}]')
            print("-" * 40)
            start_time = time.time()
            
            # Train and validate
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            # Learning rate scheduling
            previous_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Track metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            epoch_time = time.time() - start_time
            
            # Print epoch summary
            print("\nEpoch Summary:")
            print("-" * 60)
            print(f"Training Loss: {train_loss:.4f}")
            print(f"Training Accuracy: {train_acc:.2f}%")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Accuracy: {val_acc:.2f}%")
            print(f"Time taken: {epoch_time:.2f}s")
            
            if current_lr != previous_lr:
                print(f"Learning rate decreased: {previous_lr:.6f} → {current_lr:.6f}")
            
            # Save best model
            if val_acc > best_val_accuracy:
                improvement = val_acc - best_val_accuracy
                best_val_accuracy = val_acc
                self.save_model('best_model.pt')
                print(f"\nNew best model saved! (Improvement: +{improvement:.2f}%)")
                no_improve = 0
            else:
                no_improve += 1
                print(f"\nNo improvement for {no_improve} epochs")
            
            # Early stopping check
            if no_improve >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"Best validation accuracy achieved: {best_val_accuracy:.2f}%")
                break
            
            # Regular checkpoints
            if (epoch + 1) % 5 == 0:
                self.save_model(f'checkpoint_epoch_{epoch+1}.pt')
                print(f"\nCheckpoint saved: checkpoint_epoch_{epoch+1}.pt")

    def save_model(self, filename: str):
        """Save model checkpoint with additional information"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
        torch.save(checkpoint, self.model_dir / filename)

    def load_model(self, filename: str):
        """Load model checkpoint with additional information"""
        checkpoint = torch.load(self.model_dir / filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_accuracies = checkpoint['train_accuracies']
        self.val_accuracies = checkpoint['val_accuracies']

def main():
    # Set device and CUDA settings
    print("CUDA debug info:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # Set CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print(f'CUDA Version: {torch.version.cuda}')
    else:
        device = torch.device('cpu')
        print('CUDA not available, using CPU')
    
    # Create data loaders
    train_loader, val_loader, char_to_idx, idx_to_char = create_data_loaders(
        dataset_path="amharic_dataset",
        batch_size=32
    )
    
    # Create model
    model = AmharicCNN(num_classes=len(char_to_idx))
    
    # Create trainer
    trainer = AmharicOCRTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=0.001
    )
    
    # Train model
    trainer.train(num_epochs=50)

if __name__ == "__main__":
    main()