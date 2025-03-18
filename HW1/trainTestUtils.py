import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time


# Deep Learning Models

class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network for audio classification.
    Combines CNNs with RNNs for temporal modeling.
    """
    def __init__(self, num_classes=6):
        super(CRNN, self).__init__()
        # CNN layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        
        # RNN layer
        self.rnn = nn.GRU(input_size=32*4, hidden_size=64, num_layers=2, 
                          batch_first=True, bidirectional=True)
        
        # Fully connected layer
        self.fc = nn.Linear(64*2, num_classes)  # bidirectional -> *2
        
    def forward(self, x):
        # Add channel dimension if needed
        if x.dim() == 3:  # [batch, mels, time]
            x = x.unsqueeze(1)  # [batch, channels, mels, time]
            
        # CNN feature extraction
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Reshape for RNN: [batch, channels, freq, time] -> [batch, time, channels*freq]
        batch, channels, freq, time = x.size()
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous().view(batch, time, channels * freq)
        
        # RNN sequence modeling
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)
        
        # Take the last time step
        x = x[:, -1, :]
        
        # Classification
        x = self.fc(x)
        return x

class ResidualBlock(nn.Module):
    """Residual block for ResNet architecture"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class AudioResNet(nn.Module):
    """ResNet-inspired model for audio classification"""
    def __init__(self, num_blocks, num_classes=6):
        super(AudioResNet, self).__init__()
        self.in_channels = 16
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Create residual layers
        self.layer1 = self._make_layer(16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(64, num_blocks[2], stride=2)
        
        # Final fully connected layer
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        
    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Add channel dimension if needed
        if x.dim() == 3:  # [batch, mels, time]
            x = x.unsqueeze(1)  # [batch, channels, mels, time]
            
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out

# Traditional ML Feature Extraction
def extract_features_for_ml(data_loader):
    """
    Extract features from the MelSpectrogram for traditional ML models.
    Returns X (features) and y (labels) numpy arrays.
    """
    X = []
    y = []
    
    for batch in tqdm(data_loader, desc="Extracting features"):
        waveforms, labels = batch
        
        # Process each sample in the batch
        for i in range(waveforms.shape[0]):
            mel_spec = waveforms[i]
            
            # Convert to numpy for feature extraction
            if isinstance(mel_spec, torch.Tensor):
                mel_spec = mel_spec.numpy()
            
            # Extract features (simple statistical features)
            features = []
            
            # Mean and std for each mel band
            mean_per_band = np.mean(mel_spec, axis=-1)
            std_per_band = np.std(mel_spec, axis=-1)
            
            # Add to features
            features.extend(mean_per_band.flatten())
            features.extend(std_per_band.flatten())
            
            # Add more features
            features.append(np.mean(mel_spec))  # Overall mean
            features.append(np.std(mel_spec))   # Overall std
            features.append(np.max(mel_spec))   # Max value
            features.append(np.min(mel_spec))   # Min value
            
            # Add to dataset
            X.append(features)
            y.append(labels[i].item())
    
    return np.array(X), np.array(y)

# Training Functions
def train_deep_model(model, train_loader, val_loader, device, num_epochs=50, patience=10):
    """
    Train a deep learning model with early stopping.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Testing data loader
        device: Device to train on ('cuda' or 'cpu')
        num_epochs: Maximum number of epochs to train
        patience: Early stopping patience
    
    Returns:
        Trained model and training history
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # For early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # For tracking metrics
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Print statistics
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history

def evaluate_model(model, test_loader, device, class_names=None):
    """
    Evaluate a trained deep learning model.
    
    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        device: Device to evaluate on
        class_names: List of class names for the confusion matrix
    
    Returns:
        Tuple: (accuracy, classification report, confusion matrix)
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(conf_matrix)
    
    # Plot the confusion matrix if class names are provided
    if class_names is not None:
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()
    
    return accuracy, report, conf_matrix
