import os
import glob
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
from settings import sampleRate, samplingStrategy
import torch.nn.functional as F
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from collections import Counter

def collate_fn(batch):
    waveforms, labels = zip(*batch)
    
    # For MelSpectrogram output: [channels, n_mels, time]
    # Determine the maximum time length in the batch
    max_length = max(w.shape[2] for w in waveforms)  # Use the last dimension (time)
    
    padded_waveforms = []
    for w in waveforms:
        pad_amount = max_length - w.shape[2]
        # Pad along the time dimension (last dimension); (left_pad, right_pad)
        padded = F.pad(w, (0, pad_amount), "constant", 0)
        padded_waveforms.append(padded)
    
    padded_waveforms = torch.stack(padded_waveforms, dim=0)
    labels_tensor = torch.tensor(labels)
    
    return padded_waveforms, labels_tensor

def isNumber(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
    
class ChordDataset(Dataset):
    """
    A PyTorch Dataset that loads chord audio data from folders labeled 0..5.
    Each folder's name is interpreted as the class label.
    """
    def __init__(self, root_dir, transform=None):
        """
        :param root_dir: Path to the directory containing subfolders 0..5.
        :param transform: Optional torchaudio or custom transform.
        """
        self.root_dir = root_dir
        self.transform = transform

        # Gather all (file_path, label) pairs
        self.samples = []
        # We expect subfolders named 0,1,2,3,4,5 each containing .wav files
        for label_str in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label_str)
            if not os.path.isdir(label_dir) or not isNumber(label_str):
                continue
            # Attempt to convert folder name to int label
            label = int(label_str)  

            # Collect all .wav files in this subfolder
            file_paths = glob.glob(os.path.join(label_dir, "*.wav"))
            for fp in file_paths:
                self.samples.append((fp, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        # Load audio with torchaudio; shape = [channels, num_frames]
        waveform, sample_rate = torchaudio.load(file_path)  

        if self.transform is not None:
            waveform = self.transform(waveform)

        # Return (audio_tensor, label)
        return waveform, label


def create_train_test_dataloaders(
    dataset_dir = "C:/Users/inpir/OneDrive/AI_Capstone/HW1/datasets",
    transform=None,
    train_ratio=0.8,
    batch_size=16,
    shuffle=True,
    num_workers=0,
    apply_balancing=None  # Options: None, 'smote', 'adasyn', 'random_oversample'
):
    """
    Utility function to create train/test DataLoaders from the ChordDataset.
    Optionally, apply balancing (oversampling) to both train and test sets.
    """
    full_dataset = ChordDataset(root_dir=dataset_dir, transform=transform)
    dataset_size = len(full_dataset)

    # Compute split lengths
    train_size = int(train_ratio * dataset_size)
    test_size = dataset_size - train_size

    # Shuffle once if requested
    if shuffle:
        indices = torch.randperm(dataset_size)
    else:
        indices = torch.arange(dataset_size)

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    test_subset = torch.utils.data.Subset(full_dataset, test_indices)
    
    # Standard dataloader creation function for a given subset
    def create_balanced_loader(subset, target_samples):
        loader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle, 
                            num_workers=num_workers, collate_fn=collate_fn)
        if apply_balancing and apply_balancing in ['smote', 'adasyn', 'random_oversample']:
            # Load all data from the subset and find maximum feature length
            all_waveforms = []
            all_labels = []
            max_feature_length = 0
            for i in range(len(subset)):
                waveform, label = subset[i]
                if isinstance(waveform, torch.Tensor):
                    waveform_np = waveform.numpy()
                else:
                    waveform_np = np.array(waveform)
                feature_length = np.prod(waveform_np.shape)
                max_feature_length = max(max_feature_length, feature_length)
                all_waveforms.append(waveform_np)
                all_labels.append(label)
            X = []
            y = all_labels
            for waveform in all_waveforms:
                flattened = waveform.flatten()
                if len(flattened) < max_feature_length:
                    padded = np.pad(flattened, (0, max_feature_length - len(flattened)), 'constant')
                    X.append(padded)
                else:
                    X.append(flattened)
            X = np.array(X)
            y = np.array(y)
            print("Class distribution before balancing:", Counter(y))
            try:
                # Determine sampling strategy
                if apply_balancing == 'smote':
                    if isinstance(target_samples, dict):
                        sampling_strategy = target_samples
                    elif isinstance(target_samples, (int, float)):
                        sampling_strategy = {cls: target_samples for cls in np.unique(y)}
                    else:
                        sampling_strategy = 1.0
                    sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=3)
                elif apply_balancing == 'adasyn':
                    sampler = ADASYN(random_state=42)
                elif apply_balancing == 'random_oversample':
                    if target_samples:
                        sampling_strategy = {cls: target_samples for cls in np.unique(y)}
                        sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
                    else:
                        sampler = RandomOverSampler(random_state=42)
                # Apply resampling
                X_resampled, y_resampled = sampler.fit_resample(X, y)
                print("Class distribution after balancing:", Counter(y_resampled))
                # Create a custom dataset from the resampled data
                balanced_dataset = BalancedFlatDataset(
                    X_resampled, 
                    y_resampled, 
                    reference_shape=all_waveforms[0].shape
                )
                loader = DataLoader(balanced_dataset, batch_size=batch_size, shuffle=shuffle, 
                                    num_workers=num_workers, collate_fn=collate_fn)
            except Exception as e:
                print(f"Error applying balancing technique: {e}")
                print("Falling back to original unbalanced subset.")
                loader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle, 
                                    num_workers=num_workers, collate_fn=collate_fn)
        return loader

    # Create loaders for train and test subsets (balanced if requested)
    def split_sampling_strategy(strategy_dict, factor):
        """
        Splits a sampling strategy dictionary into two dictionaries based on a factor.
        
        Args:
            strategy_dict (dict): A dictionary where keys are class labels and values are integers.
            factor (float): A float between 0 and 1. Each value in strategy_dict will be split into:
                            - majority: value * factor
                            - minority: value * (1 - factor)
                            
        Returns:
            tuple: Two dictionaries (majority_dict, minority_dict) with integer values.
        """
        if not (0 <= factor <= 1):
            raise ValueError("Factor must be between 0 and 1.")
            
        majority = {}
        minority = {}
        
        for cls, count in strategy_dict.items():
            # Compute majority and minority counts, rounding to integers
            majority[cls] = int(round(count * factor))
            minority[cls] = int(round(count * (1 - factor)))
            
        return majority, minority

    train_target_samples, test_target_samples = split_sampling_strategy(samplingStrategy, train_ratio)
    train_loader = create_balanced_loader(train_subset, train_target_samples)
    test_loader = create_balanced_loader(test_subset, test_target_samples)

    return train_loader, test_loader


class BalancedFlatDataset(Dataset):
    """
    A PyTorch Dataset that wraps the resampled data after applying SMOTE or other techniques.
    """
    def __init__(self, X, y, reference_shape):
        """
        :param X: Resampled features (numpy array)
        :param y: Resampled labels (numpy array)
        :param reference_shape: Reference shape to reshape features
        """
        self.X = X
        self.y = y
        self.reference_shape = reference_shape
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # Get the feature and label
        flat_feature = self.X[idx]
        label = self.y[idx]
        
        # Reshape to match the reference shape (but it might be padded)
        # So we'll take just enough values to fill the reference shape
        needed_values = np.prod(self.reference_shape)
        relevant_feature = flat_feature[:needed_values]
        
        try:
            # Try to reshape to the reference shape
            reshaped_feature = relevant_feature.reshape(self.reference_shape)
        except ValueError:
            reshaped_feature = np.zeros(self.reference_shape)
            flat_reshaped = reshaped_feature.flatten()
            flat_reshaped[:len(relevant_feature)] = relevant_feature
            reshaped_feature = flat_reshaped.reshape(self.reference_shape)
        
        # Convert back to torch tensor
        feature_tensor = torch.from_numpy(reshaped_feature).float()
        
        return feature_tensor, label

if __name__ == "__main__":
    # Example usage:
    dataset_path = "C:/Users/inpir/OneDrive/AI_Capstone/HW1/datasets"  
    
    # Define a MelSpectrogram transform
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sampleRate,    
        n_mels=16
    )

    # Create dataloaders with SMOTE balancing
    train_dl, test_dl = create_train_test_dataloaders(
        dataset_dir=dataset_path,
        transform=mel_transform,
        train_ratio=0.8,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        apply_balancing='smote',  # Options: None, 'smote', 'adasyn', 'random_oversample'
        target_samples=None       # Set a number or None for auto-balancing
    )

    # print(len(train_dl))
    # Quick test
    for batch in train_dl:
        waveforms, labels = batch
        print("Waveforms shape:", waveforms.shape)
        print("Labels:", labels)
        print("Class distribution in batch:", Counter(labels.numpy()))
        break