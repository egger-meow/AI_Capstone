import os
import argparse
import torch
import pickle
import torchaudio
from chorDataset import create_train_test_dataloaders
from trainTestUtils import CRNN, evaluate_model ,extract_features_for_ml  # for deep model
from settings import sampleRate, modeSelection
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

def test_deep(args):
    """Testing pipeline for a deep learning model."""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sampleRate,
        n_mels=16
    )
    
    # Create test loader
    _, test_dl = create_train_test_dataloaders(
        dataset_dir=args.dataset_path,
        transform=mel_transform,
        train_ratio=0.0,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN(num_classes=6).to(device)
    
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    class_names = ['C','D','E','F','G','A']
    evaluate_model(model, test_dl, device, class_names=class_names)

def test_traditional(args):
    """Testing pipeline for a traditional ML model."""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sampleRate,
        n_mels=16
    )
    # Create test loader (use all data from test folder)
    _, test_dl = create_train_test_dataloaders(
        dataset_dir=args.dataset_path,
        transform=mel_transform,
        train_ratio=0.0,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print("Extracting features for traditional model testing...")
    X_test, y_test = extract_features_for_ml(test_dl)
    
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    with open(args.checkpoint, 'rb') as f:
        clf = pickle.load(f)
    
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, target_names=['C','D','E','F','G','A'])
    cm = confusion_matrix(y_test, preds)
    
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)

def test_main(args):
    if args.model_type == 'deep':
        test_deep(args)
    elif args.model_type == 'traditional':
        test_traditional(args)
    else:
        raise ValueError("Unsupported model_type. Choose 'deep' or 'traditional'.")

def parse_test_args():
    parser = argparse.ArgumentParser(description="Test a chord classification model.")
    parser.add_argument("--dataset_path", type=str, default="C:/Users/inpir/OneDrive/AI_Capstone/HW1/dataset/test",
                        help="Path to the top-level folder with subfolders 0..5 for testing.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/traditional_model.pkl",
                        help="Path to the saved model checkpoint. For traditional models, use a .pkl file.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for testing.")
    parser.add_argument("--model_type", type=str, default=f"{modeSelection}", choices=["deep", "traditional"],
                        help="Model type to use for testing: 'deep' or 'traditional'.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    test_args = parse_test_args()
    test_main(test_args)
