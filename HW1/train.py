import os
import torch
import argparse
import pickle
from torch import nn, optim
import torchaudio
from chorDataset import create_train_test_dataloaders   # adjust as needed
from trainTestUtils import train_deep_model, evaluate_model, extract_features_for_ml, CRNN  # deep model utilities
from settings import sampleRate, modeSelection

# Traditional ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

def train_deep(args):
    """Training pipeline for a deep learning model."""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sampleRate, 
        n_mels=16
    )
    train_dl, val_dl = create_train_test_dataloaders(
        dataset_dir=args.dataset_path,
        transform=mel_transform,
        train_ratio=0.80,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        apply_balancing='smote'
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN(num_classes=6).to(device)
    
    model, history = train_deep_model(
        model=model,
        train_loader=train_dl,
        val_loader=val_dl,
        device=device,
        num_epochs=args.num_epochs,
        patience=args.patience
    )
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, "best_deep_model.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Deep model saved to {checkpoint_path}")
    
    if args.evaluate_after_train:
        class_names = ['C','D','E','F','G','A']
        accuracy, report, conf_matrix = evaluate_model(model, val_dl, device, class_names=class_names)
        print("Evaluation done after training.")

def train_traditional(args):
    """Training pipeline for a traditional ML model."""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sampleRate,
        n_mels=16
    )
    # Use all available data for feature extraction in training
    train_dl, _ = create_train_test_dataloaders(
        dataset_dir=args.dataset_path,
        transform=mel_transform,
        train_ratio=1.0,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print("Extracting features for traditional ML model...")
    X_train, y_train = extract_features_for_ml(train_dl)
    
    # Select classifier based on argument
    if args.traditional_model == "random_forest":
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    elif args.traditional_model == "gradient_boosting":
        clf = GradientBoostingClassifier(random_state=42)
    elif args.traditional_model == "svm":
        clf = SVC(random_state=42)
    else:
        raise ValueError("Unsupported traditional model type.")
    
    clf.fit(X_train, y_train)
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    model_path = os.path.join(args.checkpoint_dir, "traditional_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"Traditional ML model saved to {model_path}")
    
    preds = clf.predict(X_train)
    print("Training performance:")
    print(classification_report(y_train, preds, target_names=['C','D','E','F','G','A']))

def train_main(args):
    if args.model_type == 'deep':
        train_deep(args)
    elif args.model_type == 'traditional':
        train_traditional(args)
    else:
        raise ValueError("Unsupported model_type. Choose 'deep' or 'traditional'.")

def parse_train_args():
    parser = argparse.ArgumentParser(description="Train a chord classification model.")
    parser.add_argument("--dataset_path", type=str, default="C:/Users/inpir/OneDrive/AI_Capstone/HW1/dataset/train",
                        help="Path to the top-level folder with subfolders 0..5 for training.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Where to save the model checkpoint.")
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training.")
    parser.add_argument("--patience", type=int, default=100,
                        help="Patience for early stopping.")
    parser.add_argument("--evaluate_after_train", action="store_true",
                        help="Evaluate on validation set immediately after training.")
    parser.add_argument("--model_type", type=str, default=f"{modeSelection}", choices=["deep", "traditional"],
                        help="Model type to use: 'deep' or 'traditional'.")
    parser.add_argument("--traditional_model", type=str, default="random_forest", choices=["random_forest", "gradient_boosting", "svm"],
                        help="For traditional models: choose among 'random_forest', 'gradient_boosting', or 'svm'.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    train_args = parse_train_args()
    train_main(train_args)
