import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import tifffile
from tqdm import tqdm
import torch.nn.functional as F


# ==================================================
# Updated Dataset for Processed Data
# ==================================================
class SarcomaDataset(Dataset):
    def __init__(self, data_paths: List[Tuple[str, int]], transform=None):
        """
        Dataset class for preprocessed sarcoma 3D images
        
        Args:
            data_paths: List of tuples (file_path, label)
            transform: Optional transform to apply to images
        """
        self.data_paths = data_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        file_path, label = self.data_paths[idx]
        
        # Load preprocessed TIFF image (already cropped and resized)
        image = tifffile.imread(file_path)
        
        # Convert to float32
        image = image.astype(np.float32)
        
        # Add channel dimension if needed: (D, H, W) → (1, D, H, W)
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Normalize
        image = (image - image.mean()) / (image.std() + 1e-8)
        
        image = torch.from_numpy(image)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


# ==================================================
# Updated Custom 3D CNN
# ==================================================
class Custom3DCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(Custom3DCNN, self).__init__()
        
        # First conv block
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=20, kernel_size=7)
        self.pool1 = nn.MaxPool3d(kernel_size=5)
        self.bn1 = nn.BatchNorm3d(20)
        self.dropout1 = nn.Dropout3d(0.1)
        
        # Second conv block
        self.conv2 = nn.Conv3d(in_channels=20, out_channels=40, kernel_size=5)
        self.pool2 = nn.MaxPool3d(kernel_size=5)
        self.bn2 = nn.BatchNorm3d(40)
        self.dropout2 = nn.Dropout3d(0.1)
        
        # Third conv block
        self.conv3 = nn.Conv3d(in_channels=40, out_channels=80, kernel_size=3)
        self.bn3 = nn.BatchNorm3d(80)
        self.dropout3 = nn.Dropout3d(0.1)
        
        # Adaptive pooling - handles variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool3d((2, 2, 2))
        
        # Dense layers
        self.fc1 = nn.Linear(80 * 2 * 2 * 2, 256)
        self.dropout4 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # First conv block
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        
        # Second conv block
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        
        # Third conv block
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.dropout3(x)
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = self.fc2(x)
        
        return x


# ==================================================
# Updated Dataset Loading for Processed Data
# ==================================================
def load_sarcoma_dataset(data_dir: str) -> Dict[str, List[Tuple[str, int]]]:
    """
    Load processed dataset and organize by subject ID
    """
    data_root = Path(data_dir) / "sarcoma_3d_processed"  # Updated path
    subject_data = {}
    label_map = {"normal": 0, "tumor": 1}

    if not data_root.exists():
        print(f"ERROR: Processed data not found at {data_root}")
        print("Please run the preprocessing script first!")
        return {}

    for subject_folder in data_root.iterdir():
        if subject_folder.is_dir() and subject_folder.name.startswith("subject_"):
            subject_id = subject_folder.name
            subject_data[subject_id] = []
            print(f"Processing {subject_id}...")

            for class_folder in subject_folder.iterdir():
                if class_folder.is_dir():
                    cname = class_folder.name.lower()
                    if "normal" in cname or "not_tumor" in cname or "non_tumor" in cname:
                        label = label_map["normal"]
                    elif "tumor" in cname:
                        label = label_map["tumor"]
                    else:
                        print(f"  Skipping unknown class: {class_folder.name}")
                        continue

                    # Look for processed TIFF files
                    tiff_files = list(class_folder.glob("processed_*.tiff")) + list(class_folder.glob("processed_*.tif"))
                    print(f"  Found {len(tiff_files)} processed files in {class_folder.name}")
                    for f in tiff_files:
                        subject_data[subject_id].append((str(f), label))

            print(f"  Total samples for {subject_id}: {len(subject_data[subject_id])}")
    
    return subject_data


def create_subject_splits(subject_data: Dict[str, List[Tuple[str, int]]], test_subject: str):
    """
    Create train/test split based on subject ID
    """
    train_paths = []
    test_paths = []
    
    for subject_id, paths in subject_data.items():
        if subject_id == test_subject:
            test_paths.extend(paths)
        else:
            train_paths.extend(paths)
    
    return train_paths, test_paths


# ==================================================
# Training Functions
# ==================================================
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training", ncols=100, leave=False)
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predictions = output.argmax(dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(target.cpu().numpy())
        
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    if len(dataloader) == 0:
        print("Warning: Empty test set, skipping evaluation")
        return 0.0, 0.0, 0.0, 0.0, 0.0, [], []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", ncols=100, leave=False)
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            predictions = output.argmax(dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro', zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1, all_predictions, all_labels


# ==================================================
# Main Function
# ==================================================
def main():
    # Configuration
    config = {
        'data_dir': '/home/haoyang27',
        'model_type': 'custom',
        'num_classes': 2,
        'batch_size': 4,  # Can increase since images are smaller now
        'learning_rate': 1e-4,
        'num_epochs': 20,  # Can train longer since it's faster
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Using device: {config['device']}")
    
    # Load processed dataset
    print("Loading processed dataset...")
    subject_data = load_sarcoma_dataset(config['data_dir'])
    
    if not subject_data:
        print("No data found! Please run preprocessing first.")
        return
    
    subject_ids = sorted(list(subject_data.keys()))  # Sort for consistent ordering
    print(f"Found subjects: {subject_ids}")
    
    # Get example shape from first file
    if subject_data:
        first_subject = list(subject_data.keys())[0]
        first_file = subject_data[first_subject][0][0]
        example_image = tifffile.imread(first_file)
        print(f"Processed image shape: {example_image.shape}")
    
    # Choose which subjects to test on
    print("\nTest subject selection:")
    print("1. Test on all subjects")
    print("2. Test on specific subjects")
    print("3. Test starting from a specific subject")
    
    selection = input("Enter choice (1/2/3): ").strip()
    
    if selection == "2":
        print(f"Available subjects: {subject_ids}")
        test_subjects_input = input("Enter subject IDs to test (comma-separated, e.g., subject_01,subject_03): ").strip()
        test_subject_list = [s.strip() for s in test_subjects_input.split(',')]
        test_subject_list = [s for s in test_subject_list if s in subject_ids]
        if not test_subject_list:
            print("No valid subjects selected, using all subjects")
            test_subject_list = subject_ids
    elif selection == "3":
        print(f"Available subjects: {subject_ids}")
        start_subject = input("Enter starting subject (e.g., subject_02): ").strip()
        if start_subject in subject_ids:
            start_idx = subject_ids.index(start_subject)
            test_subject_list = subject_ids[start_idx:]
            print(f"Will test on: {test_subject_list}")
        else:
            print(f"Subject {start_subject} not found, using all subjects")
            test_subject_list = subject_ids
    else:
        test_subject_list = subject_ids
    
    print(f"\nTesting on {len(test_subject_list)} subjects: {test_subject_list}")
    
    # Results storage
    all_results = {}
    
    # Iterate through each subject as test set
    for test_subject in test_subject_list:
        print(f"\n{'='*60}")
        print(f"Testing on {test_subject}, training on others")
        print(f"{'='*60}")
        
        # Create splits
        train_paths, test_paths = create_subject_splits(subject_data, test_subject)
        print(f"Training samples: {len(train_paths)}")
        print(f"Testing samples: {len(test_paths)}")
        
        # Skip if no test samples
        if len(test_paths) == 0:
            print(f"Warning: No test samples for {test_subject}, skipping...")
            continue
            
        # Skip if no training samples
        if len(train_paths) == 0:
            print(f"Warning: No training samples available, skipping...")
            continue
        
        # Create datasets and dataloaders
        train_dataset = SarcomaDataset(train_paths)
        test_dataset = SarcomaDataset(test_paths)
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                                shuffle=True, num_workers=4)  # Can increase workers
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                               shuffle=False, num_workers=4)
        
        # Create model - no need to specify dimensions, adaptive pooling handles it
        model = Custom3DCNN(num_classes=config['num_classes'])
        model = model.to(config['device'])
        
        # Print model parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        # optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], 
        #                           momentum=0.9, weight_decay=1e-4)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
        # Training history
        history = {
            'train_loss': [],
            'train_acc': []
        }
        
        # Training loop
        best_train_acc = 0
        for epoch in range(config['num_epochs']):
            print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config['device'])
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            
            # Save best model
            if train_acc > best_train_acc:
                best_train_acc = train_acc
        
        # Final evaluation on test set
        print(f"\n{'='*50}")
        print("FINAL TEST EVALUATION")
        print(f"{'='*50}")
        
        test_loss, test_acc, test_prec, test_rec, test_f1, test_preds, test_labels = evaluate(
            model, test_loader, criterion, config['device']
        )
        
        # Update history with test results
        history.update({
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_precision': test_prec,
            'test_recall': test_rec,
            'test_f1': test_f1
        })
        
        # Print results
        if len(test_paths) > 0:
            print(f"\nFinal Results for {test_subject}:")
            print(f"Best Training Accuracy: {best_train_acc:.4f}")
            print(f"Test Accuracy: {test_acc:.4f}")
            print(f"Test Precision: {test_prec:.4f}, Recall: {test_rec:.4f}, F1: {test_f1:.4f}")
            
            # Confusion matrix
            if len(test_labels) > 0:
                cm = confusion_matrix(test_labels, test_preds)
                print(f"Confusion Matrix:\n{cm}")
            else:
                cm = np.zeros((2, 2))
        else:
            print(f"\nNo test samples for {test_subject}")
            cm = np.zeros((2, 2))
            test_acc = test_prec = test_rec = test_f1 = 0.0
        
        # Store results
        all_results[test_subject] = {
            'best_train_acc': best_train_acc,
            'final_test_metrics': {
                'accuracy': test_acc,
                'precision': test_prec,
                'recall': test_rec,
                'f1': test_f1,
                'loss': test_loss
            },
            'confusion_matrix': cm.tolist(),
            'history': history
        }
        
        # Save individual results
        with open(f'training_history_{test_subject}_processed_sarcoma.json', 'w') as f:
            json.dump(history, f, indent=2)

        with open(f'results_{test_subject}_processed_sarcoma.pickle', 'wb') as f:
            pickle.dump(all_results[test_subject], f)
    
    # Save overall results
    with open('all_results_processed_sarcoma.json', 'w') as f:
        json_results = {}
        for k, v in all_results.items():
            json_results[k] = {
                'best_train_acc': v['best_train_acc'],
                'final_test_metrics': v['final_test_metrics'],
                'confusion_matrix': v['confusion_matrix']
            }
        json.dump(json_results, f, indent=2)

    with open('all_results_processed_sarcoma.pickle', 'wb') as f:
        pickle.dump(all_results, f)
    
    # Print overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    
    test_accuracies = [result['final_test_metrics']['accuracy'] for result in all_results.values()]
    if test_accuracies:
        print(f"Mean Test Accuracy: {np.mean(test_accuracies):.4f} ± {np.std(test_accuracies):.4f}")
        print(f"Min Test Accuracy: {np.min(test_accuracies):.4f}")
        print(f"Max Test Accuracy: {np.max(test_accuracies):.4f}")
        
        for subject_id, result in all_results.items():
            print(f"{subject_id}: Train={result['best_train_acc']:.4f}, Test={result['final_test_metrics']['accuracy']:.4f}")


if __name__ == "__main__":
    main()