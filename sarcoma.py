import os
import json
import argparse
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import glob
from torchvision.models import convnext_base, swin_b
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
import time


class BinaryImageDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, model_name=None):
        self.data = df.reset_index(drop=True)
        self.image_dir = image_dir
        

        if transform is not None:
            self.transform = transform

        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        filename = row['filename']
        
        # Recursively search for the file inside image_dir
        found = glob.glob(os.path.join(self.image_dir, "**", filename), recursive=True)
        if not found:
            raise FileNotFoundError(f"Image not found: {filename}")
        
        image = Image.open(found[0]).convert('RGB')
        label = int(row['Label'])
        return self.transform(image), label


def load_fold_data(label_csv, image_dir, test_fold_idx, model_name=None):
    df = pd.read_csv(label_csv)

    test_mask = df['filename'].str.lower().str.contains(f"_s{test_fold_idx}_")
    test_df = df[test_mask].copy()
    train_df = df[~test_mask].copy()

    if test_df.empty:
        raise ValueError(f"[Fold {test_fold_idx}] Test set is empty! Check filename pattern or fold naming.")

    print(f"[Fold {test_fold_idx}] Sample test images:")
    print(test_df['filename'].head(2).tolist())

    train_dataset = BinaryImageDataset(train_df, image_dir, model_name=model_name)
    test_dataset = BinaryImageDataset(test_df, image_dir, model_name=model_name)
    return train_dataset, test_dataset


def build_convnext_binary():
    model = convnext_base(pretrained=True)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 1)
    return model


def build_swin_binary():
    model = swin_b(pretrained=True)
    model.head = nn.Linear(model.head.in_features, 1)
    return model



def train_model(model, train_loader, test_loader, device, fold, model_name, output_dir, epochs=10, lr=1e-4):
    """
    Train model and save all necessary outputs for evaluation.
    
    Saves:
    - Model checkpoint (.pt)
    - Predictions with probabilities (.csv)
    - Confusion matrix (.json)
    - Test time (.json)
    """
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for images, labels in loop:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.sigmoid(logits) > 0.5
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=loss.item())

        avg_loss = train_loss / len(train_loader)
        acc = correct / total * 100
        print(f"✅ Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f} | Train Acc: {acc:.2f}%")

    # Save model checkpoint
    checkpoint_path = os.path.join(output_dir, f"fold{fold}_checkpoint.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"💾 Saved model checkpoint to {checkpoint_path}")

    # Evaluation on test set with timing
    model.eval()
    preds, gts, probs = [], [], []
    eval_loop = tqdm(test_loader, desc=f"Evaluating Fold {fold}", leave=False)
    
    # Start timing total test evaluation (includes data loading)
    test_start_time = time.time()
    
    # Track pure inference time (only model forward pass)
    pure_inference_time = 0.0
    
    with torch.no_grad():
        for images, labels in eval_loop:
            images = images.to(device)
            
            # Time only the model forward pass
            inference_start = time.time()
            logits = model(images)
            torch.cuda.synchronize()  # Ensure GPU computation is complete
            inference_end = time.time()
            pure_inference_time += (inference_end - inference_start)
            
            sigmoid_probs = torch.sigmoid(logits).cpu().numpy().flatten()
            
            prob_1 = sigmoid_probs
            prob_0 = 1 - sigmoid_probs
            pred = (prob_1 > 0.5).astype(int)

            preds.extend(pred)
            gts.extend(labels.numpy())
            probs.extend(zip(prob_0, prob_1))
    
    # End timing total test evaluation
    test_end_time = time.time()
    total_test_time_seconds = test_end_time - test_start_time

    # Save test time metrics
    time_dict = {
        "total_test_time_seconds": total_test_time_seconds,
        "pure_inference_time_seconds": pure_inference_time,
        "data_loading_overhead_seconds": total_test_time_seconds - pure_inference_time,
        "test_samples": len(gts),
        "total_time_per_sample_ms": (total_test_time_seconds / len(gts)) * 1000 if len(gts) > 0 else 0,
        "pure_inference_per_sample_ms": (pure_inference_time / len(gts)) * 1000 if len(gts) > 0 else 0,
        "batch_size": test_loader.batch_size
    }
    
    time_path = os.path.join(output_dir, f"fold{fold}_test_time.json")
    with open(time_path, 'w') as f:
        json.dump(time_dict, f, indent=2)
    print(f"⏱️  Total test time: {total_test_time_seconds:.2f}s ({time_dict['total_time_per_sample_ms']:.2f}ms per sample)")
    print(f"⏱️  Pure inference time: {pure_inference_time:.2f}s ({time_dict['pure_inference_per_sample_ms']:.2f}ms per sample)")
    print(f"⏱️  Data loading overhead: {time_dict['data_loading_overhead_seconds']:.2f}s")

    # Save predictions with probabilities (for ROC curves)
    prob_df = pd.DataFrame(probs, columns=["prob_class_0", "prob_class_1"])
    prob_df["true_label"] = gts
    prob_df["predicted_label"] = preds
    
    probs_path = os.path.join(output_dir, f"fold{fold}_predictions.csv")
    prob_df.to_csv(probs_path, index=False)
    print(f"📁 Saved predictions to {probs_path}")

    # Calculate and save confusion matrix
    cm = confusion_matrix(gts, preds)
    cm_dict = {
        "confusion_matrix": cm.tolist(),
        "true_negatives": int(cm[0, 0]),
        "false_positives": int(cm[0, 1]),
        "false_negatives": int(cm[1, 0]),
        "true_positives": int(cm[1, 1])
    }
    
    cm_path = os.path.join(output_dir, f"fold{fold}_confusion_matrix.json")
    with open(cm_path, 'w') as f:
        json.dump(cm_dict, f, indent=2)
    print(f"📁 Saved confusion matrix to {cm_path}")

    # Calculate metrics
    accuracy = accuracy_score(gts, preds)
    
    return {
        "fold": fold,
        "accuracy": accuracy,
        "test_samples": len(gts),
        "total_test_time_seconds": total_test_time_seconds,
        "pure_inference_time_seconds": pure_inference_time
    }


def run_cross_testing(model_name, modality, epochs=10, lr=1e-4, batch_size=32):
    """
    Run 5-fold cross-validation with organized output structure.
    
    Args:
        model_name: 'convnext', 'swin',
        modality: 'opticaxis', 'intensity', etc.
        epochs: number of training epochs per fold
        lr: learning rate
        batch_size: batch size for training and testing
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define paths
    label_csv = f"/scratch/cui0011/labels_{modality}.csv"
    image_dir = f"/scratch/cui0011/sarcoma/sarcoma_{modality}"
    
    # Create organized output directory
    base_output_dir = f"/scratch/cui0011/sarcoma_revision_incepttorch/{model_name}_{modality}"
    os.makedirs(base_output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Running {model_name.upper()} on {modality.upper()}")
    print(f"Epochs: {epochs} | LR: {lr} | Batch size: {batch_size}")
    print(f"Output directory: {base_output_dir}")
    print(f"{'='*60}\n")
    
    metrics = []

    for fold in range(1, 6):  # s1 to s5
        print(f"\n🔁 Fold {fold}/5")
        
        # Create fold-specific output directory
        fold_output_dir = os.path.join(base_output_dir, f"fold{fold}")
        os.makedirs(fold_output_dir, exist_ok=True)
        
        # Load data
        train_dataset, test_dataset = load_fold_data(label_csv, image_dir, test_fold_idx=fold, model_name=model_name)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

        # Build model
        if model_name == 'convnext':
            model = build_convnext_binary()
        elif model_name == 'swin':
            model = build_swin_binary()

        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Train and evaluate
        result = train_model(
            model, train_loader, test_loader, device, 
            fold=fold, model_name=model_name, output_dir=fold_output_dir, epochs=epochs, lr=lr
        )
        
        print(f"Fold {fold} Results: {result}")
        metrics.append(result)

    # Save overall summary
    results_df = pd.DataFrame(metrics)
    summary_path = os.path.join(base_output_dir, "cross_validation_summary.csv")
    results_df.to_csv(summary_path, index=False)
    
    # Calculate and save aggregated statistics
    stats = {
        "model": model_name,
        "modality": modality,
        "num_folds": len(metrics),
        "mean_accuracy": float(results_df["accuracy"].mean()),
        "std_accuracy": float(results_df["accuracy"].std()),
        "min_accuracy": float(results_df["accuracy"].min()),
        "max_accuracy": float(results_df["accuracy"].max()),
        "mean_total_test_time_seconds": float(results_df["total_test_time_seconds"].mean()),
        "total_total_test_time_seconds": float(results_df["total_test_time_seconds"].sum()),
        "mean_pure_inference_time_seconds": float(results_df["pure_inference_time_seconds"].mean()),
        "total_pure_inference_time_seconds": float(results_df["pure_inference_time_seconds"].sum())
    }
    
    stats_path = os.path.join(base_output_dir, "statistics.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"📊 Cross-Validation Summary:")
    print(f"Mean Accuracy: {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f}")
    print(f"Min/Max: {stats['min_accuracy']:.4f} / {stats['max_accuracy']:.4f}")
    print(f"Mean Total Test Time: {stats['mean_total_test_time_seconds']:.2f}s")
    print(f"Mean Pure Inference Time: {stats['mean_pure_inference_time_seconds']:.2f}s")
    print(f"Total Test Time: {stats['total_total_test_time_seconds']:.2f}s")
    print(f"Total Inference Time: {stats['total_pure_inference_time_seconds']:.2f}s")
    print(f"\n📁 All results saved to: {base_output_dir}")
    print(f"{'='*60}\n")
    
    return metrics


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train ConvNeXt, Swin models on sarcoma binary classification')
    parser.add_argument('--model', type=str, required=True, choices=['convnext', 'swin'],
                        help='Model name: convnext or swin')
    parser.add_argument('--modality', type=str, required=True,
                        help='Modality name (e.g., opticaxis, intensity, dopu, retardation)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs per fold (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    args = parser.parse_args()

    print("\n" + "="*60)
    print(f"STARTING {args.model.upper()} TRAINING - {args.modality.upper()}")
    print("="*60)
    results = run_cross_testing(
        model_name=args.model,
        modality=args.modality,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size
    )

    print("\n" + "="*60)
    print("✅ ALL TRAINING COMPLETE")
    print("="*60)
