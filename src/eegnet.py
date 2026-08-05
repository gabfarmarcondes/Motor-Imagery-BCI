"""
EEGNet (Lawhern et al., 2018) implementation adn training script

Runs leave-one-run-out cross-validation (via split.py's run_based_splits) on the wideband pre-processed data
(data/processed_dl), producing results comparable in formato to baseline.py's CSP+LDA output: per-fold accuracy, confusion matrix, saved to figures/.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader, Subset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from eeg_dataset import MotorImageryDataset
from split import run_based_splits

class EEGNet(nn.Module):
    """
    PyTorch reimplementation of EEGNet (Lawhern et al., 2018, "EEGNet: A Compact Convolutional Network for EEG-based BCIs").

    Expects input shaped (batch, 1, n_channels, n_timepoints) matches what MotorImageryDataset already produces.
    """

    def __init__(self, n_channels=22, n_timepoints=1126, n_classes=4,
                 F1=8, D=2, F2=None, kernel_length=125, dropout_rate=0.5):
        super().__init__()
        F2 = F2 or F1 * D

        # Block 1: temporal filtering, then spatial (per-channel) filtering
        self.block1_conv = nn.Conv2d(1, F1, (1, kernel_length), padding='same', bias=False)
        self.block1_bn1 = nn.BatchNorm2d(F1)

        #Depthwise conv: one spatial filter (across all 22 channels) per temporal
        # filter (groups=F1). No padding, this collapses the channel dimension.
        # from 22 down to 1 on purpose, it's the learned analogue of CSP
        self.depthwise_conv = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.block1_bn2 = nn.BatchNorm2d(F1 * D)
        self.block1_pool = nn.AvgPool2d((1, 4))
        self.block1_drop = nn.Dropout(dropout_rate)

        # Block 2: separable conv (depthwise temporal + pointwise channel mix)
        # Fewer parameters than a regular conv here. Important given how little
        # data we have per subject (~240 training trials per fold)
        self.sep_depthwise = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding='same', groups=F1 * D, bias=False)
        self.sep_pointwise = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.block2_bn = nn.BatchNorm2d(F2)
        self.block2_pool = nn.AvgPool2d((1, 8))
        self.block2_drop = nn.Dropout(dropout_rate)

        self.elu = nn.ELU()

        # Flattened feature size depends on n_timepoints via the two pooling layers.
        # Computed once with a dummy forward pass instead of hardcoding a number
        # that would silently go stale if n_timepoints ever changes.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_timepoints)
            flattened_size = self._forward_features(dummy).view(1, -1).shape[1]

        self.classifier = nn.Linear(flattened_size, n_classes)

    def _forward_features(self, x):
        x = self.block1_conv(x)
        x = self.block1_bn1(x)
        x = self.depthwise_conv(x)
        x = self.block1_bn2(x)
        x = self.elu(x)
        x = self.block1_pool(x)
        x = self.block1_drop(x)

        x = self.sep_depthwise(x)
        x = self.sep_pointwise(x)
        x = self.block2_bn(x)
        x = self.elu(x)
        x = self.block2_pool(x)
        x = self.block2_drop(x)
        return x

    def forward(self, x):
        # Returns raw logits (no softmax); nn.CrossEntropyLoss applies it internally
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def apply_max_norm_constraint(module: nn.Module, max_norm:float):
    """
    Manually enforces a max-norm weight constraint, matching the original KEras EEGNet's kernel_constraint=max_norm(...). Pytorch optimizers don't support
    this natively. It must be called by hand after every optimizer.step().

    Rescales each output filter's weights so their L2 norm doesn't exceed max_norm; filters already within the limit are left untouched.
    """
    with torch.no_grad():
        w = module.weight
        norm = w.norm(p=2, dim=tuple(range(1, w.dim())), keepdim=True)
        desired = torch.clamp(norm, max=max_norm)
        w *= desired / (norm + 1e-8)

def train_one_fold(model, train_loader, val_loader, device, epochs=150, lr=1e-3, patience=20):
    """
    Trains model, evaluating on val_loader, every epoch. Keeps the weights from whichever
    epoch had the best validation accracy, and stops early if that hasn't improved
    for patience epochs. With ~240 training trials, running a fixed large number of epochs
    risks overfitting well past the useful point.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            # Applied after optimizer.step(), corrects whatever the gradient
            # update just did, not before
            apply_max_norm_constraint(model.depthwise_conv, max_norm=1.0)
            apply_max_norm_constraint(model.classifier, max_norm=0.25)

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)
        val_acc, _, _ = evaluate(model, val_loader, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"    epoch {epoch:3d}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}")

        if epochs_without_improvement >= patience:
            print(f"    early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    model.load_state_dict(best_state)
    return model, best_val_acc

def evaluate(model, data_loader, device):
    # Returns (accuracy, y_true, y_pred) for data_loader
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = outputs.argmax(dim=1).cpu()
            y_true.extend(y_batch.tolist())
            y_pred.extend(preds.tolist())

    accuracy = float(np.mean(np.array(y_true) == np.array(y_pred)))
    return accuracy, y_true, y_pred

def run_eegnet_cv(subject_id: int, epochs=150, batch_size=32, device=None, lr=1e-3, patience=20):
    """
    Runs leave-one-run-out cross-validation for subject_id, mirroring
    baseline.py's role but with EEGNet + the run-based split instead of CSP+LDA + random ShuffleSplit
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = MotorImageryDataset(subject_id=subject_id)
    run_ids = dataset.run_ids

    fold_accuracies = []
    all_y_true, all_y_pred = [], []

    for fold, (train_idx, val_idx) in enumerate(run_based_splits(run_ids)):
        print(f"\n  Fold {fold} (val run {np.unique(run_ids[val_idx]).tolist()})")

        train_loader = TorchDataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
        val_loader = TorchDataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)

        # Fresh model per fold. Folds must stay independent
        model = EEGNet(n_channels=22, n_timepoints=dataset.X.shape[-1], n_classes=4)
        model, val_acc = train_one_fold(model, train_loader, val_loader, device, epochs=epochs, lr=lr, patience=patience)

        _, y_true, y_pred = evaluate(model, val_loader, device)
        fold_accuracies.append(val_acc)
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

        print(f"  Fold {fold} best val_acc: {val_acc:.4f}")

    mean_acc = float(np.mean(fold_accuracies))
    std_acc = float(np.std(fold_accuracies))

    save_confusion_matrix(all_y_true, all_y_pred, subject_id)

    return mean_acc, std_acc, fold_accuracies


def save_confusion_matrix(y_true, y_pred, subject_id):
    # Saves a configuration matrix figure, same convention as baseline.py
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)

    labels = ["Left Hand", "Right Hand", "Foot", "Tongue"]
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"EEGNet - Subject {subject_id} (leave-one-run-out), pooled folds")
    plt.tight_layout()

    save_path = figures_dir / f"eegnet_confusion_matrix_subject_{subject_id}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"  Confusion matrix saved to: {save_path}")


# test block
if __name__ == "__main__":
    try:
        mean_acc, std_acc, fold_accuracies = run_eegnet_cv(subject_id=1, epochs=150)

        print("\n--- EEGNet Summary (Subject 1) ---")
        print(f"Per-fold accuracies: {[f'{a:.4f}' for a in fold_accuracies]}")
        print(f"Mean accuracy: {mean_acc:.2%}  (std: {std_acc:.2%})")
        print("CSP+LDA baseline for subject 1 was ~70.34% (see results/baseline_results.csv)")

    except Exception as e:
        print(f"Error: {e}")