"""
Final per-subject model training + persistence. Produces the trained models the real-time PyGame interface will use

Unlike run_eegnet_cv, this trains one model per subject and saves it to disk.

Split choice: 5 runs for training, the last run held out for validation. Rationale:
- Keeps the run-based split that split.py validated (no within-run leakage).
- Using the most recent run as validation mimics the real-time game setup: the model learns from
    past data and judged on what comes next.

The held-out run drives early stopping and provides the reported accuracy. The number the game will
actually deliver, which is more honest than the 6-fold cv mean.

patience=40 is used, but early stopping stays honest: it stops when this model stops improving
on the held-out validation run, not blindly.
"""

import random
from pathlib import Path

import numpy as np
import torch

from eeg_dataset import MotorImageryDataset
from eegnet import EEGNet, train_one_fold
from torch.utils.data import DataLoader as TorchDataLoader, Subset


def set_seed(seed=42):
    """
    Fixes every RNG this training path draws from, so re-training produces the
    same model and the same held-out accuracy instead of a new one each run.

    EEGNet has two sources of randomness: weight initialization (torch) and the
    training DataLoader's shuffle (also torch). numpy and random are seeded too
    because the dataset/split code touches them, and leaving either free would
    make "the same seed" only partly true.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# Metadata saved alongside weights so a model can be rebuilt and used without
# re-deriving these from code that might change later.
MODEL_META = {
    "n_channels": 22,
    "n_classes": 4,
    "preprocessing": "wideband 4-40Hz, per-trial z-score, artifact-rejected",
    "label_map": {0: "Left Hand", 1: "Right Hand", 2: "Foot", 3: "Tongue"}
}

def train_final_model(subject_id, val_run=5, patience=40, epochs=300, batch_size=32, device=None):
    """
    Trains one EEGNet's subject_id, holding out val_run for validation/early stopping,
    and returns the trained model + its held-out accuracy.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MotorImageryDataset(subject_id=subject_id)
    run_ids = dataset.run_ids

    # Everything that isn't the held-out is training data.
    val_idx = np.where(run_ids == val_run)[0]
    train_idx = np.where(run_ids != val_run)[0]

    if len(val_idx) == 0:
        raise ValueError(f"val_run {val_run} not found in subject {subject_id}'s runs "
                         f"({sorted(set(run_ids.tolist()))}).")

    train_loader = TorchDataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = TorchDataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)

    model = EEGNet(n_channels=22, n_timepoints=dataset.X.shape[-1], n_classes=4)
    model, val_acc = train_one_fold(model, train_loader, val_loader, device, epochs=epochs, patience=patience)

    return model, val_acc, dataset.X.shape[-1]


def save_model(model, subject_id, val_acc, n_timepoints):
    # Sves weights + metadata to models/subject_{id}_eegnet.pt
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    save_path = models_dir / f"subject_{subject_id}_eegnet.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "n_timepoints": n_timepoints,
        "val_accuracy": val_acc,
        "subject_id": subject_id,
        **MODEL_META,
    }, save_path)

    return save_path


def load_model(subject_id, device=None):
    """
    Reloads a saved model for inference. Rebuilds the architecture from the saved metadata, loads the weights, and puts it in eval mode.
    Ready for the game to call on single trials.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_path = Path("models") / f"subject_{subject_id}_eegnet.pt"
    if not load_path.exists():
        raise FileNotFoundError(f"{load_path} not found. Run train_final_model first.")

    checkpoint = torch.load(load_path, map_location=device, weights_only=False)

    model = EEGNet(
        n_channels=checkpoint["n_channels"],
        n_timepoints=checkpoint["n_timepoints"],
        n_classes=checkpoint["n_classes"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


def train_all_final_models(patience=40, epochs=300):
    # Trains and saves a final model for every subject, reporting held-out accuracy
    results = []

    for subject_id in range(1, 10):
        print(f"\n{'='*50}")
        print(f"Final model: subject {subject_id}")
        print(f"{'='*50}")

        try:
            model, val_acc, n_timepoints = train_final_model(subject_id, patience=patience, epochs=epochs)
            save_path = save_model(model, subject_id, val_acc, n_timepoints)
            results.append({"subject": subject_id, "val_accuracy": val_acc})
            print(f"\nSubject {subject_id}: held-out val_acc={val_acc:.2f}, saved to {save_path}")
        except Exception as e:
            print(f"Failed subject: {subject_id}: {e}")

    return results


def collect_results_from_checkpoints(subject_ids=range(1, 10)):
    """
    Rebuilds the results table by reading val_accuracy back out of each saved
    .pt, rather than from the in-memory value returned by training.

    This is what keeps final_model_results.csv honest: the CSV then describes
    exactly the models sitting in models/, so it can never end up reporting one
    run's numbers next to another run's weights.
    """
    rows = []

    for subject_id in subject_ids:
        load_path = Path("models") / f"subject_{subject_id}_eegnet.pt"
        if not load_path.exists():
            print(f"No checkpoint for subject {subject_id}, skipping ({load_path} missing).")
            continue

        checkpoint = torch.load(load_path, map_location="cpu", weights_only=False)
        rows.append({
            "subject": checkpoint["subject_id"],
            "val_accuracy": checkpoint["val_accuracy"],
        })

    return rows


if __name__ == "__main__":
    import pandas as pd

    # Seed before anything trains, so this whole run is reproducible.
    set_seed(42)

    train_all_final_models(patience=40, epochs=300)

    # Read the accuracies back off disk instead of reusing the in-memory ones.
    results = collect_results_from_checkpoints()

    df = pd.DataFrame(results)
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    save_path = results_dir / "final_model_results.csv"
    df.to_csv(save_path, index=False)

    print("\n--- Final Per-Subject Model Accuracy (held-out run) ---")
    print(df.to_string(index=False))
    print(f"\nMean held-out accuracy: {df['val_accuracy'].mean():.2%}")
    print(f"Results saved to: {save_path}")

    # Sanity check: reload subject 1's model and confirm it runs inference.
    print("\n--- Reload sanity check (subject 1) ---")
    model, checkpoint = load_model(1)
    dataset = MotorImageryDataset(subject_id=1)
    sample = dataset.X[0:1].to(next(model.parameters()).device)  # one trial, shape [1, 1, 22, T]
    with torch.no_grad():
        pred = model(sample).argmax(dim=1).item()
    print(f"Loaded model (saved val_acc {checkpoint['val_accuracy']:.2%}) "
          f"predicted class {pred} ({checkpoint['label_map'][pred]}) for trial 0.")