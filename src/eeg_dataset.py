"""
Normalization strategy for EEGNet input.

Decision: Per-trial, per-channel z-score normalization. Each trial is normalized
using only its own mean/std across time, per channel. No statistics are computed
across trials, subjects, or folds.

Why not per-subject or per-dataset normalization (i.e. fit mean/std once, on the training data, then apply to both train and val)? Because
we're using a run-based leave-one-run-out split (see split.py): the training set is different for each of the 6 folds. Computing global stats
over training data would mean recomputing them per fold, and any global stat computed from data overlapping with what a different fold considers
validation is a subtle leakage risk. Per-trial normalization sidestep this entirely: it never looks outside the trial being normalized, so it's leakage-free
by construction, at the cost of throwing away between-trial amplitude information (e.g. if a subject's signal drifts to lower amplitude over the session, 
per-trial normalization erases that, an acceptable tradeoff here since CSP+LDA baseline already showed accuracy varies more by subject than by within-session
drift)

Raw MNE is in volts (~1e-5 scale), without this step EEGNet's weights would start updating from gradients computed on a tiny, poorly-scaled input, 
which commonly prevents or slows convergence.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from data_loader import DataLoader as GDFLoader
from split import get_motor_imagery_run_ids

# Maps the dataset's raw event codes to contiguos class indices.
# nn.CrossEntropyLoss requires class labels in [0, num_classes], so 769-772
# (MNE's event codes) must be remmaped, using then directly would either
# chash or silently train against the wrong number of classes
LABEL_REMAP = {769:0, 770:1, 771:2, 772:3}

def zscore_per_trial(epochs_data: np.ndarray) -> np.ndarray:
    """
    Normalization each trial independently, channel by channel, using that
    trial's own mean/std across the time axis.

    Args:
        epochs_data: array of shape (n_trials, n_channels, n_timepoints),
            e.g. from `epochs.get_data()
        
    Returns:
        Normalized array, same shape, each (trial, channel) row with
        mean ~0 and std ~1 across time.
    """

    # keepdims=True so broadcasting works directly againts the (trials, channels, time) array
    mean = epochs_data.mean(axis=2, keepdims=True)
    std = epochs_data.std(axis=2, keepdims=True)

    # Guard against a flat/silent channel producing division by zero
    std = np.where(std < 1e-8, 1e-8, std)

    return (epochs_data - mean) / std

class MotorImageryDataset(Dataset):
    """
    PyTorch Dataset for EEGNet training.

    Reads wideband epochs from data/processed_dl/ (see preprocess_for_deep_learning
    in preprocessing.py), applies label remapping and per-trial normalization, and
    exposes run_ids so this Dataset can be used directly with split.py's run_based_split()
    via a SubsetRandomSampler or index-based Subset.
    """

    def __init__(self, subject_id: int, training: bool = True):
        from pathlib import Path
        import mne
        from artifact import get_artifact_trial_mask

        fif_path = Path("data/processed_dl") / f"subject_{subject_id}_epo.fif"
        if not fif_path.exists():
            raise FileNotFoundError(
                f"{fif_path} not found. Run preprocess_for_deep_learning({subject_id} first.)"
            )

        epochs = mne.read_epochs(fif_path, preload=True, verbose=False)

        # Raw data: (n_trials, n_channels, n_timepoints), in volts
        raw_data = epochs.get_data()

        # Normalization BEFORE converting to tensor, keeps this testable/inspectable
        # as plain numpy in the smoke test below
        normalized_data = zscore_per_trial(raw_data)

        # EEGNet expects (batch, 1, channel, time), add the 1 input channel
        # dimension (analogus to a single-channel image)
        self.X = torch.tensor(normalized_data, dtype=torch.float32).unsqueeze(1)

        # events[:, -1] holds the original 769-772 codes; remap to 0-3
        event_codes = epochs.events[:, -1]
        self.y = torch.tensor([LABEL_REMAP[code] for code in event_codes], dtype=torch.long)

        # Needed so a training script can build run-based folds directly from
        # this Dataset. run_ids is computed from the original raw session, so it
        # must have the same artifact-rejection mask applied that was used when
        # this subject's .fif was generated. Otherwise run_ids (288 entries)
        # won't match self.y (fewer, post-rejection entries)
        loader = GDFLoader()
        raw, _ = loader.load_session(subject_id, training=training)

        run_ids_all_trials = get_motor_imagery_run_ids(raw)
        artifact_mask = get_artifact_trial_mask(raw)
        self.run_ids = run_ids_all_trials[~artifact_mask]

        assert len(self.run_ids) == len(self.y), (
            f"run_ids ({len(self.run_ids)}) and labels ({len(self.y)}) length "
            f"mismatch -- check that this .fif was generated with the same "
            f"artifact-rejection setting this Dataset assumes."
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# test block
if __name__ == "__main__":
    dataset = MotorImageryDataset(subject_id=1)

    print(f"Dataset size: {len(dataset)} trials")
    # 273, not 288: subject 1 has 15 trials flagged with the '1023' artifact
    # annotation, and preprocess_for_deep_learning drops them by default.
    print(f"X shape: {dataset.X.shape}  (expected: [273, 1, 22, ~1126])")
    print(f"y shape: {dataset.y.shape}, unique labels: {dataset.y.unique().tolist()}")
    print(f"run_ids: {len(dataset.run_ids)} values, unique runs: {sorted(set(dataset.run_ids.tolist()))}")

    # Sanity check: confirm normalization actually did what it claims to do.
    # Pick one trial, one channel, check mean/std across time.
    sample_trial, sample_channel = 0, 0
    trial_signal = dataset.X[sample_trial, 0, sample_channel, :]
    print(f"\nSanity check (trial {sample_trial}, channel {sample_channel}):")
    print(f"  mean = {trial_signal.mean().item():.6f}  (expected: ~0.0)")
    print(f"  std  = {trial_signal.std().item():.6f}  (expected: ~1.0)")