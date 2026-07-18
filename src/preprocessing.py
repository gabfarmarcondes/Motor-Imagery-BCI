import mne
import numpy as np
from pathlib import Path
from data_loader import DataLoader

# Two preprocessing "profiles" are used in this project:
# - CSP/LDA baseline: narrow band (8-30Hz), matches the mu/beta rhythms CSP relies on.
# - Deep learning (EEGNet): wider band (4-40Hz), so the network can learn its own
#   temporal filters instead of being fed data that's already been band-limited for it.
#
# Both profiles run through the SAME pipeline (notch -> bandpass -> epoch), just with
# different filter bounds and different output folders. This means the CSP baseline
# cache in data/processed/ is never touched by anything done for the deep learning path.


def preprocessing_subject(
        subject_id: int, 
        training: bool = True,
        l_freq: float = 8.0,
        h_freq: float = 30.0,
        output_dir: str = "data/processed"
        ):
    """
    Loads, filters, and epochs the data for a specific subject

    Args:
        subject_id = Subject number (1-9)
        training = if True, loads the T (training) session. If false, loads E (evaluation)
        l_freq = lower bandpass cutoff in Hz. Default 8.0 matches the CSP baseline (mu/beta band)
        h_freq = upper bandpass cutoff in Hz. Default 30.0 matches the CSP baseline
        output_dir = where to save the resulting .fif file. Use a different folder per processing profile
            (e.g. "data/processed" for CSP, "data/processed_dl for deep learning) so the two never overwrite each other

    Returns: epochs (mne.Epochs): the processed, epoched data
    """

    # 1. Setup paths
    processed_dir = Path(output_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load Raw Data
    loader = DataLoader()

    # Call load_session to get the raw EEG data and the event mapping
    print(f"Loading Subject: {subject_id}")

    # FIX: was hardcoded as 'training=True' which ignores meant this function ignored the training argument it receveid and always loaded the T (train) file.
    # Now it actually respects the flag, so callers can request the E (eval) file
    raw, event_mapping = loader.load_session(subject_id, training=training)

    # 3. Apply Notch Filter (50Hz)
    # This always stays the same regardless of profile. It just removes electrical
    # line noise, it's not part of the which band do we keep decision below.
    raw.notch_filter(50.0, picks='eeg', verbose=False)

    # 4. Apply Bandpass Filter. This is the part that chances between profiles.
    # CSP baseline (default): 8-30Hz, narrow, matches the rhythms CSP looks for.
    # EEGNet / deep learning: wider band (e.g. ~4-40Hz), so the network has more raw information avaliable to learn its own temporal filters from,
    # instead of us pre-deciding which frequencies matter.
    print(f"Applying bandpass filter: {l_freq}-{h_freq} Hz")
    raw.filter(l_freq, h_freq, fir_design='firwin', skip_by_annotation='edge', picks='eeg', verbose=False)

    # We must tell MNE exactly how to translate the GDF strings to integers
    annotation_map = {
        '769': 769,
        '770': 770,
        '771': 771,
        '772': 772
    }


    # 5. Extract Events
    # Finds timestamps where the arrows appeared on the screen.
    events, _ = mne.events_from_annotations(raw, event_id=annotation_map, verbose=False)

    # Map the dataset codes to clear names for our model
    event_id = {
        'Left Hand': 769,
        'Right Hand': 770,
        'Foot': 771,
        'Tongue': 772
    }

    # 6. Create epochs
    # Slicing the continous signal into small windows
    # tmin = -0.5, tmax = 4.0. We take 0.5s before the cue and 4.0s after the cue
    # baseline=(-0.5, 0): We use 0.5s of silence before the cue to zero out the signal
    # This ensures that we are measuring the change in brain activity, not just the background level
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=-0.5,
        tmax=4.0,
        baseline=(-0.5,0),
        preload=True,
        picks='eeg',
        on_missing='warn' # warn if a class is missing, but doesn't crash
    )

    # 7. Save process data
    # .fif file is MNE's native format
    save_path = processed_dir / f"subject_{subject_id}_epo.fif"
    epochs.save(save_path, overwrite=True)
    print(f"Saved to: {save_path}")

    return epochs

def preprocess_for_baseline(subject_id: int, training: bool = True):
    # CSP+LDA baseline profile: 8-30Hz, saved to data/processed/
    return preprocessing_subject(
        subject_id,
        training=training,
        l_freq=8.0,
        h_freq=30.0,
        output_dir="data/processed",
    )

def preprocess_for_deep_learning(subject_id: int, training: bool = True):
    # EEGNet / deep learning profile: 4-40Hz, saved to data/processed_dl
        return preprocessing_subject(
        subject_id,
        training=training,
        l_freq=4.0,
        h_freq=40.0,
        output_dir="data/processed_dl",
    )
    

# test block
if __name__ == "__main__":
    try:
        # Smoke test: run both profiles for subject 1 and confirm they don't collide.
        print("Baseline profile (8-30Hz -> data/processed)")
        baseline_epochs = preprocess_for_baseline(1)
        print(baseline_epochs)

        print("\nDeep Learning profile (4-40Hz -> data/processed_dl)")
        dl_epochs = preprocess_for_deep_learning(1)
        print(dl_epochs)
        
    except Exception as e:
        print(f"Error: {e}")