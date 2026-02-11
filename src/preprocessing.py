import mne
import numpy as np
from pathlib import Path
from data_loader import DataLoader

def preprocessing_subject(subject_id: int, training: bool = True):
    # Loads, filters, and epochs the data for a specific subject

    # 1. Setup paths
    processed_dir = Path('data/processed')
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load Raw Data
    loader = DataLoader()

    # Call load_session to get the raw EEG data and the event mapping
    print(f"Loading Subject: {subject_id}")
    raw, event_mapping = loader.load_session(subject_id, training=True)

    # 3. Apply Notch Filter (50Hz)
    raw.notch_filter(50.0, picks='eeg', verbose=False)

    # 4. Apply Bandpass Filter (8-30Hz)
    raw.filter(8.0, 30.0, fir_design='firwin', skip_by_annotation='edge', picks='eeg', verbose=False)

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

    return epochs

# test block
if __name__ == "__main__":
    try:
        processed_epochs = preprocessing_subject(1)

        # Verify the result
        print("\n--- Processing Summary ---")
        print(processed_epochs)
        
    except Exception as e:
        print(f"Error: {e}")