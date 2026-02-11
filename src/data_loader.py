import mne
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional

class DataLoader:
    # Handles loading and preprocessing of BCI Competition IV 2a dataset.

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)

        self.standard_channels = [
            'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
            'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz',
            'EOG-left', 'EOG-central', 'EOG-right'
        ]

        self.event_mapping = {
            '769': 'Left Hand',
            '770': 'Right Hand',
            '771': 'Foot',
            '772': 'Tongue'
        }

    def load_session (self, subject_id: int, training: bool = True) -> Tuple[mne.io.Raw, Dict]:
        """
        Load a specific subjection's session, renames channels, and sets montage.
        
        Args:
            subject_id (int): Subject number (1-9)
            training (bool): If true, loads "T" (Training) file. If false, loads "E" (Evaluation).
        
        Returns:
            raw (mne.io.Raw): The loaded and corrected Raw object.
            events (dict): The events dictionary.
        """

        # 1. Construct file path
        session_type = 'T' if training else 'E'
        file_name = f"A0{subject_id}{session_type}.gdf"
        file_path = self.data_dir / file_name

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        print(f"Loading: {file_name}")

        # 2. Load GDF (Preload memory)
        # MNE will warn about channel names
        raw = mne.io.read_raw_gdf(file_path, preload=True, verbose='error')

        # 3. Rename channels (Index-based memory)
        current_names = raw.ch_names
        if len(current_names) < 25:
            raise ValueError(f"Expected at least 25 channels, found {len(current_names)}")

        mapping = {current_names[i]: self.standard_channels[i] for i in range(25)}
        raw.rename_channels(mapping)

        # 4. Set channel Types (EEG vs EOG)
        raw.set_channel_types({
            'EOG-left': 'eog',
            'EOG-central': 'eog',
            'EOG-right': 'eog'
        })

        # 5. Set montage (10-20 standard system)
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)

        # 6. Extract events
        events, _ = mne.events_from_annotations(raw, verbose=False)

        print(f"Successfully loaded Subject {subject_id}. Found {len(events)} events.")
        return raw, self.event_mapping
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    try:
        loader = DataLoader()
        # Test loading subject 1
        raw, events = loader.load_session(1, training=True)
        print("Raw info: ", raw.info)

        figure_dir = Path("figures")
        figure_dir.mkdir(exist_ok=True)

        fig = raw.plot_sensors(show_names=True, show=False)

        save_path = figure_dir / f"sensors_montage_subject_1.png"
        fig.savefig(save_path)

        print(f"[SUCCESS] Sensor plot saved to: {save_path}")
    except Exception as e:
        print(f"Error: {e}")