"""
Artifact-trial detection for the BCI IV 2a dataset.

REVISED APPROACH: the original version used a time-window heuristic (bisect over
onset timestamps) to associate '1023' with nearby cues, but it produced zero matches
across all subjects when tested. A bug in that time-arithmetic logic, not a
parameter tuning issue.

Diagnostic output (inspect_artifact_context.py) revealed the actual structure
directly: '1023' always occurs at the SAME onset timestamp as a '768' (trial start)
marker, and the trial's cue (769-772) always follows next in the sequence, ~2s later.
So instead of comparing timestamps, we do a straightforward sequential scan: '768'
resets a "current trial rejected" flag to False, '1023' sets it to True, and whenever
a cue code is hit, we record the current flag value and move to the next trial.
This encodes the exact pattern observed, with no timing arithmetic to get wrong.
"""

import numpy as np


def get_artifact_trial_mask(
    raw,
    cue_codes=('769', '770', '771', '772'),
    reject_code='1023',
    trial_start_code='768',
):
    """
    Returns a boolean array, one entry per motor imagery cue in chronological order
    (same order mne.Epochs produces from the same raw + cue_codes), True where that
    trial is marked as containing an artifact.
    """
    onsets = raw.annotations.onset
    descriptions = list(raw.annotations.description)
    pairs = sorted(zip(onsets, descriptions), key=lambda p: p[0])

    mask = []
    current_trial_rejected = False

    for _, desc in pairs:
        if desc == trial_start_code:
            current_trial_rejected = False
        elif desc == reject_code:
            current_trial_rejected = True
        elif desc in cue_codes:
            mask.append(current_trial_rejected)
            # No need to reset here, the next '768' always resets before the
            # next cue is reached.

    return np.array(mask, dtype=bool)


# test block
if __name__ == "__main__":
    from data_loader import DataLoader

    expected_counts = {1: 15, 2: 18, 3: 18, 4: 26, 5: 26, 6: 69, 7: 17, 8: 24, 9: 51}

    loader = DataLoader()
    for subject_id, expected in expected_counts.items():
        raw, _ = loader.load_session(subject_id, training=True)
        mask = get_artifact_trial_mask(raw)
        status = "OK" if mask.sum() == expected else "MISMATCH"
        print(f"Subject {subject_id}: {mask.sum()} flagged (expected {expected}) [{status}]")