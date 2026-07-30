"""
Run-based train/validation split for the BCI IV 2a Motor Imagery Data.

baseline.py uses a random ShuffleSplit at the TRIAL level, which is fine for CSP+LDA
(and shouldn't be changed, to keep prior baseline results comparable). For EEGNet,
a random trial-level split risks leakage: trials recorded in the same run are
temporally close and can share confounds (electrode drift, fatigue, etc.), so a 
random split can make validation accuracy look better than it would on genuinely
unseen data.

We confirmed via inspect_runs.py that every subject has exactly 6 motor imagery runs
of 48 trials each (some subjects also have 1-3 extra EOG calibration blocks before 
the runs, which we exclude here since they contain no motor imagery cues).

This module provides:
    - get_motor_imagery_run_ids(): recovers which run (0-5) each epoch/trial belongs to,
    directly from the raw annotation (the .fif epoch cache doesn't store this by
    default, so this needs the original 'raw' object, not just 'epochs').
    - run_based_splits(): wraps sklearn's GroupKFold so each fold holds out one ENTIRE run
    for validation, never mixing trials from the same run across train/validation.
"""

import bisect
import numpy as np
from sklearn.model_selection import GroupKFold

from data_loader import DataLoader

def get_motor_imagery_run_ids(raw, cue_codes=('769', '770', '771', '772'), run_marker='32766'):
    """
    Returns a run_id (0-5) each motor imagery cue/trial in 'raw' in chronological order, matching
    the order mne.Epochs will produce when built from the same raw object and the same cue codes.

    Any '32766' boundary that isn't followed by cue trials (e.g. EOG calibration blocks) is
    automatically excludes: we only assign run numbers based on boundaries that actually
    contains cues, then renumber then 0-5 in order.
    """

    onsets = raw.annotations.onset
    descriptions = list(raw.annotations.description)

    boundary_times = sorted(onsets[i] for i, d in enumerate(descriptions) if d== run_marker)
    cue_times = sorted(onsets[i] for i, d in enumerate(descriptions) if d in cue_codes)

    if not boundary_times:
        raise ValueError(f"No '{run_marker}' run boundaries found in this recording.")

    # For each cue, find which boundary it falls after (bisect_right - 1 gives the
    # index of the last boundary at or before cue's onset time.)
    absolute_run_id = [bisect.bisect_right(boundary_times, t) - 1 for t in cue_times]

    # Only boundaries that actually have cues assigned to then are real MI runs.
    # Renumber those, in chronological order, to a clean 0..5 range, this is what makes
    # subject 4 (fewer calibration blocks, different abosolute boundary indices)
    # line up with every other subject.
    ids_with_cues_in_order = sorted(set(absolute_run_id))
    remap = {old: new for new, old in enumerate(ids_with_cues_in_order)}
    run_ids = np.array([remap[r] for r in absolute_run_id])

    return run_ids

def run_based_splits(run_ids):
    """
    Yields (train_idx, val_idx) pairs, one per unique run, using sklearn's
    GroupKFold so each holds out one entire run for validation, no trial
    from that run appears in training for that fold.

    n_splits is set to the number of unique runs (6 for this dataset), giving
    a leave-one-run-out cross-validation. The run-based analogue of baseline.py's
    10x ShuffleSplit.
    """
    n_runs = len(np.unique(run_ids))
    gfk = GroupKFold(n_splits=n_runs)

    # X is required by sklearn's API but unique for the split logic itselt
    # GroupdKFold only looks at groups. We pass a dummy array if the right length.
    dummy_X = np.zeros(len(run_ids))

    return gfk.split(dummy_X, groups=run_ids)

# test block
if __name__ == "__main__":
    loader = DataLoader()
    raw, _ = loader.load_session(1, training=True)

    run_ids = get_motor_imagery_run_ids(raw)
    print(f"Found {len(run_ids)} motor imagery trials across {len(np.unique(run_ids))}")
    print(f"Trials per run: {np.bincount(run_ids)}")

    print("\nRun based Split (leave-one-run-out)")
    for fold, (train_idx, val_idx) in enumerate(run_based_splits(run_ids)):
        val_runs = np.unique(run_ids[val_idx])
        print(f"Fold {fold}: train{len(train_idx)} trials, val={len(val_idx)} trials "
              f"val run(s): {val_runs.tolist()}")