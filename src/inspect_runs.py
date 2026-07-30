"""
Diagnostic script: inspects the raw annotations of a subject's GDF file to confirm
how motor imagery trials are organized into runs.

We work with plain Python lists/loops here instead of numpy vectorized comparisons
on raw.annotations.description -- a numpy "inhomogeneous shape" error showed up when
comparing that array directly, likely a quirk of how MNE's GDF reader structures the
annotations for this specific dataset. Plain iteration sidesteps it entirely, and is
just as fast for a few hundred annotations.

We also now distinguish "any 32766 boundary" (includes EOG calibration blocks, e.g.
eyes open/closed) from "runs that actually contain motor imagery cues" -- only the
second group matters for the train/validation split.
"""

from collections import Counter
from data_loader import DataLoader


def inspect_runs(subject_id: int, training: bool = True):
    loader = DataLoader()
    raw, _ = loader.load_session(subject_id, training=training)

    onsets = raw.annotations.onset
    descriptions = raw.annotations.description
    pairs = list(zip(onsets, descriptions))

    code_counts = Counter(desc for _, desc in pairs)
    print(f"\nSubject {subject_id} - all annotation codes found")
    for code, count in sorted(code_counts.items()):
        print(f"  {code}: {count}")

    run_marker = '32766'
    cue_codes = {'769', '770', '771', '772'}

    if run_marker not in code_counts:
        print(f"\nWARNING: run marker '{run_marker}' not found for subject {subject_id}.")
        return None

    run_start_times = sorted(onset for onset, desc in pairs if desc == run_marker)
    cue_times = sorted(onset for onset, desc in pairs if desc in cue_codes)

    print(f"\nTotal '32766' boundaries: {len(run_start_times)} "
          f"(includes non-motor-imagery blocks like eyes-open/closed/eye-movement).")

    # Assign each cue to the run boundary immediately before it.
    run_ids = []
    for t in cue_times:
        run_id = sum(1 for rs in run_start_times if rs <= t) - 1
        run_ids.append(run_id)

    trials_per_run = Counter(run_ids)
    print(f"Trials per boundary-block for subject {subject_id} (run_id: trial_count):")
    for run_id in sorted(trials_per_run):
        print(f"  Run {run_id}: {trials_per_run[run_id]} trials")

    # Only boundaries that actually contain cue trials are real motor imagery runs.
    # EOG calibration blocks will simply not appear here (0 cues assigned to them).
    print(f"Motor imagery runs (non-empty): {len(trials_per_run)}")

    return trials_per_run


if __name__ == "__main__":
    for subject_id in range(1, 10):
        try:
            inspect_runs(subject_id)
        except Exception as e:
            print(f"Subject {subject_id} FAILED: {e}")