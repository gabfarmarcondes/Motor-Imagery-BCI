""""
Runs the full pipeline (preprocessing -> CSP+LDA baseline) across all the 9 subjects
of the BCI Competition IV 2a Dataset, and saves a comparison table.

This file doesn't duplicate any logic from preprocessing.py or baseline.py. It just imports and calls their functions in a loop, and collectes the results
that were previously only printed to the console for a single subject.
"""

from pathlib import Path
import pandas as pd

from preprocessing import preprocess_for_baseline, preprocess_for_deep_learning
from baseline import run_baseline

# BCI Competition IV 2a has exactly 9 subjcts, numbered 1 to 9
ALL_SUBJECTS = range(1,10)

def preprocess_all_subjects(preprocess_fn=preprocess_for_baseline):
    """
    Runs preprocessing_subject() for every subject

    Wrapped in try/except per subject: if one .gdf file is missing or corrupted, we dont't want that to kill whole batch.
    We skip it, log it, and keep going.
    """

    succeeded = []
    failed = []

    for subject_id in ALL_SUBJECTS:
        print(f"\nPreprocessing subject {subject_id}")
        try:
            preprocess_fn(subject_id, training=True)
            succeeded.append(subject_id)
        except Exception as e:
            print(f"\nFailed subject {subject_id}: {e}")
            failed.append(subject_id)
    
    print(f"\nPreprocessing done: {len(succeeded)}/9 subjects succeeded")

    if failed:
        print(f"Failed subjects: {failed}")
    
    return succeeded

def run_baseline_all_subjects(subject_ids):
    """
    Runs the CSP+LDA baseline for each successfully preprocessed subject.

    Instead of just printing the result (like baseline.py's __main__ block does for a single subject),
    we collect everything into a list of dicts so it can become a proper comparison table.
    """

    results = []

    for subject_id in subject_ids:
        print(f"\nBaseline for subject {subject_id}")
        try:
            mean_acc, std_acc = run_baseline(subject_id)
            results.append({
                "subject": subject_id,
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc
            })
        except Exception as e:
            print(f"\nFailed subject: {subject_id}: {e}")
        
    return results

def save_results_table(results):
    """
    Saves the per-subject results as a CSV under results/, and prints a quick summary (mean across all subjects)
    so you get a single headline number without opening the file.
    """

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    df = pd.DataFrame(results)
    save_path = results_dir / "baseline_results.csv"
    df.to_csv(save_path, index=False)

    print("\nBaseline Results (all subjects)")
    print(df.to_string(index=False))
    print(f"\nOverall mean accuracy: {df['mean_accuracy'].mean():.2%}")
    print(f"Results saved to: {save_path}")

    return df

if __name__ == "__main__":
    processed_subjects = preprocess_all_subjects()
    results = run_baseline_all_subjects(processed_subjects)
    save_results_table(results)
