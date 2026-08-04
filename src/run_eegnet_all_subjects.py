"""
Runs EEGNet leave-one-run-out cross-validation acrpss all 9 subjects, and merges the results with the existing CSP+LDA baseline
(results/baseline_results.csv) into a single comparison table.

Mirros run_all_subjects.py's structure: per-subjects try/except so one failure doesn't kill the whole batch,
plus a final saved CSV.

Runtime warning: on CPU, subject 1 alone took several minutes across 6 folds (each up to 150 epochs, 
with early stopping typically firing between 40-90 epochs). Across 9 subjects this can run well 
over an hour. Consider running this in the background (e.g. `nohup python src/run_eegnet_all_subjects.py &`)
rather than waiting on it interactively
"""

import time
from pathlib import Path

import pandas as pd

from eegnet import run_eegnet_cv

ALL_SUBJECTS = range(1, 10)

def run_eegnet_all_subjects(epochs=150):
    """
    Runs run_eegnet_cv() for every subject, collecting mean/std accuracy per subject.
    Each subject's confusion matrix is still saved individually by run_eegnet_cv (figures/eegnet_confusion_matrix_subject_{id}.png).
    This function only agreements the summary numbers.
    """

    results = []

    for subject_id in ALL_SUBJECTS:
        print(f"\n{'='*50}")
        print(f"=== EEGNet: subject {subject_id} ===")
        print(f"{'='*50}")

        start = time.time()
        try:
            mean_acc, std_acc, fold_accuracies = run_eegnet_cv(subject_id, epochs=epochs)
            elapsed = time.time() - start

            results.append({
                "subject": subject_id,
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc,
                "min_fold_accuracy": min(fold_accuracies),
                "max_fold_accuracy": max(fold_accuracies),
            })

            print(f"\nSubject {subject_id} done in {elapsed/60:.1f} min -- "
                  f"mean_acc={mean_acc:.2%}")

        except Exception as e:
            print(f"  -> FAILED subject {subject_id}: {e}")

    return results

def save_eegnet_results(results):
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    df = pd.DataFrame(results)
    save_path = results_dir / "eegnet_results.csv"
    df.to_csv(save_path, index=False)

    print("\n--- EEGNet Results (all subjects) ---")
    print(df.to_string(index=False))
    print(f"\nOverall mean accuracy: {df['mean_accuracy'].mean():.2%}")
    print(f"Results saved to: {save_path}")

    return df

def compare_with_baseline(eegnet_df):
    """
    Merges EEGNet results with the existing CSP+LDA baseline results, subject by
    subject, and reports which method won for each, and by how much, since a method
    winning by less than its own fold-to-fold std isn't a meaninful win (see subject 1's
    result: EEGNet's std alone was larger than its margin over baseline).
    """
    baseline_path = Path("results/baseline_results.csv")
    if not baseline_path.exists():
        print(f"\nNo baseline results found at {baseline_path} -- skipping comparison.")
        return None

    baseline_df = pd.read_csv(baseline_path)

    merged = baseline_df.merge(
        eegnet_df, on="subject", suffixes=("_baseline", "_eegnet")
    )

    merged["diff"] = merged["mean_accuracy_eegnet"] - merged["mean_accuracy_baseline"]

    # A diff is only meaningful win if it's larger than EEGNet's own fold std.
    # Otherwise the improvement is within the method's own noise.
    merged["meaningful_win"] = merged["diff"].abs() > merged["std_accuracy_eegnet"]

    comparison_path = Path("results") / "comparison_baseline_vs_eegnet.csv"
    merged.to_csv(comparison_path, index=False)

    print("\nBaseline vs EEGNet")
    display_cols = ["subject", "mean_accuracy_baseline", "mean_accuracy_eegnet", "diff", "meaningful_win"]
    print(merged[display_cols].to_string(index=False))
    print(f"\nComparison saved to: {comparison_path}")

    n_eegnet_wins = ((merged["diff"] > 0) & merged["meaningful_win"]).sum()
    n_baseline_wins = ((merged["diff"] < 0) & merged["meaningful_win"]).sum()
    n_ties = len(merged) - n_eegnet_wins - n_baseline_wins
    print(f"\nMeaningful EEGNet wins: {n_eegnet_wins}/9 | "
          f"Meaningful baseline wins: {n_baseline_wins}/9 | "
          f"Within noise (tie): {n_ties}/9")

    return merged

# test block
if __name__ == "__main__":
    overall_start = time.time()

    results = run_eegnet_all_subjects(epochs=150)
    eegnet_df = save_eegnet_results(results)
    compare_with_baseline(eegnet_df)

    total_elapsed = time.time() - overall_start
    print(f"\nTotal runtime: {total_elapsed/60:.1f} minutes")