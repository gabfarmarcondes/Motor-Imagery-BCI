"""
Generates the wideband (4-40Hz) preprocessing cache for all 9 subjects,
used later by EEGNet. Does NOT touch data/processed/ or results/baseline_results.csv.
Those belong to the CSP baseline pipeline only
"""

from run_all_subjects import preprocess_all_subjects
from preprocessing import preprocess_for_deep_learning

if __name__ == "__main__":
    succeeded = preprocess_all_subjects(preprocess_fn=preprocess_for_deep_learning)
    print(f"\nWideband cache for subjects: {succeeded}")
    print("Saved under data/processed_dl/")