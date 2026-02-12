import mne
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from mne.decoding import CSP

def run_baseline(subject_id: int):
    # Run the CSP + LDA baseline model for a specific subject

    # 1. Load the Preprocessed data
    data_path = Path(f"data/processed/subject_{subject_id}_epo.fif")
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data not found: {data_path}")
    
    epochs = mne.read_epochs(data_path, preload=True, verbose=False)

    # 2. Extract Data (X) and Labels (y)
    # X Shape: (n_epochs, n_times) -> e.g., (288, 22, 1125)
    # y Shape: (n_epochs) -> e.g., [0, 1, 2, 3, ...]
    # We copy the data to avoid messing up the original MNE object
    
    # MNE objects are complex. Scikit-learn expects simple NumPy arrays. This command extract the raw numbers into a matrix X.
    X = epochs.get_data(copy=True) 
    y = epochs.events[:, -1] # last column contains the event ID (769, 770, 771, 772)

    # 3.Define the Machine Learning Pipeline
    # CSP: finds spatial filters that maximize the difference between classes
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
        # It finds filters (direction in space) where the signal variance is high for one class and low for the other
        # We tell it to find the 4 best filters (usually the most discriminative ones)
    # LDA: draw a line to separate the classes based on CSP features
    lda = LinearDiscriminantAnalysis()

    # It bundles CSP and LDA into a single object
    # When we call clf.fit(), it automatically runs CSP first, transforms the data, and then passes it to LDA
    clf = Pipeline([('CSP', csp), ('LDA', lda)])

    # 4. Define Cross-Validation Strategy
    # ShuffleSplit: Randomly shuffles data and splits it 10 times.
    # Instead training once, we train and test 10 times on different random chuncks of data.
    # test_size = 0.2, means 20% of data is used for testing and 80% for training
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

    # 5. Run Cross-Validation (Training Loop)
    # cross_val_Score automates the loop: Split -> Train -> Test -> Score
    # It returns a list of 10 accuracy scores (one for each split)
    scores = cross_val_score(clf, X, y, cv=cv, n_jobs=1)

    # Calculate average accuracy
    mean_score = np.mean(scores)
    std_score = np.std(scores)

    # 6. Generate Confusion Matrix (Verification)
    # to draw a matrix, we need to manually predict on one split
    # we grab the indices of the first split
    train_idx, test_idx = next(cv.split(X,y))
    # We manually split the data one last time to generate the Confusion Matrix. 
    # We need predictions (y_pred) to compare against the truth (y_test) to see what the model is getting wrong (e.g., confusing Feet with Tongue)
    y_train, y_test = y[train_idx], y[test_idx]
    X_train, X_test = X[train_idx], X[test_idx]

    # Fit on train, predict on test
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Create the Confusion Matrix plot
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=epochs.event_id.keys())

# Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap='Blues', ax=ax)
    plt.title(f"Confusion Matrix (Subject {subject_id})\nMean Accuracy: {mean_score:.2%}")
    
    # Save the figure
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)
    save_path = figures_dir / f"confusion_matrix_subject_{subject_id}.png"
    plt.savefig(save_path)
    
    print(f"\nResults for Subject {subject_id}:")
    print(f"Mean Accuracy: {mean_score:.2%} (+/- {std_score:.2%})")
    print(f"Chance Level: 25.00% (4 classes)")
    print(f"Confusion Matrix saved to: {save_path}")

    return mean_score

# --- Main Execution Block ---
if __name__ == "__main__":
    try:
        # Run baseline for Subject 1
        acc = run_baseline(1)
        
        # Check Acceptance Criteria
        if acc > 0.65:
            print("\nSUCCESS: Accuracy > 65%. Baseline established!")
        else:
            print("\nWARNING: Accuracy is low. Check data quality.")
            
    except Exception as e:
        print(f"Error: {e}")