import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE


# ============================================================
# 1. Load Dataset
# ============================================================

print("=" * 70)
print("SMOTE + RANDOM FOREST - 5-FOLD CROSS VALIDATION")
print("=" * 70)

df = pd.read_csv("IDS_Clean.csv")

X = df.drop("Target", axis=1)
y = df["Target"]

print(f"\nDataset shape: {df.shape}")
print(f"Features: {X.shape[1]}")
print(f"Samples: {len(df)}")

print("\nClass distribution:")
print(y.value_counts())


# ============================================================
# 2. Cross Validation
# ============================================================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


results = []


# ============================================================
# 3. Run 5 Folds
# ============================================================

for fold, (train_idx, test_idx) in enumerate(
        skf.split(X, y), start=1):

    print("\n")
    print("=" * 50)
    print(f"FOLD {fold}")
    print("=" * 50)

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()

    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")


    # ========================================================
    # 4. Feature Selection
    #    IMPORTANT: only training data
    # ========================================================

    print("\nSelecting features...")

    selector_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    selector_model.fit(X_train, y_train)

    importances = pd.Series(
        selector_model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)

    selected_features = importances.head(20).index.tolist()

    print("\nTop 20 features:")

    for i, feature in enumerate(selected_features, 1):
        print(f"{i}. {feature}")


    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]


    # ========================================================
    # 5. SMOTE - ONLY TRAINING DATA
    # ========================================================

    print("\nClass distribution BEFORE SMOTE:")

    print(y_train.value_counts())

    smote = SMOTE(
        random_state=42
    )

    X_train_smote, y_train_smote = smote.fit_resample(
        X_train_selected,
        y_train
    )

    print("\nClass distribution AFTER SMOTE:")

    print(pd.Series(y_train_smote).value_counts())

    print(
        f"\nTraining samples after SMOTE: "
        f"{len(X_train_smote)}"
    )


    # ========================================================
    # 6. Random Forest
    # ========================================================

    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train_smote,
        y_train_smote
    )


    # ========================================================
    # 7. Prediction
    # ========================================================

    y_pred = model.predict(X_test_selected)


    # ========================================================
    # 8. Metrics
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn)


    # ========================================================
    # 9. Print Results
    # ========================================================

    print("\nSMOTE + RF Results:")

    print(
        f"Accuracy : {accuracy * 100:.4f}%"
    )

    print(
        f"Precision: {precision * 100:.4f}%"
    )

    print(
        f"Recall   : {recall * 100:.4f}%"
    )

    print(
        f"F1       : {f1 * 100:.4f}%"
    )

    print(
        f"FPR      : {fpr * 100:.6f}%"
    )

    print("\nConfusion Matrix:")
    print(cm)


    # ========================================================
    # 10. Save Fold Results
    # ========================================================

    results.append({
        "Fold": fold,
        "Accuracy": accuracy * 100,
        "Precision": precision * 100,
        "Recall": recall * 100,
        "F1": f1 * 100,
        "FPR": fpr * 100
    })


# ============================================================
# 11. Final Results
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "SMOTE_RF_5Fold_Results.csv",
    index=False
)


# ============================================================
# 12. Mean and Standard Deviation
# ============================================================

print("\n")
print("=" * 70)
print("FINAL 5-FOLD CROSS-VALIDATION RESULTS")
print("=" * 70)

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "FPR"
]

means = {}
stds = {}

for metric in metrics:

    mean = results_df[metric].mean()
    std = results_df[metric].std()

    means[metric] = mean
    stds[metric] = std

    print(
        f"{metric:<10}: "
        f"{mean:.4f}% +/- {std:.4f}%"
    )


# ============================================================
# 13. Save Summary
# ============================================================

summary_df = pd.DataFrame({
    "Metric": metrics,
    "Mean": [means[m] for m in metrics],
    "Std": [stds[m] for m in metrics]
})

summary_df.to_csv(
    "SMOTE_RF_5Fold_Summary.csv",
    index=False
)


print("\n")
print("Files saved:")
print("SMOTE_RF_5Fold_Results.csv")
print("SMOTE_RF_5Fold_Summary.csv")

print("\n5-Fold SMOTE evaluation completed.")