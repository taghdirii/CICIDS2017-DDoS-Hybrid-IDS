import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE


# =========================================================
# 1. Load Dataset
# =========================================================

print("Loading dataset...")

df = pd.read_csv("IDS_Clean.csv")

print("Dataset shape:", df.shape)


# =========================================================
# 2. Load Selected Features
# =========================================================

selected_features_df = pd.read_csv(
    "Selected_Features.csv"
)

selected_features = (
    selected_features_df["Feature"].tolist()
)

print("\nSelected Features:")

for i, feature in enumerate(selected_features, 1):
    print(f"{i}. {feature}")


# Check missing features
missing_features = [
    feature
    for feature in selected_features
    if feature not in df.columns
]

if missing_features:

    print("\nERROR: Missing features:")

    for feature in missing_features:
        print(feature)

    raise ValueError(
        "Some selected features do not exist."
    )


X = df[selected_features]
y = df["Target"]


# =========================================================
# 3. Train / Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================================================
# 4. Class Distribution Before SMOTE
# =========================================================

print("\nClass distribution BEFORE SMOTE:")

print(y_train.value_counts())


# =========================================================
# 5. Apply SMOTE ONLY to Training Data
# =========================================================

print("\nApplying SMOTE...")

smote = SMOTE(
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)


# =========================================================
# 6. Class Distribution After SMOTE
# =========================================================

print("\nClass distribution AFTER SMOTE:")

print(
    pd.Series(y_train_smote).value_counts()
)

print(
    "\nTraining samples after SMOTE:",
    len(X_train_smote)
)


# =========================================================
# 7. Train Random Forest
# =========================================================

print("\nTraining Random Forest...")

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf.fit(
    X_train_smote,
    y_train_smote
)


# =========================================================
# 8. Prediction
# =========================================================

y_probability = rf.predict_proba(
    X_test
)[:, 1]

y_prediction = (
    y_probability >= 0.5
).astype(int)


# =========================================================
# 9. Evaluation
# =========================================================

accuracy = accuracy_score(
    y_test,
    y_prediction
)

precision = precision_score(
    y_test,
    y_prediction,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_prediction,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_prediction,
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    y_prediction
)

tn, fp, fn, tp = cm.ravel()

fpr = fp / (fp + tn)


# =========================================================
# 10. Print Results
# =========================================================

print("\n" + "=" * 50)
print("SMOTE + RANDOM FOREST RESULTS")
print("=" * 50)

print(
    f"Accuracy : {accuracy:.6f} "
    f"({accuracy * 100:.4f}%)"
)

print(
    f"Precision: {precision:.6f} "
    f"({precision * 100:.4f}%)"
)

print(
    f"Recall   : {recall:.6f} "
    f"({recall * 100:.4f}%)"
)

print(
    f"F1-Score : {f1:.6f} "
    f"({f1 * 100:.4f}%)"
)

print(
    f"FPR      : {fpr:.6f} "
    f"({fpr * 100:.6f}%)"
)

print("\nConfusion Matrix:")

print(cm)


# =========================================================
# 11. Save Results
# =========================================================

results = pd.DataFrame({

    "Model": [
        "SMOTE + Random Forest"
    ],

    "Accuracy": [
        accuracy
    ],

    "Precision": [
        precision
    ],

    "Recall": [
        recall
    ],

    "F1": [
        f1
    ],

    "FPR": [
        fpr
    ],

    "TN": [
        tn
    ],

    "FP": [
        fp
    ],

    "FN": [
        fn
    ],

    "TP": [
        tp
    ]

})


results.to_csv(
    "SMOTE_RF_Results.csv",
    index=False
)

print(
    "\nResults saved to: "
    "SMOTE_RF_Results.csv"
)