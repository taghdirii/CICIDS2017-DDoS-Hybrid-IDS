import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import skfuzzy as fuzz
from skfuzzy import control as ctrl


# =========================================================
# 1. Load Dataset
# =========================================================

print("Loading dataset...")

df = pd.read_csv("IDS_Clean.csv")

print("Dataset shape:", df.shape)

X_all = df.drop("Target", axis=1)
y = df["Target"]


# =========================================================
# 2. Check Fuzzy Features
# =========================================================

fuzzy_features = [
    "Flow Packets/s",
    "Flow Bytes/s",
    "SYN Flag Count"
]

missing = [
    f for f in fuzzy_features
    if f not in X_all.columns
]

if missing:
    raise ValueError(
        f"Missing fuzzy features: {missing}"
    )


# =========================================================
# 3. Normalize Fuzzy Feature
#    Parameters learned ONLY from training fold
# =========================================================

def normalize_train_test(train_series, test_series):

    train_series = (
        train_series
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    test_series = (
        test_series
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    low = train_series.quantile(0.01)
    high = train_series.quantile(0.99)

    if high <= low:

        return (
            np.zeros(len(train_series)),
            np.zeros(len(test_series))
        )

    train_norm = (
        (train_series - low) /
        (high - low)
    ).clip(0, 1)

    test_norm = (
        (test_series - low) /
        (high - low)
    ).clip(0, 1)

    return (
        train_norm.values,
        test_norm.values
    )


# =========================================================
# 4. Build Fuzzy System
# =========================================================

def build_fuzzy_system():

    packets = ctrl.Antecedent(
        np.arange(0, 1.01, 0.01),
        "packets"
    )

    bytes_flow = ctrl.Antecedent(
        np.arange(0, 1.01, 0.01),
        "bytes_flow"
    )

    syn = ctrl.Antecedent(
        np.arange(0, 21, 1),
        "syn"
    )

    risk = ctrl.Consequent(
        np.arange(0, 101, 1),
        "risk"
    )

    # -----------------------------------------------------
    # Membership Functions
    # -----------------------------------------------------

    packets["low"] = fuzz.trimf(
        packets.universe,
        [0, 0, 0.4]
    )

    packets["medium"] = fuzz.trimf(
        packets.universe,
        [0.2, 0.5, 0.8]
    )

    packets["high"] = fuzz.trimf(
        packets.universe,
        [0.6, 1, 1]
    )

    bytes_flow["low"] = fuzz.trimf(
        bytes_flow.universe,
        [0, 0, 0.4]
    )

    bytes_flow["medium"] = fuzz.trimf(
        bytes_flow.universe,
        [0.2, 0.5, 0.8]
    )

    bytes_flow["high"] = fuzz.trimf(
        bytes_flow.universe,
        [0.6, 1, 1]
    )

    syn["low"] = fuzz.trimf(
        syn.universe,
        [0, 0, 5]
    )

    syn["medium"] = fuzz.trimf(
        syn.universe,
        [2, 8, 14]
    )

    syn["high"] = fuzz.trimf(
        syn.universe,
        [10, 20, 20]
    )

    risk["low"] = fuzz.trimf(
        risk.universe,
        [0, 0, 40]
    )

    risk["medium"] = fuzz.trimf(
        risk.universe,
        [25, 50, 75]
    )

    risk["high"] = fuzz.trimf(
        risk.universe,
        [60, 100, 100]
    )

    # -----------------------------------------------------
    # Rules
    # -----------------------------------------------------

    rule1 = ctrl.Rule(
        packets["high"] &
        bytes_flow["high"] &
        syn["high"],
        risk["high"]
    )

    rule2 = ctrl.Rule(
        packets["high"] &
        bytes_flow["high"],
        risk["high"]
    )

    rule3 = ctrl.Rule(
        packets["high"] &
        syn["high"],
        risk["high"]
    )

    rule4 = ctrl.Rule(
        bytes_flow["high"] &
        syn["high"],
        risk["high"]
    )

    rule5 = ctrl.Rule(
        packets["medium"] &
        bytes_flow["medium"] &
        syn["medium"],
        risk["medium"]
    )

    rule6 = ctrl.Rule(
        packets["medium"] &
        syn["medium"],
        risk["medium"]
    )

    rule7 = ctrl.Rule(
        bytes_flow["medium"] &
        syn["medium"],
        risk["medium"]
    )

    rule8 = ctrl.Rule(
        packets["low"] &
        bytes_flow["low"] &
        syn["low"],
        risk["low"]
    )

    rule9 = ctrl.Rule(
        packets["low"] &
        syn["low"],
        risk["low"]
    )

    rule10 = ctrl.Rule(
        bytes_flow["low"] &
        syn["low"],
        risk["low"]
    )

    system = ctrl.ControlSystem([
        rule1,
        rule2,
        rule3,
        rule4,
        rule5,
        rule6,
        rule7,
        rule8,
        rule9,
        rule10
    ])

    return system


# =========================================================
# 5. Fuzzy Prediction
# =========================================================

def fuzzy_predict(
    fuzzy_system,
    packets_test,
    bytes_test,
    syn_test
):

    probabilities = []

    for i in range(len(packets_test)):

        try:

            simulation = ctrl.ControlSystemSimulation(
                fuzzy_system
            )

            simulation.input["packets"] = float(
                packets_test[i]
            )

            simulation.input["bytes_flow"] = float(
                bytes_test[i]
            )

            simulation.input["syn"] = float(
                syn_test[i]
            )

            simulation.compute()

            risk_value = simulation.output["risk"]

            probabilities.append(
                risk_value / 100
            )

        except Exception:

            probabilities.append(0.0)

    return np.array(probabilities)


# =========================================================
# 6. 5-Fold Stratified Cross Validation
# =========================================================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


rf_results = []
hybrid_results = []


print("\n")
print("=" * 70)
print("5-FOLD STRATIFIED CROSS-VALIDATION")
print("=" * 70)


# =========================================================
# 7. Fold Loop
# =========================================================

for fold, (train_idx, test_idx) in enumerate(
    skf.split(X_all, y),
    1
):

    print("\n")
    print("=" * 50)
    print(f"FOLD {fold}")
    print("=" * 50)

    X_train_all = X_all.iloc[train_idx]
    X_test_all = X_all.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    print(
        "Training samples:",
        len(X_train_all)
    )

    print(
        "Testing samples:",
        len(X_test_all)
    )


    # =====================================================
    # 8. Feature Selection ONLY on Training Fold
    # =====================================================

    print("\nSelecting features...")

    selector = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    selector.fit(
        X_train_all,
        y_train
    )

    importance = pd.DataFrame({
        "Feature": X_train_all.columns,
        "Importance": selector.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    selected_features = (
        importance
        .head(20)["Feature"]
        .tolist()
    )

    print("\nTop 20 features:")

    for i, feature in enumerate(
        selected_features,
        1
    ):
        print(
            f"{i}. {feature}"
        )


    # =====================================================
    # 9. Prepare Selected Features
    # =====================================================

    X_train = X_train_all[
        selected_features
    ]

    X_test = X_test_all[
        selected_features
    ]


    # =====================================================
    # 10. Random Forest
    # =====================================================

    print("\nTraining Random Forest...")

    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(
        X_train,
        y_train
    )

    rf_probability = rf_model.predict_proba(
        X_test
    )[:, 1]

    rf_prediction = (
        rf_probability >= 0.5
    ).astype(int)


    # =====================================================
    # 11. RF Evaluation
    # =====================================================

    rf_accuracy = accuracy_score(
        y_test,
        rf_prediction
    )

    rf_precision = precision_score(
        y_test,
        rf_prediction,
        zero_division=0
    )

    rf_recall = recall_score(
        y_test,
        rf_prediction,
        zero_division=0
    )

    rf_f1 = f1_score(
        y_test,
        rf_prediction,
        zero_division=0
    )

    rf_cm = confusion_matrix(
        y_test,
        rf_prediction
    )

    tn, fp, fn, tp = rf_cm.ravel()

    rf_fpr = fp / (fp + tn)


    # =====================================================
    # 12. Fuzzy Normalization
    # =====================================================

    print("\nPreparing fuzzy inputs...")

    _, packets_test = normalize_train_test(
        X_train_all["Flow Packets/s"],
        X_test_all["Flow Packets/s"]
    )

    _, bytes_test = normalize_train_test(
        X_train_all["Flow Bytes/s"],
        X_test_all["Flow Bytes/s"]
    )

    syn_test = (
        X_test_all["SYN Flag Count"]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
        .clip(0, 20)
        .values
    )


    # =====================================================
    # 13. Fuzzy System
    # =====================================================

    print("\nRunning fuzzy inference...")

    fuzzy_system = build_fuzzy_system()

    fuzzy_probability = fuzzy_predict(
        fuzzy_system,
        packets_test,
        bytes_test,
        syn_test
    )


    # =====================================================
    # 14. Hybrid Model
    # =====================================================

    hybrid_score = (
        0.7 * rf_probability +
        0.3 * fuzzy_probability
    )

    hybrid_prediction = (
        hybrid_score >= 0.5
    ).astype(int)


    # =====================================================
    # 15. Hybrid Evaluation
    # =====================================================

    hybrid_accuracy = accuracy_score(
        y_test,
        hybrid_prediction
    )

    hybrid_precision = precision_score(
        y_test,
        hybrid_prediction,
        zero_division=0
    )

    hybrid_recall = recall_score(
        y_test,
        hybrid_prediction,
        zero_division=0
    )

    hybrid_f1 = f1_score(
        y_test,
        hybrid_prediction,
        zero_division=0
    )

    hybrid_cm = confusion_matrix(
        y_test,
        hybrid_prediction
    )

    h_tn, h_fp, h_fn, h_tp = (
        hybrid_cm.ravel()
    )

    hybrid_fpr = (
        h_fp /
        (h_fp + h_tn)
    )


    # =====================================================
    # 16. Save Fold Results
    # =====================================================

    rf_results.append({
        "Fold": fold,
        "Accuracy": rf_accuracy,
        "Precision": rf_precision,
        "Recall": rf_recall,
        "F1": rf_f1,
        "FPR": rf_fpr
    })

    hybrid_results.append({
        "Fold": fold,
        "Accuracy": hybrid_accuracy,
        "Precision": hybrid_precision,
        "Recall": hybrid_recall,
        "F1": hybrid_f1,
        "FPR": hybrid_fpr
    })


    # =====================================================
    # 17. Print Fold Results
    # =====================================================

    print("\nRF Results:")

    print(
        f"Accuracy : {rf_accuracy * 100:.4f}%"
    )

    print(
        f"Precision: {rf_precision * 100:.4f}%"
    )

    print(
        f"Recall   : {rf_recall * 100:.4f}%"
    )

    print(
        f"F1       : {rf_f1 * 100:.4f}%"
    )

    print(
        f"FPR      : {rf_fpr * 100:.6f}%"
    )


    print("\nHybrid Results:")

    print(
        f"Accuracy : {hybrid_accuracy * 100:.4f}%"
    )

    print(
        f"Precision: {hybrid_precision * 100:.4f}%"
    )

    print(
        f"Recall   : {hybrid_recall * 100:.4f}%"
    )

    print(
        f"F1       : {hybrid_f1 * 100:.4f}%"
    )

    print(
        f"FPR      : {hybrid_fpr * 100:.6f}%"
    )


# =========================================================
# 18. Convert Results to DataFrame
# =========================================================

rf_df = pd.DataFrame(
    rf_results
)

hybrid_df = pd.DataFrame(
    hybrid_results
)


# =========================================================
# 19. Save Fold Results
# =========================================================

rf_df.to_csv(
    "RF_5Fold_Results.csv",
    index=False
)

hybrid_df.to_csv(
    "Hybrid_5Fold_Results.csv",
    index=False
)


# =========================================================
# 20. Calculate Mean and Standard Deviation
# =========================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "FPR"
]


rf_summary = {}

hybrid_summary = {}


for metric in metrics:

    rf_summary[metric] = {
        "Mean": rf_df[metric].mean(),
        "Std": rf_df[metric].std()
    }

    hybrid_summary[metric] = {
        "Mean": hybrid_df[metric].mean(),
        "Std": hybrid_df[metric].std()
    }


# =========================================================
# 21. Print Final Summary
# =========================================================

print("\n")
print("=" * 70)
print("FINAL 5-FOLD CROSS-VALIDATION RESULTS")
print("=" * 70)


print("\nRandom Forest:")

for metric in metrics:

    mean = rf_summary[metric]["Mean"]
    std = rf_summary[metric]["Std"]

    if metric == "FPR":

        print(
            f"{metric:<10}: "
            f"{mean * 100:.6f}% "
            f"+/- {std * 100:.6f}%"
        )

    else:

        print(
            f"{metric:<10}: "
            f"{mean * 100:.4f}% "
            f"+/- {std * 100:.4f}%"
        )


print("\nRF + Fuzzy Hybrid:")

for metric in metrics:

    mean = hybrid_summary[metric]["Mean"]
    std = hybrid_summary[metric]["Std"]

    if metric == "FPR":

        print(
            f"{metric:<10}: "
            f"{mean * 100:.6f}% "
            f"+/- {std * 100:.6f}%"
        )

    else:

        print(
            f"{metric:<10}: "
            f"{mean * 100:.4f}% "
            f"+/- {std * 100:.4f}%"
        )


# =========================================================
# 22. Comparison
# =========================================================

print("\n")
print("=" * 70)
print("COMPARISON")
print("=" * 70)


rf_fpr_mean = rf_df["FPR"].mean()
hybrid_fpr_mean = hybrid_df["FPR"].mean()


if rf_fpr_mean > 0:

    reduction = (
        (rf_fpr_mean - hybrid_fpr_mean)
        / rf_fpr_mean
    ) * 100

    print(
        f"\nFPR Reduction: "
        f"{reduction:.2f}%"
    )


print("\nFiles saved:")

print(
    "RF_5Fold_Results.csv"
)

print(
    "Hybrid_5Fold_Results.csv"
)

print("\nCross-validation completed.")