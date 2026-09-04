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

import skfuzzy as fuzz
from skfuzzy import control as ctrl


# =========================================================
# 1. Load Dataset
# =========================================================

df = pd.read_csv("IDS_Clean.csv")

print("Dataset shape:", df.shape)

X_all = df.drop("Target", axis=1)
y = df["Target"]


# =========================================================
# 2. Train / Test Split
# =========================================================

X_train_all, X_test_all, y_train, y_test = train_test_split(
    X_all,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train_all))
print("Testing samples:", len(X_test_all))


# =========================================================
# 3. Load Features Selected ONLY from Training Data
# =========================================================

selected_features_df = pd.read_csv("Selected_Features.csv")

selected_features = selected_features_df["Feature"].tolist()

print("\nSelected Features:")

for i, feature in enumerate(selected_features, 1):
    print(f"{i}. {feature}")


# بررسی وجود ویژگی‌ها
missing_features = [
    feature for feature in selected_features
    if feature not in X_all.columns
]

if missing_features:
    print("\nERROR: Missing features:")
    for feature in missing_features:
        print(feature)
    raise ValueError("Some selected features do not exist in dataset.")


X_train = X_train_all[selected_features]
X_test = X_test_all[selected_features]


# =========================================================
# 4. Random Forest Model
# =========================================================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Probability of attack
rf_probability = rf_model.predict_proba(X_test)[:, 1]


# =========================================================
# 5. Fuzzy Normalization
#    Parameters are calculated ONLY from TRAINING DATA
# =========================================================

def normalize_using_train(train_series, test_series):

    train_series = train_series.replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    test_series = test_series.replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    # Learn limits only from training data
    low = train_series.quantile(0.01)
    high = train_series.quantile(0.99)

    # Avoid division by zero
    if high <= low:
        return (
            np.zeros(len(train_series)),
            np.zeros(len(test_series))
        )

    train_normalized = (
        (train_series - low) /
        (high - low)
    ).clip(0, 1)

    test_normalized = (
        (test_series - low) /
        (high - low)
    ).clip(0, 1)

    return train_normalized.values, test_normalized.values


# =========================================================
# 6. Fuzzy Input Variables
# =========================================================

required_fuzzy_features = [
    "Flow Packets/s",
    "Flow Bytes/s",
    "SYN Flag Count"
]

missing_fuzzy = [
    feature for feature in required_fuzzy_features
    if feature not in df.columns
]

if missing_fuzzy:
    print("\nERROR: Missing fuzzy features:")
    for feature in missing_fuzzy:
        print(feature)
    raise ValueError("Fuzzy input features are missing.")


# Training and testing indices
train_indices = X_train_all.index
test_indices = X_test_all.index


# Flow Packets/s
_, test_packets = normalize_using_train(
    df.loc[train_indices, "Flow Packets/s"],
    df.loc[test_indices, "Flow Packets/s"]
)


# Flow Bytes/s
_, test_bytes = normalize_using_train(
    df.loc[train_indices, "Flow Bytes/s"],
    df.loc[test_indices, "Flow Bytes/s"]
)


# SYN Flag Count
syn_test = (
    df.loc[test_indices, "SYN Flag Count"]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
    .clip(0, 20)
    .values
)


# =========================================================
# 7. Define Fuzzy Variables
# =========================================================

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


# =========================================================
# 8. Membership Functions
# =========================================================

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


# =========================================================
# 9. Fuzzy Rules
# =========================================================

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


# =========================================================
# 10. Build Fuzzy Control System
# =========================================================

risk_control = ctrl.ControlSystem([
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


# =========================================================
# 11. Fuzzy Inference
# =========================================================

fuzzy_probability = []

print("\nRunning fuzzy inference...")

for i in range(len(X_test)):

    try:

        simulation = ctrl.ControlSystemSimulation(
            risk_control
        )

        simulation.input["packets"] = float(
            test_packets[i]
        )

        simulation.input["bytes_flow"] = float(
            test_bytes[i]
        )

        simulation.input["syn"] = float(
            syn_test[i]
        )

        simulation.compute()

        fuzzy_risk = simulation.output["risk"]

        # Convert 0-100 to 0-1
        fuzzy_probability.append(
            fuzzy_risk / 100
        )

    except Exception:

        # If no fuzzy rule is activated
        fuzzy_probability.append(0.0)


fuzzy_probability = np.array(
    fuzzy_probability
)


# =========================================================
# 12. Hybrid Model
# =========================================================

# Combination:
# 70% Random Forest
# 30% Fuzzy System

hybrid_score = (
    0.7 * rf_probability +
    0.3 * fuzzy_probability
)


# Final classification threshold
hybrid_prediction = (
    hybrid_score >= 0.5
).astype(int)


# =========================================================
# 13. Evaluation
# =========================================================

accuracy = accuracy_score(
    y_test,
    hybrid_prediction
)

precision = precision_score(
    y_test,
    hybrid_prediction,
    zero_division=0
)

recall = recall_score(
    y_test,
    hybrid_prediction,
    zero_division=0
)

f1 = f1_score(
    y_test,
    hybrid_prediction,
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    hybrid_prediction
)

tn, fp, fn, tp = cm.ravel()

fpr = fp / (fp + tn)


# =========================================================
# 14. Print Results
# =========================================================

print("\n" + "=" * 50)
print("HYBRID RF + FUZZY RESULTS")
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
# 15. Save Results
# =========================================================

results = pd.DataFrame({
    "Model": ["RF + Fuzzy Hybrid"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1": [f1],
    "FPR": [fpr],
    "TN": [tn],
    "FP": [fp],
    "FN": [fn],
    "TP": [tp]
})

results.to_csv(
    "Hybrid_Results.csv",
    index=False
)

print("\nResults saved to: Hybrid_Results.csv")