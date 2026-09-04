import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# =========================================================
# 1. Load Dataset
# =========================================================

print("Loading dataset...")

data = pd.read_csv("IDS_Clean.csv")

print("Dataset shape:", data.shape)


# =========================================================
# 2. Load Selected Features
# =========================================================

selected_features = pd.read_csv(
    "Selected_Features.csv"
)["Feature"].tolist()

print("\nSelected Features:")
for i, feature in enumerate(selected_features, 1):
    print(f"{i}. {feature}")


# =========================================================
# 3. Prepare X and y
# =========================================================

X = data[selected_features]
y = data["Target"]


# =========================================================
# 4. Train / Test Split
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
# 5. Train Random Forest
# =========================================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# =========================================================
# 6. Prediction
# =========================================================

y_pred = model.predict(X_test)


# =========================================================
# 7. Evaluation
# =========================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

fpr = fp / (fp + tn)


# =========================================================
# 8. Results
# =========================================================

print("\n====================================")
print(" RANDOM FOREST RESULTS")
print("====================================")

print(f"Accuracy : {accuracy:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall   : {recall:.6f}")
print(f"F1-Score : {f1:.6f}")
print(f"FPR      : {fpr:.6f}")

print("\nPercentage:")
print(f"Accuracy : {accuracy * 100:.4f}%")
print(f"Precision: {precision * 100:.4f}%")
print(f"Recall   : {recall * 100:.4f}%")
print(f"F1-Score : {f1 * 100:.4f}%")
print(f"FPR      : {fpr * 100:.6f}%")

print("\nConfusion Matrix:")
print(cm)


# =========================================================
# 9. Save Results
# =========================================================

results = pd.DataFrame({
    "Model": ["Random Forest"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1": [f1],
    "FPR": [fpr]
})

results.to_csv(
    "Random_Forest_Results.csv",
    index=False
)

print("\nResults saved to Random_Forest_Results.csv")