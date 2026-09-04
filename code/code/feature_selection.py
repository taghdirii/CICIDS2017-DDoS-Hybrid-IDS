import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# =========================================================
# 1. Load Dataset
# =========================================================

print("Loading dataset...")

data = pd.read_csv("IDS_Clean.csv")

X = data.drop("Target", axis=1)
y = data["Target"]

print("Dataset shape:", X.shape)


# =========================================================
# 2. Train / Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================================================
# 3. Feature Selection
# IMPORTANT:
# Feature selection is performed ONLY on training data
# to prevent data leakage.
# =========================================================

print("\nSelecting features using training data...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# =========================================================
# 4. Feature Importance
# =========================================================

importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

top_features = importance.head(20)

print("\nTop 20 Features:")
print(top_features)


# =========================================================
# 5. Save Feature Importance
# =========================================================

importance.to_csv(
    "Feature_Importance.csv",
    index=False
)

# ذخیره فقط نام 20 ویژگی منتخب
top_features[["Feature"]].to_csv(
    "Selected_Features.csv",
    index=False
)

print("\nFeature selection completed.")
print("Saved: Feature_Importance.csv")
print("Saved: Selected_Features.csv")