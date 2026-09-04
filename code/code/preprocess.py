import pandas as pd

# خواندن داده
data = pd.read_csv("IDS_Dataset.csv")

print("Before:", data.shape)

# حذف مقادیر بی‌نهایت
data.replace([float("inf"), -float("inf")], pd.NA, inplace=True)

# حذف ردیف‌های دارای مقدار خالی
data.dropna(inplace=True)

# حذف داده‌های تکراری
data.drop_duplicates(inplace=True)

# حذف ستون Label (متنی)
data.drop("Label", axis=1, inplace=True)

print("After:", data.shape)

# ذخیره نسخه پاک شده
data.to_csv("IDS_Clean.csv", index=False)

print("Cleaning finished!")