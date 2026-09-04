import pandas as pd

normal_file = "Monday-WorkingHours.pcap_ISCX.csv"
attack_file = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

print("Loading files...")

normal = pd.read_csv(normal_file)
attack = pd.read_csv(attack_file)

print("Normal data:", normal.shape)
print("Attack data:", attack.shape)

# ترکیب داده‌ها
data = pd.concat([normal, attack], ignore_index=True)

# اصلاح نام ستون‌ها
data.columns = data.columns.str.strip()

# اصلاح Label
data["Label"] = data["Label"].str.strip()

# تبدیل کلاس‌ها
data["Target"] = data["Label"].apply(
    lambda x: 0 if x == "BENIGN" else 1
)

print("\nClass distribution:")
print(data["Target"].value_counts())

# ذخیره
data.to_csv("IDS_Dataset.csv", index=False)

print("\nDone!")