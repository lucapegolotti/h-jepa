import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# --- Config ---
CLINICAL_PATH = "./data/raw/clinical_info.csv"
TARGET_VARIABLE = "bmi"
BMI_BUCKETS = [(0, 18.5), (18.5, 25), (25, 30), (30, 35), (35, 40), (40, 100)]
BMI_LABELS = ["Underweight", "Normal", "Overweight", "Obese I", "Obese II", "Obese III"]
BUCKET_COLORS = ["#56B4E9", "#009E73", "#F0E442", "#E69F00", "#D55E00", "#CC79A7"]

# --- Load data ---
df = pd.read_csv(CLINICAL_PATH)
df = df[["caseid", TARGET_VARIABLE]].dropna()
bmis = df[TARGET_VARIABLE].values


# --- Assign bucket index to each BMI ---
def bucketize_bmi(bmi):
    for i, (low, high) in enumerate(BMI_BUCKETS):
        if low <= bmi < high:
            return i
    return len(BMI_BUCKETS) - 1


bucket_indices = np.array([bucketize_bmi(b) for b in bmis])
colors = [BUCKET_COLORS[i] for i in bucket_indices]

# --- Plot ---
plt.figure(figsize=(12, 6))
sns.kdeplot(bmis, color="black", label="KDE", linewidth=2)

# Plot histogram with colored bars
bins = np.linspace(min(bmis), max(bmis), 50)
bin_centers = 0.5 * (bins[1:] + bins[:-1])
bin_counts, _ = np.histogram(bmis, bins)

# Assign a dominant bucket color to each bin based on the midpoint
bin_colors = []
for center in bin_centers:
    idx = bucketize_bmi(center)
    bin_colors.append(BUCKET_COLORS[idx])

plt.bar(
    bin_centers,
    bin_counts,
    width=(bins[1] - bins[0]),
    color=bin_colors,
    edgecolor="black",
    alpha=0.8,
)

# Add legend manually
for label, color in zip(BMI_LABELS, BUCKET_COLORS):
    plt.bar(0, 0, color=color, label=label)

plt.title("BMI Distribution in the Population")
plt.xlabel("BMI")
plt.ylabel("Count")
plt.legend(title="BMI Categories")
plt.grid(True)
plt.tight_layout()
plt.show()
