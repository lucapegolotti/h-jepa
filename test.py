import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# --- Load data ---
embeddings = torch.load("./data/postprocess/jepa_embeddings.pt").numpy()
bmi_labels = torch.load("./data/postprocess/jepa_targets_bmi.pt").numpy()
caseids = torch.load("./data/preprocess/ids.pt").numpy()

# Filter to match embeddings
clinical_df = pd.read_csv("./data/raw/clinical_info.csv")[["caseid", "bmi"]].dropna()
match_caseid_set = set(clinical_df["caseid"])
match_indices = [i for i, cid in enumerate(caseids) if cid in match_caseid_set]

embeddings = embeddings[match_indices]
bmi_labels = bmi_labels[match_indices]
caseids = caseids[match_indices]

# Optional: reduce caseid to strings to avoid too many unique colors
caseid_strs = pd.factorize(caseids)[0]

# --- Dimensionality reduction ---
def reduce_and_plot(X, labels, title, palette="tab10", label_name="BMI Bucket"):
    # reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    reducer = PCA(n_components=2)  # Try this too
    reduced = reducer.fit_transform(X)

    df = pd.DataFrame({
        "X": reduced[:, 0],
        "Y": reduced[:, 1],
        "Label": labels
    })

    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df, x="X", y="Y", hue="Label", palette=palette, s=20)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title=label_name)
    plt.tight_layout()
    plt.show()

# --- Plot by BMI bucket ---
reduce_and_plot(embeddings, bmi_labels, title="JEPA Embeddings Colored by BMI Bucket", label_name="BMI Bucket")

# --- Plot by CaseID ---
reduce_and_plot(embeddings, caseid_strs, title="JEPA Embeddings Colored by Subject", palette="husl", label_name="Case ID")
