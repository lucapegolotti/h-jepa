import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# --- Config ---
EMBED_PATH = "./data/postprocess/jepa_embeddings.pt"
IDS_PATH = "./data/preprocess/preprocessed_ids.pt"
SAMPLE_LIMIT = 5000  # Limit for t-SNE performance

# --- Load cached data ---
print("📥 Loading embeddings and IDs...")
X = torch.load(EMBED_PATH).numpy()
ids = torch.load(IDS_PATH).numpy()

# --- Reduce size if needed ---
if len(X) > SAMPLE_LIMIT:
    idx = np.random.choice(len(X), SAMPLE_LIMIT, replace=False)
    X = X[idx]
    ids = ids[idx]

# --- Run t-SNE ---
print("🔍 Running t-SNE...")
tsne = TSNE(
    n_components=2, perplexity=30, learning_rate="auto", init="pca", random_state=42
)
X_2d = tsne.fit_transform(X)

# --- Plot ---
print("📊 Plotting projection...")
plt.figure(figsize=(10, 8))
palette = sns.color_palette("hls", n_colors=len(np.unique(ids)))
sns.scatterplot(
    x=X_2d[:, 0],
    y=X_2d[:, 1],
    hue=ids,
    palette=palette,
    s=10,
    linewidth=0,
    legend=False,
)
plt.title("t-SNE Projection of JEPA Embeddings by Subject ID")
plt.xlabel("t-SNE Dim 1")
plt.ylabel("t-SNE Dim 2")
plt.tight_layout()
plt.show()
