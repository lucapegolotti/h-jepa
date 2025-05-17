import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, top_k_accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns
from train import ConvEncoder

# --- CONFIG ---
DATASET_PATH = "./data/preprocess/preprocessed_data.pt"
IDS_PATH = "./data/preprocess/preprocessed_ids.pt"
MODEL_PATH = "./data/model/jepa_best_model.pth"
CLINICAL_PATH = "./data/raw/clinical_info.csv"
TARGET_VARIABLE = "age"
BATCH_SIZE = 512
EMBED_DIM = 128
EMBED_CACHE = "./data/postprocess/jepa_embeddings.pt"
TARGET_CACHE = "./data/postprocess/jepa_targets.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "mps"
NUM_BUCKETS = 10

# --- Ensure output dir exists ---
os.makedirs("./data/postprocess", exist_ok=True)


# --- Dataset Loader ---
class JEPAEmbeddingDataset(Dataset):
    def __init__(self, tensor_data):
        self.data = tensor_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# --- Plotting ---
def plot_confusion_matrix(conf, labels):
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        conf,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted Age Bucket")
    plt.ylabel("True Age Bucket")
    plt.title("Normalized Confusion Matrix (Age Buckets)")
    plt.tight_layout()
    plt.show()


# --- Main ---
def main():
    print("📦 Loading raw data...")
    raw_data = torch.load(DATASET_PATH)
    caseids = torch.load(IDS_PATH).numpy()
    clinical_df = pd.read_csv(CLINICAL_PATH)
    clinical_df = clinical_df[["caseid", TARGET_VARIABLE]].dropna()

    # Map caseid → target
    caseid_to_target = dict(zip(clinical_df["caseid"], clinical_df[TARGET_VARIABLE]))
    matched_targets = []
    matched_indices = []

    for idx, cid in enumerate(caseids):
        if cid in caseid_to_target:
            matched_targets.append(caseid_to_target[cid])
            matched_indices.append(idx)

    if not matched_indices:
        raise RuntimeError("❌ No caseids matched with target variable.")

    filtered_data = raw_data[matched_indices]
    raw_ages = np.array(matched_targets, dtype=np.float32)

    # === Bucketize ages: 0–10, 10–20, ..., 90+ → labels 0–9
    y_bucketed = (raw_ages // 10).astype(int)
    y_bucketed = np.clip(y_bucketed, 0, NUM_BUCKETS - 1)

    # === Cache check ===
    if os.path.exists(EMBED_CACHE) and os.path.exists(TARGET_CACHE):
        print("💾 Loading cached embeddings...")
        X = torch.load(EMBED_CACHE).numpy()
        y = torch.Tensor(y_bucketed)
    else:
        print("🧠 Extracting embeddings from JEPA model...")
        encoder = ConvEncoder(in_channels=2, embed_dim=EMBED_DIM)
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        encoder.load_state_dict(
            {
                k.replace("encoder_context.", ""): v
                for k, v in state_dict.items()
                if "encoder_context" in k
            }
        )
        encoder.to(DEVICE)
        encoder.eval()

        dataset = JEPAEmbeddingDataset(filtered_data)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        all_embeddings = []
        with torch.no_grad():
            for batch in tqdm(loader):
                batch = batch.to(DEVICE)
                emb = encoder(batch)
                all_embeddings.append(emb.cpu())

        X = torch.cat(all_embeddings).numpy()
        y = y_bucketed
        torch.save(torch.tensor(X), EMBED_CACHE)
        torch.save(torch.tensor(y), TARGET_CACHE)
        print(f"✅ Saved embeddings and labels.")

    # === Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"📈 Training weighted MLP classifier for '{TARGET_VARIABLE}' buckets...")
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=1024,
        learning_rate_init=1e-3,
        max_iter=20,
        early_stopping=True,
        random_state=42,
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    # === Evaluation
    acc = accuracy_score(y_test, y_pred)
    top2_acc = top_k_accuracy_score(y_test, y_prob, k=2)
    conf_raw = confusion_matrix(y_test, y_pred)
    conf_norm = conf_raw / conf_raw.sum(axis=1, keepdims=True)

    print(f"\n📊 Classification results for '{TARGET_VARIABLE}_bucket':")
    print(f"    Accuracy     : {acc:.4f}")
    print(f"    Top-2 Accuracy: {top2_acc:.4f}")

    # === Plot ===
    age_labels = [f"{i*10}-{i*10+9}" for i in range(9)] + ["90+"]
    plot_confusion_matrix(conf_norm, labels=age_labels)


if __name__ == "__main__":
    main()
