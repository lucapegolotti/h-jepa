import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns

# --- CONFIG ---
DATASET_PATH = "./data/preprocess/data.pt"
IDS_PATH = "./data/preprocess/ids.pt"
MODEL_PATH = "./data/model/jepa_best_model.pth"
CLINICAL_PATH = "./data/raw/clinical_info.csv"
TARGET_VARIABLE = "bmi"
BATCH_SIZE = 512
EMBED_DIM = 128
EMBED_CACHE = "./data/postprocess/jepa_embeddings.pt"
TARGET_CACHE = "./data/postprocess/jepa_targets_bmi.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")

BMI_BUCKETS = [(0, 18.5), (18.5, 25), (25, 30), (30, 100)]
BMI_LABELS = ["Underweight", "Normal", "Overweight", "Obese"]
NUM_BUCKETS = len(BMI_BUCKETS)

os.makedirs("./data/postprocess", exist_ok=True)

# ---- Encoder ----
class ConvEncoder(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, 5, stride=2, padding=2),
            nn.BatchNorm1d(32, momentum=0.05),
            nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm1d(64, momentum=0.05),
            nn.ReLU(),
            nn.Conv1d(64, 128, 2, stride=2, padding=1),
            nn.BatchNorm1d(128, momentum=0.05),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x):
        return self.net(x)


class JEPAEmbeddingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLPClassifierTorch(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


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
    plt.xlabel("Predicted BMI Category")
    plt.ylabel("True BMI Category")
    plt.title("Normalized Confusion Matrix (BMI Buckets)")
    plt.tight_layout()
    plt.show()


def main():
    print("📦 Loading raw data...")
    raw_data = torch.load(DATASET_PATH)
    caseids = torch.load(IDS_PATH).numpy()
    clinical_df = pd.read_csv(CLINICAL_PATH)
    clinical_df = clinical_df[["caseid", TARGET_VARIABLE]].dropna()

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
    raw_bmi = np.array(matched_targets, dtype=np.float32)

    def bucketize_bmi(bmi):
        for i, (low, high) in enumerate(BMI_BUCKETS):
            if low <= bmi < high:
                return i
        return NUM_BUCKETS - 1

    y_bucketed = np.array([bucketize_bmi(b) for b in raw_bmi], dtype=np.int64)

    # Build DataFrame to link indices to caseids and targets
    match_df = pd.DataFrame({
        "idx": matched_indices,
        "caseid": caseids[matched_indices],
        "target": y_bucketed
    })

    # Show overall class distribution
    print("📊 Overall class distribution:")
    print(match_df["target"].value_counts(normalize=True).sort_index().rename(lambda x: BMI_LABELS[x]))

    # Stratified split based on caseid-bucket pairing
    caseid_bucket_df = match_df.drop_duplicates(subset="caseid")[["caseid", "target"]]
    train_caseids, test_caseids = train_test_split(
        caseid_bucket_df["caseid"],
        test_size=0.2,
        random_state=42,
        stratify=caseid_bucket_df["target"]
    )

    train_df = match_df[match_df["caseid"].isin(train_caseids)]
    test_df = match_df[match_df["caseid"].isin(test_caseids)]

    # Show split distributions
    print("\n📊 Train class distribution:")
    print(train_df["target"].value_counts(normalize=True).sort_index().rename(lambda x: BMI_LABELS[x]))
    print("\n📊 Test class distribution:")
    print(test_df["target"].value_counts(normalize=True).sort_index().rename(lambda x: BMI_LABELS[x]))

    # Load or compute embeddings
    if os.path.exists(EMBED_CACHE):
        print("💾 Loading cached embeddings...")
        X_all = torch.load(EMBED_CACHE).numpy()
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

        dataset = torch.utils.data.TensorDataset(filtered_data)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        all_embeddings = []
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(DEVICE)
                emb = encoder(batch)
                all_embeddings.append(emb.cpu())

        X_all = torch.cat(all_embeddings).numpy()
        torch.save(torch.tensor(X_all), EMBED_CACHE)
        torch.save(torch.tensor(y_bucketed), TARGET_CACHE)
        print("✅ Saved embeddings and labels.")

    # Final split based on index
    X_train = X_all[train_df["idx"].values]
    y_train = train_df["target"].values
    X_test = X_all[test_df["idx"].values]
    y_test = test_df["target"].values

    # Dataset
    train_dataset = JEPAEmbeddingDataset(X_train, y_train)
    test_dataset = JEPAEmbeddingDataset(X_test, y_test)

    # Compute stable class weights
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_BUCKETS),
        y=y_train
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    # Weighted sampling for training
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=128, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Model + Optimizer + Loss
    model = MLPClassifierTorch(input_dim=EMBED_DIM, num_classes=NUM_BUCKETS).to(DEVICE)
    criterion = nn.CrossEntropyLoss()# (weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("🚀 Training...")
    model.train()
    for epoch in range(10):
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    print("🔍 Evaluating...")
    model.eval()
    all_preds, all_probs, all_targets = [], [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())
            all_targets.append(yb)

    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_targets).numpy()

    acc = accuracy_score(y_true, y_pred)
    top2_acc = top_k_accuracy_score(y_true, y_prob, k=2)
    conf_raw = confusion_matrix(y_true, y_pred)
    conf_norm = conf_raw / conf_raw.sum(axis=1, keepdims=True)

    print(f"\n📊 Classification results for '{TARGET_VARIABLE}_bucket':")
    print(f"    Accuracy     : {acc:.4f}")
    print(f"    Top-2 Accuracy: {top2_acc:.4f}")

    plot_confusion_matrix(conf_norm, labels=BMI_LABELS)


if __name__ == "__main__":
    main()
