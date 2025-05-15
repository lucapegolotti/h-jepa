import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
from tqdm import tqdm 

# --- CONFIG ---
DATASET_PATH = './data/preprocess/preprocessed_data.pt'
IDS_PATH = './data/preprocess/preprocessed_ids.pt'
MODEL_PATH = './data/model/jepa_model.pth'
CLINICAL_PATH = './data/raw/clinical_info.csv'
TARGET_VARIABLE = 'age'
BATCH_SIZE = 512
EMBED_CACHE = './postprocess/jepa_embeddings.pt'
TARGET_CACHE = './postprocess/jepa_targets.pt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Ensure output dir exists ---
os.makedirs('./data/postprocess', exist_ok=True)

# --- Dataset Loader ---
class JEPAEmbeddingDataset(Dataset):
    def __init__(self, tensor_data):
        self.data = tensor_data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# --- Encoder Only ---
class ConvEncoder(nn.Module):
    def __init__(self, in_channels=2, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, 5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, embed_dim)
        )
    def forward(self, x):
        return self.net(x)

# --- Plotting ---
def plot_predictions(y_true, y_pred, target_var):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.3, edgecolor='k')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label="Perfect")
    plt.xlabel('Ground Truth')
    plt.ylabel('Predicted')
    r2 = r2_score(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    plt.title(f"{target_var} Prediction\nR² = {r2:.2f}, Corr = {corr:.2f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Main ---
def main():
    print("📦 Loading raw data...")
    raw_data = torch.load(DATASET_PATH)
    caseids = torch.load(IDS_PATH).numpy()
    clinical_df = pd.read_csv(CLINICAL_PATH)
    clinical_df = clinical_df[['caseid', TARGET_VARIABLE]].dropna()

    # Map caseid → target
    caseid_to_target = dict(zip(clinical_df['caseid'], clinical_df[TARGET_VARIABLE]))
    matched_targets = []
    matched_indices = []

    for idx, cid in enumerate(caseids):
        if cid in caseid_to_target:
            matched_targets.append(caseid_to_target[cid])
            matched_indices.append(idx)

    if not matched_indices:
        raise RuntimeError("❌ No caseids matched with target variable.")

    filtered_data = raw_data[matched_indices]
    y = np.array(matched_targets, dtype=np.float32)

    # === Cache check ===
    if os.path.exists(EMBED_CACHE) and os.path.exists(TARGET_CACHE):
        print("💾 Loading cached embeddings...")
        X = torch.load(EMBED_CACHE).numpy()
        y = torch.load(TARGET_CACHE).numpy()
    else:
        print("🧠 Extracting embeddings from JEPA model...")
        encoder = ConvEncoder()
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        encoder.load_state_dict({k.replace('encoder_context.', ''): v for k, v in state_dict.items() if 'encoder_context' in k})
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
        torch.save(torch.tensor(X), EMBED_CACHE)
        torch.save(torch.tensor(y), TARGET_CACHE)
        print(f"✅ Saved embeddings to {EMBED_CACHE}")

    # === Regression (Nonlinear) ===
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"📈 Training MLP regressor for '{TARGET_VARIABLE}'...")
    reg = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=20,
        early_stopping=True,
        random_state=42
    )
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    # === Evaluation ===
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    corr = np.corrcoef(y_test, y_pred)[0, 1]

    print(f"\n📊 Evaluation for '{TARGET_VARIABLE}':")
    print(f"    MSE  : {mse:.4f}")
    print(f"    R²   : {r2:.4f}")
    print(f"    Corr : {corr:.4f}")

    # === Plot ===
    plot_predictions(y_test, y_pred, TARGET_VARIABLE)

if __name__ == "__main__":
    main()
