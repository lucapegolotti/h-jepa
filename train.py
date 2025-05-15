import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import random
import torch.nn.functional as F
import copy
from tqdm import tqdm

# ---- Config ----
DATA_DIR = './data/preprocess'
MODEL_DIR = './data/model'
os.makedirs(MODEL_DIR, exist_ok=True)

DATASET_PATH = os.path.join(DATA_DIR, 'preprocessed_data.pt')
MODEL_PATH = os.path.join(MODEL_DIR, 'jepa_model.pth')

WINDOW_SIZE = 600
MASK_SIZE = 100
BATCH_SIZE = 128
EMBED_DIM = 128
EPOCHS = 100
LR = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps'
EMA_BETA = 0.99

# ---- Dataset ----
class JEPA_Dataset(Dataset):
    def __init__(self, data_tensor):
        self.data = data_tensor  # shape: (N, C, T)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

# ---- Encoder ----
class ConvEncoder(nn.Module):
    def __init__(self, in_channels, embed_dim):
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

# ---- JEPA Model ----
class JEPA(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.encoder_context = ConvEncoder(in_channels, embed_dim)
        self.encoder_target = copy.deepcopy(self.encoder_context)
        for param in self.encoder_target.parameters():
            param.requires_grad = False

        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x, mask_start):
        mask = torch.ones_like(x)
        mask[:, :, mask_start:mask_start+MASK_SIZE] = 0

        context_input = x * mask
        target_input = x[:, :, mask_start:mask_start+MASK_SIZE]

        z_context = self.encoder_context(context_input)
        z_target = self.encoder_target(target_input)

        z_pred = F.normalize(self.predictor(z_context), dim=-1)
        z_target = F.normalize(z_target, dim=-1)

        return z_pred, z_target

    def update_target_encoder(self, beta=EMA_BETA):
        for param_q, param_k in zip(self.encoder_context.parameters(), self.encoder_target.parameters()):
            param_k.data = beta * param_k.data + (1 - beta) * param_q.data

# ---- Train ----
def train():
    data = torch.load(DATASET_PATH)
    dataset = JEPA_Dataset(data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = JEPA(in_channels=2, embed_dim=EMBED_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in tqdm(dataloader):
            batch = batch.to(DEVICE)
            mask_start = random.randint(0, WINDOW_SIZE - MASK_SIZE - 1)

            z_pred, z_target = model(batch, mask_start)
            loss = 2 - 2 * (z_pred * z_target).sum(dim=1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_target_encoder()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} - std(pred): {z_pred.std().item():.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
