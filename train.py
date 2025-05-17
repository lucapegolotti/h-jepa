import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import os
import random
import torch.nn.functional as F
import copy
from tqdm import tqdm
import time

# ---- Config ----
torch.set_float32_matmul_precision("high")

DATA_DIR = './data/preprocess'
MODEL_DIR = './data/model'
os.makedirs(MODEL_DIR, exist_ok=True)

DATASET_PATH = os.path.join(DATA_DIR, 'preprocessed_data.pt')
MODEL_PATH = os.path.join(MODEL_DIR, 'jepa_model.pth')
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'jepa_best_model.pth')

WINDOW_SIZE = 600
MASK_SIZE = 100
BATCH_SIZE = 512
EMBED_DIM = 128
EPOCHS = 1000
LR = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps'
EMA_BETA = 0.99
EARLY_STOPPING_PATIENCE = 20
TEMPERATURE = 0.1

# ---- Dataset ----
class JEPA_Dataset(Dataset):
    def __init__(self, data_tensor):
        self.data = data_tensor
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
            nn.BatchNorm1d(32, momentum=0.05),
            nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm1d(64, momentum=0.05),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm1d(128, momentum=0.05),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim)  # helps with scale stability
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

        self.target_projector = nn.Sequential(
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
        with torch.no_grad():
            z_target = self.encoder_target(target_input)
            z_target = self.target_projector(z_target)

        z_pred = F.normalize(self.predictor(z_context), dim=-1, eps=1e-6)
        z_target = F.normalize(z_target, dim=-1, eps=1e-6)

        return z_pred, z_target

    def update_target_encoder(self, beta=EMA_BETA):
        for param_q, param_k in zip(self.encoder_context.parameters(), self.encoder_target.parameters()):
            param_k.data = beta * param_k.data + (1 - beta) * param_q.data

# ---- Training ----
def run_epoch(model, dataloader, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0

    with torch.set_grad_enabled(is_train):
        for batch in dataloader:
            batch = batch.to(DEVICE)
            mask_start = random.randint(0, WINDOW_SIZE - MASK_SIZE - 1)
            z_pred, z_target = model(batch, mask_start)
            loss = -(z_pred * z_target).sum(dim=1).mean() / TEMPERATURE

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                model.update_target_encoder()

            total_loss += loss.item()

    return total_loss / len(dataloader)

# ---- Main Train Function ----
def train():
    print("📦 Loading dataset...")
    data = torch.load(DATASET_PATH, map_location='cpu')
    full_dataset = JEPA_Dataset(data)

    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=2, persistent_workers=True)

    model = JEPA(in_channels=2, embed_dim=EMBED_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float('inf')
    no_improve_count = 0

    print("🚀 Starting training...")
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        train_loss = run_epoch(model, train_loader, optimizer)
        val_loss = run_epoch(model, val_loader)

        scheduler.step()
        epoch_duration = time.time() - epoch_start

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | ⏱️ {epoch_duration:.2f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"💾 Best model saved to {BEST_MODEL_PATH}")
        else:
            no_improve_count += 1
            if no_improve_count >= EARLY_STOPPING_PATIENCE:
                print(f"⛔ Early stopping triggered after {epoch} epochs.")
                break

        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f'jepa_epoch_{epoch}.pth'))

    torch.save(model.state_dict(), MODEL_PATH)
    duration = time.time() - start_time
    print(f"\n✅ Final model saved to {MODEL_PATH}")
    print(f"⏱️ Total training time: {duration:.2f} seconds")

if __name__ == "__main__":
    train()
