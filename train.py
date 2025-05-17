import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import os
import random
import torch.nn.functional as F
import copy
from tqdm import tqdm
import time
import yaml
from core.jepa_dataset import JEPA_Dataset
from core.model import JEPA

# ---- Config ----
torch.set_float32_matmul_precision("high")

DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)

# ---- Training ----
def run_epoch(config, model, dataloader, optimizer=None):
    """
    Executes a single epoch of training or evaluation for the given model.

    Args:
        config (dict): Configuration dictionary containing training parameters such as
            "window_size", "mask_size", "temperature", and "ema_beta".
        model (torch.nn.Module): The model to train or evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader providing the input data.
        optimizer (torch.optim.Optimizer, optional): Optimizer for training. If None, 
            the function runs in evaluation mode.

    Returns:
        float: The average loss over all batches in the dataloader.

    Notes:
        - In training mode, the function updates the model's parameters using the optimizer
          and applies gradient clipping with a maximum norm of 1.0.
        - The function also updates the target encoder of the model using an exponential
          moving average (EMA) with the beta value specified in the configuration.
        - In evaluation mode, gradients are disabled, and no parameter updates are performed.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0

    with torch.set_grad_enabled(is_train):
        for batch in dataloader:
            batch = batch.to(DEVICE)
            mask_start = random.randint(
                0,
                config["training"]["window_size"] - config["training"]["mask_size"] - 1,
            )
            z_pred, z_target = model(batch, mask_start)
            loss = (
                -(z_pred * z_target).sum(dim=1).mean()
                / config["training"]["temperature"]
            )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                model.update_target_encoder(config["training"]["ema_beta"])

            total_loss += loss.item()

    return total_loss / len(dataloader)


# ---- Main Train Function ----
def train(config):
    """
    Trains a JEPA model using the provided configuration.

    This function handles the entire training process, including loading the dataset,
    splitting it into training and validation sets, initializing the model, optimizer,
    and learning rate scheduler, and performing training epochs. It also implements
    early stopping and saves the best model and periodic checkpoints.

    Args:
        config (dict): A dictionary containing the configuration for training. It should
            include the following keys:
            - "paths": A dictionary with paths for preprocessed data and model directory.
                - "preprocessed_data" (str): Path to the preprocessed dataset file.
                - "model_dir" (str): Directory to save the model checkpoints.
            - "training": A dictionary with training parameters.
                - "batch_size" (int): Batch size for training and validation.
                - "learning_rate" (float): Learning rate for the optimizer.
                - "epochs" (int): Number of training epochs.
                - "early_stopping_patience" (int): Number of epochs to wait for improvement
                  in validation loss before triggering early stopping.

    Returns:
        None
    """
    print("📦 Loading dataset...")
    data = torch.load(config["paths"]["preprocessed_data"], map_location="cpu")
    full_dataset = JEPA_Dataset(data)

    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        config["training"]["batch_size"],
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
    )

    model = JEPA(config).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["training"]["learning_rate"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["epochs"]
    )

    best_val_loss = float("inf")
    no_improve_count = 0

    print("🚀 Starting training...")
    start_time = time.time()

    for epoch in range(1, config["training"]["epochs"] + 1):
        epoch_start = time.time()

        train_loss = run_epoch(config, model, train_loader, optimizer)
        val_loss = run_epoch(config, model, val_loader)

        scheduler.step()
        epoch_duration = time.time() - epoch_start

        print(
            f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | ⏱️ {epoch_duration:.2f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
            model_path = os.path.join(
                config["paths"]["model_dir"], f"jepa_best_model.pth"
            )
            torch.save(model.state_dict(), model_path)
            print(f"💾 Best model saved to {model_path}")
        else:
            no_improve_count += 1
            if no_improve_count >= config["training"]["early_stopping_patience"]:
                print(f"⛔ Early stopping triggered after {epoch} epochs.")
                break

        if epoch % 10 == 0:
            model_path = (
                os.path.join(config["paths"]["model_dir"], f"jepa_epoch_{epoch}.pth"),
            )
            torch.save(model.state_dict(), model_path)

    model_path = os.path.join(config["paths"]["model_dir"], "jepa_model_final.pth")
    torch.save(model.state_dict(), model_path)
    duration = time.time() - start_time
    print(f"\n✅ Final model saved to jepa_model_final.pth")
    print(f"⏱️ Total training time: {duration:.2f} seconds")


if __name__ == "__main__":
    CONFIG_PATH = "config.yml"

    print("📖 Loading configuration...")
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)

    train(config)
