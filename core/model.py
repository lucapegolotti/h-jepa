import torch
import torch.nn as nn
import copy
import torch.nn.functional as F


# ---- Encoder ----
class ConvEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                config["model"]["in_channels"],
                config["model"]["conv1_out_channels"],
                config["model"]["conv1_kernel_size"],
                stride=config["model"]["conv1_stride"],
                padding=config["model"]["conv1_padding"],
            ),
            nn.BatchNorm1d(
                config["model"]["conv1_out_channels"],
                momentum=config["model"]["batch_norm_momentum"],
            ),
            nn.ReLU(),
            nn.Conv1d(
                config["model"]["conv1_out_channels"],
                config["model"]["conv2_out_channels"],
                config["model"]["conv2_kernel_size"],
                stride=config["model"]["conv2_stride"],
                padding=config["model"]["conv2_padding"],
            ),
            nn.BatchNorm1d(
                config["model"]["conv2_out_channels"],
                momentum=config["model"]["batch_norm_momentum"],
            ),
            nn.ReLU(),
            nn.Conv1d(
                config["model"]["conv2_out_channels"],
                config["model"]["conv3_out_channels"],
                config["model"]["conv3_kernel_size"],
                stride=config["model"]["conv3_stride"],
                padding=config["model"]["conv3_padding"],
            ),
            nn.BatchNorm1d(
                config["model"]["conv3_out_channels"],
                momentum=config["model"]["batch_norm_momentum"],
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(config["model"]["adaptive_pool_output_size"]),
            nn.Flatten(),
            nn.Linear(
                config["model"]["conv3_out_channels"], config["model"]["embed_dim"]
            ),
            nn.LayerNorm(config["model"]["embed_dim"]),  # helps with scale stability
        )

    def forward(self, x):
        return self.net(x)


# ---- JEPA Model ----
class JEPA(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.encoder_context = ConvEncoder(config)
        self.encoder_target = copy.deepcopy(self.encoder_context)
        for param in self.encoder_target.parameters():
            param.requires_grad = False

        self.predictor = nn.Sequential(
            nn.Linear(config["model"]["embed_dim"], config["model"]["embed_dim"]),
            nn.ReLU(),
            nn.Linear(config["model"]["embed_dim"], config["model"]["embed_dim"]),
        )

        self.target_projector = nn.Sequential(
            nn.Linear(config["model"]["embed_dim"], config["model"]["embed_dim"]),
            nn.ReLU(),
            nn.Linear(config["model"]["embed_dim"], config["model"]["embed_dim"]),
        )

    def forward(self, x, mask_start):
        mask = torch.ones_like(x)
        mask[:, :, mask_start : mask_start + self.config["training"]["mask_size"]] = 0

        context_input = x * mask
        target_input = x[
            :, :, mask_start : mask_start + self.config["training"]["mask_size"]
        ]

        z_context = self.encoder_context(context_input)
        with torch.no_grad():
            z_target = self.encoder_target(target_input)
            z_target = self.target_projector(z_target)

        z_pred = F.normalize(self.predictor(z_context), dim=-1, eps=1e-6)
        z_target = F.normalize(z_target, dim=-1, eps=1e-6)

        return z_pred, z_target

    def update_target_encoder(self, beta):
        for param_q, param_k in zip(
            self.encoder_context.parameters(), self.encoder_target.parameters()
        ):
            param_k.data = beta * param_k.data + (1 - beta) * param_q.data
