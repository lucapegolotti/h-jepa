import torch
import torch.nn as nn
import copy
import torch.nn.functional as F


# ---- Encoder ----
class ConvEncoder(nn.Module):
    """
    A convolutional encoder module for processing 1D input data.

    This class implements a sequence of convolutional layers, batch normalization,
    activation functions, and pooling operations to encode input data into a fixed-size
    embedding vector. The architecture is configurable via a dictionary-based configuration.

    Attributes:
        net (nn.Sequential): A sequential container of layers that defines the encoder's
            architecture, including convolutional layers, batch normalization, activation
            functions, pooling, and a final linear layer for embedding generation.

    Args:
        config (dict): A dictionary containing the configuration parameters for the encoder.
            The expected keys in the configuration dictionary are:
                - model.in_channels (int): Number of input channels.
                - model.conv1_out_channels (int): Number of output channels for the first convolutional layer.
                - model.conv1_kernel_size (int): Kernel size for the first convolutional layer.
                - model.conv1_stride (int): Stride for the first convolutional layer.
                - model.conv1_padding (int): Padding for the first convolutional layer.
                - model.conv2_out_channels (int): Number of output channels for the second convolutional layer.
                - model.conv2_kernel_size (int): Kernel size for the second convolutional layer.
                - model.conv2_stride (int): Stride for the second convolutional layer.
                - model.conv2_padding (int): Padding for the second convolutional layer.
                - model.conv3_out_channels (int): Number of output channels for the third convolutional layer.
                - model.conv3_kernel_size (int): Kernel size for the third convolutional layer.
                - model.conv3_stride (int): Stride for the third convolutional layer.
                - model.conv3_padding (int): Padding for the third convolutional layer.
                - model.batch_norm_momentum (float): Momentum for batch normalization layers.
                - model.adaptive_pool_output_size (int): Output size for the adaptive average pooling layer.
                - model.embed_dim (int): Dimension of the final embedding vector.

    Methods:
        forward(x):
            Passes the input tensor through the encoder network and returns the resulting embedding.

    Example:
        config = {
            "model": {
                "in_channels": 1,
                "conv1_out_channels": 16,
                "conv1_kernel_size": 3,
                "conv1_stride": 1,
                "conv1_padding": 1,
                "conv2_out_channels": 32,
                "conv2_kernel_size": 3,
                "conv2_stride": 1,
                "conv2_padding": 1,
                "conv3_out_channels": 64,
                "conv3_kernel_size": 3,
                "conv3_stride": 1,
                "conv3_padding": 1,
                "batch_norm_momentum": 0.1,
                "adaptive_pool_output_size": 1,
                "embed_dim": 128,
            }
        }
        encoder = ConvEncoder(config)
        input_tensor = torch.randn(8, 1, 128)  # Batch of 8, 1 channel, 128 length
        output = encoder(input_tensor)
        print(output.shape)  # Output shape: (8, 128)
    """

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
    """
    Joint Embedding Predictive Architecture (JEPA) model.

    This class implements a JEPA model, which consists of two encoders (context and target),
    a predictor, and a target projector. The model is designed to predict a normalized
    embedding of a masked target region from the context region of the input.

    Attributes:
        config (dict): Configuration dictionary containing model and training parameters.
        encoder_context (nn.Module): Encoder for the context region of the input.
        encoder_target (nn.Module): Encoder for the target region of the input, initialized
            as a deep copy of `encoder_context` and updated using an exponential moving average.
        predictor (nn.Sequential): A feedforward network that predicts the target embedding
            from the context embedding.
        target_projector (nn.Sequential): A feedforward network that projects the target
            embedding to the same space as the predicted embedding.

    Methods:
        forward(x, mask_start):
            Computes the predicted and target embeddings for the input tensor `x`.

        update_target_encoder(beta):
            Updates the target encoder parameters using an exponential moving average
            with the given momentum `beta`.
    """

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
