# I used chatgpt to help me
from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(self, n_track: int = 10, n_waypoints: int = 3):
        super().__init__()
        self.n_track = n_track
        self.n_waypoints = n_waypoints

        # Define the layers for the MLP
        self.fc1 = nn.Linear(n_track * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_waypoints * 2)  # Output shape matches n_waypoints * 2

    def forward(self, track_left: torch.Tensor, track_right: torch.Tensor, **kwargs) -> torch.Tensor:
        # Concatenate left and right track boundaries
        x = torch.cat([track_left, track_right], dim=-1)  # Shape: (B, n_track, 4)
        x = x.view(x.size(0), -1)  # Flatten to shape (B, n_track * 4)

        # Pass through MLP layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        # Reshape to (B, n_waypoints, 2)
        return x.view(-1, self.n_waypoints, 2)


class TransformerPlanner(nn.Module):
    def __init__(self, n_track: int = 10, n_waypoints: int = 3, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model
        self.nhead = nhead

        # Ensure d_model is divisible by nhead
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        # Embedding layer for input track points
        self.track_embedding = nn.Linear(2, d_model)  # Each track point is 2D

        # Query embeddings for each waypoint
        self.query_embed = nn.Embedding(n_waypoints, d_model)  # n_waypoints x d_model

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output layer to project back to 2D waypoints
        self.output_layer = nn.Linear(d_model, 2)  # Output 2D waypoints

    def forward(self, track_left: torch.Tensor, track_right: torch.Tensor, **kwargs) -> torch.Tensor:
        # Concatenate track_left and track_right along the sequence dimension
        track = torch.cat([track_left, track_right], dim=1)  # Shape: (B, 2 * n_track, 2)
        #print(f"Track shape (before embedding): {track.shape}")

        # Embed track points
        track_embedded = self.track_embedding(track)  # Shape: (B, 2 * n_track, d_model)
        #print(f"Track embedded shape: {track_embedded.shape}")

        # Permute track_embedded to (seq_len, batch_size, d_model) for the Transformer
        track_embedded = track_embedded.permute(1, 0, 2)  # Shape: (2 * n_track, B, d_model)
        #print(f"Track permuted shape (for Transformer): {track_embedded.shape}")

        # Query embeddings for waypoints (n_waypoints, batch_size, d_model)
        query = self.query_embed.weight.unsqueeze(1).repeat(1, track_embedded.size(1), 1)  # (n_waypoints, B, d_model)
        #print(f"Query shape: {query.shape}")

        # Decode the track information using the Transformer decoder
        decoded = self.decoder(query, track_embedded)  # Shape: (n_waypoints, B, d_model)
        #print(f"Decoded shape (Transformer output): {decoded.shape}")

        # Permute decoded back to (B, n_waypoints, d_model)
        decoded = decoded.permute(1, 0, 2)  # Shape: (B, n_waypoints, d_model)
        #print(f"Decoded permuted shape (for output): {decoded.shape}")

        # Project the decoded output to 2D waypoints
        output = self.output_layer(decoded)  # Shape: (B, n_waypoints, 2)
        #print(f"Output shape: {output.shape}")

        return output  # Final output: (B, n_waypoints, 2)


class CNNPlanner(nn.Module):
    def __init__(self, n_waypoints: int = 3):
        super().__init__()
        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        # Define CNN layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 6 * 8, 128)
        self.fc2 = nn.Linear(128, n_waypoints * 2)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        # Normalize input
        x = (image - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # CNN feature extraction
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))

        # Flatten and pass through fully connected layers
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        # Reshape to (B, n_waypoints, 2)
        return x.view(-1, self.n_waypoints, 2)

MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
