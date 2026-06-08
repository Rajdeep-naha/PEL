import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from .config import TEXT_EMBED_DIM, TEXT_HEADS, TEXT_LAYERS, TEXT_MAX_LEN, TEXT_VOCAB_SIZE


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_resnet_encoder(modality="vision"):
    """Return a ResNet-18 encoder ending at global average pooling."""
    model = tv_models.resnet18(weights=None)
    dim = model.fc.in_features
    encoder = nn.Sequential(*(list(model.children())[:-1]))
    return encoder, dim


class M11(nn.Module):
    """Compact 1D CNN encoder used for audio downstream and pretraining."""

    def __init__(self, n_input=1, n_output=35, stride=4, n_channel=64):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)

        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)

        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)

        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.AdaptiveAvgPool1d(1)
        self.feature_dim = 2 * n_channel
        self.n_output = n_output

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() == 4:
            x = x.view(x.size(0), 1, -1)

        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        return x


class TextPeLEncoder(nn.Module):
    def __init__(self, pad_idx=1):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(TEXT_VOCAB_SIZE, TEXT_EMBED_DIM)
        self.pos_embedding = nn.Parameter(torch.randn(1, TEXT_MAX_LEN, TEXT_EMBED_DIM))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=TEXT_EMBED_DIM,
            nhead=TEXT_HEADS,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=TEXT_LAYERS)

    def forward(self, x):
        padding_mask = x == self.pad_idx
        x = self.embedding(x) + self.pos_embedding[:, : x.size(1), :]
        out = self.transformer(x, src_key_padding_mask=padding_mask)
        return out, padding_mask


def pool_encoder_output(encoder_output):
    """Single downstream pooling rule used by Scratch, FineTune, and PeL-Frozen."""
    if isinstance(encoder_output, tuple):
        hidden, padding_mask = encoder_output
        valid = (~padding_mask).float().unsqueeze(-1)
        summed = torch.sum(hidden * valid, dim=1)
        count = torch.clamp(valid.sum(dim=1), min=1e-9)
        return summed / count

    return torch.flatten(encoder_output, 1)


class PooledEncoderClassifier(nn.Module):
    """Encoder + shared pooling + one linear classifier."""

    def __init__(self, encoder, feature_dim, num_classes):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.head(pool_encoder_output(self.encoder(x)))


class SimCLRProjector(nn.Module):
    def __init__(self, encoder, feature_dim, projection_dim=128):
        super().__init__()
        self.encoder = encoder
        self.net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, projection_dim),
        )

    def forward(self, x):
        return self.net(pool_encoder_output(self.encoder(x)))


class ProjectionMLP(nn.Module):
    """Legacy helper retained for old appendix scripts; not used in the rerun tracks."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x):
        return self.net(pool_encoder_output(x) if isinstance(x, tuple) or x.dim() > 2 else x)
