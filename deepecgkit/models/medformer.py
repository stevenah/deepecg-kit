import torch
from torch import nn

from deepecgkit.registry import register_model


class PatchEmbedding1D(nn.Module):
    """
    Converts 1D signal into patch embeddings at a given patch size.

    Args:
        input_channels: Number of input ECG channels
        d_model: Output embedding dimension
        patch_size: Size of each non-overlapping patch
    """

    def __init__(self, input_channels: int, d_model: int, patch_size: int):
        super().__init__()
        self.proj = nn.Conv1d(input_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.transpose(1, 2)


class CrossChannelAttention(nn.Module):
    """
    Cross-channel attention for multi-lead ECG signals.

    For single-lead ECG this acts as an identity-like operation.

    Args:
        d_model: Embedding dimension
        nhead: Number of attention heads (default: 4)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x, _ = self.attn(x, x, x)
        return self.norm(x + residual)


class IntraGranularityBlock(nn.Module):
    """
    Self-attention within patches of one granularity level.

    Args:
        d_model: Embedding dimension
        nhead: Number of attention heads (default: 8)
        dim_feedforward: Feedforward dimension (default: 256)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class InterGranularityAttention(nn.Module):
    """
    Cross-attention to fuse information across different granularities.

    Pools each granularity to a single vector, applies self-attention
    across granularities, and returns updated representations.

    Args:
        d_model: Embedding dimension
        num_granularities: Number of granularity levels
        nhead: Number of attention heads (default: 4)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        d_model: int,
        num_granularities: int,
        nhead: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.num_granularities = num_granularities

    def forward(self, granularity_features: list[torch.Tensor]) -> list[torch.Tensor]:
        pooled = [f.mean(dim=1) for f in granularity_features]
        stacked = torch.stack(pooled, dim=1)
        attn_out, _ = self.attn(stacked, stacked, stacked)
        attn_out = self.norm(attn_out + stacked)

        updated = []
        for i, feat in enumerate(granularity_features):
            scale = attn_out[:, i].unsqueeze(1)
            updated.append(feat + scale)
        return updated


@register_model(
    name="medformer",
    description="Multi-granularity patching Transformer for medical time series (NeurIPS 2024)",
)
class Medformer(nn.Module):
    """
    Medformer: Multi-Granularity Patching Transformer for medical time series.

    Uses multiple patch sizes to capture fine, medium, and coarse temporal
    patterns, with intra-granularity self-attention and inter-granularity
    cross-attention for information fusion.

    Args:
        input_channels: Number of input channels (default: 1 for single-lead ECG)
        output_size: Number of output classes
        d_model: Transformer model dimension (default: 128)
        patch_sizes: Tuple of patch sizes for different granularities (default: (10, 25, 50))
        num_encoder_layers: Number of encoder layers (default: 2)
        nhead: Number of attention heads (default: 8)
        dim_feedforward: Feedforward dimension (default: 256)
        dropout_rate: Dropout probability (default: 0.1)
        max_patches: Maximum number of patches per granularity (default: 500)

    Example:
        >>> model = Medformer(input_channels=1, output_size=4)
        >>> x = torch.randn(32, 1, 3000)
        >>> output = model(x)
        >>> print(output.shape)

        >>> features = model.extract_features(x)
        >>> print(features.shape)  # (32, 384)
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_size: int = 4,
        d_model: int = 128,
        patch_sizes: tuple[int, ...] = (10, 25, 50),
        num_encoder_layers: int = 2,
        nhead: int = 8,
        dim_feedforward: int = 256,
        dropout_rate: float = 0.1,
        max_patches: int = 500,
    ):
        super().__init__()

        self.patch_sizes = patch_sizes
        self.d_model = d_model
        num_granularities = len(patch_sizes)

        self.patch_embeddings = nn.ModuleList(
            [PatchEmbedding1D(input_channels, d_model, ps) for ps in patch_sizes]
        )

        self.pos_embeddings = nn.ParameterList(
            [nn.Parameter(torch.randn(1, max_patches, d_model) * 0.02) for _ in patch_sizes]
        )

        self.cross_channel_attn = CrossChannelAttention(
            d_model,
            nhead=min(4, nhead),
            dropout=dropout_rate,
        )

        self.intra_blocks = nn.ModuleList()
        self.inter_blocks = nn.ModuleList()
        for _ in range(num_encoder_layers):
            self.intra_blocks.append(
                nn.ModuleList(
                    [
                        IntraGranularityBlock(d_model, nhead, dim_feedforward, dropout_rate)
                        for _ in patch_sizes
                    ]
                )
            )
            self.inter_blocks.append(
                InterGranularityAttention(d_model, num_granularities, min(4, nhead), dropout_rate)
            )

        self.dropout = nn.Dropout(dropout_rate)
        self._feature_dim = d_model * num_granularities

        self.classifier = nn.Linear(self._feature_dim, output_size)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        granularity_features = []
        for patch_emb, pos_emb in zip(self.patch_embeddings, self.pos_embeddings):
            patches = patch_emb(x)
            num_patches = patches.size(1)
            patches = patches + pos_emb[:, :num_patches]
            granularity_features.append(patches)

        for intra_layer, inter_layer in zip(self.intra_blocks, self.inter_blocks):
            for i, intra_block in enumerate(intra_layer):
                granularity_features[i] = intra_block(granularity_features[i])
            granularity_features = inter_layer(granularity_features)

        pooled = [f.mean(dim=1) for f in granularity_features]
        return torch.cat(pooled, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extract_features(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
