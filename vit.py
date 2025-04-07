import torch
from torch import nn
from einops.layers.torch import Rearrange


# Converts the image into patches and projects them to a higher-dimensional space
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, hidden_size, img_size):
        super().__init__()
        self.in_channels = in_channels

        # Ensure image dimensions are divisible by patch size
        assert img_size[-1] % patch_size == 0 and img_size[-2] % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        # Output channel = total pixels in image / patch area
        self.out_channel = ((img_size[-3]*img_size[-2]*img_size[-1])//patch_size**2)
        self.hidden_size = hidden_size

        # Convolution to split image into patches
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=self.out_channel, kernel_size=patch_size, stride=patch_size, padding=0)

        # Linear projection of each patch to the hidden dimension
        self.linear = nn.Linear(in_features=self.out_channel, out_features=self.hidden_size)

    def forward(self, x):
        x = self.conv(x)  # Convert image to patch embeddings
        x = torch.flatten(x, start_dim=2, end_dim=3)  # Flatten spatial dimensions
        x = x.permute(0, 2, 1)  # Rearrange to (Batch, num_patches, features)
        x = self.linear(x)  # Project to hidden size
        return x


# Generate sinusoidal positional embeddings
def get_positional_encoding(embedding_size, sequence):
    embedding_range_half = torch.arange(0, int(embedding_size/2), step=1)
    denominator = 10000 ** ((2 * embedding_range_half) / embedding_size)

    # Expand sequence positions and apply sinusoidal transformation
    timesteps_dimensionality_half = sequence[:, None].repeat(1, embedding_size // 2) / denominator
    cos_seq = torch.cos(timesteps_dimensionality_half)
    sin_seq = torch.sin(timesteps_dimensionality_half)

    positional_embed = torch.cat([cos_seq, sin_seq], dim=-1)
    return positional_embed


# Adds class token and positional embeddings
class Embeddings(nn.Module):
    def __init__(self, hidden_size, n_patches):
        super().__init__()
        # Learnable classification token
        self.classification_token = nn.Parameter(torch.rand(1, hidden_size))

        # Fixed positional embedding (non-trainable)
        self.positional_embedding = nn.Parameter(get_positional_encoding(hidden_size, torch.arange(0, n_patches + 1)))
        self.positional_embedding.requires_grad = False

    def forward(self, x):
        batch_size, n, _ = x.shape
        cls_token = self.classification_token.expand(batch_size, 1, -1)  # Duplicate classification token across batch
        x = torch.cat((cls_token, x), dim=1)  # Prepend class token
        x = x + self.positional_embedding  # Add positional embedding
        return x


# Single Transformer encoder block
class Encoder(nn.Module):
    def __init__(self, hidden_size, n_heads):
        super().__init__()
        self.hidden_size = hidden_size

        self.layer_normalization1 = nn.LayerNorm(self.hidden_size)
        self.layer_normalization2 = nn.LayerNorm(self.hidden_size)

        self.msa = Attention(n_heads, self.hidden_size)

        # Feed-forward MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

    def forward(self, x):
        # Multi-head self-attention with skip connection
        out = x + self.msa(self.layer_normalization1(x))
        out = self.layer_normalization2(out)
        out = out + self.mlp(out)  # Feed-forward with skip connection
        return out


# Multi-head self-attention mechanism
class Attention(nn.Module):
    def __init__(self, n_heads, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads

        if (3 * self.hidden_size) % self.n_heads != 0:
            print("Numero di head incompatibile")  # Warning for incompatible heads

        # Linear projection to get Q, K, V
        self.qkv = nn.Linear(hidden_size, 3 * self.hidden_size * self.n_heads)
        self.softmax = nn.Softmax(dim=-1)
        self.linear = nn.Linear(self.n_heads * self.hidden_size, self.hidden_size)

    def forward(self, x):
        batch_size = x.shape[0]
        patch_n = x.shape[1]

        qkv = self.qkv(x)  # Linear projection to (Batch, Patches, QKV)
        qkv = qkv.view(batch_size, patch_n, self.n_heads, -1)
        qkv = qkv.transpose(1, 2)  # (Batch, Heads, Patches, Features)

        q, k, v = torch.chunk(qkv, 3, dim=-1)  # Split into Q, K, V

        qk = torch.matmul(q, k.transpose(-1, -2))  # Attention scores
        qk_softm = self.softmax(qk / (q.shape[-1] ** 0.5))  # Apply softmax

        attention = torch.matmul(qk_softm, v)  # Weighted sum
        attention = attention.reshape(batch_size, patch_n, -1)  # Merge heads

        attention = self.linear(attention)  # Final linear projection
        return attention


# Vision Transformer model
class Vit(nn.Module):
    def __init__(self, in_channels, hidden_size, img_size, num_classes, patch_size):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_dim = in_channels * patch_size * patch_size  # Flattened patch dimension

        # Converts image to flattened patch embeddings
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, hidden_size),
            nn.LayerNorm(hidden_size)
        )

        # Calculate number of patches in the image
        self.npatches = (img_size[-1] // patch_size) ** 2

        # Combine patch embeddings with positional and class token embeddings
        self.embedding_pos_cls = Embeddings(hidden_size=self.hidden_size, n_patches=self.npatches)

        # Transformer encoder block
        self.encoder = Encoder(hidden_size=self.hidden_size, n_heads=2)

        # Final classifier
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, num_classes)
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)  # Convert image to patch embeddings
        x = self.embedding_pos_cls(x)  # Add positional and class token embeddings
        x = self.encoder(x)  # Process with Transformer encoder
        x = x[:, 0]  # Take output of class token
        x = self.mlp(x)  # Final classification head
        return x






