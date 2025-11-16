import torch
import torch.nn as nn

# Transformer-based decoder (structure consider MAE decoder)
class TransformerDecoder(nn.Module):
    def __init__(self, encoder_embed_dim, decoder_embed_dim, num_layers, num_heads, dropout,
                 img_size=224, patch_size=32, num_channels=3):
        super(TransformerDecoder, self).__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        # map encoder output to decoder dimension
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim)
        # learnable position encoding, size: [1, num_patches, decoder_embed_dim]
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_embed_dim))
        # stack multiple Transformer blocks (use TransformerEncoderLayer, set batch_first=True)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=decoder_embed_dim, nhead=num_heads,
                                       dropout=dropout, activation="gelu", batch_first=True)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(decoder_embed_dim)
        # final prediction head, map each token to the pixel of corresponding patch (patch_size*patch_size*num_channels)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size * patch_size * num_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # position encoding initialization (use truncated normal distribution)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, features):
        """
        features: [seq_len, batch, encoder_embed_dim]
        1. remove class token (assume the first token is class token), get patch tokens
        2. map to decoder dimension through linear layer, and add position encoding
        3. pass through Transformer blocks and normalization
        4. predict the pixel of each patch, and reshape to complete image
        """
        # remove class token, get patch tokens shape: [num_patches, batch, encoder_embed_dim]
        patch_tokens = features[1:, :, :]
        # transpose to [batch, num_patches, encoder_embed_dim]
        patch_tokens = patch_tokens.permute(1, 0, 2)
        x = self.decoder_embed(patch_tokens)  # [batch, num_patches, decoder_embed_dim]
        x = x + self.pos_embed  # add position encoding
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.decoder_pred(x)  # [batch, num_patches, patch_size*patch_size*num_channels]
        # reshape to [batch, grid_size, grid_size, patch_size, patch_size, num_channels]
        batch_size = x.shape[0]
        x = x.view(batch_size, self.grid_size, self.grid_size, self.patch_size, self.patch_size, -1)
        # reshape to [batch, num_channels, img_size, img_size]
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(batch_size, -1, self.img_size, self.img_size)
        x = torch.sigmoid(x)
        return x