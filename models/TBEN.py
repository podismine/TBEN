import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):

        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)

        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x

class TBEN(nn.Module):
    def __init__(self, tr_drop = 0.,depth = 6, heads = 8, pool = 'cls', dim = 128,mlp_dim=512):
        super(TBEN, self).__init__()
        self.feature_encode1 = self.conv_layer(1, 32)
        self.feature_encode2 = self.conv_layer(32, 64)
        self.tr = ViT(
                    image_size = 22,
                    patch_size = 27,
                    num_classes = 84,
                    dim = dim,
                    depth = depth,
                    heads = heads,
                    mlp_dim = mlp_dim,
                    pool = pool,
                    dim_head = 64,
                    channels = 64,
                    dropout = tr_drop
                  )

    def conv_layer(self, in_channel, out_channel):
        layer = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, padding=1, kernel_size=3),
            nn.BatchNorm3d(out_channel),
            nn.MaxPool3d(2, stride=2),
            nn.ReLU(inplace = True),
            )
        return layer
    def forward(self, x):
        x = self.feature_encode1(x)
        x = self.feature_encode2(x)
        x = self.tr(x)
        return x

class ViT(nn.Module):
    # this code is modified from *****, link: *********
    # Specially, the only code should be modified is the rearrange

    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 1, dim_head = 64, dropout = 0.5, emb_dropout = 0.):
        super().__init__()

        num_patches = image_size * image_size
        patch_dim = channels * patch_size
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.patch_size = channels * patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(self.patch_size, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    def forward(self, img):
        p = img.size(3)
        x = rearrange(img, 'b c (h) (w p) (d) -> b (h w d) (p c)', p = p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        x1 = self.mlp_head(x)
        out1 = F.log_softmax(x1, dim=1)
        return out1

