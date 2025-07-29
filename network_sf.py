import torch
import torch.nn as nn
from einops import rearrange
from mamba_ssm import Mamba
from timm.models.layers import DropPath, to_2tuple, trunc_normal_, lecun_normal_
import math
from functools import partial

class LateralConnection(nn.Module):
    def __init__(self, fast_channels=36, slow_channels=72):
        super(LateralConnection, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(fast_channels, slow_channels, 3, 2, 1),   
            nn.BatchNorm1d(slow_channels),
            nn.ReLU(),
        )
        
    def forward(self, slow_path, fast_path):
        fast_path = rearrange(fast_path, 'b t c -> b c t')
        fast_path = self.conv(fast_path)
        fast_path = rearrange(fast_path, 'b c t -> b t c')
        return fast_path + slow_path
    
class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=64, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  
            d_state=d_state,  
            d_conv=d_conv, 
            expand=expand  
        )
    def forward(self, x):
        B, N, C = x.shape
        x_norm = self.norm(x)
        x_mamba = self.mamba(x_norm)    
        return x_mamba


class MambaBlock(nn.Module):
    def __init__(self, 
        dim, 
        mlp_ratio,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = MambaLayer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, D, C = x.size()
        x = x + self.drop_path(self.norm1(self.attn(x)))

        return x


class MambaSF(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.Stem_slow = nn.Sequential(
            nn.Conv1d(dim, dim*2, 3, 2, 1),
            nn.BatchNorm1d(dim*2),
            nn.ReLU(inplace=True),
        )

        dpr = [x.item() for x in torch.linspace(0, 0.2, 4)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        
        self.block1 = nn.ModuleList([MambaBlock(
            dim = dim*2, 
            mlp_ratio = 4,
            drop_path=inter_dpr[i], 
            norm_layer=nn.LayerNorm,)
        for i in range(4)])
        self.block2 = nn.ModuleList([MambaBlock(
            dim = dim*2, 
            mlp_ratio = 4,
            drop_path=inter_dpr[i], 
            norm_layer=nn.LayerNorm,)
        for i in range(4)])
        self.block3 = nn.ModuleList([MambaBlock(
            dim = dim*2, 
            mlp_ratio = 4,
            drop_path=inter_dpr[i], 
            norm_layer=nn.LayerNorm,)
        for i in range(4)])

        self.block1_fast = nn.ModuleList([MambaBlock(
            dim = dim, 
            mlp_ratio = 4,
            drop_path=inter_dpr[i], 
            norm_layer=nn.LayerNorm,)
        for i in range(4)]) 
        self.block2_fast = nn.ModuleList([MambaBlock(
            dim = dim, 
            mlp_ratio = 4,
            drop_path=inter_dpr[i], 
            norm_layer=nn.LayerNorm,)
        for i in range(4)])
        self.block3_fast = nn.ModuleList([MambaBlock(
            dim = dim, 
            mlp_ratio = 4,
            drop_path=inter_dpr[i], 
            norm_layer=nn.LayerNorm,)
        for i in range(4)])

        self.fuse_1 = LateralConnection(fast_channels=dim, slow_channels=dim*2)
        self.fuse_2 = LateralConnection(fast_channels=dim, slow_channels=dim*2)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x_fast = rearrange(x, 'b c t -> b t c')
        x_slow = self.Stem_slow(x)
        x_slow = rearrange(x_slow, 'b c t -> b t c')

        for blk in self.block1:
            x_slow = blk(x_slow)
        for blk in self.block1_fast:
            x_fast = blk(x_fast)
        x_slow = self.fuse_1(x_slow,x_fast)

        for blk in self.block2:
            x_slow = blk(x_slow)
        for blk in self.block2_fast:
            x_fast = blk(x_fast)
        x_slow = self.fuse_2(x_slow,x_fast)

        for blk in self.block3:
            x_slow = blk(x_slow)
        for blk in self.block3_fast:
            x_fast = blk(x_fast)
        x_slow = rearrange(x_slow, 'b t c -> b c t')
        x_slow = self.upsample(x_slow)
        x_slow = rearrange(x_slow, 'b c t -> b t c')
        x_fusion = torch.cat((x_slow, x_fast), dim=2) 
        return x_fusion
    
# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class METST_SF(nn.Module):
    def __init__(self, out_channels=3):
        super(METST_SF, self).__init__()
        dim = 128
        self.Stem = nn.Sequential(
            nn.Conv1d(36, dim, 3, 1, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, 3, 1, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
        )

        self.spot_pathway = MambaSF(dim)
        self.recog_pathway = MambaSF(dim)

        self.fc_spot = nn.Linear(in_features=dim*3, out_features=1)
        self.fc_recog = nn.Linear(in_features=dim*3, out_features=out_channels)
        self.sigmoid = nn.Sigmoid()

        # init
        self.apply(segm_init_weights)
        self.apply(partial(_init_weights,n_layer=4,**({}), ))

    
    def forward(self, x):
        x = x.squeeze(1)
        x = self.Stem(x)

        x_spot = self.spot_pathway(x)
        x_spot = self.fc_spot(x_spot)
        x_spot = self.sigmoid(x_spot)
        x_spot = x_spot.squeeze(-1)

        x_recog = self.recog_pathway(x)
        x_recog = self.fc_recog(x_recog)

        return x_spot, x_recog