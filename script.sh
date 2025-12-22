#!/bin/bash
cat > /tmp/full_patch.py << 'EOF'
filepath = "/home/bilal-linux/.pyenv/versions/vitcod_py38/lib/python3.8/site-packages/timm/models/vision_transformer.py"

with open(filepath, 'r') as f:
    content = f.read()

# ============ PATCH 1: Attention class ============
old_attention_init = '''class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)'''

new_attention_init = '''class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., need_weight=False, attn_mask=None, svd_type=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # ViTCoD additions
        self.need_weight = need_weight
        self.attn_mask = attn_mask
        self.svd_type = svd_type
        if need_weight:
            self.register_buffer('attention_sum', torch.zeros([self.num_heads, 197, 197]))
            self.register_buffer('num_attention', torch.zeros([1]))'''

content = content.replace(old_attention_init, new_attention_init)

# ============ PATCH 2: Attention forward ============
old_attention_forward = '''    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x'''

new_attention_forward = '''    def forward(self, x, evaluate=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # ViTCoD: track attention weights
        if self.need_weight and not evaluate:
            self.attention_sum += attn.sum(dim=0)
            self.num_attention += attn.size(0)
        
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x'''

content = content.replace(old_attention_forward, new_attention_forward)

# ============ PATCH 3: Block class ============
old_block = '''class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)'''

new_block = '''class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, need_weight=False, attn_mask=None, svd_type=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            need_weight=need_weight, attn_mask=attn_mask, svd_type=svd_type)'''

content = content.replace(old_block, new_block)

# ============ PATCH 4: Block forward ============
old_block_forward = '''    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x'''

new_block_forward = '''    def forward(self, x, evaluate=False):
        x = x + self.drop_path(self.attn(self.norm1(x), evaluate=evaluate))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x'''

content = content.replace(old_block_forward, new_block_forward)

# ============ PATCH 5: VisionTransformer __init__ ============
old_vit_init = '''class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm):'''

new_vit_init = '''class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm,
                 need_weight=False, attn_mask=None, svd_type=None):'''

content = content.replace(old_vit_init, new_vit_init)

# ============ PATCH 6: Block creation in VisionTransformer ============
old_blocks_create = '''        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])'''

new_blocks_create = '''        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                need_weight=need_weight, attn_mask=attn_mask, svd_type=svd_type)
            for i in range(depth)])'''

content = content.replace(old_blocks_create, new_blocks_create)

# ============ PATCH 7: VisionTransformer forward ============
old_vit_forward = '''    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x'''

new_vit_forward = '''    def forward(self, x, evaluate=False):
        x = self.forward_features(x)
        x = self.head(x)
        recon_loss = torch.tensor(0.0, device=x.device)
        return x, recon_loss'''

content = content.replace(old_vit_forward, new_vit_forward)

with open(filepath, 'w') as f:
    f.write(content)

print("All patches applied successfully!")
EOF

python /tmp/full_patch.py