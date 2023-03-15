import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def make_model(args, parent=False):
    return swinOIR(args)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)  
        self.act = act_layer() 
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    
    B, H, W, C = x.shape  
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):

    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        flops = 0
        flops += N * self.dim * 3 * self.dim
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        flops += N * self.dim * self.dim
        return flops

class SwinTransformerBlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None
        
        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):

        flops = 0
        H, W = self.input_resolution
        flops += self.dim * H * W
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        flops += self.dim * H * W
        return flops

    def forward(self, x):
        block1 = self.gelu(self.block1(x))
        block2 = self.gelu(self.block2(block1))
        b2_dense = self.gelu(torch.cat([block1, block2],1))
        block3 = self.gelu(self.block3(b2_dense))
        b3_dense = self.gelu(torch.cat([block1, block2, block3], 1))
        block4 = self.gelu(self.block4(b3_dense))
        b4_dense = self.gelu(torch.cat([block1, block3, block4], 1))
        block5 = self.gelu(self.block5(b4_dense))
        c5_dense = self.gelu(torch.cat([block1, block2, block4, block5], 1))
        block6 = self.gelu(self.block6(c5_dense))
        c6_dense = self.gelu(torch.cat([block1, block3, block5, block6], 1))


        return c6_dense

class DSTB8(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size, use_checkpoint=False):
        
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.use_checkpoint = use_checkpoint

        self.gelu = nn.GELU()
        self.block1 = SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
        self.block2 = SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
        self.block3 = SwinTransformerBlock(dim=6, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
        self.block4 = SwinTransformerBlock(dim=9, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
        self.block5 = SwinTransformerBlock(dim=9, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
        self.block6 = SwinTransformerBlock(dim=12, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
        self.block7 = SwinTransformerBlock(dim=12, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
        self.block8 = SwinTransformerBlock(dim=15, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
        self.conv = nn.Conv2d(in_channels=15, out_channels=15, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        block1 = self.gelu(self.block1(x))
        block2 = self.gelu(self.block2(block1))
        b2_dense = self.gelu(torch.cat([block1, block2],1))
        block3 = self.gelu(self.block3(b2_dense))
        b3_dense = self.gelu(torch.cat([block1, block2, block3], 1))
        block4 = self.gelu(self.block4(b3_dense))
        b4_dense = self.gelu(torch.cat([block1, block3, block4], 1))
        block5 = self.gelu(self.block5(b4_dense))
        c5_dense = self.gelu(torch.cat([block1, block2, block4, block5], 1))
        block6 = self.gelu(self.block6(c5_dense))
        c6_dense = self.gelu(torch.cat([block1, block3, block5, block6], 1))
        block7 = self.gelu(self.block7(c6_dense))
        c7_dense = self.gelu(torch.cat([block1, block2, block4, block6, block7], 1))
        block8 = self.gelu(self.block8(c7_dense))
        c8_dense = self.gelu(torch.cat([block1, block3, block5, block7, block8], 1))
        # c8_dense = self.conv(c8_dense)

        return c8_dense

class DSTB10(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size, use_checkpoint=False):
        
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.use_checkpoint = use_checkpoint

        self.gelu = nn.GELU()
        self.block1 = SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
        self.block2 = SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
        self.block3 = SwinTransformerBlock(dim=6, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
        self.block4 = SwinTransformerBlock(dim=9, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
        self.block5 = SwinTransformerBlock(dim=9, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
        self.block6 = SwinTransformerBlock(dim=12, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
        self.block7 = SwinTransformerBlock(dim=12, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
        self.block8 = SwinTransformerBlock(dim=15, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
        self.block9 = SwinTransformerBlock(dim=15, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
        self.block10 = SwinTransformerBlock(dim=18, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)


    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        b2_dense = torch.cat([block1, block2],1)
        block3 = self.block3(b2_dense)
        b3_dense = torch.cat([block1, block2, block3], 1)
        block4 = self.block4(b3_dense)
        b4_dense = torch.cat([block1, block3, block4], 1)
        block5 = self.block5(b4_dense)
        c5_dense = torch.cat([block1, block2, block4, block5], 1)
        block6 = self.block6(c5_dense)
        c6_dense = torch.cat([block1, block3, block5, block6], 1)
        block7 = self.block7(c6_dense)
        c7_dense = torch.cat([block1, block2, block4, block6, block7], 1)
        block8 = self.block8(c7_dense)
        c8_dense = torch.cat([block1, block3, block5, block7, block8], 1)
        block9 = self.block9(c8_dense)
        c9_dense = torch.cat([block1, block2, block4, block6, block8, block9], 1)
        block10 = self.block10(c9_dense)
        c10_dense = torch.cat([block1, block3, block5, block7, block9, block10], 1)

        return c10_dense

class DSTB12(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size, use_checkpoint=False):
        
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.use_checkpoint = use_checkpoint

        self.gelu = nn.GELU()
        self.block1 = SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
        self.block2 = SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
        self.block3 = SwinTransformerBlock(dim=6, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
        self.block4 = SwinTransformerBlock(dim=9, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
        self.block5 = SwinTransformerBlock(dim=9, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
        self.block6 = SwinTransformerBlock(dim=12, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
        self.block7 = SwinTransformerBlock(dim=12, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
        self.block8 = SwinTransformerBlock(dim=15, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
        self.block9 = SwinTransformerBlock(dim=15, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
        self.block10 = SwinTransformerBlock(dim=18, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
        self.block11 = SwinTransformerBlock(dim=18, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
        self.block12 = SwinTransformerBlock(dim=21, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        b2_dense = torch.cat([block1, block2],1)
        block3 = self.block3(b2_dense)
        b3_dense = torch.cat([block1, block2, block3], 1)
        block4 = self.block4(b3_dense)
        b4_dense = torch.cat([block1, block3, block4], 1)
        block5 = self.block5(b4_dense)
        c5_dense = torch.cat([block1, block2, block4, block5], 1)
        block6 = self.block6(c5_dense)
        c6_dense = torch.cat([block1, block3, block5, block6], 1)
        block7 = self.block7(c6_dense)
        c7_dense = torch.cat([block1, block2, block4, block6, block7], 1)
        block8 = self.block8(c7_dense)
        c8_dense = torch.cat([block1, block3, block5, block7, block8], 1)
        block9 = self.block9(c8_dense)
        c9_dense = torch.cat([block1, block2, block4, block6, block8, block9], 1)
        block10 = self.block10(c9_dense)
        c10_dense = torch.cat([block1, block3, block5, block7, block9, block10], 1)
        block11 = self.block11(c10_dense)
        c11_dense = torch.cat([block1, block2, block4, block6, block8, block10, block11], 1)
        block12 = self.block12(c11_dense)
        c12_dense = torch.cat([block1, block3, block5, block7, block9, block11, block12], 1)
        # c12_dense = self.conv(c12_dense)

        return c12_dense

class DSTB14(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size, use_checkpoint=False):
        
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.use_checkpoint = use_checkpoint

        self.gelu = nn.GELU()
        self.block1 = SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
        self.block2 = SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
        self.block3 = SwinTransformerBlock(dim=6, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
        self.block4 = SwinTransformerBlock(dim=9, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
        self.block5 = SwinTransformerBlock(dim=9, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
        self.block6 = SwinTransformerBlock(dim=12, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
        self.block7 = SwinTransformerBlock(dim=12, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
        self.block8 = SwinTransformerBlock(dim=15, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
        self.block9 = SwinTransformerBlock(dim=15, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
        self.block10 = SwinTransformerBlock(dim=18, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
        self.block11 = SwinTransformerBlock(dim=18, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
        self.block12 = SwinTransformerBlock(dim=21, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
        self.block13 = SwinTransformerBlock(dim=21, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
        self.block14 = SwinTransformerBlock(dim=24, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        b2_dense = torch.cat([block1, block2],1)
        block3 = self.block3(b2_dense)
        b3_dense = torch.cat([block1, block2, block3], 1)
        block4 = self.block4(b3_dense)
        b4_dense = torch.cat([block1, block3, block4], 1)
        block5 = self.block5(b4_dense)
        c5_dense = torch.cat([block1, block2, block4, block5], 1)
        block6 = self.block6(c5_dense)
        c6_dense = torch.cat([block1, block3, block5, block6], 1)
        block7 = self.block7(c6_dense)
        c7_dense = torch.cat([block1, block2, block4, block6, block7], 1)
        block8 = self.block8(c7_dense)
        c8_dense = torch.cat([block1, block3, block5, block7, block8], 1)
        block9 = self.block9(c8_dense)
        c9_dense = torch.cat([block1, block2, block4, block6, block8, block9], 1)
        block10 = self.block10(c9_dense)
        c10_dense = torch.cat([block1, block3, block5, block7, block9, block10], 1)
        block11 = self.block11(c10_dense)
        c11_dense = torch.cat([block1, block2, block4, block6, block8, block10, block11], 1)
        block12 = self.block12(c11_dense)
        c12_dense = torch.cat([block1, block3, block5, block7, block9, block11, block12], 1)
        block13 = self.block13(c12_dense)
        c13_dense = torch.cat([block1, block2, block4, block6, block8, block10, block12, block13], 1)
        block14 = self.block14(c13_dense)
        c14_dense = torch.cat([block1, block3, block5, block7, block9, block11, block13, block14], 1)

        return c14_dense

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops

class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])
        return x

    def flops(self):
        flops = 0
        return flops

class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

class UpsampleOneStep(nn.Sequential):

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat                    #中間特徵的通道數數量
        self.input_resolution = input_resolution    #輸入的解析度大小
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))            #使用PixelShuffle上採樣scale倍
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.num_feat * 3 * 9
        return flops


class swinOIR(nn.Module):

    def __init__(self, img_size=64, patch_size=1, in_chans=3,
                 embed_dim=96, num_heads=[6],
                 window_size=8, mlp_ratio=4., drop_rate=0.,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 upscale=4, img_range=1., upsampler='pixelshuffledirect', resi_connection='1conv',
                 **kwargs):
        super(swinOIR, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        
        self.layers = nn.ModuleList()

        layer1 = DSTB8(dim=in_chans, input_resolution=(patches_resolution[0],patches_resolution[1]),
                            num_heads=num_heads, window_size=window_size)
        layer2 = DSTB10(dim=in_chans, input_resolution=(patches_resolution[0],patches_resolution[1]),
                            num_heads=num_heads, window_size=window_size)
        layer3 = DSTB12(dim=in_chans, input_resolution=(patches_resolution[0],patches_resolution[1]),
                            num_heads=num_heads, window_size=window_size)
        layer4 = DSTB14(dim=in_chans, input_resolution=(patches_resolution[0],patches_resolution[1]),
                            num_heads=num_heads, window_size=window_size)
        layer5 = DSTB12(dim=in_chans, input_resolution=(patches_resolution[0],patches_resolution[1]),
                            num_heads=num_heads, window_size=window_size)
        layer6 = DSTB10(dim=in_chans, input_resolution=(patches_resolution[0],patches_resolution[1]),
                            num_heads=num_heads, window_size=window_size)
        layer7 = DSTB8(dim=in_chans, input_resolution=(patches_resolution[0],patches_resolution[1]),
                            num_heads=num_heads, window_size=window_size)

        self.layers.append(layer2)
        self.layers.append(layer3)
        self.layers.append(layer4)
        self.layers.append(layer5)
        self.layers.append(layer6)

        self.norm = norm_layer(self.num_features)

        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        if self.upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':    

            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))

        self.apply(self._init_weights)

    
    def _make_transition_layer(self, layer, in_channels, out_channels):
        modules = []
        modules.append(layer(in_channels, out_channels))
        return nn.Sequential(*modules)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:                        
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
      
        x = x / self.img_range + self.mean

        return x[:, :, :H*self.upscale, :W*self.upscale]

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops