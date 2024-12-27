import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.fft as fft


def inv_mag(x):
  fft_ = torch.fft.fft2(x)
  fft_ = torch.fft.ifft2(1*torch.exp(1j*(fft_.angle())))
  return fft_.real


class AGSSF(nn.Module):
    def __init__(self, channels, b=1, gamma=2):
        super(AGSSF, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channels = channels
        self.b = b
        self.gamma = gamma
        self.conv = nn.Conv1d(1, 1, kernel_size=self.kernel_size(), padding=(self.kernel_size() - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def kernel_size(self):
        k = int(abs((math.log2(self.channels)/self.gamma)+ self.b/self.gamma))
        out = k if k % 2 else k+1
        return out

    def forward(self, x):

        x1=inv_mag(x)
        # feature descriptor on the global spatial information
        y = self.avg_pool(x1)


        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)


        # Multi-scale information fusion
        y = self.sigmoid(y)


        return x * y.expand_as(x)


class SFCA(nn.Module):
    def __init__(self, channels, relu_slope=0.2, gamma=2):
        super(SFCA, self).__init__()
        self.identity1 = nn.Conv2d(channels, channels, 1)
        self.identity2 = nn.Conv2d(channels, channels, 1)
        self.conv_1 = nn.Conv2d(channels, 2*channels, kernel_size=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope)
        self.conv_2 = nn.Conv2d(2*channels, channels, kernel_size=3, padding=1, groups=channels, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope)

        self.conv_f1 = nn.Conv2d(channels, 2*channels, kernel_size=1)
        self.conv_f2 = nn.Conv2d(2*channels, channels, kernel_size=1)
        self.con2X1 = nn.Conv2d(2*channels, channels, kernel_size=1)
        self.agssf = AGSSF(channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):

        b_, l_, c_ = x.shape
        h_ = int(l_ ** 0.5)
        w_ = h_
        x = x.view(b_, c_, h_, w_)

        out = self.conv_1(x)
        out_1, out_2 = torch.chunk(out, 2, dim=1)
        out = torch.cat([out_1, out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        # print(self.identity1(x).shape, out.shape)
        out += self.identity1(x)

        x_fft = fft.fftn(x, dim=(-2, -1)).real
        x_fft = F.gelu(self.conv_f1(x_fft))
        x_fft = self.conv_f2(x_fft)
        x_reconstructed = fft.ifftn(x_fft, dim=(-2, -1)).real
        x_reconstructed += self.identity2(x)

        f_out = self.con2X1(torch.cat([out, x_reconstructed], dim=1))
        f_out = self.agssf(f_out)
        f_out = f_out.view(b_, -1, c_)
        f_out = self.norm(f_out)


        return f_out


def to_4d(x):
        b, n, c = x.shape
        h = int(n ** 0.5)
        w = h
        x = x.permute(1, 0, 2)
        x = x.view(h, w, b, c)
        x = x.permute(2, 3, 0, 1)  # (b, c, h, w)
        return x


def to_3d(x):
        b, c, h, w  = x.shape
        x = x.contiguous().view(b, h * w, c)
        return x


class FeedForward(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.):
        super(FeedForward, self).__init__()

        self.project_in = nn.Conv2d(in_features, hidden_features * 2, kernel_size=1, bias=False)

        self.dwconv3x3 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=False)
        self.dwconv5x5 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=5, stride=1, padding=2, groups=hidden_features * 2, bias=False)
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()

        self.dwconv3x3_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features , bias=False)
        self.dwconv5x5_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features , bias=False)

        self.gelu3_1 = F.gelu
        self.gelu5_1 = F.gelu

        self.drop = nn.Dropout(drop)
        self.project_out = nn.Conv2d(hidden_features * 2, in_features, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.project_in(x)
        x1_3, x2_3 = self.relu3(self.dwconv3x3(x)).chunk(2, dim=1)
        x1_5, x2_5 = self.relu5(self.dwconv5x5(x)).chunk(2, dim=1)

        x1 = torch.cat([x1_3, x1_5], dim=1)
        x2 = torch.cat([x2_3, x2_5], dim=1)
        x1 = self.drop(x1)
        x2 = self.drop(x2)

        x1 = self.gelu3_1(self.dwconv3x3_1(x1))
        x2 = self.gelu5_1(self.dwconv5x5_1(x2))

        x = torch.cat([x1, x2], dim=1)
        x = self.drop(x)

        x = self.project_out(x)

        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows  #  (B * (H // window_size) * (W // window_size), window_size, window_size, C)


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x  # (B , H , W , C)


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w],indexing='ij')) # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

        self.register_buffer("relative_position_index", relative_position_index)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=True)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=True)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)


        # frequency

        self.q1X1_1 = nn.Conv2d(dim, dim , kernel_size=1, bias=False)
        self.q1X1_2 = nn.Conv2d(dim, dim , kernel_size=1, bias=False)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape

        Hf = int(N ** 0.5)
        Wf = Hf
        x_f = x.permute(1, 0, 2)
        x_f = x_f.view(Hf, Wf, B_, C)
        x_f = x_f.permute(2, 3, 0, 1)
        x_fft = fft.fftn(x_f, dim=(-2, -1)).real
        x_fft1 = self.q1X1_1(x_fft)
        x_fft2 = F.gelu(x_fft1)
        x_fft3 = self.q1X1_2(x_fft2)
        x_ifft = fft.ifftn(x_fft3, dim=(-2, -1)).real  # (B_, C, Hw, WW)
        x_ifft = x_ifft.reshape(B_, C, -1)
        x_ifft = x_ifft.reshape(B_, self.num_heads, N, C // self.num_heads)

        temp = self.qkv_dwconv(self.qkv(x.permute(2,0,1))).permute(1,2,0)
        qkv = temp.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q + x_ifft
        k = k + x_ifft
        v = v + x_ifft

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)  # 1, nH, Wh*Ww, Wh*Ww

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)  #  * C
        x = self.proj(x)
        x = self.proj_drop(x)
        
        #  print(torch.matmul(attnf, vf).shape)

        return x

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

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
            # if window size is larger than input resolution, we don't partition windows
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
        self.ffn = FeedForward(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        self.norm3 = norm_layer(dim)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
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

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # nW, window_size, window_size
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift8*
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = self.norm2(x)
        x_4d = to_4d(x)
        x_4d = self.ffn(x_4d)
        x_3d = to_3d(x_4d)
        x = x + self.drop_path(self.norm3(x_3d))

        return x


# Downsample
class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


# UpSample

class FU(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super(FU, self).__init__()
        self.factor = scale_factor

        if self.factor == 2:
            self.amp_fuse = nn.Sequential(nn.Conv2d(in_channels, in_channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(in_channels, in_channels,1,1,0))
            self.pha_fuse = nn.Sequential(nn.Conv2d(in_channels, in_channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(in_channels, in_channels,1,1,0))
            self.post = nn.Conv2d(in_channels, in_channels//2, 1, 1, 0)

        elif self.factor == 0: # enhance
            self.amp_fuse = nn.Sequential(nn.Conv2d(in_channels, in_channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(in_channels, in_channels,1,1,0))
            self.pha_fuse = nn.Sequential(nn.Conv2d(in_channels, in_channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(in_channels, in_channels,1,1,0))
            self.post = nn.Conv2d(in_channels, in_channels, 1, 1, 0)

    def forward(self, x):

        fft_x = torch.fft.fft2(x)
        mag_x = torch.abs(fft_x)
        pha_x = torch.angle(fft_x)
        Mag = self.amp_fuse(mag_x)
        Pha = self.pha_fuse(pha_x)
        amp_fuse = torch.tile(Mag, (2, 2))
        pha_fuse = torch.tile(Pha, (2, 2))
        real = amp_fuse * torch.cos(pha_fuse)
        imag = amp_fuse * torch.sin(pha_fuse)
        out = torch.complex(real, imag)
        output = torch.fft.ifft2(out)
        output = torch.abs(output)  
        x_fu = self.post(output)

        return x_fu


class DU(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super(DU, self).__init__()
        self.factor = scale_factor


        if self.factor == 2:
            self.conv = nn.Conv2d(in_channels, in_channels//2, 1, 1, 0, bias=False)
            self.up_p = nn.Sequential(nn.Conv2d(in_channels, 2*in_channels, 1, 1, 0, bias=False),
                                      nn.PReLU(),
                                      nn.PixelShuffle(scale_factor),
                                      nn.Conv2d(in_channels//2, in_channels//2, 1, stride=1, padding=0, bias=False))

            self.up_b = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, 1, 0),
                                      nn.PReLU(),
                                      nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
                                      nn.Conv2d(in_channels, in_channels // 2, 1, stride=1, padding=0, bias=False))

        elif self.factor == 0: # enhance
            self.conv = nn.Conv2d(2*in_channels, in_channels, 1, 1, 0, bias=False)
            self.up_p = nn.Sequential(nn.Conv2d(in_channels, 4 * in_channels, 1, 1, 0, bias=False),
                                      nn.PReLU(),
                                      nn.PixelShuffle(2), # old: scale_factor
                                      nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=False))

            self.up_b = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, 1, 0),
                                      nn.PReLU(),
                                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                      nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        """
        x: B, L = H*W, C
        """
        x_p = self.up_p(x)  # pixel shuffle
        x_b = self.up_b(x)  # bilinear
        x_du = self.conv(torch.cat([x_p, x_b], dim=1)) 

        return x_du


class UpSample(nn.Module):
    def __init__(self, input_resolution, in_channels, scale_factor):
        super(UpSample, self).__init__()
        self.input_resolution = input_resolution
        self.factor = scale_factor
        self.fu = FU(in_channels, scale_factor)
        self.du = DU(in_channels, scale_factor)

        if self.factor == 2:
            self.conv = nn.Conv2d(in_channels, in_channels//2, 1, 1, 0, bias=False)

        elif self.factor == 0: # enhance
            self.conv = nn.Conv2d(2*in_channels, in_channels, 1, 1, 0, bias=False)

    def forward(self, x):

        if type(self.input_resolution) == int:
            H = self.input_resolution
            W = self.input_resolution

        elif type(self.input_resolution) == tuple:
            H, W = self.input_resolution

        B, L, C = x.shape
        x = x.view(B, H, W, C)  # B, H, W, C
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        x_fu = self.fu(x)
        x_du = self.du(x)
        out = self.conv(torch.cat([x_fu, x_du], dim=1))
        out = out.permute(0, 2, 3, 1)  # B, H, W, C/2
        if self.factor == 2:
            out = out.view(B, -1, C // 2)

        return out


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = UpSample(input_resolution, in_channels=dim, scale_factor=2)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

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

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class URSCT(nn.Module):
    def __init__(self, opt, ch = [128, 64, 32]):
        super(URSCT, self).__init__()
        self.img_size = opt['IMG_SIZE']  # [256, 256]
        self.in_chans = opt['IN_CHANS']  # 3
        self.out_chans = opt['OUT_CHANS']  # 3
        self.patch_size = opt['PATCH_SIZE']  # 2
        self.depths = opt['DEPTH_EN']  # [8, 8, 8, 8]
        self.num_layers = len(self.depths)  # 4
        self.embed_dim = opt['EMB_DIM']  # 32
        self.ape = opt['APE']  # False
        self.patch_norm = opt['PATCH_NORM']  # True
        self.num_features = int(self.embed_dim * 2 ** (self.num_layers - 1))  # 256
        self.num_features_up = int(self.embed_dim * 2)  # 64
        self.mlp_ratio = opt['MLP_RATIO']  # 4.0
        self.qkv_bias = opt['QKV_BIAS']  # True
        self.qk_scale = opt['QK_SCALE']  # 8
        self.num_heads = opt['HEAD_NUM']  # [8, 8, 8, 8]
        self.window_size = opt['WIN_SIZE']  # 8
        self.drop_rate = opt['DROP_RATE']  # 0
        self.drop_path_rate = opt['DROP_PATH_RATE']  # 0.1
        self.attn_drop_rate = opt['ATTN_DROP_RATE']  # 0.
        self.use_checkpoint = opt['USE_CHECKPOINTS']  # False

        self.prelu = nn.PReLU()
        self.norm_layer = nn.LayerNorm
        self.conv_first = nn.Conv2d(in_channels=self.in_chans, out_channels=self.embed_dim, kernel_size=3, stride=1, padding=1)
        self.attnfusion = nn.ModuleList([SFCA(channels) for channels in ch]) 
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size= self.img_size, patch_size=self.patch_size, in_chans=self.embed_dim, embed_dim=self.embed_dim,
            norm_layer=self.norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches  # 16384
        patches_resolution = self.patch_embed.patches_resolution  # [128, 128]
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=self.drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]  # stochastic depth decay rule

        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(self.embed_dim * 2 ** i_layer),  # 32, 64, 128, 256
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),  # (128, 128), (64, 64), (32, 32), (16, 16)
                               depth=self.depths[i_layer],
                               num_heads=self.num_heads[i_layer],
                               window_size=self.window_size,  # 8
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                               drop=self.drop_rate, attn_drop=self.attn_drop_rate,
                               drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                               norm_layer=self.norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=self.use_checkpoint)
            self.layers.append(layer)

        # build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2 * int(self.embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      int(self.embed_dim * 2 ** (
                                              self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = UpSample(input_resolution=patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),  # 16
                                    in_channels=int(self.embed_dim * 2 ** (self.num_layers - 1 - i_layer)), scale_factor=2)  # (16, 256, 2)
            else:
                layer_up = BasicLayer_up(dim=int(self.embed_dim * 2 ** (self.num_layers - 1 - i_layer)),  # 128, 64, 32
                                         input_resolution=(
                                             patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                             patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),  # (32, 32), (64, 64), (128, 128)
                                         depth=self.depths[(self.num_layers - 1 - i_layer)],
                                         num_heads=self.num_heads[(self.num_layers - 1 - i_layer)],
                                         window_size=self.window_size,
                                         mlp_ratio=self.mlp_ratio,
                                         qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                                         drop=self.drop_rate, attn_drop=self.attn_drop_rate,
                                         drop_path=dpr[sum(self.depths[:(self.num_layers - 1 - i_layer)]):sum(
                                             self.depths[:(self.num_layers - 1 - i_layer) + 1])],
                                         norm_layer=self.norm_layer,
                                         upsample=UpSample if (i_layer < self.num_layers - 1) else None,
                                         use_checkpoint=self.use_checkpoint)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = self.norm_layer(self.num_features)
        self.norm_up = self.norm_layer(self.embed_dim)

        # final_upsample
        self.up = UpSample(input_resolution=(self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size),  # (128, 128)
                           in_channels=self.embed_dim, scale_factor=0)
        self.output = nn.Conv2d(in_channels=self.embed_dim, out_channels=self.out_chans, kernel_size=3, stride=1,padding=1, bias=False)  # kernel = 1
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    # Encoder and Bottleneck
    def forward_features(self, x):
        residual = x
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []

        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)

        x = self.norm(x)  # B L C

        return x, residual, x_downsample

    # Decoder and Skip connection
    def forward_up_features(self, x, x_downsample):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, self.attnfusion[inx-1](x_downsample[3-inx])], -1)  # concat last dimension
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

        x = self.norm_up(x)  # B L C

        return x


    def up_enhance(self, x):
        x = self.up(x)
        x = x.permute(0, 3, 1, 2)  # B,C,H,W
        return x

    def forward(self, x):
        if self.in_chans == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.conv_first(x)
        x, residual, x_downsample = self.forward_features(x)
        x = self.forward_up_features(x, x_downsample)
        x = self.up_enhance(x)
        out = self.output(x)
        return out