# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Block modules."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn
from torch.cuda.amp import autocast
from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ELAN1",
    "ADown",
    "AConv",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "C3k2",
    "C2fPSA",
    "C2PSA",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "SCDown",
    "TorchVision",
    "WEC",
    "SSF"
)


#" ----------------- Channel Attention (CBAM 风格) -----------------
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(8, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        w = self.sigmoid(avg_out + max_out)
        return x * w


# ----------------- Spatial Attention (CBAM 风格) -----------------
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: B,C,H,W
        avg_out = x.mean(dim=1, keepdim=True)           # B,1,H,W
        max_out, _ = x.max(dim=1, keepdim=True)         # B,1,H,W
        x_cat = torch.cat([avg_out, max_out], dim=1)    # B,2,H,W
        attn = self.sigmoid(self.conv(x_cat))           # B,1,H,W
        return x * attn


# ----------------- Attention Bottleneck（代替原 Bottleneck） -----------------
class AttnBottleneck(nn.Module):
    """
    Bottleneck + Channel Attention + Spatial Attention

    就是论文图 2 那个：
      Conv1 -> BN -> ReLU
      Conv2 -> BN
      -> CA -> SA
      -> 残差相加 -> ReLU
    """
    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: tuple[int, int] = (3, 3),
        e: float = 0.5,
        ca_reduction: int = 16,
        sa_kernel: int = 7,
    ):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

        self.ca = ChannelAttention(c2, reduction=ca_reduction)
        self.sa = SpatialAttention(kernel_size=sa_kernel)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        y = self.cv2(self.cv1(x))   # 两层 conv
        y = self.ca(y)              # Channel attention
        y = self.sa(y)              # Spatial attention
        if self.add:
            y = x + y               # 残差
        return self.act(y)



class SpatialAttention_CBAM(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        assert kernel_size in (3, 5, 7)
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        s = torch.cat([avg, mx], dim=1)
        s = self.sigmoid(self.conv(s))
        return x * s

# class CSPNextUWUnit(nn.Module):
#     def __init__(self, c, shortcut=True, e=0.5):
#         super().__init__()
#         c_hidden = int(c * e)
#
#         # 1×1 降维
#         self.cv1 = Conv(c, c_hidden, k=1, s=1)
#
#         # 多尺度并行分支
#         self.local = Conv(c_hidden, c_hidden, k=3, s=1)          # 普通 3×3
#         self.dilated = Conv(c_hidden, c_hidden, k=3, s=1, d=2)   # 膨胀 3×3
#
#         # 融合两个分支
#         self.fuse = Conv(2 * c_hidden, c_hidden, k=1, s=1)
#
#
#         # 边缘空间注意
#         self.edge_attn = EdgeSpatialAttention(c_hidden)
#         # self.edge_attn = SpatialAttention_CBAM(3)
#         # 投影回原通道
#         self.proj = Conv(c_hidden, c, k=1, s=1)
#
#         self.shortcut = shortcut
#
#     def forward(self, x):
#         # x: B,C,H,W
#         y = self.cv1(x)                  # B,c_hidden,H,W
#
#         y1 = self.local(y)               # B,c_hidden,H,W
#         y2 = self.dilated(y)             # B,c_hidden,H,W
#         y = torch.cat((y1, y2), dim=1)   # B,2*c_hidden,H,W
#
#         y = self.fuse(y)                 # B,c_hidden,H,W
#         y = self.edge_attn(y)            # B,c_hidden,H,W
#         y = self.proj(y)                 # B,C,H,W
#
#         return x + y if self.shortcut else y



class CSPNextUWUnitLite(nn.Module):
    """
    Unit 内不放 Wave / Edge（避免 n 次堆叠重复扰动 + 冗余计算）
    只做：1x1降维 -> (3x3 + dilated3x3) -> fuse -> post -> proj -> (shortcut)
    """
    def __init__(self, c, shortcut=True, e=0.5):
        super().__init__()
        c_hidden = int(c * e)

        self.cv1 = Conv(c, c_hidden, k=1, s=1)
        self.local = Conv(c_hidden, c_hidden, k=3, s=1)
        self.dilated = Conv(c_hidden, c_hidden, k=3, s=1, d=2)

        self.fuse1 = Conv(2 * c_hidden, c_hidden, k=1, s=1)
        self.post = Conv(c_hidden, c_hidden, k=3, s=1)
        self.proj = Conv(c_hidden, c, k=1, s=1)

        self.shortcut = shortcut

    def forward(self, x):
        y = self.cv1(x)
        y1 = self.local(y)
        y2 = self.dilated(y)
        y = self.fuse1(torch.cat([y1, y2], dim=1))
        y = self.post(y)
        y = self.proj(y)
        return x + y if self.shortcut else y

class EdgeSpatialAttention(nn.Module):
    def __init__(self, channels, beta=0.5, learnable=True):
        super().__init__()
        self.beta = beta

        # depthwise conv 做 Gx, Gy
        self.conv_gx = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.conv_gy = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)

        # Sobel 核
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1,-2,-1],
                                [ 0, 0, 0],
                                [ 1, 2, 1]], dtype=torch.float32)

        with torch.no_grad():
            wgx = sobel_x.view(1,1,3,3).repeat(channels,1,1,1)
            wgy = sobel_y.view(1,1,3,3).repeat(channels,1,1,1)
            self.conv_gx.weight.copy_(wgx)
            self.conv_gy.weight.copy_(wgy)


        # 是否允许微调
        self.conv_gx.weight.requires_grad_(learnable)
        self.conv_gy.weight.requires_grad_(learnable)

        # 把多通道梯度合并成 1 通道
        self.reduce = nn.Conv2d(channels, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: B,C,H,W
        gx = self.conv_gx(x)
        gy = self.conv_gy(x)
        grad_mag = torch.sqrt(gx * gx + gy * gy + 1e-6)  # B,C,H,W

        edge = self.reduce(grad_mag)                     # B,1,H,W
        # 这里不做复杂归一化，直接 sigmoid
        mask = self.sigmoid(edge)                        # [0,1]

        # 残差式调节，避免过猛
        # mask-0.5 ∈ [-0.5,0.5] → 1 + beta*(mask-0.5) 在 [1-0.5*beta, 1+0.5*beta]
        scale = 1 + self.beta * (mask - 0.5)
        return x * scale
class WaveletGateLite(nn.Module):
    def __init__(self, C, Cw=None, se_r=16, refine=False):
        super().__init__()
        if Cw is None:
            Cw = max(32, C // 4)   # 重点：C//4 通常就够
        self.down = Conv(C, Cw, k=1, s=1)
        self.wg   = WaveletGate(Cw, se_r=se_r, refine=refine)
        self.up   = Conv(Cw, C, k=1, s=1)

    def forward(self, x):
        z = self.down(x)
        z = self.wg(z)
        return self.up(z)
class WaveletGate(nn.Module):
    """
    参考你图里的思路：
      - DWT -> 高频(LH/HL/HH) 与 低频(LL)
      - 从高/低频各自预测一个通道权重 A_high/A_low
      - 对输入 x 做三路投影：id/ high/ low，然后分别门控再 concat -> 输出仍为 C
      - 可选 1x1 refine（很轻，且对mAP通常更稳）
    """
    def __init__(self, C: int, se_r: int = 16, refine: bool = True):
        super().__init__()
        self.C = C
        self.dwt = HaarDWT2D()

        # 三路通道分配：C/2, C/4, 剩余给 bypass
        c_high = C // 2
        c_low  = C // 4
        c_id   = C - c_high - c_low
        self.c_high, self.c_low, self.c_id = c_high, c_low, c_id

        # 投影：保持标准1x1（别用DW，避免你担心的mAP上限）
        self.proj_id   = Conv(C, c_id,   k=1, s=1)
        self.proj_high = Conv(C, c_high, k=1, s=1)
        self.proj_low  = Conv(C, c_low,  k=1, s=1)

        # 从频域子带提取门控信息
        # 高频：cat(LH,HL,HH) -> conv -> 通道注意力
        self.high_conv = Conv(3 * C, c_high, k=1, s=1)
        self.high_att  = FCattLite(c_high, r=se_r)  # 输出 sigmoid 权重 [B,c_high,1,1]

        # 低频：LL -> conv -> 通道注意力（图里可以只有 sigmoid，但这里用轻SE更稳）
        self.low_conv = Conv(C, c_low, k=1, s=1)
        self.low_att  = FCattLite(c_low, r=se_r)

        self.refine = Conv(C, C, k=1, s=1) if refine else nn.Identity()

    def forward(self, x):
        # DWT
        LL, LH, HL, HH = self.dwt(x)  # each [B,C,H/2,W/2]

        # gate weights
        high_feat = self.high_conv(torch.cat([LH, HL, HH], dim=1))  # [B,c_high,h,w]
        A_high = self.high_att(high_feat)                           # [B,c_high,1,1]

        low_feat = self.low_conv(LL)                                # [B,c_low,h,w]
        A_low = self.low_att(low_feat)                              # [B,c_low,1,1]

        # three projections from spatial x
        xid   = self.proj_id(x)                                     # [B,c_id,H,W]
        xhigh = self.proj_high(x) * A_high                          # [B,c_high,H,W]
        xlow  = self.proj_low(x)  * A_low                           # [B,c_low,H,W]

        y = torch.cat([xid, xhigh, xlow], dim=1)                    # [B,C,H,W]
        y = self.refine(y)
        return y
class WEC(nn.Module):
    def __init__(
        self,
        c1,
        c2,
        n=2,
        e=0.5,
        shortcut=True,
        use_wave=True,
        beta_init=0.05,
        use_edge=True,
        use_ca=True,
        se_r=16,
        wave_refine=True
    ):
        super().__init__()
        self.use_wave = bool(use_wave)
        self.use_edge = bool(use_edge)
        self.use_ca = bool(use_ca)

        # 统一通道
        self.cv1 = Conv(c1, c2, k=1, s=1)

        c_hidden = int(c2 * e)

        # CSP 两条分支
        self.cv2 = Conv(c2, c_hidden, k=1, s=1)  # 走 blocks 的分支
        self.cv3 = Conv(c2, c_hidden, k=1, s=1)  # 直连分支

        # 内部 n 层 Unit（不含 Wave/Edge）
        self.blocks = nn.Sequential(*[
            CSPNextUWUnitLite(c_hidden, shortcut=shortcut, e=0.5)
            for _ in range(int(n))
        ])

        if self.use_wave:
            self.wave = WaveletGateLite(c_hidden, se_r=se_r, refine=wave_refine)
            self.beta = nn.Parameter(torch.tensor(float(beta_init)))

        # 拼接融合
        self.cv4 = Conv(2 * c_hidden, c2, k=1, s=1)

        # CA：只做一次（输出处）
        if self.use_ca:
            self.ca = ECA(c2)  # 你实现若是“返回特征”就直接 y=self.ca(y)

        # Edge/Spatial：只做一次（输出处）
        if self.use_edge:
            self.edge_attn = EdgeSpatialAttention(c2)

    def forward(self, x):
        x = self.cv1(x)                 # [B,c2,H,W]q
        y1 = self.blocks(self.cv2(x))   # [B,c_hidden,H,W]
        y2 = self.cv3(x)                # [B,c_hidden,H,W]

        # Wave 只注入一次：补偿式（小beta更稳）
        if self.use_wave:
            y1 = y1 + self.beta * self.wave(y1)

        y = self.cv4(torch.cat([y1, y2], dim=1))  # [B,c2,H,W]

        # CA 只做一次（输出处）
        if self.use_ca:
            y = self.ca(y)

        # Edge 只做一次（输出处）
        if self.use_edge:
            y = self.edge_attn(y)

        return y


class FCattLite(nn.Module):
    """通道权重 [B,C,1,1]，带压缩，参数显著少于 C->C 的 1x1"""
    def __init__(self, c: int, r: int = 16):
        super().__init__()
        c_mid = max(8, c // r)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c, c_mid, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(c_mid, c, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        w = self.pool(x)
        w = self.fc2(self.act(self.fc1(w)))
        return self.gate(w)

class HaarDWT2D(nn.Module):
    """Haar DWT: x(B,C,H,W) -> LL,LH,HL,HH (each B,C,H/2,W/2)."""
    def __init__(self):
        super().__init__()
        # 2x2 haar filters (normalize by 2)
        ll = torch.tensor([[1., 1.],
                           [1., 1.]]) * 0.5
        lh = torch.tensor([[-1., 1.],
                           [-1., 1.]]) * 0.5
        hl = torch.tensor([[-1., -1.],
                           [ 1.,  1.]]) * 0.5
        hh = torch.tensor([[ 1., -1.],
                           [-1.,  1.]]) * 0.5
        base = torch.stack([ll, lh, hl, hh], dim=0)  # [4,2,2]
        self.register_buffer("base", base[None, ...])  # [1,4,2,2]

    def forward(self, x):
        B, C, H, W = x.shape
        pad_h = H & 1
        pad_w = W & 1
        if pad_h or pad_w:
            # 只为保证偶数尺寸；reflect更稳一点
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        # weight: [4C,1,2,2] with groups=C
        w = self.base.repeat(C, 1, 1, 1)  # [C,4,2,2]
        w = w.view(4 * C, 1, 2, 2).contiguous()

        y = F.conv2d(x, w, stride=2, padding=0, groups=C)  # [B,4C,H/2,W/2]
        LL, LH, HL, HH = torch.chunk(y, 4, dim=1)
        return LL, LH, HL, HH





def channel_shuffle(x, groups: int = 2):
    b, c, h, w = x.shape
    if groups <= 1 or (c % groups) != 0:
        return x
    x = x.view(b, groups, c // groups, h, w).transpose(1, 2).contiguous()
    return x.view(b, c, h, w)

class DWCon(nn.Module):
    """Depthwise conv with dilation, BN + SiLU"""
    def __init__(self, c, k=3, d=1):
        super().__init__()
        p = (k // 2) * d
        self.conv = nn.Conv2d(c, c, k, 1, p, dilation=d, groups=c, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)  # 如果你想按图用 PReLU：nn.PReLU(c)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class PDPM(nn.Module):
    """
    PDPM: Parallel Dilated Pyramid (lite)
      1x1 reduce -> parallel dilated DWConv -> concat -> 1x1 expand
    """
    def __init__(self, c, r=2, rates=(1, 2, 3)):
        super().__init__()
        cm = max(8, c // r)
        self.pre = Conv(c, cm, k=1, s=1)
        self.branches = nn.ModuleList([DWCon(cm, k=3, d=int(rt)) for rt in rates])
        self.post = Conv(len(rates) * cm, c, k=1, s=1)

    def forward(self, x):
        x = self.pre(x)
        ys = [b(x) for b in self.branches]
        return self.post(torch.cat(ys, dim=1))
class DualGateFuse2(nn.Module):
    """Fuse(y1,y2): cat -> conv -> channel gate -> spatial gate"""
    def __init__(self, c_in, c_out=None, r=8):
        super().__init__()
        c_out = c_in if c_out is None else c_out

        # cat 后就是 c_in（在 PDPMC3K2 里就是 c2）
        self.fuse = Conv(c_in, c_out, k=1, s=1)

        cr = max(8, c_out // r)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.ch = nn.Sequential(
            nn.Conv2d(c_out, cr, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(cr, c_out, 1, bias=True),
            nn.Sigmoid()
        )

        self.sp = nn.Sequential(
            DWCon(c_out, k=3, d=1),
            nn.Conv2d(c_out, 1, 1, bias=True),
            nn.Sigmoid()
        )

        self.out = Conv(c_out, c_out, k=1, s=1)

    def forward(self, a, b):
        ab = torch.cat([a, b], dim=1)   # (B, c_in, H, W)
        f = self.fuse(ab)               # (B, c_out, H, W)

        f = f * self.ch(self.avg(f))    # channel gate
        f = f * self.sp(f)              # spatial gate
        return self.out(f)
# class PDPMC3K2(nn.Module):
#     """ 0.871
#     PDPMC3K2: 用图里的框架替换 C3k2（neck 用）
#     - 输入:  (B, c1, H, W)
#     - 输出:  (B, c2, H, W)
#     结构（对应图）：
#       Align(1x1) -> Split
#         -> (1x1) -> PDPM
#         -> (1x1) -> PDPM
#       Concat -> Add(input) -> ChannelShuffle
#       Skip: DW3x3 -> 1x1
#       Out = main + skip
#     """
#     def __init__(self, c1, c2, e=0.5, rates=(1, 2, 3), shuffle_g=2):
#         super().__init__()
#         self.c2 = int(c2)
#         self.shuffle_g = int(shuffle_g)
#
#         # 0) 对齐到 c2（替换 C3k2 必须保证输出通道一致）
#         self.align = Conv(c1, c2, k=1, s=1) if c1 != c2 else nn.Identity()
#
#         # 1) split 两支
#         c_ = max(1, int(c2 * float(e)))
#         c_ = min(c_ , c2 - 1)  # 保证两支都非 0
#         self.c_ = c_
#
#         self.pre1 = Conv(c_, c_, k=1, s=1)
#         self.pre2 = Conv(c2 - c_, c2 - c_, k=1, s=1)
#
#         self.pdpm1 = PDPM(c_, r=2, rates=tuple(rates))
#         self.pdpm2 = PDPM(c2 - c_, r=2, rates=tuple(rates))
#
#         # 2) skip 分支（对应图上方 3x3DW + 1x1）
#         self.skip_dw = DWCon(c2, k=3, d=1)
#         self.skip_pw = Conv(c2, c2, k=1, s=1)
#
#         # 3) 输出前再轻混一下（可选，但对 neck 通常更稳）
#         self.mix = Conv(c2, c2, k=1, s=1)
#
#         self.fuse2 = DualGateFuse2(c2)
#     def forward(self, x):
#         x = self.align(x)  # (B,c2,H,W)
#
#         x1 = x[:, :self.c_]
#         x2 = x[:, self.c_:]
#
#         y1 = self.pdpm1(self.pre1(x1))
#         y2 = self.pdpm2(self.pre2(x2))
#
#         y = self.fuse2(y1, y2)  # 需要 __init__ 里 self.fuse2 = DualGateFuse2(c2)
#         y = y + x
#
#         # y = channel_shuffle(y, self.shuffle_g)
#
#         # 对应图上方 skip path
#         s = self.skip_pw(self.skip_dw(x))
#
#         out = self.mix(y + s)
#         return out
class DualGateFuse2Residual(nn.Module):
    """
    Fuse(y1,y2): cat -> 1x1 conv -> residual channel gate -> residual spatial gate -> 1x1 conv
    - 残差式门控：不会把特征直接乘没（比 f=f*w 更稳）
    """
    def __init__(self, c_in, c_out=None, r=8):
        super().__init__()
        c_out = c_in if c_out is None else c_out

        # cat 后就是 c_in（在 PDPMC3K2 里通常就是 c2）
        self.fuse = Conv(c_in, c_out, k=1, s=1)

        cr = max(8, c_out // r)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.ch = nn.Sequential(
            nn.Conv2d(c_out, cr, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(cr, c_out, 1, bias=True),
            nn.Sigmoid()
        )

        # spatial gate：从融合后的 f 生成 (B,1,H,W)
        self.sp_dw = DWCon(c_out, k=3, d=1)
        self.sp_pw = nn.Conv2d(c_out, 1, 1, bias=True)
        self.sp_act = nn.Sigmoid()

        self.out = Conv(c_out, c_out, k=1, s=1)

    def forward(self, a, b):
        ab = torch.cat([a, b], dim=1)     # (B, c_in, H, W)
        f = self.fuse(ab)                 # (B, c_out, H, W)

        w_ch = self.ch(self.avg(f))       # (B, c_out, 1, 1)
        f = f * (1.0 + w_ch)              # residual channel gate

        w_sp = self.sp_act(self.sp_pw(self.sp_dw(f)))  # (B,1,H,W)
        f = f * (1.0 + w_sp)              # residual spatial gate

        return self.out(f)

class SelectivePDPM(nn.Module):
    """
    Selective-PDPM (ASPP-lite -> Adaptive RF selection)
      1x1 reduce -> parallel dilated DWConv -> softmax branch weights -> weighted sum -> 1x1 expand
    - 不再 concat，减少带宽/显存，并显式学习“选哪个 dilation”
    """
    def __init__(self, c, r=2, rates=(1, 2, 3), gate_r=4):
        super().__init__()
        self.rates = tuple(rates)
        self.n = len(self.rates)

        cm = max(8, c // r)
        self.pre = Conv(c, cm, k=1, s=1)

        self.branches = nn.ModuleList([DWCon(cm, k=3, d=int(rt)) for rt in self.rates])

        # gating：从 reduced feature (cm) 预测每个 dilation 分支权重
        hidden = max(8, cm // gate_r)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(cm, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, self.n, 1, bias=True)  # logits
        )

        self.post = Conv(cm, c, k=1, s=1)

    def forward(self, x):
        z = self.pre(x)                          # (B, cm, H, W)
        ys = [b(z) for b in self.branches]       # list of (B, cm, H, W)

        logits = self.gate(self.pool(z))         # (B, n, 1, 1)
        w = torch.softmax(logits, dim=1)         # (B, n, 1, 1)

        y = 0
        for i, yi in enumerate(ys):
            y = y + yi * w[:, i:i+1]             # broadcast to (B,1,H,W)

        return self.post(y)
# -------------------------------------------------------
# 3) 推荐版 PDPMC3K2：一支增强（SelectivePDPM），一支轻保真
# -------------------------------------------------------
class PDPMC3K2(nn.Module):
    """
    PDPMC3K2 (recommended):
      Align -> Split (CSP style)
        - Enhance branch: 1x1 -> SelectivePDPM
        - Preserve branch: light 1x1 (or Identity)
      Fuse (DualGateFuse2Residual) -> +x
      (optional) skip path: DW3x3 -> 1x1
      out = mix(y + skip)
    """
    def __init__(
        self,
        c1,
        c2,
        e=0.5,                 # 增强分支通道占比：建议 0.33~0.5 之间试
        rates=(1, 2, 3),
        use_skip=True,
        preserve_identity=True  # True: 保真分支=Identity；False: 1x1轻混合
    ):
        super().__init__()
        self.c2 = int(c2)
        self.use_skip = bool(use_skip)

        # 0) 对齐到 c2
        self.align = Conv(c1, c2, k=1, s=1) if c1 != c2 else nn.Identity()

        # 1) split：一支增强，一支保真
        c_ = max(1, int(c2 * float(e)))
        c_ = min(c_, c2 - 1)
        self.c_ = c_

        # Enhance branch (更“重”)
        self.enh_pre = Conv(c_, c_, k=1, s=1)
        self.enh = SelectivePDPM(c_, r=2, rates=tuple(rates))

        # Preserve branch (更“轻”)
        if preserve_identity:
            self.pre_pre = nn.Identity()
        else:
            self.pre_pre = Conv(c2 - c_, c2 - c_, k=1, s=1)

        # 2) 融合：图里那种 channel+spatial gate（残差式）
        self.fuse2 = DualGateFuse2Residual(c_in=c2, c_out=c2, r=8)

        # 3) skip 分支（可选，先保留更稳；无效再消融删）
        if self.use_skip:
            self.skip_dw = DWCon(c2, k=3, d=1)
            self.skip_pw = Conv(c2, c2, k=1, s=1)

        # 4) 输出再混一下（可选）
        self.mix = Conv(c2, c2, k=1, s=1)

        # 可学习缩放（让模块更容易“显现贡献”，也更稳）
        self.beta = nn.Parameter(torch.zeros(1))  # 主分支缩放，初始0更稳
        self.gamma = nn.Parameter(torch.ones(1))  # skip 缩放

    def forward(self, x):
        x = self.align(x)  # (B, c2, H, W)

        x1 = x[:, :self.c_]      # enhance
        x2 = x[:, self.c_:]      # preserve

        y1 = self.enh(self.enh_pre(x1))
        y2 = self.pre_pre(x2)

        y = self.fuse2(y1, y2)   # (B, c2, H, W)

        # 残差：用 beta 控制“增强幅度”（初始0，训练更稳，也更容易观察是否学到东西）
        y = x + self.beta * y

        if self.use_skip:
            s = self.skip_pw(self.skip_dw(x))
            out = self.mix(y + self.gamma * s)
        else:
            out = self.mix(y)

        return out
class SelectivePDPM_v2(nn.Module):
    """
    Selective-PDPM v2:
      1x1 reduce -> [Identity + parallel dilated DWConv] -> softmax weights -> weighted sum -> 1x1 expand
    关键：多一个 Identity 分支，允许“别动特征”，对小目标更友好。
    """
    def __init__(self, c, r=2, rates=(1, 2, 3), gate_r=4, tau=1.0, learnable_tau=False):
        super().__init__()
        self.rates = tuple(int(x) for x in rates)
        self.use_tau = bool(learnable_tau)

        cm = max(8, c // r)
        self.pre = Conv(c, cm, k=1, s=1)

        # 分支：0=Identity，其余=dilated DWConv
        self.branches = nn.ModuleList([nn.Identity()] + [DWCon(cm, k=3, d=int(rt)) for rt in self.rates])
        self.n = len(self.branches)

        hidden = max(8, cm // gate_r)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(cm, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, self.n, 1, bias=True)  # logits
        )

        if self.use_tau:
            self.tau = nn.Parameter(torch.tensor(float(tau)))
        else:
            self.register_buffer("tau", torch.tensor(float(tau)))

        self.post = Conv(cm, c, k=1, s=1)

    def forward(self, x):
        z = self.pre(x)                      # (B, cm, H, W)
        ys = [b(z) for b in self.branches]   # list of (B, cm, H, W)

        logits = self.gate(self.pool(z))     # (B, n, 1, 1)

        tau = self.tau
        if self.use_tau:
            tau = tau.clamp(0.5, 2.0)        # 防止数值发散
        w = torch.softmax(logits / tau, dim=1)

        y = 0.0
        for i, yi in enumerate(ys):
            y = y + yi * w[:, i:i+1]
        return self.post(y)
class DualGateFuse2Residual_v2(nn.Module):
    def __init__(self, c_in, c_out=None, r=8):
        super().__init__()
        c_out = c_in if c_out is None else c_out
        r = max(1, int(r))

        self.fuse = Conv(c_in, c_out, k=1, s=1)

        cr = max(8, c_out // r)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.ch = nn.Sequential(
            nn.Conv2d(c_out, cr, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(cr, c_out, 1, bias=True),
            nn.Sigmoid()
        )

        self.sp_dw = DWCon(c_out, k=3, d=1)
        self.sp_pw = nn.Conv2d(c_out, 1, 1, bias=True)
        self.sp_act = nn.Sigmoid()

        # 两个强度参数，初始0 => 初始不改变特征
        self.alpha_ch = nn.Parameter(torch.zeros(1))
        self.alpha_sp = nn.Parameter(torch.zeros(1))

        self.out = Conv(c_out, c_out, k=1, s=1)

    def forward(self, a, b):
        f = self.fuse(torch.cat([a, b], dim=1))

        w_ch = self.ch(self.avg(f))          # (0,1)
        w_ch = 2.0 * (w_ch - 0.5)            # (-1,1)
        f = f * (1.0 + self.alpha_ch * w_ch) # 初始≈1

        w_sp = self.sp_act(self.sp_pw(self.sp_dw(f)))  # (0,1)
        w_sp = 2.0 * (w_sp - 0.5)                       # (-1,1)
        f = f * (1.0 + self.alpha_sp * w_sp)

        return self.out(f)
class SSF(nn.Module):
    def __init__(self, c1, c2, e=0.5, rates=(1,2,3), use_skip=True, preserve_identity=True):
        super().__init__()
        self.c2 = int(c2)
        self.use_skip = bool(use_skip)

        self.align = Conv(c1, c2, k=1, s=1) if c1 != c2 else nn.Identity()

        c_ = max(1, int(c2 * float(e)))
        c_ = min(c_, c2 - 1)
        self.c_ = c_

        self.enh_pre = Conv(c_, c_, k=1, s=1)
        self.enh = SelectivePDPM_v2(c_, r=2, rates=tuple(rates), gate_r=4, tau=1.2, learnable_tau=False)

        self.pre_pre = nn.Identity() if preserve_identity else Conv(c2 - c_, c2 - c_, k=1, s=1)

        self.fuse2 = DualGateFuse2Residual_v2(c_in=c2, c_out=c2, r=8)

        if self.use_skip:
            self.skip_dw = DWCon(c2, k=3, d=1)
            self.skip_pw = Conv(c2, c2, k=1, s=1)

        self.mix = Conv(c2, c2, k=1, s=1)

        self.beta = nn.Parameter(torch.zeros(1))
        self.gamma = nn.Parameter(torch.zeros(1))  # ✅ 改这里（原来 ones）

    def forward(self, x):
        x = self.align(x)
        x1, x2 = x[:, :self.c_], x[:, self.c_:]

        y1 = self.enh(self.enh_pre(x1))
        y2 = self.pre_pre(x2)

        y = self.fuse2(y1, y2)
        y = x + self.beta * y

        if self.use_skip:
            s = self.skip_pw(self.skip_dw(x))
            out = self.mix(y + self.gamma * s)
        else:
            out = self.mix(y)

        return out

class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1: int = 16):
        """
        Initialize a convolutional layer with a given number of input channels.

        Args:
            c1 (int): Number of input channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the DFL module to input tensor and return transformed output."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """Ultralytics YOLO models mask Proto module for segmentation models."""

    def __init__(self, c1: int, c_: int = 256, c2: int = 32):
        """
        Initialize the Ultralytics YOLO models mask Proto module with specified number of protos and masks.

        Args:
            c1 (int): Input channels.
            c_ (int): Intermediate channels.
            c2 (int): Output channels (number of protos).
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1: int, cm: int, c2: int):
        """
        Initialize the StemBlock of PPHGNetV2.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(
        self,
        c1: int,
        cm: int,
        c2: int,
        k: int = 3,
        n: int = 6,
        lightconv: bool = False,
        shortcut: bool = False,
        act: nn.Module = nn.ReLU(),
    ):
        """
        Initialize HGBlock with specified parameters.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            n (int): Number of LightConv or Conv blocks.
            lightconv (bool): Whether to use LightConv.
            shortcut (bool): Whether to use shortcut connection.
            act (nn.Module): Activation function.
        """
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1: int, c2: int, k: tuple[int, ...] = (5, 9, 13)):
        """
        Initialize the SPP layer with input/output channels and pooling kernel sizes.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (tuple): Kernel sizes for max pooling.
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1: int, c2: int, k: int = 5):
        """
        Initialize the SPPF layer with given input/output channels and kernel size.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.

        Notes:
            This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sequential pooling operations to input and return concatenated feature maps."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1: int, c2: int, n: int = 1):
        """
        Initialize the CSP Bottleneck with 1 convolution.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of convolutions.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution and residual connection to input tensor."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize a CSP Bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """
        Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class HaarDWT(nn.Module):
    """
    简单的 2D Haar DWT：
      输入:  x, 形状 [B, C, H, W] （H,W 需要是偶数）
      输出:  LL, LH, HL, HH, 每个都是 [B, C, H/2, W/2]
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        assert h % 2 == 0 and w % 2 == 0, "H, W 必须是 2 的整数倍才能做 DWT"

        # 四个子采样位置
        a = x[:, :, 0::2, 0::2]  # top-left
        b_ = x[:, :, 0::2, 1::2]  # top-right
        c_ = x[:, :, 1::2, 0::2]  # bottom-left
        d_ = x[:, :, 1::2, 1::2]  # bottom-right

        # Haar 小波变换（正交变换，缩放因子这里用 0.5）
        # 经典形式：LL, LH, HL, HH
        LL = (a + b_ + c_ + d_) * 0.5
        LH = (a - b_ + c_ - d_) * 0.5
        HL = (a + b_ - c_ - d_) * 0.5
        HH = (a - b_ - c_ + d_) * 0.5

        return LL, LH, HL, HH


class HaarIDWT(nn.Module):
    """
    简单的 2D Haar IDWT：
      输入:  LL, LH, HL, HH, 每个 [B, C, H/2, W/2]
      输出:  x, 形状 [B, C, H, W]
    """
    def __init__(self):
        super().__init__()

    def forward(self, LL: torch.Tensor, LH: torch.Tensor,
                HL: torch.Tensor, HH: torch.Tensor):
        b, c, h, w = LL.shape

        # 逆变换（与上面的正变换成对）
        a = (LL + LH + HL + HH) * 0.5
        b_ = (LL - LH + HL - HH) * 0.5
        c_ = (LL + LH - HL - HH) * 0.5
        d_ = (LL - LH - HL + HH) * 0.5

        # 还原回原始空间尺寸
        x = torch.zeros(b, c, h * 2, w * 2, device=LL.device, dtype=LL.dtype)
        x[:, :, 0::2, 0::2] = a
        x[:, :, 0::2, 1::2] = b_
        x[:, :, 1::2, 0::2] = c_
        x[:, :, 1::2, 1::2] = d_

        return x
import math
def eca_kernel(channels, gamma=2, b=1):
    k = int(abs((math.log2(channels) / gamma) + b))
    return k if k % 2 == 1 else k + 1

class ECA(nn.Module):
    def __init__(self, channels, k_size=None):
        super().__init__()
        if k_size is None: k_size = max(3, eca_kernel(channels))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.gap(x)
        y = self.conv(y.squeeze(-1).transpose(1, 2))
        y = self.sigmoid(y).transpose(1, 2).unsqueeze(-1)
        return x * y
class PartialUMDCNeckBlock(nn.Module):
    """
    Partial-UMDC：
      输入: x [B, C, H, W]
      步骤:
        1) 1x1 Conv -> c_hidden
        2) 通道划分：c_md (多域) + c_id (轻量)
        3) 多域分支 (UMDC):
             - spatial_md: DWConv 3x3
             - dwt_md    : Haar DWT -> 子带卷积 -> Haar IDWT
             - branch attention 在 spatial / dwt 两个分支之间做权重
        4) identity 分支:
             - 可选 DWConv 3x3 或纯 identity
        5) concat -> 1x1 Conv -> 残差

    ratio_md: 多域分支占 c_hidden 的比例 (0~1)，例如 0.5 表示一半通道走 UMDC。
    """
    def __init__(
        self,
        c: int,
        e: float = 0.5,
        ratio_md: float = 0.5,
        shortcut: bool = True,
        use_dwt: bool = True,
        use_id_conv: bool = True,
        use_cbam:bool = True
    ):
        super().__init__()
        self.shortcut = shortcut
        self.use_dwt = use_dwt
        self.use_id_conv = use_id_conv
        self.use_cbam=use_cbam
        c_hidden = int(c * e)
        assert c_hidden > 0, "c_hidden 必须 > 0"

        # 1x1 降维
        self.cv1 = Conv(c, c_hidden, k=1, s=1)

        # 通道划分
        c_md = int(c_hidden * ratio_md)
        c_md = max(1, c_md)              # 至少 1
        c_md = min(c_md, c_hidden)       # 不超过总通道
        c_id = c_hidden - c_md           # 剩余通道

        self.c_hidden = c_hidden
        self.c_md = c_md
        self.c_id = c_id

        # 多域分支: 只对 c_md 通道做 UMDC
        self.spatial_md = Conv(c_md, c_md, k=3, s=1, g=c_md)

        if use_dwt:
            self.dwt = HaarDWT()
            self.idwt = HaarIDWT()
            self.conv_ll = Conv(c_md, c_md, k=3, s=1)
            self.conv_h = Conv(3 * c_md, 3 * c_md, k=3, s=1)
            self.branch_attn = nn.Sequential(
                nn.Linear(2, 2, bias=False),
                nn.Softmax(dim=-1)
            )

        # identity 分支: 对 c_id 通道，做一个很轻的 DWConv 或直接 Identity
        if self.c_id > 0:
            if use_id_conv:
                self.id_conv = Conv(c_id, c_id, k=3, s=1, g=c_id)
            else:
                self.id_conv = nn.Identity()
        else:
            self.id_conv = None

        # 1x1 投影回原通道
        self.cv2 = Conv(c_hidden, c, k=1, s=1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor):
        # x: [B, C, H, W]
        y = self.cv1(x)  # [B, c_hidden, H, W]
        b, c_hidden, h, w = y.shape
        orig_h, orig_w = h, w

        # 通道划分: [c_md, c_id]
        if self.c_id > 0:
            y_md, y_id = torch.split(y, [self.c_md, self.c_id], dim=1)
        else:
            y_md, y_id = y, None

        # -----------------------------
        # 多域分支 (UMDC) on y_md
        # -----------------------------
        # 为了 DWT，保证 H,W 为偶数，必要时向下/右 pad 1 像素
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            y_md_pad = F.pad(y_md, (0, pad_w, 0, pad_h))  # (left, right, top, bottom)
        else:
            y_md_pad = y_md

        # spatial 分支
        y_spatial = self.spatial_md(y_md_pad)

        # 如果没开 DWT，就只用 spatial
        if not self.use_dwt:
            # 裁剪回原尺寸
            y_spatial = y_spatial[:, :, :orig_h, :orig_w]
            y_md_fused = y_spatial
        else:
            # DWT 子带分解
            _, _, h_pad, w_pad = y_md_pad.shape
            assert h_pad % 2 == 0 and w_pad % 2 == 0, "pad 后 H,W 必须为偶数"

            LL, LH, HL, HH = self.dwt(y_md_pad)  # 每个 [B, c_md, H/2, W/2]

            # 低频卷积
            LL = self.conv_ll(LL)

            # 高频三路 concat 一起卷积，再拆分
            high = torch.cat([LH, HL, HH], dim=1)         # [B, 3*c_md, H/2, W/2]
            high = self.conv_h(high)
            LH2, HL2, HH2 = torch.chunk(high, 3, dim=1)

            # 逆变换还原
            y_dwt = self.idwt(LL, LH2, HL2, HH2)          # [B, c_md, H_pad, W_pad]

            # 裁剪回原尺寸
            y_spatial = y_spatial[:, :, :orig_h, :orig_w]
            y_dwt = y_dwt[:, :, :orig_h, :orig_w]

            # 分支级注意力：在 spatial / dwt 两个分支之间做权重
            v_spatial = y_spatial.mean(dim=(1, 2, 3))  # [B]
            v_dwt = y_dwt.mean(dim=(1, 2, 3))          # [B]

            scores = torch.stack([v_spatial, v_dwt], dim=1)  # [B, 2]
            weights = self.branch_attn(scores)               # [B, 2]
            weights = weights.view(b, 2, 1, 1, 1)            # broadcast

            feats = torch.stack([y_spatial, y_dwt], dim=1)   # [B, 2, c_md, H, W]
            y_md_fused = (feats * weights).sum(dim=1)        # [B, c_md, H, W]

        # -----------------------------
        # identity 分支 on y_id
        # -----------------------------
        if self.c_id > 0 and y_id is not None:
            y_id_out = self.id_conv(y_id)  # [B, c_id, H, W]
            # 保护一次，万一 conv 修改了尺寸，裁回原 H,W
            if y_id_out.shape[2] != orig_h or y_id_out.shape[3] != orig_w:
                y_id_out = y_id_out[:, :, :orig_h, :orig_w]

            y_cat = torch.cat([y_md_fused, y_id_out], dim=1)  # [B, c_hidden, H, W]
        else:
            y_cat = y_md_fused  # 所有通道都走 UMDC

        # 1x1 投影 + 残差
        out = self.cv2(y_cat)

        if self.shortcut and out.shape == x.shape:
            out = x + out
        return self.act(out)








class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize the CSP Bottleneck with 3 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 3 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize C3 module with cross-convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1: int, c2: int, n: int = 3, e: float = 1.0):
        """
        Initialize CSP Bottleneck with a single convolution.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepConv blocks.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of RepC3 module."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize C3 module with TransformerBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Transformer blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize C3 module with GhostBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Ghost bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/Efficient-AI-Backbones."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1):
        """
        Initialize Ghost Bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """
        Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize CSP Bottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1: int, c2: int, s: int = 1, e: int = 4):
        """
        Initialize ResNet block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            e (int): Expansion ratio.
        """
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1: int, c2: int, s: int = 1, is_first: bool = False, n: int = 1, e: int = 4):
        """
        Initialize ResNet layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            is_first (bool): Whether this is the first layer.
            n (int): Number of ResNet blocks.
            e (int): Expansion ratio.
        """
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1: int, c2: int, nh: int = 1, ec: int = 128, gc: int = 512, scale: bool = False):
        """
        Initialize MaxSigmoidAttnBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            nh (int): Number of heads.
            ec (int): Embedding channels.
            gc (int): Guide channels.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of MaxSigmoidAttnBlock.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor.

        Returns:
            (torch.Tensor): Output tensor after attention.
        """
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, guide.shape[1], self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        ec: int = 128,
        nh: int = 1,
        gc: int = 512,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ):
        """
        Initialize C2f module with attention mechanism.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            ec (int): Embedding channels for attention.
            nh (int): Number of heads for attention.
            gc (int): Guide channels for attention.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through C2f layer with attention.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using split() instead of chunk().

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(
        self, ec: int = 256, ch: tuple[int, ...] = (), ct: int = 512, nh: int = 8, k: int = 3, scale: bool = False
    ):
        """
        Initialize ImagePoolingAttn module.

        Args:
            ec (int): Embedding channels.
            ch (tuple): Channel dimensions for feature maps.
            ct (int): Channel dimension for text embeddings.
            nh (int): Number of attention heads.
            k (int): Kernel size for pooling.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x: list[torch.Tensor], text: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of ImagePoolingAttn.

        Args:
            x (list[torch.Tensor]): List of input feature maps.
            text (torch.Tensor): Text embeddings.

        Returns:
            (torch.Tensor): Enhanced text embeddings.
        """
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Forward function of contrastive learning.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """
        Initialize BNContrastiveHead.

        Args:
            embed_dims (int): Embedding dimensions for features.
        """
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def fuse(self):
        """Fuse the batch normalization layer in the BNContrastiveHead module."""
        del self.norm
        del self.bias
        del self.logit_scale
        self.forward = self.forward_fuse

    def forward_fuse(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Passes input out unchanged."""
        return x

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Forward function of contrastive learning with batch normalization.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)

        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """
        Initialize RepBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize RepCSP layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepBottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int, n: int = 1):
        """
        Initialize CSP-ELAN layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for RepCSP.
            n (int): Number of RepCSP blocks.
        """
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int):
        """
        Initialize ELAN1 layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for convolutions.
        """
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1: int, c2: int):
        """
        Initialize AConv module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1: int, c2: int):
        """
        Initialize ADown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, k: int = 5):
        """
        Initialize SPP-ELAN block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            k (int): Kernel size for max pooling.
        """
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1: int, c2s: list[int], k: int = 1, s: int = 1, p: int | None = None, g: int = 1):
        """
        Initialize CBLinear module.

        Args:
            c1 (int): Input channels.
            c2s (list[int]): List of output channel sizes.
            k (int): Kernel size.
            s (int): Stride.
            p (int | None): Padding.
            g (int): Groups.
        """
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx: list[int]):
        """
        Initialize CBFuse module.

        Args:
            idx (list[int]): Indices for feature selection.
        """
        super().__init__()
        self.idx = idx

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through CBFuse layer.

        Args:
            xs (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Fused output tensor.
        """
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """
        Initialize CSP bottleneck layer with two convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C3f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))




class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, g: int = 1, shortcut: bool = True
    ):
        """
        Initialize C3k2 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        """
        Initialize C3k module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
            k (int): Kernel size.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed: int) -> None:
        """
        Initialize RepVGGDW module.

        Args:
            ed (int): Input and output channels.
        """
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """
        Fuse the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """
    Conditional Identity Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5, lk: bool = False):
        """
        Initialize the CIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            e (float): Expansion ratio.
            lk (bool): Whether to use RepVGGDW.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """
    C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(
        self, c1: int, c2: int, n: int = 1, shortcut: bool = False, lk: bool = False, g: int = 1, e: float = 0.5
    ):
        """
        Initialize C2fCIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of CIB modules.
            shortcut (bool): Whether to use shortcut connection.
            lk (bool): Whether to use local key connection.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        """
        Initialize multi-head attention module.

        Args:
            dim (int): Input dimension.
            num_heads (int): Number of attention heads.
            attn_ratio (float): Attention ratio for key dimension.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
        """
        Initialize the PSABlock.

        Args:
            c (int): Input and output channels.
            attn_ratio (float): Attention ratio for key dimension.
            num_heads (int): Number of attention heads.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute a forward pass through PSABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1: int, c2: int, e: float = 0.5):
        """
        Initialize PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute forward pass in PSA module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))




class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """
        Initialize C2PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the input tensor through a series of PSA blocks.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """
    C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA

        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """
        Initialize C2fPSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


class SCDown(nn.Module):
    """
    SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1: int, c2: int, k: int, s: int):
        """
        Initialize SCDown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply convolution and downsampling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Downsampled output tensor.
        """
        return self.cv2(self.cv1(x))


class TorchVision(nn.Module):
    """
    TorchVision module to allow loading any torchvision model.

    This class provides a way to load a model from the torchvision library, optionally load pre-trained weights, and customize the model by truncating or unwrapping layers.

    Attributes:
        m (nn.Module): The loaded torchvision model, possibly truncated and unwrapped.

    Args:
        model (str): Name of the torchvision model to load.
        weights (str, optional): Pre-trained weights to load. Default is "DEFAULT".
        unwrap (bool, optional): If True, unwraps the model to a sequential containing all but the last `truncate` layers. Default is True.
        truncate (int, optional): Number of layers to truncate from the end if `unwrap` is True. Default is 2.
        split (bool, optional): Returns output from intermediate child modules as list. Default is False.
    """

    def __init__(
        self, model: str, weights: str = "DEFAULT", unwrap: bool = True, truncate: int = 2, split: bool = False
    ):
        """
        Load the model and weights from torchvision.

        Args:
            model (str): Name of the torchvision model to load.
            weights (str): Pre-trained weights to load.
            unwrap (bool): Whether to unwrap the model.
            truncate (int): Number of layers to truncate.
            split (bool): Whether to split the output.
        """
        import torchvision  # scope for faster 'import ultralytics'

        super().__init__()
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = list(self.m.children())
            if isinstance(layers[0], nn.Sequential):  # Second-level for some models like EfficientNet, Swin
                layers = [*list(layers[0].children()), *layers[1:]]
            self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))
            self.split = split
        else:
            self.split = False
            self.m.head = self.m.heads = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor | list[torch.Tensor]): Output tensor or list of tensors.
        """
        if self.split:
            y = [x]
            y.extend(m(y[-1]) for m in self.m)
        else:
            y = self.m(x)
        return y


class AAttn(nn.Module):
    """
    Area-attention module for YOLO models, providing efficient attention mechanisms.

    This module implements an area-based attention mechanism that processes input features in a spatially-aware manner,
    making it particularly effective for object detection tasks.

    Attributes:
        area (int): Number of areas the feature map is divided.
        num_heads (int): Number of heads into which the attention mechanism is divided.
        head_dim (int): Dimension of each attention head.
        qkv (Conv): Convolution layer for computing query, key and value tensors.
        proj (Conv): Projection convolution layer.
        pe (Conv): Position encoding convolution layer.

    Methods:
        forward: Applies area-attention to input tensor.

    Examples:
        >>> attn = AAttn(dim=256, num_heads=8, area=4)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = attn(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, area: int = 1):
        """
        Initialize an Area-attention module for YOLO models.

        Args:
            dim (int): Number of hidden channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            area (int): Number of areas the feature map is divided.
        """
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, dim, 7, 1, 3, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the input tensor through the area-attention.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention.
        """
        B, C, H, W = x.shape
        N = H * W

        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
            B, N, _ = qkv.shape
        q, k, v = (
            qkv.view(B, N, self.num_heads, self.head_dim * 3)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        x = v @ attn.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            v = v.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        x = x + self.pe(v)
        return self.proj(x)


class ABlock(nn.Module):
    """
    Area-attention block module for efficient feature extraction in YOLO models.

    This module implements an area-attention mechanism combined with a feed-forward network for processing feature maps.
    It uses a novel area-based attention approach that is more efficient than traditional self-attention while
    maintaining effectiveness.

    Attributes:
        attn (AAttn): Area-attention module for processing spatial features.
        mlp (nn.Sequential): Multi-layer perceptron for feature transformation.

    Methods:
        _init_weights: Initializes module weights using truncated normal distribution.
        forward: Applies area-attention and feed-forward processing to input tensor.

    Examples:
        >>> block = ABlock(dim=256, num_heads=8, mlp_ratio=1.2, area=1)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 1.2, area: int = 1):
        """
        Initialize an Area-attention block module.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            area (int): Number of areas the feature map is divided.
        """
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        Initialize weights using a truncated normal distribution.

        Args:
            m (nn.Module): Module to initialize.
        """
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention and feed-forward processing.
        """
        x = x + self.attn(x)
        return x + self.mlp(x)


class A2C2f(nn.Module):
    """
    Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.

    This module extends the C2f architecture by incorporating area-attention and ABlock layers for improved feature
    processing. It supports both area-attention and standard convolution modes.

    Attributes:
        cv1 (Conv): Initial 1x1 convolution layer that reduces input channels to hidden channels.
        cv2 (Conv): Final 1x1 convolution layer that processes concatenated features.
        gamma (nn.Parameter | None): Learnable parameter for residual scaling when using area attention.
        m (nn.ModuleList): List of either ABlock or C3k modules for feature processing.

    Methods:
        forward: Processes input through area-attention or standard convolution pathway.

    Examples:
        >>> m = A2C2f(512, 512, n=1, a2=True, area=1)
        >>> x = torch.randn(1, 512, 32, 32)
        >>> output = m(x)
        >>> print(output.shape)
        torch.Size([1, 512, 32, 32])
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        a2: bool = True,
        area: int = 1,
        residual: bool = False,
        mlp_ratio: float = 2.0,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
    ):
        """
        Initialize Area-Attention C2f module.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of ABlock or C3k modules to stack.
            a2 (bool): Whether to use area attention blocks. If False, uses C3k blocks instead.
            area (int): Number of areas the feature map is divided.
            residual (bool): Whether to use residual connections with learnable gamma parameter.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            e (float): Channel expansion ratio for hidden channels.
            g (int): Number of groups for grouped convolutions.
            shortcut (bool): Whether to use shortcut connections in C3k blocks.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through A2C2f layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(-1, self.gamma.shape[0], 1, 1) * y
        return y


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network for transformer-based architectures."""

    def __init__(self, gc: int, ec: int, e: int = 4) -> None:
        """
        Initialize SwiGLU FFN with input dimension, output dimension, and expansion factor.

        Args:
            gc (int): Guide channels.
            ec (int): Embedding channels.
            e (int): Expansion factor.
        """
        super().__init__()
        self.w12 = nn.Linear(gc, e * ec)
        self.w3 = nn.Linear(e * ec // 2, ec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU transformation to input features."""
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class Residual(nn.Module):
    """Residual connection wrapper for neural network modules."""

    def __init__(self, m: nn.Module) -> None:
        """
        Initialize residual module with the wrapped module.

        Args:
            m (nn.Module): Module to wrap with residual connection.
        """
        super().__init__()
        self.m = m
        nn.init.zeros_(self.m.w3.bias)
        # For models with l scale, please change the initialization to
        # nn.init.constant_(self.m.w3.weight, 1e-6)
        nn.init.zeros_(self.m.w3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual connection to input features."""
        return x + self.m(x)


class SAVPE(nn.Module):
    """Spatial-Aware Visual Prompt Embedding module for feature enhancement."""

    def __init__(self, ch: list[int], c3: int, embed: int):
        """
        Initialize SAVPE module with channels, intermediate channels, and embedding dimension.

        Args:
            ch (list[int]): List of input channel dimensions.
            c3 (int): Intermediate channels.
            embed (int): Embedding dimension.
        """
        super().__init__()
        self.cv1 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3), Conv(c3, c3, 3), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity()
            )
            for i, x in enumerate(ch)
        )

        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 1), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity())
            for i, x in enumerate(ch)
        )

        self.c = 16
        self.cv3 = nn.Conv2d(3 * c3, embed, 1)
        self.cv4 = nn.Conv2d(3 * c3, self.c, 3, padding=1)
        self.cv5 = nn.Conv2d(1, self.c, 3, padding=1)
        self.cv6 = nn.Sequential(Conv(2 * self.c, self.c, 3), nn.Conv2d(self.c, self.c, 3, padding=1))

    def forward(self, x: list[torch.Tensor], vp: torch.Tensor) -> torch.Tensor:
        """Process input features and visual prompts to generate enhanced embeddings."""
        y = [self.cv2[i](xi) for i, xi in enumerate(x)]
        y = self.cv4(torch.cat(y, dim=1))

        x = [self.cv1[i](xi) for i, xi in enumerate(x)]
        x = self.cv3(torch.cat(x, dim=1))

        B, C, H, W = x.shape

        Q = vp.shape[1]

        x = x.view(B, C, -1)

        y = y.reshape(B, 1, self.c, H, W).expand(-1, Q, -1, -1, -1).reshape(B * Q, self.c, H, W)
        vp = vp.reshape(B, Q, 1, H, W).reshape(B * Q, 1, H, W)

        y = self.cv6(torch.cat((y, self.cv5(vp)), dim=1))

        y = y.reshape(B, Q, self.c, -1)
        vp = vp.reshape(B, Q, 1, -1)

        score = y * vp + torch.logical_not(vp) * torch.finfo(y.dtype).min
        score = F.softmax(score, dim=-1).to(y.dtype)
        aggregated = score.transpose(-2, -3) @ x.reshape(B, self.c, C // self.c, -1).transpose(-1, -2)

        return F.normalize(aggregated.transpose(-2, -3).reshape(B, Q, -1), dim=-1, p=2)
