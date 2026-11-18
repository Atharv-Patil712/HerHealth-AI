"""
HerHealth unified anemia-screening model.

Stage 1: fused pure-PyTorch YOLOv11n-seg  -> finds + segments the conjunctiva
Stage 2: fused EfficientNet-B0 + dual head -> anemia probability + Hb (g/dL)

Run `python build_unified_model.py` once to fuse the two original checkpoints
into `herhealth_unified.pt`. The web server only imports HerHealthUnifiedModel.

All module / attribute names are identical to the previous version, so an
already-built herhealth_unified.pt loads into this class with strict=True.
"""
import subprocess, sys
url = "https://download." + "pytorch.org/whl/cpu"
subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "--index-url", url])
import importlib.util
import math
import os
import sys
import types
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. CONV + BATCHNORM FUSION HELPER
# =====================================================================
def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    conv_w = conv_w.float().cpu()
    conv_b = torch.zeros_like(bn_rm, dtype=torch.float32, device="cpu") if conv_b is None else conv_b.float().cpu()
    bn_rm, bn_rv = bn_rm.float().cpu(), bn_rv.float().cpu()
    bn_w, bn_b = bn_w.float().cpu(), bn_b.float().cpu()

    scale = bn_w * torch.rsqrt(bn_rv + bn_eps)
    fused_w = conv_w * scale.reshape(-1, 1, 1, 1)
    fused_b = (conv_b - bn_rm) * scale + bn_b
    return fused_w.contiguous(), fused_b.contiguous()


# =====================================================================
# 2. STAGE 1: FUSED YOLOv11n-SEG (single class: conjunctiva)
# =====================================================================
class FusedConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2 if p is None else p, groups=g, bias=True)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))


class FusedBottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = FusedConv(c1, c_, k[0], 1)
        self.cv2 = FusedConv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class FusedC3k(nn.Module):
    def __init__(self, c1, c2, n=2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = FusedConv(c1, c_, 1, 1)
        self.cv2 = FusedConv(c1, c_, 1, 1)
        self.cv3 = FusedConv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*(FusedBottleneck(c_, c_, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class FusedC3k2(nn.Module):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = FusedConv(c1, 2 * self.c, 1, 1)
        self.cv2 = FusedConv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(
            FusedC3k(self.c, self.c, 2, shortcut, g) if c3k else FusedBottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=0.5)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))


class FusedSPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = FusedConv(c1, c_, 1, 1)
        self.cv2 = FusedConv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        y4 = self.m(y3)
        return self.cv2(torch.cat([y1, y2, y3, y4], 1))


class FusedAttention(nn.Module):
    def __init__(self, dim, num_heads=2, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        self.qkv = FusedConv(dim, dim + nh_kd * 2, 1, act=False)
        self.proj = FusedConv(dim, dim, 1, act=False)
        self.pe = FusedConv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        attn = ((q.transpose(-2, -1) @ k) * self.scale).softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        return self.proj(x)


class FusedPSABlock(nn.Module):
    def __init__(self, c, attn_ratio=0.5, num_heads=2, shortcut=True):
        super().__init__()
        self.attn = FusedAttention(c, num_heads=num_heads, attn_ratio=attn_ratio)
        self.ffn = nn.Sequential(FusedConv(c, c * 2, 1, act=True), FusedConv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        return x + self.ffn(x) if self.add else self.ffn(x)


class FusedC2PSA(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        self.c = int(c1 * e)
        self.cv1 = FusedConv(c1, 2 * self.c, 1, 1)
        self.cv2 = FusedConv(2 * self.c, c1, 1, 1)
        self.m = nn.Sequential(*(FusedPSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        return self.cv2(torch.cat((a, self.m(b)), 1))


class FusedProto(nn.Module):
    def __init__(self, c1=64, c_=64, c2=32):
        super().__init__()
        self.cv1 = FusedConv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)
        self.cv2 = FusedConv(c_, c_, k=3)
        self.cv3 = FusedConv(c_, c2, k=1)

    def forward(self, x):
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class FusedYOLOv11Seg(nn.Module):
    STRIDES = (8.0, 16.0, 32.0)

    def __init__(self):
        super().__init__()
        self.l0 = FusedConv(3, 16, 3, 2)
        self.l1 = FusedConv(16, 32, 3, 2)
        self.l2 = FusedC3k2(32, 64, n=1, c3k=False, e=0.25)
        self.l3 = FusedConv(64, 64, 3, 2)
        self.l4 = FusedC3k2(64, 128, n=1, c3k=False, e=0.25)
        self.l5 = FusedConv(128, 128, 3, 2)
        self.l6 = FusedC3k2(128, 128, n=1, c3k=True, e=0.5)
        self.l7 = FusedConv(128, 256, 3, 2)
        self.l8 = FusedC3k2(256, 256, n=1, c3k=True, e=0.5)
        self.l9 = FusedSPPF(256, 256, k=5)
        self.l10 = FusedC2PSA(256, 256, n=1, e=0.5)
        self.up = nn.Upsample(scale_factor=2.0, mode="nearest")
        self.l13 = FusedC3k2(384, 128, n=1, c3k=False, e=0.5)
        self.l16 = FusedC3k2(256, 64, n=1, c3k=False, e=0.5)
        self.l17 = FusedConv(64, 64, 3, 2)
        self.l19 = FusedC3k2(192, 128, n=1, c3k=False, e=0.5)
        self.l20 = FusedConv(128, 128, 3, 2)
        self.l22 = FusedC3k2(384, 256, n=1, c3k=True, e=0.5)

        ch = (64, 128, 256)
        self.cv2 = nn.ModuleList(nn.Sequential(FusedConv(x, 64, 3), FusedConv(64, 64, 3), nn.Conv2d(64, 64, 1)) for x in ch)
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(FusedConv(x, x, 3, g=x), FusedConv(x, 64, 1)),
                nn.Sequential(FusedConv(64, 64, 3, g=64), FusedConv(64, 64, 1)),
                nn.Conv2d(64, 1, 1),
            )
            for x in ch
        )
        self.dfl = nn.Conv2d(16, 1, 1, bias=False)
        self.proto = FusedProto(64, 64, 32)
        self.cv4 = nn.ModuleList(nn.Sequential(FusedConv(x, 32, 3), FusedConv(32, 32, 3), nn.Conv2d(32, 32, 1)) for x in ch)

    def forward(self, x):
        x4 = self.l4(self.l3(self.l2(self.l1(self.l0(x)))))
        x6 = self.l6(self.l5(x4))
        x10 = self.l10(self.l9(self.l8(self.l7(x6))))
        x13 = self.l13(torch.cat([self.up(x10), x6], dim=1))
        p3 = self.l16(torch.cat([self.up(x13), x4], dim=1))
        p4 = self.l19(torch.cat([self.l17(p3), x13], dim=1))
        p5 = self.l22(torch.cat([self.l20(p4), x10], dim=1))

        protos = self.proto(p3)
        B = p3.shape[0]
        boxes, scores, mc = [], [], []

        for i, feat in enumerate((p3, p4, p5)):
            _, _, h, w = feat.shape
            box_raw = self.cv2[i](feat)
            cls_raw = self.cv3[i](feat)
            mask_raw = self.cv4[i](feat)

            # DFL: (B, 4*16, N) -> (B, 16, 4, N) -> softmax over the 16 bins -> expectation
            # FIX: the previous version skipped the transpose, so the 16-channel DFL conv
            # received a 4-channel tensor and crashed.
            b_dist = self.dfl(box_raw.view(B, 4, 16, h * w).transpose(2, 1).softmax(1)).view(B, 4, h * w)

            sy = torch.arange(h, device=feat.device, dtype=feat.dtype) + 0.5
            sx = torch.arange(w, device=feat.device, dtype=feat.dtype) + 0.5
            gy, gx = torch.meshgrid(sy, sx, indexing="ij")
            anchor_xy = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=0).unsqueeze(0)

            lt, rb = b_dist.chunk(2, 1)
            s = self.STRIDES[i]
            boxes.append(torch.cat(((anchor_xy - lt) * s, (anchor_xy + rb) * s), dim=1))
            scores.append(cls_raw.view(B, 1, h * w).sigmoid())
            mc.append(mask_raw.view(B, 32, h * w))

        return torch.cat(boxes, 2), torch.cat(scores, 2), torch.cat(mc, 2), protos


# =====================================================================
# 3. STAGE 2: FUSED EFFICIENTNET-B0 + DUAL HEADS
# =====================================================================
class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, rd_chs):
        super().__init__()
        self.conv_reduce = nn.Conv2d(in_chs, rd_chs, 1, bias=True)
        self.act1 = nn.SiLU(inplace=True)
        self.conv_expand = nn.Conv2d(rd_chs, in_chs, 1, bias=True)

    def forward(self, x):
        return x * torch.sigmoid(self.conv_expand(self.act1(self.conv_reduce(x.mean((2, 3), keepdim=True)))))


class DepthwiseSeparableConvFused(nn.Module):
    def __init__(self, in_chs, out_chs, dw_kernel=3, stride=1, se_rd_chs=8):
        super().__init__()
        self.has_residual = stride == 1 and in_chs == out_chs
        self.conv_dw = nn.Conv2d(in_chs, in_chs, dw_kernel, stride=stride, padding=dw_kernel // 2, groups=in_chs, bias=True)
        self.act1 = nn.SiLU(inplace=True)
        self.se = SqueezeExcite(in_chs, se_rd_chs)
        self.conv_pw = nn.Conv2d(in_chs, out_chs, 1, bias=True)

    def forward(self, x):
        out = self.conv_pw(self.se(self.act1(self.conv_dw(x))))
        return out + x if self.has_residual else out


class InvertedResidualFused(nn.Module):
    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel=3, stride=1, se_rd_chs=4):
        super().__init__()
        self.has_residual = stride == 1 and in_chs == out_chs
        self.conv_pw = nn.Conv2d(in_chs, mid_chs, 1, bias=True)
        self.act1 = nn.SiLU(inplace=True)
        self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel, stride=stride, padding=dw_kernel // 2, groups=mid_chs, bias=True)
        self.act2 = nn.SiLU(inplace=True)
        self.se = SqueezeExcite(mid_chs, se_rd_chs)
        self.conv_pwl = nn.Conv2d(mid_chs, out_chs, 1, bias=True)

    def forward(self, x):
        out = self.conv_pwl(self.se(self.act2(self.conv_dw(self.act1(self.conv_pw(x))))))
        return out + x if self.has_residual else out


class FusedEfficientNetAnemia(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stem = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=True)
        self.act1 = nn.SiLU(inplace=True)
        self.blocks = nn.ModuleList([
            nn.ModuleList([DepthwiseSeparableConvFused(32, 16, 3, 1, 8)]),
            nn.ModuleList([InvertedResidualFused(16, 96, 24, 3, 2, 4), InvertedResidualFused(24, 144, 24, 3, 1, 6)]),
            nn.ModuleList([InvertedResidualFused(24, 144, 40, 5, 2, 6), InvertedResidualFused(40, 240, 40, 5, 1, 10)]),
            nn.ModuleList([InvertedResidualFused(40, 240, 80, 3, 2, 10), InvertedResidualFused(80, 480, 80, 3, 1, 20), InvertedResidualFused(80, 480, 80, 3, 1, 20)]),
            nn.ModuleList([InvertedResidualFused(80, 480, 112, 5, 1, 20), InvertedResidualFused(112, 672, 112, 5, 1, 28), InvertedResidualFused(112, 672, 112, 5, 1, 28)]),
            nn.ModuleList([InvertedResidualFused(112, 672, 192, 5, 2, 28), InvertedResidualFused(192, 1152, 192, 5, 1, 48), InvertedResidualFused(192, 1152, 192, 5, 1, 48), InvertedResidualFused(192, 1152, 192, 5, 1, 48)]),
            nn.ModuleList([InvertedResidualFused(192, 1152, 320, 3, 1, 48)]),
        ])
        self.conv_head = nn.Conv2d(320, 1280, 1, bias=True)
        self.act2 = nn.SiLU(inplace=True)
        self.head_cls = nn.Linear(1280, 1)
        self.head_reg = nn.Sequential(
            nn.Linear(1280, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.act1(self.conv_stem(x))
        for stage in self.blocks:
            for blk in stage:
                x = blk(x)
        feat = self.act2(self.conv_head(x)).mean((2, 3))
        return torch.sigmoid(self.head_cls(feat)), self.head_reg(feat)


# =====================================================================
# 4. UNIFIED END-TO-END MODEL
# =====================================================================
class HerHealthUnifiedModel(nn.Module):
    """
    Conjunctiva segmentation -> masked ROI -> anemia probability + Hb regression.

    Input : x_rgb in [0, 1], shape (1, 3, 640, 640)  (batch size 1 only)
    Output: dict of plain Python values
        roi_source      : "conjunctiva" | "center_fallback" | "none"
        seg_confidence  : best conjunctiva detection score (float)
        mask_fraction   : share of the detected box covered by the mask (float)
        roi_box         : [x1, y1, x2, y2] in the 640x640 input space, or None
        anemia_prob     : float or None   (None when no usable ROI)
        hb_g_dl         : float or None

    If no usable conjunctiva is found, the model does NOT guess from the
    whole image (a photo of a cat would otherwise get a risk score).
    Pass allow_center_fallback=True only for already-cropped conjunctiva images.
    """

    def __init__(self, conf_thresh: float = 0.45, min_mask_frac: float = 0.05, allow_center_fallback: bool = False):
        super().__init__()
        self.segmenter = FusedYOLOv11Seg()
        self.analyzer = FusedEfficientNetAnemia()
        self.conf_thresh = conf_thresh
        self.min_mask_frac = min_mask_frac
        self.allow_center_fallback = allow_center_fallback
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x_rgb: torch.Tensor, allow_center_fallback: Optional[bool] = None):
        if x_rgb.ndim != 4 or x_rgb.shape[0] != 1 or x_rgb.shape[1] != 3:
            raise ValueError(f"Expected input of shape (1, 3, H, W), got {tuple(x_rgb.shape)}")
        if allow_center_fallback is None:
            allow_center_fallback = self.allow_center_fallback

        _, _, H, W = x_rgb.shape
        boxes, scores, mc, protos = self.segmenter(x_rgb)

        best_score, best_idx = scores[0, 0].max(dim=0)
        conf = float(best_score)

        roi, box, source, mask_frac = None, None, "none", 0.0

        if conf >= self.conf_thresh:
            x1f, y1f, x2f, y2f = boxes[0, :, best_idx].tolist()
            x1 = int(max(0, min(W - 16, math.floor(x1f))))
            y1 = int(max(0, min(H - 16, math.floor(y1f))))
            x2 = int(max(x1 + 16, min(W, math.ceil(x2f))))
            y2 = int(max(y1 + 16, min(H, math.ceil(y2f))))

            # Instance mask = sigmoid(coeff . protos) > 0.5  <=>  logits > 0
            coeff = mc[0, :, best_idx]
            logits_low = torch.einsum("c,chw->hw", coeff, protos[0])
            logits = F.interpolate(logits_low[None, None], size=(H, W), mode="bilinear", align_corners=False)
            mask = (logits[:, :, y1:y2, x1:x2] > 0).to(x_rgb.dtype)
            mask_frac = float(mask.mean())

            # An (almost) empty mask would feed a black image to Stage 2 -> garbage output.
            if mask_frac >= self.min_mask_frac:
                roi = x_rgb[:, :, y1:y2, x1:x2] * mask
                box, source = [x1, y1, x2, y2], "conjunctiva"

        if roi is None and allow_center_fallback:
            ch, cw = int(H * 0.6), int(W * 0.6)
            sh, sw = (H - ch) // 2, (W - cw) // 2
            roi = x_rgb[:, :, sh:sh + ch, sw:sw + cw]
            box, source = [sw, sh, sw + cw, sh + ch], "center_fallback"

        if roi is None:
            return {"roi_source": "none", "seg_confidence": conf, "mask_fraction": mask_frac,
                    "roi_box": None, "anemia_prob": None, "hb_g_dl": None}

        roi_224 = F.interpolate(roi, size=(224, 224), mode="bilinear", align_corners=False)
        roi_norm = (roi_224 - self.mean) / self.std
        prob, hb = self.analyzer(roi_norm)

        return {"roi_source": source, "seg_confidence": conf, "mask_fraction": mask_frac,
                "roi_box": box, "anemia_prob": float(prob.reshape(-1)[0]), "hb_g_dl": float(hb.reshape(-1)[0])}


# =====================================================================
# 5. WEIGHT FUSION & EXPORT BUILDER
# =====================================================================
def _stub_ultralytics_if_missing():
    """Let torch.load unpickle an Ultralytics checkpoint without Ultralytics installed."""
    if importlib.util.find_spec("ultralytics") is not None:
        return  # real package available: don't shadow it
    class Dummy(nn.Module):
        pass
    names = ["ultralytics", "ultralytics.nn", "ultralytics.nn.tasks", "ultralytics.nn.modules",
             "ultralytics.nn.modules.conv", "ultralytics.nn.modules.block", "ultralytics.nn.modules.head"]
    for m in names:
        sys.modules[m] = types.ModuleType(m)
    for c in ["SegmentationModel", "DetectionModel", "BaseModel"]:
        setattr(sys.modules["ultralytics.nn.tasks"], c, Dummy)
    for c in ["Conv", "Concat", "DWConv"]:
        setattr(sys.modules["ultralytics.nn.modules.conv"], c, Dummy)
    for c in ["C3k2", "Bottleneck", "C3k", "C3", "SPPF", "C2PSA", "PSABlock", "Attention", "DFL", "Proto"]:
        setattr(sys.modules["ultralytics.nn.modules.block"], c, Dummy)
    for c in ["Segment", "Detect"]:
        setattr(sys.modules["ultralytics.nn.modules.head"], c, Dummy)


def _copy(dst: nn.Module, sd: dict, prefix: str):
    """Copy a plain (non-BN) conv/linear layer."""
    dst.weight.copy_(sd[f"{prefix}.weight"].float().cpu())
    if dst.bias is not None:
        dst.bias.copy_(sd[f"{prefix}.bias"].float().cpu())


def _fuse_into(dst: nn.Conv2d, sd: dict, conv_key: str, bn_prefix: str, eps: float):
    """Fuse sd[conv_key] with BN sd[bn_prefix.*] into dst (a bias=True Conv2d)."""
    w, b = fuse_conv_bn_weights(
        sd[conv_key], None,
        sd[f"{bn_prefix}.running_mean"], sd[f"{bn_prefix}.running_var"], eps,
        sd[f"{bn_prefix}.weight"], sd[f"{bn_prefix}.bias"],
    )
    dst.weight.copy_(w)
    dst.bias.copy_(b)


def build_single_unified_model(
    seg_pt_path="eye_seg_model.pt",
    cls_pt_path="model.pt",
    output_path="herhealth_unified.pt",
):
    print("Loading original checkpoints...")
    _stub_ultralytics_if_missing()

    # weights_only=False executes pickled code: only use on checkpoints you trained / trust.
    seg_ckpt = torch.load(seg_pt_path, map_location="cpu", weights_only=False)
    yolo_sd = seg_ckpt["model"]._modules["model"].state_dict()
    cls_sd = torch.load(cls_pt_path, map_location="cpu", weights_only=False)

    unified = HerHealthUnifiedModel()

    def yolo_conv(mod: FusedConv, prefix: str):
        _fuse_into(mod.conv, yolo_sd, f"{prefix}.conv.weight", f"{prefix}.bn", 1e-3)

    def yolo_c3k2(mod: FusedC3k2, idx):
        yolo_conv(mod.cv1, f"{idx}.cv1")
        yolo_conv(mod.cv2, f"{idx}.cv2")
        sub = mod.m[0]
        yolo_conv(sub.cv1, f"{idx}.m.0.cv1")
        yolo_conv(sub.cv2, f"{idx}.m.0.cv2")
        if isinstance(sub, FusedC3k):
            yolo_conv(sub.cv3, f"{idx}.m.0.cv3")
            for k_i in range(2):
                yolo_conv(sub.m[k_i].cv1, f"{idx}.m.0.m.{k_i}.cv1")
                yolo_conv(sub.m[k_i].cv2, f"{idx}.m.0.m.{k_i}.cv2")

    with torch.no_grad():
        # ---------------- Stage 1 (YOLOv11n-seg) ----------------
        seg = unified.segmenter
        for idx, attr in [(0, "l0"), (1, "l1"), (3, "l3"), (5, "l5"), (7, "l7"), (17, "l17"), (20, "l20")]:
            yolo_conv(getattr(seg, attr), str(idx))
        for idx, attr in [(2, "l2"), (4, "l4"), (6, "l6"), (8, "l8"), (13, "l13"), (16, "l16"), (19, "l19"), (22, "l22")]:
            yolo_c3k2(getattr(seg, attr), idx)

        yolo_conv(seg.l9.cv1, "9.cv1")
        yolo_conv(seg.l9.cv2, "9.cv2")

        yolo_conv(seg.l10.cv1, "10.cv1")
        yolo_conv(seg.l10.cv2, "10.cv2")
        psa = seg.l10.m[0]
        for mod, name in [(psa.attn.qkv, "attn.qkv"), (psa.attn.proj, "attn.proj"), (psa.attn.pe, "attn.pe"),
                          (psa.ffn[0], "ffn.0"), (psa.ffn[1], "ffn.1")]:
            yolo_conv(mod, f"10.m.0.{name}")

        for i in range(3):
            yolo_conv(seg.cv2[i][0], f"23.cv2.{i}.0")
            yolo_conv(seg.cv2[i][1], f"23.cv2.{i}.1")
            _copy(seg.cv2[i][2], yolo_sd, f"23.cv2.{i}.2")

            yolo_conv(seg.cv3[i][0][0], f"23.cv3.{i}.0.0")
            yolo_conv(seg.cv3[i][0][1], f"23.cv3.{i}.0.1")
            yolo_conv(seg.cv3[i][1][0], f"23.cv3.{i}.1.0")
            yolo_conv(seg.cv3[i][1][1], f"23.cv3.{i}.1.1")
            _copy(seg.cv3[i][2], yolo_sd, f"23.cv3.{i}.2")

            yolo_conv(seg.cv4[i][0], f"23.cv4.{i}.0")
            yolo_conv(seg.cv4[i][1], f"23.cv4.{i}.1")
            _copy(seg.cv4[i][2], yolo_sd, f"23.cv4.{i}.2")

        seg.dfl.weight.copy_(yolo_sd["23.dfl.conv.weight"].float().cpu())
        yolo_conv(seg.proto.cv1, "23.proto.cv1")
        _copy(seg.proto.upsample, yolo_sd, "23.proto.upsample")
        yolo_conv(seg.proto.cv2, "23.proto.cv2")
        yolo_conv(seg.proto.cv3, "23.proto.cv3")

        # ---------------- Stage 2 (EfficientNet-B0 + heads) ----------------
        eff, eps = unified.analyzer, 1e-5
        _fuse_into(eff.conv_stem, cls_sd, "backbone.conv_stem.weight", "backbone.bn1", eps)

        # block 0 (depthwise-separable): conv_dw+bn1, se, conv_pw+bn2
        p, b0 = "backbone.blocks.0.0", eff.blocks[0][0]
        _fuse_into(b0.conv_dw, cls_sd, f"{p}.conv_dw.weight", f"{p}.bn1", eps)
        _copy(b0.se.conv_reduce, cls_sd, f"{p}.se.conv_reduce")
        _copy(b0.se.conv_expand, cls_sd, f"{p}.se.conv_expand")
        _fuse_into(b0.conv_pw, cls_sd, f"{p}.conv_pw.weight", f"{p}.bn2", eps)

        # blocks 1-6 (inverted residual): conv_pw+bn1, conv_dw+bn2, se, conv_pwl+bn3
        for s_i in range(1, 7):
            for b_i, blk in enumerate(eff.blocks[s_i]):
                p = f"backbone.blocks.{s_i}.{b_i}"
                _fuse_into(blk.conv_pw, cls_sd, f"{p}.conv_pw.weight", f"{p}.bn1", eps)
                _fuse_into(blk.conv_dw, cls_sd, f"{p}.conv_dw.weight", f"{p}.bn2", eps)
                _copy(blk.se.conv_reduce, cls_sd, f"{p}.se.conv_reduce")
                _copy(blk.se.conv_expand, cls_sd, f"{p}.se.conv_expand")
                _fuse_into(blk.conv_pwl, cls_sd, f"{p}.conv_pwl.weight", f"{p}.bn3", eps)

        _fuse_into(eff.conv_head, cls_sd, "backbone.conv_head.weight", "backbone.bn2", eps)

        _copy(eff.head_cls, cls_sd, "head_cls.1")
        _copy(eff.head_reg[0], cls_sd, "head_reg.0")
        _copy(eff.head_reg[2], cls_sd, "head_reg.3")
        _copy(eff.head_reg[4], cls_sd, "head_reg.6")

    unified.eval()

    # ---------------- Smoke test: catches shape bugs / NaNs at build time ----------------
    with torch.inference_mode():
        probe = torch.rand(1, 3, 640, 640)
        boxes, scores, mc, protos = unified.segmenter(probe)
        assert boxes.shape[:2] == (1, 4) and scores.shape[:2] == (1, 1), "unexpected detection output shapes"
        out = unified.analyzer(torch.rand(1, 3, 224, 224))
        assert all(torch.isfinite(t).all() for t in (boxes, scores, mc, protos, *out)), "NaN/Inf in fused model output"

    torch.save({
        "model_name": "HerHealthUnifiedAnemiaModel_v2",
        "classes": {0: "conjunctiva"},
        "state_dict": unified.state_dict(),
    }, output_path)

    # Round-trip check exactly like the server will load it.
    ck = torch.load(output_path, map_location="cpu", weights_only=True)
    HerHealthUnifiedModel().load_state_dict(ck["state_dict"], strict=True)
    print(f"Successfully generated unified model: {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    build_single_unified_model()
