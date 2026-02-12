from __future__ import annotations

import argparse
import numpy as np
import cv2
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict, Any
from PIL import Image, ImageOps

try:
    import rawpy

    HAS_RAW = True
except Exception:
    HAS_RAW = False

RAW_EXTS = {
    ".arw",
    ".dng",
    ".cr2",
    ".cr3",
    ".nef", ".rw2",
}
IMG_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
}
ALL_EXTS = RAW_EXTS | IMG_EXTS


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1.0 + a)) ** 2.4).astype(
        np.float32
    )


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    a = 0.055
    return np.where(
        x <= 0.0031308, x * 12.92, (1.0 + a) * (x ** (1.0 / 2.4)) - a
    ).astype(np.float32)


def bgr16_to_float01(bgr16: np.ndarray) -> np.ndarray:
    return np.clip(bgr16.astype(np.float32) / 65535.0, 0.0, 1.0)


def float01_to_bgr8(bgr01: np.ndarray) -> np.ndarray:
    return np.clip(bgr01 * 255.0 + 0.5, 0, 255).astype(np.uint8)


def luma01_from_bgr01_srgb(bgr01: np.ndarray) -> np.ndarray:
    return 0.2126 * bgr01[..., 2] + 0.7152 * bgr01[..., 1] + 0.0722 * bgr01[..., 0]


def luma01_from_bgr01_linear(bgr01: np.ndarray) -> np.ndarray:
    rgb = bgr01[..., ::-1].astype(np.float32)
    rgb_lin = srgb_to_linear(rgb)
    return (
        0.2126 * rgb_lin[..., 0] + 0.7152 * rgb_lin[..., 1] + 0.0722 * rgb_lin[..., 2]
    ).astype(np.float32)


def smoothstep(x: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def apply_gain_linear_rgb(bgr01: np.ndarray, gain: float) -> np.ndarray:
    rgb = bgr01[..., ::-1].astype(np.float32)
    rgb_lin = srgb_to_linear(rgb)
    rgb_lin = np.clip(rgb_lin * float(gain), 0.0, 1.0)
    rgb_srgb = linear_to_srgb(rgb_lin)
    out = rgb_srgb[..., ::-1].copy()
    return np.clip(out, 0.0, 1.0)


@dataclass
class Config:
    jpeg_quality: int = 95

    raw_use_camera_wb: bool = True
    raw_use_auto_wb: bool = True
    raw_no_auto_bright: bool = True
    raw_output_color_srgb: bool = True
    raw_highlight_mode: int = 1

    xrite_name_hint: str = "xrite"
    xrite_skip_export: bool = True
    xrite_required: bool = False

    wb_sat_max: float = 0.20
    wb_y_min: float = 0.10
    wb_y_max: float = 0.95
    wb_max_channel: float = 0.995
    wb_min_pixels: int = 2500
    wb_gain_clip_min: float = 0.60
    wb_gain_clip_max: float = 1.70
    wb_strength: float = 1.00

    use_opencv_wb: bool = False
    opencv_wb_mode: str = "simple"
    opencv_wb_p: float = 0.5
    opencv_wb_saturation_thresh: float = 0.98

    bg_v_min: float = 0.62
    bg_s_max: float = 0.30
    bg_min_pixels: int = 1500
    border_frac: float = 0.10
    min_bg_for_border: int = 800

    spec_v_min: float = 0.95
    spec_s_max: float = 0.22
    spec_grad_min: float = 0.02

    exp_pctl: float = 90.0
    exp_target: float = 0.995
    exp_gain_min: float = 1.20
    exp_gain_max: float = 4.50
    exp_highlight_pctl: float = 99.6
    exp_highlight_cap: float = 0.998

    bg_whiten_enable: bool = True
    bg_whiten_pctl: float = 95.0
    bg_whiten_target: float = 0.992
    bg_whiten_gain_max: float = 2.25

    shadows_lift: float = 0.20
    shadows_start: float = 0.00
    shadows_end: float = 0.52
    shadows_max_boost: float = 2.10

    midtone_target: float = 0.85
    subject_white_cut: float = 0.965
    p_min: float = 1.00
    p_max: float = 3.00
    lift_strength: float = 0.75

    pop_strength: float = 0.65
    contrast_factor: float = 1.10
    contrast_pivot: float = 0.62
    microcontrast_amount: float = 0.12
    microcontrast_radius: float = 10.0

    vibrance: float = 0.08
    vibrance_max_boost: float = 1.12
    vibrance_highlight_start: float = 0.88
    vibrance_highlight_end: float = 0.99
    vibrance_neutral_chroma_max: float = 28.0
    vibrance_neutral_soft: float = 56.0

    highlight_protect_start: float = 0.88
    highlight_protect_end: float = 0.99
    highlight_protect_strength: float = 0.80

    highlight_roll_start: float = 0.92
    highlight_roll_strength: float = 0.55

    highlight_detint_enable: bool = True
    highlight_detint_start: float = 0.94
    highlight_detint_end: float = 0.995
    highlight_detint_strength: float = 0.65

    sharpness_amount: float = 1.0
    sharpness_radius: float = 1.0
    sharpness_threshold: int = 0

    final_brightness: float = 1.03


CFG = Config()


def iter_images(folder: Path) -> Iterable[Path]:
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in ALL_EXTS:
            yield p


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_jpeg_like_to_bgr16(path: Path) -> Optional[np.ndarray]:
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
            rgb8 = np.array(im, dtype=np.uint8)
        bgr8 = cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)
        return bgr8.astype(np.uint16) * 257
    except Exception as e:
        print(f"[WARN] PIL read failed {path}: {e}")
        return None


def read_raw_to_bgr16(path: Path, cfg: Config) -> Optional[np.ndarray]:
    if not HAS_RAW:
        print("[WARN] rawpy not installed; skipping RAW.")
        return None
    try:
        with rawpy.imread(str(path)) as raw:
            kwargs: Dict[str, Any] = dict(
                output_bps=16,
                no_auto_bright=cfg.raw_no_auto_bright,
                highlight_mode=int(cfg.raw_highlight_mode),
                user_sat=None,
            )

            if cfg.raw_use_auto_wb:
                kwargs["use_auto_wb"] = True
                kwargs["use_camera_wb"] = False
            else:
                kwargs["use_camera_wb"] = True
                kwargs["use_auto_wb"] = False

            if cfg.raw_output_color_srgb:
                kwargs["output_color"] = rawpy.ColorSpace.sRGB

            rgb16 = raw.postprocess(**kwargs)

        return cv2.cvtColor(rgb16, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"[WARN] RAW read failed {path}: {e}")
        return None


def read_image_any_to_bgr16(path: Path, cfg: Config) -> Optional[np.ndarray]:
    ext = path.suffix.lower()
    if ext in RAW_EXTS:
        return read_raw_to_bgr16(path, cfg)
    return read_jpeg_like_to_bgr16(path)


def write_jpeg(path: Path, bgr8: np.ndarray, quality: int) -> None:
    ensure_dir(path.parent)
    ok = cv2.imwrite(str(path), bgr8, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for {path}")


def _is_xrite_file(p: Path, cfg: Config) -> bool:
    hint = (cfg.xrite_name_hint or "").lower().strip()
    if not hint:
        return False
    return hint in p.name.lower()


def opencv_white_balance_bgr16(
    bgr16: np.ndarray, cfg: Config
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if not cfg.use_opencv_wb:
        return bgr16, {"applied": False, "reason": "disabled"}
    if not hasattr(cv2, "xphoto"):
        return bgr16, {"applied": False, "reason": "cv2.xphoto missing"}

    bgr8 = (bgr16 >> 8).astype(np.uint8)
    mode = cfg.opencv_wb_mode.lower().strip()
    try:
        if mode == "grayworld":
            wb = cv2.xphoto.createGrayworldWB()
            wb.setSaturationThreshold(float(cfg.opencv_wb_saturation_thresh))
        else:
            wb = cv2.xphoto.createSimpleWB()
            if hasattr(wb, "setP"):
                wb.setP(float(cfg.opencv_wb_p))

        wb8 = wb.balanceWhite(bgr8)

        mask = cv2.cvtColor(bgr8, cv2.COLOR_BGR2GRAY) > 16
        if int(mask.sum()) < 1000:
            mask = np.ones(bgr8.shape[:2], dtype=bool)

        src_med = np.median(bgr8[mask].reshape(-1, 3), axis=0).astype(np.float32) + 1e-6
        dst_med = np.median(wb8[mask].reshape(-1, 3), axis=0).astype(np.float32)

        gains = (dst_med / src_med).astype(np.float32)
        gains = np.clip(gains, cfg.wb_gain_clip_min, cfg.wb_gain_clip_max)

        out16 = np.clip(
            bgr16.astype(np.float32) * gains[None, None, :], 0.0, 65535.0
        ).astype(np.uint16)
        return out16, {"applied": True, "method": "opencv", "gains_bgr": gains.tolist()}
    except Exception as e:
        return bgr16, {"applied": False, "reason": f"wb_failed: {e}"}


def compute_anchor_wb_from_xrite_bgr16(
    bgr16: np.ndarray, cfg: Config
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    bgr01 = bgr16_to_float01(bgr16)
    bgr8 = float01_to_bgr8(bgr01)
    hsv = cv2.cvtColor(bgr8, cv2.COLOR_BGR2HSV).astype(np.float32)
    S = hsv[..., 1] / 255.0
    Y = luma01_from_bgr01_srgb(bgr01)

    mask = (
        (S <= float(cfg.wb_sat_max))
        & (Y >= float(cfg.wb_y_min))
        & (Y <= float(cfg.wb_y_max))
        & (np.max(bgr01, axis=2) <= float(cfg.wb_max_channel))
    )

    n = int(mask.sum())
    if n < int(cfg.wb_min_pixels):
        mask = (
            (S <= float(cfg.wb_sat_max) * 1.5)
            & (Y >= float(cfg.wb_y_min) * 0.7)
            & (Y <= float(cfg.wb_y_max) * 1.05)
        )
        n = int(mask.sum())

    if n < 500:
        return None, {"ok": False, "reason": "too_few_neutral_pixels", "n": n}

    rgb01 = bgr01[..., ::-1].astype(np.float32)
    rgb_lin = srgb_to_linear(rgb01)

    samp = rgb_lin[mask].reshape(-1, 3).astype(np.float32)
    med = np.median(samp, axis=0).astype(np.float32) + 1e-8

    tgt = float(np.mean(med))
    gains = (tgt / med).astype(np.float32)
    gains = np.clip(gains, cfg.wb_gain_clip_min, cfg.wb_gain_clip_max)

    a = float(np.clip(cfg.wb_strength, 0.0, 1.0))
    gains = 1.0 + a * (gains - 1.0)

    return gains, {
        "ok": True,
        "n": n,
        "rgb_lin_median": med.tolist(),
        "wb_gains_rgb_lin": gains.tolist(),
        "strength": a,
    }


def apply_anchor_wb_linear_rgb(
    bgr01: np.ndarray, wb_gains_rgb_lin: np.ndarray
) -> np.ndarray:
    rgb = bgr01[..., ::-1].astype(np.float32)
    rgb_lin = srgb_to_linear(rgb)
    rgb_lin = np.clip(rgb_lin * wb_gains_rgb_lin[None, None, :], 0.0, 1.0)
    rgb_srgb = linear_to_srgb(rgb_lin)
    out_bgr = rgb_srgb[..., ::-1].copy()
    return np.clip(out_bgr, 0.0, 1.0)


def detect_white_product_scene(bgr01: np.ndarray) -> Dict[str, Any]:
    Y = luma01_from_bgr01_srgb(bgr01)
    y_med = float(np.median(Y))
    y_p95 = float(np.percentile(Y, 95.0))

    is_dark_subject = y_med < 0.25

    bgr8 = float01_to_bgr8(bgr01)
    hsv = cv2.cvtColor(bgr8, cv2.COLOR_BGR2HSV).astype(np.float32)
    s_med = float(np.median(hsv[..., 1] / 255.0))
    is_whiteish = (s_med < 0.10) and (y_p95 > 0.88)

    return {
        "is_whiteish": bool(is_whiteish),
        "is_dark_subject": bool(is_dark_subject),
        "s_med": s_med,
        "y_med": y_med,
    }


def _border_mask(h: int, w: int, frac: float) -> np.ndarray:
    b = int(max(1, round(min(h, w) * float(frac))))
    m = np.zeros((h, w), dtype=bool)
    m[:b, :] = True
    m[-b:, :] = True
    m[:, :b] = True
    m[:, -b:] = True
    return m


def compute_bg_mask(bgr01: np.ndarray, cfg: Config) -> np.ndarray:
    bgr8 = float01_to_bgr8(bgr01)
    hsv = cv2.cvtColor(bgr8, cv2.COLOR_BGR2HSV).astype(np.float32)
    S = hsv[..., 1] / 255.0
    V = hsv[..., 2] / 255.0

    v_thr = float(cfg.bg_v_min)
    s_thr = float(cfg.bg_s_max)
    bg = (V >= v_thr) & (S <= s_thr)

    if int(bg.sum()) < int(cfg.min_bg_for_border):
        h, w = bg.shape
        bm = _border_mask(h, w, float(cfg.border_frac))
        s_border = S[bm]
        v_border = V[bm]

        s_p35 = float(np.percentile(s_border, 35.0))
        v_p65 = float(np.percentile(v_border, 65.0))

        s_thr2 = min(s_thr, max(0.10, s_p35))
        v_thr2 = max(v_thr, min(0.92, v_p65))

        bg2 = (V >= v_thr2) & (S <= s_thr2)
        if int(bg2.sum()) > int(bg.sum()):
            bg = bg2

    bg_u8 = bg.astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bg_u8 = cv2.morphologyEx(bg_u8, cv2.MORPH_OPEN, k, iterations=1)
    bg_u8 = cv2.morphologyEx(bg_u8, cv2.MORPH_CLOSE, k, iterations=2)
    return bg_u8 > 0


def compute_specular_mask(bgr01: np.ndarray, cfg: Config) -> np.ndarray:
    bgr8 = float01_to_bgr8(bgr01)
    hsv = cv2.cvtColor(bgr8, cv2.COLOR_BGR2HSV).astype(np.float32)
    S = hsv[..., 1] / 255.0
    V = hsv[..., 2] / 255.0

    Y = luma01_from_bgr01_linear(bgr01)
    gx = cv2.Sobel(Y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(Y, cv2.CV_32F, 0, 1, ksize=3)
    g = np.sqrt(gx * gx + gy * gy)
    g = np.clip(g, 0.0, 1.0)

    spec = (
        (V >= float(cfg.spec_v_min))
        & (S <= float(cfg.spec_s_max))
        & (g >= float(cfg.spec_grad_min))
    )

    spec_u8 = spec.astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    spec_u8 = cv2.dilate(spec_u8, k, iterations=1)
    return spec_u8 > 0


def compute_subject_mask(bg_mask: np.ndarray) -> np.ndarray:
    subj = ~bg_mask
    subj_u8 = subj.astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    subj_u8 = cv2.morphologyEx(subj_u8, cv2.MORPH_CLOSE, k, iterations=2)
    subj_u8 = cv2.morphologyEx(subj_u8, cv2.MORPH_OPEN, k, iterations=1)
    return subj_u8 > 0


def apply_exposure_from_bg(
    bgr01: np.ndarray, bg_mask: np.ndarray, spec_mask: np.ndarray, cfg: Config
) -> Tuple[np.ndarray, Dict[str, Any]]:
    Y = luma01_from_bgr01_linear(bgr01)

    usable = bg_mask & (~spec_mask)
    if int(usable.sum()) < int(cfg.bg_min_pixels):
        usable = np.ones_like(bg_mask, dtype=bool)
        ref = "all"
    else:
        ref = "bg"

    base = float(np.percentile(Y[usable], float(cfg.exp_pctl)))
    base = max(base, 1e-6)
    gain_target = float(cfg.exp_target) / base

    hi = float(np.percentile(Y[usable], float(cfg.exp_highlight_pctl)))
    hi = max(hi, 1e-6)
    gain_hi = float(cfg.exp_highlight_cap) / hi

    gain = float(min(gain_target, gain_hi))
    gain = float(np.clip(gain, cfg.exp_gain_min, cfg.exp_gain_max))

    out = apply_gain_linear_rgb(bgr01, gain)
    return out, {"ref": ref, "base": base, "gain": gain}


def bg_whiten_failsafe(
    bgr01: np.ndarray, bg_mask: np.ndarray, spec_mask: np.ndarray, cfg: Config
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if not cfg.bg_whiten_enable:
        return bgr01, {"applied": False}

    Y = luma01_from_bgr01_linear(bgr01)
    usable = bg_mask & (~spec_mask)
    if int(usable.sum()) < int(cfg.bg_min_pixels):
        return bgr01, {"applied": False, "reason": "too_few_bg_pixels"}

    base = float(np.percentile(Y[usable], float(cfg.bg_whiten_pctl)))
    base = max(base, 1e-6)

    if base >= float(cfg.bg_whiten_target) - 1e-3:
        return bgr01, {"applied": False, "base": base}

    gain = float(cfg.bg_whiten_target) / base
    gain = float(np.clip(gain, 1.0, float(cfg.bg_whiten_gain_max)))

    out = apply_gain_linear_rgb(bgr01, gain)
    return out, {"applied": True, "base": base, "gain": gain}


def apply_shadows_lift(
    bgr01: np.ndarray, cfg: Config
) -> Tuple[np.ndarray, Dict[str, Any]]:
    strength = float(np.clip(cfg.shadows_lift, 0.0, 1.0))
    if strength <= 1e-6:
        return bgr01, {"applied": False}

    Y = luma01_from_bgr01_srgb(bgr01)
    s0 = float(np.clip(cfg.shadows_start, 0.0, 0.95))
    s1 = float(np.clip(cfg.shadows_end, s0 + 1e-3, 1.0))

    w = 1.0 - smoothstep(Y, s0, s1)
    w *= strength

    gamma = 1.0 / (1.0 + 1.8 * strength)
    Y_lift = np.power(np.clip(Y, 0.0, 1.0), gamma)

    ratio = Y_lift / np.maximum(Y, 1e-4)
    ratio = np.clip(ratio, 1.0, float(cfg.shadows_max_boost)).astype(np.float32)

    out = np.clip(bgr01 * (1.0 + w[..., None] * (ratio[..., None] - 1.0)), 0.0, 1.0)
    return out, {"applied": True, "strength": strength, "gamma": gamma, "end": s1}


def adaptive_midtone_lift(
    bgr01: np.ndarray, cfg: Config, scene_info: Dict
) -> Tuple[np.ndarray, Dict[str, Any]]:
    Y = luma01_from_bgr01_srgb(bgr01)

    subj_mask = Y < cfg.subject_white_cut
    if int(subj_mask.sum()) < 2000:
        subj_mask = np.ones(Y.shape, dtype=bool)

    med = float(np.median(Y[subj_mask]))
    tgt = float(cfg.midtone_target)

    is_dark_scene = False
    if med < 0.25:
        is_dark_scene = True
        tgt = max(med * 2.5, 0.40)
        tgt = min(tgt, cfg.midtone_target)

    med = min(max(med, 1e-6), 0.999999)
    one_minus_med = max(1.0 - med, 1e-6)
    one_minus_tgt = max(1.0 - tgt, 1e-6)
    p = float(np.log(one_minus_tgt) / np.log(one_minus_med))
    p = float(np.clip(p, cfg.p_min, cfg.p_max))

    Yp = 1.0 - np.power(np.clip(1.0 - Y, 0.0, 1.0), p)
    scale = Yp / np.maximum(Y, 1e-4)
    scale = np.clip(scale, 0.75, 2.2).astype(np.float32)

    lifted = np.clip(bgr01 * scale[..., None], 0.0, 1.0)
    a = float(np.clip(cfg.lift_strength, 0.0, 1.0))
    if scene_info.get("is_whiteish", False):
        a *= 0.75

    out = np.clip(bgr01 * (1.0 - a) + lifted * a, 0.0, 1.0)
    return out, {
        "applied": True,
        "subject_median": med,
        "target": tgt,
        "is_dark_fix": is_dark_scene,
        "p": p,
    }


def highlight_protect_blend(original: np.ndarray, edited: np.ndarray, cfg: Config):
    Yo = luma01_from_bgr01_srgb(original)
    w = smoothstep(Yo, cfg.highlight_protect_start, cfg.highlight_protect_end)
    strength = float(np.clip(cfg.highlight_protect_strength, 0.0, 1.0))
    w = w * strength
    out = np.clip(edited * (1.0 - w[..., None]) + original * w[..., None], 0.0, 1.0)
    return out, {"applied": True}


def highlight_rolloff(
    bgr01: np.ndarray, cfg: Config
) -> Tuple[np.ndarray, Dict[str, Any]]:
    strength = float(np.clip(cfg.highlight_roll_strength, 0.0, 1.0))
    start = float(np.clip(cfg.highlight_roll_start, 0.0, 0.999))
    if strength <= 1e-6:
        return bgr01, {"applied": False}

    Y = luma01_from_bgr01_srgb(bgr01)
    t = np.clip((Y - start) / max(1.0 - start, 1e-6), 0.0, 1.0)
    Y2 = Y - strength * (t * t) * (Y - start)
    Y2 = np.clip(Y2, 0.0, 1.0)

    ratio = Y2 / np.maximum(Y, 1e-4)
    ratio = np.clip(ratio, 0.80, 1.0).astype(np.float32)
    out = np.clip(bgr01 * ratio[..., None], 0.0, 1.0)
    return out, {"applied": True}


def highlight_detint(
    bgr01: np.ndarray, cfg: Config
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if not cfg.highlight_detint_enable:
        return bgr01, {"applied": False}

    Y = luma01_from_bgr01_srgb(bgr01)
    w = smoothstep(
        Y, float(cfg.highlight_detint_start), float(cfg.highlight_detint_end)
    )
    a = float(np.clip(cfg.highlight_detint_strength, 0.0, 1.0))
    w = w * a
    if float(w.max()) <= 1e-6:
        return bgr01, {"applied": False}

    m = np.max(bgr01, axis=2, keepdims=True)
    out = np.clip(bgr01 * (1.0 - w[..., None]) + m * w[..., None], 0.0, 1.0)
    return out, {"applied": True}


def apply_vibrance_ycrcb_protected(
    bgr01: np.ndarray, cfg: Config
) -> Tuple[np.ndarray, Dict[str, Any]]:
    v = float(np.clip(cfg.vibrance, 0.0, 2.0))
    if v <= 1e-6:
        return bgr01, {"applied": False}

    Y_orig = luma01_from_bgr01_srgb(bgr01).astype(np.float32)

    bgr8 = float01_to_bgr8(bgr01)
    ycrcb = cv2.cvtColor(bgr8, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    Y8 = ycrcb[..., 0]
    Cr = ycrcb[..., 1]
    Cb = ycrcb[..., 2]

    dCr = Cr - 128.0
    dCb = Cb - 128.0
    chroma = np.sqrt(dCr * dCr + dCb * dCb)

    chroma_norm = np.clip(chroma / 128.0, 0.0, 1.0)
    base_boost = 1.0 + v * (1.0 - chroma_norm)
    base_boost = np.clip(base_boost, 1.0, float(cfg.vibrance_max_boost))

    Y01 = (Y8 / 255.0).astype(np.float32)
    wh = smoothstep(Y01, cfg.vibrance_highlight_start, cfg.vibrance_highlight_end)
    highlight_keep = 1.0 - wh

    c0 = float(max(cfg.vibrance_neutral_chroma_max, 0.0))
    c1 = float(max(cfg.vibrance_neutral_soft, c0 + 1.0))
    neutral_ramp = np.clip((chroma - c0) / (c1 - c0), 0.0, 1.0)

    eff = 1.0 + (base_boost - 1.0) * (highlight_keep * neutral_ramp)

    dCr2 = dCr * eff
    dCb2 = dCb * eff

    Cr2 = np.clip(128.0 + dCr2, 0.0, 255.0)
    Cb2 = np.clip(128.0 + dCb2, 0.0, 255.0)
    ycrcb[..., 1] = Cr2
    ycrcb[..., 2] = Cb2

    out8 = cv2.cvtColor(
        np.clip(ycrcb + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR
    )
    out = out8.astype(np.float32) / 255.0

    Y_new = luma01_from_bgr01_srgb(out).astype(np.float32)
    ratio = Y_orig / np.maximum(Y_new, 1e-4)
    ratio = np.clip(ratio, 0.95, 1.05).astype(np.float32)
    out = np.clip(out * ratio[..., None], 0.0, 1.0)

    return np.clip(out, 0.0, 1.0), {"applied": True}


def apply_unsharp_mask(bgr8: np.ndarray, cfg: Config) -> np.ndarray:
    if cfg.sharpness_amount <= 0:
        return bgr8
    blurred = cv2.GaussianBlur(bgr8, (0, 0), cfg.sharpness_radius)
    sharpened = cv2.addWeighted(
        bgr8, 1.0 + cfg.sharpness_amount, blurred, -cfg.sharpness_amount, 0
    )
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_subject_contrast(
    bgr01: np.ndarray, subject_mask: np.ndarray, cfg: Config
) -> Tuple[np.ndarray, Dict[str, Any]]:
    pop = float(np.clip(cfg.pop_strength, 0.0, 1.0))
    if pop <= 1e-6:
        return bgr01, {"applied": False}

    cf = 1.0 + (float(cfg.contrast_factor) - 1.0) * pop
    Y = luma01_from_bgr01_srgb(bgr01)
    pivot = float(np.clip(cfg.contrast_pivot, 0.0, 1.0))
    Yc = np.clip(pivot + cf * (Y - pivot), 0.0, 1.0)

    ratio = Yc / np.maximum(Y, 1e-4)
    ratio = np.clip(ratio, 0.65, 1.35).astype(np.float32)

    out = bgr01.copy()
    out_subj = np.clip(bgr01 * ratio[..., None], 0.0, 1.0)
    out[subject_mask] = out_subj[subject_mask]
    return out, {"applied": True, "cf": cf, "pivot": pivot}


def apply_subject_microcontrast(
    bgr01: np.ndarray, subject_mask: np.ndarray, cfg: Config
) -> Tuple[np.ndarray, Dict[str, Any]]:
    pop = float(np.clip(cfg.pop_strength, 0.0, 1.0))
    amt = float(np.clip(cfg.microcontrast_amount, 0.0, 2.0)) * pop
    if amt <= 1e-6:
        return bgr01, {"applied": False}

    Y = luma01_from_bgr01_srgb(bgr01)
    blur = cv2.GaussianBlur(Y, (0, 0), float(cfg.microcontrast_radius))
    detail = Y - blur
    Y2 = np.clip(Y + amt * detail, 0.0, 1.0)
    ratio = Y2 / np.maximum(Y, 1e-4)
    ratio = np.clip(ratio, 0.90, 1.20).astype(np.float32)

    out = bgr01.copy()
    out_subj = np.clip(bgr01 * ratio[..., None], 0.0, 1.0)
    out[subject_mask] = out_subj[subject_mask]
    return out, {"applied": True, "amt": amt, "radius": float(cfg.microcontrast_radius)}


def process_bgr16(
    bgr16: np.ndarray,
    cfg: Config,
    anchor_wb_gains_rgb_lin: Optional[np.ndarray],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    dbg: Dict[str, Any] = {}

    bgr16_2, wb16_dbg = opencv_white_balance_bgr16(bgr16, cfg)
    dbg["wb_opencv"] = wb16_dbg

    bgr01 = bgr16_to_float01(bgr16_2)

    if anchor_wb_gains_rgb_lin is not None:
        bgr01 = apply_anchor_wb_linear_rgb(bgr01, anchor_wb_gains_rgb_lin)
        dbg["wb_anchor"] = {
            "applied": True,
            "gains_rgb_lin": anchor_wb_gains_rgb_lin.tolist(),
        }
    else:
        dbg["wb_anchor"] = {"applied": False, "reason": "no_anchor_gains"}

    scene = detect_white_product_scene(bgr01)
    dbg["scene"] = scene

    bg_mask = compute_bg_mask(bgr01, cfg)
    spec_mask = compute_specular_mask(bgr01, cfg)

    pre_exposure_ref = bgr01.copy()

    bgr01, exp_dbg = apply_exposure_from_bg(bgr01, bg_mask, spec_mask, cfg)
    dbg["exposure_bg_linear"] = exp_dbg

    fb = float(cfg.final_brightness)
    if abs(fb - 1.0) > 1e-4:
        bgr01 = apply_gain_linear_rgb(bgr01, fb)
        dbg["final_brightness"] = {"applied": True, "gain": fb}
    else:
        dbg["final_brightness"] = {"applied": False}

    bg_mask2 = compute_bg_mask(bgr01, cfg)
    spec_mask2 = compute_specular_mask(bgr01, cfg)
    bgr01, wb_bg_dbg = bg_whiten_failsafe(bgr01, bg_mask2, spec_mask2, cfg)
    dbg["bg_whiten_failsafe"] = wb_bg_dbg

    bgr01, hp_dbg = highlight_protect_blend(pre_exposure_ref, bgr01, cfg)
    dbg["highlight_protect"] = hp_dbg

    bg_mask3 = compute_bg_mask(bgr01, cfg)
    spec_mask3 = compute_specular_mask(bgr01, cfg)
    subj_mask = compute_subject_mask(bg_mask3)
    dbg["mask_counts"] = {
        "bg": int(bg_mask3.sum()),
        "subj": int(subj_mask.sum()),
        "spec": int(spec_mask3.sum()),
    }

    bgr01, sh_dbg = apply_shadows_lift(bgr01, cfg)
    dbg["shadows_lift"] = sh_dbg

    bgr01, lift_dbg = adaptive_midtone_lift(bgr01, cfg, scene)
    dbg["midtone_lift"] = lift_dbg

    bgr01, v_dbg = apply_vibrance_ycrcb_protected(bgr01, cfg)
    dbg["vibrance_ycrcb"] = v_dbg

    bgr01, c_dbg = apply_subject_contrast(bgr01, subj_mask, cfg)
    bgr01, mc_dbg = apply_subject_microcontrast(bgr01, subj_mask, cfg)
    dbg["subject_contrast"] = c_dbg
    dbg["subject_microcontrast"] = mc_dbg

    bgr01, hr_dbg = highlight_rolloff(bgr01, cfg)
    bgr01, hd_dbg = highlight_detint(bgr01, cfg)
    dbg["highlight_rolloff"] = hr_dbg
    dbg["highlight_detint"] = hd_dbg

    bgr8 = float01_to_bgr8(bgr01)
    bgr8 = apply_unsharp_mask(bgr8, cfg)
    return bgr8, dbg


def find_xrite_file(input_dir: Path, cfg: Config) -> Optional[Path]:
    candidates = [p for p in iter_images(input_dir) if _is_xrite_file(p, cfg)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: str(p).lower())
    return candidates[0]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="input", help="Input folder")
    ap.add_argument("--output", type=str, default="output", help="Output folder")
    ap.add_argument(
        "--quality", type=int, default=CFG.jpeg_quality, help="JPEG quality"
    )

    ap.add_argument(
        "--raw-auto-wb",
        action="store_true",
        help="Use rawpy auto WB (instead of camera WB) for the main render",
    )

    ap.add_argument("--sharpness", type=float, default=CFG.sharpness_amount)
    ap.add_argument("--vibrance", type=float, default=CFG.vibrance)

    ap.add_argument("--contrast", type=float, default=CFG.contrast_factor)
    ap.add_argument("--contrast-pivot", type=float, default=CFG.contrast_pivot)
    ap.add_argument("--pop", type=float, default=CFG.pop_strength)
    ap.add_argument("--microcontrast", type=float, default=CFG.microcontrast_amount)
    ap.add_argument("--micro-radius", type=float, default=CFG.microcontrast_radius)

    ap.add_argument(
        "--shadows", type=float, default=CFG.shadows_lift, help="Shadows lift (0..1)"
    )
    ap.add_argument(
        "--shadows-end",
        type=float,
        default=CFG.shadows_end,
        help="Shadows fade-out luma (0..1)",
    )

    ap.add_argument("--xrite-hint", type=str, default=CFG.xrite_name_hint)
    ap.add_argument("--no-skip-xrite-export", action="store_true")
    ap.add_argument("--xrite-required", action="store_true")

    ap.add_argument("--wb-strength", type=float, default=CFG.wb_strength)
    ap.add_argument("--wb-sat-max", type=float, default=CFG.wb_sat_max)
    ap.add_argument("--wb-y-min", type=float, default=CFG.wb_y_min)
    ap.add_argument("--wb-y-max", type=float, default=CFG.wb_y_max)
    ap.add_argument("--wb-min-pixels", type=int, default=CFG.wb_min_pixels)

    ap.add_argument("--opencv-wb", action="store_true")
    ap.add_argument(
        "--opencv-wb-mode",
        type=str,
        default=CFG.opencv_wb_mode,
        choices=["simple", "grayworld"],
    )

    ap.add_argument("--no-bg-whiten", action="store_true")
    ap.add_argument("--bg-whiten-target", type=float, default=CFG.bg_whiten_target)
    ap.add_argument("--bg-whiten-max", type=float, default=CFG.bg_whiten_gain_max)

    return ap.parse_args()


def main() -> None:
    t0 = time.perf_counter()
    args = parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    if not input_dir.exists():
        raise SystemExit(f"Input folder does not exist: {input_dir}")

    cfg = Config()
    cfg.jpeg_quality = int(args.quality)
    cfg.raw_use_auto_wb = bool(args.raw_auto_wb)
    cfg.raw_use_camera_wb = not cfg.raw_use_auto_wb

    cfg.sharpness_amount = float(args.sharpness)
    cfg.vibrance = float(args.vibrance)

    cfg.contrast_factor = float(args.contrast)
    cfg.contrast_pivot = float(args.contrast_pivot)
    cfg.pop_strength = float(args.pop)
    cfg.microcontrast_amount = float(args.microcontrast)
    cfg.microcontrast_radius = float(args.micro_radius)

    cfg.shadows_lift = float(args.shadows)
    cfg.shadows_end = float(args.shadows_end)

    cfg.xrite_name_hint = str(args.xrite_hint or "").strip()
    cfg.xrite_skip_export = not bool(args.no_skip_xrite_export)
    cfg.xrite_required = bool(args.xrite_required)

    cfg.wb_strength = float(np.clip(args.wb_strength, 0.0, 1.0))
    cfg.wb_sat_max = float(np.clip(args.wb_sat_max, 0.01, 1.0))
    cfg.wb_y_min = float(np.clip(args.wb_y_min, 0.0, 0.95))
    cfg.wb_y_max = float(np.clip(args.wb_y_max, cfg.wb_y_min + 1e-3, 1.0))
    cfg.wb_min_pixels = int(max(0, args.wb_min_pixels))

    cfg.use_opencv_wb = bool(args.opencv_wb)
    cfg.opencv_wb_mode = str(args.opencv_wb_mode).lower().strip()

    cfg.bg_whiten_enable = not bool(args.no_bg_whiten)
    cfg.bg_whiten_target = float(np.clip(args.bg_whiten_target, 0.0, 1.0))
    cfg.bg_whiten_gain_max = float(max(1.0, args.bg_whiten_max))

    ensure_dir(output_dir)

    anchor_path = find_xrite_file(input_dir, cfg)
    anchor_wb: Optional[np.ndarray] = None

    if anchor_path is None:
        msg = f"[WARN] No xrite anchor found (hint='{cfg.xrite_name_hint}')."
        if cfg.xrite_required:
            raise SystemExit(msg + " (xrite-required set)")
        print(msg + " Proceeding without anchor WB.")
    else:
        bgr16_anchor = read_image_any_to_bgr16(anchor_path, cfg)
        if bgr16_anchor is None:
            msg = f"[WARN] Could not read xrite anchor: {anchor_path}"
            if cfg.xrite_required:
                raise SystemExit(msg)
            print(msg + " Proceeding without anchor WB.")
        else:
            anchor_wb, wb_dbg = compute_anchor_wb_from_xrite_bgr16(bgr16_anchor, cfg)
            if anchor_wb is None:
                msg = f"[WARN] Xrite read OK but WB estimation failed: {wb_dbg}"
                if cfg.xrite_required:
                    raise SystemExit(msg)
                print(msg + " Proceeding without anchor WB.")
            else:
                print(f"[INFO] Xrite anchor: {anchor_path.name}")
                print(
                    f"[INFO] Anchor WB gains (RGB linear): {wb_dbg.get('wb_gains_rgb_lin')} n={wb_dbg.get('n')}"
                )

    count_in = 0
    count_out = 0

    print(f"Processing images from {input_dir} -> {output_dir}")
    print(
        f"Settings: Pop={cfg.pop_strength} | Contrast={cfg.contrast_factor} (pivot {cfg.contrast_pivot}) | "
        f"Micro={cfg.microcontrast_amount}@{cfg.microcontrast_radius} | "
        f"Vibrance={cfg.vibrance} | Shadows={cfg.shadows_lift} | "
        f"AnchorWB={'on' if anchor_wb is not None else 'off'} | "
        f"BGWhiten={'on' if cfg.bg_whiten_enable else 'off'}"
    )

    for p in iter_images(input_dir):
        count_in += 1

        if cfg.xrite_skip_export and _is_xrite_file(p, cfg):
            print(f"Skip (xrite ref): {p.name}")
            continue

        bgr16 = read_image_any_to_bgr16(p, cfg)
        if bgr16 is None:
            print(f"[WARN] Could not read {p}")
            continue

        out8, dbg = process_bgr16(bgr16, cfg, anchor_wb)

        rel = p.relative_to(input_dir)
        out_path = (output_dir / rel).with_suffix(".jpg")
        write_jpeg(out_path, out8, cfg.jpeg_quality)
        count_out += 1

        exp = dbg.get("exposure_bg_linear", {})
        wb_bg = dbg.get("bg_whiten_failsafe", {})
        c = dbg.get("subject_contrast", {})
        mc = dbg.get("subject_microcontrast", {})

        print(
            f"Saved: {out_path.name} | "
            f"ExpGain={exp.get('gain',0):.2f}({exp.get('ref','?')}) "
            f"BGWhiten={('on' if wb_bg.get('applied') else 'off')} "
            f"SubContrast={('on' if c.get('applied') else 'off')} "
            f"Micro={('on' if mc.get('applied') else 'off')}"
        )

    total = time.perf_counter() - t0
    rate = (count_out / total) if total > 0 else 0.0
    print(
        f"Done. Read {count_in} files, wrote {count_out} JPEGs.\nTotal time: {total:.2f}s ({rate:.2f} img/s)"
    )


if __name__ == "__main__":
    main()
