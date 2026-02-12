from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict, Any

import numpy as np
from PIL import Image, ImageOps
import cv2
import time

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
    ".nef",
    ".rw2",
}
IMG_EXTS = {".jpg", ".jpeg", ".png"}
ALL_EXTS = RAW_EXTS | IMG_EXTS


@dataclass
class Config:
    jpeg_quality: int = 95

    raw_use_camera_wb: bool = True
    raw_use_auto_wb: bool = True
    raw_no_auto_bright: bool = True
    raw_output_color_srgb: bool = True
    raw_highlight_mode: int = 1

    wb_method: str = "neutral"
    wb_strength: float = 0.60
    wb_gain_clip_min: float = 0.70
    wb_gain_clip_max: float = 1.45

    neutral_sat_max: float = 0.18
    neutral_y_min: float = 0.12
    neutral_y_max: float = 0.92
    neutral_min_pixels: int = 2500

    use_opencv_wb: bool = False
    opencv_wb_mode: str = "simple"
    opencv_wb_p: float = 0.5
    opencv_wb_saturation_thresh: float = 0.98

    exp_pctl: float = 90.0
    exp_target: float = 0.990
    exp_gain_min: float = 1.20
    exp_gain_max: float = 6.00
    exp_highlight_pctl: float = 99.6
    exp_highlight_cap: float = 0.998

    bg_v_min: float = 0.62
    bg_s_max: float = 0.30
    bg_min_pixels: int = 1500

    shadows_lift: float = 0.32
    shadows_start: float = 0.00
    shadows_end: float = 0.52
    shadows_max_boost: float = 2.10

    midtone_target: float = 0.90
    subject_white_cut: float = 0.965
    p_min: float = 1.00
    p_max: float = 3.00
    lift_strength: float = 0.85

    contrast_factor: float = 1.10
    contrast_scale_min: float = 0.75
    contrast_scale_max: float = 1.45

    vibrance: float = 0.25
    vibrance_max_boost: float = 1.35
    vibrance_highlight_start: float = 0.88
    vibrance_highlight_end: float = 0.99
    vibrance_neutral_chroma_max: float = 10.0
    vibrance_neutral_soft: float = 22.0

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

    ref_name_hint: str = "xrite"
    ref_skip_export: bool = True
    ref_neutral_tol: float = 0.035
    ref_adapt_strength: float = 0.35
    ref_max_wb_delta: float = 0.20

    ref_vibrance_adapt: bool = True
    ref_vibrance_min_scale: float = 0.90
    ref_vibrance_max_scale: float = 1.10


CFG = Config()


def iter_images(folder: Path) -> Iterable[Path]:
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in ALL_EXTS:
            yield p


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def bgr16_to_float01(bgr16: np.ndarray) -> np.ndarray:
    return np.clip(bgr16.astype(np.float32) / 65535.0, 0.0, 1.0)


def float01_to_bgr8(bgr01: np.ndarray) -> np.ndarray:
    return np.clip(bgr01 * 255.0 + 0.5, 0, 255).astype(np.uint8)


def luma01_from_bgr01(bgr01: np.ndarray) -> np.ndarray:
    return 0.2126 * bgr01[..., 2] + 0.7152 * bgr01[..., 1] + 0.0722 * bgr01[..., 0]


def smoothstep(x: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def robust_chroma_from_bgr01(
    bgr01: np.ndarray,
    neutral_chroma_cut: float = 6.0,
    pctl: float = 75.0,
    min_pixels: int = 500,
) -> float:
    bgr8 = float01_to_bgr8(bgr01)
    ycrcb = cv2.cvtColor(bgr8, cv2.COLOR_BGR2YCrCb).astype(np.float32)

    Cr = ycrcb[..., 1]
    Cb = ycrcb[..., 2]
    dCr = Cr - 128.0
    dCb = Cb - 128.0

    chroma = np.sqrt(dCr * dCr + dCb * dCb).reshape(-1)

    chroma = chroma[chroma >= float(neutral_chroma_cut)]

    if chroma.size < int(min_pixels):
        chroma_all = np.sqrt(dCr * dCr + dCb * dCb).reshape(-1)
        return float(np.median(chroma_all))

    return float(np.percentile(chroma, float(pctl)))


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
                kwargs["use_camera_wb"] = bool(cfg.raw_use_camera_wb)

            if cfg.raw_output_color_srgb:
                kwargs["output_color"] = rawpy.ColorSpace.sRGB

            rgb16 = raw.postprocess(**kwargs)

        return cv2.cvtColor(rgb16, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"[WARN] RAW read failed {path}: {e}")
        return None


def wb_gains_from_bright_neutrals(
    bgr01: np.ndarray, cfg: Config
) -> Optional[np.ndarray]:
    bgr8 = float01_to_bgr8(bgr01)
    hsv = cv2.cvtColor(bgr8, cv2.COLOR_BGR2HSV).astype(np.float32)
    v = hsv[..., 2] / 255.0
    s = hsv[..., 1] / 255.0

    low_sat = s <= min(0.35, cfg.bg_s_max + 0.05)
    if int(low_sat.sum()) < 500:
        low_sat = s <= 0.45

    v_thr = (
        float(np.percentile(v[low_sat], 95.0))
        if int(low_sat.sum())
        else float(np.percentile(v, 95.0))
    )
    mask = low_sat & (v >= v_thr)

    return wb_gains_from_mask_bgr(bgr01, mask, cfg)


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


def bg_mask_from_bgr01(
    bgr01: np.ndarray, cfg: Config, allow_clipped: bool = False
) -> np.ndarray:
    bgr8 = float01_to_bgr8(bgr01)
    hsv = cv2.cvtColor(bgr8, cv2.COLOR_BGR2HSV).astype(np.float32)
    v = hsv[..., 2] / 255.0
    s = hsv[..., 1] / 255.0
    clip_ok = 1.0 if allow_clipped else 0.995
    mask = (
        (v >= cfg.bg_v_min) & (s <= cfg.bg_s_max) & (np.max(bgr01, axis=2) <= clip_ok)
    )
    return mask


def wb_gains_from_mask_bgr(
    bgr01: np.ndarray, mask: np.ndarray, cfg: Config
) -> Optional[np.ndarray]:
    n = int(mask.sum())
    if n < max(200, cfg.bg_min_pixels // 4):
        return None
    samp = bgr01[mask].reshape(-1, 3).astype(np.float32)
    med = np.median(samp, axis=0) + 1e-6
    g_ref = float(med[1])
    gains = np.array([g_ref / med[0], 1.0, g_ref / med[2]], dtype=np.float32)
    gains = np.clip(gains, cfg.wb_gain_clip_min, cfg.wb_gain_clip_max)
    return gains


def apply_bgr_gains(
    bgr01: np.ndarray, gains_bgr: np.ndarray, strength: float = 1.0
) -> np.ndarray:
    a = float(np.clip(strength, 0.0, 1.0))
    g = 1.0 + a * (gains_bgr - 1.0)
    return np.clip(bgr01 * g[None, None, :], 0.0, 1.0)


def median_subject_chroma(bgr01: np.ndarray, bg_mask: np.ndarray) -> float:
    subj = ~bg_mask
    if int(subj.sum()) < 500:
        subj = np.ones(bg_mask.shape, dtype=bool)

    bgr8 = float01_to_bgr8(bgr01)
    ycrcb = cv2.cvtColor(bgr8, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    Cr = ycrcb[..., 1][subj]
    Cb = ycrcb[..., 2][subj]
    dCr = Cr - 128.0
    dCb = Cb - 128.0
    chroma = np.sqrt(dCr * dCr + dCb * dCb)
    return float(np.median(chroma))


def detect_white_product_scene(bgr01: np.ndarray) -> Dict[str, Any]:
    Y = luma01_from_bgr01(bgr01)
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


def neutral_pixel_white_balance(
    bgr01: np.ndarray, cfg: Config
) -> Tuple[np.ndarray, Dict[str, Any]]:
    method = cfg.wb_method.lower().strip()
    if method != "neutral":
        return bgr01, {"applied": False, "method": method}

    bgr8 = float01_to_bgr8(bgr01)
    hsv = cv2.cvtColor(bgr8, cv2.COLOR_BGR2HSV).astype(np.float32)
    H = hsv[..., 0]
    S = hsv[..., 1] / 255.0
    Y = luma01_from_bgr01(bgr01)

    mask = (
        (S <= cfg.neutral_sat_max)
        & (Y >= cfg.neutral_y_min)
        & (Y <= cfg.neutral_y_max)
        & (np.max(bgr01, axis=2) <= 0.995)
    )

    n = int(mask.sum())
    if n < cfg.neutral_min_pixels:
        mask = (
            (S <= (cfg.neutral_sat_max * 1.5))
            & (Y >= (cfg.neutral_y_min * 0.7))
            & (Y <= (cfg.neutral_y_max * 1.05))
        )
        n = int(mask.sum())

    if n < 500:
        return bgr01, {"applied": False, "reason": "too_few_neutral_pixels", "n": n}

    samp_h = H[mask]
    if len(samp_h) > 0:
        rads = np.deg2rad(samp_h * 2.0)
        R = np.sqrt(np.sum(np.sin(rads)) ** 2 + np.sum(np.cos(rads)) ** 2) / len(samp_h)
        hue_variance = 1.0 - R
        if hue_variance < 0.4:
            return bgr01, {
                "applied": False,
                "reason": "dominant_hue_detected_in_neutral",
                "hue_var": float(hue_variance),
            }

    samp = bgr01[mask].reshape(-1, 3).astype(np.float32)
    mean_bgr = samp.mean(axis=0) + 1e-6
    g_ref = float(mean_bgr[1])
    gains = np.array([g_ref / mean_bgr[0], 1.0, g_ref / mean_bgr[2]], dtype=np.float32)
    gains = np.clip(gains, cfg.wb_gain_clip_min, cfg.wb_gain_clip_max)

    a = float(np.clip(cfg.wb_strength, 0.0, 1.0))
    gains = 1.0 + a * (gains - 1.0)

    out = np.clip(bgr01 * gains[None, None, :], 0.0, 1.0)
    return out, {
        "applied": True,
        "method": "neutral",
        "n": n,
        "gains_bgr": gains.tolist(),
        "strength": a,
    }


def opencv_white_balance_bgr16(
    bgr16: np.ndarray, cfg: Config
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if not cfg.use_opencv_wb or cfg.wb_method.lower().strip() != "opencv":
        return bgr16, {"applied": False, "reason": "disabled_or_not_selected"}
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

        a = float(np.clip(cfg.wb_strength, 0.0, 1.0))
        gains = 1.0 + a * (gains - 1.0)

        out16 = np.clip(
            bgr16.astype(np.float32) * gains[None, None, :], 0.0, 65535.0
        ).astype(np.uint16)
        return out16, {"applied": True, "method": "opencv", "gains_bgr": gains.tolist()}
    except Exception as e:
        return bgr16, {"applied": False, "reason": f"wb_failed: {e}"}


def apply_exposure_gain_on_luma_brighten_only(
    bgr01: np.ndarray, cfg: Config
) -> Tuple[np.ndarray, Dict[str, Any]]:
    bgr8 = float01_to_bgr8(bgr01)
    hsv = cv2.cvtColor(bgr8, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2].astype(np.float32) / 255.0
    s = hsv[..., 1].astype(np.float32) / 255.0
    bg_mask = (v >= cfg.bg_v_min) & (s <= cfg.bg_s_max)

    Y = luma01_from_bgr01(bgr01)
    bg_count = int(bg_mask.sum())

    if bg_count >= cfg.bg_min_pixels:
        base = float(np.percentile(Y[bg_mask], cfg.exp_pctl))
        ref = "bg"
    else:
        base = float(np.percentile(Y, cfg.exp_pctl))
        ref = "all"

    base = max(base, 1e-6)
    gain_target = cfg.exp_target / base

    hi = float(np.percentile(Y, cfg.exp_highlight_pctl))
    hi = max(hi, 1e-6)
    gain_hi = cfg.exp_highlight_cap / hi

    gain = float(min(gain_target, gain_hi))
    gain = float(np.clip(gain, cfg.exp_gain_min, cfg.exp_gain_max))
    gain = max(gain, 1.0)

    Y2 = np.clip(Y * gain, 0.0, 1.0)
    ratio = Y2 / np.maximum(Y, 1e-4)
    ratio = np.clip(ratio, 0.75, 8.0).astype(np.float32)

    out = np.clip(bgr01 * ratio[..., None], 0.0, 1.0)
    return out, {"gain": gain, "ref": ref, "base_val": base}


def apply_shadows_lift(
    bgr01: np.ndarray, cfg: Config
) -> Tuple[np.ndarray, Dict[str, Any]]:
    strength = float(np.clip(cfg.shadows_lift, 0.0, 1.0))
    if strength <= 1e-6:
        return bgr01, {"applied": False}

    Y = luma01_from_bgr01(bgr01)
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


def apply_contrast_on_luma(
    bgr01: np.ndarray, cfg: Config
) -> Tuple[np.ndarray, Dict[str, Any]]:
    cf = float(cfg.contrast_factor)
    if abs(cf - 1.0) < 1e-4:
        return bgr01, {"applied": False}

    Y = luma01_from_bgr01(bgr01)
    pivot = float(np.median(Y))
    pivot = float(np.clip(pivot, 0.2, 0.8))

    Yc = pivot + cf * (Y - pivot)
    Yc = np.clip(Yc, 0.0, 1.0)

    ratio = Yc / np.maximum(Y, 1e-4)
    ratio = np.clip(ratio, cfg.contrast_scale_min, cfg.contrast_scale_max).astype(
        np.float32
    )

    out = np.clip(bgr01 * ratio[..., None], 0.0, 1.0)
    return out, {"applied": True, "contrast_factor": cf, "pivot": pivot}


def highlight_protect_blend(
    original: np.ndarray, edited: np.ndarray, cfg: Config
) -> Tuple[np.ndarray, Dict[str, Any]]:
    Yo = luma01_from_bgr01(original)
    w = smoothstep(Yo, cfg.highlight_protect_start, cfg.highlight_protect_end)
    strength = float(np.clip(cfg.highlight_protect_strength, 0.0, 1.0))
    w = w * strength
    out = np.clip(edited * (1.0 - w[..., None]) + original * w[..., None], 0.0, 1.0)
    return out, {"applied": True}


def adaptive_midtone_lift(
    bgr01: np.ndarray, cfg: Config, scene_info: Dict[str, Any]
) -> Tuple[np.ndarray, Dict[str, Any]]:
    Y = luma01_from_bgr01(bgr01)

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


def highlight_rolloff(
    bgr01: np.ndarray, cfg: Config
) -> Tuple[np.ndarray, Dict[str, Any]]:
    strength = float(np.clip(cfg.highlight_roll_strength, 0.0, 1.0))
    start = float(np.clip(cfg.highlight_roll_start, 0.0, 0.999))
    if strength <= 1e-6:
        return bgr01, {"applied": False}

    Y = luma01_from_bgr01(bgr01)
    t = np.clip((Y - start) / max(1.0 - start, 1e-6), 0.0, 1.0)
    Y2 = Y - strength * (t * t) * (Y - start)
    Y2 = np.clip(Y2, 0.0, 1.0)

    ratio = Y2 / np.maximum(Y, 1e-4)
    ratio = np.clip(ratio, 0.7, 1.0).astype(np.float32)
    out = np.clip(bgr01 * ratio[..., None], 0.0, 1.0)
    return out, {"applied": True}


def highlight_detint(
    bgr01: np.ndarray, cfg: Config
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if not cfg.highlight_detint_enable:
        return bgr01, {"applied": False}

    Y = luma01_from_bgr01(bgr01)
    w = smoothstep(Y, cfg.highlight_detint_start, cfg.highlight_detint_end)
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
    return np.clip(out, 0.0, 1.0), {"applied": True}


def apply_unsharp_mask(bgr8: np.ndarray, cfg: Config) -> np.ndarray:
    if cfg.sharpness_amount <= 0:
        return bgr8
    blurred = cv2.GaussianBlur(bgr8, (0, 0), cfg.sharpness_radius)
    sharpened = cv2.addWeighted(
        bgr8, 1.0 + cfg.sharpness_amount, blurred, -cfg.sharpness_amount, 0
    )
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def process_bgr16(
    bgr16: np.ndarray,
    cfg: Config,
    ref_wb_gains: Optional[np.ndarray],
    ref_chroma: Optional[float],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    dbg: Dict[str, Any] = {}
    if ref_wb_gains is not None:
        dbg["wb_opencv"] = {"applied": False, "reason": "skipped_due_to_ref_lock"}

        bgr01 = bgr16_to_float01(bgr16)

        bgr01 = apply_bgr_gains(bgr01, ref_wb_gains, strength=1.0)
        dbg["wb_ref"] = {
            "applied": True,
            "gains_bgr": ref_wb_gains.tolist(),
            "locked": True,
        }

        dbg["wb_bg_off"] = None
        dbg["wb_bg_mean_bgr"] = None
        dbg["wb_adapt"] = {"applied": False, "reason": "locked_to_ref"}
        dbg["wb_fallback"] = {"applied": False, "reason": "locked_to_ref"}

    else:
        bgr16_2, wb16_dbg = opencv_white_balance_bgr16(bgr16, cfg)
        dbg["wb_opencv"] = wb16_dbg

        bgr01 = bgr16_to_float01(bgr16_2)
        dbg["wb_ref"] = {"applied": False, "locked": False}

        bg = bg_mask_from_bgr01(bgr01, cfg)
        cur_gains = wb_gains_from_mask_bgr(bgr01, bg, cfg)

        did_adapt = False

        if not did_adapt:
            if cfg.wb_method == "opencv":
                bgr16_tmp = (bgr01 * 65535.0 + 0.5).astype(np.uint16)
                bgr16_tmp, wb_dbg = opencv_white_balance_bgr16(bgr16_tmp, cfg)
                bgr01 = bgr16_to_float01(bgr16_tmp)
            elif cfg.wb_method == "neutral":
                bgr01, wb_dbg = neutral_pixel_white_balance(bgr01, cfg)
            else:
                wb_dbg = {"applied": False, "reason": "wb_method_none"}

            dbg["wb_fallback"] = wb_dbg

    scene = detect_white_product_scene(bgr01)
    dbg["scene"] = scene

    pre_exp = bgr01
    bgr01, exp_dbg = apply_exposure_gain_on_luma_brighten_only(bgr01, cfg)
    dbg["exposure_luma"] = exp_dbg

    bgr01, hp_dbg = highlight_protect_blend(pre_exp, bgr01, cfg)
    dbg["highlight_protect"] = hp_dbg

    bgr01, sh_dbg = apply_shadows_lift(bgr01, cfg)
    dbg["shadows_lift"] = sh_dbg

    bgr01, lift_dbg = adaptive_midtone_lift(bgr01, cfg, scene)
    dbg["midtone_lift"] = lift_dbg

    bgr01, c_dbg = apply_contrast_on_luma(bgr01, cfg)
    dbg["contrast_luma"] = c_dbg

    eff_vibrance = cfg.vibrance
    if cfg.ref_vibrance_adapt and (ref_chroma is not None):
        cur_ch = robust_chroma_from_bgr01(bgr01, neutral_chroma_cut=8.0, pctl=60.0)
        if cur_ch > 1e-3:
            scale = (ref_chroma / cur_ch) ** 0.5
            scale = float(
                np.clip(scale, cfg.ref_vibrance_min_scale, cfg.ref_vibrance_max_scale)
            )
            eff_vibrance = float(cfg.vibrance * scale)
            dbg["vibrance_ref"] = {
                "cur_chroma": cur_ch,
                "ref_chroma": ref_chroma,
                "scale": scale,
                "eff": eff_vibrance,
            }

    old_v = cfg.vibrance
    cfg.vibrance = eff_vibrance
    bgr01, v_dbg = apply_vibrance_ycrcb_protected(bgr01, cfg)
    cfg.vibrance = old_v
    dbg["vibrance_ycrcb"] = v_dbg

    bgr01, hr_dbg = highlight_rolloff(bgr01, cfg)
    bgr01, hd_dbg = highlight_detint(bgr01, cfg)
    dbg["highlight_rolloff"] = hr_dbg
    dbg["highlight_detint"] = hd_dbg

    fb = float(cfg.final_brightness)
    if abs(fb - 1.0) > 1e-4:
        bgr01 = np.clip(bgr01 * fb, 0.0, 1.0)

    bgr8 = float01_to_bgr8(bgr01)
    bgr8 = apply_unsharp_mask(bgr8, cfg)
    return bgr8, dbg


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="input", help="Input folder")
    ap.add_argument("--output", type=str, default="output", help="Output folder")
    ap.add_argument(
        "--quality", type=int, default=CFG.jpeg_quality, help="JPEG quality"
    )

    ap.add_argument("--raw-auto-wb", action="store_true", help="Use rawpy auto WB")
    ap.add_argument("--sharpness", type=float, default=CFG.sharpness_amount)
    ap.add_argument(
        "--wb-method",
        type=str,
        default=CFG.wb_method,
        choices=["none", "neutral", "opencv"],
    )
    ap.add_argument("--contrast", type=float, default=CFG.contrast_factor)
    ap.add_argument("--vibrance", type=float, default=CFG.vibrance)
    ap.add_argument(
        "--shadows", type=float, default=CFG.shadows_lift, help="Shadows lift (0..1)"
    )
    ap.add_argument(
        "--shadows-end",
        type=float,
        default=CFG.shadows_end,
        help="Shadows fade-out luma (0..1)",
    )

    ap.add_argument(
        "--ref-hint",
        type=str,
        default=CFG.ref_name_hint,
        help="Filename hint for reference (e.g. xrite)",
    )
    ap.add_argument("--ref-neutral-tol", type=float, default=CFG.ref_neutral_tol)
    ap.add_argument("--ref-adapt", type=float, default=CFG.ref_adapt_strength)
    ap.add_argument(
        "--ref-skip",
        action="store_true",
        help="Skip exporting reference image (overrides config)",
    )
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

    cfg.wb_method = str(args.wb_method).lower().strip()
    cfg.contrast_factor = float(args.contrast)
    cfg.vibrance = float(args.vibrance)
    cfg.sharpness_amount = float(args.sharpness)
    cfg.shadows_lift = float(args.shadows)
    cfg.shadows_end = float(args.shadows_end)

    cfg.ref_name_hint = str(args.ref_hint).strip()
    cfg.ref_neutral_tol = float(args.ref_neutral_tol)
    cfg.ref_adapt_strength = float(np.clip(args.ref_adapt, 0.0, 1.0))
    if args.ref_skip:
        cfg.ref_skip_export = True

    ensure_dir(output_dir)

    files = sorted(iter_images(input_dir), key=lambda p: str(p).lower())
    if not files:
        raise SystemExit("No images found in input folder.")

    ref_path: Optional[Path] = None
    hint = cfg.ref_name_hint.lower().strip()
    if hint:
        for p in files:
            if hint in p.name.lower():
                ref_path = p
                break
    if ref_path is None:
        ref_path = files[0]

    ref_wb_gains: Optional[np.ndarray] = None
    ref_chroma: Optional[float] = None

    ref_bgr16 = read_image_any_to_bgr16(ref_path, cfg)
    if ref_bgr16 is not None:
        ref01 = bgr16_to_float01(ref_bgr16)

        ref_wb_gains = wb_gains_from_bright_neutrals(ref01, cfg)

        ref01_wb = (
            apply_bgr_gains(ref01, ref_wb_gains, 1.0)
            if ref_wb_gains is not None
            else ref01
        )

        ref_chroma = robust_chroma_from_bgr01(
            ref01_wb, neutral_chroma_cut=8.0, pctl=60.0
        )

    print(f"[INFO] Reference file: {ref_path.name if ref_path else 'NONE'}")
    print(
        f"[INFO] Reference WB gains (BGR): {ref_wb_gains.tolist() if ref_wb_gains is not None else None}"
    )
    print(
        f"[INFO] Reference subject chroma: {ref_chroma if ref_chroma is not None else None}"
    )
    print(
        "[INFO] ref_wb_gains:", None if ref_wb_gains is None else ref_wb_gains.tolist()
    )
    print("[INFO] ref_neutral pixels:", "NA" if ref_wb_gains is None else "ok")
    print("[INFO] ref_chroma_robust:", ref_chroma)

    count_in = 0
    count_out = 0

    print(f"Processing images from {input_dir} -> {output_dir}")
    print(
        f"Settings: Sharpness={cfg.sharpness_amount} | Contrast={cfg.contrast_factor} | "
        f"WB={cfg.wb_method} | Vibrance={cfg.vibrance} | Shadows={cfg.shadows_lift} | "
        f"RefHint={cfg.ref_name_hint}"
    )

    for p in files:
        count_in += 1

        if (
            cfg.ref_skip_export
            and ref_path is not None
            and p.resolve() == ref_path.resolve()
        ):
            print(f"Skip (reference): {p.name}")
            continue

        bgr16 = read_image_any_to_bgr16(p, cfg)
        if bgr16 is None:
            print(f"[WARN] Could not read {p}")
            continue

        out8, dbg = process_bgr16(bgr16, cfg, ref_wb_gains, ref_chroma)

        rel = p.relative_to(input_dir)
        out_path = (output_dir / rel).with_suffix(".jpg")
        write_jpeg(out_path, out8, cfg.jpeg_quality)
        count_out += 1

        c = dbg.get("contrast_luma", {})
        sh = dbg.get("shadows_lift", {})
        sc = dbg.get("scene", {})
        off = dbg.get("wb_bg_off", None)
        vib = dbg.get("vibrance_ref", {})

        print(
            f"Saved: {out_path.name} | "
            f"Piv={c.get('pivot',0):.2f} "
            f"Shad={('on' if sh.get('applied') else 'off')} "
            f"IsDark={sc.get('is_dark_subject', False)} "
            f"BGoff={(off if off is not None else 'NA')} "
            f"Veff={vib.get('eff', cfg.vibrance):.3f}"
        )

    total = time.perf_counter() - t0
    rate = (count_out / total) if total > 0 else 0.0
    print(
        f"Done. Read {count_in} files, wrote {count_out} JPEGs.\nTotal time: {total:.2f}s ({rate:.2f} img/s)"
    )


if __name__ == "__main__":
    main()
