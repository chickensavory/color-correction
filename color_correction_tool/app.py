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
}
IMG_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
}
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
    exp_target: float = 0.985
    exp_gain_min: float = 1.00
    exp_gain_max: float = 4.50
    exp_highlight_pctl: float = 99.6
    exp_highlight_cap: float = 0.990

    bg_v_min: float = 0.62
    bg_s_max: float = 0.30
    bg_min_pixels: int = 1500

    midtone_target: float = 0.86
    subject_white_cut: float = 0.965
    p_min: float = 1.00
    p_max: float = 2.60
    lift_strength: float = 0.85

    contrast_factor: float = 1.00
    contrast_pivot: float = 0.60
    contrast_scale_min: float = 0.75
    contrast_scale_max: float = 1.45

    vibrance: float = 0.18
    vibrance_max_boost: float = 1.35

    highlight_protect_start: float = 0.88
    highlight_protect_end: float = 0.99
    highlight_protect_strength: float = 0.80

    highlight_roll_start: float = 0.92
    highlight_roll_strength: float = 0.55

    highlight_detint_enable: bool = True
    highlight_detint_start: float = 0.94
    highlight_detint_end: float = 0.995
    highlight_detint_strength: float = 0.65

    final_brightness: float = 1.01


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


def detect_white_product_scene(bgr01: np.ndarray) -> Dict[str, Any]:
    bgr8 = float01_to_bgr8(bgr01)
    hsv = cv2.cvtColor(bgr8, cv2.COLOR_BGR2HSV).astype(np.float32)
    S = hsv[..., 1] / 255.0
    Y = luma01_from_bgr01(bgr01)

    s_med = float(np.median(S))
    y_p95 = float(np.percentile(Y, 95.0))

    is_whiteish = (s_med < 0.10) and (y_p95 > 0.88)
    return {"is_whiteish": bool(is_whiteish), "s_med": s_med, "y_p95": y_p95}


def neutral_pixel_white_balance(
    bgr01: np.ndarray, cfg: Config
) -> Tuple[np.ndarray, Dict[str, Any]]:
    method = cfg.wb_method.lower().strip()
    if method != "neutral":
        return bgr01, {"applied": False, "method": method}

    bgr8 = float01_to_bgr8(bgr01)
    hsv = cv2.cvtColor(bgr8, cv2.COLOR_BGR2HSV).astype(np.float32)
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
            (S <= (cfg.neutral_sat_max * 1.6))
            & (Y >= (cfg.neutral_y_min * 0.7))
            & (Y <= (cfg.neutral_y_max * 1.05))
        )
        n = int(mask.sum())

    if n < 500:
        return bgr01, {"applied": False, "reason": "too_few_neutral_pixels", "n": n}

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
        "mean_bgr": mean_bgr.tolist(),
        "gains_bgr": gains.tolist(),
        "strength": a,
        "mask_sat_max": float(cfg.neutral_sat_max),
        "mask_y_min": float(cfg.neutral_y_min),
        "mask_y_max": float(cfg.neutral_y_max),
    }


def opencv_white_balance_bgr16(
    bgr16: np.ndarray, cfg: Config
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if not cfg.use_opencv_wb or cfg.wb_method.lower().strip() != "opencv":
        return bgr16, {"applied": False, "reason": "disabled_or_not_selected"}

    if not hasattr(cv2, "xphoto"):
        return bgr16, {
            "applied": False,
            "reason": "cv2.xphoto missing (need opencv-contrib-python)",
        }

    bgr8 = (bgr16 >> 8).astype(np.uint8)
    mode = cfg.opencv_wb_mode.lower().strip()
    if mode == "grayworld":
        wb = cv2.xphoto.createGrayworldWB()
        wb.setSaturationThreshold(float(cfg.opencv_wb_saturation_thresh))
    else:
        wb = cv2.xphoto.createSimpleWB()
        if hasattr(wb, "setP"):
            wb.setP(float(cfg.opencv_wb_p))

    try:
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
        return out16, {
            "applied": True,
            "method": "opencv",
            "mode": mode,
            "gains_bgr": gains.tolist(),
            "strength": a,
        }
    except Exception as e:
        return bgr16, {"applied": False, "reason": f"wb_failed: {e}"}


def apply_exposure_gain_brighten_only(
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
    gain_bg = cfg.exp_target / base

    hi = float(np.percentile(Y, cfg.exp_highlight_pctl))
    hi = max(hi, 1e-6)
    gain_hi = cfg.exp_highlight_cap / hi

    gain = float(min(gain_bg, gain_hi))
    gain = float(np.clip(gain, cfg.exp_gain_min, cfg.exp_gain_max))
    gain = max(gain, 1.0)

    out = np.clip(bgr01 * gain, 0.0, 1.0)
    dbg = {
        "gain": gain,
        "ref": ref,
        "bg_pixels": bg_count,
        "base_val": base,
        "gain_bg": float(gain_bg),
        "hi_val": hi,
        "gain_hi": float(gain_hi),
    }
    return out, dbg


def highlight_protect_blend(
    original: np.ndarray, edited: np.ndarray, cfg: Config
) -> Tuple[np.ndarray, Dict[str, Any]]:
    Yo = luma01_from_bgr01(original)
    w = smoothstep(Yo, cfg.highlight_protect_start, cfg.highlight_protect_end)
    strength = float(np.clip(cfg.highlight_protect_strength, 0.0, 1.0))
    w = w * strength
    out = np.clip(edited * (1.0 - w[..., None]) + original * w[..., None], 0.0, 1.0)
    dbg = {"applied": bool(strength > 1e-6), "strength": strength}
    return out, dbg


def adaptive_midtone_lift(
    bgr01: np.ndarray, cfg: Config
) -> Tuple[np.ndarray, Dict[str, Any]]:
    Y = luma01_from_bgr01(bgr01)

    subj_mask = Y < cfg.subject_white_cut
    n = int(subj_mask.sum())
    if n < 2000:
        subj_mask = np.ones(Y.shape, dtype=bool)
        n = int(subj_mask.sum())

    med = float(np.median(Y[subj_mask]))
    tgt = float(cfg.midtone_target)
    tgt = min(max(tgt, 0.05), 0.98)
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
    out = np.clip(bgr01 * (1.0 - a) + lifted * a, 0.0, 1.0)

    dbg = {
        "applied": True,
        "subject_pixels": n,
        "subject_median": med,
        "target": tgt,
        "p": p,
        "strength": a,
    }
    return out, dbg


def apply_contrast_luma(
    bgr01: np.ndarray, cfg: Config
) -> Tuple[np.ndarray, Dict[str, Any]]:
    cf = float(cfg.contrast_factor)
    if abs(cf - 1.0) < 1e-4:
        return bgr01, {"applied": False, "contrast_factor": 1.0}

    Y = luma01_from_bgr01(bgr01)
    p = float(np.clip(cfg.contrast_pivot, 0.0, 1.0))

    Yc = p + cf * (Y - p)
    Yc = np.clip(Yc, 0.0, 1.0)

    ratio = Yc / np.maximum(Y, 1e-4)
    ratio = np.clip(ratio, cfg.contrast_scale_min, cfg.contrast_scale_max).astype(
        np.float32
    )
    out = np.clip(bgr01 * ratio[..., None], 0.0, 1.0)
    return out, {"applied": True, "contrast_factor": cf, "pivot": p}


def apply_vibrance(bgr01: np.ndarray, cfg: Config) -> Tuple[np.ndarray, Dict[str, Any]]:
    v = float(np.clip(cfg.vibrance, 0.0, 2.0))
    if v <= 1e-6:
        return bgr01, {"applied": False, "vibrance": 0.0}

    bgr8 = float01_to_bgr8(bgr01)
    hsv = cv2.cvtColor(bgr8, cv2.COLOR_BGR2HSV).astype(np.float32)

    S = hsv[..., 1] / 255.0
    boost = 1.0 + v * (1.0 - S)
    boost = np.clip(boost, 1.0, float(cfg.vibrance_max_boost))

    hsv[..., 1] = np.clip(hsv[..., 1] * boost, 0.0, 255.0)
    hsv8 = np.clip(hsv + 0.5, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(hsv8, cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

    return np.clip(out, 0.0, 1.0), {
        "applied": True,
        "vibrance": v,
        "max_boost": float(cfg.vibrance_max_boost),
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

    return out, {"applied": True, "start": start, "strength": strength}


def highlight_detint(
    bgr01: np.ndarray, cfg: Config
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if not cfg.highlight_detint_enable:
        return bgr01, {"applied": False, "reason": "disabled"}

    Y = luma01_from_bgr01(bgr01)
    w = smoothstep(Y, cfg.highlight_detint_start, cfg.highlight_detint_end)
    a = float(np.clip(cfg.highlight_detint_strength, 0.0, 1.0))
    w = w * a
    if float(w.max()) <= 1e-6:
        return bgr01, {"applied": False, "reason": "no_highlights_in_range"}

    m = np.max(bgr01, axis=2, keepdims=True)
    out = np.clip(bgr01 * (1.0 - w[..., None]) + m * w[..., None], 0.0, 1.0)
    return out, {
        "applied": True,
        "strength": a,
        "start": cfg.highlight_detint_start,
        "end": cfg.highlight_detint_end,
    }


def process_bgr16(bgr16: np.ndarray, cfg: Config) -> Tuple[np.ndarray, Dict[str, Any]]:
    dbg: Dict[str, Any] = {}

    bgr16_2, wb16_dbg = opencv_white_balance_bgr16(bgr16, cfg)
    dbg["wb_opencv"] = wb16_dbg

    bgr01 = bgr16_to_float01(bgr16_2)

    bgr01, wb_dbg = neutral_pixel_white_balance(bgr01, cfg)
    dbg["wb"] = wb_dbg

    scene = detect_white_product_scene(bgr01)
    dbg["scene"] = scene

    pre_exp = bgr01
    bgr01, exp_dbg = apply_exposure_gain_brighten_only(bgr01, cfg)
    dbg["exposure"] = exp_dbg

    bgr01, hp_dbg = highlight_protect_blend(pre_exp, bgr01, cfg)
    dbg["highlight_protect"] = hp_dbg

    if scene["is_whiteish"]:
        saved_strength = cfg.lift_strength
        cfg.lift_strength = float(np.clip(cfg.lift_strength * 0.75, 0.0, 1.0))
        bgr01, lift_dbg = adaptive_midtone_lift(bgr01, cfg)
        cfg.lift_strength = saved_strength
        lift_dbg["whiteish_tempered"] = True
    else:
        bgr01, lift_dbg = adaptive_midtone_lift(bgr01, cfg)
        lift_dbg["whiteish_tempered"] = False
    dbg["midtone_lift"] = lift_dbg

    bgr01, c_dbg = apply_contrast_luma(bgr01, cfg)
    dbg["contrast"] = c_dbg

    bgr01, v_dbg = apply_vibrance(bgr01, cfg)
    dbg["vibrance"] = v_dbg

    bgr01, hr_dbg = highlight_rolloff(bgr01, cfg)
    dbg["highlight_rolloff"] = hr_dbg

    bgr01, hd_dbg = highlight_detint(bgr01, cfg)
    dbg["highlight_detint"] = hd_dbg

    fb = float(cfg.final_brightness)
    if abs(fb - 1.0) > 1e-4:
        bgr01 = np.clip(bgr01 * fb, 0.0, 1.0)
        dbg["final_brightness"] = fb

    return float01_to_bgr8(bgr01), dbg


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="input", help="Input folder")
    ap.add_argument("--output", type=str, default="output", help="Output folder")
    ap.add_argument(
        "--quality", type=int, default=CFG.jpeg_quality, help="JPEG quality (1-100)"
    )

    ap.add_argument(
        "--raw-auto-wb",
        action="store_true",
        help="Use rawpy auto WB instead of camera WB",
    )

    ap.add_argument(
        "--wb-method",
        type=str,
        default=CFG.wb_method,
        choices=["none", "neutral", "opencv"],
        help="WB method. Recommended for RAW: neutral (or none).",
    )
    ap.add_argument(
        "--wb-strength",
        type=float,
        default=CFG.wb_strength,
        help="WB blend strength (0..1)",
    )
    ap.add_argument(
        "--opencv-wb-mode",
        type=str,
        default=CFG.opencv_wb_mode,
        choices=["simple", "grayworld"],
    )
    ap.add_argument(
        "--use-opencv-wb",
        action="store_true",
        help="Enable OpenCV WB step (mostly for JPEGs)",
    )

    ap.add_argument("--midtone-target", type=float, default=CFG.midtone_target)
    ap.add_argument("--lift-strength", type=float, default=CFG.lift_strength)
    ap.add_argument("--final-brightness", type=float, default=CFG.final_brightness)

    ap.add_argument("--contrast", type=float, default=CFG.contrast_factor)
    ap.add_argument("--vibrance", type=float, default=CFG.vibrance)

    ap.add_argument("--hp-strength", type=float, default=CFG.highlight_protect_strength)
    ap.add_argument("--roll-strength", type=float, default=CFG.highlight_roll_strength)

    ap.add_argument(
        "--detint-off", action="store_true", help="Disable highlight de-tint step"
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
    cfg.wb_strength = float(np.clip(args.wb_strength, 0.0, 1.0))

    cfg.use_opencv_wb = bool(args.use_opencv_wb)
    cfg.opencv_wb_mode = str(args.opencv_wb_mode)

    cfg.midtone_target = float(args.midtone_target)
    cfg.lift_strength = float(args.lift_strength)
    cfg.final_brightness = float(args.final_brightness)

    cfg.contrast_factor = float(args.contrast)
    cfg.vibrance = float(args.vibrance)

    cfg.highlight_protect_strength = float(np.clip(args.hp_strength, 0.0, 1.0))
    cfg.highlight_roll_strength = float(np.clip(args.roll_strength, 0.0, 1.0))

    cfg.highlight_detint_enable = not bool(args.detint_off)

    ensure_dir(output_dir)

    count_in = 0
    count_out = 0

    for p in iter_images(input_dir):
        count_in += 1
        bgr16 = read_image_any_to_bgr16(p, cfg)
        if bgr16 is None:
            print(f"[WARN] Could not read {p}")
            continue

        out8, dbg = process_bgr16(bgr16, cfg)

        rel = p.relative_to(input_dir)
        out_path = (output_dir / rel).with_suffix(".jpg")
        write_jpeg(out_path, out8, cfg.jpeg_quality)
        count_out += 1

        ex = dbg.get("exposure", {})
        ml = dbg.get("midtone_lift", {})
        wb = dbg.get("wb", {})
        sc = dbg.get("scene", {})
        hd = dbg.get("highlight_detint", {})
        print(
            f"Saved: {out_path} | "
            f"WB={wb.get('applied', False)}({wb.get('method','')}) "
            f"gain={ex.get('gain', 1.0):.3f} ref={ex.get('ref','')} | "
            f"med={ml.get('subject_median', 0):.3f} p={ml.get('p', 1.0):.3f} | "
            f"whiteish={sc.get('is_whiteish', False)} s_med={sc.get('s_med', 0):.3f} "
            f"detint={hd.get('applied', False)} "
            f"vib={cfg.vibrance:.3f} con={cfg.contrast_factor:.3f}"
        )

    total = time.perf_counter() - t0
    rate = (count_out / total) if total > 0 else 0.0
    print(
        f"Done. Read {count_in} files, wrote {count_out} JPEGs to {output_dir}. "
        f"Total={total:.2f}s ({rate:.2f} img/s, {rate*60:.1f} img/min)"
    )


if __name__ == "__main__":
    main()
