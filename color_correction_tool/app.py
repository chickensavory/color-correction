from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import json
import getpass
import urllib.request
import urllib.error
import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict, Any
from datetime import date
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import time

from color_correction_tool.xmp import write_processed_tags

import numpy as np
from PIL import Image, ImageOps
import cv2

try:
    import rawpy

    HAS_RAW = True
except Exception:
    HAS_RAW = False

TRACK_ENDPOINT = os.environ.get(
    "COLOR_CORRECTION_ENDPOINT", "https://sofiakris-colorcorrection.hf.space/track"
)  # TODO THIS MIGHT BE WRONG

KEYCHAIN_SERVICE = "color-correction-tracker"
KEYCHAIN_ACCOUNT_HF = "hf_token"
KEYCHAIN_ACCOUNT_TRACKER = "tracker_token"


def _keychain_get(account: str) -> Optional[str]:
    try:
        res = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-s",
                KEYCHAIN_SERVICE,
                "-a",
                account,
                "-w",
            ],
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            return None
        return (res.stdout or "").strip() or None
    except Exception:
        return None


def _keychain_set(account: str, secret: str) -> bool:
    try:
        subprocess.run(
            [
                "security",
                "delete-generic-password",
                "-s",
                KEYCHAIN_SERVICE,
                "-a",
                account,
            ],
            capture_output=True,
            text=True,
        )
        res = subprocess.run(
            [
                "security",
                "add-generic-password",
                "-s",
                KEYCHAIN_SERVICE,
                "-a",
                account,
                "-w",
                secret,
            ],
            capture_output=True,
            text=True,
        )
        return res.returncode == 0
    except Exception:
        return False


def _ensure_tokens_or_prompt_once() -> Tuple[Optional[str], Optional[str]]:
    hf_token = _keychain_get(KEYCHAIN_ACCOUNT_HF)
    tracker_token = _keychain_get(KEYCHAIN_ACCOUNT_TRACKER)

    env_hf = os.environ.get("HF_ACCESS_TOKEN")
    env_tracker = os.environ.get("TRACKER_TOKEN")

    if not hf_token and env_hf:
        hf_token = env_hf.strip()
        _keychain_set(KEYCHAIN_ACCOUNT_HF, hf_token)

    if not tracker_token and env_tracker:
        tracker_token = env_tracker.strip()
        _keychain_set(KEYCHAIN_ACCOUNT_TRACKER, tracker_token)

    if not hf_token:
        print("[TRACK] Hugging Face access token not found in Keychain.")
        hf_token = getpass.getpass("Enter Hugging Face access token (hf...): ").strip()
        if hf_token:
            _keychain_set(KEYCHAIN_ACCOUNT_HF, hf_token)

    if not tracker_token:
        print("[TRACK] Tracker token not found in Keychain.")
        tracker_token = getpass.getpass(
            "Enter tracker token (hex / random string): "
        ).strip()
        if tracker_token:
            _keychain_set(KEYCHAIN_ACCOUNT_TRACKER, tracker_token)

    if not hf_token or not tracker_token:
        print("[TRACK] Tracking disabled: missing token(s).")
        return None, None

    return hf_token, tracker_token


def _post_run_counts(
    *,
    hf_token: str,
    tracker_token: str,
    processed: int,
    unprocessed: int,
) -> None:
    payload = json.dumps(
        {"processed": int(processed), "unprocessed": int(unprocessed)}
    ).encode("utf-8")
    req = urllib.request.Request(
        TRACK_ENDPOINT,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {hf_token}",
            "X-Tracker-Token": tracker_token,
            "User-Agent": "color-correction/1.0",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=6) as resp:
            _ = resp.read()
        print(
            f"[TRACK] Sent counts to API: processed={processed}, unprocessed={unprocessed}"
        )
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="ignore")
        except Exception:
            body = ""
        print(f"[TRACK] FAILED ({e.code}): {body[:700].strip()}")
    except Exception as e:
        print(f"[TRACK] FAILED: {type(e).__name__}: {e}")


try:
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)
except Exception:
    pass

RAW_EXTS = {
    ".arw",
    ".dng",
    ".cr2",
    ".cr3",
    ".nef",
    ".crw",
    ".rw2",
}
IMG_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
}
ALL_EXTS = RAW_EXTS | IMG_EXTS

PROCESS_TOOL = "color-correction"


_N = 65536


def _ensure_contiguous(arr: np.ndarray) -> np.ndarray:
    if not arr.flags["C_CONTIGUOUS"]:
        return np.ascontiguousarray(arr)
    return arr


_LUMA_KERNEL_BGR = np.array([[0.0722, 0.7152, 0.2126]], dtype=np.float32)


_M_BGR2YCrCb = np.array(
    [
        [0.114, 0.587, 0.299, 0.0],
        [-0.08131, -0.41869, 0.5, 0.5],
        [0.5, -0.33126, -0.16874, 0.5],
    ],
    dtype=np.float32,
)

_M_YCrCb2BGR = np.array(
    [
        [1.0, 0.0, 1.772, -0.886],
        [1.0, -0.714136, -0.344136, 0.529136],
        [1.0, 1.402, 0.0, -0.701],
    ],
    dtype=np.float32,
)


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1.0 + a)) ** 2.4).astype(
        np.float32, copy=False
    )


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)
    a = 0.055
    return np.where(
        x <= 0.0031308, x * 12.92, (1.0 + a) * (x ** (1.0 / 2.4)) - a
    ).astype(np.float32, copy=False)


def bgr16_to_float01(bgr16: np.ndarray) -> np.ndarray:
    return np.clip(bgr16.astype(np.float32) / 65535.0, 0.0, 1.0).astype(
        np.float32, copy=False
    )


def float01_to_bgr8(bgr01: np.ndarray) -> np.ndarray:
    out = np.empty(bgr01.shape, dtype=np.float32)
    np.multiply(bgr01, 255.0, out=out)
    np.add(out, 0.5, out=out)
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)


def luma01_from_bgr01_srgb(bgr01: np.ndarray) -> np.ndarray:
    bgr01 = _ensure_contiguous(bgr01.astype(np.float32, copy=False))
    return (
        cv2.transform(bgr01, _LUMA_KERNEL_BGR)
        .reshape(bgr01.shape[:2])
        .astype(np.float32, copy=False)
    )


def luma01_from_bgr01_linear(bgr01: np.ndarray) -> np.ndarray:
    bgr_lin = srgb_to_linear(bgr01)
    bgr_lin = _ensure_contiguous(bgr_lin)
    return (
        cv2.transform(bgr_lin, _LUMA_KERNEL_BGR)
        .reshape(bgr_lin.shape[:2])
        .astype(np.float32, copy=False)
    )


def smoothstep(x: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    result = t * t
    np.multiply(result, 3.0 - 2.0 * t, out=result)
    return result.astype(np.float32, copy=False)


def apply_gain_linear_bgr(bgr01: np.ndarray, gain: float) -> np.ndarray:
    t = np.linspace(0.0, 1.0, _N, dtype=np.float64)
    a = 0.055
    t_lin = np.where(t <= 0.04045, t / 12.92, ((t + a) / (1.0 + a)) ** 2.4)
    t_lin = np.clip(t_lin * float(gain), 0.0, 1.0)
    t_out = np.where(
        t_lin <= 0.0031308, t_lin * 12.92, (1.0 + a) * (t_lin ** (1.0 / 2.4)) - a
    )
    lut = np.clip(t_out, 0.0, 1.0).astype(np.float32)
    idx = np.clip(bgr01 * np.float32(_N - 1) + np.float32(0.5), 0, _N - 1).astype(
        np.uint16
    )
    return lut[idx]


def _apply_y_ratio_lut(
    bgr01: np.ndarray,
    Y: np.ndarray,
    ratio_lut: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    idx = np.clip(Y * np.float32(_N - 1) + np.float32(0.5), 0, _N - 1).astype(np.uint16)
    ratio = ratio_lut[idx]
    if mask is not None:
        ratio = np.where(mask, ratio, np.float32(1.0))
    out = bgr01 * ratio[..., None]
    np.clip(out, 0.0, 1.0, out=out)
    return out


@dataclass
class Config:
    jpeg_quality: int = 95

    raw_use_camera_wb: bool = True
    raw_use_auto_wb: bool = False
    raw_no_auto_bright: bool = True
    raw_output_color_srgb: bool = True
    raw_highlight_mode: int = 1

    xrite_name_hint: str = "xrite"
    xrite_skip_export: bool = True
    xrite_required: bool = False

    wb_sat_max: float = 0.16
    wb_y_min: float = 0.14
    wb_y_max: float = 0.95
    wb_max_channel: float = 0.995
    wb_min_pixels: int = 2500

    wb_gain_clip_min: float = 0.60
    wb_gain_clip_max: float = 1.70
    wb_strength: float = 0.85
    use_opencv_wb: bool = True
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

    shadows_lift: float = 0.18
    shadows_start: float = 0.00
    shadows_end: float = 0.52
    shadows_max_boost: float = 2.10

    midtone_target: float = 0.90
    subject_white_cut: float = 0.965
    p_min: float = 1.00
    p_max: float = 3.00
    lift_strength: float = 0.85

    pop_strength: float = 1.0
    contrast_factor: float = 0.97
    contrast_pivot: float = 0.58
    microcontrast_amount: float = 0.50
    microcontrast_radius: float = 6.0

    vibrance: float = 0.90
    vibrance_max_boost: float = 1.12
    vibrance_highlight_start: float = 0.88
    vibrance_highlight_end: float = 0.99
    vibrance_neutral_chroma_max: float = 28.0
    vibrance_neutral_soft: float = 56.0

    highlight_protect_start: float = 0.88
    highlight_protect_end: float = 0.99
    highlight_protect_strength: float = 0.70

    highlight_roll_start: float = 0.92
    highlight_roll_strength: float = 0.55

    highlight_detint_enable: bool = True
    highlight_detint_start: float = 0.94
    highlight_detint_end: float = 0.995
    highlight_detint_strength: float = 0.65

    sharpness_amount: float = 1.8
    sharpness_radius: float = 1.0
    sharpness_threshold: int = 0
    final_brightness: float = 1.01

    think_long_edge: int = 1600
    think_min_edge: int = 640


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
        return (bgr8.astype(np.uint16) * 257).astype(np.uint16, copy=False)
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


def _thinking_resize_bgr01(bgr01, cfg):
    h, w = bgr01.shape[:2]
    long_edge = max(h, w)
    short_edge = min(h, w)

    tgt_long = int(max(1, cfg.think_long_edge))
    tgt_min = int(max(1, cfg.think_min_edge))

    if long_edge <= tgt_long and short_edge <= tgt_min:
        return bgr01, {"applied": False, "scale": 1.0, "size": (w, h)}
    scale = min(max(tgt_long / float(long_edge), tgt_min / float(short_edge)), 1.0)
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))

    if nh == h and nw == w:
        return bgr01, {"applied": False, "scale": 1.0, "size": (w, h)}

    small = cv2.resize(bgr01, (nw, nh), interpolation=cv2.INTER_AREA)
    if small.dtype != np.float32:
        small = small.astype(np.float32, copy=False)
    return small, {
        "applied": True,
        "scale": float(scale),
        "size": (nw, nh),
        "orig": (w, h),
    }


def _upsample_mask_nn(mask_small, target_hw):
    th, tw = target_hw
    up = cv2.resize(
        mask_small.astype(np.uint8) * 255, (tw, th), interpolation=cv2.INTER_NEAREST
    )
    return up > 0


def opencv_white_balance_bgr16(bgr16, cfg):
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

        gains = np.clip(
            (dst_med / src_med).astype(np.float32),
            cfg.wb_gain_clip_min,
            cfg.wb_gain_clip_max,
        )
        out16 = np.clip(
            bgr16.astype(np.float32) * gains[None, None, :], 0.0, 65535.0
        ).astype(np.uint16)
        return out16, {"applied": True, "method": "opencv", "gains_bgr": gains.tolist()}
    except Exception as e:
        return bgr16, {"applied": False, "reason": f"wb_failed: {e}"}


def compute_anchor_wb_from_xrite_bgr16(bgr16, cfg):
    bgr01 = bgr16_to_float01(bgr16)
    bgr8 = float01_to_bgr8(bgr01)
    hsv = cv2.cvtColor(bgr8, cv2.COLOR_BGR2HSV).astype(np.float32)
    S = hsv[..., 1] / 255.0
    Y = luma01_from_bgr01_linear(bgr01)

    mask = (
        (S <= float(cfg.wb_sat_max) * 1.5)
        & (Y >= float(cfg.wb_y_min) * 0.7)
        & (Y <= float(cfg.wb_y_max) * 1.05)
    )
    n = int(mask.sum())

    if n < 500:
        return None, {"ok": False, "reason": "too_few_neutrals_pixels", "n": n}
    bgr_lin = srgb_to_linear(bgr01)
    med_bgr = np.median(bgr_lin[mask].reshape(-1, 3), axis=0).astype(np.float32) + 1e-8
    med_rgb = med_bgr[::-1].copy()
    tgt = float(np.mean(med_rgb))
    gains_rgb = np.clip(
        (tgt / med_rgb).astype(np.float32), cfg.wb_gain_clip_min, cfg.wb_gain_clip_max
    )
    a = float(np.clip(cfg.wb_strength, 0.0, 1.0))
    gains_rgb = 1.0 + a * (gains_rgb - 1.0)
    return gains_rgb, {
        "ok": True,
        "n": n,
        "rgb_lin_median": med_rgb.tolist(),
        "wb_gains_rgb_lin": gains_rgb.tolist(),
        "strength": a,
    }


def apply_anchor_wb_linear_rgb(bgr01, wb_gains_rgb_lin):
    bgr01 = bgr01.astype(np.float32, copy=False)
    bgr_lin = srgb_to_linear(bgr01)
    gains_bgr = np.array(
        [wb_gains_rgb_lin[2], wb_gains_rgb_lin[1], wb_gains_rgb_lin[0]],
        dtype=np.float32,
    )
    bgr_lin = np.clip(bgr_lin * gains_bgr[None, None, :], 0.0, 1.0)
    return np.clip(linear_to_srgb(bgr_lin), 0.0, 1.0).astype(np.float32, copy=False)


def detect_white_product_scene(bgr01, bgr8=None):
    Y = luma01_from_bgr01_srgb(bgr01)
    y_med = float(np.median(Y))
    y_p95 = float(np.percentile(Y, 95.0))

    if bgr8 is None:
        bgr8 = float01_to_bgr8(bgr01)
    hsv = cv2.cvtColor(bgr8, cv2.COLOR_BGR2HSV).astype(np.float32)
    s_med = float(np.median(hsv[..., 1] / 255.0))

    return {
        "is_whiteish": (s_med < 0.10) and (y_p95 > 0.88),
        "is_dark_subject": y_med < 0.25,
        "s_med": s_med,
        "y_med": y_med,
    }


def _border_mask(h, w, frac):
    b = int(max(1, round(min(h, w) * float(frac))))
    m = np.zeros((h, w), dtype=bool)
    m[:b, :] = True
    m[-b:, :] = True
    m[:, :b] = True
    m[:, -b:] = True
    return m


def compute_bg_mask(bgr01, cfg, hsv=None):
    if hsv is None:
        hsv = cv2.cvtColor(float01_to_bgr8(bgr01), cv2.COLOR_BGR2HSV).astype(np.float32)
    S = hsv[..., 1] / 255.0
    V = hsv[..., 2] / 255.0

    v_thr = float(cfg.bg_v_min)
    s_thr = float(cfg.bg_s_max)
    bg = (V >= v_thr) & (S <= s_thr)

    if int(bg.sum()) < int(cfg.min_bg_for_border):
        h, w = bg.shape
        bm = _border_mask(h, w, float(cfg.border_frac))
        s_p35 = float(np.percentile(S[bm], 35.0))
        v_p65 = float(np.percentile(V[bm], 65.0))
        bg2 = (V >= max(v_thr, min(0.92, v_p65))) & (S <= min(s_thr, max(0.10, s_p35)))
        if int(bg2.sum()) > int(bg.sum()):
            bg = bg2

    bg_u8 = bg.astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bg_u8 = cv2.morphologyEx(bg_u8, cv2.MORPH_OPEN, k, iterations=1)
    bg_u8 = cv2.morphologyEx(bg_u8, cv2.MORPH_OPEN, k, iterations=2)
    return bg_u8 > 0


def compute_specular_mask(bgr01, cfg, hsv=None, Ylin=None):
    if hsv is None:
        hsv = cv2.cvtColor(float01_to_bgr8(bgr01), cv2.COLOR_BGR2HSV).astype(np.float32)
    S = hsv[..., 1] / 255.0
    V = hsv[..., 2] / 255.0

    if Ylin is None:
        Ylin = luma01_from_bgr01_linear(bgr01)
    gx = cv2.Sobel(Ylin, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(Ylin, cv2.CV_32F, 0, 1, ksize=3)
    g = cv2.magnitude(gx, gy)
    np.clip(g, 0.0, 1.0, out=g)
    spec = (
        (V >= float(cfg.spec_v_min))
        & (S <= float(cfg.spec_s_max))
        & (g >= float(cfg.spec_grad_min))
    )

    spec_u8 = spec.astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(spec_u8, k, iterations=1) > 0


def compute_subject_mask(bg_mask):
    subj_u8 = (~bg_mask).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    subj_u8 = cv2.morphologyEx(subj_u8, cv2.MORPH_CLOSE, k, iterations=2)
    subj_u8 = cv2.morphologyEx(subj_u8, cv2.MORPH_OPEN, k, iterations=1)
    return subj_u8 > 0


def compute_exposure_gain_from_bg(bgr01, bg_mask, spec_mask, cfg):
    Y = luma01_from_bgr01_linear(bgr01)
    usable = bg_mask & (~spec_mask)
    if int(usable.sum()) < int(cfg.bg_min_pixels):
        usable = np.ones_like(bg_mask, dtype=bool)
        ref = "all"
    else:
        ref = "bg"
    base = max(float(np.percentile(Y[usable], float(cfg.exp_pctl))), 1e-6)
    hi = max(float(np.percentile(Y[usable], float(cfg.exp_highlight_pctl))), 1e-6)
    gain = float(
        np.clip(
            min(float(cfg.exp_target) / base, float(cfg.exp_highlight_cap) / hi),
            cfg.exp_gain_min,
            cfg.exp_gain_max,
        )
    )
    return {"ref": ref, "base": base, "gain": gain}


def compute_bg_whiten_gain(bgr01, bg_mask, spec_mask, cfg):
    if not cfg.bg_whiten_enable:
        return {"applied": True, "gain": 1.0}

    Y = luma01_from_bgr01_linear(bgr01)
    usable = bg_mask & (~spec_mask)
    if int(usable.sum()) < int(cfg.bg_min_pixels):
        return {"applied": False, "reason": "too_few_bg_pixels", "gain": 1.0}

    base = max(float(np.percentile(Y[usable], float(cfg.bg_whiten_pctl))), 1e-6)
    if base >= float(cfg.bg_whiten_target) - 1e-3:
        return {"applied": False, "base": base, "gain": 1.0}

    gain = float(
        np.clip(float(cfg.bg_whiten_target) / base, 1.0, float(cfg.bg_whiten_gain_max))
    )
    return {"applied": True, "base": base, "gain": gain}


def apply_shadows_and_midtone_fused(
    bgr01: np.ndarray, cfg: Config, scene_info: Dict
) -> Tuple[np.ndarray, Dict[str, Any]]:
    Y = luma01_from_bgr01_srgb(bgr01)
    yv = np.linspace(0.0, 1.0, _N, dtype=np.float64)

    strength = float(np.clip(cfg.shadows_lift, 0.0, 1.0))
    sh_mult = np.ones(_N, dtype=np.float64)
    sh_dbg = {"applied": False}
    if strength > 1e-6:
        s0 = float(np.clip(cfg.shadows_start, 0.0, 0.95))
        s1 = float(np.clip(cfg.shadows_end, s0 + 1e-3, 1.0))
        gamma = 1.0 / (1.0 + 1.8 * strength)
        t = np.clip((yv - s0) / max(s1 - s0, 1e-6), 0.0, 1.0)
        w = (1.0 - t * t * (3.0 - 2.0 * t)) * strength
        y_lift = np.power(np.clip(yv, 0.0, 1.0), gamma)
        ratio = np.clip(
            y_lift / np.maximum(yv, 1e-4), 1.0, float(cfg.shadows_max_boost)
        )
        sh_mult = 1.0 + w * (ratio - 1.0)
        sh_dbg = {"applied": True, "strength": strength, "gamma": gamma, "end": s1}

    y_after_sh = np.clip(yv * sh_mult, 0.0, 1.0)

    subj_mask = Y < cfg.subject_white_cut
    if int(subj_mask.sum()) < 2000:
        subj_mask = np.ones(Y.shape, dtype=bool)

    med = min(max(float(np.median(Y[subj_mask])), 1e-6), 0.999999)
    tgt = float(cfg.midtone_target)
    is_dark_scene = False
    if med < 0.25:
        is_dark_scene = True
        tgt = min(max(med * 2.5, 0.40), cfg.midtone_target)
    p = float(
        np.clip(
            np.log(max(1.0 - tgt, 1e-6)) / np.log(max(1.0 - med, 1e-6)),
            cfg.p_min,
            cfg.p_max,
        )
    )
    a = float(np.clip(cfg.lift_strength, 0.0, 1.0))
    if scene_info.get("is_whiteish", False):
        a *= 0.75

    Yp = 1.0 - np.power(np.clip(1.0 - y_after_sh, 0.0, 1.0), p)
    scale = np.clip(Yp / np.maximum(y_after_sh, 1e-4), 0.75, 2.2)
    mt_mult = 1.0 + a * (scale - 1.0)

    combined = (sh_mult * mt_mult).astype(np.float32)

    out = _apply_y_ratio_lut(bgr01, Y, combined)
    return out, {
        "shadows": sh_dbg,
        "midtone": {
            "applied": True,
            "subject_median": med,
            "target": tgt,
            "is_dark_fix": is_dark_scene,
            "p": p,
        },
    }


def highlight_protect_blend(original, edited, cfg, Y_original=None):
    if Y_original is None:
        Y_original = luma01_from_bgr01_srgb(original)
    w = smoothstep(Y_original, cfg.highlight_protect_start, cfg.highlight_protect_end)
    w *= float(np.clip(cfg.highlight_protect_strength, 0.0, 1.0))
    diff = np.subtract(original, edited)
    np.multiply(diff, w[..., None], out=diff)
    np.add(edited, diff, out=diff)
    np.clip(diff, 0.0, 1.0, out=diff)
    return diff.astype(np.float32, copy=False), {"applied": True}


def highlight_roll_and_detint(bgr01, cfg):
    Y = luma01_from_bgr01_srgb(bgr01)
    hr_dbg = {"applied": False}
    hd_dbg = {"applied": False}

    strength_r = float(np.clip(cfg.highlight_roll_strength, 0.0, 1.0))
    start_r = float(np.clip(cfg.highlight_roll_start, 0.0, 0.999))
    if strength_r > 1e-6:
        yv = np.linspace(0.0, 1.0, _N, dtype=np.float64)
        t = np.clip((yv - start_r) / max(1.0 - start_r, 1e-6), 0.0, 1.0)
        Y2 = np.clip(yv - strength_r * (t * t) * (yv - start_r), 0.0, 1.0)
        ratio = np.clip(Y2 / np.maximum(yv, 1e-4), 0.80, 1.0).astype(np.float32)
        bgr01 = _apply_y_ratio_lut(bgr01, Y, ratio)
        idx = np.clip(Y * np.float32(_N - 1) + np.float32(0.5), 0, _N - 1).astype(
            np.uint16
        )
        Y = np.clip(Y * ratio[idx], 0.0, 1.0)
        hr_dbg = {"applied": True}

    if cfg.highlight_detint_enable:
        w = smoothstep(
            Y, float(cfg.highlight_detint_start), float(cfg.highlight_detint_end)
        )
        a = float(np.clip(cfg.highlight_detint_strength, 0.0, 1.0))
        w *= a
        if float(w.max()) > 1e-6:
            m = np.max(bgr01, axis=2, keepdims=True)
            w3 = w[..., None]
            diff = np.subtract(m, bgr01)
            np.multiply(diff, w3, out=diff)
            np.add(bgr01, diff, out=diff)
            np.clip(diff, 0.0, 1.0, out=diff)
            bgr01 = diff.astype(np.float32, copy=False)
            hd_dbg = {"applied": True}

    return bgr01, hr_dbg, hd_dbg


def apply_subject_pop(bgr01, subject_mask, cfg):
    pop = float(np.clip(cfg.pop_strength, 0.0, 1.0))
    c_dbg = {"applied": False}
    mc_dbg = {"applied": False}
    if pop <= 1e-6:
        return bgr01, c_dbg, mc_dbg

    Y = luma01_from_bgr01_srgb(bgr01)

    cf = 1.0 + (float(cfg.contrast_factor) - 1.0) * pop
    pivot = float(np.clip(cfg.contrast_pivot, 0.0, 1.0))
    yv = np.linspace(0.0, 1.0, _N, dtype=np.float64)
    Yc = np.clip(pivot + cf * (yv - pivot), 0.0, 1.0)
    ratio_c = np.clip(Yc / np.maximum(yv, 1e-4), 0.85, 1.35).astype(np.float32)
    bgr01 = _apply_y_ratio_lut(bgr01, Y, ratio_c, mask=subject_mask)
    c_dbg = {"applied": True, "cf": cf, "pivot": pivot}

    idx = np.clip(Y * np.float32(_N - 1) + np.float32(0.5), 0, _N - 1).astype(np.uint16)
    ratio_at_Y = np.where(subject_mask, ratio_c[idx], np.float32(1.0))
    Y = np.clip(Y * ratio_at_Y, 0.0, 1.0)

    amt = float(np.clip(cfg.microcontrast_amount, 0.0, 2.0)) * pop
    if amt > 1e-6:
        blur = cv2.GaussianBlur(Y, (0, 0), float(cfg.microcontrast_radius))
        Y2 = np.clip(Y + amt * (Y - blur), 0.0, 1.0)
        ratio_mc = Y2 / np.maximum(Y, 1e-4)
        np.clip(ratio_mc, 0.90, 1.20, out=ratio_mc)
        ratio_mc = np.where(subject_mask, ratio_mc, np.float32(1.0))
        out = bgr01 * ratio_mc[..., None]
        np.clip(out, 0.0, 1.0, out=out)
        bgr01 = out.astype(np.float32, copy=False)
        mc_dbg = {
            "applied": True,
            "amt": amt,
            "radius": float(cfg.microcontrast_radius),
        }

    return bgr01, c_dbg, mc_dbg


def apply_vibrance_ycrb_protected(bgr01, cfg):
    v = float(np.clip(cfg.vibrance, 0.0, 2.0))
    if v <= 1e-6:
        return bgr01, {"applied": False}

    Y_orig = luma01_from_bgr01_srgb(bgr01)

    bgr_c = _ensure_contiguous(bgr01.astype(np.float32, copy=False))
    ycrcb = cv2.transform(bgr_c, _M_BGR2YCrCb)

    dCr = ycrcb[..., 1] - 0.5
    dCb = ycrcb[..., 2] - 0.5
    chroma = cv2.magnitude(dCr, dCb)

    c0 = float(max(cfg.vibrance_neutral_chroma_max, 0.0)) / 255.0
    c1 = (
        float(max(cfg.vibrance_neutral_soft, cfg.vibrance_neutral_chroma_max + 1.0))
        / 255.0
    )
    neutral_ramp = np.clip((chroma - c0) / max(c1 - c0, 1e-6), 0.0, 1.0)

    wh = smoothstep(
        ycrcb[..., 0], cfg.vibrance_highlight_start, cfg.vibrance_highlight_end
    )
    np.subtract(1.0, wh, out=wh)

    np.multiply(wh, neutral_ramp, out=wh)

    chroma_norm = np.clip(chroma / 0.5, 0.0, 1.0)
    np.subtract(1.0, chroma_norm, out=chroma_norm)
    np.multiply(chroma_norm, v, out=chroma_norm)
    np.clip(chroma_norm, 0.0, float(cfg.vibrance_max_boost) - 1.0, out=chroma_norm)

    np.multiply(chroma_norm, wh, out=chroma_norm)
    np.add(chroma_norm, 1.0, out=chroma_norm)

    np.multiply(dCr, chroma_norm, out=dCr)
    np.multiply(dCb, chroma_norm, out=dCb)
    np.add(dCr, 0.5, out=dCr)
    np.add(dCb, 0.5, out=dCb)
    ycrcb[..., 1] = dCr
    ycrcb[..., 2] = dCb

    ycrcb = _ensure_contiguous(ycrcb)
    out = cv2.transform(ycrcb, _M_YCrCb2BGR)
    np.clip(out, 0.0, 1.0, out=out)

    Y_new = luma01_from_bgr01_srgb(out)
    ratio = Y_orig / np.maximum(Y_new, 1e-4)
    np.clip(ratio, 0.95, 1.05, out=ratio)
    np.multiply(out, ratio[..., None], out=out)
    np.clip(out, 0.0, 1.0, out=out)
    return out.astype(np.float32, copy=False), {"applied": True}


def apply_unsharp_mask(bgr8, cfg):
    if cfg.sharpness_amount <= 0:
        return bgr8
    blurred = cv2.GaussianBlur(bgr8, (0, 0), cfg.sharpness_radius)
    sharpened = cv2.addWeighted(
        bgr8, 1.0 + cfg.sharpness_amount, blurred, -cfg.sharpness_amount, 0
    )
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def process_bgr16(bgr16, cfg, anchor_wb_gains_rgb_lin, _profile=False):
    dbg = {}
    timings = {}

    def _t():
        return time.perf_counter()

    t_start = _t()

    t0 = _t()
    bgr16_2, wb16_dbg = opencv_white_balance_bgr16(bgr16, cfg)
    dbg["wb_opencv"] = wb16_dbg
    timings["wb_opencv"] = _t() - t0

    t0 = _t()
    bgr01_full = bgr16_to_float01(bgr16_2)
    timings["bgr16_to_float"] = _t() - t0

    t0 = _t()
    if anchor_wb_gains_rgb_lin is not None:
        bgr01_full = apply_anchor_wb_linear_rgb(bgr01_full, anchor_wb_gains_rgb_lin)
        dbg["wb_anchor"] = {
            "applied": True,
            "gains_rgb_lin": anchor_wb_gains_rgb_lin.tolist(),
        }
    else:
        dbg["wb_anchor"] = {"applied": False, "reason": "no_anchor_gains"}
    timings["anchor_wb"] = _t() - t0

    t0 = _t()
    bgr01_think, think_dbg = _thinking_resize_bgr01(bgr01_full, cfg)
    dbg["thinking"] = think_dbg
    timings["think_resize_1"] = _t() - t0

    t0 = _t()
    bgr8_t = float01_to_bgr8(bgr01_think)
    hsv_t = cv2.cvtColor(bgr8_t, cv2.COLOR_BGR2HSV).astype(np.float32)
    Ylin_t = luma01_from_bgr01_linear(bgr01_think)
    timings["think_convert_1"] = _t() - t0

    t0 = _t()
    scene = detect_white_product_scene(bgr01_think, bgr8=bgr8_t)
    dbg["scene"] = scene
    timings["scene_detect"] = _t() - t0

    t0 = _t()
    bg_mask_t = compute_bg_mask(bgr01_think, cfg, hsv=hsv_t)
    spec_mask_t = compute_specular_mask(bgr01_think, cfg, hsv=hsv_t, Ylin=Ylin_t)
    timings["masks_1"] = _t() - t0

    t0 = _t()
    Y_pre_exp = luma01_from_bgr01_srgb(bgr01_full)
    pre_exposure_ref_full = bgr01_full.copy()
    timings["copy_pre_exp"] = _t() - t0

    t0 = _t()
    exp_dbg = compute_exposure_gain_from_bg(bgr01_think, bg_mask_t, spec_mask_t, cfg)
    dbg["exposure_bg_linear"] = exp_dbg
    timings["exposure_calc"] = _t() - t0

    gain_total = float(exp_dbg["gain"])
    fb = float(cfg.final_brightness)
    if abs(fb - 1.0) > 1e-4:
        gain_total *= fb
        dbg["final_brightness"] = {"applied": True, "gain": fb}
    else:
        dbg["final_brightness"] = {"applied": False}

    t0 = _t()
    if abs(gain_total - 1.0) > 1e-6:
        bgr01_think = apply_gain_linear_bgr(bgr01_think, gain_total)
    timings["gain_think"] = _t() - t0

    t0 = _t()
    bgr8_t2 = float01_to_bgr8(bgr01_think)
    hsv_t2 = cv2.cvtColor(bgr8_t2, cv2.COLOR_BGR2HSV).astype(np.float32)
    Ylin_t2 = luma01_from_bgr01_linear(bgr01_think)
    timings["think_convert_2"] = _t() - t0

    t0 = _t()
    bg_mask2_t = compute_bg_mask(bgr01_think, cfg, hsv=hsv_t2)
    spec_mask2_t = compute_specular_mask(bgr01_think, cfg, hsv=hsv_t2, Ylin=Ylin_t2)
    timings["masks_2"] = _t() - t0

    t0 = _t()
    wb_bg_dbg = compute_bg_whiten_gain(bgr01_think, bg_mask2_t, spec_mask2_t, cfg)
    dbg["bg_whiten_failsafe"] = wb_bg_dbg
    timings["bg_whiten_calc"] = _t() - t0

    g_bg = float(wb_bg_dbg.get("gain", 1.0))
    if g_bg != 1.0:
        gain_total *= g_bg

    t0 = _t()
    if abs(gain_total - 1.0) > 1e-6:
        bgr01_full = apply_gain_linear_bgr(bgr01_full, gain_total)
    timings["gain_full"] = _t() - t0

    t0 = _t()
    bgr01_full, hp_dbg = highlight_protect_blend(
        pre_exposure_ref_full, bgr01_full, cfg, Y_original=Y_pre_exp
    )
    dbg["highlight_protect"] = hp_dbg
    timings["highlight_protect"] = _t() - t0
    del pre_exposure_ref_full, Y_pre_exp

    t0 = _t()
    bgr01_think, _ = _thinking_resize_bgr01(bgr01_full, cfg)
    bgr8_t3 = float01_to_bgr8(bgr01_think)
    hsv_t3 = cv2.cvtColor(bgr8_t3, cv2.COLOR_BGR2HSV).astype(np.float32)
    Ylin_t3 = luma01_from_bgr01_linear(bgr01_think)
    timings["think_resize_convert_3"] = _t() - t0

    t0 = _t()
    bg_mask3_t = compute_bg_mask(bgr01_think, cfg, hsv=hsv_t3)
    spec_mask3_t = compute_specular_mask(bgr01_think, cfg, hsv=hsv_t3, Ylin=Ylin_t3)
    subj_mask_t = compute_subject_mask(bg_mask3_t)
    timings["masks_3"] = _t() - t0

    t0 = _t()
    subj_mask_full = _upsample_mask_nn(subj_mask_t, bgr01_full.shape[:2])
    bg_mask_full = _upsample_mask_nn(bg_mask3_t, bgr01_full.shape[:2])
    spec_mask_full = _upsample_mask_nn(spec_mask3_t, bgr01_full.shape[:2])
    timings["mask_upsample"] = _t() - t0

    dbg["mask_counts"] = {
        "bg": int(bg_mask_full.sum()),
        "subj": int(subj_mask_full.sum()),
        "spec": int(spec_mask_full.sum()),
    }

    t0 = _t()
    bgr01_full, sh_mt_dbg = apply_shadows_and_midtone_fused(bgr01_full, cfg, scene)
    dbg["shadows_lift"] = sh_mt_dbg.get("shadows", {})
    dbg["midtone_lift"] = sh_mt_dbg.get("midtone", {})
    timings["shadows_midtone_fused"] = _t() - t0

    t0 = _t()
    bgr01_full, v_dbg = apply_vibrance_ycrb_protected(bgr01_full, cfg)
    dbg["vibrance_ycrcb"] = v_dbg
    timings["vibrance"] = _t() - t0

    t0 = _t()
    bgr01_full, c_dbg, mc_dbg = apply_subject_pop(bgr01_full, subj_mask_full, cfg)
    dbg["subject_contrast"] = c_dbg
    dbg["subject_microcontrast"] = mc_dbg
    timings["subject_pop"] = _t() - t0

    t0 = _t()
    bgr01_full, hr_dbg, hd_dbg = highlight_roll_and_detint(bgr01_full, cfg)
    dbg["highlight_rolloff"] = hr_dbg
    dbg["highlight_detint"] = hd_dbg
    timings["highlight_roll_detint"] = _t() - t0

    t0 = _t()
    bgr8 = float01_to_bgr8(bgr01_full)
    bgr8 = apply_unsharp_mask(bgr8, cfg)
    timings["sharpen_output"] = _t() - t0

    timings["TOTAL"] = _t() - t_start

    if _profile:
        print("\n── PIPELINE PROFILE ──")
        for name, dur in sorted(timings.items(), key=lambda kv: -kv[1]):
            pct = dur / timings["TOTAL"] * 100.0
            print(f"  {name:<30s} {dur:6.3f}s  {pct:5.1f}%  {'█' * int(pct / 2)}")
        print()

    dbg["timings"] = timings
    return bgr8, dbg


def iter_images_prefetched(input_dir, cfg):
    file_list = list(iter_images(input_dir))
    if not file_list:
        return
    with ThreadPoolExecutor(max_workers=2) as io_pool:
        fut = io_pool.submit(read_image_any_to_bgr16, file_list[0], cfg)
        for i, p in enumerate(file_list):
            bgr16 = fut.result()
            if i + 1 < len(file_list):
                fut = io_pool.submit(read_image_any_to_bgr16, file_list[i + 1], cfg)
            if bgr16 is not None:
                yield p, bgr16


def find_xrite_file(input_dir, cfg):
    candidates = [p for p in iter_images(input_dir) if _is_xrite_file(p, cfg)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: str(p).lower())
    return candidates[0]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="input")
    ap.add_argument("--output", type=str, default="output")
    ap.add_argument("--quality", type=int, default=CFG.jpeg_quality)
    ap.add_argument("--sharpness", type=float, default=CFG.sharpness_amount)
    ap.add_argument("--vibrance", type=float, default=CFG.vibrance)
    ap.add_argument("--contrast", type=float, default=CFG.contrast_factor)
    ap.add_argument("--contrast-pivot", type=float, default=CFG.contrast_pivot)
    ap.add_argument("--pop", type=float, default=CFG.pop_strength)
    ap.add_argument("--microcontrast", type=float, default=CFG.microcontrast_amount)
    ap.add_argument("--micro-radius", type=float, default=CFG.microcontrast_radius)
    ap.add_argument("--shadows", type=float, default=CFG.shadows_lift)
    ap.add_argument("--shadows-end", type=float, default=CFG.shadows_end)
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
    ap.add_argument("--think-long-edge", type=int, default=CFG.think_long_edge)
    ap.add_argument("--think-min-edge", type=int, default=CFG.think_min_edge)
    ap.add_argument(
        "--profile", action="store_true", help="Print per-step timing for first image."
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Parallel workers for batch processing (default 2, use 1 for sequential).",
    )
    return ap.parse_args()


def _process_single(args_tuple):
    """Worker for ProcessPoolExecutor. Receives serialized args to avoid pickle issues."""
    (
        img_path_str,
        input_dir_str,
        output_dir_str,
        cfg_dict,
        anchor_wb_list,
        jpeg_quality,
    ) = args_tuple
    cfg = Config()
    for k, v in cfg_dict.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    p = Path(img_path_str)
    anchor_wb = np.array(anchor_wb_list, dtype=np.float32) if anchor_wb_list else None
    bgr16 = read_image_any_to_bgr16(p, cfg)
    if bgr16 is None:
        return (p.name, False, "read_failed")
    out8, dbg = process_bgr16(bgr16, cfg, anchor_wb)
    rel = p.relative_to(input_dir_str)
    out_path = (Path(output_dir_str) / rel).with_suffix(".jpg")
    write_jpeg(out_path, out8, jpeg_quality)
    exp = dbg.get("exposure_bg_linear", {})
    think = dbg.get("thinking", {})
    return (
        p.name,
        True,
        f"ExpGain={exp.get('gain',0):.2f}({exp.get('ref','?')}) "
        f"ThinkScale={think.get('scale',1.0):.3f}",
    )


def main():
    t0 = time.perf_counter()
    args = parse_args()
    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    if not input_dir.exists():
        raise SystemExit(f"Input folder does not exist: {input_dir}")

    cfg = Config()
    cfg.jpeg_quality = int(args.quality)

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

    cfg.think_long_edge = int(max(256, args.think_long_edge))
    cfg.think_min_edge = int(max(256, args.think_min_edge))

    ensure_dir(output_dir)

    anchor_path = find_xrite_file(input_dir, cfg)
    anchor_wb = None
    if anchor_path is None:
        msg = f"[WARN] No xrite anchor found (hint='{cfg.xrite_name_hint}')"
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
                print(msg + " Proceeding without anchor WB")
            else:
                print(f"[INFO] Xrite anchor: {anchor_path.name}")
                print(
                    f"[INFO] Anchor WB gains (RGB linear): {wb_dbg.get('wb_gains_rgb_lin')} n={wb_dbg.get('n')}"
                )

    count_in = 0
    count_out = 0
    count_tagged = 0

    run_date = date.today().isoformat()
    do_profile = bool(args.profile)
    n_workers = max(1, int(args.workers))

    print(f"Processing images from {input_dir} -> {output_dir}")
    print(
        f"Settings: Pop={cfg.pop_strength} | Contrast={cfg.contrast_factor} (pivot {cfg.contrast_pivot}) | "
        f"Micro={cfg.microcontrast_amount}@{cfg.microcontrast_radius} | "
        f"Vibrance={cfg.vibrance} | Shadows={cfg.shadows_lift} | "
        f"AnchorWB={'on' if anchor_wb is not None else 'off'} | "
        f"BGWhiten={'on' if cfg.bg_whiten_enable else 'off'} | "
        f"ThinkLongEdge={cfg.think_long_edge} (min {cfg.think_min_edge}) | "
        f"Workers={n_workers}"
    )

    all_files = [
        p
        for p in iter_images(input_dir)
        if not (cfg.xrite_skip_export and _is_xrite_file(p, cfg))
    ]
    count_in = len(all_files)

    if n_workers >= 2 and count_in >= 2:
        if do_profile and all_files:
            bgr16_first = read_image_any_to_bgr16(all_files[0], cfg)
            if bgr16_first is not None:
                process_bgr16(bgr16_first, cfg, anchor_wb, _profile=True)
                del bgr16_first

        cfg_dict = {k: getattr(cfg, k) for k in vars(cfg) if not k.startswith("_")}
        anchor_list = anchor_wb.tolist() if anchor_wb is not None else None

        work_items = [
            (
                str(p),
                str(input_dir),
                str(output_dir),
                cfg_dict,
                anchor_list,
                cfg.jpeg_quality,
            )
            for p in all_files
        ]

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_process_single, item): item[0] for item in work_items
            }
            for fut in as_completed(futures):
                name, ok, msg = fut.result()
                if ok:
                    count_out += 1
                print(
                    f"[{count_out}/{count_in}] {'OK' if ok else 'FAIL'} {name}  {msg}"
                )

    else:
        for p, bgr16 in iter_images_prefetched(input_dir, cfg):
            count_in += 1
            if cfg.xrite_skip_export and _is_xrite_file(p, cfg):
                print(f"Skip (xrite ref): {p.name}")
                continue

            profile_this = do_profile and (count_out == 0)
            out8, dbg = process_bgr16(bgr16, cfg, anchor_wb, _profile=profile_this)

            rel = p.relative_to(input_dir)
            out_path = (output_dir / rel).with_suffix(".jpg")
            write_jpeg(out_path, out8, cfg.jpeg_quality)
            count_out += 1

            exp = dbg.get("exposure_bg_linear", {})
            wb_bg = dbg.get("bg_whiten_failsafe", {})
            c = dbg.get("subject_contrast", {})
            mc = dbg.get("subject_microcontrast", {})
            think = dbg.get("thinking", {})
            print(
                f"Saved: {output_dir.name} | "
                f"ExpGain={exp.get('gain',0):.2f}({exp.get('ref', '?')}) "
                f"BGWhiten={('on' if wb_bg.get('applied') else 'off')} "
                f"SubContrast={('on' if c.get('applied') else 'off')} "
                f"Micro={('on' if mc.get('applied') else 'off')} "
                f"ThinkScale={think.get('scale', 1.0):.3f}"
            )

    total = time.perf_counter() - t0
    rate = (count_out / total) if total > 0 else 0.0
    print(
        f"Done. Read {count_in} files, wrote {count_out} JPEGs.\nTotal time: {total:.2f}s ({rate:.2f} img/s)"
    )


if __name__ == "__main__":
    main()
