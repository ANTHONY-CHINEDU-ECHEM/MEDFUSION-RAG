"""Deterministic procedural rendering of CT key images and histology tiles.

Images are not stored in the CSV. Each row carries the geometry (nodule
position, diameter, type, margin, calcification, emphysema, lymph nodes) and a
seed, and this module renders a stylised but anatomically plausible image on
demand. Rendering is fully deterministic, so the dataset remains compact and
exactly reproducible while still giving the vision encoders real pixels whose
content is causally tied to the labels.

CT key image: 128 x 128 single channel, nominal pixel spacing 2.6 mm
(333 mm field of view), lung window intensities scaled to [0, 1].
Histology tile: 64 x 64 RGB, haematoxylin and eosin like colour model.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter

from medfusion.utils import neg, sub

CT_SIZE = 128
WSI_SIZE = 64
PIXEL_SPACING_MM = 2.6


def _num(row: dict, key: str, default: float):
    value = row.get(key, default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return default if value != value else value


def _grid(size: int):
    coords = (np.arange(size) + 0.5) / size
    return np.meshgrid(coords, coords)


def _ellipse(xx, yy, cx, cy, rx, ry):
    return (np.subtract(xx, cx) / rx) ** 2 + (np.subtract(yy, cy) / ry) ** 2 <= 1.0


def render_ct(row: dict):
    """Render a lung window axial key image for one study row."""
    rng = np.random.default_rng(int(row["ct_image_seed"]))
    xx, yy = _grid(CT_SIZE)
    image = np.zeros((CT_SIZE, CT_SIZE), dtype=np.float32)
    body = _ellipse(xx, yy, 0.5, 0.52, 0.46 + rng.normal(0, 0.01), 0.36 + rng.normal(0, 0.01))
    image[body] = 0.62
    # Lungs: two ellipses with patient specific asymmetry
    lung_right = _ellipse(xx, yy, 0.31, 0.5, 0.15 + rng.normal(0, 0.008), 0.27 + rng.normal(0, 0.01))
    lung_left = _ellipse(xx, yy, 0.69, 0.5, 0.14 + rng.normal(0, 0.008), 0.27 + rng.normal(0, 0.01))
    lungs = lung_right | lung_left
    image[lungs] = 0.08
    # Mediastinum, spine and aorta
    image[_ellipse(xx, yy, 0.5, 0.82, 0.05, 0.05)] = 0.95
    image[_ellipse(xx, yy, 0.55, 0.47, 0.04, 0.04)] = 0.7
    # Vascular markings inside lungs
    vessels = np.zeros_like(image)
    for _ in range(int(rng.integers(26, 40))):
        side = 0.31 if rng.random() < 0.5 else 0.69
        vx = side + rng.normal(0, 0.07)
        vy = 0.5 + rng.normal(0, 0.14)
        vessels += np.exp(neg(((np.subtract(xx, vx)) ** 2 + (np.subtract(yy, vy)) ** 2) / (2 * (0.006 + rng.random() * 0.004) ** 2)))
    image = image + 0.28 * np.clip(vessels, 0, 1) * lungs
    # Emphysema: low attenuation patches
    severity = {"none": 0, "mild": 8, "moderate": 18, "severe": 32}.get(row.get("emphysema_severity", "none"), 0)
    for _ in range(severity):
        side = 0.31 if rng.random() < 0.5 else 0.69
        ex, ey = side + rng.normal(0, 0.07), 0.45 + rng.normal(0, 0.12)
        patch = _ellipse(xx, yy, ex, ey, 0.012 + rng.random() * 0.02, 0.012 + rng.random() * 0.02) & lungs
        image[patch] = 0.0
    # Mediastinal lymph nodes
    if int(row.get("mediastinal_lymphadenopathy", 0)):
        for _ in range(int(rng.integers(1, 3))):
            nx, ny = 0.5 + rng.normal(0, 0.03), 0.42 + rng.normal(0, 0.04)
            image[_ellipse(xx, yy, nx, ny, 0.028, 0.024)] = 0.9
    # Dominant nodule and satellites
    if int(row.get("nodule_present", 0)):
        _draw_nodule(image, xx, yy, rng, row)
        for _ in range(max(0, sub(int(row.get("nodule_count", 1)), 1))):
            side = 0.31 if rng.random() < 0.5 else 0.69
            sx, sy = side + rng.normal(0, 0.06), 0.5 + rng.normal(0, 0.12)
            r = (3.0 / PIXEL_SPACING_MM) / CT_SIZE / 2
            blob = np.exp(neg(((np.subtract(xx, sx)) ** 2 + (np.subtract(yy, sy)) ** 2) / (2 * r ** 2)))
            image = np.maximum(image, 0.55 * blob * lungs)
    image = gaussian_filter(image, 0.6)
    dose = float(row.get("ctdivol_mgy", 2.0))
    noise_sigma = 0.028 / math.sqrt(max(dose, 0.5)) + 0.004
    image = image + rng.normal(0, noise_sigma, image.shape).astype(np.float32)
    return np.clip(image, 0, 1).astype(np.float32)


def _draw_nodule(image, xx, yy, rng, row):
    cx, cy = float(row["nodule_x_norm"]), float(row["nodule_y_norm"])
    diameter = float(row["nodule_diameter_mm"])
    radius = max(diameter / PIXEL_SPACING_MM / 2.0, 0.9) / CT_SIZE
    dist = np.sqrt(np.subtract(xx, cx) ** 2 + np.subtract(yy, cy) ** 2)
    angle = np.arctan2(np.subtract(yy, cy), np.subtract(xx, cx))
    margin = row.get("margin", "smooth")
    ntype = row.get("nodule_type", "solid")
    if margin == "lobulated":
        radius_field = radius * (1.0 + 0.18 * np.sin(3 * angle + rng.random() * 6.28))
    else:
        radius_field = radius * np.ones_like(dist)
    soft = np.clip(sub(1.0, sub(dist, radius_field) / (0.6 / CT_SIZE)), 0, 1)
    if margin == "spiculated":
        n_spikes = int(rng.integers(6, 11))
        for k in range(n_spikes):
            theta = 2 * math.pi * k / n_spikes + rng.normal(0, 0.2)
            length = radius * (1.6 + rng.random() * 1.4)
            along = (np.subtract(xx, cx)) * math.cos(theta) + (np.subtract(yy, cy)) * math.sin(theta)
            across = np.abs(sub((np.subtract(yy, cy)) * math.cos(theta), (np.subtract(xx, cx)) * math.sin(theta)))
            spike = (along > 0) & (along < length) & (across < 0.6 / CT_SIZE)
            soft = np.maximum(soft, spike * 0.75)
    if ntype == "solid":
        value = 0.72 * soft
    elif ntype == "part solid":
        solid_r = max(_num(row, "solid_component_mm", diameter / 2) / PIXEL_SPACING_MM / 2.0, 0.7) / CT_SIZE
        core = np.clip(sub(1.0, sub(dist, solid_r) / (0.6 / CT_SIZE)), 0, 1)
        value = np.maximum(0.33 * soft, 0.72 * core)
    elif ntype == "ground glass":
        value = 0.3 * soft + 0.03 * rng.random(image.shape) * soft
    else:
        value = np.maximum(0.7 * soft, 1.0 * (dist < radius * 0.55))
    np.maximum(image, value.astype(np.float32), out=image)


# Histology colour model in RGB (0 to 1)
EOSIN_BG = np.array([0.93, 0.78, 0.85], dtype=np.float32)
STROMA = np.array([0.88, 0.62, 0.74], dtype=np.float32)
HAEMATOXYLIN = np.array([0.30, 0.18, 0.52], dtype=np.float32)
KERATIN = np.array([0.95, 0.45, 0.55], dtype=np.float32)


def render_wsi(row: dict):
    """Render a 64 x 64 H and E like tile reflecting the histological diagnosis."""
    seed = row.get("wsi_image_seed")
    if seed is None or seed != seed:
        return np.zeros((3, WSI_SIZE, WSI_SIZE), dtype=np.float32)
    rng = np.random.default_rng(int(seed))
    xx, yy = _grid(WSI_SIZE)
    subtype = row.get("histology_subtype", "granuloma")
    tile = np.ones((WSI_SIZE, WSI_SIZE, 3), dtype=np.float32) * EOSIN_BG
    stroma = gaussian_filter(rng.random((WSI_SIZE, WSI_SIZE)), 3.0)
    tile = tile + (stroma[..., None] > 0.5) * np.subtract(STROMA, EOSIN_BG) * 0.8
    cellularity = _num(row, "tumour_cellularity_pct", 20.0)
    pleo = _num(row, "nuclear_pleomorphism_score", 1.0)
    n_nuclei = int(30 + cellularity * 1.5)
    base_r = {"small cell carcinoma": 0.010, "carcinoid tumour": 0.012}.get(subtype, 0.016)
    if subtype in ("granuloma", "organising pneumonia", "hamartoma", "nondiagnostic sample"):
        n_nuclei = int(n_nuclei * 0.6)
    nucleus_map = np.zeros((WSI_SIZE, WSI_SIZE), dtype=np.float32)
    centres = rng.random((n_nuclei, 2))
    if subtype == "adenocarcinoma":
        # Arrange nuclei around glandular lumens
        n_glands = int(rng.integers(5, 9))
        gland_c = rng.random((n_glands, 2)) * 0.7 + 0.15
        for g in range(n_glands):
            lumen = _ellipse(xx, yy, gland_c[g, 0], gland_c[g, 1], 0.055, 0.045)
            tile[lumen] = np.array([0.98, 0.94, 0.96], dtype=np.float32)
        idx = rng.integers(0, n_glands, n_nuclei)
        theta = rng.random(n_nuclei) * 2 * math.pi
        ring = gland_c[idx] + np.stack([np.cos(theta) * 0.065, np.sin(theta) * 0.055], axis=1)
        keep = rng.random(n_nuclei) < 0.7
        centres = np.where(keep[:, None], ring, centres)
    elif subtype == "squamous cell carcinoma":
        for _ in range(int(rng.integers(2, 4))):
            kx, ky = rng.random(2) * 0.7 + 0.15
            pearl = _ellipse(xx, yy, kx, ky, 0.09, 0.09)
            tile[pearl] = KERATIN
    elif subtype == "granuloma":
        gx, gy = rng.random(2) * 0.4 + 0.3
        tile[_ellipse(xx, yy, gx, gy, 0.22, 0.2)] = np.array([0.90, 0.70, 0.80], dtype=np.float32)
        if int(row.get("necrosis_present", 0)):
            tile[_ellipse(xx, yy, gx, gy, 0.1, 0.09)] = np.array([0.96, 0.85, 0.88], dtype=np.float32)
        theta = rng.random(n_nuclei) * 2 * math.pi
        rad = 0.2 + rng.normal(0, 0.03, n_nuclei)
        centres = np.stack([gx + np.cos(theta) * rad, gy + np.sin(theta) * rad], axis=1)
    elif subtype == "hamartoma":
        tile[_ellipse(xx, yy, 0.5, 0.5, 0.3, 0.25)] = np.array([0.75, 0.75, 0.88], dtype=np.float32)
    elif int(row.get("necrosis_present", 0)):
        nx, ny = rng.random(2) * 0.6 + 0.2
        tile[_ellipse(xx, yy, nx, ny, 0.12, 0.1)] = np.array([0.97, 0.88, 0.9], dtype=np.float32)
    for cx, cy in centres:
        r = base_r * (1 + (pleo + rng.normal(0, 0.15)) * 0.22 * rng.random())
        nucleus_map += np.exp(neg((np.subtract(xx, cx) ** 2 + np.subtract(yy, cy) ** 2) / (2 * r ** 2)))
    mitoses = _num(row, "mitoses_per_2mm2", 0.0)
    for _ in range(min(12, int(mitoses / 8))):
        mx, my = rng.random(2)
        nucleus_map += 1.6 * np.exp(neg((np.subtract(xx, mx) ** 2 + np.subtract(yy, my) ** 2) / (2 * 0.008 ** 2)))
    alpha = np.clip(nucleus_map, 0, 1)[..., None]
    tile = tile * sub(1.0, alpha) + HAEMATOXYLIN * alpha
    tile = tile + rng.normal(0, 0.02, tile.shape).astype(np.float32)
    return np.clip(tile, 0, 1).transpose(2, 0, 1).astype(np.float32)


def nodule_heatmap(row: dict, size: int = 32):
    """Gaussian target heatmap for localisation supervision."""
    if not int(row.get("nodule_present", 0)):
        return np.zeros((size, size), dtype=np.float32)
    xx, yy = _grid(size)
    cx, cy = float(row["nodule_x_norm"]), float(row["nodule_y_norm"])
    sigma = max(float(row["nodule_diameter_mm"]) / PIXEL_SPACING_MM / CT_SIZE / 2.0, 1.2 / size)
    heat = np.exp(neg((np.subtract(xx, cx) ** 2 + np.subtract(yy, cy) ** 2) / (2 * sigma ** 2)))
    return (heat / heat.max()).astype(np.float32)
