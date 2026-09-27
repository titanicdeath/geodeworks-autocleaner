"""Registration: how far the slide content moved between two frames.

Small shifts: phase correlation (vision.phase_shift). Large shifts: a prior
(from the minimap view ring, ~±15 px, or from the pan model) plus template
matching inside a small search window. The template is the largest square
that shows slide in both frames, so the fixed rim/HUD/vignette never
dominates the match.

Sign convention everywhere: shift (dx, dy) > 0 means content moved right/down
on screen, i.e. the view moved left/up over the slide.
"""
import cv2
import numpy as np

from . import vision as v


def highpass(frame, sigma=8):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return g - cv2.GaussianBlur(g, (0, 0), sigma)


def _shift_mask(mask, px, py):
    """out[y, x] = mask[y + py, x + px] (False outside)."""
    h, w = mask.shape
    out = np.zeros_like(mask)
    y0, y1 = max(0, -py), min(h, h - py)
    x0, x1 = max(0, -px), min(w, w - px)
    if y0 < y1 and x0 < x1:
        out[y0:y1, x0:x1] = mask[y0 + py:y1 + py, x0 + px:x1 + px]
    return out


def _subpixel(res, x, y):
    def para(a, b, c):
        d = a - 2 * b + c
        return 0.0 if abs(d) < 1e-9 else 0.5 * (a - c) / d
    dx = para(res[y, x - 1], res[y, x], res[y, x + 1]) if 0 < x < res.shape[1] - 1 else 0.0
    dy = para(res[y - 1, x], res[y, x], res[y + 1, x]) if 0 < y < res.shape[0] - 1 else 0.0
    return float(np.clip(dx, -0.5, 0.5)), float(np.clip(dy, -0.5, 0.5))


def register(a, b, layout, prior=(0, 0), search=40, shrink=0.92, max_half=260, min_half=50,
             hp=None):
    """Content shift a -> b near `prior`. Returns (dx, dy, score).

    score is the normalized cross-correlation peak (1 = perfect). Below ~0.4 the
    match is unreliable (featureless view, or the prior was off by > search).
    """
    px, py = int(round(prior[0])), int(round(prior[1]))
    valid = layout.valid_mask(shrink)
    both = (valid & _shift_mask(valid, px, py)).astype(np.uint8)
    dist = cv2.distanceTransform(both, cv2.DIST_C, 3)
    cy, cx = np.unravel_index(int(np.argmax(dist)), dist.shape)
    half = int(min(dist[cy, cx] - search - 2, max_half))
    if half < min_half:
        return 0.0, 0.0, 0.0
    ha = hp[0] if hp else highpass(a)
    hb = hp[1] if hp else highpass(b)
    tmpl = ha[cy - half:cy + half, cx - half:cx + half]
    if tmpl.std() < 0.3:
        return float(px), float(py), 0.0
    y0, x0 = cy + py - half - search, cx + px - half - search
    region = hb[y0:y0 + 2 * (half + search), x0:x0 + 2 * (half + search)]
    res = cv2.matchTemplate(region, tmpl, cv2.TM_CCOEFF_NORMED)
    _, peak, _, (mx, my) = cv2.minMaxLoc(res)
    sx, sy = _subpixel(res, mx, my)
    return float(px + mx - search + sx), float(py + my - search + sy), float(peak)


def ring_center(frame, layout, ring_frac=None):
    if not layout.minimap:
        return None
    ring = v.minimap_features(frame, layout.minimap, ring_frac or v.RING_R_FRAC).ring
    return None if ring is None else np.array(ring[:2], float)


def scale_vec(ring_scale):
    """Minimap scale as an (x, y) pair; older calibrations stored one number."""
    s = np.asarray(ring_scale, float)
    return np.array([float(s), float(s)]) if s.ndim == 0 else s


def ring_agreement_tol(ring_est):
    # The ring is drawn on a 0.5 px grid (+-0.25 ring px ~ +-6.5 slide px per reading,
    # so ~13 px worst case for a difference), plus a few % for scale error.
    return 20.0 + 0.04 * float(np.linalg.norm(ring_est))


def measure_shift(a, b, layout, ring_a=None, ring_b=None, ring_scale=None, predicted=None,
                  search=40, min_score=0.45):
    """Best estimate of the content shift a -> b, using every cue available.

    Priors are tried in order: minimap ring delta (independent, coarse), the
    model prediction, zero (phase correlation). The highest-scoring match that
    agrees with the ring (when the ring is known) wins; if none is reliable the
    ring estimate itself is returned with source='minimap'.
    """
    ring_est = None
    if ring_a is not None and ring_b is not None and ring_scale is not None:
        ring_est = -scale_vec(ring_scale) * (np.asarray(ring_b) - np.asarray(ring_a))
    priors = []
    if ring_est is not None:
        priors.append(('ring', ring_est))
    if predicted is not None:
        priors.append(('model', np.asarray(predicted, float)))
    priors.append(('zero', np.zeros(2)))
    hp = (highpass(a), highpass(b))
    best = None
    tried = []
    for name, p in priors:
        if any(np.linalg.norm(p - q) < search * 0.5 for q in tried):
            continue
        tried.append(p)
        dx, dy, score = register(a, b, layout, p, search=search, hp=hp)
        est = np.array([dx, dy])
        agrees = ring_est is None or np.linalg.norm(est - ring_est) < ring_agreement_tol(ring_est)
        if score >= min_score and agrees and (best is None or score > best[1]):
            best = (est, score, name)
    if best is None and ring_est is None and predicted is not None:
        # No ring to lean on: one wide search around the model, held to a stricter bar.
        dx, dy, score = register(a, b, layout, predicted, search=3 * search, hp=hp)
        if score >= max(min_score, 0.6):
            best = (np.array([dx, dy]), score, 'wide')
    if best is not None:
        return best[0], best[1], 'image'
    if ring_est is not None:
        return ring_est, 0.0, 'minimap'
    return np.zeros(2), 0.0, 'none'
