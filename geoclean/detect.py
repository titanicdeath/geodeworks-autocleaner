"""Imperfection detection and mode classification.

Colours vary per mineral, so everything is relative to the LOCAL background
(a large median filter), in Lab (dE ~ perceptual colour difference).

Mode classification uses frames of the same view in several lighting modes:
an imperfection is rendered at full contrast in its own mode and faded in the
others, so each pixel belongs to the mode where it contrasts most with the
background. Overlapping objects of different modes separate naturally.
With only a Survey frame, `survey_guess` classifies by texture/contrast
(full-contrast rock blocks are textured; faded objects are smooth).
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

MODE_NAMES = ('survey', 'bright', 'polarized')


@dataclass
class ContrastMap:
    x0: int                  # crop offset in the frame
    y0: int
    valid: np.ndarray        # bool, crop size
    dE: np.ndarray           # float32 colour difference from local background
    tex: np.ndarray          # float32 local texture (high-pass magnitude, smoothed)
    lab: np.ndarray          # float32 Lab of the crop
    bg_lab: np.ndarray       # float32 Lab of the local background


def scope_box(layout, pad=4):
    cx, cy, r = layout.scope
    w, h = layout.size
    return (max(0, int(cx - r - pad)), max(0, int(cy - r - pad)),
            min(w, int(cx + r + pad)), min(h, int(cy + r + pad)))


def contrast_map(frame, layout, shrink=0.94, k_bg=51):
    x0, y0, x1, y1 = scope_box(layout)
    crop = frame[y0:y1, x0:x1]
    valid = layout.valid_mask(shrink)[y0:y1, x0:x1]
    filled = crop.copy()
    filled[~valid] = np.median(crop[valid], axis=0).astype(np.uint8)
    # Background: large median at half resolution (objects are <= ~70 px, lines thin).
    small = cv2.resize(filled, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.medianBlur(small, k_bg), (crop.shape[1], crop.shape[0]),
                    interpolation=cv2.INTER_LINEAR)
    lab = cv2.cvtColor(filled, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(bg, cv2.COLOR_BGR2LAB).astype(np.float32)
    dE = np.linalg.norm(lab - bg_lab, axis=2)
    dE[~valid] = 0
    g = cv2.cvtColor(filled, cv2.COLOR_BGR2GRAY).astype(np.float32)
    tex = cv2.GaussianBlur(np.abs(g - cv2.GaussianBlur(g, (0, 0), 1.5)), (0, 0), 2)
    tex[~valid] = 0
    return ContrastMap(x0, y0, valid, dE.astype(np.float32), tex, lab, bg_lab)


@dataclass
class Obj:
    mode: str                # 'survey'/'bright'/'polarized', or '?' if unclassified
    area: int
    centroid: tuple          # frame (screen) px
    click: tuple             # interior point: deepest pixel of the object mask
    bbox: tuple              # frame px (x, y, w, h)
    contrast: dict           # mean dE of the object's pixels in each available mode
    dominance: float         # own-mode contrast / best other-mode contrast
    texture: float           # mean texture in the Survey frame
    elongation: float        # long / short side of the min-area rectangle
    partial: bool            # touches the edge of the usable view
    survey_guess: str = ''   # 'full' / 'faded' from the Survey frame alone
    method: str = ''         # how the mode was decided: 'contrast' or 'alpha' (transparency model)
    alpha_scores: dict = None
    mask: np.ndarray = field(default=None, repr=False)   # crop of the object mask (bbox)


def fill_holes(m):
    """Fill enclosed holes (rock texture can leave background-coloured pixels inside)."""
    pad = np.pad(m.astype(np.uint8), 1)
    flood = pad.copy()
    cv2.floodFill(flood, None, (0, 0), 2)
    return (pad == 1)[1:-1, 1:-1] | (flood == 0)[1:-1, 1:-1]


def solidity(sub):
    m = fill_holes(sub).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.contourArea(cv2.convexHull(max(cnts, key=cv2.contourArea))) if cnts else 0
    return float(m.sum() / hull) if hull > 0 else 1.0


def split_touching(sub, typical_area=None, min_peak=10.0, min_sep=28, rel_peak=0.7, max_solidity=0.88):
    """Split touching blocks inside one component crop: one watershed region per
    clearly separate distance-transform peak.

    Only blobs that look like several blocks are split: clearly non-convex (single
    blocks measured 0.97-1.01 solidity; clusters 0.2-0.87), or much larger than this
    frame's typical block. Blocks are compact (peak ~ half their side); lines are
    thin (peak ~3-5 px), so lines never split. Returns a list of bool masks."""
    m = fill_holes(sub).astype(np.uint8)
    dist = cv2.distanceTransform(np.pad(m, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    if dist.max() < min_peak:
        return [sub]
    big = typical_area is not None and m.sum() > 1.6 * typical_area
    if solidity(sub) > max_solidity and not big:
        return [sub]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * min_sep + 1, 2 * min_sep + 1))
    peaks = (dist >= cv2.dilate(dist, k) - 1e-3) & (dist >= max(min_peak, (0.5 if big else rel_peak) * dist.max()))
    n, markers, _, cents = cv2.connectedComponentsWithStats(peaks.astype(np.uint8))
    if n <= 2:
        return [sub]
    # Each pixel goes to the block whose centre it is most "inside": distance to
    # the peak minus that peak's radius (a power diagram). OpenCV's watershed on a
    # smooth distance map cut live blocks into thin slices instead.
    pk = np.array(cents[1:])                                   # (k, 2) x, y
    rk = np.array([dist[int(round(c[1])), int(round(c[0]))] for c in pk])
    # Keep only peaks that are separate blocks: biggest first, each at least 1.2x
    # the larger radius from every kept one, and no more than the area allows.
    cap = int(round(m.sum() / typical_area)) + 1 if typical_area else len(pk)
    keep = []
    for i in np.argsort(-rk):
        if len(keep) >= cap:
            break
        if all(np.hypot(*(pk[i] - pk[j])) >= 1.2 * max(rk[i], rk[j]) for j in keep):
            keep.append(i)
    if len(keep) < 2:
        return [sub]
    pk, rk = pk[keep], rk[keep]
    ys, xs = np.nonzero(sub)
    score = np.hypot(xs[:, None] - pk[None, :, 0], ys[:, None] - pk[None, :, 1]) - rk[None, :]
    lab = score.argmin(axis=1)
    parts = []
    for i in range(len(pk)):
        p = np.zeros_like(sub)
        p[ys[lab == i], xs[lab == i]] = True
        if p.any():
            parts.append(p)
    return parts


def _components(mask, min_area, split=True):
    """Yield (mask_crop, (x, y, w, h, area), centroid) in the coordinates of `mask`."""
    # Close first: low-contrast rock texture can break one block into fragments.
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            comps.append((x, y, labels[y:y + h, x:x + w] == i, int(area)))
    # Typical single block in this frame: compact, convex, block-sized blobs.
    singles = [a for _, _, s, a in comps if a >= 1500 and solidity(s) >= 0.95]
    typical = float(np.median(singles)) if len(singles) >= 2 else None
    out = []
    for x, y, sub, area in comps:
        for p in (split_touching(sub, typical) if split else [sub]):
            ys, xs = np.nonzero(p)
            if len(xs) < min_area:
                continue
            px, py = xs.min(), ys.min()
            crop = p[py:ys.max() + 1, px:xs.max() + 1]
            out.append((crop, (x + px, y + py, crop.shape[1], crop.shape[0], len(xs)),
                        (x + xs.mean(), y + ys.mean())))
    return out


def _absorb_fragments(label, anyobj, n_modes, max_area=250):
    """Small same-label islands inside a larger object of another label take that
    object's label (edge flicker, not a separate object)."""
    label = label.copy()
    for m in range(n_modes):
        mask = (anyobj & (label == m)).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] > max_area:
                continue
            x, y, w, h = st[i, :4]
            x0, y0 = max(0, x - 2), max(0, y - 2)
            isl = lab[y0:y + h + 2, x0:x + w + 2] == i
            ring = cv2.dilate(isl.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~isl
            around = label[y0:y + h + 2, x0:x + w + 2][ring & anyobj[y0:y + h + 2, x0:x + w + 2]]
            around = around[around != m]
            if around.size >= 0.3 * ring.sum():
                label[y0:y + h + 2, x0:x + w + 2][isl] = np.bincount(around).argmax()
    return label


def alpha_mode(obs, max_fade=0.85):
    """Which mode shows this object at full opacity?

    obs: {mode: (mean Lab of the object's pixels, mean Lab of the local background
    under them)}. Faded imperfections are drawn semi-transparent (Roblox part
    transparency = alpha blending), so if mode h is the object's own mode, then in
    every other mode m:  I_m ~= bg_m + f * (I_h - bg_m)  with 0 <= f < 1.
    Returns (best_mode, {mode: score}); lower score = better fit."""
    scores = {}
    for h, (Ih, _) in obs.items():
        s = 0.0
        for m, (Im, bm) in obs.items():
            if m == h:
                continue
            d = Ih - bm
            nd = float(np.dot(d, d))
            if nd < 4.0:                         # own colour ~ that mode's background: no info
                s += 5.0
                continue
            f = float(np.dot(Im - bm, d) / nd)
            resid = float(np.linalg.norm(Im - bm - f * d))
            s += resid + 40.0 * max(0.0, f - max_fade) + 40.0 * max(0.0, -f)
        scores[h] = round(s, 2)
    best = min(scores, key=scores.get)
    return best, scores


def thin(mask, max_iter=100):
    """Zhang-Suen thinning to a 1-px skeleton (numpy, fine for object-sized masks)."""
    img = np.pad(mask.astype(np.uint8), 1)
    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            P = [img[1:-1, 1:-1], img[:-2, 1:-1], img[:-2, 2:], img[1:-1, 2:], img[2:, 2:],
                 img[2:, 1:-1], img[2:, :-2], img[1:-1, :-2], img[:-2, :-2]]
            p1, p2, p3, p4, p5, p6, p7, p8, p9 = P
            B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            A = sum(((seq[k] == 0) & (seq[k + 1] == 1)).astype(np.uint8) for k in range(8))
            if step == 0:
                c = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                c = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            rm = (p1 == 1) & (B >= 2) & (B <= 6) & (A == 1) & c
            if rm.any():
                img[1:-1, 1:-1][rm] = 0
                changed = True
        if not changed:
            break
    return img[1:-1, 1:-1].astype(bool)


def is_line_like(mask, max_half_width=15.0, min_area=250, min_length=80):
    """Thin and long. Live: lines are ~18 px wide (distance-to-edge max 9-13 px),
    rock blocks >= 20 even with texture holes; slivers of blocks cut off by the
    view edge are thin but short, hence the skeleton-length test."""
    if mask.sum() < min_area:
        return False
    dist = cv2.distanceTransform(np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2, 5)
    if float(dist.max()) > max_half_width:
        return False
    # A faded block can be detected as just its outline, which is thin too, but it
    # is closed: filled in, it is thick. Lines are open shapes and stay thin.
    filled = cv2.distanceTransform(np.pad(fill_holes(mask).astype(np.uint8), 1), cv2.DIST_L2, 5)
    if float(filled.max()) > max_half_width:
        return False
    return int(thin(mask).sum()) >= min_length


def line_segments(mask, min_len=28, merge_dist=10.0):
    """Straight segments of a (possibly jointed) line object: [(p0, p1), ...] in
    mask coordinates. Live, a bent line is several separate imperfections, one per
    straight segment (the Polarized minimap shows one dot per segment)."""
    sk = thin(mask).astype(np.uint8) * 255
    found = cv2.HoughLinesP(sk, 1, np.pi / 180, threshold=12, minLineLength=min_len, maxLineGap=6)
    if found is None:
        return []
    segs = [(np.array(s[0][:2], float), np.array(s[0][2:], float)) for s in found]
    segs.sort(key=lambda s: -np.linalg.norm(s[1] - s[0]))
    kept = []

    def near(pt, seg):
        a, b = seg
        d = b - a
        t = np.clip(np.dot(pt - a, d) / max(1e-9, np.dot(d, d)), 0, 1)
        return np.linalg.norm(pt - (a + t * d)) < merge_dist
    for a, b in segs:
        samples = [a + (b - a) * t for t in np.linspace(0, 1, 9)]
        covered = sum(any(near(p, k) for k in kept) for p in samples)
        if covered < 0.6 * len(samples):
            kept.append((a, b))
    return _fuse_collinear(kept, merge_dist)


def _fuse_collinear(segs, dist_tol, angle_tol=12.0, gap=15.0):
    """Join segments on the same straight line that overlap or nearly touch
    (Hough and misaligned views return pieces of one straight part)."""
    segs = [(np.asarray(a, float), np.asarray(b, float)) for a, b in segs]
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                a1, b1 = segs[i]
                a2, b2 = segs[j]
                d1, d2 = b1 - a1, b2 - a2
                ang = np.degrees(np.arccos(min(1.0, abs(np.dot(d1, d2)) / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-9))))
                if ang > angle_tol:
                    continue
                u = d1 / (np.linalg.norm(d1) + 1e-9)
                nrm = np.array([-u[1], u[0]])
                if max(abs(np.dot(a2 - a1, nrm)), abs(np.dot(b2 - a1, nrm))) > dist_tol:
                    continue
                t = sorted([0.0, float(np.dot(d1, u)), float(np.dot(a2 - a1, u)), float(np.dot(b2 - a1, u))])
                s1 = sorted([0.0, float(np.dot(d1, u))])
                s2 = sorted([float(np.dot(a2 - a1, u)), float(np.dot(b2 - a1, u))])
                if s2[0] > s1[1] + gap or s1[0] > s2[1] + gap:
                    continue
                segs[i] = (a1 + u * t[0], a1 + u * t[-1])
                segs.pop(j)
                changed = True
                break
            if changed:
                break
    return segs


def is_speck(area, contrast, elongation):
    """Small bright squares (~16 px wide, very high contrast). Live they carry no
    minimap dot in any mode, so they may not be cleanable imperfections."""
    return area < 500 and contrast > 60 and elongation < 2.2


def survey_guess(texture, contrast):
    """Full-contrast Survey objects (rock blocks) are textured; faded ones are smooth."""
    return 'full' if texture >= 4.0 and contrast >= 18 else 'faded'


def detect_objects(maps, t=10.0, min_area=60, min_dominance=1.15, min_mean_contrast=12.0):
    """maps: {mode: ContrastMap} of the SAME view. Returns [Obj] in frame coords."""
    modes = [m for m in MODE_NAMES if m in maps]
    ref = maps[modes[0]]
    D = np.stack([maps[m].dE for m in modes])
    valid = ref.valid
    anyobj = (D.max(axis=0) > t) & valid
    # Per-pixel mode = where the pixel contrasts most. Smooth first so antialiased
    # object edges don't flicker between modes and leave fragments.
    Ds = np.stack([cv2.GaussianBlur(d, (0, 0), 2.5) for d in D]) if len(modes) > 1 else D
    label = Ds.argmax(axis=0)
    if len(modes) > 1:
        label = _absorb_fragments(label, anyobj, len(modes))
    inner = cv2.erode(valid.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    survey_tex = maps['survey'].tex if 'survey' in maps else None
    objs = []
    for mi, m in enumerate(modes):
        mask_m = anyobj & (label == mi) if len(modes) > 1 else anyobj
        for sub, st, c in _components(mask_m, min_area):
            x, y, w, h, area = (int(v) for v in st)
            win = (slice(y, y + h), slice(x, x + w))
            cont = {mm: round(float(maps[mm].dE[win][sub].mean()), 2) for mm in modes}
            mode, method, a_scores = m, 'contrast', None
            if len(modes) > 1:
                obs = {mm: (maps[mm].lab[win][sub].mean(axis=0), maps[mm].bg_lab[win][sub].mean(axis=0))
                       for mm in modes}
                a_mode, a_scores = alpha_mode(obs)
                ranked = sorted(a_scores.values())
                # Use the transparency model when it clearly prefers one mode.
                if a_mode != m and ranked[0] < 0.6 * ranked[1]:
                    mode, method = a_mode, 'alpha'
            own = cont[mode]
            if own < min_mean_contrast:
                continue          # background mottling near the pixel threshold, not an object
            others = [cont[o] for o in modes if o != mode]
            dom = own / max(1e-6, max(others)) if others else 1.0
            # Click point: deepest pixel of the (hole-filled) object, so on a line it
            # lies on the line and on a block well inside it.
            solid = fill_holes(sub).astype(np.uint8)
            dist = cv2.distanceTransform(np.pad(solid, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
            dist[~sub] = 0
            iy, ix = np.unravel_index(int(np.argmax(dist)), dist.shape)
            pts = np.column_stack(np.nonzero(sub))[:, ::-1].astype(np.float32)
            (_, _), (rw, rh), _ = cv2.minAreaRect(pts) if len(pts) >= 3 else ((0, 0), (1, 1), 0)
            elong = float(max(rw, rh) / max(1.0, min(rw, rh)))
            tex = float(survey_tex[win][sub].mean()) if survey_tex is not None else float('nan')
            partial = bool((sub & ~inner[win]).any())
            o = Obj(mode=mode if len(modes) > 1 else '?', area=area, method=method, alpha_scores=a_scores,
                    centroid=(float(c[0] + ref.x0), float(c[1] + ref.y0)),
                    click=(float(x + ix + ref.x0), float(y + iy + ref.y0)),
                    bbox=(x + ref.x0, y + ref.y0, w, h), contrast=cont, dominance=round(float(dom), 3),
                    texture=round(tex, 2), elongation=round(elong, 2), partial=partial, mask=sub)
            if survey_tex is not None:
                o.survey_guess = survey_guess(tex, cont['survey'])
            objs.append(o)
    return objs
