"""Screen-only vision. All images are BGR uint8, coordinates are image pixels.

Nothing here assumes fixed mineral colors: backgrounds are estimated per frame.
"""
from dataclasses import dataclass, asdict

import cv2
import numpy as np


# --------------------------------------------------------------------------
# Layout: microscope circle, minimap circle, UI exclusion
# --------------------------------------------------------------------------

_MASKS = {}


@dataclass
class Layout:
    scope: tuple          # (cx, cy, r)
    minimap: tuple        # (mx, my, mr) or None
    size: tuple           # (w, h)

    # UI bands measured from the samples, as fractions of the scope radius.
    HUD_TOP = -0.80       # HUD (purity/money/progress bar) sits above cy-0.80r
    BUTTONS_TOP = 0.84    # mode buttons start around cy+0.88r

    def as_dict(self):
        return dict(scope=list(map(float, self.scope)),
                    minimap=list(map(float, self.minimap)) if self.minimap else None,
                    size=list(self.size))

    def valid_mask(self, shrink=0.96):
        """Pixels of the slide not covered by the rim, HUD or mode buttons."""
        key = (tuple(self.scope), tuple(self.size), shrink)
        if key not in _MASKS:
            w, h = self.size
            cx, cy, r = self.scope
            yy, xx = np.ogrid[:h, :w]
            m = (xx - cx) ** 2 + (yy - cy) ** 2 < (r * shrink) ** 2
            m &= (yy > cy + self.HUD_TOP * r) & (yy < cy + self.BUTTONS_TOP * r)
            m.flags.writeable = False
            _MASKS.clear() if len(_MASKS) > 16 else None
            _MASKS[key] = m
        return _MASKS[key]

    def is_clickable_area(self, x, y, margin=0.9):
        cx, cy, r = self.scope
        return (np.hypot(x - cx, y - cy) < r * margin and
                cy + (self.HUD_TOP + 0.03) * r < y < cy + (self.BUTTONS_TOP - 0.03) * r)

    def hud_box(self):
        """Purity row + progress bar (excludes the event banner and its countdown)."""
        cx, cy, r = self.scope
        return (int(cx - 0.45 * r), int(cy - 0.95 * r), int(cx + 0.45 * r), int(cy - 0.80 * r))

    def inner_square(self, frac=0.60):
        """Square fully inside the usable slide, for registration."""
        cx, cy, r = self.scope
        half = int(r * frac)
        half = min(half, int(cy - (cy + (self.HUD_TOP + 0.05) * r)))  # stay below HUD
        return (int(cx - half), int(cy - half), int(cx + half), int(cy + half))


def _fit_circle(pts):
    x, y = pts[:, 0].astype(float), pts[:, 1].astype(float)
    a = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
    cx, cy, c = np.linalg.lstsq(a, x * x + y * y, rcond=None)[0]
    return cx, cy, float(np.sqrt(max(c + cx * cx + cy * cy, 0)))


def _robust_circle(pts, init, tol=4.0, rounds=4):
    cx, cy, r = init
    for _ in range(rounds):
        d = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
        keep = pts[d < max(tol, np.percentile(d, 40))]
        if len(keep) < 20:
            break
        cx, cy, r = _fit_circle(keep)
    return cx, cy, r


def find_scope(frame):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = (gray > 45).astype('uint8')
    mask[:, : int(w * 0.12)] = 0
    mask[:, int(w * 0.88):] = 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        raise ValueError('No bright microscope disk found.')
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = labels == big
    if stats[big, cv2.CC_STAT_AREA] < 0.1 * w * h:
        raise ValueError('Microscope disk too small; is the microscope open?')
    # Left/right edge of each row in the middle band: the HUD (top) and mode
    # buttons (bottom) never touch these, so they are clean rim points.
    top, height = stats[big, cv2.CC_STAT_TOP], stats[big, cv2.CC_STAT_HEIGHT]
    pts = []
    for y in range(int(top + 0.2 * height), int(top + 0.8 * height), 2):
        xs = np.flatnonzero(comp[y])
        if len(xs) > 0.2 * w:
            pts += [(xs[0], y), (xs[-1], y)]
    pts = np.array(pts, float)
    if len(pts) < 40:
        raise ValueError('Could not trace the microscope rim.')
    cx, cy, r = _robust_circle(pts, _fit_circle(pts), tol=2.0)
    if not (0.25 * h < r < 0.6 * h):
        raise ValueError('Microscope view not found (open the microscope).')
    # The microscope sits on black: the band just outside its left/right rim is dark.
    ys = np.arange(int(cy - 0.5 * r), int(cy + 0.5 * r), 4)
    xs = np.concatenate([np.full_like(ys, int(cx - 1.06 * r)), np.full_like(ys, int(cx + 1.06 * r))])
    ys = np.concatenate([ys, ys])
    ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    if ok.sum() < 20 or np.mean(gray[ys[ok], xs[ok]] < 30) < 0.9:
        raise ValueError('Microscope view not found (open the microscope).')
    return float(cx), float(cy), float(r)


def find_minimap(frame):
    """Largest roughly circular non-black blob in the top-right corner."""
    h, w = frame.shape[:2]
    x0, y0 = int(w * 0.85), 0
    crop = frame[y0: int(h * 0.30), x0:]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return _circle_blob(gray, x0, y0, min_r=25, max_r=0.12 * h)


def _circle_blob(gray, x0=0, y0=0, min_r=10, max_r=1e9):
    best = None
    for thresh in (40, 55, 70):
        contours, _ = cv2.findContours((gray > thresh).astype('uint8'), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        for c in contours:
            (mx, my), mr = cv2.minEnclosingCircle(c)
            if not (min_r < mr < max_r):
                continue
            fill = cv2.contourArea(c) / (np.pi * mr * mr)
            if fill > 0.75 and (best is None or mr > best[2]):
                best = (float(mx + x0), float(my + y0), float(mr))
        if best:
            return best
    return None


def detect_layout(frame):
    h, w = frame.shape[:2]
    return Layout(scope=find_scope(frame), minimap=find_minimap(frame), size=(w, h))


# --------------------------------------------------------------------------
# Minimap: remaining-imperfection dots and the current-view ring
# --------------------------------------------------------------------------

def disk(shape, circle, fraction=1.0):
    x, y, r = circle
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (xx - x) ** 2 + (yy - y) ** 2 < (r * fraction) ** 2


@dataclass
class MapInfo:
    dots: list            # [[x, y], ...] image coords
    ring: tuple           # (x, y, r) or None
    background: list      # Lab median


RING_R_FRAC = 0.3835      # view-ring radius / minimap radius (constant in every sample)
# The ring shrinks with the mode's zoom (live 2026-09-26: 19.26 / 14.15 / 10.12 px at
# minimap radius 49.8, i.e. 17.9/zoom + 1.4 px of line width); centre unchanged.
RING_R_FRAC_MODE = {'survey': RING_R_FRAC, 'bright': 0.2841, 'polarized': 0.2032}


def _coverage(pts, c, r, tol=2.0):
    d = np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1])
    near = np.abs(d - r) < tol
    ang = np.arctan2(pts[near, 1] - c[1], pts[near, 0] - c[0])
    return len(np.unique(np.floor((ang + np.pi) / (2 * np.pi) * 16))), int(near.sum())


def _fit_fixed_radius(pts, r, c0, iters=25):
    """Gauss-Newton for the centre of a circle of known radius r (robust to a short arc)."""
    c = np.array(c0, float)
    use = pts
    for i in range(iters):
        d = use - c
        n = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-6)
        res = n - r
        J = -d / n[:, None]
        step, *_ = np.linalg.lstsq(J, -res, rcond=None)
        c = c + step
        if i == 5:   # after settling, drop outliers (dots, antialiasing specks)
            dd = np.abs(np.hypot(*(pts - c).T) - r)
            use = pts[dd < 2.5] if (dd < 2.5).sum() >= 10 else pts
        if np.hypot(*step) < 1e-3 and i > 6:
            break
    res = np.abs(np.hypot(*(use - c).T) - r)
    return c, float(np.median(res)) if len(res) else 1e9


def find_ring(pts, r_expected):
    """Centre (x, y, r) of the view ring from its pale pixels, or None."""
    # The ring's radius never changes, so fitting only its centre with the radius
    # fixed is more precise than a free fit, and works on a partial arc too.
    free = _robust_circle(pts, _fit_circle(pts), tol=1.5)
    starts = [free[:2], pts.mean(axis=0)]
    best = None
    for s in starts:
        c, med = _fit_fixed_radius(pts, r_expected, s)
        cov, n = _coverage(pts, c, r_expected)
        if med < 1.2 and cov >= 4 and n >= 25 and (best is None or n > best[1]):
            best = (c, n)
    if best is None:
        return None
    return float(best[0][0]), float(best[0][1]), float(r_expected)


def minimap_features(frame, circle, ring_frac=RING_R_FRAC):
    """ring_frac: expected view-ring radius / minimap radius (see RING_R_FRAC_MODE)."""
    mx, my, mr = circle
    x0, y0 = max(0, int(mx - mr - 2)), max(0, int(my - mr - 2))
    crop = frame[y0: int(my + mr + 3), x0: int(mx + mr + 3)]
    local = (mx - x0, my - y0, mr)
    inside = disk(crop.shape, local, 0.92)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype('float32')
    bg = np.median(lab[inside], axis=0)
    contrast = np.linalg.norm(lab - bg, axis=2)
    # View ring: pale, unsaturated, and different from the map background. Near the
    # slide edges part of it runs off the minimap, so look almost to the outline and
    # accept a partial arc when it matches the ring's known radius.
    bright = (hsv[:, :, 1] < 60) & (hsv[:, :, 2] > 175) & (contrast > 15)
    ring = None
    # The ring is drawn in full even where it leaves the minimap (over black), so
    # search a wider crop. 1.3*mr stays clear of the close button's white "X".
    er = 1.3 * mr
    ex0, ey0 = max(0, int(mx - er)), max(0, int(my - er))
    wide = frame[ey0: int(my + er) + 1, ex0: int(mx + er) + 1]
    whsv = cv2.cvtColor(wide, cv2.COLOR_BGR2HSV)
    wlab = cv2.cvtColor(wide, cv2.COLOR_BGR2LAB).astype('float32')
    wpale = ((whsv[:, :, 1] < 60) & (whsv[:, :, 2] > 175) & (np.linalg.norm(wlab - bg, axis=2) > 15)
             & disk(wide.shape, (mx - ex0, my - ey0, er)))
    yy, xx = np.nonzero(wpale)
    if len(xx) >= 12:
        found = find_ring(np.column_stack([xx, yy]).astype(float), ring_frac * mr)
        if found is not None:
            ring = (float(found[0] + ex0), float(found[1] + ey0), float(found[2]))
    # Dot fills: differ from the background but are not the dark outlines, so
    # touching dots stay separate (each fill is enclosed by its own outline).
    # Dots are drawn on top of the view ring, so the ring does not need masking;
    # its pale antialiased edge is rejected by the saturation test.
    fg = (contrast > 22) & inside & ~bright & (hsv[:, :, 2] > 130) & (hsv[:, :, 1] >= 60)
    fg = fg.astype('uint8')
    n, labels, stats, centers = cv2.connectedComponentsWithStats(fg)
    # One dot's fill is a 5x5 disc = 21 px at minimap radius 49.8 (measured live).
    # Overlapping dots merge their fills, so count k = area / single, and place the
    # k dots by clustering the blob's pixels. Tiny specks (<6 px) are ring/edge
    # antialiasing, and a dot hidden entirely under another cannot be recovered.
    single = 21.0 * (mr / 49.8) ** 2
    # Thickness, not size, separates dots from ring slivers: live 2026-09-27 ~10
    # overlapping dots merged into one 31x23 px blob, which an old width limit
    # (0.45 mr) threw away - the run then believed Survey was empty.
    thick = cv2.distanceTransform(fg, cv2.DIST_L2, 3)
    dots = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 0.28 * single:
            continue
        if area > 2 * single and thick[labels == i].max() < 1.9 * (mr / 49.8):
            continue            # long thin arc: antialiased view-ring edge
        k = int(min(16, max(1, round(area / single))))
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if k == 1 and (min(bw, bh) < 3 or area < 0.3 * single or area / float(bw * bh) < 0.35):
            continue            # thin/sparse: a sliver of the view ring's antialiased edge
        if k == 1:
            dots.append([float(centers[i][0] + x0), float(centers[i][1] + y0)])
            continue
        ys, xs = np.nonzero(labels == i)
        pts = np.column_stack([xs, ys]).astype(np.float32)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        _, _, cent = cv2.kmeans(pts, k, None, crit, 5, cv2.KMEANS_PP_CENTERS)
        dots += [[float(c[0] + x0), float(c[1] + y0)] for c in cent]
    return MapInfo(dots, ring, bg.tolist())


# --------------------------------------------------------------------------
# Registration (how far the slide content moved between two frames)
# --------------------------------------------------------------------------

def _prep(frame, box):
    x0, y0, x1, y1 = box
    g = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype('float32')
    # Remove illumination/vignetting that is fixed to the view, not the slide.
    return g - cv2.GaussianBlur(g, (0, 0), 12)


def phase_shift(a, b, layout, box=None):
    """Content translation from frame a to frame b: returns (dx, dy, response).

    dx > 0 means the slide content moved right on screen. Response ~0.05 or less
    means little texture/unreliable; >0.2 is a confident match.
    """
    box = box or layout.inner_square()
    pa, pb = _prep(a, box), _prep(b, box)
    if pa.std() < 0.5 or pb.std() < 0.5:
        return 0.0, 0.0, 0.0
    win = cv2.createHanningWindow(pa.shape[::-1], cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(pa, pb, win)
    return float(dx), float(dy), float(resp)


# --------------------------------------------------------------------------
# Change metrics for click verification
# --------------------------------------------------------------------------

def slide_background(frame, layout):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype('float32')
    return lab, np.median(lab[layout.valid_mask()], axis=0)


def object_mask(frame, layout, contrast=18.0):
    lab, bg = slide_background(frame, layout)
    return (np.linalg.norm(lab - bg, axis=2) > contrast) & layout.valid_mask(), bg


def local_change(before, after, point, layout, radius=40):
    """How much of the object under `point` disappeared between two frames."""
    x, y = map(int, point)
    h, w = before.shape[:2]
    x0, y0, x1, y1 = max(0, x - radius), max(0, y - radius), min(w, x + radius + 1), min(h, y + radius + 1)
    mb, bg_b = object_mask(before, layout)
    ma, _ = object_mask(after, layout)
    pb, pa = mb[y0:y1, x0:x1], ma[y0:y1, x0:x1]
    # Restrict to the connected object touching the point (or nearest to it).
    n, labels = cv2.connectedComponents(pb.astype('uint8'))
    ly, lx = y - y0, x - x0
    if n > 1:
        lab_id = labels[ly, lx]
        if lab_id == 0:
            ys, xs = np.nonzero(labels)
            k = int(np.argmin((xs - lx) ** 2 + (ys - ly) ** 2))
            lab_id = labels[ys[k], xs[k]]
            on_object = bool(np.hypot(xs[k] - lx, ys[k] - ly) <= 3)
        else:
            on_object = True
        obj = labels == lab_id
    else:
        obj = np.zeros_like(pb)
        on_object = False
    lab_b = cv2.cvtColor(before[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).astype('float32')
    lab_a = cv2.cvtColor(after[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).astype('float32')
    delta = np.linalg.norm(lab_a - lab_b, axis=2)
    obj_px = int(obj.sum())
    gone = obj & ~pa           # object pixels that now look like background
    changed = obj & (delta > 15)
    # Core under the click point: overlapping objects merge into one component,
    # so judge the clicked spot itself as well as the whole component.
    yy, xx = np.ogrid[:pb.shape[0], :pb.shape[1]]
    core = obj & ((xx - lx) ** 2 + (yy - ly) ** 2 <= 8 ** 2)
    core_px = int(core.sum())
    return dict(object_px=obj_px, cursor_on_object=on_object,
                core_gone_frac=round(float((core & ~pa).sum()) / core_px, 3) if core_px else 0.0,
                gone_frac=round(float(gone.sum()) / obj_px, 3) if obj_px else 0.0,
                changed_frac=round(float(changed.sum()) / obj_px, 3) if obj_px else 0.0,
                patch_changed_px=int((delta > 15).sum()))


def region_changed_px(a, b, box, thresh=30):
    x0, y0, x1, y1 = box
    d = cv2.absdiff(a[y0:y1, x0:x1], b[y0:y1, x0:x1]).max(axis=2)
    return int((d > thresh).sum())


def global_change_frac(a, b, layout, thresh=25):
    m = layout.valid_mask()
    d = cv2.absdiff(a, b).max(axis=2)
    return round(float(((d > thresh) & m).sum()) / m.sum(), 4)


def judge_click(before, after, point, layout):
    loc = local_change(before, after, point, layout)
    hud = region_changed_px(before, after, layout.hud_box())
    glob = global_change_frac(before, after, layout)
    dots_b = dots_a = None
    if layout.minimap:
        dots_b = len(minimap_features(before, layout.minimap).dots)
        dots_a = len(minimap_features(after, layout.minimap).dots)
    removed = loc['object_px'] >= 25 and (loc['gone_frac'] >= 0.6 or
                                          (loc['cursor_on_object'] and loc['core_gone_frac'] >= 0.8))
    verdict = ('REMOVED' if removed else 'no change' if loc['changed_frac'] < 0.2 else 'changed?')
    if glob > 0.15:
        verdict += ' (view moved - unreliable)'
    return dict(verdict=verdict, removed=bool(removed), local=loc, hud_changed_px=hud,
                global_change_frac=glob, minimap_dots_before=dots_b, minimap_dots_after=dots_a)


# --------------------------------------------------------------------------
# Drawing helpers for diagnostics
# --------------------------------------------------------------------------

def draw_layout(frame, layout, extra_points=()):
    out = frame.copy()
    cx, cy, r = layout.scope
    cv2.circle(out, (int(cx), int(cy)), int(r), (0, 255, 255), 2)
    ov = out.copy()
    ov[~layout.valid_mask()] = (ov[~layout.valid_mask()] * 0.5).astype('uint8')
    out = cv2.addWeighted(out, 0.4, ov, 0.6, 0)
    x0, y0, x1, y1 = layout.hud_box()
    cv2.rectangle(out, (x0, y0), (x1, y1), (255, 0, 255), 1)
    x0, y0, x1, y1 = layout.inner_square()
    cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 0), 1)
    if layout.minimap:
        mx, my, mr = layout.minimap
        cv2.circle(out, (int(mx), int(my)), int(mr), (0, 255, 255), 1)
        info = minimap_features(frame, layout.minimap)
        for d in info.dots:
            cv2.circle(out, (int(d[0]), int(d[1])), 3, (0, 0, 255), 1)
        if info.ring:
            cv2.circle(out, (int(info.ring[0]), int(info.ring[1])), int(info.ring[2]), (255, 0, 0), 1)
    for p in extra_points:
        cv2.drawMarker(out, (int(p[0]), int(p[1])), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
    return out


def zoom_strip(frames, point, labels, radius=60, scale=3):
    """Side-by-side zoomed crops around `point`, one per frame, with a crosshair."""
    x, y = map(int, point)
    tiles = []
    for f, lab in zip(frames, labels):
        h, w = f.shape[:2]
        pad = cv2.copyMakeBorder(f, radius, radius, radius, radius, cv2.BORDER_CONSTANT)
        tile = pad[y: y + 2 * radius + 1, x: x + 2 * radius + 1].copy()
        tile = cv2.resize(tile, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        c = radius * scale + scale // 2
        for a, b in [((c - 14, c), (c - 4, c)), ((c + 4, c), (c + 14, c)),
                     ((c, c - 14), (c, c - 4)), ((c, c + 4), (c, c + 14))]:
            cv2.line(tile, a, b, (255, 0, 255), 2)
        cv2.rectangle(tile, (0, 0), (tile.shape[1], 22), (20, 20, 20), -1)
        cv2.putText(tile, lab, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(tile)
    return np.hstack(tiles)


def as_jsonable(obj):
    if hasattr(obj, '__dataclass_fields__'):
        return as_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {k: as_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [as_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
