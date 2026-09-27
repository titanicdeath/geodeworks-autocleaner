"""M2 slide mapping: sweep plan, capture, stitching, detection merge, minimap check.

Slide coordinates S are screen px at the microscope's zoom, measured from the
minimap centre (same frame as pan.Panner.V). A pixel at screen (x, y) in a frame
taken with the view centred on V is at S = V + (x - cx, y - cy).

The sweep (live) and the analysis (offline-capable) are separate: the sweep saves
every frame crop and the stop positions; analyze() only reads those files, so a
live run can be re-analysed without the game.
"""
import json
import math
from dataclasses import asdict

import cv2
import numpy as np

from . import detect as D
from . import register as R
from . import vision as v

MODE_COLORS = {'survey': (80, 220, 80), 'bright': (40, 200, 255), 'polarized': (255, 80, 255), '?': (200, 200, 200),
               'survey?': (150, 200, 150), 'faded?': (200, 200, 200), 'speck?': (255, 255, 255)}


# --------------------------------------------------------------------------
# Sweep plan
# --------------------------------------------------------------------------

def plan_grid(limit_xy, spacing=540.0, margin=20.0):
    """Serpentine grid of view centres covering |x| <= lx, |y| <= ly.

    With the view's usable radius ~438 px and the HUD cutting it at -373 px, a
    spacing up to ~540 px leaves no gaps (worst point ~(270, 255) from a centre)."""
    lx, ly = (max(0.0, float(l) - margin) for l in limit_xy)
    nx = int(math.ceil(2 * lx / spacing)) + 1
    ny = int(math.ceil(2 * ly / spacing)) + 1
    xs = np.linspace(-lx, lx, nx)
    ys = np.linspace(-ly, ly, ny)
    pts = []
    for j, y in enumerate(ys):
        row = xs if j % 2 == 0 else xs[::-1]
        pts += [(float(x), float(y)) for x in row]
    return pts


def crop_layout(layout):
    """Layout for frames cropped to the microscope box (what the sweep saves)."""
    x0, y0, x1, y1 = D.scope_box(layout)
    cx, cy, r = layout.scope
    return v.Layout(scope=(cx - x0, cy - y0, r), minimap=None, size=(x1 - x0, y1 - y0)), (x0, y0)


# --------------------------------------------------------------------------
# Analysis (reads a sweep folder)
# --------------------------------------------------------------------------

def load_sweep(folder):
    meta = json.loads((folder / 'sweep.json').read_text(encoding='utf-8'))
    frames = {}
    for s in meta['stops']:
        for m, name in s['frames'].items():
            frames[(s['i'], m)] = cv2.imread(str(folder / name))
    return meta, frames


def refine_positions(meta, frames, clayout, mode='survey', max_dist=650, min_score=0.5, anchor_w=0.05):
    """Least-squares stop positions from pairwise registration of overlapping stops.

    Measurements: P_j - P_i = -(content shift i->j). Weak anchors to the tracked V
    fix the overall position. Returns (P, pairs) with P as an (n, 2) array."""
    stops = meta['stops']
    n = len(stops)
    V = np.array([s['V'] for s in stops], float)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(V[i] - V[j]) > max_dist:
                continue
            a, b = frames.get((stops[i]['i'], mode)), frames.get((stops[j]['i'], mode))
            if a is None or b is None:
                continue
            dx, dy, score = R.register(a, b, clayout, prior=V[i] - V[j], search=40)
            if score >= min_score:
                pairs.append(dict(i=i, j=j, shift=[dx, dy], score=round(score, 3),
                                  vs_tracked=round(float(np.hypot(dx - (V[i, 0] - V[j, 0]),
                                                                  dy - (V[i, 1] - V[j, 1]))), 2)))
    rows, rhs, w = [], [], []
    for p in pairs:
        r = np.zeros(n)
        r[p['j']], r[p['i']] = 1, -1
        rows.append(r)
        rhs.append(-np.array(p['shift']))
        w.append(p['score'])
    for i in range(n):
        r = np.zeros(n)
        r[i] = 1
        rows.append(r)
        rhs.append(V[i])
        w.append(anchor_w)
    A = np.array(rows) * np.sqrt(np.array(w))[:, None]
    B = np.array(rhs) * np.sqrt(np.array(w))[:, None]
    P, *_ = np.linalg.lstsq(A, B, rcond=None)
    for p in pairs:
        pred = P[p['j']] - P[p['i']]
        p['residual'] = round(float(np.hypot(*(pred + np.array(p['shift'])))), 2)
    return P, pairs


def mosaic(frames, P, stops, mode, clayout, limit_xy, shrink=0.94, pad=20):
    """Stitch one mode's frames; each canvas pixel comes from the frame whose
    centre it is closest to (least vignetting/perspective)."""
    cx, cy, r = clayout.scope
    lx, ly = limit_xy
    W, H = int(2 * (lx + r + pad)), int(2 * (ly + r + pad))
    origin = np.array([W / 2, H / 2])
    canvas = np.zeros((H, W, 3), np.uint8)
    best = np.full((H, W), np.inf, np.float32)
    valid = clayout.valid_mask(shrink)
    h, w = valid.shape
    yy, xx = np.mgrid[:h, :w]
    dist = np.hypot(xx - cx, yy - cy).astype(np.float32)
    dist[~valid] = np.inf
    for k, s in enumerate(stops):
        f = frames.get((s['i'], mode))
        if f is None:
            continue
        ox, oy = np.round(origin + P[k] - (cx, cy)).astype(int)
        x0, y0 = max(0, ox), max(0, oy)
        x1, y1 = min(W, ox + w), min(H, oy + h)
        if x0 >= x1 or y0 >= y1:
            continue
        sub = dist[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        win = best[y0:y1, x0:x1]
        take = sub < win
        win[take] = sub[take]
        canvas[y0:y1, x0:x1][take] = f[y0 - oy:y1 - oy, x0 - ox:x1 - ox][take]
    return canvas, origin, np.isfinite(best)


def collect_objects(meta, frames, P, clayout, modes):
    """Detect per stop (all modes of one stop are the same view), then express in
    slide coordinates."""
    cx, cy, r = clayout.scope
    out = []
    for k, s in enumerate(meta['stops']):
        maps = {m: D.contrast_map(frames[(s['i'], m)], clayout) for m in modes if (s['i'], m) in frames}
        if not maps:
            continue
        for o in D.detect_objects(maps):
            off = P[k] - (cx, cy)
            out.append(dict(stop=k, obj=o, S=np.array(o.centroid) + off, click=np.array(o.click) + off,
                            mask_origin=np.round(np.array(o.bbox[:2]) + off).astype(int),
                            centre_dist=float(np.hypot(o.centroid[0] - cx, o.centroid[1] - cy))))
    return out


def merge_objects(recs, min_overlap=0.3):
    """Deduplicate detections of the same object seen from several stops.

    Two detections are the same object when their masks overlap in slide
    coordinates (>= min_overlap of the smaller one); centroids alone fail for
    long lines, of which each view may show a different part. Best view first
    (not touching the view edge, then nearest the view centre), so it becomes
    the representative. The mode is voted over the views, weighted by dominance."""
    if not recs:
        return []
    lo = np.min([r['mask_origin'] for r in recs], axis=0) - 2
    hi = np.max([r['mask_origin'] + r['obj'].mask.shape[::-1] for r in recs], axis=0) + 2
    canvas = np.zeros((hi[1] - lo[1], hi[0] - lo[0]), np.int32)       # group id + 1
    order = sorted(recs, key=lambda r: (r['obj'].partial, r['centre_dist']))
    groups = []
    for r in order:
        m = r['obj'].mask
        x0, y0 = r['mask_origin'] - lo
        win = canvas[y0:y0 + m.shape[0], x0:x0 + m.shape[1]]
        hits = win[m]
        hits = hits[hits > 0]
        g = None
        if hits.size:
            gid = int(np.bincount(hits).argmax())
            cand = groups[gid - 1]
            share = np.count_nonzero(hits == gid) / min(m.sum(), cand['obj'].mask.sum())
            if share >= min_overlap:
                g = cand
        if g is None:
            groups.append(dict(r, views=1, votes={r['obj'].mode: r['obj'].dominance}))
            win[m & (win == 0)] = len(groups)
        else:
            g['views'] += 1
            g['votes'][r['obj'].mode] = g['votes'].get(r['obj'].mode, 0) + r['obj'].dominance
            win[m & (win == 0)] = groups.index(g) + 1
    absorb_partials(groups, canvas)
    merge_line_groups(groups, canvas)
    targets = []
    for n, g in enumerate(groups):
        if g.get('absorbed'):
            continue
        o = g['obj']
        mode = max(g['votes'], key=g['votes'].get)
        base = dict(mode=mode, area=o.area, contrast=o.contrast, dominance=o.dominance, texture=o.texture,
                    elongation=o.elongation, survey_guess=o.survey_guess, method=o.method,
                    alpha_scores=o.alpha_scores, partial=o.partial, views=g['views'],
                    votes={k: round(val, 2) for k, val in g['votes'].items()}, best_stop=g['stop'])
        # The union of every view's mask of this object, in slide coordinates.
        ys, xs = np.nonzero(canvas == n + 1)
        union = (canvas[ys.min():ys.max() + 1, xs.min():xs.max() + 1] == n + 1) if len(xs) else None
        uorig = np.array([xs.min(), ys.min()]) + lo if len(xs) else None
        if union is not None and (g.get("line_like") or D.is_line_like(o.mask)):
            segs = D.line_segments(union)
            if len(segs) >= 1:
                for a, b in segs:
                    mid = (a + b) / 2
                    # Click on the line: nearest mask pixel to the segment midpoint.
                    my, mx = np.nonzero(union)
                    k = int(np.argmin((mx - mid[0]) ** 2 + (my - mid[1]) ** 2))
                    targets.append(dict(base, kind='line', S=np.round(mid + uorig, 1).tolist(),
                                        click=[float(mx[k] + uorig[0]), float(my[k] + uorig[1])],
                                        segment=[np.round(a + uorig, 1).tolist(), np.round(b + uorig, 1).tolist()],
                                        area=int(np.linalg.norm(b - a) * 10), parent=n + 1))
                continue
        kind = 'speck' if D.is_speck(o.area, o.contrast.get('survey', max(o.contrast.values())), o.elongation) \
            else ('line' if o.elongation >= 4 else 'block')
        targets.append(dict(base, kind=kind, S=np.round(g['S'], 1).tolist(), click=np.round(g['click'], 1).tolist()))
    for i, t in enumerate(targets):
        t['id'] = i + 1
    return targets


def absorb_partials(groups, canvas, reach=20, min_inside=0.6):
    """Merge objects that were only ever seen cut off at a view edge into a whole
    object right next to them. Blocks are 3D, so between views their outline
    shifts (parallax ~3.5% of the view offset); an edge sliver of a block seen from
    a far stop can then fail the overlap test and pose as a thin 'line'."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1))
    for gi, g in enumerate(groups):
        if not g['obj'].partial:
            continue
        ys, xs = np.nonzero(canvas == gi + 1)
        if not len(xs):
            continue
        y0, y1 = max(0, ys.min() - reach), min(canvas.shape[0], ys.max() + reach + 1)
        x0, x1 = max(0, xs.min() - reach), min(canvas.shape[1], xs.max() + reach + 1)
        win = canvas[y0:y1, x0:x1]
        mine = win == gi + 1
        best, best_frac = None, 0.0
        for h in np.unique(win):
            if h == 0 or h == gi + 1 or groups[h - 1]['obj'].partial or groups[h - 1].get('absorbed'):
                continue
            near = cv2.dilate((win == h).astype(np.uint8), k).astype(bool)
            frac = float((mine & near).sum()) / mine.sum()
            if frac > best_frac:
                best, best_frac = h, frac
        if best is not None and best_frac >= min_inside:
            g['absorbed'] = True
            groups[best - 1]['views'] += g['views']
            win[mine] = best


def merge_line_groups(groups, canvas, reach=12):
    """Pieces of the same bent line seen from different stops can fail the overlap
    test (views misaligned by a few px, thin masks). Join line-like groups whose
    masks come within `reach` px of each other."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1))
    ids = []
    for gi, g in enumerate(groups):
        if g.get('absorbed'):
            continue
        ys, xs = np.nonzero(canvas == gi + 1)
        if len(xs) and D.is_line_like(g["obj"].mask):
            ids.append(gi)
    for gi in ids:
        if groups[gi].get('absorbed'):
            continue
        near = cv2.dilate((canvas == gi + 1).astype(np.uint8), k).astype(bool)
        for gj in ids:
            if gj == gi or groups[gj].get('absorbed'):
                continue
            if (near & (canvas == gj + 1)).any():
                groups[gi]['line_like'] = True
                groups[gj]['absorbed'] = True
                groups[gi]['views'] += groups[gj]['views']
                canvas[canvas == gj + 1] = gi + 1
                near = cv2.dilate((canvas == gi + 1).astype(np.uint8), k).astype(bool)


def split_specks(targets):
    """Specks (small bright squares) are kept out of the target list. Live they
    carry no minimap dot in any mode and show parallax between views (~3.5% of the
    view offset), i.e. they float above the slide - most likely decoration."""
    specks = [t for t in targets if t['kind'] == 'speck']
    keep = [t for t in targets if t['kind'] != 'speck']
    for i, t in enumerate(keep):
        t['id'] = i + 1
    return keep, specks


def minimap_dots_slide(meta, scale):
    """Minimap dots per mode, converted to slide coordinates."""
    mm = np.array(meta['minimap'][:2])
    out = {}
    for m, dots in meta.get('minimap_dots', {}).items():
        out[m] = [((np.array(d) - mm) * R.scale_vec(scale)).round(1).tolist() for d in dots]
    return out


def assign_modes(targets, dots, radius=60.0, mismatch=25.0):
    """Give each target a mode by matching it to the minimap dots.

    The minimap in each mode shows that mode's remaining imperfections. Dot
    positions are coarse (0.5 minimap px ~ 13 slide px, plus dot size), hence the
    generous radius. Survey-only appearance nudges the choice: a full-contrast
    object prefers a Survey dot, a faded one a Bright/Polarized dot (+mismatch px).
    One dot per target; greedy by cost. Unmatched targets keep a guess:
    'survey?' (full contrast) or 'faded?'. Returns per-mode statistics."""
    cand = []
    for ti, t in enumerate(targets):
        S = np.array(t['S'])
        for m, pts in dots.items():
            for di, pnt in enumerate(pts):
                d = float(np.linalg.norm(S - pnt))
                if d >= radius:
                    continue
                pen = 0.0
                if t.get('survey_guess') == 'full' and m != 'survey':
                    pen = mismatch
                elif t.get('survey_guess') == 'faded' and m == 'survey':
                    pen = mismatch
                cand.append((d + pen, ti, m, di, d))
    used_t, used_d = set(), set()
    for cost, ti, m, di, d in sorted(cand):
        if ti in used_t or (m, di) in used_d:
            continue
        used_t.add(ti)
        used_d.add((m, di))
        targets[ti]['mode'] = m
        targets[ti]['dot_dist'] = round(d, 1)
    # Segments of one line are one object's pieces: an unmatched segment takes the
    # mode its matched siblings got (the minimap may show fewer dots than segments).
    parent_mode = {}
    for ti in used_t:
        t = targets[ti]
        if t.get('parent'):
            parent_mode.setdefault(t['parent'], []).append(t['mode'])
    for ti, t in enumerate(targets):
        if ti not in used_t:
            sib = parent_mode.get(t.get('parent'))
            if sib:
                t['mode'] = max(set(sib), key=sib.count)
                t['mode_from'] = 'sibling segment'
            else:
                t['mode'] = ('speck?' if t.get('kind') == 'speck' else
                             'survey?' if t.get('survey_guess') == 'full' else 'faded?')
            t['dot_dist'] = None
    stats = {}
    for m, pts in dots.items():
        n_match = sum(1 for (mm, _) in used_d if mm == m)
        stats[m] = dict(dots=len(pts), matched=n_match, dots_without_target=len(pts) - n_match)
    stats['targets_without_dot'] = len(targets) - len(used_t)
    # How well would the Survey-only look (full vs faded) have predicted the mode?
    matched = [t for t in targets if t['dot_dist'] is not None and t.get('survey_guess')]
    ok = sum((t['survey_guess'] == 'full') == (t['mode'] == 'survey') for t in matched)
    stats['survey_guess_agreement'] = round(ok / len(matched), 3) if matched else None
    return stats


def draw_map(canvas, origin, targets, dots, P, scale=0.5):
    img = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    to = lambda S: tuple(np.round((origin + np.asarray(S)) * scale).astype(int))
    for m, pts in dots.items():
        for p in pts:
            cv2.circle(img, to(p), 11, MODE_COLORS.get(m, (255, 255, 0)), 1, cv2.LINE_AA)
    for t in targets:
        col = MODE_COLORS.get(t['mode'], (200, 200, 200))
        c = to(t['S'])
        if t.get('segment'):
            cv2.line(img, to(t['segment'][0]), to(t['segment'][1]), col, 2, cv2.LINE_AA)
        rad = max(6, int(math.sqrt(t['area']) * scale * 0.6)) if t['kind'] != 'line' else 7
        cv2.circle(img, c, rad, col, 2)
        cv2.drawMarker(img, to(t['click']), (0, 0, 255), cv2.MARKER_CROSS, 8, 1)
        cv2.putText(img, f'{t["id"]}{t["mode"][0].upper()}', (c[0] + rad, c[1] - rad),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
    for k, p in enumerate(P):
        cv2.putText(img, str(k), to(p), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)
    y = 18
    for m, col in MODE_COLORS.items():
        if m == '?':
            continue
        cv2.putText(img, m, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        y += 18
    cv2.putText(img, 'thick: found objects (mode colour; grey = no minimap dot)  thin rings: minimap dots  '
                     '+: click points',
                (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
    return img


def analyze(folder, log=print):
    """Full offline analysis of a sweep folder. Writes map outputs next to it."""
    meta, frames = load_sweep(folder)
    L = meta['crop_layout']
    clayout = v.Layout(scope=tuple(L['scope']), minimap=None, size=tuple(L['size']))
    stops = meta['stops']
    limit = meta['limit_xy']
    # Only Survey frames are used: the other modes zoom the view (Bright ~1.40x,
    # Polarized ~2.08x), and in Survey every imperfection is visible.
    P, pairs = refine_positions(meta, frames, clayout, mode='survey')
    V = np.array([s['V'] for s in stops])
    corr = np.hypot(*(P - V).T)
    resid = [p['residual'] for p in pairs]
    recs = collect_objects(meta, frames, P, clayout, ['survey'])
    targets, specks = split_specks(merge_objects(recs))
    # Re-read the minimap dots from the saved screenshots when present, so a
    # re-analysis always uses the current dot detector.
    for m in list(meta.get('minimap_dots', {})):
        png = folder / f'minimap-{m}.png'
        if png.exists() and meta.get('minimap'):
            meta['minimap_dots'][m] = np.round(
                v.minimap_features(cv2.imread(str(png)), tuple(meta['minimap'])).dots, 2).tolist()
    dots = minimap_dots_slide(meta, meta['ring_scale'])
    match = assign_modes(targets, dots)
    canvas, origin, covered = mosaic(frames, P, stops, 'survey', clayout, limit)
    cv2.imwrite(str(folder / 'mosaic-survey.jpg'), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img = draw_map(canvas, origin, targets, dots, P)
    cv2.imwrite(str(folder / 'map.png'), img)
    by_mode = {}
    for t in targets:
        by_mode[t['mode']] = by_mode.get(t['mode'], 0) + 1
    summary = dict(stops=len(stops), targets=len(targets), by_mode=by_mode, specks=len(specks),
                   pairs=len(pairs), pair_residual_px=dict(median=float(np.median(resid)) if resid else None,
                                                           max=float(max(resid)) if resid else None),
                   tracked_vs_refined_px=dict(median=float(np.median(corr)), max=float(corr.max())),
                   minimap=match, coverage_frac=float(covered.mean()))
    (folder / 'targets.json').write_text(json.dumps(dict(summary=summary, targets=targets, specks=specks,
                                                         positions=P.round(1).tolist(), pairs=pairs,
                                                         minimap_dots=dots), indent=2), encoding='utf-8')
    rep = report(summary, targets)
    (folder / 'map-report.txt').write_text(rep, encoding='utf-8')
    log(rep)
    return summary, targets


def report(s, targets):
    m = s['minimap']
    L = ['M2 slide map report', '',
         f'Stops: {s["stops"]} (Survey), slide coverage {s["coverage_frac"]*100:.0f}% of the canvas',
         f'Stitching: {s["pairs"]} overlapping pairs registered; residual after least squares median '
         f'{s["pair_residual_px"]["median"]} px, max {s["pair_residual_px"]["max"]} px; tracked positions moved by '
         f'median {s["tracked_vs_refined_px"]["median"]:.1f} px, max {s["tracked_vs_refined_px"]["max"]:.1f} px',
         f'Imperfections found: {s["targets"]}  by mode: {s["by_mode"]}  (+ {s.get("specks", 0)} floating specks, not counted)']
    for mode in ('survey', 'bright', 'polarized'):
        if mode in m:
            r = m[mode]
            L.append(f'  minimap in {mode}: {r["dots"]} dots, {r["matched"]} matched to a found object, '
                     f'{r["dots_without_target"]} with nothing found nearby')
    L.append(f'  found objects with no minimap dot nearby: {m.get("targets_without_dot")}')
    if m.get('survey_guess_agreement') is not None:
        L.append(f'Survey-only look (full contrast vs faded) agrees with the minimap mode for '
                 f'{m["survey_guess_agreement"]*100:.0f}% of matched objects')
    L.append('')
    L.append(f'{"id":>3} {"mode":10} {"kind":6} {"S (slide px)":>14} {"area":>6} {"dE":>6} {"tex":>5} '
             f'{"look":6} {"dot_px":>6} {"views":>5}')
    for t in targets:
        L.append(f'{t["id"]:>3} {t["mode"]:10} {t["kind"]:6} {str(tuple(round(c) for c in t["S"])):>14} {t["area"]:>6} '
                 f'{t["contrast"].get("survey", 0):>6} {t["texture"]:>5} {t["survey_guess"]:6} '
                 f'{str(t.get("dot_dist")):>6} {t["views"]:>5}')
    return '\n'.join(L)
