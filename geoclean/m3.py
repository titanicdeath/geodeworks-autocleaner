"""M3: clean the slide, driven by the minimap.

The minimap in each mode shows exactly that mode's remaining imperfections, so
it is the to-do list: no mapping sweep is needed, and the floating specks (no
dot in any mode; they cannot be removed) are never targeted.

For each mode, repeat until its minimap is empty (or nothing more happens):
  1. read the dots (slide coordinates),
  2. group them into views; pan to each view in Survey with one quick drag
     (position from the minimap ring),
  3. in Bright/Polarized, switch to that mode (the view zooms 1.40x / 2.08x),
  4. detect once, pick the object at each dot (the highest-contrast object near
     it = the one at full contrast in this mode), click them all in a burst,
  5. verify all clicks from one after-frame; a click that revealed another block
     underneath gets a follow-up.
Every click lands inside a freshly detected object; 5 clicks in a row without
effect stop the run.
"""
import json
import time

import cv2
import numpy as np

from . import detect as D
from . import modes as MO
from . import register as R
from . import vision as v
from . import winapi as w
from . import zoom as Z
from .inputs import MouseDriver, Strategy
from .m0 import setup_session, wait_key, wait_still
from .m2 import load_calibration
from .pan import Panner
from .safety import Stop

# Move like S3 (absolute + wiggle; M0: registers as a real hover), shorter dwell/hold.
CLICK = Strategy('click', 'abs', wiggle=True, dwell=0.06, hold=0.05)
MODE_ORDER = ('survey', 'bright', 'polarized')
MIN_TARGET_AREA = 600
# Survey imperfections are rough rock. Interior texture on live samples: rock 2.0-3.8,
# faded blocks <= 1.1, lines ~0.4 (v0.9 live: all 5 Survey misses were smooth cyan lines).
SURVEY_MIN_TEXTURE = 1.5     # live: blocks ~4200 px, line pieces ~1000+, specks 250-300


class Cleaner:
    def __init__(self, guard, driver, api, screen, layout, panner, run, limit, max_fail_streak=5):
        self.guard, self.driver, self.api, self.screen = guard, driver, api, screen
        self.layout, self.p, self.run, self.limit = layout, panner, run, limit
        self.mode = 'survey'
        cx, cy, _ = layout.scope
        # Zoom constants: live Bright measured 1.401x, 1.401x and 1.408x about the scope
        # centre (SIFT); measuring cost ~0.3 s per run and often lacked features.
        self.zoom = {m: dict(zoom=Z.DEFAULT_ZOOM[m], centre=[cx, cy], measured=True) for m in MODE_ORDER}
        self.records = []
        self.fail_streak = 0
        self.max_fail_streak = max_fail_streak
        self.mm = np.array(layout.minimap[:2]) if layout.minimap else None
        self.bursts = 0
        self.pan_style = self.PAN_SPEEDS[0]   # no primer: calibration showed no benefit, saves a frame gap
        self.dead = {}                         # mode -> slide spots where a click did nothing
        self.unresolved = 0                    # dots for which no object was picked

    # --- view / mode -------------------------------------------------------
    def to_mode(self, m):
        """Switch lighting mode (raises MO.ModeDone if its button is greyed out)."""
        if MO.read_buttons(self.screen.grab(), self.layout)['active'] == m:
            self.mode = m
            self.p.set_mode(m, self.zoom[m]['zoom'], v.RING_R_FRAC_MODE[m])
            return
        # Zoom is measured against a Survey frame only (the only unzoomed view).
        before = self.screen.grab() if (not self.zoom[m]['measured'] and self.mode == 'survey') else None
        MO.switch_mode(m, self.driver, self.guard, self.screen, self.layout, self.run.log, steady_timeout=3.0)
        self.mode = m
        if before is not None:
            res = Z.measure_zoom(before, self.screen.grab(), self.layout, Z.DEFAULT_ZOOM[m])
            self.zoom[m]['measured'] = True
            if res:
                self.zoom[m].update(zoom=res['zoom'], centre=res['centre'], inliers=res['inliers'])
                self.run.log(f'  {m} zoom measured: {res["zoom"]:.3f}x about {res["centre"]}')
            else:
                self.run.log(f'  {m} zoom: too few features to measure, using {self.zoom[m]["zoom"]}x')
        self.p.set_mode(m, self.zoom[m]['zoom'], v.RING_R_FRAC_MODE[m])

    def finished(self, mode, frame=None):
        return mode in MO.read_buttons(self.screen.grab() if frame is None else frame, self.layout)['done']

    def pred(self, S):
        zi = self.zoom[self.mode]
        return np.array(zi['centre']) + zi['zoom'] * (np.asarray(S, float) - self.p.V)

    def view_radius(self, mode):
        """Slide px around the view centre that are comfortably clickable in `mode`."""
        return 0.68 * self.layout.scope[2] / self.zoom[mode]['zoom']

    def dots(self, frame=None):
        if self.mm is None:
            return []
        frame = self.screen.grab() if frame is None else frame
        scale = R.scale_vec(self.p.model.ring_scale)
        return [(np.array(d) - self.mm) * scale for d in v.minimap_features(frame, self.layout.minimap).dots]

    # --- picking -----------------------------------------------------------
    def candidates(self, frame):
        cm = D.contrast_map(frame, self.layout)
        self._dEs = cv2.GaussianBlur(cm.dE, (0, 0), 3)          # smoothed contrast, crop coords
        self._cm = cm
        objs = []
        k = np.ones((9, 9), np.uint8)
        for o in D.detect_objects({'survey': cm}):
            if o.area < MIN_TARGET_AREA and not D.is_line_like(o.mask):
                continue                 # specks and fragments (specks: cannot be removed)
            x, y, w, h = o.bbox
            ys, xs = np.nonzero(o.mask)
            # Texture INSIDE the object (4 px in from its edge, so a thin line's
            # edges don't count): rock blocks are rough, faded objects smooth.
            inner = cv2.erode(D.fill_holes(o.mask).astype(np.uint8), k).astype(bool) & o.mask
            tex = cm.tex[y - cm.y0:y - cm.y0 + h, x - cm.x0:x - cm.x0 + w]
            itex = float(tex[inner].mean()) if inner.sum() >= 30 else 0.0
            objs.append(dict(obj=o, xs=xs + x, ys=ys + y, itex=itex))
        return objs

    def strong_part(self, c, p):
        """A full-contrast block lying on faded blocks merges with them into one blob
        (live: a rock block at dE ~100 on two faded Bright blocks at ~50; the blob's
        deepest point was on a faded block and 4 clicks did nothing). Keep the pixels
        near the blob's top contrast and take the piece nearest the dot."""
        o = c['obj']
        x, y, w, h = o.bbox
        dE = self._dEs[y - self._cm.y0:y - self._cm.y0 + h, x - self._cm.x0:x - self._cm.x0 + w]
        vals = dE[o.mask]
        if vals.size < 50 or np.percentile(vals, 95) < 1.5 * np.percentile(vals, 20):
            return None                                          # uniform: one object
        strong = (o.mask & (dE >= 0.7 * np.percentile(vals, 95))).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(strong)
        best = None
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] < 300:
                continue
            ys, xs = np.nonzero(lab == i)
            d = float(np.sqrt(((xs + x - p[0]) ** 2 + (ys + y - p[1]) ** 2).min()))
            if best is None or d < best[0]:
                best = (d, lab == i)
        if best is None:
            return None
        part = D.fill_holes(best[1]).astype(np.uint8)
        dist = cv2.distanceTransform(np.pad(part, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
        iy, ix = np.unravel_index(int(np.argmax(dist)), dist.shape)
        return float(x + ix), float(y + iy)

    def pick(self, objs, S, radius, taken):
        """The object at dot S, among objects reaching within `radius` of where S
        should be and not much farther than the closest one:
        - Survey: the ROUGHEST (interior texture). Survey imperfections are rock
          blocks; live on a dark mineral the faded Polarized lines had 3-4x the
          contrast of the rock blocks, so "highest contrast" clicked faded lines.
        - Bright/Polarized: the highest contrast (their own objects are opaque there)."""
        p = self.pred(S)
        near = []
        for i, c in enumerate(objs):
            if i in taken:
                continue
            if self.mode == 'survey' and c['itex'] < SURVEY_MIN_TEXTURE:
                continue        # smooth: another mode's object (live: every Survey miss was a cyan line)
            d = float(np.sqrt(((c['xs'] - p[0]) ** 2 + (c['ys'] - p[1]) ** 2).min()))
            if d <= radius:
                near.append((d, i))
        if not near:
            return None
        dmin = min(d for d, _ in near)
        if self.mode == 'survey':
            key = lambda i: (round(objs[i]['itex'], 1), objs[i]['obj'].contrast['survey'])
        else:
            key = lambda i: objs[i]['obj'].contrast['survey']
        for i in sorted((i for d, i in near if d <= dmin + 25), key=key, reverse=True):
            pt, kind = self.click_point(objs[i], p)
            if not self.dead_spot(pt):
                return i, p, pt, kind
        return None

    def to_slide(self, pt):
        zi = self.zoom[self.mode]
        return self.p.V + (np.asarray(pt, float) - zi['centre']) / zi['zoom']

    def dead_spot(self, pt, radius=25.0):
        """A spot where a click in this mode already did nothing (live: the same
        faded line was clicked 3 times)."""
        S = self.to_slide(pt)
        return any(np.linalg.norm(S - q) < radius for q in self.dead.get(self.mode, []))

    def click_point(self, c, p):
        o = c['obj']
        if D.is_line_like(o.mask):
            x, y, _, _ = o.bbox
            dist = cv2.distanceTransform(np.pad(o.mask.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
            ys, xs = np.nonzero(dist >= max(2.0, 0.6 * dist.max()))
            k = int(np.argmin((xs + x - p[0]) ** 2 + (ys + y - p[1]) ** 2))
            return (float(xs[k] + x), float(ys[k] + y)), 'line'
        strong = self.strong_part(c, p)
        if strong is not None:
            return strong, 'block'
        return (float(o.click[0]), float(o.click[1])), 'block'

    def click(self, pt):
        self.driver.move_to(self.screen.to_screen(pt), CLICK)
        self.guard.sleep(CLICK.dwell)
        self.driver.click('left', CLICK.hold)

    # --- one view ----------------------------------------------------------
    def clean_view(self, dots_here, radius):
        """Burst-click every dot's object visible from here, verify from one frame."""
        for follow_up in range(3):
            before = self.screen.grab()
            objs = self.candidates(before)
            picks, taken = [], set()
            for S in dots_here:
                hit = self.pick(objs, S, radius, taken)
                if hit is None:
                    self.unresolved += 1
                    if self.unresolved <= 8:
                        p = self.pred(S)
                        name = f'unresolved/{self.unresolved:02d}-{self.mode}.jpg'
                        self.run.image(name, v.zoom_strip([before], p, ['no object picked'], radius=110), quality=85)
                        self.run.log(f'    {self.mode}: no object picked for the dot at {np.round(p).tolist()} '
                                     f'({name})', console=False)
                    continue
                i, p, pt, kind = hit
                taken.add(i)
                picks.append(dict(S=np.round(S, 1).tolist(), click=pt, kind=kind, area=objs[i]['obj'].area,
                                  contrast=objs[i]['obj'].contrast['survey']))
            if not picks:
                return 0
            for pk in picks:
                self.click(pk['click'])
            self.bursts += 1
            cm_before = self._cm
            self.guard.sleep(0.15)
            after = self.screen.grab()
            judged = [self.quick_judge(cm_before, after, pk['click']) for pk in picks]
            if any(j['outcome'] == 'no_change' for j in judged):
                self.guard.sleep(0.3)                # slow removal? look once more
                after = self.screen.grab()
                judged = [self.quick_judge(cm_before, after, pk['click']) for pk in picks]
            again = []
            for pk, j in zip(picks, judged):
                outcome = j['outcome']
                if outcome == 'no_change':
                    self.dead.setdefault(self.mode, []).append(self.to_slide(pk['click']))
                rec = dict(pk, n=len(self.records) + 1, mode=self.mode, outcome=outcome,
                           gone=j['gone'], V=np.round(self.p.V, 1).tolist(), burst=self.bursts)
                rec['click'] = [round(c, 1) for c in pk['click']]
                if outcome != 'removed' or len(self.records) < 12:     # keep the log small
                    name = f'clicks/{rec["n"]:03d}-{self.mode}-{outcome}.jpg'
                    self.run.image(name, v.zoom_strip([before, after], pk['click'], ['before', outcome], radius=60),
                                   quality=85)
                    rec['image'] = name
                self.records.append(rec)
                self.run.log(f'    #{rec["n"]:<3} {self.mode:9} {pk["kind"]:5} -> {outcome}', console=False)
                if outcome == 'no_change':
                    self.fail_streak += 1
                    if self.fail_streak >= self.max_fail_streak:
                        raise Stop(f'{self.fail_streak} clicks in a row changed nothing; stopping to look at why.')
                else:
                    self.fail_streak = 0
                if outcome == 'changed':
                    again.append(pk['S'])               # another block revealed underneath
            if not again:
                break
            dots_here = again
        return len(picks)

    def quick_judge(self, cm, after, pt, t=12.0):
        """Did the object at pt disappear? Only a patch around the click is examined
        (the view does not move while clicking); replaces the full-frame judge
        (~150 ms per click). The background is the median of the patch border, not
        the contrast map's local median: inside a big zoomed block that median is
        the block itself."""
        r = int(45 * self.zoom[self.mode]['zoom']) + 15
        x, y = int(round(pt[0])) - cm.x0, int(round(pt[1])) - cm.y0
        h, w = cm.dE.shape
        y0, y1, x0, x1 = max(0, y - r), min(h, y + r + 1), max(0, x - r), min(w, x + r + 1)
        lab_b, valid = cm.lab[y0:y1, x0:x1], cm.valid[y0:y1, x0:x1]
        crop = after[y0 + cm.y0:y1 + cm.y0, x0 + cm.x0:x1 + cm.x0]
        lab_a = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
        border = np.zeros(valid.shape, bool)
        border[:5], border[-5:], border[:, :5], border[:, -5:] = True, True, True, True
        ring = lab_b[border & valid]
        if len(ring) < 20:
            return dict(outcome='no_change', gone=0.0)
        bgc = np.median(ring, axis=0)
        db = np.linalg.norm(lab_b - bgc, axis=2)
        da = np.linalg.norm(lab_a - bgc, axis=2)
        ob = D.fill_holes((db > t) & valid) & valid
        n, lab = cv2.connectedComponents(ob.astype(np.uint8))
        ly, lx = y - y0, x - x0
        lid = lab[ly, lx]
        if lid == 0 and n > 1:
            ys, xs = np.nonzero(lab)
            k = int(np.argmin((xs - lx) ** 2 + (ys - ly) ** 2))
            lid = lab[ys[k], xs[k]] if (xs[k] - lx) ** 2 + (ys[k] - ly) ** 2 <= 144 else 0
        obj = (lab == lid) if lid else np.zeros_like(ob)
        npx = int(obj.sum())
        if npx < 25:
            return dict(outcome='no_change', gone=0.0)
        gone = obj & (da <= np.maximum(t, 0.35 * db))
        yy, xx = np.ogrid[:obj.shape[0], :obj.shape[1]]
        core = obj & ((xx - lx) ** 2 + (yy - ly) ** 2 <= 64)
        core_gone = float((core & gone).sum()) / max(1, int(core.sum()))
        changed = float((obj & (np.linalg.norm(lab_a - lab_b, axis=2) > 15)).sum()) / npx
        gone_frac = float(gone.sum()) / npx
        outcome = ('removed' if gone_frac >= 0.6 or (core.sum() >= 20 and core_gone >= 0.8) else
                   'changed' if changed >= 0.5 else 'no_change')
        return dict(outcome=outcome, gone=round(gone_frac, 3))

    # --- route -------------------------------------------------------------
    def plan_views(self, dots, R):
        """Order of views covering all dots. Greedy: next view = the centre that
        covers the most remaining dots, discounted by travel (one extra dot is
        worth travelling 2.5 view radii). Live, one pan per dot took ~18 pans for
        15 dots; each pan costs ~0.6-0.9 s."""
        remaining = [np.asarray(d, float) for d in dots]
        pos = self.p.V.copy()
        views = []
        while remaining:
            cands = list(remaining)
            for d in remaining:
                for reach in (R, 1.6 * R):
                    near = [e for e in remaining if np.linalg.norm(e - d) <= reach]
                    cands.append(np.mean(near, axis=0))
            best, best_score, best_group = None, -1e9, None
            for c in cands:
                group = [e for e in remaining if np.linalg.norm(e - c) <= R]
                score = len(group) - np.linalg.norm(c - pos) / (2.5 * R)
                if group and score > best_score:
                    best, best_score, best_group = c, score, group
            views.append((best, best_group))
            remaining = [e for e in remaining if not any(e is g for g in best_group)]
            pos = best
        return views

    PAN_SPEEDS = ('v4500np', 'v3000np', 'v1600np')

    def quick_pan(self, target, tol, tag):
        """Fast pan: one long drag that may end anywhere in the window (up to
        ~1280 px across), at 4500 px/s. If the game loses too much of such fast
        drags (moved < 75% of the request), step down to 3000, then 1600 px/s.
        Live at 3000 px/s the drags moved 89-105% of the request."""
        n0 = len(self.p.history)
        self.p.quick_pan_to(target, tol=tol, limit=self.limit, tag=tag, style=self.pan_style,
                            max_cursor=1400, wide=True)
        ratios = [h['moved_ratio'] for h in self.p.history[n0:]
                  if h.get('moved_ratio') is not None and np.linalg.norm(h['cursor']) > 300]
        if ratios and min(ratios) < 0.75 and self.pan_style != self.PAN_SPEEDS[-1]:
            self.pan_style = self.PAN_SPEEDS[self.PAN_SPEEDS.index(self.pan_style) + 1]
            self.run.log(f'  fast drags lose too much (moved {min(ratios):.2f} of the request); '
                         f'slowing to {self.pan_style[1:5]} px/s')

    # --- a whole mode ------------------------------------------------------
    def clean_mode(self, mode, max_rounds=4):
        """Clean one mode until its button greys out (the game's own "done"), its
        minimap is empty, or a round achieves nothing."""
        t0 = time.perf_counter()
        stats = dict(rounds=0, views=0, finished=False)
        try:
            for rnd in range(max_rounds):
                self.to_mode(mode)
                todo = self.dots()
                stats.setdefault('dots_per_round', []).append(len(todo))
                self.run.log(f'{mode}: minimap shows {len(todo)} (round {rnd + 1})')
                if not todo:
                    if not self.finished(mode):
                        self.run.log(f'  WARNING: {mode} minimap looks empty but its button is not greyed out '
                                     '(dots hidden or unreadable?)')
                    break
                stats['rounds'] += 1
                n_before = sum(r['outcome'] in ('removed', 'changed') for r in self.records)
                R_view = self.view_radius(mode)
                radius = 55 * self.zoom[mode]['zoom']
                for centre, group in self.plan_views(todo, R_view):
                    if np.linalg.norm(centre - self.p.V) > 0.35 * R_view:
                        # Pan in this mode: a finished mode's button greys out, so Survey
                        # may not be available, and switching costs zoom animations.
                        self.quick_pan(centre, tol=0.35 * R_view, tag=f'{mode}-view')
                    self.clean_view(group, radius)
                    stats['views'] += 1
                    if self.finished(mode):
                        raise MO.ModeDone(f'{mode} finished')
                n_after = sum(r['outcome'] in ('removed', 'changed') for r in self.records)
                if n_after == n_before:
                    break                               # nothing more to be done here
            stats['left'] = len(self.dots())
        except MO.ModeDone:
            stats['finished'] = True
            stats['left'] = 0
        if not stats['finished'] and self.finished(mode):
            stats['finished'] = True
        recs = [r for r in self.records if r['mode'] == mode]
        stats.update(clicks=len(recs), removed=sum(r['outcome'] == 'removed' for r in recs),
                     changed=sum(r['outcome'] == 'changed' for r in recs),
                     no_change=sum(r['outcome'] == 'no_change' for r in recs),
                     seconds=round(time.perf_counter() - t0, 1))
        return stats


def hud(frame, layout):
    x0, y0, x1, y1 = layout.hud_box()
    return frame[y0:y1, x0:x1].copy()


def clean(api, screen, run, args):
    model, limit = load_calibration()
    modes = [m for m in MODE_ORDER if m in args.modes]
    guard, layout = setup_session(api, screen, run, args, max_clicks=args.max_clicks)
    driver = MouseDriver(guard, api, run.log)
    p = Panner(guard, driver, api, screen, layout, model, run.log)
    c = Cleaner(guard, driver, api, screen, layout, p, run, limit)
    out = dict(modes=modes, results={})
    t0 = time.perf_counter()
    try:
        run.log('')
        run.log(f'Cleaning {", ".join(modes)}. Survey mode, any position. Tap F6, then take your hand OFF the mouse.')
        wait_key(guard, api, keys=(w.VK_F6,))
        if wait_still(guard, api) is None:
            raise Stop('Mouse never settled.')
        t0 = time.perf_counter()
        state = MO.read_buttons(screen.grab(), layout)
        run.log(f'Mode buttons: {dict(zip(MO.MODES, state["states"]))}')
        if sum(state['present']) < 3:
            raise Stop('Mode buttons not recognised; not clicking anything.')
        if state['done']:
            run.log(f'Already finished (greyed out): {", ".join(state["done"])}')
        c.mode = state['active'] or 'survey'
        p.start()
        if state['active']:
            p.set_mode(c.mode, c.zoom[c.mode]['zoom'], v.RING_R_FRAC_MODE[c.mode])
        run.image('hud-before.png', hud(screen.grab(), layout))
        # Pass 1 over every mode, then revisit whatever is not greyed out yet (the
        # game's own "finished"): the minimap can hide dots under one another, and
        # a mode may end a pass with a few imperfections left.
        for pass_no in range(1 + getattr(args, "revisits", 2)):
            todo = [m for m in modes if m not in MO.read_buttons(screen.grab(), layout)['done']]
            if pass_no and todo:
                run.log(f'Revisit {pass_no}: not finished yet: {", ".join(todo)}')
            for m in modes:
                if m not in todo:
                    if pass_no == 0:
                        out['results'][m] = dict(finished=True, already=True, rounds=0, views=0, clicks=0,
                                                 removed=0, changed=0, no_change=0, left=0, seconds=0.0)
                    continue
                r = c.clean_mode(m)
                prev = out['results'].get(m)
                if prev and not prev.get('already'):
                    for k in ('rounds', 'views'):
                        r[k] += prev[k]
                    r['seconds'] = round(r['seconds'] + prev['seconds'], 1)
                    r['dots_per_round'] = prev.get('dots_per_round', []) + r.get('dots_per_round', [])
                out['results'][m] = r
                run.log(f'{m}: {r["removed"]} removed (+{r["changed"]} revealed another), {r["no_change"]} without '
                        f'effect, {"FINISHED" if r["finished"] else str(r["left"]) + " left on the minimap"}, '
                        f'{r["views"]} views, {r["seconds"]}s')
            if not todo:
                break
    except Stop as exc:
        run.log(f'Stopped: {exc}')
        out['stopped'] = str(exc)
    finally:
        guard.close()
        out['seconds'] = round(time.perf_counter() - t0, 1)
        out['zoom'] = c.zoom
        out['clicks'] = c.records
        out['pans'] = p.history
        try:
            run.image('hud-after.png', hud(screen.grab(), layout))
        except Exception:
            pass
        run.json('clean.json', out)
    rep = report(out)
    run.text('clean-report.txt', rep)
    run.log('')
    run.log(rep)
    return out


def report(out):
    L = ['M3 cleaning report', '']
    for m, r in out['results'].items():
        if r.get('already'):
            L.append(f'{m:10} already finished (button greyed out) - skipped')
            continue
        L.append(f'{m:10} {r["removed"]} removed (+{r["changed"]} revealed another), {r["no_change"]} clicks without '
                 f'effect; minimap per round {r.get("dots_per_round")} -> '
                 f'{"FINISHED (button greyed out)" if r["finished"] else str(r["left"]) + " left"}; '
                 f'{r["views"]} views in {r["seconds"]}s')
    z = out.get('zoom', {})
    L.append('Zoom: ' + ', '.join(f'{m} {z[m]["zoom"]}x{" (measured)" if z[m].get("inliers") else ""}'
                                  for m in z if m != 'survey'))
    if out.get('stopped'):
        L.append(f'STOPPED: {out["stopped"]}')
    L.append(f'Total {out.get("seconds")}s after F6. Purity bar: hud-before.png / hud-after.png. '
             f'Clicks: clicks/NNN-mode-outcome.jpg')
    return '\n'.join(L)
