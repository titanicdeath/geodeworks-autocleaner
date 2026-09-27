"""Right-drag panning with closed-loop position tracking.

Coordinates
-----------
V (view position) is the centre of the current view in *slide pixels* (screen
pixels at the current zoom), measured from the minimap centre, which is taken
to be the slide centre. A content shift s on screen moves the view by -s:
    V_after = V_before - s
The minimap view ring gives an absolute but coarse V (ring offset * ring_scale).
Image registration gives precise relative moves. Panner.V fuses both.

M0 found that the game accumulates cursor deltas while right is held (with
some loss), so every drag is measured, never assumed.
"""
import json
import time
from dataclasses import dataclass, field, asdict

import numpy as np

from . import register as R
from .inputs import Strategy

# Moving to a drag's start point. Live M1 re-runs showed that if the game hasn't
# processed this move before the button goes down, the jump from the previous
# drag's end is counted as part of the new drag. So: wiggle (relative events),
# and wait whole game frames between move, press, motion and release.
ANCHOR_MOVE = Strategy('anchor', 'abs', wiggle=True, dwell=0.0)
# anchor/press/release are "frame gaps" (>= 2 game frames); pan-calibrate rescales
# them from the measured latency. event = interval between held-cursor events.
DEFAULT_TIMING = dict(anchor=0.08, press=0.08, event=0.005, step_delay=0.008, settle=0.15,
                      release=0.08)


def frame_gaps(latency_s):
    """Waits that safely span >= 2 game frames, from the measured view latency
    (latency is roughly 1-2 frames)."""
    gap = float(np.clip(1.5 * latency_s, 0.06, 0.2))
    return dict(anchor=gap, press=gap, release=gap, settle=float(np.clip(1.5 * latency_s + 0.03, 0.08, 0.4)))


@dataclass
class PanModel:
    G: list = field(default_factory=lambda: [[0.87, 0.0], [0.0, 0.87]])  # content px per cursor px
    ring_scale: list = field(default_factory=lambda: [26.2, 25.0])  # slide px per ring px, (x, y)
    ring_blend: float = 0.25          # how strongly each drag pulls the tracked V toward the ring
    style: str = 'v600'               # held-cursor motion: vN = N px/s (see _move_held)
    max_cursor: float = 420.0         # longest cursor movement in one drag
    timing: dict = field(default_factory=lambda: dict(DEFAULT_TIMING))

    def predict(self, cursor_delta):
        return np.asarray(self.G, float) @ np.asarray(cursor_delta, float)

    def cursor_for(self, content_shift):
        return np.linalg.solve(np.asarray(self.G, float), np.asarray(content_shift, float))

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        keys = cls.__dataclass_fields__
        return cls(**{k: d[k] for k in keys if k in d})


def save_model(path, model, extra=None):
    data = dict(kind='geoclean-pan-calibration', version=1,
                saved=time.strftime('%Y-%m-%d %H:%M:%S'), model=model.as_dict())
    data.update(extra or {})
    path.write_text(json.dumps(data, indent=2, default=float), encoding='utf-8')


def load_model(path):
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('kind') != 'geoclean-pan-calibration':
        raise ValueError(f'{path} is not a pan calibration file.')
    return PanModel.from_dict(data['model']), data


def fit_gain(cursor, shift):
    """Least-squares G with shift = G @ cursor. Returns (G, rms, relative_rms)."""
    C, S = np.asarray(cursor, float), np.asarray(shift, float)
    Gt, *_ = np.linalg.lstsq(C, S, rcond=None)
    G = Gt.T
    resid = S - C @ Gt
    rms = float(np.sqrt(np.mean(np.sum(resid ** 2, axis=1))))
    rel = rms / max(1e-6, float(np.sqrt(np.mean(np.sum(S ** 2, axis=1)))))
    return G, rms, rel


class Panner:
    def __init__(self, guard, driver, api, screen, layout, model, log=print, record=None):
        self.guard, self.driver, self.api, self.screen = guard, driver, api, screen
        self.layout, self.model, self.log = layout, model, log
        self.record = record            # callback(dict, frame) for diagnostics
        self.frame, self.ring, self.V = None, None, None
        # Current lighting mode's zoom (Bright 1.40x, Polarized 2.08x) and minimap
        # view-ring size; quick pans in a zoomed mode move the slide ~1/zoom as much
        # per cursor px. gain_adj learns the exact per-mode factor from the ring.
        self.zoom, self.ring_frac, self.mode_key = 1.0, None, 'survey'
        self.gain_adj = {}
        self.history = []
        cx, cy, r = layout.scope
        # Cursor must stay over the slide: away from rim, HUD and mode buttons.
        self.box = (cx - 0.78 * r, cy - 0.68 * r, cx + 0.78 * r, cy + 0.76 * r)

    # --- observation --------------------------------------------------------
    def observe(self):
        self.frame = self.screen.grab()
        self.ring = R.ring_center(self.frame, self.layout, self.ring_frac)
        return self.frame

    def v_from_ring(self, ring=None):
        ring = self.ring if ring is None else ring
        if ring is None or not self.layout.minimap:
            return None
        mx, my, _ = self.layout.minimap
        return (np.asarray(ring) - (mx, my)) * R.scale_vec(self.model.ring_scale)

    def start(self):
        self.observe()
        self.V = self.v_from_ring()
        if self.V is None:
            self.V = np.zeros(2)
            self.log('WARNING: minimap view ring not found; position starts at (0,0) (relative only).')
        return self.V

    # --- one drag -----------------------------------------------------------
    def window_box(self):
        """Where a held drag may END: most of the Roblox window (the user: dragging
        to the side of the display keeps panning). Clear of the top bar/HUD row and
        the bottom edge."""
        w, h = self.layout.size
        return (40.0, 90.0, w - 40.0, h - 70.0)

    def wide_limits(self):
        """Per axis, the most negative / positive cursor move that starts inside the
        microscope box and ends inside the window box."""
        A0x, A0y, A1x, A1y = self.box
        B0x, B0y, B1x, B1y = self.window_box()
        return np.array([B0x - A1x, B0y - A1y]), np.array([B1x - A0x, B1y - A0y])

    def _endpoints_wide(self, d):
        d = np.asarray(d, float)
        cx, cy, _ = self.layout.scope
        A0, A1 = np.array(self.box[:2]), np.array(self.box[2:])
        B0, B1 = np.array(self.window_box()[:2]), np.array(self.window_box()[2:])
        lo = np.maximum(A0, B0 - d)
        hi = np.minimum(A1, B1 - d)
        if np.any(lo > hi + 1e-6):
            raise ValueError(f'Cursor move {d} does not fit in the window.')
        a = np.clip(np.array([cx, cy]) - d / 2, lo, hi)
        return a, a + d

    def _endpoints(self, d):
        cx, cy, _ = self.layout.scope
        d = np.asarray(d, float)
        x0, y0, x1, y1 = self.box
        if abs(d[0]) > x1 - x0 or abs(d[1]) > y1 - y0:
            raise ValueError(f'Cursor move {d} does not fit inside the microscope.')
        # Centre the motion on the microscope, then clamp so both ends stay in the box.
        lo = np.array([x0, y0]) - np.minimum(d, 0)
        hi = np.array([x1, y1]) - np.maximum(d, 0)
        a = np.clip(np.array([cx, cy]) - d / 2, lo, hi)
        return a, a + d

    def _move_held(self, start, end, style):
        """Move the held cursor from start to end.

        vN    - at N px/s, one event every timing['event'] s. The game appears to cap
                how far the view moves per frame (one big jump barely pans; 3 px
                steps pan ~1:1), so speed, not step size, is what matters.
        stepN - N px per event every timing['step_delay'] s (M1 first run).
        jump  - a single event.
        """
        start, end = np.asarray(start, float), np.asarray(end, float)
        dist = float(np.abs(end - start).max())
        T = self.model.timing
        if style == 'jump':
            n, dt = 1, 0.0
        elif style.startswith('step'):
            n, dt = max(1, int(np.ceil(dist / float(style[4:])))), T['step_delay']
        elif style.startswith('v'):
            dt = T['event']
            n = max(1, int(np.ceil(dist / (float(style[1:]) * dt))))
        else:
            raise ValueError(f'Unknown drag style {style!r}')
        t0 = time.perf_counter()
        for i in range(1, n + 1):
            p = self.screen.to_screen(start + (end - start) * i / n)
            self.driver._abs(p[0], p[1], True)
            if i < n:
                # Pace against the clock so sleep overshoot doesn't slow the drag.
                self.guard.sleep(max(0.0, t0 + i * dt - time.perf_counter()))
        return n

    def drag(self, cursor_delta, style=None, wide=False):
        """Press right, move the cursor by cursor_delta, release. No measuring.
        wide: the drag may end outside the microscope (anywhere in the window)."""
        style = style or self.model.style
        T = self.model.timing
        a, b = self._endpoints_wide(cursor_delta) if wide else self._endpoints(cursor_delta)
        self.driver.move_to(self.screen.to_screen(a), ANCHOR_MOVE)
        self.guard.sleep(T['anchor'])
        t0 = time.perf_counter()
        self.driver.press('right')
        try:
            self.guard.sleep(T['press'])
            if not style.endswith('np'):
                # Primer: live, a lone held-cursor jump either counted fully or not at
                # all, and stepped drags lost ~1-2 events' worth - as if the first
                # move after the press can be swallowed. Let it swallow 1 px instead.
                u = np.asarray(cursor_delta, float)
                u = u / max(1e-9, np.abs(u).max())
                p = self.screen.to_screen(a + u)
                self.driver._abs(p[0], p[1], True)
                self.guard.sleep(T['press'])
            events = self._move_held(a, b, style.removesuffix('np'))
            self.guard.sleep(T['settle'])
        finally:
            if 'right' in self.guard.held:
                self.driver.release('right')
        self.guard.sleep(T['release'])
        return dict(anchor=a.round(1).tolist(), end=b.round(1).tolist(), events=events,
                    held_ms=round((time.perf_counter() - t0) * 1000))

    def pan_cursor(self, cursor_delta, style=None, tag=''):
        """Drag and measure. Updates V. Returns a record dict."""
        if self.frame is None:
            self.start()
        before, ring_b, V_b = self.frame, self.ring, self.V.copy()
        predicted = self.model.predict(cursor_delta)
        info = self.drag(cursor_delta, style)
        after = self.observe()
        shift, score, source = R.measure_shift(before, after, self.layout, ring_b, self.ring,
                                               self.model.ring_scale, predicted)
        if source == 'none':
            shift, source = predicted, 'model'   # unmeasurable: assume the model, not "no move"
        V_chain = V_b - shift
        V_ring = self.v_from_ring()
        # Fuse: image registration is precise per drag but its small errors add up
        # (live: 28-43 px after ~100 drags); the ring is absolute and unbiased but
        # coarse (0.5 px grid). A gentle pull toward the ring on every drag bounds drift.
        if V_ring is None:
            self.V = V_chain
        elif source == 'minimap':
            self.V = V_ring
        else:
            self.V = V_chain + self.model.ring_blend * (V_ring - V_chain)
        rec = dict(n=len(self.history) + 1, tag=tag, style=style or self.model.style,
                   cursor=np.round(cursor_delta, 1).tolist(), predicted=np.round(predicted, 1).tolist(),
                   shift=np.round(shift, 1).tolist(), score=round(score, 3), source=source,
                   V_before=np.round(V_b, 1).tolist(), V=np.round(self.V, 1).tolist(),
                   V_chain=np.round(V_chain, 1).tolist(),
                   V_ring=None if V_ring is None else np.round(V_ring, 1).tolist(),
                   ring=None if self.ring is None else np.round(self.ring, 2).tolist(),
                   ring_before=None if ring_b is None else np.round(ring_b, 2).tolist(), **info)
        # Gross disagreement (a wrong match that slipped through): jump to the ring.
        limit = 60 if source == 'image' else 40
        if V_ring is not None and np.linalg.norm(self.V - V_ring) > limit:
            rec['resynced_from'] = rec['V']
            self.V = V_ring
            rec['V'] = np.round(self.V, 1).tolist()
            self.log(f'    note: tracked position disagreed with the minimap by '
                     f'{np.linalg.norm(np.array(rec["resynced_from"]) - V_ring):.0f}px; resynced.')
        self.history.append(rec)
        if self.record:
            self.record(rec, after)
        return rec

    def pan_content(self, content_shift, style=None, tag=''):
        d = self.model.cursor_for(content_shift)
        n = np.abs(d).max() / self.model.max_cursor
        if n > 1:
            d = d / n
        return self.pan_cursor(d, style, tag)

    def set_mode(self, key, zoom, ring_frac):
        """Tell the panner which lighting mode the view is in (zoom, ring size)."""
        self.mode_key, self.zoom, self.ring_frac = key, float(zoom), ring_frac
        self.observe()
        V_ring = self.v_from_ring()
        if V_ring is not None:
            self.V = V_ring

    def quick_pan_to(self, target, tol=60.0, max_drags=3, limit=None, tag='quick', style=None, max_cursor=None,
                     wide=False):
        """Fast approach for cleaning: targets are re-detected on screen anyway, so
        position comes straight from the minimap ring (+-7 px) after each drag - no
        image registration - and one drag usually suffices. Works in the zoomed
        modes too (live: a finished mode's button greys out, so Survey may no
        longer be available for panning)."""
        target = np.asarray(target, float)
        if limit is not None:
            target = np.clip(target, -np.asarray(limit, float), np.asarray(limit, float))
        x0, y0, x1, y1 = self.box
        span = 0.96 * np.array([x1 - x0, y1 - y0])
        drags = 0
        for _ in range(max_drags):
            err = target - self.V
            if np.linalg.norm(err) <= tol:
                break
            adj = self.gain_adj.get(self.mode_key, 1.0) / self.zoom     # slide px per survey-model px
            d = self.model.cursor_for(-err) / adj
            n = np.abs(d).max() / (max_cursor or self.model.max_cursor)
            if n > 1:
                d = d / n
            if wide:
                neg, pos = self.wide_limits()
                lim = np.where(d >= 0, 0.97 * pos, 0.97 * -neg)
                fit = np.abs(d) / np.maximum(lim, 1.0)
            else:
                fit = np.abs(d) / span              # never longer than the safe cursor box
            if fit.max() > 1:
                d = d / fit.max()
            V_b = self.V.copy()
            info = self.drag(d, style, wide=wide)
            self.observe()
            drags += 1
            V_ring = self.v_from_ring()
            want = -np.asarray(self.model.predict(d)) * adj
            self.V = V_ring if V_ring is not None else V_b + want
            got = self.V - V_b
            ratio = float(np.dot(got, want) / max(1e-6, np.dot(want, want))) if V_ring is not None else None
            if ratio is not None and np.linalg.norm(want) > 120 and 0.3 < ratio < 3.0:
                # Learn this mode's real gain (blend, so one odd drag can't derail it).
                self.gain_adj[self.mode_key] = self.gain_adj.get(self.mode_key, 1.0) * (0.5 + 0.5 * ratio)
            self.history.append(dict(n=len(self.history) + 1, tag=tag, mode=self.mode_key, zoom=self.zoom,
                                     cursor=np.round(d, 1).tolist(),
                                     V_before=np.round(V_b, 1).tolist(), V=np.round(self.V, 1).tolist(),
                                     style=style or self.model.style,
                                     moved_ratio=None if ratio is None else round(ratio, 3),
                                     gain_adj=round(self.gain_adj.get(self.mode_key, 1.0), 3),
                                     source='minimap' if V_ring is not None else 'model', **info))
        return dict(target=target.tolist(), V=np.round(self.V, 1).tolist(), drags=drags,
                    error=round(float(np.linalg.norm(target - self.V)), 1))

    def pan_to(self, target, tol=6.0, max_drags=12, tag='pan_to'):
        """Closed loop: drag until the tracked view position is within tol of target."""
        target = np.asarray(target, float)
        recs, stuck, no_gain = [], False, 0
        for _ in range(max_drags):
            err = target - self.V
            if np.linalg.norm(err) <= tol:
                break
            recs.append(self.pan_content(-err, tag=tag))   # view += err  <=>  content -= err
            new_err = np.linalg.norm(target - self.V)
            if (recs[-1]['source'] in ('image', 'minimap') and np.linalg.norm(recs[-1]['shift']) < 1.0
                    and np.linalg.norm(err) > 3 * tol):
                stuck = True                                # at an edge: target unreachable
                break
            # Two drags in a row that don't help: the remaining error is below what
            # a drag can resolve (lossy/quantized panning). Accept where we are.
            no_gain = no_gain + 1 if new_err > np.linalg.norm(err) - 1.0 else 0
            if no_gain >= 2:
                break
        return dict(target=target.tolist(), V=np.round(self.V, 1).tolist(),
                    error=round(float(np.linalg.norm(target - self.V)), 2), drags=len(recs),
                    stuck=stuck)
