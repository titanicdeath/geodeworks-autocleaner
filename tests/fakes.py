"""Fake Windows backend + fake Roblox for offline tests. No real input is sent."""
import threading

import numpy as np

from geoclean import winapi as w


class FakeApi:
    """Simulates the cursor (with pointer acceleration on relative moves), keys,
    the foreground window, and a Roblox that updates its own pointer position
    only from the moves listed in `roblox_sees` (the hypothesis under test)."""

    HWND = 1234

    def __init__(self, cursor=(500, 500), rel_gain=1.0, roblox_sees=('rel', 'physical')):
        self.lock = threading.RLock()
        self.pos = np.array(cursor, float)
        self.roblox_pos = np.array(cursor, float)
        self.rel_gain = rel_gain
        self.roblox_sees = set(roblox_sees)
        self.events = []
        self.keys = set()
        self.fg = self.HWND
        self.rect = (0, 0, 1920, 1080)
        self.on_press = None      # callback(roblox_pos, button)
        self.on_release = None
        self.on_move = None       # callback(delta) when Roblox's pointer moves
        self.locked = False       # simulate a cursor lock while right-dragging
        self.key_script = []      # list of (vk, polls_down) consumed by key_down

    # input
    def send_mouse(self, flags, dx=0, dy=0):
        with self.lock:
            old = self.roblox_pos.copy()
            self._send(flags, dx, dy)
            if self.on_move and np.any(self.roblox_pos != old):
                self.on_move(self.roblox_pos - old)

    def _send(self, flags, dx, dy):
        with self.lock:
            self.events.append((flags, dx, dy))
            if flags & w.MOVE:
                if flags & w.ABSOLUTE:
                    left, top, width, height = self.virtual_bounds()
                    new = np.array([left + dx * (width - 1) / 65535, top + dy * (height - 1) / 65535])
                    if not self.locked:
                        self.pos = np.rint(new)
                    if 'abs' in self.roblox_sees:
                        self.roblox_pos = self.pos.copy()
                else:
                    if not self.locked:
                        self.pos = np.rint(self.pos + np.array([dx, dy]) * self.rel_gain)
                    if 'rel' in self.roblox_sees:
                        self.roblox_pos = self.pos.copy()
            for name, (down, up) in w.BUTTONS.items():
                if flags & down and self.on_press:
                    self.on_press(self.roblox_pos.copy(), name)
                if flags & up and self.on_release:
                    self.on_release(self.roblox_pos.copy(), name)

    def set_cursor(self, x, y):
        with self.lock:
            self.events.append(('SetCursorPos', x, y))
            self.pos = np.array([x, y], float)
            if 'setcursor' in self.roblox_sees:
                self.roblox_pos = self.pos.copy()

    def physical_move(self, x, y):
        with self.lock:
            self.pos = np.array([x, y], float)
            if 'physical' in self.roblox_sees:
                self.roblox_pos = self.pos.copy()

    def get_cursor(self):
        with self.lock:
            return (int(self.pos[0]), int(self.pos[1]))

    def virtual_bounds(self):
        return (0, 0, 3840, 1080)

    def press_key(self, vk, up_polls=3, down_polls=1, then=None):
        """Script a key tap: up for `up_polls` polls, down for `down_polls`, then
        call `then()` (e.g. the simulated user moving the mouse away)."""
        with self.lock:
            self.key_script.append([vk, up_polls, down_polls, then])

    def key_down(self, vk):
        with self.lock:
            for item in self.key_script:
                if item[0] != vk:
                    continue
                if item[1] > 0:
                    item[1] -= 1
                    return False
                if item[2] > 0:
                    item[2] -= 1
                    if item[2] == 0:
                        self.key_script.remove(item)
                        if item[3]:
                            threading.Timer(0.2, item[3]).start()
                    return True
                return False
            return vk in self.keys

    # windows
    def foreground(self):
        return self.fg

    def window_title(self, hwnd):
        return 'Roblox' if hwnd == self.HWND else 'Console'

    def window_exe(self, hwnd):
        return 'RobloxPlayerBeta.exe' if hwnd == self.HWND else 'cmd.exe'

    def window_rect(self, hwnd):
        return self.rect

    def client_rect(self, hwnd):
        return self.rect

    def environment(self, hwnd=None):
        return dict(dpi_awareness='fake', virtual_bounds=self.virtual_bounds(), swap_buttons=False)


class FakeScreen:
    """Serves `before` until the fake Roblox registers a left press on the target."""

    def __init__(self, before, after):
        self.before, self.after = before, after
        self.removed = False
        self.origin = (0, 0)

    def grab(self):
        return (self.after if self.removed else self.before).copy()

    def to_screen(self, p):
        return (int(round(p[0])), int(round(p[1])))

    def to_image(self, p):
        return (p[0], p[1])

    def info(self):
        return {}

    def close(self):
        pass


def make_slide(half, seed=0, px_per_block=9000, grain=4.0, content_radius=None):
    """Synthetic slide texture: beige ground, optional grain, blocks and lines.
    With content_radius, features only appear within that distance of the centre
    (live slides are empty near their edges)."""
    import cv2
    rng = np.random.default_rng(seed)
    size = 2 * half
    img = np.empty((size, size, 3), np.float32)
    img[:] = (150, 165, 178)
    if grain:
        img += cv2.GaussianBlur(rng.normal(0, grain, (size, size, 3)).astype(np.float32), (0, 0), 1.5)

    def spot():
        while True:
            c = rng.uniform(0, size, 2)
            if content_radius is None or np.hypot(*(c - half)) < content_radius:
                return c
    for _ in range(int(size * size / px_per_block)):
        c = spot()
        s = rng.uniform(25, 60)
        box = cv2.boxPoints(((float(c[0]), float(c[1])), (s, s), rng.uniform(0, 360))).astype(np.int32)
        cv2.fillPoly(img, [box], tuple(float(x) for x in rng.uniform(70, 140, 3)))
    for _ in range(int(size * size / (px_per_block * 7))):
        p = spot()
        q = p + rng.normal(0, 90, 2)
        cv2.line(img, tuple(p.astype(int).tolist()), tuple(q.astype(int).tolist()),
                 tuple(float(x) for x in rng.uniform(80, 230, 3)), 6)
    return np.clip(img, 0, 255).astype(np.uint8)


class FakeSlide:
    """A microscope whose slide is larger than the view.

    While right is held, every Roblox pointer move of delta moves the content by
    delta*gain (view by -delta*gain):
      quantum    - each event's movement is truncated to multiples of q (loss)
      max_event  - events larger than this are ignored (live: one big jump barely moved)
      bounds     - ('circle', r) or ('square', (hx, hy)): limits of the view centre
      overshoot  - the internal position may run this far past the bounds; the
                   rendered view is clamped but the minimap ring shows the internal
                   position, and pulling back first unwinds the overshoot (live: the
                   ring kept moving at a corner while the view had stopped)
      fps        - frame mode: the game samples the pointer once per frame. While
                   dragging, the view moves by (this frame's sample - last one), capped
                   at max_frame_delta px per frame. When the button goes down, the drag
                   starts from the PREVIOUS frame's sample, so a move to the drag
                   start that lands in the same frame as the press is counted as part
                   of the drag (the live M1 re-run failure).
    """

    def __init__(self, api, base, layout, gain=0.87, radius=600, scale=26.0, quantum=0,
                 start=(120, -80), seed=0, px_per_block=9000, bounds=None, overshoot=0.0,
                 max_event=None, grain=4.0, content_radius=None, fps=None, max_frame_delta=None,
                 ring_grid=None):
        """scale: slide px per minimap px, a number or (x, y). ring_grid: snap the
        drawn ring centre to this grid in minimap px (live: 0.5)."""
        self.ring_grid = ring_grid
        import cv2
        import time
        self.cv2, self.time = cv2, time
        self.fps, self.max_frame_delta = fps, max_frame_delta
        if fps:
            self.lock = threading.RLock()
            self.events = [(time.perf_counter(), np.array(api.roblox_pos, float), False)]
            self.t_tick = time.perf_counter()
            self.dragging, self.prev_sample, self.last_pos = False, None, None
        self.api, self.layout = api, layout
        self.gain, self.quantum = gain, quantum
        self.scale = np.broadcast_to(np.asarray(scale, float), (2,)).copy()
        self.bounds = bounds or ('circle', radius)
        self.overshoot, self.max_event = overshoot, max_event
        self.P = np.array(start, float)        # internal position (what the minimap shows)
        self.held = False
        cx, cy, r = layout.scope
        reach = radius if self.bounds[0] == 'circle' else float(np.hypot(*self.bounds[1]))
        self.half = int(reach + overshoot + r + 60)
        self.slide = make_slide(self.half, seed, px_per_block, grain, content_radius)
        self.base = base.copy()
        mx, my, mr = layout.minimap
        mm = v_disk(base.shape, (mx, my, mr * 0.93))
        self.base[mm] = np.median(base[mm], axis=0).astype(np.uint8)
        self.ys, self.xs = np.nonzero(layout.valid_mask(0.985))
        self.origin = (0, 0)
        api.on_press, api.on_release, api.on_move = self.press, self.release, self.move

    def _clamp(self, P, extra=0.0):
        kind, lim = self.bounds
        if kind == 'circle':
            n = np.linalg.norm(P)
            return P * (lim + extra) / n if n > lim + extra else P
        return np.clip(P, -np.array(lim) - extra, np.array(lim) + extra)

    @property
    def V(self):
        return self._clamp(self.P)

    @V.setter
    def V(self, value):
        self.P = np.asarray(value, float)

    # --- frame mode -----------------------------------------------------------
    def _record(self, down=None):
        with self.lock:
            now = self.time.perf_counter()
            self._advance(now)
            d = self.events[-1][2] if down is None else down
            self.events.append((now, np.array(self.api.roblox_pos, float), d))

    def _advance(self, now):
        dt = 1.0 / self.fps
        with self.lock:
            while self.t_tick + dt <= now:
                self.t_tick += dt
                while len(self.events) > 1 and self.events[1][0] <= self.t_tick:
                    self.events.pop(0)
                _, pos, down = self.events[0]
                if down and not self.dragging:
                    self.dragging = True
                    self.last_pos = self.prev_sample if self.prev_sample is not None else pos
                if self.dragging and down:
                    delta = pos - self.last_pos
                    n = np.linalg.norm(delta)
                    if self.max_frame_delta and n > self.max_frame_delta:
                        delta = delta * self.max_frame_delta / n
                    self.P = self._clamp(self.P - delta * self.gain, self.overshoot)
                    self.last_pos = pos
                if not down:
                    self.dragging = False
                self.prev_sample = pos

    # --- event mode -----------------------------------------------------------
    def press(self, pos, button):
        if button == 'right':
            if self.fps:
                self._record(True)
            self.held = True

    def release(self, pos, button):
        if button == 'right':
            if self.fps:
                self._record(False)
            self.held = False

    def move(self, delta):
        if self.fps:
            self._record()
            return
        if not self.held:
            return
        if self.max_event and np.abs(delta).max() > self.max_event:
            return
        m = np.asarray(delta, float) * self.gain * self._pan_scale()
        if self.quantum:
            m = np.trunc(m / self.quantum) * self.quantum
        # The internal position may run up to `overshoot` past the bounds; the view
        # (self.V) stays clamped, so pulling back first unwinds the overshoot.
        self.P = self._clamp(self.P - m, self.overshoot)

    def grab(self):
        cv2 = self.cv2
        if self.fps:
            self._advance(self.time.perf_counter())
        cx, cy, _ = self.layout.scope
        out = self._base_image().copy()
        vx, vy = np.round(self.V).astype(int)
        z = self._zoom()
        if z == 1.0:
            sy = self.ys - int(round(cy)) + self.half + vy
            sx = self.xs - int(round(cx)) + self.half + vx
        else:   # zoomed about the microscope centre (live: Bright 1.40x, Polarized 2.08x)
            sy = np.round((self.ys - cy) / z + self.half + self.V[1]).astype(int)
            sx = np.round((self.xs - cx) / z + self.half + self.V[0]).astype(int)
        out[self.ys, self.xs] = self._slide_image()[sy, sx]
        mx, my, _ = self.layout.minimap
        off = self.P / self.scale
        if self.ring_grid:
            off = np.round(off / self.ring_grid) * self.ring_grid
        c = (int(round((mx + off[0]) * 8)), int(round((my + off[1]) * 8)))
        cv2.circle(out, c, int(round(self._ring_radius() * 8)), (235, 235, 235), 2, cv2.LINE_AA, shift=3)
        self._draw_minimap(out)        # live: dots are drawn on top of the view ring
        return out

    # hooks for FakeScope
    def _zoom(self):
        return 1.0

    def _pan_scale(self):
        return 1.0

    def _ring_radius(self):
        return 19.0

    def _base_image(self):
        return self.base

    def _slide_image(self):
        return self.slide

    def _draw_minimap(self, out):
        pass

    def to_screen(self, p):
        return (int(round(p[0])), int(round(p[1])))

    def to_image(self, p):
        return (p[0], p[1])

    def info(self):
        return {}

    def close(self):
        pass


def v_disk(shape, circle):
    x, y, r = circle
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (xx - x) ** 2 + (yy - y) ** 2 < r * r


class FakeScope(FakeSlide):
    """FakeSlide with lighting modes, clickable mode buttons and minimap dots.

    Each imperfection belongs to one mode: in that mode it is drawn at full
    contrast (blocks with rock texture), in the other modes blended at `fade`
    opacity with no texture. The background colour changes per mode. Clicking a
    button (left press inside it) switches the mode; the active button is teal.
    The minimap shows a dot for each imperfection of the current mode.
    Ground truth: self.objects (slide coordinates).
    """

    MODE_BG = {'survey': (150, 165, 178), 'bright': (182, 194, 204), 'polarized': (128, 112, 132)}
    ZOOM = {'survey': 1.0, 'bright': 1.40, 'polarized': 2.08}

    def __init__(self, api, base, layout, n_objects=30, fade=0.5, obj_seed=7, spread=None, min_gap=120, **kw):
        import cv2
        from geoclean import modes as MO
        kw.setdefault('px_per_block', 10 ** 9)       # no random texture objects from make_slide
        kw.setdefault('grain', 1.5)
        super().__init__(api, base, layout, **kw)
        self.MO = MO
        self.mode = 'survey'
        self._fade = fade
        rng = np.random.default_rng(obj_seed)
        lim = spread or (self.half - layout.scope[2] - 40)
        self.objects = []
        names = ('survey', 'bright', 'polarized')
        tries = 0
        while len(self.objects) < n_objects and tries < 5000:
            tries += 1
            p = rng.uniform(-lim, lim, 2)
            if any(np.linalg.norm(p - o['S']) < min_gap for o in self.objects):
                continue
            kind = 'line' if rng.random() < 0.25 else 'block'
            mode = names[len(self.objects) % 3]
            if mode != 'polarized':
                kind = 'block'      # live so far: lines are Polarized; Survey = rock, Bright = smooth blocks
            self.objects.append(dict(S=p, mode=mode, kind=kind,
                                     size=float(rng.uniform(50, 66)), angle=float(rng.uniform(0, 180)),
                                     length=float(rng.uniform(120, 220)),
                                     color=self._colour(rng)))
        self.slides = {m: self._render(m, rng) for m in names}
        # Buttons drawn onto per-mode copies of the base frame.
        self.bases = {}
        cx, cy, r = layout.scope
        for m in names:
            img = self.base.copy()
            for k, (x, y) in enumerate(MO.button_centers(layout)):
                col = (80, 92, 41) if MO.MODES[k] == m else (39, 29, 25)
                cv2.rectangle(img, (int(x - 0.12 * r), int(y - 0.043 * r)), (int(x + 0.12 * r), int(y + 0.043 * r)),
                              col, -1)
            self.bases[m] = img
        self.clicks = []
        self.zoomed_drags = 0
        self.removed_log = []
        self.missed_clicks = []
        self.done = set()
        self.active_shown = True
        # Objects whose minimap dot only appears once their mode has been left and
        # re-entered (live: dots can hide under one another, so a pass can end
        # with the mode unfinished although the minimap looked empty).
        self.hidden_dots = set()
        self.left_modes = set()

    def _render(self, mode, rng):
        cv2 = self.cv2
        img = self.slide.astype(np.float32)
        img += np.array(self.MODE_BG[mode], np.float32) - np.array((150, 165, 178), np.float32)
        tex = rng.normal(0, 24, img.shape[:2]).astype(np.float32)   # interior texture ~2-3 like live rock
        tex = cv2.GaussianBlur(tex, (0, 0), 1.2)
        for o in self.objects:
            if o.get('removed'):
                continue
            R = int(max(o['size'], o['length'] / 2)) + 12
            c0 = np.round(o['S'] + self.half).astype(int)
            x0, y0 = c0 - R
            win = (slice(y0, y0 + 2 * R), slice(x0, x0 + 2 * R))
            m = np.zeros((2 * R, 2 * R), np.uint8)
            c = o['S'] + self.half - (x0, y0)
            if o['kind'] == 'block':
                box = cv2.boxPoints(((float(c[0]), float(c[1])), (o['size'], o['size']), o['angle'])).astype(np.int32)
                cv2.fillPoly(m, [box], 1)
            else:
                d = np.array([np.cos(np.radians(o['angle'])), np.sin(np.radians(o['angle']))]) * o['length'] / 2
                cv2.line(m, tuple((c - d).astype(int).tolist()), tuple((c + d).astype(int).tolist()), 1, 9)
            sel = m.astype(bool)
            sub = img[win]
            col = np.array(o['color'], np.float32)
            if o['mode'] == mode:
                sub[sel] = col + (tex[win][sel, None] if o['kind'] == 'block' else 0)
            else:
                sub[sel] = (1 - self.fade) * sub[sel] + self.fade * col
        return np.clip(img, 0, 255).astype(np.uint8)

    @property
    def fade(self):
        return self._fade

    def _colour(self, rng):
        # Game imperfections are designed to be visible: keep colours well away
        # from the Survey background, so their faded versions still show.
        bg = np.array(self.MODE_BG['survey'], float)
        while True:
            c = rng.uniform(40, 220, 3)
            if np.linalg.norm(c - bg) > 90:
                return tuple(float(x) for x in c)

    def _zoom(self):
        return self.ZOOM[self.mode]

    def _pan_scale(self):
        # Live assumption (M3): in a zoomed mode a cursor px moves the slide 1/zoom
        # as much. The 1.15 is a deliberate error the cleaner must learn.
        return 1.0 if self.mode == 'survey' else 1.15 / self.ZOOM[self.mode]

    def _ring_radius(self):
        from geoclean import vision as v
        return v.RING_R_FRAC_MODE[self.mode] * self.layout.minimap[2]

    def _base_image(self):
        cv2 = self.cv2
        img = self.base.copy()
        cx, cy, r = self.layout.scope
        for k, (x, y) in enumerate(self.MO.button_centers(self.layout)):
            name = self.MO.MODES[k]
            if name in self.done:
                col = (64, 56, 52)                      # greyed: mode finished (live)
            elif name == self.mode and self.active_shown:
                col = (80, 92, 41)
            else:
                col = (39, 29, 25)
            cv2.rectangle(img, (int(x - 0.12 * r), int(y - 0.043 * r)), (int(x + 0.12 * r), int(y + 0.043 * r)),
                          col, -1)
        return img

    def _slide_image(self):
        return self.slides[self.mode]

    def _draw_minimap(self, out):
        cv2 = self.cv2
        mx, my, _ = self.layout.minimap
        for o in self.objects:
            if o.get('removed'):
                continue
            if o['mode'] != self.mode:
                continue
            if id(o) in self.hidden_dots and self.mode not in self.left_modes:
                continue
            p = (int(round((mx + o['S'][0] / self.scale[0]) * 8)), int(round((my + o['S'][1] / self.scale[1]) * 8)))
            cv2.circle(out, p, 3 * 8, (193, 226, 121), -1, cv2.LINE_AA, shift=3)
            cv2.circle(out, p, 3 * 8, (64, 67, 39), 1, cv2.LINE_AA, shift=3)

    def _hit(self, o, S):
        if o['kind'] == 'block':
            d = S - o['S']
            a = np.radians(o['angle'])
            u = np.array([np.cos(a), np.sin(a)])
            n = np.array([-u[1], u[0]])
            return abs(np.dot(d, u)) <= o['size'] / 2 and abs(np.dot(d, n)) <= o['size'] / 2
        d = np.array([np.cos(np.radians(o['angle'])), np.sin(np.radians(o['angle']))])
        t = np.clip(np.dot(S - o['S'], d), -o['length'] / 2, o['length'] / 2)
        return np.linalg.norm(S - (o['S'] + t * d)) <= 5.0

    def _click_slide(self, pos):
        """A left click removes the topmost object of the CURRENT mode under it."""
        cx, cy, _ = self.layout.scope
        S = self.V + (pos - (cx, cy)) / self._zoom()
        for o in reversed(self.objects):
            if not o.get('removed') and o['mode'] == self.mode and self._hit(o, S):
                o['removed'] = True
                rng = np.random.default_rng(1)
                self.slides = {m: self._render(m, rng) for m in self.slides}
                self.removed_log.append((self.mode, np.round(S).tolist()))
                if not any(not q.get('removed') and q['mode'] == self.mode for q in self.objects):
                    self.done.add(self.mode)            # live: button greys, no mode highlighted
                    self.active_shown = False
                return True
        self.missed_clicks.append((self.mode, np.round(S).tolist()))
        return False

    def press(self, pos, button):
        if button == 'left':
            cx, cy, r = self.layout.scope
            on_button = False
            for k, (x, y) in enumerate(self.MO.button_centers(self.layout)[:3]):
                if abs(pos[0] - x) < 0.12 * r and abs(pos[1] - y) < 0.043 * r:
                    if self.MO.MODES[k] not in self.done:   # a greyed button does nothing
                        if self.MO.MODES[k] != self.mode:
                            self.left_modes.add(self.mode)
                        self.mode = self.MO.MODES[k]
                        self.active_shown = True
                    on_button = True
            self.clicks.append((tuple(pos), self.mode))
            if not on_button:
                self._click_slide(np.asarray(pos, float))
            return
        if button == 'right' and self.mode != 'survey':
            self.zoomed_drags += 1          # panning while zoomed: the M2 re-run bug
        super().press(pos, button)
