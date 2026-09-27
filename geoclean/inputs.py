"""Interchangeable mouse movement strategies.

Why several: v0.3/v0.4 put the cursor exactly on a block (GetCursorPos agreed)
but Roblox ignored the click, while a press with a *physically* placed cursor
worked. So Roblox's idea of the pointer apparently didn't follow SetCursorPos or
absolute SendInput moves. Relative moves (MOUSEEVENTF_MOVE without ABSOLUTE)
look like a real mouse, including in raw input, so they are the main candidate.

Relative moves are scaled by Windows pointer speed/acceleration, so they run
closed-loop: send a small step, read the cursor back, correct.
"""
import time
from dataclasses import dataclass, asdict

import numpy as np

from . import winapi as w
from .safety import Stop


@dataclass(frozen=True)
class Strategy:
    name: str
    move: str                 # none | setcursor | abs | abs_path | rel
    wiggle: bool = False      # +1/-1 relative nudge after arriving
    nocoalesce: bool = True   # MOUSEEVENTF_MOVE_NOCOALESCE on SendInput moves
    dwell: float = 0.15       # pause between arriving and pressing
    hold: float = 0.08        # button down -> up
    steps: int = 12           # abs_path: number of absolute events
    step_px: int = 12         # rel: max counts per event
    step_delay: float = 0.008
    description: str = ''

    def as_dict(self):
        return asdict(self)


# Order matters for M0: control first (proves the block is cleanable and the
# press works), then a replica of the failing v0.4 path, then the candidates.
M0_STRATEGIES = [
    Strategy('S0-control', 'none', dwell=0.05, hold=0.08,
             description='No movement. You keep the cursor on the block; the tool only presses.'),
    Strategy('S1-abs-path', 'abs_path', nocoalesce=False, dwell=0.20, hold=0.10, steps=6,
             description='Replica of v0.4: 6 absolute SendInput moves (expected to fail).'),
    Strategy('S2-setcursor-wiggle', 'setcursor', wiggle=True, dwell=0.15, hold=0.08,
             description='SetCursorPos jump, then a 1-count relative wiggle.'),
    Strategy('S3-abs-wiggle', 'abs', wiggle=True, dwell=0.15, hold=0.08,
             description='One absolute SendInput move, then a relative wiggle.'),
    Strategy('S4-rel', 'rel', dwell=0.15, hold=0.08,
             description='Relative SendInput steps only, closed-loop.'),
    Strategy('S5-rel-wiggle-slow', 'rel', wiggle=True, step_px=6, step_delay=0.012,
             dwell=0.30, hold=0.15,
             description='Slow relative steps, wiggle, long dwell and hold.'),
]

STRATEGIES = {s.name: s for s in M0_STRATEGIES}


class MouseDriver:
    def __init__(self, guard, api, log=print):
        self.guard = guard
        self.api = api
        self.log = log

    def cursor(self):
        return tuple(self.api.get_cursor())

    # --- primitives --------------------------------------------------------
    def _abs(self, x, y, nocoalesce):
        dx, dy = w.normalize_absolute(int(x), int(y), self.api.virtual_bounds())
        flags = w.MOVE | w.ABSOLUTE | w.VIRTUALDESK | (w.MOVE_NOCOALESCE if nocoalesce else 0)
        self.guard.send(flags, dx, dy)

    def _rel(self, dx, dy, nocoalesce=True):
        if dx or dy:
            self.guard.send(w.MOVE | (w.MOVE_NOCOALESCE if nocoalesce else 0), dx, dy)

    def rel_closed_loop(self, target, step_px=12, step_delay=0.008, nocoalesce=True,
                        max_events=800, stall_limit=40):
        """Relative steps until the cursor is exactly on target (or can't get closer)."""
        target = np.array(target, int)
        events, stall = 0, 0
        best = None
        while events < max_events:
            cur = np.array(self.cursor())
            err = target - cur
            dist = int(np.abs(err).max())
            if best is None or dist < best:
                best, stall = dist, 0
            else:
                stall += 1
            if dist == 0 or stall >= stall_limit:
                break
            step = np.clip(err, -step_px, step_px)
            self._rel(int(step[0]), int(step[1]), nocoalesce)
            events += 1
            self.guard.sleep(step_delay)
        return events

    def rel_open_loop(self, dx, dy, step_px=8, step_delay=0.008, nocoalesce=True):
        """Send exactly (dx, dy) counts in small steps without feedback (cursor may be locked)."""
        remaining = np.array([dx, dy], int)
        events = 0
        while remaining.any():
            step = np.clip(remaining, -step_px, step_px)
            self._rel(int(step[0]), int(step[1]), nocoalesce)
            remaining -= step
            events += 1
            self.guard.sleep(step_delay)
        return events

    def wiggle(self, target, nocoalesce=True):
        for dx in (1, -1):
            self._rel(dx, 0, nocoalesce)
            self.guard.sleep(0.015)
        for dy in (1, -1):
            self._rel(0, dy, nocoalesce)
            self.guard.sleep(0.015)
        # Pointer acceleration can leave a 1 px residue; fix it relatively.
        return self.rel_closed_loop(target, step_px=2, step_delay=0.01, nocoalesce=nocoalesce,
                                    max_events=40, stall_limit=10)

    # --- strategies --------------------------------------------------------
    def move_to(self, target, s: Strategy):
        """Move with strategy s. Returns a trace dict for diagnostics."""
        t0 = time.perf_counter()
        start = self.cursor()
        trace = dict(strategy=s.name, target=list(map(int, target)), start=list(start))
        if self.guard.dry_run:
            trace.update(dry_run=True, end=list(start), final_error_px=None)
            return trace
        x, y = map(int, target)
        events = 0
        if s.move == 'none':
            pass
        elif s.move == 'setcursor':
            self.guard.set_cursor(x, y)
            events = 1
            self.guard.sleep(0.01)
        elif s.move == 'abs':
            self._abs(x, y, s.nocoalesce)
            events = 1
            self.guard.sleep(0.01)
        elif s.move == 'abs_path':
            a, b = np.array(start, float), np.array([x, y], float)
            for i in range(1, s.steps + 1):
                px, py = np.rint(a + (b - a) * i / s.steps).astype(int)
                self._abs(px, py, s.nocoalesce)
                events += 1
                self.guard.sleep(0.01)
        elif s.move == 'rel':
            events = self.rel_closed_loop((x, y), s.step_px, s.step_delay, s.nocoalesce)
        else:
            raise ValueError(f'Unknown move method {s.move!r}')
        arrived = self.cursor()
        wiggle_events = self.wiggle((x, y), s.nocoalesce) + 4 if s.wiggle else 0
        end = self.cursor()
        trace.update(events=events, wiggle_events=wiggle_events, arrived=list(arrived),
                     end=list(end), final_error_px=float(np.hypot(end[0] - x, end[1] - y)),
                     move_ms=round((time.perf_counter() - t0) * 1000, 1))
        return trace

    def click(self, button='left', hold=0.08):
        down, up = w.BUTTONS[button]
        self.guard.send(down)
        try:
            self.guard.sleep(hold)
        finally:
            self.guard.send(up)

    def press(self, button):
        self.guard.send(w.BUTTONS[button][0])

    def release(self, button):
        self.guard.send(w.BUTTONS[button][1])


def strategy_list(names):
    if not names:
        return list(M0_STRATEGIES)
    out = []
    for n in names:
        match = [s for s in M0_STRATEGIES if s.name == n or s.name.split('-')[0] == n]
        if not match:
            raise Stop(f'Unknown strategy {n!r}. Known: {", ".join(STRATEGIES)}')
        out.append(match[0])
    return out
