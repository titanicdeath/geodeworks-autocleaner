"""Lighting-mode buttons (Survey / Bright / Polarized / Darkfield) under the microscope.

Their positions are fixed relative to the microscope circle (identical in every
sample): y = cy + 0.93 r, x = cx + (-0.386, -0.129, +0.129, +0.386) r. Inactive
buttons are dark navy (BGR ~39,29,25); the active one is teal (BGR ~80,92,41).
A mode click is only sent after the buttons have been recognised, and it is
verified by watching the teal highlight move.
"""
import time

import cv2
import numpy as np

from .inputs import STRATEGIES
from .safety import Stop

MODES = ('survey', 'bright', 'polarized', 'darkfield')
_X = (-0.386, -0.129, 0.129, 0.386)
_Y = 0.93


def button_centers(layout):
    cx, cy, r = layout.scope
    return [(cx + dx * r, cy + _Y * r) for dx in _X]


def read_buttons(frame, layout):
    """Returns dict(present=[bool]*4, states=[...]*4, active=name or None,
    done=[names], colors=[BGR]*4).

    States: 'active' (teal, BGR ~80,92,41), 'enabled' (navy ~39,29,25) and
    'disabled' (grey ~64,56,52). Live 2026-09-27: once every imperfection of a mode
    is cleaned, the game greys that mode's button and it can no longer be
    selected - the definitive "this mode is done" signal. (Darkfield, locked
    behind a game pass, keeps a navy background.)
    Samples a patch on the left part of each button (clear of its label text)."""
    cx, cy, r = layout.scope
    out = dict(present=[], states=[], active=None, done=[], colors=[])
    for k, (x, y) in enumerate(button_centers(layout)):
        x0, y0 = int(x - 0.085 * r), int(y - 0.02 * r)
        patch = frame[y0:y0 + max(4, int(0.04 * r)), x0:x0 + max(4, int(0.035 * r))]
        if patch.size == 0:
            out['present'].append(False)
            out['states'].append('unknown')
            out['colors'].append(None)
            continue
        b, g, rr = np.median(patch.reshape(-1, 3), axis=0)
        hsv = cv2.cvtColor(np.uint8([[[b, g, rr]]]), cv2.COLOR_BGR2HSV)[0, 0]
        spread = max(b, g, rr) - min(b, g, rr)
        if g > b and g > rr + 25 and hsv[1] > 90 and hsv[2] > 60:
            state = 'active'
        elif 50 <= hsv[2] < 110 and spread <= 22:
            state = 'disabled'
        elif 12 < hsv[2] < 50 and b > rr:
            state = 'enabled'
        else:
            state = 'unknown'
        out['states'].append(state)
        out['present'].append(state != 'unknown')
        out['colors'].append([int(b), int(g), int(rr)])
    if out['states'].count('active') == 1:
        out['active'] = MODES[out['states'].index('active')]
    out['done'] = [MODES[k] for k in range(3) if out['states'][k] == 'disabled']
    return out


class ModeDone(Stop):
    """The mode's button is greyed out: everything in that mode is cleaned."""


def switch_mode(name, driver, guard, screen, layout, log=print, timeout=2.0, steady_timeout=1.0):
    """Click a mode button and wait until it is highlighted and the view is steady.

    Live: Bright zooms the view in ~1.40x and Polarized ~2.08x (about the
    microscope centre), with an animation; going back zooms out again."""
    if name not in MODES[:3]:
        raise ValueError(f'Unsupported mode {name!r}')
    state = read_buttons(screen.grab(), layout)
    if state['active'] == name:
        return dict(clicked=False, active=name, waited_ms=0)
    if sum(state['present']) < 3:
        raise Stop(f'Mode buttons not recognised (states={state["states"]}, colors={state["colors"]}); '
                   'not clicking.')
    if name in state['done']:
        raise ModeDone(f'{name} is finished (its button is greyed out).')
    # No mode highlighted is fine: live, after a mode is finished the game can leave
    # none selected.
    x, y = button_centers(layout)[MODES.index(name)]
    driver.move_to(screen.to_screen((x, y)), STRATEGIES['S3-abs-wiggle'])
    guard.sleep(0.05)
    driver.click('left', 0.08)
    # Move off the button so hover effects don't change what we capture.
    cx, cy, _ = layout.scope
    driver.move_to(screen.to_screen((cx, cy)), STRATEGIES['S3-abs-wiggle'])
    t0 = time.perf_counter()
    prev, switched_at = None, None
    while time.perf_counter() - t0 < timeout + steady_timeout:
        guard.sleep(0.04)
        f = screen.grab()
        if read_buttons(f, layout)['active'] == name:
            switched_at = switched_at or time.perf_counter()
            # Wait for the view itself to settle (lighting transitions)...
            small = cv2.resize(f, None, fx=0.25, fy=0.25)
            if prev is not None and cv2.absdiff(small, prev).max() < 12:
                return dict(clicked=True, active=name, steady=True,
                            waited_ms=round((time.perf_counter() - t0) * 1000))
            prev = small
            # ...but a constantly animated scene must not stall the run.
            if time.perf_counter() - switched_at > steady_timeout:
                log(f'    note: view still changing {steady_timeout:.1f}s after switching to {name}; continuing')
                return dict(clicked=True, active=name, steady=False,
                            waited_ms=round((time.perf_counter() - t0) * 1000))
        elif time.perf_counter() - t0 > timeout:
            break
    raise Stop(f'Clicked the {name} button but the mode did not change within {timeout}s.')
