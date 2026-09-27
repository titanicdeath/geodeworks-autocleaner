"""M2: map the whole slide (hands-off, under a minute).

1. At the start position, read the minimap in each lighting mode (it shows that
   mode's remaining imperfections as dots). Bright and Polarized ZOOM the view
   (~1.40x, ~2.08x), so no panning or view capture happens in those modes; the
   tool switches back to Survey and checks the view came back unchanged.
2. Pan a serpentine grid over the slide in Survey mode only (the mode the pan
   calibration was made in) and save one frame per stop.
mapping.analyze() then stitches, detects every imperfection (all are visible in
Survey: own ones at full contrast, the others faded) and assigns each one a mode
by matching it to the minimap dots.
"""
import time

import numpy as np

from . import mapping as MP
from . import modes as MO
from . import register as R
from . import vision as v
from . import winapi as w
from .diag import ROOT
from .inputs import MouseDriver
from .m0 import setup_session, wait_key, wait_still
from .pan import Panner, load_model
from .safety import Stop

CALIBRATION = ROOT / 'pan_calibration.json'


def read_minimap_modes(p, driver, guard, screen, layout, run, modes):
    """Minimap dots in every mode, without panning. Returns {mode: [[x, y], ...]}."""
    before = screen.grab()
    ring0 = R.ring_center(before, layout)
    dots, info = {}, {}
    for m in modes:
        sw = MO.switch_mode(m, driver, guard, screen, layout, run.log, steady_timeout=3.0)
        f = screen.grab()
        run.image(f'minimap-{m}.png', f)
        feat = v.minimap_features(f, layout.minimap)
        dots[m] = np.round(feat.dots, 2).tolist()
        info[m] = dict(switch=sw, dots=len(feat.dots),
                       ring=None if feat.ring is None else np.round(feat.ring, 2).tolist())
        run.log(f'  minimap in {m}: {len(feat.dots)} dots')
    if MO.read_buttons(screen.grab(), layout)['active'] != 'survey':
        info['return'] = MO.switch_mode('survey', driver, guard, screen, layout, run.log, steady_timeout=3.0)
    after = p.observe()
    # The view must be where it was: same ring, and the image registers at ~0 shift.
    dx, dy, resp = v.phase_shift(before, after, layout)
    moved = float(np.hypot(dx, dy))
    ring_moved = None if (ring0 is None or p.ring is None) else float(np.linalg.norm(p.ring - ring0))
    info['view_after_return'] = dict(shift_px=round(moved, 2), response=round(resp, 3),
                                     ring_moved_px=None if ring_moved is None else round(ring_moved, 2))
    run.log(f'  back in Survey: view moved {moved:.1f}px (image), ring moved {ring_moved} minimap px')
    if resp > 0.2 and moved > 8:
        raise Stop(f'Switching modes moved the Survey view by {moved:.0f}px; not mapping on a moved view.')
    return dots, info


def load_calibration():
    if not CALIBRATION.exists():
        raise Stop('No pan_calibration.json - run pan-calibrate (M1) first.')
    model, cal = load_model(CALIBRATION)
    limit = (cal.get('bounds') or {}).get('V_limit_from_ring')
    if not limit:
        raise Stop('The calibration has no slide bounds; re-run pan-calibrate.')
    return model, limit


def ensure_buttons(driver, guard, screen, layout, run):
    """Mode buttons must be recognised before anything is clicked; end in Survey."""
    state = MO.read_buttons(screen.grab(), layout)
    run.log(f'Mode buttons: present={state["present"]}, active={state["active"]}')
    if sum(state['present']) < 3 or state['active'] is None:
        raise Stop('Mode buttons not recognised; not clicking anything.')
    if state['active'] != 'survey':
        MO.switch_mode('survey', driver, guard, screen, layout, run.log, steady_timeout=3.0)


def sweep_core(p, driver, guard, screen, layout, run, modes, limit, spacing, model):
    """Minimap per mode, then the Survey sweep. Writes sweep.json (even on Stop,
    which is re-raised). The Panner must already be started, in Survey."""
    targets = MP.plan_grid(limit, spacing=spacing)
    clayout, (x0, y0) = MP.crop_layout(layout)
    x1, y1 = x0 + clayout.size[0], y0 + clayout.size[1]
    meta = dict(version=2, modes=modes, limit_xy=limit, spacing=spacing, ring_scale=model.ring_scale,
                minimap=list(layout.minimap) if layout.minimap else None,
                crop_origin=[x0, y0], crop_layout=clayout.as_dict(), stops=[], minimap_dots={},
                calibration=str(CALIBRATION))
    t0 = time.perf_counter()
    try:
        run.log(f'Mapping plan: minimap in {", ".join(modes)}, then {len(targets)} stops in Survey.')
        run.log('Map 1/2 Reading the minimap in each mode...')
        meta['minimap_dots'], meta['minimap_info'] = read_minimap_modes(p, driver, guard, screen, layout, run, modes)
        meta['minimap_V'] = np.round(p.V, 1).tolist()
        run.log('Map 2/2 Sweeping in Survey...')
        for i, T in enumerate(targets):
            if MO.read_buttons(p.frame, layout)['active'] != 'survey':
                raise Stop('Not in Survey mode any more (did something else change the mode?).')
            res = p.pan_to(T, tol=15, tag=f'stop{i}')
            name = f'frames/stop{i:02d}-survey.jpg'
            run.image(name, p.frame[y0:y1, x0:x1], quality=95)
            meta['stops'].append(dict(i=i, target=list(T), V=np.round(p.V, 1).tolist(), pan=res,
                                      frames={'survey': name},
                                      ring=None if p.ring is None else np.round(p.ring, 2).tolist()))
            run.log(f'  stop {i + 1}/{len(targets)} at V={np.round(p.V).tolist()} ({res["drags"]} drags)')
    except Stop as exc:
        meta['stopped'] = str(exc)
        raise
    finally:
        meta['seconds'] = round(time.perf_counter() - t0, 1)
        meta['pan_history'] = list(p.history)
        run.json('sweep.json', meta)
    return meta


def sweep(api, screen, run, args):
    model, limit = load_calibration()
    modes = list(args.modes)
    guard, layout = setup_session(api, screen, run, args, max_clicks=12)
    driver = MouseDriver(guard, api, run.log)
    p = Panner(guard, driver, api, screen, layout, model, run.log)
    meta = {}
    try:
        run.log('')
        run.log('Survey mode, any position. Tap F6, then take your hand OFF the mouse (~1 minute).')
        wait_key(guard, api, keys=(w.VK_F6,))
        if wait_still(guard, api) is None:
            raise Stop('Mouse never settled.')
        ensure_buttons(driver, guard, screen, layout, run)
        p.start()
        run.log(f'Start position from minimap: V={np.round(p.V).tolist()}')
        meta = sweep_core(p, driver, guard, screen, layout, run, modes, limit, args.spacing, model)
    except Stop as exc:
        run.log(f'Stopped: {exc}')
    finally:
        guard.close()
    if (run.folder / 'sweep.json').exists():
        import json as _json
        meta = _json.loads((run.folder / 'sweep.json').read_text(encoding='utf-8'))
    run.log(f'Sweep finished in {meta.get("seconds")}s ({len(meta.get("stops", []))} stops). Analysing...')
    if meta.get('stops'):
        MP.analyze(run.folder, run.log)
        run.log(f'Map: {run.folder / "map.png"}')
    return meta
