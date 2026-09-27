"""M0: input diagnostics.

click-test: you hover a known cleanable block and tap F6; you move the mouse away
and let go; the tool moves back with strategy N and clicks. Repeated for each
strategy. Before/after frames judge whether the block was removed.

drag-test: the tool holds the right button at the microscope centre and moves the
cursor through a fixed pattern of offsets, capturing frames at each stop, to
measure how cursor offset maps to view movement (displacement vs velocity,
gain, sign, return-to-start).
"""
import time

import cv2
import numpy as np

from . import winapi as w
from .inputs import MouseDriver, Strategy, strategy_list
from .safety import Guard, Stop
from . import vision as v


# --------------------------------------------------------------------------
# Small interaction helpers
# --------------------------------------------------------------------------

def wait_key(guard, api, keys=(w.VK_F6, w.VK_F7), timeout=None):
    """Wait for a fresh press of one of `keys`. Returns the key or None on timeout."""
    prev = {k: api.key_down(k) for k in keys}
    end = time.monotonic() + timeout if timeout else None
    while True:
        guard.check()
        for k in keys:
            down = api.key_down(k)
            if down and not prev[k]:
                return k
            prev[k] = down
        if end and time.monotonic() > end:
            return None
        time.sleep(0.01)


def wait_away_and_still(guard, api, point, min_dist=70, still=0.5, timeout=25):
    """Wait until the cursor is >= min_dist from point and hasn't moved for `still` s."""
    end = time.monotonic() + timeout
    last, since = None, time.monotonic()
    while time.monotonic() < end:
        guard.check()
        c = api.get_cursor()
        if c != last:
            last, since = c, time.monotonic()
        elif np.hypot(c[0] - point[0], c[1] - point[1]) >= min_dist and \
                time.monotonic() - since >= still:
            return c
        time.sleep(0.01)
    return None


def wait_still(guard, api, still=0.5, timeout=20):
    end = time.monotonic() + timeout
    last, since = None, time.monotonic()
    while time.monotonic() < end:
        guard.check()
        c = api.get_cursor()
        if c != last:
            last, since = c, time.monotonic()
        elif time.monotonic() - since >= still:
            return c
        time.sleep(0.01)
    return None


def wait_for_microscope(guard, screen, run, timeout=90):
    """Poll until the microscope circle is detectable. Returns (frame, layout)."""
    end = time.monotonic() + timeout
    told = False
    while True:
        frame = screen.grab()
        try:
            return frame, v.detect_layout(frame)
        except ValueError as exc:
            if time.monotonic() > end:
                run.image('no-microscope.png', frame)
                raise Stop(f'{exc} Gave up after {timeout}s; frame saved as no-microscope.png.')
            if not told:
                run.log(f'{exc} Waiting for the microscope view...')
                told = True
        guard.sleep(0.5)      # also stops on F8/Esc or focus loss


def setup_session(api, screen, run, args, max_clicks):
    guard = Guard(api, run.log, max_seconds=args.seconds, max_clicks=max_clicks,
                  dry_run=args.dry_run)
    if args.dry_run:
        run.log('DRY RUN: no mouse input will be sent.')
    guard.wait_for_roblox()
    frame, layout = wait_for_microscope(guard, screen, run)
    env = dict(args=vars(args), screen=screen.info(), windows=api.environment(guard.hwnd),
               layout=layout.as_dict())
    run.json('env.json', env)
    run.image('layout.png', v.draw_layout(frame, layout))
    run.log(f'Microscope circle {tuple(round(x) for x in layout.scope)}, '
            f'minimap {layout.minimap and tuple(round(x) for x in layout.minimap)}')
    run.log(f'Window: {env["windows"].get("window_exe")} rect={env["windows"].get("window_rect")} '
            f'dpi={env["windows"].get("window_dpi")} awareness={env["windows"]["dpi_awareness"]} '
            f'mouse speed={env["windows"].get("mouse_speed_1_20")} '
            f'enhance precision={env["windows"].get("enhance_pointer_precision")}')
    if env['windows']['swap_buttons']:
        run.log('WARNING: primary/secondary mouse buttons are swapped in Windows settings.')
    guard.start()
    return guard, layout


# --------------------------------------------------------------------------
# Click test
# --------------------------------------------------------------------------

def click_trial(idx, total, s: Strategy, api, screen, guard, driver, layout, run):
    tag = f'trial-{idx:02d}-{s.name}'
    control = s.move == 'none'
    run.log('')
    run.log(f'=== Trial {idx}/{total}: {s.name} - {s.description}')
    if control:
        run.log('    Hover the CENTRE of a full-contrast block, tap F6, and KEEP THE MOUSE STILL.')
    else:
        run.log('    Hover the CENTRE of a full-contrast block and tap F6.')
    run.log('    (F7 skips this strategy, F8 stops everything.)')
    while True:
        key = wait_key(guard, api)
        if key == w.VK_F7:
            run.log('    Skipped.')
            return dict(trial=idx, strategy=s.name, verdict='skipped', removed=None)
        p_screen = api.get_cursor()
        p = screen.to_image(p_screen)
        if layout.is_clickable_area(*p):
            break
        run.log(f'    {p} is too close to the rim/HUD/buttons. Pick a block nearer the centre.')
    run.log(f'    Target recorded at {p}.')
    if control:
        guard.sleep(0.3)
        away = p_screen
    else:
        run.log('    Now move the mouse ~3 cm away onto empty background and LET GO of it.')
        away = wait_away_and_still(guard, api, p_screen)
        if away is None:
            run.log('    Timed out waiting for the mouse to move away and settle. Skipped.')
            return dict(trial=idx, strategy=s.name, verdict='timeout', removed=None)
        run.log(f'    Mouse parked at {screen.to_image(away)}. Moving back with {s.name}...')
    before = screen.grab()
    trace = driver.move_to(p_screen, s)
    guard.sleep(s.dwell)
    at_press = api.get_cursor()
    hover = screen.grab()
    disturbed = np.hypot(at_press[0] - p_screen[0], at_press[1] - p_screen[1]) > 2
    if disturbed and not guard.dry_run:
        run.log(f'    Cursor is at {screen.to_image(at_press)}, not the target. Not clicking.')
        result = dict(verdict='not clicked (cursor off target)', removed=None)
    else:
        t_press = time.perf_counter()
        if guard.dry_run:
            run.log('    DRY RUN: would press now.')
        else:
            driver.click('left', s.hold)
        guard.sleep(0.9)
        after = screen.grab()
        press_to_capture = round((time.perf_counter() - t_press) * 1000)
        result = v.judge_click(before, after, p, layout)
        result['press_to_capture_ms'] = press_to_capture
        run.image(f'{tag}/after.png', after)
        strip = v.zoom_strip([before, hover, after], p,
                             ['before (mouse away)', 'hovered, pre-press', 'after click'])
        run.image(f'{tag}/zoom.png', strip)
    hover_local = v.local_change(before, hover, p, layout)
    run.image(f'{tag}/before.png', before)
    rec = dict(trial=idx, strategy=s.name, strategy_params=s.as_dict(), target=list(p),
               parked_at=list(screen.to_image(away)), cursor_at_press=list(screen.to_image(at_press)),
               move_trace=trace, hover_patch_changed_px=hover_local['patch_changed_px'], **result)
    run.json(f'{tag}/trial.json', rec)
    run.log(f'    RESULT: {rec["verdict"]}  (object gone {result.get("local", {}).get("gone_frac")}, '
            f'HUD px changed {result.get("hud_changed_px")}, minimap dots '
            f'{result.get("minimap_dots_before")}->{result.get("minimap_dots_after")}, '
            f'move {trace.get("move_ms")} ms, final error {trace.get("final_error_px")} px)')
    return rec


def click_test(api, screen, run, args):
    strategies = strategy_list(args.strategies)
    guard, layout = setup_session(api, screen, run, args, max_clicks=len(strategies))
    driver = MouseDriver(guard, api, run.log)
    results = []
    try:
        for i, s in enumerate(strategies, 1):
            results.append(click_trial(i, len(strategies), s, api, screen, guard, driver, layout, run))
            run.json('results.json', results)
    except Stop as exc:
        run.log(f'Stopped: {exc}')
    finally:
        guard.close()
        run.json('results.json', results)
        run.text('summary.txt', click_summary(results))
        run.log('')
        run.log(click_summary(results))
    return results


def click_summary(results):
    lines = ['M0 click test summary', '',
             f'{"#":>2}  {"strategy":22} {"verdict":34} {"gone":>5} {"HUDpx":>6} {"dots":>7} {"err_px":>6} {"move_ms":>7}']
    for r in results:
        loc = r.get('local') or {}
        tr = r.get('move_trace') or {}
        dots = (f'{r.get("minimap_dots_before")}->{r.get("minimap_dots_after")}'
                if r.get('minimap_dots_before') is not None else '')
        lines.append(f'{r["trial"]:>2}  {r["strategy"]:22} {r["verdict"]:34} '
                     f'{loc.get("gone_frac", ""):>5} {r.get("hud_changed_px", ""):>6} {dots:>7} '
                     f'{tr.get("final_error_px", ""):>6} {tr.get("move_ms", ""):>7}')
    worked = [r['strategy'] for r in results if r.get('removed')]
    control = [r for r in results if r['strategy'].startswith('S0')]
    lines.append('')
    if control and not control[0].get('removed'):
        lines.append('NOTE: the control (no movement) did not remove its block. That block was probably '
                     'not cleanable in the current mode; results for other strategies are unreliable.')
    lines.append('Removed by: ' + (', '.join(worked) if worked else 'none'))
    return '\n'.join(lines)


# --------------------------------------------------------------------------
# Drag test
# --------------------------------------------------------------------------

def drag_waypoints(amp):
    a, b = amp // 2, amp
    return [(a, 0), (b, 0), (0, 0), (0, a), (0, b), (0, 0), (-a, 0), (-b, 0), (0, 0),
            (0, -a), (0, -b), (0, 0)]


def move_during_drag(driver, method, target, run):
    """Move the (possibly held) cursor to `target` (screen px) with `method`."""
    cur = np.array(driver.cursor(), float)
    target = np.array(target, float)
    if method == 'abs':
        n = max(1, int(np.ceil(np.abs(target - cur).max() / 10)))
        for i in range(1, n + 1):
            px, py = np.rint(cur + (target - cur) * i / n).astype(int)
            driver._abs(px, py, True)
            driver.guard.sleep(0.008)
        return dict(events=n)
    if method == 'rel':
        events = driver.rel_closed_loop(target, step_px=10, step_delay=0.008, stall_limit=15)
        return dict(events=events)
    if method == 'relopen':
        # Pixels commanded as raw counts; the cursor may not follow if Roblox locks it.
        d = np.rint(target - driver._open_pos).astype(int)
        events = driver.rel_open_loop(int(d[0]), int(d[1]), step_px=8)
        driver._open_pos = target
        return dict(events=events, counts=d.tolist())
    raise ValueError(method)


def drag_run(method, api, screen, guard, driver, layout, run, amp):
    run.log('')
    run.log(f'=== Right-drag using {method} cursor moves')
    cx, cy, _ = layout.scope
    anchor = screen.to_screen((cx, cy))
    # Pre-position with a real-looking relative move so Roblox knows where the
    # cursor is at the moment the right button goes down.
    from .inputs import STRATEGIES
    driver.move_to(anchor, STRATEGIES['S5-rel-wiggle-slow'])
    guard.sleep(0.4)
    driver._open_pos = np.array(anchor, float)
    records = []

    def snap(label, offset_cmd, t_ref):
        f = screen.grab()
        c = api.get_cursor()
        name = f'drag-{method}/{len(records):02d}-{label}.jpg'
        run.image(name, f, quality=92)
        records.append(dict(label=label, offset_cmd=list(offset_cmd),
                            cursor=[c[0] - anchor[0], c[1] - anchor[1]],
                            t=round(time.perf_counter() - t_ref, 3), frame=name))
        return f

    t0 = time.perf_counter()
    snap('start', (0, 0), t0)
    guard.sleep(0.25)
    snap('start-again', (0, 0), t0)
    try:
        driver.press('right')
        guard.sleep(0.2)
        snap('pressed', (0, 0), t0)
        for ox, oy in drag_waypoints(amp):
            info = move_during_drag(driver, method, (anchor[0] + ox, anchor[1] + oy), run)
            t_arr = time.perf_counter()
            guard.sleep(0.25)
            snap(f'at{ox:+d}{oy:+d}-0.25s', (ox, oy), t0)
            guard.sleep(0.5)
            snap(f'at{ox:+d}{oy:+d}-0.75s', (ox, oy), t0)
            records[-1]['move_events'] = info.get('events')
            run.log(f'    offset {ox:+4d},{oy:+4d}: cursor now {records[-1]["cursor"]} '
                    f'({time.perf_counter() - t_arr:.2f}s after arrival)')
    finally:
        driver.release('right') if 'right' in guard.held else None
        guard.release_all()
    guard.sleep(0.4)
    snap('released', (0, 0), t0)
    run.json(f'drag-{method}/records.json', records)
    analysis = analyze_drag(run.folder, method, records, layout)
    run.json(f'drag-{method}/analysis.json', analysis)
    run.text(f'drag-{method}/analysis.txt', analysis['text'])
    run.log(analysis['text'])
    return analysis


def analyze_drag(folder, method, records, layout):
    frames = [cv2.imread(str(folder / r['frame'])) for r in records]
    base = frames[0]
    ring0 = v.minimap_features(base, layout.minimap).ring if layout.minimap else None
    directs, steps, rings = [], [], []
    for i, f in enumerate(frames):
        directs.append(v.phase_shift(base, f, layout))
        steps.append(v.phase_shift(frames[i - 1], f, layout) if i else (0.0, 0.0, 1.0))
        ring = v.minimap_features(f, layout.minimap).ring if layout.minimap else None
        rings.append(np.array([ring[0] - ring0[0], ring[1] - ring0[1]]) if (ring and ring0) else None)
    scale = ring_scale(directs, rings)
    rows = []
    cum = np.zeros(2)
    for r, d, s, ring in zip(records, directs, steps, rings):
        # Candidates: straight against the start frame, or the previous estimate plus
        # a stop-to-stop step. Accept only those the minimap ring agrees with
        # (ring * scale is coarse, ~±15 px, but independent); prefer higher response.
        cands = [(np.array(d[:2]), d[2], 'direct'), (cum + np.array(s[:2]), s[2], 'chained')]
        pred = -scale * ring if (ring is not None and scale) else None
        if pred is not None:
            ok = [c for c in cands if np.linalg.norm(c[0] - pred) < 25 and c[1] > 0.1]
            best = max(ok, key=lambda c: c[1]) if ok else (pred, 0.0, 'minimap')
        else:
            best = max(cands, key=lambda c: c[1])
        cum = best[0]
        rows.append(dict(label=r['label'], offset_cmd=r['offset_cmd'], cursor=r['cursor'],
                         shift=[round(float(cum[0]), 1), round(float(cum[1]), 1)], source=best[2],
                         direct=[round(d[0], 1), round(d[1], 1)], response=round(d[2], 3),
                         step=[round(s[0], 1), round(s[1], 1)], step_response=round(s[2], 3),
                         ring_shift=None if ring is None else [round(float(x), 2) for x in ring]))
    # Interpretation
    lines = [f'Right-drag analysis ({method} moves). shift = best estimate of slide content movement since '
             f'start (px; src says which registration was used). direct = straight against the start '
             f'frame; step = against the previous capture; ring = minimap view-circle movement (px). '
             f'Content px per minimap px: {scale and round(scale, 2)}.', '',
             f'{"label":22} {"cmd":>10} {"cursor":>11} {"shift":>14} {"src":>7} {"direct":>14} {"resp":>5} '
             f'{"step":>14} {"sresp":>5} {"ring":>15}']
    for r in rows:
        lines.append(f'{r["label"]:22} {str(tuple(r["offset_cmd"])):>10} {str(tuple(r["cursor"])):>11} '
                     f'{str(tuple(r["shift"])):>14} {r["source"]:>7} {str(tuple(r["direct"])):>14} '
                     f'{r["response"]:>5} {str(tuple(r["step"])):>14} {r["step_response"]:>5} '
                     f'{str(r["ring_shift"]):>15}')
    moving = [r for r in rows if r['offset_cmd'] != [0, 0] and r['label'].endswith('0.75s')]
    responded = any(max(abs(r['shift'][0]), abs(r['shift'][1])) > 3 for r in rows)
    cursor_followed = any(max(abs(r['cursor'][0]), abs(r['cursor'][1])) > 3 for r in moving)
    lines.append('')
    lines.append(f'Cursor followed the commanded offsets: {cursor_followed}')
    lines.append(f'View moved at all: {responded}')
    gains, drift = [], []
    by_label = {r['label']: r for r in rows}
    for r in moving:
        if by_label.get(r['label'].replace('0.75s', '0.25s')) and r['step_response'] > 0.1:
            drift.append(float(np.hypot(*r['step'])))   # 0.25s -> 0.75s at the same stop
        c = np.array(r['cursor'], float)
        if np.linalg.norm(c) < 10:          # cursor locked/not followed: use the command
            c = np.array(r['offset_cmd'], float)
        if np.linalg.norm(c) > 10:
            gains.append(float(np.dot(r['shift'], c) / np.dot(c, c)))
    if drift:
        mode = 'DISPLACEMENT-like (view holds still while cursor holds still)' if max(drift) < 4 else \
               'VELOCITY-like (view keeps moving while cursor holds still)'
        lines.append(f'Drift between 0.25s and 0.75s at a held offset: max {max(drift):.1f}px -> {mode}')
    if gains:
        lines.append(f'Gain (content shift / cursor offset, along the offset): median {np.median(gains):+.3f} '
                     f'(range {min(gains):+.3f}..{max(gains):+.3f}). Positive = content moves with the cursor.')
    returns = [r for r in rows if r['offset_cmd'] == [0, 0] and r['label'].startswith('at')]
    if returns:
        worst = max(np.hypot(*r['shift']) for r in returns)
        lines.append(f'Back at the anchor (0,0) while held: worst residual shift {worst:.1f}px')
    return dict(method=method, rows=rows, cursor_followed=cursor_followed, view_moved=responded,
                gains=gains, drift=drift, ring_scale=scale, text='\n'.join(lines))


def ring_scale(directs, rings):
    """Content px moved per minimap-ring px (ring moves opposite to the content),
    from confident direct registrations with a clearly moved ring."""
    ks = []
    for d, ring in zip(directs, rings):
        if ring is not None and d[2] > 0.5 and np.linalg.norm(ring) > 1.0:
            ks.append(-float(np.dot(d[:2], ring) / np.dot(ring, ring)))
    return float(np.median(ks)) if len(ks) >= 2 else None


def drag_test(api, screen, run, args):
    guard, layout = setup_session(api, screen, run, args, max_clicks=0)
    driver = MouseDriver(guard, api, run.log)
    out = {}
    try:
        run.log('')
        run.log('Pan (by hand) so the view is near the MIDDLE of the slide with a few blocks visible.')
        run.log('Then tap F6 and LET GO of the mouse. The tool will hold the right button and move the')
        run.log(f'cursor up to {args.amplitude}px from the microscope centre and back, several times.')
        wait_key(guard, api, keys=(w.VK_F6,))
        run.log('Waiting for the mouse to be still...')
        if wait_still(guard, api) is None:
            raise Stop('Mouse never settled.')
        for method in args.methods:
            out[method] = drag_run(method, api, screen, guard, driver, layout, run, args.amplitude)
            guard.sleep(0.6)
    except Stop as exc:
        run.log(f'Stopped: {exc}')
    finally:
        guard.close()
    run.json('drag-summary.json', {k: {kk: vv for kk, vv in a.items() if kk != 'text'} for k, a in out.items()})
    return out
