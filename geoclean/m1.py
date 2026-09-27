"""M1: pan calibration (hands-off, about a minute).

1. latency  - how long after a held cursor move the view settles
2. styles   - drags of 60/200 px in 4 directions, moving the held cursor as one
              jump, 10 px steps or 3 px steps; fit content-shift = G @ cursor per
              style and keep the most consistent one
3. edges    - from the slide centre, walk in 8 directions until the view stops
              (slide boundary), returning to the centre each time
4. accuracy - closed-loop pan_to() a set of targets
5. closure  - return to the centre and compare with the first centre frame
Writes pan_calibration.json (project root) plus a full report in the run folder.
"""
import json
import time

import cv2
import numpy as np

from . import register as R
from . import vision as v
from .diag import ROOT
from .inputs import MouseDriver
from .m0 import setup_session, wait_key, wait_still
from .pan import ANCHOR_MOVE, Panner, PanModel, fit_gain, frame_gaps, save_model
from .safety import Stop
from . import winapi as w

CALIBRATION = ROOT / 'pan_calibration.json'
# Held-cursor speed in px/s, fastest first. 'np' = no primer move after the press
# (diagnostic: shows whether the primer matters on this machine).
STYLES = ('v1600', 'v800', 'v400', 'v800np')
DIRS8 = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]


def latency_probe(p, dx, burst=0.5):
    """Hold right, move the cursor dx at a slow, safe speed, then grab frames as
    fast as possible. Latency = time from the end of the motion until the view
    stops changing."""
    T = p.model.timing
    before, ring_b, V_b = p.frame, p.ring, p.V.copy()
    a, b = p._endpoints((dx, 0))
    p.driver.move_to(p.screen.to_screen(a), ANCHOR_MOVE)
    p.guard.sleep(T['anchor'])
    p.driver.press('right')
    samples = []
    try:
        p.guard.sleep(T['press'])
        p._move_held(a, b, 'v400')
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < burst:
            f = p.screen.grab()
            samples.append((time.perf_counter() - t0, f))
    finally:
        if 'right' in p.guard.held:
            p.driver.release('right')
    p.guard.sleep(T['release'])
    after = p.observe()
    shift, score, source = R.measure_shift(before, after, p.layout, ring_b, p.ring,
                                           p.model.ring_scale, p.model.predict((dx, 0)))
    p.V = V_b - shift
    series = []
    for t, f in samples:
        s = v.phase_shift(before, f, p.layout)
        series.append((round(t * 1000, 1), round(s[0], 1), round(s[2], 2)))
    final = shift[0]
    settled = next((t for t, sx, _ in series if abs(sx - final) <= 1.5 and
                    all(abs(x - final) <= 1.5 for tt, x, _ in series if tt >= t)), None)
    p.history.append(dict(n=len(p.history) + 1, tag='latency', cursor=[dx, 0],
                          shift=np.round(shift, 1).tolist(), score=round(score, 3), source=source,
                          V=np.round(p.V, 1).tolist()))
    return dict(cursor=dx, final_shift=round(float(final), 1), settled_ms=settled,
                frames=len(series), series=series)


def style_test(p, styles, run):
    vecs = [(m * sx, m * sy) for m in (60, 200) for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
    fits = {}
    for style in styles:
        recs = [p.pan_cursor(d, style=style, tag=f'style-{style}') for d in vecs]
        good = [r for r in recs if r['source'] == 'image']
        if len(good) < 4:
            run.log(f'  {style}: too few reliable measurements ({len(good)}/8)')
            continue
        G, rms, rel = fit_gain([r['cursor'] for r in good], [r['shift'] for r in good])
        if abs(np.linalg.det(G)) < 0.2:
            run.log(f'  {style}: the view barely moved (G={G.round(3).tolist()}); not usable.')
            fits[style] = dict(G=G.round(4).tolist(), usable=False)
            continue
        along = {f'{d}': round(float(np.dot(r['shift'], d) / np.dot(d, d)), 3)
                 for r, d in zip(recs, vecs) if r['source'] == 'image'}
        coupling = float(max(abs(G[0, 1]), abs(G[1, 0])) / max(1e-6, min(abs(G[0, 0]), abs(G[1, 1]))))
        # Live good runs: 1.4-5% scatter, coupling <= 0.10. Broken-timing runs: 35-50%
        # scatter, coupling 0.2-1.1. The limits sit between the two with margin.
        fits[style] = dict(usable=True, G=G.round(4).tolist(), rms_px=round(rms, 2), rel_rms=round(rel, 4),
                           coupling=round(coupling, 3), consistent=bool(rel <= 0.12 and coupling <= 0.15),
                           gain_along=along, held_ms=int(np.median([r['held_ms'] for r in recs])),
                           n=len(good))
        run.log(f'  {style:7}: G=[[{G[0,0]:+.3f} {G[0,1]:+.3f}] [{G[1,0]:+.3f} {G[1,1]:+.3f}]]  '
                f'residual {rms:.1f}px ({rel*100:.1f}%)  hold {fits[style]["held_ms"]}ms')
    usable = {s: f for s, f in fits.items() if f['usable']}
    if not usable:
        raise Stop('No drag style produced reliable measurements. Is the view featureless? '
                   'Pan to an area with several blocks and retry.')
    good = [s for s in styles if s in usable and usable[s]['consistent']]
    if not good:
        f = min(usable.values(), key=lambda f: f['rel_rms'])
        raise Stop(f'Drags are inconsistent in every style (best: scatter {f["rel_rms"]*100:.0f}%, '
                   f'cross-axis {f["coupling"]:.2f}; limits 12% and 0.15). The game is probably not '
                   f'seeing the cursor where the tool thinks it is - an input timing problem. '
                   f'Calibration NOT saved.')
    # Styles are listed fastest first. Among consistent ones, take the fastest whose
    # scatter is within 3 points of the best (every pan is measured and corrected
    # anyway) and that still pans ~1:1 (lossy speeds depend on the game's frame rate).
    best = min(usable[s]['rel_rms'] for s in good)
    ok = [s for s in good if usable[s]['rel_rms'] <= best + 0.03]
    gain = lambda s: min(abs(usable[s]['G'][0][0]), abs(usable[s]['G'][1][1]))
    full = [s for s in ok if gain(s) >= 0.85]
    chosen = full[0] if full else max(ok, key=gain)
    return fits, chosen


def fit_ring_scale(history, min_ring_move=3.0):
    """Slide px per ring px, separately for x and y, from per-drag pairs
    (image-measured shift vs ring delta). Per-drag pairs don't accumulate drift.
    Live: x 26.2 in two runs, y 24.9-25.2 (the view is slightly foreshortened in y).
    Returns ([sx, sy] or None, n, residual rms px per axis)."""
    pairs = [(np.array(r['shift'], float), np.array(r['ring'], float) - np.array(r['ring_before'], float))
             for r in history
             if r.get('source') == 'image' and r.get('score', 0) >= 0.6 and r.get('ring') and r.get('ring_before')]
    k, rms, n = [], [], []
    for ax in (0, 1):
        use = [(s[ax], d[ax]) for s, d in pairs if abs(d[ax]) >= min_ring_move]
        if len(use) < 3:
            return None, len(use), None
        s = np.array([u[0] for u in use])
        d = np.array([u[1] for u in use])
        # Robust LS: fit, drop gross outliers (wrong matches), refit.
        kk = -float(np.dot(s, d) / np.dot(d, d))
        keep = np.abs(s + kk * d) < 25
        if keep.sum() >= 3:
            kk = -float(np.dot(s[keep], d[keep]) / np.dot(d[keep], d[keep]))
        k.append(kk)
        rms.append(round(float(np.sqrt(np.mean((s[keep] + kk * d[keep]) ** 2))), 1))
        n.append(int(keep.sum()))
    return [round(k[0], 3), round(k[1], 3)], min(n), rms


def edge_probe(p, u, run, amount=150.0):
    """At an edge: push further, then pull back the same amount. Compare what the
    image and the minimap ring say each time (live, the ring kept moving at a
    corner while the view had stopped)."""
    out = {}
    for tag, sign in (('push', 1), ('pull', -1)):
        ring_b = p.ring
        rec = p.pan_content(-u * amount * sign, tag=f'probe-{tag}')
        # Both numbers: how far the VIEW moved in the requested direction (u*sign).
        ring_move = None if (ring_b is None or p.ring is None) else round(float(np.dot(
            (np.array(p.ring) - ring_b) * R.scale_vec(p.model.ring_scale), u * sign)), 1)
        out[tag] = dict(requested=amount, image_shift=rec['shift'], source=rec['source'],
                        score=rec['score'], view_moved_by_image=round(float(np.dot(rec['shift'], -u * sign)), 1)
                        if rec['source'] == 'image' else None,
                        ring_moved=ring_move)
    run.log(f'  edge probe: push {amount:.0f} -> image {out["push"]["view_moved_by_image"]}, '
            f'minimap {out["push"]["ring_moved"]}; pull back -> image {out["pull"]["view_moved_by_image"]}, '
            f'minimap {out["pull"]["ring_moved"]}  (px of slide; None = image had no features)')
    return out


def explore_edges(p, run, step=350.0, max_drags=8):
    edges, probe = [], None
    for d in DIRS8:
        u = np.array(d, float) / np.linalg.norm(d)
        found, sources = None, []
        for _ in range(max_drags):
            rec = p.pan_content(-u * step, tag=f'edge{d}')
            sources.append(rec['source'])
            if rec['source'] == 'model':
                continue            # unmeasured: can't tell whether we hit the edge
            progress = -float(np.dot(rec['shift'], u))
            requested = -float(np.dot(rec['predicted'], u))
            if progress < 0.3 * requested:
                found = p.V.copy()
                break
        extent = None if found is None else float(np.dot(found, u))
        # 'image' = measured from slide features; 'minimap' = from the ring only,
        # which may run past the real edge.
        mm = p.layout.minimap
        edges.append(dict(direction=list(d), measured_by=sorted(set(sources[-2:])),
                          edge_V=None if found is None else np.round(found, 1).tolist(),
                          extent=None if extent is None else round(extent, 1),
                          edge_V_ring=None if p.v_from_ring() is None else np.round(p.v_from_ring(), 1).tolist(),
                          edge_ring_px=None if (p.ring is None or found is None) else
                          np.round(np.array(p.ring) - mm[:2], 2).tolist()))
        run.log(f'  direction {str(d):8}: ' + (f'edge at V={np.round(found).tolist()} (extent {extent:.0f}px)'
                                              if found is not None else f'no edge within {max_drags} drags'))
        if d == (1, 0) and found is not None:
            probe = edge_probe(p, u, run)
        for _ in range(2):          # every direction starts from the centre
            back = p.pan_to((0, 0), tol=15)
            if not back['stuck'] and back['error'] <= 15:
                break
        else:
            run.log(f'  WARNING: could not get back to the centre: {back}')
            p.return_failures = getattr(p, 'return_failures', 0) + 1
    return edges, probe


def bounds_from_edges(edges):
    """Axis limits of the view centre, and whether the boundary is square or round."""
    by = {tuple(e['direction']): e for e in edges if e['extent']}
    b = {}
    for key, d, axis, sign in (('x_max', (1, 0), 0, 1), ('x_min', (-1, 0), 0, -1),
                               ('y_max', (0, 1), 1, 1), ('y_min', (0, -1), 1, -1)):
        if d in by:
            b[key] = by[d]['edge_V'][axis]
    axis_ext = [by[d]['extent'] for d in ((1, 0), (-1, 0), (0, 1), (0, -1)) if d in by]
    diag_ext = [by[d]['extent'] for d in ((1, 1), (-1, 1), (-1, -1), (1, -1)) if d in by]
    if len(axis_ext) >= 3 and len(diag_ext) >= 3:
        ratio = np.mean(diag_ext) / np.mean(axis_ext)
        b['diag_to_axis'] = round(float(ratio), 3)
        b['shape'] = ('square' if abs(ratio - np.sqrt(2)) < 0.15 else
                      'round' if abs(ratio - 1) < 0.1 else 'unclear')
    # The minimap ring is the absolute reference: at every edge it stops the same
    # number of minimap px from the centre (live: exactly 30.0 in both good runs).
    ring_lim = [abs(e['edge_ring_px'][i]) for e in edges if e.get('edge_ring_px')
                for i, c in enumerate(e['direction']) if c != 0]
    if len(ring_lim) >= 4:
        b['ring_limit_px'] = round(float(np.median(ring_lim)), 2)
        b['ring_limit_spread'] = round(float(np.max(ring_lim) - np.min(ring_lim)), 2)
    return b


def accuracy_test(p, run, radius):
    r = radius
    targets = [(r, 0), (-r, 0.6 * r), (0, -r), (0.7 * r, 0.7 * r), (-0.5 * r, -0.5 * r), (0.3 * r, -0.2 * r)]
    results = []
    for t in targets:
        res = p.pan_to(t, tol=5)
        vr = p.v_from_ring()
        res['ring_disagreement'] = None if vr is None else round(float(np.linalg.norm(p.V - vr)), 1)
        results.append(res)
        run.log(f'  target {np.round(t).tolist()}: {res["drags"]} drags, tracked error {res["error"]:.1f}px, '
                f'minimap says {res["ring_disagreement"]}px off')
    return results


def draw_map(p, edges, model, path_img):
    """Slide-coordinate overview: 1 px = 4 slide px."""
    k = 4.0
    size = 700
    img = np.full((size, size, 3), 30, np.uint8)
    c = np.array([size / 2, size / 2])
    to = lambda V: tuple(np.round(c + np.asarray(V) / k).astype(int))
    if p.layout.minimap:
        s = R.scale_vec(model.ring_scale)
        cv2.ellipse(img, to((0, 0)), (int(p.layout.minimap[2] * s[0] / k), int(p.layout.minimap[2] * s[1] / k)),
                    0, 0, 360, (90, 90, 90), 1)
    pts = [h['V'] for h in p.history if h.get('V')]
    for a, b in zip(pts, pts[1:]):
        cv2.line(img, to(a), to(b), (200, 160, 60), 1)
    for h in p.history:
        if h.get('V_ring'):
            cv2.circle(img, to(h['V_ring']), 2, (120, 120, 255), -1)
    for e in edges:
        if e['edge_V']:
            cv2.drawMarker(img, to(e['edge_V']), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 12, 2)
    view_r = int(p.layout.scope[2] / k)
    cv2.circle(img, to((0, 0)), view_r, (80, 200, 80), 1)
    cv2.putText(img, 'blue: tracked path  red dots: minimap estimate  X: edges  green: one view at centre',
                (8, size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
    cv2.imwrite(str(path_img), img)


def pan_calibrate(api, screen, run, args, timing=None, styles=STYLES, fixed_timing=None):
    """timing: initial waits; fixed_timing: waits forced after the latency step
    (tests use it to reproduce timing bugs)."""
    guard, layout = setup_session(api, screen, run, args, max_clicks=0)
    driver = MouseDriver(guard, api, run.log)
    model = PanModel()
    if timing:
        model.timing.update(timing)
    drag_log = run.path('drags.jsonl').open('a', encoding='utf-8')

    def record(rec, frame):
        drag_log.write(json.dumps(rec) + '\n')
        drag_log.flush()
        run.image(f'drags/{rec["n"]:03d}.jpg', frame, quality=85)
        run.log(f'    #{rec["n"]:<3} {rec["tag"]:14} cursor {str(rec["cursor"]):16} shift {str(rec["shift"]):16} '
                f'({rec["source"]} {rec["score"]:.2f})  V {rec["V"]}', console=False)

    p = Panner(guard, driver, api, screen, layout, model, run.log, record)
    out = dict(layout=layout.as_dict())
    try:
        run.log('')
        run.log('Survey mode, any position. Tap F6, then take your hand OFF the mouse for about a minute.')
        wait_key(guard, api, keys=(w.VK_F6,))
        if wait_still(guard, api) is None:
            raise Stop('Mouse never settled.')
        p.start()
        run.log(f'Start position from minimap: V={np.round(p.V).tolist()}')
        run.image('start.png', p.frame)

        run.log('1/5 Latency...')
        lat = [latency_probe(p, dx) for dx in (60, -60)]
        settled = [l['settled_ms'] for l in lat if l['settled_ms'] is not None]
        latency_ms = max(settled) if settled else None
        if latency_ms is not None:
            model.timing.update(frame_gaps(latency_ms / 1000))
        if fixed_timing:
            model.timing.update(fixed_timing)
        out['latency'] = dict(probes=lat, latency_ms=latency_ms, settle_s=model.timing['settle'],
                              gap_s=model.timing['press'])
        run.log(f'  view settles {latency_ms} ms after a held move -> settle {model.timing["settle"]:.2f}s, '
                f'gap between move/press/release {model.timing["press"]:.2f}s')

        run.log('2/5 Drag styles...')
        fits, chosen = style_test(p, styles, run)
        model.G, model.style = fits[chosen]['G'], chosen
        scale, n, _ = fit_ring_scale(p.history)
        if scale:
            model.ring_scale = scale
        out['styles'] = dict(fits=fits, chosen=chosen, ring_scale=model.ring_scale, ring_scale_n=n)
        run.log(f'  chosen style: {chosen}; minimap scale x {model.ring_scale[0]:.2f}, '
                f'y {model.ring_scale[1]:.2f} slide px per minimap px (n={n})')
        if p.v_from_ring() is not None:
            p.V = p.v_from_ring()      # re-anchor with the fitted scale

        run.log('3/5 Slide edges (8 directions)...')
        centre = p.pan_to((0, 0), tol=5)
        centre_frame = p.frame.copy()
        run.image('centre-first.png', centre_frame)
        out['centre'] = centre
        edges, probe = explore_edges(p, run)
        out['edges'] = edges
        out['edge_probe'] = probe
        out['bounds'] = bounds_from_edges(edges)
        run.log(f'  view-centre bounds: {out["bounds"]}')
        extents = [e['extent'] for e in edges if e['extent']]
        out['extent_min'] = min(extents) if extents else None
        out['extent_max'] = max(extents) if extents else None
        # Refit the scale with the long edge-walk drags (big ring moves -> the 0.5 px
        # ring grid matters less).
        k, n, rms = fit_ring_scale(p.history)
        out['ring_scale_global'] = dict(scale=k, rms_px=rms, n=n)
        if k and n >= 10:
            run.log(f'  minimap scale refit over the whole run: x {k[0]:.2f}, y {k[1]:.2f} '
                    f'(residual rms {rms} px, n={n})')
            model.ring_scale = k
        if out['bounds'].get('ring_limit_px'):
            lim = out['bounds']['ring_limit_px'] * R.scale_vec(model.ring_scale)
            out['bounds']['V_limit_from_ring'] = np.round(lim, 1).tolist()
            run.log(f'  minimap: the view stops {out["bounds"]["ring_limit_px"]} minimap px from the centre '
                    f'(spread {out["bounds"]["ring_limit_spread"]}) = x +-{lim[0]:.0f}, y +-{lim[1]:.0f} slide px')

        run.log('4/5 Closed-loop accuracy...')
        radius = 0.6 * (min(extents) if extents else 400)
        out['accuracy'] = accuracy_test(p, run, radius)

        run.log('5/5 Closure...')
        back = p.pan_to((0, 0), tol=3)
        dx, dy, resp = v.phase_shift(centre_frame, p.frame, layout)
        out['closure'] = dict(pan_to=back, true_residual_px=round(float(np.hypot(dx, dy)), 2),
                              residual=[round(dx, 2), round(dy, 2)], response=round(resp, 3))
        run.log(f'  back at centre: tracker says {back["error"]:.1f}px, image vs first centre visit '
                f'says {np.hypot(dx, dy):.1f}px (response {resp:.2f})')
    except Stop as exc:
        run.log(f'Stopped: {exc}')
        out['stopped'] = str(exc)
    finally:
        guard.close()
        drag_log.close()
    out['model'] = model.as_dict()
    out['drags'] = len(p.history)
    run.json('m1-results.json', out)
    report = m1_report(out)
    run.text('report.txt', report)
    if p.history:
        draw_map(p, out.get('edges', []), model, run.path('pan-map.png'))
    run.log('')
    run.log(report)
    problems = []
    if 'stopped' in out or 'closure' not in out:
        problems.append('the run did not complete')
    else:
        if getattr(p, 'return_failures', 0):
            problems.append(f'{p.return_failures} failed return(s) to the centre')
        if out['closure']['true_residual_px'] > 30:
            problems.append(f'closure error {out["closure"]["true_residual_px"]}px (> 30)')
    out['quality_problems'] = problems
    if not problems:
        save_model(CALIBRATION, model, extra=dict(run=str(run.folder), extent_min=out['extent_min'],
                                                 extent_max=out['extent_max'], edges=out['edges'],
                                                 bounds=out['bounds'], edge_probe=out['edge_probe'],
                                                 latency_ms=out['latency']['latency_ms'],
                                                 closure_px=out['closure']['true_residual_px']))
        run.log(f'Saved calibration: {CALIBRATION}')
    else:
        run.log(f'Calibration NOT saved: {"; ".join(problems)}.')
    return out


def m1_report(out):
    L = ['M1 pan calibration report', '']
    if 'latency' in out:
        L.append(f'Latency: view settled {out["latency"]["latency_ms"]} ms after a held jump '
                 f'(probes: {[p["settled_ms"] for p in out["latency"]["probes"]]}), settle used '
                 f'{out["latency"]["settle_s"]:.2f}s')
    if 'styles' in out:
        L.append('Drag styles (content shift = G @ cursor):')
        for s, f in out['styles']['fits'].items():
            if not f['usable']:
                L.append(f'  {s:7} G={f["G"]} - view barely moved, not usable')
                continue
            L.append(f'  {s:7} G={f["G"]} residual {f["rms_px"]}px ({f["rel_rms"]*100:.1f}%), cross-axis '
                     f'{f["coupling"]:.2f}, hold {f["held_ms"]}ms{"" if f["consistent"] else "  <-- INCONSISTENT"}')
            L.append(f'          gain along each test vector: {f["gain_along"]}')
        L.append(f'  chosen: {out["styles"]["chosen"]}; minimap scale (x, y) {out["styles"]["ring_scale"]} '
                 f'slide px / ring px (from the style drags)')
    g = out.get('ring_scale_global')
    if g and g['scale']:
        L.append(f'Minimap scale over the whole run (x, y): {g["scale"]} slide px / ring px '
                 f'(residual rms {g["rms_px"]} px, n={g["n"]})')
    if 'edges' in out:
        L.append('Slide edges (view-centre position where panning stopped, slide px from minimap centre):')
        for e in out['edges']:
            L.append(f'  {str(e["direction"]):8} extent {e["extent"]}  at {e["edge_V"]}  minimap says {e["edge_V_ring"]}'
                     f'  (last drags measured by {"/".join(e.get("measured_by", []))})')
        b = out.get('bounds') or {}
        L.append(f'  view-centre limits: x {b.get("x_min")}..{b.get("x_max")}, y {b.get("y_min")}..{b.get("y_max")}; '
                 f'boundary shape {b.get("shape")} (diagonal/axis extent {b.get("diag_to_axis")}; '
                 f'square = 1.41, round = 1.0)')
        if b.get('ring_limit_px'):
            L.append(f'  minimap: view stops {b["ring_limit_px"]} minimap px from centre in every direction '
                     f'(spread {b["ring_limit_spread"]}) = {b.get("V_limit_from_ring")} slide px (x, y)')
    if out.get('edge_probe'):
        e = out['edge_probe']
        L.append(f'Edge probe at +x (slide px): push {e["push"]["requested"]:.0f} further -> image says the view moved '
                 f'{e["push"]["view_moved_by_image"]}, minimap says {e["push"]["ring_moved"]}; pull back -> image '
                 f'{e["pull"]["view_moved_by_image"]}, minimap {e["pull"]["ring_moved"]} (None = no features to match)')
    if 'accuracy' in out:
        errs = [a['error'] for a in out['accuracy']]
        drags = [a['drags'] for a in out['accuracy']]
        dis = [a['ring_disagreement'] for a in out['accuracy'] if a['ring_disagreement'] is not None]
        L.append(f'Closed-loop pan_to: tracked error max {max(errs):.1f}px, drags per target {drags}, '
                 f'minimap disagreement max {max(dis) if dis else None}px')
    if 'closure' in out:
        c = out['closure']
        L.append(f'Closure (true, image-based): {c["true_residual_px"]}px {c["residual"]} after '
                 f'{out["drags"]} drags (response {c["response"]})')
    if 'stopped' in out:
        L.append(f'STOPPED EARLY: {out["stopped"]}')
    return '\n'.join(L)
