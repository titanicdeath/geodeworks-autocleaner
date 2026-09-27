"""Command line + launcher menu. Run `python -m geoclean --help`."""
import argparse
import json
import sys
import traceback
import unittest
from pathlib import Path

from . import __version__
from .diag import ROOT, Run
from .safety import Stop

MENU = f"""
Geodeworks cleaner {__version__}
  1  Clean slide   (M3): hands-off; cleans Survey, Bright and Polarized (follows the minimap)
  7  Map slide     (M2): hands-off ~1 min; maps the slide and lists every imperfection by mode
  6  Pan calibrate (M1): hands-off ~2 min; measures panning, finds slide edges, saves calibration
  2  Click test    (M0): hover a block + F6, the tool re-clicks it with 6 movement strategies
  3  Drag test     (M0): holds right-click and pans in a fixed pattern, measuring the view
  4  Snapshot         : save a screenshot + detected layout (no input sent)
  5  Self-test        : offline tests on ./samples (no game needed)
  q  Quit
Stop at any time: hold F8 (Esc also works), or switch away from Roblox.
"""


def build_parser():
    p = argparse.ArgumentParser(prog='geoclean', description=__doc__)
    sub = p.add_subparsers(dest='command')
    c = sub.add_parser('click-test', help='M0 click strategy test (hover + F6)')
    c.add_argument('--strategies', nargs='*', help='e.g. S0 S4 S5 (default: all)')
    c.add_argument('--seconds', type=float, default=300)
    c.add_argument('--dry-run', action='store_true')
    d = sub.add_parser('drag-test', help='M0 right-drag measurement')
    d.add_argument('--methods', nargs='*', default=['rel', 'abs'], choices=['rel', 'abs', 'relopen'])
    d.add_argument('--amplitude', type=int, default=120, help='max cursor offset in px (default 120)')
    d.add_argument('--seconds', type=float, default=120)
    d.add_argument('--dry-run', action='store_true')
    m = sub.add_parser('pan-calibrate', help='M1 pan calibration (hands-off)')
    m.add_argument('--seconds', type=float, default=300)
    m.add_argument('--styles', nargs='*', default=['v1600', 'v800', 'v400', 'v800np'])
    m.set_defaults(dry_run=False)
    mp = sub.add_parser('map-slide', help='M2 map the whole slide (hands-off)')
    mp.add_argument('--modes', nargs='*', default=['survey', 'bright', 'polarized'],
                    choices=['survey', 'bright', 'polarized'])
    mp.add_argument('--spacing', type=float, default=540.0, help='grid spacing in slide px (default 540)')
    mp.add_argument('--seconds', type=float, default=300)
    mp.set_defaults(dry_run=False)
    cl = sub.add_parser('clean', help='M3 clean the slide, driven by the minimap (hands-off)')
    cl.add_argument('--modes', nargs='*', default=['survey', 'bright', 'polarized'],
                    choices=['survey', 'bright', 'polarized'])
    cl.add_argument('--max-clicks', type=int, default=200)
    cl.add_argument('--revisits', type=int, default=2, help='extra passes over modes not greyed out yet')
    cl.add_argument('--seconds', type=float, default=900)
    cl.set_defaults(dry_run=False)
    am = sub.add_parser('analyze-map', help='re-run M2 analysis on a saved map-slide run folder')
    am.add_argument('folder', type=Path)
    s = sub.add_parser('snapshot', help='save a screenshot and detected layout, no input')
    s.add_argument('--seconds', type=float, default=60)
    s.add_argument('--dry-run', action='store_true')
    sub.add_parser('selftest', help='run offline tests')
    a = sub.add_parser('analyze-drag', help='re-run drag analysis on a saved run folder')
    a.add_argument('folder', type=Path)
    return p


def selftest():
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), top_level_dir=str(ROOT))
    ok = unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()
    return 0 if ok else 1


def snapshot(api, screen, run, args):
    from .safety import Guard
    from . import vision as v
    from .m0 import wait_for_microscope
    guard = Guard(api, run.log, max_seconds=args.seconds, dry_run=True)
    guard.wait_for_roblox()
    frame, layout = wait_for_microscope(guard, screen, run, timeout=args.seconds)
    run.image('frame.png', frame)
    run.image('layout.png', v.draw_layout(frame, layout))
    info = v.minimap_features(frame, layout.minimap) if layout.minimap else None
    run.json('layout.json', dict(layout=layout.as_dict(), minimap=info,
                                 windows=api.environment(guard.hwnd)))
    run.log(f'Saved {run.folder / "frame.png"}  (dots on minimap: {len(info.dots) if info else "?"}, '
            f'view ring: {info.ring if info else None})')


def analyze_saved_drag(folder):
    from . import vision as v
    from .m0 import analyze_drag
    env = json.loads((folder / 'env.json').read_text(encoding='utf-8'))
    L = env['layout']
    layout = v.Layout(scope=tuple(L['scope']), minimap=tuple(L['minimap']) if L['minimap'] else None,
                      size=tuple(L['size']))
    for rec in sorted(folder.glob('drag-*/records.json')):
        method = rec.parent.name.split('-', 1)[1]
        a = analyze_drag(folder, method, json.loads(rec.read_text(encoding='utf-8')), layout)
        print(a['text'])
        print()


def main(argv=None):
    parser = build_parser()
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(MENU)
        choice = input('Select: ').strip().lower()
        argv = {'1': ['clean'], '7': ['map-slide'], '6': ['pan-calibrate'], '2': ['click-test'], '3': ['drag-test'], '4': ['snapshot'],
                '5': ['selftest']}.get(choice)
        if not argv:
            return 0
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == 'selftest':
        return selftest()
    if args.command == 'analyze-drag':
        analyze_saved_drag(args.folder)
        return 0
    if args.command == 'analyze-map':
        from .mapping import analyze
        analyze(args.folder)
        return 0

    from .winapi import Win32
    api = Win32()            # sets DPI awareness before anything captures the screen
    from .capture import Screen
    screen = Screen()
    run = Run(args.command)
    run.log(f'geoclean {__version__} {args.command}  ->  results in {run.folder}')
    try:
        if args.command == 'click-test':
            from .m0 import click_test
            click_test(api, screen, run, args)
        elif args.command == 'drag-test':
            from .m0 import drag_test
            drag_test(api, screen, run, args)
        elif args.command == 'clean':
            from .m3 import clean
            clean(api, screen, run, args)
        elif args.command == 'map-slide':
            from .m2 import sweep
            sweep(api, screen, run, args)
        elif args.command == 'pan-calibrate':
            from .m1 import pan_calibrate
            pan_calibrate(api, screen, run, args, styles=tuple(args.styles))
        elif args.command == 'snapshot':
            snapshot(api, screen, run, args)
    except Stop as exc:
        run.log(f'Stopped: {exc}')
    except Exception:
        run.log(traceback.format_exc())
        run.log('ERROR - the traceback above is also in log.txt.')
        return 1
    finally:
        screen.close()
        run.log(f'Results folder: {run.folder}')
        run.close()
    return 0
