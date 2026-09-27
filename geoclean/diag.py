"""Per-run diagnostics folder: runs/<timestamp>-<kind>/ with log, images, JSON."""
import json
import time
from pathlib import Path

import cv2

from .vision import as_jsonable

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / 'runs'


class Run:
    def __init__(self, kind, root=RUNS):
        stamp = time.strftime('%Y%m%d-%H%M%S')
        self.folder = Path(root) / f'{stamp}-{kind}'
        self.folder.mkdir(parents=True, exist_ok=True)
        self._log = open(self.folder / 'log.txt', 'a', encoding='utf-8')
        (Path(root) / 'LATEST.txt').write_text(str(self.folder), encoding='utf-8')
        self.t0 = time.perf_counter()

    def log(self, msg='', console=True):
        line = f'[{time.perf_counter() - self.t0:8.3f}] {msg}'
        self._log.write(line + '\n')
        self._log.flush()
        if console:
            print(msg, flush=True)

    def path(self, name):
        p = self.folder / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def image(self, name, img, quality=None):
        params = [cv2.IMWRITE_JPEG_QUALITY, quality] if quality else []
        if not cv2.imwrite(str(self.path(name)), img, params):
            self.log(f'WARNING: could not save {name}')

    def json(self, name, obj):
        self.path(name).write_text(json.dumps(as_jsonable(obj), indent=2), encoding='utf-8')

    def text(self, name, s):
        self.path(name).write_text(s, encoding='utf-8')

    def close(self):
        self._log.close()
