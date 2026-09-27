# Geodeworks microscope cleaner (v0.9.2, milestone M3)

Screen capture + ordinary Windows mouse input only. No memory reading, no injection.

## Status

| Milestone | State |
|---|---|
| M0 input diagnostics | **passed** live 2026-09-25 (see findings below) |
| M1 pan calibration | **passed** live 2026-09-25 23:15 (closure 4.1 px after 100 drags; see below) |
| M2 slide mapping | **passed** (user OK 2026-09-26): 33 found vs 37 counted by the user |
| M3 cleaning | **Works live**: v0.9.1 fully cleaned 3/3 gems in 31-35 s (the user takes 14-20 s). v0.9.2: speed tweaks; awaiting live run |

### M0 live findings (2026-09-25, one run each)

- Clicking: every SendInput-based move worked (S1-S5, n=1 each) and so did the no-move control.
  v0.3's failures used SetCursorPos alone, which sends no input event. v0.4 never actually
  clicked: its target selection found no candidates.
- Right-drag works with both relative and absolute moves. The view holds still while the cursor
  holds still, so it is not velocity-based. Content moves the same way as the cursor, at about
  0.87x (60 px -> 38-61 px, 120 px -> 101-119 px). Returning the cursor to the anchor leaves a
  0-20 px residual in the direction of the last movement. So panning accumulates cursor deltas
  (with some loss) rather than mapping the cursor position absolutely, and M1 needs closed-loop
  registration.
- One minimap-ring pixel is about 25-27 px of slide content. The ring moves opposite to the content.
- Registering large jumps (>100 px) against a distant frame is unreliable. Stop-to-stop
  registration, checked against the minimap ring, is consistent.

### M1 first live run (2026-09-25, 98 drags)

- **Held-cursor speed matters.** A single 60/200 px jump barely pans (x at 0.5x, y not at all).
  10 px steps give about 0.88x, and 3 px steps give about 0.975x with the lowest scatter (2%).
  Panning is close to 1:1 when the cursor moves smoothly.
- **Latency:** the view settles about 30-35 ms after a held move.
- **The slide boundary is square, not round.** The view centre stops at about x ±790 and
  y ±775 slide px from the minimap centre. The minimap ring then sits partly outside the
  round minimap (it is still drawn over the black).
- **The view is almost empty near the edges,** so image registration there has little to
  match. At a corner the ring kept moving (about 50 px) while the view barely did.
- **Minimap scale:** 25.98 slide px per ring px, over the whole run.
- **Closed-loop pans:** 2-3 drags to within 5 px (tracked). The true closure error was 10 px
  after 98 drags.

**Bugs found and fixed:**
- The ring detector missed partial rings at the corners. It now fits the ring's known radius
  on a wider crop.
- An unmeasurable drag was recorded as "didn't move", which faked edge hits and blocked the
  returns to centre. It now assumes the model instead, and never counts it as an edge.
- Image-measured drags are trusted over the coarse ring before a resync (80 px limit).
- Faster styles (`step6`, `step10`) are preferred when within 3 points of the most
  consistent one.
- The anchor move no longer wiggles.
- An edge probe at +x measures how the ring and the view behave when pushing past the edge
  and pulling back.
- The report gives square/round bounds.

### M1 live runs 2-4 (2026-09-25): a timing regression, now fixed

After run 1, the move to each drag's start point had lost its wiggle and its waits had been
cut to 20-30 ms, while the game was also responding more slowly (view latency 50-65 ms). The
game then counted the jump from the previous drag's end as part of the new drag. Every drag
that started somewhere new came out as (requested drag + that jump), and fitting that gave big
cross-axis terms and 35-50% scatter. Those runs still saved a bad calibration; it is kept as
`pan_calibration.BAD-run-20260925-222555.json`.

**Fixes:**
- The move to the drag start wiggles again.
- The waits between that move, the press, the motion and the release are frame gaps: at least
  60 ms, rescaled to 1.5x the measured latency.
- Drags are defined by speed (`v1600`/`v800`/`v400` px/s, one event every 5 ms), not step size.
- After the press, a 1 px primer move is sent, because run 1 showed single jumps either counting
  fully or not at all, as if the first held move can be swallowed. `v800np` omits the primer,
  for comparison.
- **Quality gate:** the calibration is only saved if the chosen style is consistent (scatter
  <= 10%, cross-axis <= 0.10), every return to the centre succeeds, and the closure error is
  <= 30 px.
- The fake now models a game that samples the cursor once per frame. A test reproduces the
  failure and checks that the gate stops it.

### M1 live runs 5-7 (2026-09-25 23:02-23:05): timing fixed; drift found and fixed

- **Drags are consistent now:** 1.4-5% scatter and cross-axis <= 0.10 (runs 2-4: 35-50%).
  `v800` was chosen, and single-axis gains are about 1.0 in x and about 0.92 in y.
- **Run 5 stopped:** its fastest style sat at cross-axis 0.103 against a 0.10 limit, while
  `v400` passed. The gate now picks the fastest style that passes (limits 12% / 0.15), and only
  stops if none does.
- **The view clamps at exactly 30.0 minimap px from the centre** in every direction (both
  complete runs). There is no overshoot at the axis edges.
- **The minimap scale differs per axis:** x 26.20 in both runs, y 24.9-25.2 (the view is
  slightly foreshortened in y). The ring is drawn on a 0.5 px grid, which gives about ±6.5 slide
  px per reading.
- **Drift:** the tracked position was 28 and 43 px off after about 100 drags. The main cause was
  using one minimap scale for both axes (resyncs and featureless drags used the wrong y scale).
  Replaying the recorded drags with per-axis scales and a 0.25 minimap blend per drag gives
  4-8 px against the image truth. This is a regression test on the real logs.
- **Fixes:**
  - per-axis `ring_scale`
  - the minimap is blended into the tracked position on every drag
  - the "agrees with the minimap" check is tightened to 20 px + 4% (it was 15% of the move,
    which let 47 px mismatches through)
  - bounds are reported from the minimap
  - the edge probe's minimap sign is fixed

## M3 cleaning (`clean`)

`launch.cmd clean [--modes survey bright polarized] [--max-clicks N] [--revisits N]`

**First live run (2026-09-26 02:38), map-driven version.** It removed 2 real Survey blocks, then
clicked 5 white specks. Specks on that mineral have dE ~43 (70-84 on the earlier one), so the
contrast-based speck filter missed them. It was also far too slow: 43 s of mapping before the
first click, then ~2.4 s per click.

**Current design, driven by the minimap.** The minimap in each mode lists exactly that mode's
remaining imperfections, so it is the to-do list. There is no mapping sweep, and specks (no dot
in any mode, and the user confirmed they cannot be removed) are never targeted. Objects under
600 px that are not lines are never clicked either (live: blocks ~4200 px, specks 250-300).

For each mode, repeat until its minimap is empty, or a round removes nothing (up to 4 rounds):
1. Read the dots and group them into views (radius 0.62 r / zoom).
2. Pan to each view in Survey with `quick_pan_to`: v1600 drags, with position taken from the
   minimap ring (no image registration), and tolerance about 1/3 of the view.
3. Bright/Polarized: switch mode (zoom 1.40x / 2.08x, measured by SIFT at the first switch).
   The tool leaves the mode only when the next view needs a pan.
4. Detect once per view. Each dot's object is the highest-contrast object near where the dot
   says (the full-contrast one in this mode). Click them all in a burst (absolute move + wiggle,
   60 ms dwell, 50 ms hold).
5. Verify every click from one after-frame. A click that revealed another block underneath gets
   a follow-up; blocks hidden under others surface in the next round's minimap.

5 clicks in a row without effect stop the run.

**Second live run (2026-09-26 02:54, Survey only).** It took 35 s: 16 removed plus 1 revealed
underneath, with 1 Survey dot left. The one left was a rock block lying on two faded Bright
blocks: all three merged into one blob, whose deepest point was on a faded block, so 4 clicks did
nothing. Panning was the slow part, at 18 pans for 15 dots and ~0.9 s each.

**Changes:**
- **Aim:** in a merged blob, click the part near the blob's top contrast (the full-contrast
  object) nearest the dot. The regression sample is `white-rock-on-faded-pair.png`.
- **Route:** greedy view planning covers the most dots per view, discounted by travel. The view
  radius is 0.68 r / zoom.
- **Pan speed:** quick pans at 3000 px/s without the primer, up to 640 px per drag. The pan speed
  drops to 1600 px/s automatically if fast drags lose more than 25%.

**Third live run (2026-09-27 02:36, Survey).** Survey was cleaned in ~25 s: 24 removed, 1 click
without effect, 14 pans at 3000 px/s (the drags moved 89-100% of the request, so the fast speed
is fine). Then the run stopped on a game rule nobody knew about: **a finished mode's button
turns grey (BGR ~64,56,52) and can no longer be selected**, and sometimes no mode is highlighted
at all. The follow-up Bright runs could not start, because the design panned only in Survey.

**Changes:**
- **Button states:** `modes.read_buttons` returns active / enabled / disabled per button.
  Disabled means the mode is finished (`ModeDone`), which ends that mode's cleaning at once.
  It is never clicked, and "no mode highlighted" is accepted.
- **Panning in any mode:** each mode is panned in its own view. `Panner.set_mode` sets the zoom
  and the minimap ring size (measured: 19.26 / 14.15 / 10.12 px). `quick_pan_to` scales cursor
  moves by the zoom and learns the exact per-mode gain from the ring. This also saves the zoom
  animations of switching modes for every pan.
- **Starting state:** the run may start in any mode; modes that are already finished are skipped.

Offline (the fake now greys finished buttons, shrinks the ring when zoomed, and pans 1.15/zoom as
a deliberate error): 15/15 removed in 23 s for all three modes, the gain was learned as 1.14, and
no click was wasted.

**Repeated live runs (2026-09-27, all modes).** Overall good: some gems were fully cleaned
hands-off. Others kept a few imperfections in an earlier mode, and runs took 23-63 s, against
14-20 s by hand. In the fastest full run (37 s), clicks took ~0.16 s each, and every burst cost
~2.4 s: 1-2 pans of ~0.7 s (capped at 640 cursor px), a full-frame judge of ~150 ms per click,
a 0.45 s wait, and detection. Mode switches took 1.5-2 s. On a dark-grey mineral, faded
Polarized lines (dE 45-56) beat the low-contrast rock blocks (dE 13-16) under the
highest-contrast rule, and one dead spot was clicked 3 times.

**Changes (v0.9):**
- **Revisits:** after the pass over all modes, every requested mode whose button is not greyed
  out gets another pass (`--revisits`, default 2). The fake test hides a dot on the first pass.
- **Survey picks the roughest object:** the mean texture inside the object, 4 px in from its
  edge. On the samples, rock is 2-4, faded blocks <= 1.1 and lines ~0.4. Other modes keep
  highest contrast.
- **Dead spots:** a spot (in slide coordinates, 25 px) where a click did nothing is never
  clicked again in that mode.
- **Quick verification:** only a patch around each click is examined, against a background taken
  from the patch border, since inside big zoomed blocks the local median is the block itself. It
  agrees with the full judge on all 6 live before/after pairs. The wait is 0.3 s, plus 0.3 s
  more only if something looks unchanged.
- **Long, fast pans:** the drag starts inside the microscope but may end anywhere in the window
  (±1280 px across, about ±795 vertically), max 1400 px. It runs at 4500 px/s, stepping down to
  3000 and then 1600 when drags move < 75% of the request.
- **Mode switch** is polled every 40 ms instead of 80.

**v0.9 live run (2026-09-27 14:10, 52.6 s).** Pans were faster: every 4500 px/s drag moved
83-110% of the request, and there was no fallback. Bright and Polarized finished (Polarized on
the revisit). But ~10 Survey blocks were left, clustered in one spot:
- Their minimap dots overlapped into one 31x23 px blob, which `minimap_features` discarded with
  a width limit (0.45 mr) meant for ring slivers. The minimap therefore read 0, and the
  revisits did nothing. Fix: reject only thin blobs (distance-transform thickness), and allow
  up to 16 dots per merged blob. On the screenshot (`samples/raw/green-cluster-missed.png`) it
  now finds 9 of ~10.
- All 5 Survey clicks without effect were on bright cyan Polarized lines near a Survey dot. Fix:
  in Survey, objects with interior texture < 1.5 are never clicked (`SURVEY_MIN_TEXTURE`). The
  fake now has live-like rock texture, and lines only in Polarized.
- A warning is logged when a minimap looks empty but the button is not greyed out.

**v0.9.1 live (2026-09-27 14:33-14:35, 3 gems): all three fully cleaned, 31.3 / 31.3 / 34.8 s.**
Each view costs ~1.3 s: drag ~0.64 s (calibrated frame gaps: keep them), detection 114 ms, a
~0.16 s click, a 0.3 s wait, and grabs of 29 ms. Mode switches take 1.2-1.7 s. Twice a last
Survey dot got no click in its round and was only cleaned on the revisit (~3.5 s extra). One
short drag (162 px) read a ratio of 0.72 from the ring's 0.5 px grid, which wrongly dropped the
speed to 3000 px/s. Every long 4500 px/s drag moved 0.82-1.08 of the request.

**Changes (v0.9.2):** only drags > 300 px can slow the pan speed. The first look after a burst
comes at 0.15 s, plus 0.3 s only if something seems unchanged. No SIFT zoom measurement: Bright
measured 1.401-1.408x every time, so the constants are used. A dot that gets no click saves
`unresolved/NN-mode.jpg`, to find out why Survey sometimes needs a revisit.

**Outputs:** `clean-report.txt`, `clean.json`, `unresolved/`, `clicks/` (before/after of every click that did not
cleanly remove, and of the first dozen), `hud-before.png` / `hud-after.png`.

Offline (fake microscope with zoom and click-to-remove): 15/15 removed, 15 clicks, 47 s for all
three modes, no panning while zoomed.

## M2 slide mapping (`map-slide`, re-analyse with `analyze-map <run folder>`)

**First live run (2026-09-26 00:18): the design had to change.** Switching modes *zooms* the
view: Bright is about 1.40x and Polarized about 2.08x, about the microscope centre, measured
from the saved frames by SIFT. Frames of one stop in different modes are therefore not the
same view.

That run also panned while zoomed, because the serpentine mode order left some stops in
Polarized. That corrupted positions by up to 270 px and produced 327 bogus "objects". The
minimap turned out to be the key: in each mode it shows that mode's remaining imperfections
(19 / 8 / 5 dots that run), which is consistent with the user's counts in
`samples/mineral_counts.tsv`.

**Current design** (hands-off, under a minute; needs `pan_calibration.json` from M1):

1. **Minimap per mode:** at the start position, switch Survey → Bright → Polarized → Survey and
   read the minimap dots in each mode. The tool waits for the zoom animation, never pans or
   captures the view while zoomed, and checks that the Survey view came back unchanged.
2. **Sweep:** a serpentine grid of 16 stops, Survey only (asserted before every pan).
3. **Stitch:** overlapping stops are registered pairwise and positions solved by least squares.
   Each mosaic pixel comes from the frame whose centre is closest.
4. **Detect:** every imperfection is visible in Survey (own ones at full contrast, the others
   faded). Detection uses colour difference from the LOCAL background (large median, Lab), with
   a mean contrast >= 12. Touching blocks are split only when non-convex or oversized. Objects
   seen from several stops are merged by mask overlap in slide coordinates.
5. **Modes:** each object is matched to the nearest minimap dot of any mode (within 60 px,
   one-to-one). A full-contrast look prefers Survey dots and a faded look prefers
   Bright/Polarized dots. Objects without a dot are marked `survey?` or `faded?`.

**Outputs:**
- `map.png`: Survey mosaic, objects circled in their mode colour, minimap dots as thin rings,
  and click points
- `targets.json`
- `mosaic-survey.jpg`
- `minimap-<mode>.png`
- `map-report.txt`

Offline (fake microscope that zooms like the game, 15 objects in 3 modes): all 15 found, all
modes correct, every minimap dot matched, 3 mode clicks, no panning while zoomed.

### Live run 2026-09-26 01:06 (`samples/map-live1`)

The sweep was clean: 3 mode switches, the view came back exactly, and 16 Survey stops in 38 s.
The user then cleaned the slide by hand and counted **37** imperfections. By eye, the mosaic
shows 22 rock blocks, 5 faded blocks, about 8 line segments and 4 floating specks.

Fixes from this run (all applied offline with `analyze-map`):
- **Minimap dots:** a single dot is a 21 px fill, so merged dots are counted by area and placed
  by k-means. Dots are drawn over the view ring, so the ring no longer needs masking. Now
  16 / 5 / 7 dots (dots stacked exactly on another cannot be seen).
- **Touching blocks:** OpenCV watershed cut them into slices. They are now split with a power
  diagram over well-separated distance peaks.
- **Bent lines:** each straight segment is a separate imperfection (one Polarized dot per
  segment). The line test is half-width <= 15 px, open shape, and skeleton >= 80 px. Segments
  come from Hough on the skeleton, with collinear pieces fused, and pieces of one line seen
  from different stops are joined.
- **Edge slivers:** objects only ever seen cut off at a view edge are absorbed into the whole
  object next to them.
- **Specks:** small bright squares have no minimap dot and show parallax (~3.5% of the view
  offset), so they float above the slide. They are listed separately and not counted.

**Result:** 33 found (15 Survey + 3 `survey?`, 4 Bright, 7 Polarized, 4 `faded?`). Minimap
dots matched: Survey 15/16, Bright 4/5, Polarized 7/7. Known gaps: blocks stacked almost
exactly on top of each other count once, and a few short duplicate line pieces remain.

## M1 pan calibration (`pan-calibrate`)

Hands-off, about a minute. It writes `pan_calibration.json` to the project root, but only if
the run completes. It runs five steps:

1. **Latency:** hold right, jump the cursor 60 px, and grab frames continuously to find when
   the view settles. The settle delay is set from this.
2. **Drag styles:** 60/200 px drags in 4 directions, moving the held cursor as one jump,
   10 px steps, or 3 px steps. It fits `content shift = G @ cursor` for each style and keeps the
   most consistent one. Styles that barely move the view are rejected.
3. **Edges:** from the slide centre (minimap centre), it walks in 8 directions until panning
   stops, returning to the centre each time. This gives the slide boundary (extent per
   direction) and whether it is round.
4. **Accuracy:** closed-loop `pan_to()` six targets inside the boundary.
5. **Closure:** return to the centre and compare with the first centre frame, by image. This is
   the true accumulated error.

Position tracking: V = view centre in slide px from the minimap centre. Every drag is measured by
`register.measure_shift`: template matching around a prior from the minimap ring (coarse,
independent), with the pan model and zero as fallbacks. Only matches that agree with the ring
are accepted. If the tracked V drifts more than 40 px from the ring's absolute position, it
resyncs. After the edge walks, the minimap scale is refit over the whole run.

Outputs in `runs/<ts>-pan-calibrate/`: `report.txt`, `m1-results.json`, `drags.jsonl` (one line
per drag), `drags/NNN.jpg` (frame after each drag), `pan-map.png` (path, minimap estimates,
edges), `start.png`, `centre-first.png`.

## Setup

`setup.cmd` reuses the existing `.venv` (or creates one with Python 3.12/3.11), installs
`requirements.txt` and runs the offline self-test. Then use `launch.cmd`, which opens the menu.
You can also run a command directly: `launch.cmd click-test --strategies S0 S4`.

## Safety

- Hold **F8** (or Esc) to stop. The tool also stops if Roblox stops being the active window,
  if the window moves or resizes, if the cursor is pushed into the top-left screen corner,
  or when the time limit or click budget is reached.
- A watchdog thread releases any held mouse button when it stops; every exit path releases too.
- No input is sent until Roblox is the foreground window. `--dry-run` sends none at all.
- No extra windows open. Everything is written to `runs/<timestamp>-<command>/`.

## M0 commands

- **click-test**: 6 trials. In each one you hover a block and tap F6, then move the mouse away
  and let go. The tool moves back using one strategy and clicks. Before/hover/after frames
  decide whether the block was removed.
  - S0 control: no movement; the tool only presses
  - S1: replica of v0.4 (absolute moves)
  - S2: SetCursorPos + wiggle
  - S3: absolute + wiggle
  - S4: relative closed-loop
  - S5: slow relative + wiggle, long dwell/hold
- **drag-test**: holds the right button at the microscope centre and moves the cursor through
  ±60/±120 px offsets and back. It captures frames 0.25 s and 0.75 s after each stop and
  measures slide movement by phase correlation and by the minimap view ring. It does this once
  with relative moves and once with absolute moves.
- **snapshot**: saves a frame + detected layout (no input). Use it to collect samples.
- **selftest**: offline tests on `samples/`.
- **analyze-drag <run folder>**: re-analyses a saved drag run.

## Layout

```
geoclean/   winapi (ctypes), safety (guard/watchdog), inputs (strategies), capture (mss.MSS),
            vision (layout, minimap, registration, click judge), m0 (tests), diag, cli
tests/      offline tests + fake Windows/Roblox
samples/    raw screenshots (green mineral v0.3 before/after pairs, tan mineral, minimap crops)
previous/   v0.1-v0.4 prototype, its evidence and diagnostics (reference only)
runs/       diagnostics from live runs (LATEST.txt points to the newest)
```
