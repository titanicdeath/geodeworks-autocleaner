# Geodeworks microscope cleaner — prototype 0.4

A Windows Python prototype that captures the primary display, identifies likely
imperfections, and performs bounded mouse clicks and optional right-button drags.
It estimates each image's background instead of assuming imperfections are purple,
brown, or any other fixed color. The minimap helps distinguish current targets
from faint shapes that remain visible in other lighting modes.

**Status:** vision checked offline against your 11 screenshots. Live Windows input,
Roblox's response to simulated input, drag calibration, and end-to-end cleaning have
NOT been tested in a running game. This is a first build for supervised testing.

## Start on Windows

**Updating from v0.1/v0.2/v0.3:** close the program, copy this archive's project contents
over your existing files, and keep your `.venv` and `calibration.json`. No setup
rerun or recalibration is needed if the game window and display are unchanged.
Version 0.2 requests that calibration/preview windows stay on top, restores and
requests focus for its own window, prints each countdown stage, and aims farther
inside matched objects. If Windows prevents activation, use Alt-Tab to select
the named calibration window. Calibration still uses the same seven clicks.

Version 0.3 adds a 0.20-second delay after cursor movement and a 0.10-second
button hold, checks the actual Windows cursor position before pressing, and waits
at least 0.75 seconds before checking the result. These changes address possible
timing problems; they do not establish what caused a previous missed click.

Version 0.4 changes movement from `SetCursorPos` to ordinary Windows `SendInput`
movement events along a short path. Button presses and vision are unchanged so
the movement difference can be tested separately. The new `mouse_input.py` file
must be copied with the rest of the update. No additional packages are needed.

The supplied v0.3 screenshots place all three failed automatic click coordinates
on solid gray blocks; both three-click runs show no meaningful changes to the
microscope, minimap or purity display. The console separately reports progress for
a manually positioned F6 test. This suggests an input-state or movement-handling
issue, but does not establish its exact cause. A returned cursor coordinate alone
does not prove Roblox updated its internal pointer targeting. v0.4 tests that
hypothesis; its live behavior still needs verification.

**Next check for this update:** run option 3 again, without moving the physical
mouse during the pass. If blocks disappear, proceed to option 4. If they do not,
send the new `last-test.zip`. It now includes cursor position before movement,
position afterward, and the movement backend. Keep the existing calibration.

## If a click shows no progress

The five-click option attempts **up to** five supported targets. If only one is
proposed, it can end after one attempt. `visual progress=False` means the image
heuristic saw no clear marker/foreground removal; it does not prove that Roblox
received or rejected the click.

Use **5 — Test ONE click on a target you choose**:

1. In the game, pan a known cleanable block toward the center of the microscope.
2. Start option 5 and switch to Roblox during the countdown.
3. Hover over the center of that block and tap **F6**. Do not left-click.
4. The program tries exactly one click at your selected position and stops.
5. Note whether the object disappears. If it does not, try clicking it yourself
   after the test ends to check whether it is cleanable in the current mode.

This test bypasses automatic target selection. It retains focus and stop checks.
After option 3, 4, or 5, the program creates `last-test.zip` in its folder. Send this
file with the outcome. It contains raw before/after display screenshots,
calibration, and per-attempt metadata including the requested and actual cursor
coordinates. Nothing is uploaded automatically. The underlying timestamped runs
are kept in `diagnostics/`; delete old runs when no longer needed. A failed input
attempt may have only the before screenshot and error metadata.

The actual cursor coordinate confirms cursor placement only. Even a matching
coordinate and an `input_sent` status do not prove Roblox accepted a button press.

## First-time setup

1. Install 64-bit Python 3.11 or 3.12 with its Python launcher if needed.
2. Extract this entire ZIP into a writable folder. Double-click `setup.cmd` once.
   Setup creates a local `.venv` and downloads the packages in `requirements.txt`.
3. Open Roblox's microscope on your **primary monitor**. Keep the game window in
   the same position and size after calibration.
4. Double-click `launch.cmd`, choose **1 Calibrate**, and switch back to Roblox
   during the five-second countdown.
5. A frozen screenshot opens. Click the microscope center, its right rim,
   minimap center, its right rim, and the Survey, Bright, Polarized buttons, in that
   order. These clicks go to the calibration picture. Right-click undoes a point;
   Enter saves; Esc cancels.
6. Launch again, choose **2 Preview**, then switch back during the countdown.
   Inspect the frozen result: **red crosses are proposed clicks**, orange marks
   contrasting shapes, and yellow circles mark detected minimap blobs. Preview
   sends no mouse input. It saves `preview.png` alongside the program.
7. If the red crosses lie on cleanable features, choose **3 Try five clicks**.
   This operates only in the current view and lighting mode, without panning.
8. After the small trial works, choose **4 Pan and clean all three modes** to test
   the experimental navigation. It is limited to 120 seconds, 120 total clicks
   (including mode buttons), and 60 drags. Run again if desired after inspecting.

**Stop:** hold **Esc** or **F8**, move the mouse into the top-left corner of the
primary display, or switch away from Roblox. Stopping ends the pass; relaunch to
resume. Held buttons are released on exit. No administrator access is required.

The program controls your normal mouse while active. Keep Roblox unobstructed.
If you move/resize the window, change display resolution or scaling, switch primary
monitors, or change the UI layout, calibrate again.

## What the prototype does

- Samples the dominant microscope background in Lab color space each frame.
- Finds contrasting shapes, avoiding the rim, purity overlay, and mode buttons.
- Estimates the minimap background separately, isolates small colored markers,
  and fits the pale viewing circle.
- Projects nearby map markers into microscope coordinates and snaps to a detected
  shape interior. It refuses to click if these cues do not agree.
- Captures a fresh frame after each click; reduced marker count or local foreground
  removal is a progress signal. It stops after three clicks without clear progress.
- For panning, measures how horizontal and vertical test drags move the map viewing
  circle. It recalibrates for each lighting mode, bounds subsequent drags, and checks
  whether they move toward the chosen marker.
- Visits Survey, Bright, Polarized only when cycling is requested. Darkfield is
  excluded because it was unavailable in the supplied screenshots.

## Known limits and assumptions

- The viewing circle must represent the microscope's field of view with aligned
  axes. Offline screenshots are consistent with this, but verify red crosses in
  each mode. Perspective, a different map convention, or a changed UI can break it.
- The mineral background must cover most of the usable microscope area and be
  reasonably uniform. Subtle translucent shapes, busy backgrounds, overlapping
  objects, or backgrounds too similar to targets can be missed.
- Map markers must differ from the background and have reasonably bright colored
  interiors; the view ring must remain pale. Markers touching each other can merge.
  Markers on the viewing ring can be suppressed. Counts are estimates, not exact
  game state. A later pan can separate some overlaps.
- Visible shapes are not guaranteed to be clickable in the selected mode. Map
  guidance reduces mistakes, but cannot prove clickability from screenshots.
- A lack of markers does not prove full completion. It does not read the purity
  text, and does not assume 100% purity is achievable: your screenshots show
  mineral-specific maximum values such as 34%, 57%, and 91%.
- The program may stop on an unresolved centered target instead of finishing the
  whole mineral. Save a Preview to help improve detection. It does not click blank
  areas as a fallback or repeatedly retry one coordinate in an unchanged view.
- Pan calibration can fail near a movement boundary. Pan manually nearer the map
  center and retry. Ordinary simulated mouse input may not work with your Roblox
  client; this build stops if it cannot observe useful results.
- A five-click trial can still select a wrong feature. Whether wrong clicks have a
  gameplay penalty remains unknown; preview before trying input.
- Changes in animations/lighting or overlapping markers can make the progress
  heuristic inaccurate. This is not an unattended farming system.

## Commands and tuning

Run these from the extracted folder after setup:

```bat
.venv\Scripts\python.exe cleaner.py calibrate
.venv\Scripts\python.exe cleaner.py preview
.venv\Scripts\python.exe cleaner.py run --clicks 5 --seconds 30
.venv\Scripts\python.exe cleaner.py input-test
.venv\Scripts\python.exe cleaner.py run --pan --cycle --clicks 120 --seconds 120
.venv\Scripts\python.exe cleaner.py preview --contrast 14
.venv\Scripts\python.exe cleaner.py analyze --image example.png --output marked.png
.venv\Scripts\python.exe -m unittest -v test_cleaner.py
```

Lower `--contrast` detects subtler features but may include more background noise;
higher values are stricter. Default is 20. Use the same override for Preview and Run,
or edit `contrast` in `calibration.json`. `settle` controls the post-action delay in
seconds (default 0.4); increase it if the game responds slowly. Screenshot analysis
uses automatically estimated geometry and does not require Windows or calibration.

## Included files

- `cleaner.py`: calibration, preview, mouse controller, cleaning loop, menu.
- `vision.py`: adaptive detection and annotated preview generation.
- `mouse_input.py`: Windows movement events and virtual-display coordinates.
- `test_cleaner.py`: synthetic vision tests and mocked control-stop tests.
- `evidence/`: annotated versions of all 11 supplied screenshots and an offline
  validation report. Red crosses are proposals, not verified successful clicks.
- `setup.cmd`, `launch.cmd`, `requirements.txt`: installation and launch helpers.

The application performs no network requests. Package installation uses your
configured Python package index. Calibration and preview files remain local.

To refine the next version, share `preview.png` from each mode and the console
output of the five-click trial. A short recording of right-dragging and successful
cleaning would help validate the map projection and navigation behavior.

Implementation references:
- https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html
- https://python-mss.readthedocs.io/
- https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-sendinput
- https://learn.microsoft.com/en-us/windows/win32/api/winuser/ns-winuser-mouseinput
