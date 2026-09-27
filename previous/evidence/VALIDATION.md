# Offline validation — prototype 0.4

20 offline tests passed. Four new checks cover Win32 INPUT structure size and event flags, coordinate normalization across monitors with negative origins, rejected movement events, and controller movement without SetCursorPos. New tests cover cursor settling before button press, aborting on a cursor mismatch, selecting exactly one user-chosen target with F6, and saving raw diagnostic image pairs and cursor metadata. Earlier vision, focus, release, and window tests still pass. Windows focus behavior is mocked; live Windows window activation has not been tested here.

The vision algorithm is unchanged from v0.2, where all 11 original raw screenshots were reprocessed. Every proposed click falls inside a detected shape. The cleared local Survey example still produces no proposed clicks. The two user-supplied annotated previews were reviewed visually, not treated as raw detector test frames.

Map counts are estimates. End-to-end Roblox cleaning remains unverified. The user has confirmed that v0.1 calibration and preview do open; the initial calibration window was simply unnoticed.

| Screenshot | Shapes | Map blobs | Proposed clicks |
|---|---:|---:|---:|
| 0370e2e9-fb20-4d0f-917c-932234f5cd1d.png | 9 | 14 | 4 |
| 5a011c10-ce11-4728-976f-8eb82a563d62.png | 2 | 10 | 3 |
| 633ab452-1367-4f65-8544-c437daf5a2db.png | 4 | 5 | 2 |
| 69cb0169-7ab1-4646-bb8b-20ac8d05384c.png | 2 | 4 | 3 |
| 75ddaa33-c52b-4b0f-a27b-d69776024c17.png | 2 | 1 | 1 |
| 7e37168c-7910-4b22-beb6-5c20fe90d7a2.png | 3 | 5 | 2 |
| e29ae45b-b844-44e6-a690-f64b6e5663e3.png | 9 | 15 | 5 |
| e30ac374-feab-48dc-a9c2-15fc2ac2fec3.png | 1 | 7 | 2 |
| e4329e5a-394a-4ecc-b54f-4ed0923d7df3.png | 5 | 18 | 3 |
| e6cf68ef-76ba-400b-a0c1-1cb6bccea284.png | 8 | 7 | 1 |
| fe491f9c-cf9c-4d1b-88b0-02679eee78cc.png | 4 | 8 | 0 |

The user has now reported a v0.2 automatic click with no observed visual progress. This has not been reproduced locally; v0.3 adds a user-selected input test and raw before/after evidence to identify whether target selection, timing, or input handling is responsible.

Six v0.3 automatic attempts (two sets of three) were inspected. Requested and reported cursor coordinates match. The three points lie on visible solid blocks. Comparing each before/after pair found no pixels changing by more than 10 intensity levels in the tested microscope, purity, and minimap regions. The separately reported F6 success has no corresponding raw image pair in this upload. SendInput behavior is tested with mocks here; compatibility with Roblox remains to be checked live.
