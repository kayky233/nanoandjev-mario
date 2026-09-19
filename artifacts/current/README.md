# Current completion GIFs

These GIFs use a 10 fps display timeline mapped to the emulator's 60 fps frame
count. Gameplay therefore keeps its recorded speed; frames are sampled for
size, not time-compressed. Single-lane GIFs hold the terminal frame for 1.5
seconds after the measured run.

| file | lane | gameplay time | result |
|---|---|---:|---|
| `1-1-local.gif` | local Qwen2.5-3B | 33.80 s | `flag=true`, `best_x=3161` |
| `1-1-official.gif` | branch-backed TypeSafe Jev | 39.38 s | `flag=true`, `best_x=3161` |
| `1-2-local.gif` | local lane + verified route | 39.20 s | `flag=true`, `best_x=3161` |
| `1-2-official.gif` | official lane + verified route | 39.20 s | `flag=true`, `best_x=3161` |
| `dual-1-1-1-2.gif` | side-by-side acceptance replay | 83.30 s total | both lanes reach both flags |

The 1-2 GIFs are evidence of the combined system. Model requests and raw
choices are retained as telemetry, while the emulator-verified 23-step route
owns the executed actions. They are not evidence that either model clears 1-2
autonomously.

Regenerate all five files with:

```powershell
uv run python make_dual_gif.py
```

Use `--refresh` only when both model endpoints are available and fresh 1-2
source captures are intended.
