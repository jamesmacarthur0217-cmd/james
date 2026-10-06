# VisionLabeler: starting and stopping

## When you turn on the PC

Docker Desktop and Ollama start by themselves with Windows. You only start the app:

1. Double-click **`start.bat`** in `C:\VisionLabeler`.
2. Wait. The window shows three checks:
   ```
   [1/3] Docker is ready        (right after boot this can take about a minute; it waits for you)
   [2/3] Database is ready
   [3/3] Ollama is ready
   ```
3. The browser opens http://127.0.0.1:8000 by itself about 15 seconds later, with the live camera view.

Keep the black `start.bat` window open while you use the system. Closing it stops the app.

If something does not start, the window says what to do:

| Message | What to do |
|---|---|
| "Docker did not become ready within 3 minutes" | Open Docker Desktop, wait for "Engine running", run `start.bat` again |
| "Ollama is not responding" | The app runs with colors only. Start "Ollama" from the Start menu, then restart the app |
| "NO PICTURE FROM CAMERA" on the video | Check the lens cover; close Teams, Discord or other apps using the camera |

## Before you shut down the PC

Double-click **`stop.bat`**. It:

1. stops the app, after it saves the people in view and finishes their labels;
2. stops the database (your data is kept);
3. frees the GPU memory used by the vision model.

When it says **"VisionLabeler is fully stopped. It is now safe to shut down the PC."**, shut down Windows
as usual. This normally takes a few seconds.

## Other ways to stop

| Way | What it stops |
|---|---|
| **Stop** button, top right of the web page | The app only (database keeps running; fine if you will start again soon) |
| **Ctrl+C** in the `start.bat` window | The app only, same as the Stop button |
| `stop.bat` | Everything (use before shutting down) |

## If you shut down without stopping

Nothing breaks. The database is crash-safe and recovers by itself on the next start. The only loss: the
people in view at that exact moment do not get their final update (their earlier labels are kept).

## Quick reference

| I want to... | Do this |
|---|---|
| Start after turning on the PC | `start.bat` |
| Stop before shutting down | `stop.bat`, wait for "safe to shut down" |
| Pause for a while, PC stays on | Stop button in the web page |
| Use the demo photo instead of the webcam | `start.bat --source samples\bus.jpg` |
| See which camera number to use | `.venv\Scripts\python.exe run.py --list-cameras` |
