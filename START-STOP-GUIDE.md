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

**If the app crashes**, it restarts by itself after 5 seconds (the window says so). After 5 crashes within
10 minutes it stops trying and leaves the error message on screen.

**To read messages after the window is closed**, open `C:\VisionLabeler\data\logs\visionlabeler.log` in
Notepad (newest lines at the bottom). `supervisor.log` in the same folder lists every crash and restart.

**To change a setting** (add a camera, keep data longer, turn HTTPS on…), open **Settings** in the top
menu, change it, press **Save**, then **Restart now**. The page comes back by itself in about a minute.

**Each morning**, the top of the Live page says how many people came today, the busiest hour, alerts and camera
problems, against a usual day.

**To see what a person did**, press **▶ Video** on them in the people list (admins).

**To add old videos** (from a phone or a recorder), open **Import** in the top menu.

**To watch from another site** (another house or office), follow `docs\SECOND-SITE.md`.

**To check that everything works**, open the **Health** page (top menu, or click "health OK" on the Live
page). Each line is green when it works; red lines say what is wrong and what to do.

**If you double-click `start.bat` twice**, the second one says "already using port 8000" and closes; only
one copy ever runs.

If something does not start, the window says what to do:

| Message | What to do |
|---|---|
| "Docker did not become ready within 3 minutes" | Open Docker Desktop, wait for "Engine running", run `start.bat` again |
| "Ollama is not responding" | The app runs with colors only. Start "Ollama" from the Start menu, then restart the app |
| "NO PICTURE FROM CAMERA" on the video | Check the lens cover; close Teams, Discord or other apps using the camera |
| "disk almost full" in the page header | Delete people you no longer need, or lower `storage.retention_days` |

### Start automatically when you log in (optional)

Double-click **`install-autostart.bat`** once. From then on VisionLabeler starts by itself (minimized) when
you log in to Windows. Undo with **`remove-autostart.bat`**.

## Before you shut down the PC

Double-click **`stop.bat`**. It:

1. stops the app, after it saves the people in view and finishes their labels;
2. backs up the database (at most once a day; the app also makes one every night while it runs; the last 7
   backups are kept in `data\backups`);
3. stops the database (your data is kept);
4. frees the GPU memory used by the vision model.

When it says **"VisionLabeler is fully stopped. It is now safe to shut down the PC."**, shut down Windows
as usual. This normally takes a few seconds.

## Other ways to stop

| Way | What it stops |
|---|---|
| **Stop** button, top right of the web page | The app only (database keeps running; fine if you will start again soon) |
| **Ctrl+C** in the `start.bat` window | The app only, same as the Stop button |
| `stop.bat` | Everything, with a backup (use before shutting down) |

## If you shut down without stopping

Nothing breaks. The database is crash-safe and recovers by itself on the next start, and people who were
in view are marked finished. The only loss: those people do not get their final update.

## Restoring a backup

Stop the app first (Stop button), then run, with the backup file you want:

```bash
docker exec -i visionlabeler-db pg_restore -U visionlabeler -d visionlabeler --clean --if-exists < data/backups/visionlabeler-YYYYMMDD-HHMMSS.dump
```

## Quick reference

| I want to... | Do this |
|---|---|
| Start after turning on the PC | `start.bat` |
| Stop before shutting down | `stop.bat`, wait for "safe to shut down" |
| Pause for a while, PC stays on | Stop button in the web page |
| Start automatically at login | `install-autostart.bat` (undo: `remove-autostart.bat`) |
| Use the demo photo instead of the webcam | `start.bat --source samples\bus.jpg` |
| See which camera number to use | `.venv\Scripts\python.exe run.py --list-cameras` |
