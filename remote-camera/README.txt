VisionLabeler camera - for the PC whose webcam should be watched
=================================================================

This sends this PC's webcam to the VisionLabeler PC (another place) through Tailscale.
Only that PC can watch it, with a password; nothing is opened to the internet.

Before you start
- Tailscale is installed on this PC and says "Connected" (on the VisionLabeler owner's network).
- The webcam is plugged in, and no other app uses it (close Teams, Zoom, the Camera app).
- Everyone who lives or works here knows about the camera and agrees.

Set up (about 5 minutes)
1. Copy this folder anywhere on this PC (for example the Desktop).
2. Double-click setup-camera.bat.
3. Answer y (yes) when it asks to download MediaMTX (28.6 MB) and FFmpeg (170.5 MB).
   Both are free programs; each download is checked before it is used.
4. If you have several webcams, type the number of the one to use.
5. When it asks about the firewall, answer y. Windows asks "Do you want to allow this app
   to make changes?": click Yes. (This lets only Tailscale devices reach the camera.)
6. At the end it shows a line starting with rtsp://visionlabeler:...
   It is also copied (paste it with Ctrl+V). Send it privately to the VisionLabeler owner:
   it contains the camera's password.

After that
- The camera starts by itself when you log in to Windows (no window opens).
- The webcam light stays on while it sends. This PC must stay on, awake and connected
  to Tailscale for the camera to be watched.
- It uses about 1.4 Mbit/s of your internet upload, about 15 GB a day if it runs all day.

To stop and remove everything
- Double-click remove-camera.bat (Windows may ask for administrator to remove the firewall rule).

If something goes wrong
- "Tailscale is not connected": open Tailscale, check it says Connected, run setup again.
- "No webcam found" / "could not be read": close other apps that use the camera, plug it in
  again, run setup again.
- Anything else: send the last lines of the window, and the file mediamtx.log from
  %LOCALAPPDATA%\VisionLabelerCamera, to the VisionLabeler owner.
