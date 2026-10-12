VisionLabeler camera - for the PC whose webcam should be watched
=================================================================

This SENDS this PC's webcam to the VisionLabeler PC (another place) through Tailscale.
It only makes an outgoing connection: no open port, no firewall change, no administrator.
(On a company PC, ask first whether this is allowed.)

Before you start
- Tailscale is installed on this PC and says "Connected" (on the VisionLabeler owner's network).
- The webcam is plugged in, and no other app uses it (close Teams, Zoom, the Camera app).
- Everyone who lives or works here knows about the camera and agrees.
- The VisionLabeler owner has given you this camera's SEND ADDRESS (it starts with rtsp://
  and holds a password: keep it private).

Set up (about 5 minutes)
1. Copy this folder anywhere on this PC (for example the Desktop).
2. Double-click setup-camera.bat.
3. Paste the send address when it asks (right-click pastes in the window), press Enter.
4. Answer y (yes) when it asks to download FFmpeg (170.5 MB, a free program; the download is
   checked before it is used).
5. If you have several webcams, type the number of the one to use.
6. It sends a 5-second test picture. "received by the VisionLabeler PC" means it works.

After that
- It sends by itself whenever you are logged in to Windows (no window opens), and starts again
  by itself after a network drop or when the VisionLabeler PC restarts.
- The webcam light stays on while it sends. This PC must stay on, awake and connected to Tailscale.
- It uses about 1.5 Mbit/s of your internet upload, about 15 GB a day if it runs all day.

To stop and remove everything
- Double-click remove-camera.bat.

If something goes wrong
- "Tailscale is not connected": open Tailscale, check it says Connected, run setup again.
- "refused the password": ask for the send address again (the owner can make a new one).
- "did not answer": VisionLabeler must be running on the other PC, with its firewall rule.
- "No webcam found" / "could not be read": close other apps that use the camera, plug it in
  again, run setup again.
