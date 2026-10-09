# Watching from another site (Tailscale)

This lets a PC at another place (another house, an office) open VisionLabeler over the internet, safely: only
devices in **your own Tailscale network** can reach it, nothing is opened on your router, and everything is
encrypted. Each step says how to check it on the **Health** page of this PC (top menu → Health).

**This PC** = the PC with the camera and VisionLabeler. **Other PC** = the PC at the other site.

## 1. Tailscale on both PCs

On this PC Tailscale is already installed and signed in.

1. On the **other PC**, download Tailscale from https://tailscale.com/download and install it.
2. Sign in with **the same account** you use on this PC. (Or, for another person's PC: in the Tailscale admin
   page on this PC's account, *Share* this PC with them, and they sign in with their own account.)

**Check:** Health → "Tailscale (another site)" says **1 other device(s)** (the number goes up by one).

## 2. A login account for the other site

This PC never needs to log in; other PCs do. A viewer account sees the live video (faces blurred) and can
search, nothing more. An account already exists for `TIXUAN`. To add another one, on this PC:

```bash
.venv\Scripts\python.exe tools\users.py add <name> --role viewer
```

**Check:** Health → "other devices" says how many login accounts there are.

## 3. Let Tailscale devices through the Windows firewall (once)

Windows counts the Tailscale network as "public" and blocks it. This rule lets **only Tailscale addresses**
reach **only VisionLabeler's HTTPS port**:

1. On this PC, click Start, type `PowerShell`, right-click **Windows PowerShell** → **Run as administrator**.
2. Paste this one line and press Enter:

```bash
New-NetFirewallRule -DisplayName "VisionLabeler HTTPS (Tailscale)" -Direction Inbound -Protocol TCP -LocalPort 8443 -RemoteAddress 100.64.0.0/10 -Action Allow -Profile Any
```

**Check:** Health → "firewall for Tailscale" is green (up to 5 minutes later: the page checks every 5 minutes).

## 4. Open it on the other PC

1. Start VisionLabeler on this PC as usual (`start.bat`). When Windows asks whether Python may use the network,
   allow **Private networks** only.
2. On the other PC, open the address from Health → "Tailscale (another site)": `https://100.x.x.x:8443`
   (your numbers; the `s` in `https` matters).
3. The browser warns that the certificate is not trusted. That is expected: this PC made its own certificate.
   Click the warning's details and compare the **SHA-256 fingerprint** with the one on this PC's Health page
   ("other devices") or in the app's window at startup. If they match, continue (Advanced → Continue/Proceed).
   If they do not match, stop: you are not talking to your PC.
4. Log in with the account from step 2.

**Check:** Health → "Tailscale (another site)" shows **last login over Tailscale** with the time and the name.

## If it does not work

| What you see | What to do |
|---|---|
| The page does not load at all on the other PC | Is VisionLabeler running here? Is the firewall rule (step 3) green? Are both PCs "Connected" in Tailscale? |
| "Connection refused" | `web.https` must be on (Settings page) and the app restarted |
| The login page says the name or password is wrong | Five wrong tries block that device for 5 minutes; reset the password with `tools\users.py password <name>` |
| It worked yesterday, not today | This PC may be asleep or VisionLabeler stopped; the other PC cannot wake it |

Do **not** use Tailscale Serve/Funnel, Cloudflare Tunnel, port forwarding on the router, or remote-desktop
apps for this: they would bypass the login or open the PC to everyone.
