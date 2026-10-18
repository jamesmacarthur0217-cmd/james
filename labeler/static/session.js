// Shared by every page: shows who is logged in, hides admin controls (class "admin-only") from viewers,
// and goes to the login page when the login has run out. This PC itself is always admin and never logs in.
(() => {
  const plainFetch = window.fetch;
  window.fetch = async (...args) => {
    const r = await plainFetch(...args);
    if (r.status === 401 && location.pathname !== "/login") location.href = "/login";
    return r;
  };

  async function show() {
    let me = null;
    try {
      const r = await plainFetch("/api/me");
      if (r.ok) me = await r.json();
    } catch {}
    if (!me) return;
    if (me.role === "admin") document.body.classList.add("is-admin");
    const header = document.querySelector("header");
    if (me.local || !header) return;
    const box = document.createElement("div");
    box.className = "who";
    box.textContent = `${me.username} · ${me.role} `;
    const out = document.createElement("button");
    out.textContent = "Log out";
    out.addEventListener("click", async () => {
      await plainFetch("/api/logout", { method: "POST" });
      location.href = "/login";
    });
    box.append(out);
    header.append(box);
  }

  // --- Alert pop-ups on every page (also a phone's browser) ---------------------------------------------
  // New alerts since the page opened appear bottom-right: most for 20 seconds; a fall stays until OK is
  // clicked and beeps until then. Browsers play sound only after a click on the page, so a fall pop-up says
  // so while the sound is still blocked; any click turns it on.
  const escHtml = (s) => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
  let lastAlert = null, audio = null, beeping = null;

  function toastBox() {
    let box = document.getElementById("toasts");
    if (!box) {
      box = document.createElement("div");
      box.id = "toasts";
      box.className = "toasts";
      box.setAttribute("aria-live", "assertive");
      document.body.append(box);
    }
    return box;
  }

  function unlockSound() {
    try {
      audio = audio || new (window.AudioContext || window.webkitAudioContext)();
      if (audio.state === "suspended") audio.resume();
    } catch {}
    setTimeout(updateAlarm, 100);
  }
  document.addEventListener("pointerdown", unlockSound);
  document.addEventListener("keydown", unlockSound);

  function beep() {
    if (!audio || audio.state !== "running") return;
    const t = audio.currentTime;
    for (const [start, freq] of [[0, 880], [0.5, 660]]) {   // two tones, like an alarm
      const osc = audio.createOscillator(), gain = audio.createGain();
      osc.type = "square";
      osc.frequency.value = freq;
      gain.gain.setValueAtTime(0.0001, t + start);
      gain.gain.exponentialRampToValueAtTime(0.2, t + start + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, t + start + 0.4);
      osc.connect(gain).connect(audio.destination);
      osc.start(t + start);
      osc.stop(t + start + 0.45);
    }
  }

  function updateAlarm() {
    const open = document.querySelector(".toast.fall");
    const blocked = !audio || audio.state !== "running";
    document.querySelectorAll(".toast.fall .sound-off").forEach(el => { el.hidden = !blocked; });
    if (open && !beeping) { beep(); beeping = setInterval(beep, 2000); }
    else if (!open && beeping) { clearInterval(beeping); beeping = null; }
  }

  // Who or what an alert is about (also used by the Alerts page).
  window.VL = window.VL || {};
  window.VL.alertWho = (e) => e.unusual ? "unusual quiet" : e.quiet ? "no movement" : e.help ? `help sign · person #${e.track_id}` : e.scene_id ? "scene question" : e.animal_id ? `animal #${e.animal_id}`
    : e.vehicle_id ? `vehicle #${e.vehicle_id}` : `person #${e.track_id}`;

  function showAlert(e, admin) {
    const t = document.createElement("div");
    const picture = e.crop ? `<img src="/crops/${encodeURIComponent(e.crop)}" alt="">` : "";
    const time = new Date(e.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    if (e.help) {   // the help sign (labeler/gestures.py): red with the beep, like a fall
      t.className = "toast fall help";
      t.setAttribute("role", "alertdialog");
      t.innerHTML = `${picture}<div><b>HELP SIGN</b> · ${escHtml(e.camera)} · ${time}<br>Someone held a hand above
        their head (person #${e.track_id}).${admin ? ' <a href="/alerts">See the clip</a>' : ""}<br>
        <small class="sound-off">Click anywhere on this page to hear the alarm.</small><br>
        <button class="primary">OK</button></div>`;
      t.querySelector("button").addEventListener("click", () => { t.remove(); updateAlarm(); });
    } else if (e.fall) {
      t.className = "toast fall";
      t.setAttribute("role", "alertdialog");
      t.innerHTML = `${picture}<div><b>FALL</b> · ${escHtml(e.camera)} · ${time}<br>Someone fell and has not got up
        (person #${e.track_id}).${admin ? ' <a href="/alerts">See the clip</a>' : ""}<br>
        <small class="sound-off">Click anywhere on this page to hear the alarm.</small><br>
        <button class="primary">OK</button></div>`;
      t.querySelector("button").addEventListener("click", () => { t.remove(); updateAlarm(); });
    } else if (e.quiet || e.unusual) {   // no movement (labeler/quiet.py) or unusual quiet (labeler/unusual.py): stays
      // until OK, like a fall, but without the beep
      t.className = "toast quiet";
      t.setAttribute("role", "alertdialog");
      t.innerHTML = `<div><b>${e.unusual ? "UNUSUAL QUIET" : "NO MOVEMENT"}</b> · ${escHtml(e.camera)} · ${time}<br>${escHtml(e.message)}<br>
        <button class="primary">OK</button></div>`;
      t.querySelector("button").addEventListener("click", () => t.remove());
    } else {
      t.className = "toast";
      t.innerHTML = `${picture}<div><b>Alert</b> · ${escHtml(e.camera)}<br>${escHtml(e.message)}<br>
        <small>${window.VL.alertWho(e)}</small></div>`;
      t.addEventListener("click", () => t.remove());
      setTimeout(() => t.remove(), 20000);
    }
    toastBox().prepend(t);
    updateAlarm();
  }

  async function pollAlerts(admin) {
    try {
      const r = await plainFetch("/api/alerts/events?since_id=" + (lastAlert ?? 0) + "&limit=10");
      if (!r.ok) return;
      const data = await r.json();
      if (lastAlert === null) { lastAlert = data.latest_id; return; }   // don't replay old alerts
      for (const e of data.events.reverse()) {
        lastAlert = Math.max(lastAlert, e.id);
        showAlert(e, admin);
      }
    } catch {}
  }

  // --- Home / Away switch in the header of every page (labeler/arming.py) -----------------------------
  // Admins and viewers may switch it; a confirm box first, so a stray tap on a phone does not. Other scripts
  // read window.VL.arming and listen for the "vl-arming" event (the Live page's tile); VL.switchArming() asks too.
  const hhmm = (iso) => iso ? new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "";
  const dayTime = (iso) => {
    const d = new Date(iso), today = new Date().toDateString() === d.toDateString();
    return today ? hhmm(iso) : d.toLocaleDateString([], { weekday: "short" }) + " " + hhmm(iso);
  };
  const Nice = (m) => m === "away" ? "Away" : "Home";
  window.VL = window.VL || {};
  window.VL.armingText = (s) => {
    if (!s) return "";
    let text = s.since ? `${Nice(s.mode)} since ${dayTime(s.since)} (${s.why === "schedule" ? "schedule" : s.by})` : `${Nice(s.mode)}`;
    if (s.next) text += ` · schedule: ${Nice(s.next.mode)} at ${dayTime(s.next.at)}`;
    return text;
  };

  function drawArming(s) {
    window.VL.arming = s;
    const b = document.getElementById("arming");
    if (b) {
      b.className = "arming " + s.mode;
      b.innerHTML = s.mode === "away" ? "<span aria-hidden=\"true\">🔒</span> Away" : "<span aria-hidden=\"true\">🏠</span> Home";
      b.title = window.VL.armingText(s) + " · click to switch";
      b.setAttribute("aria-label", `${Nice(s.mode)}. Switch to ${s.mode === "away" ? "Home" : "Away"}`);
    }
    document.dispatchEvent(new CustomEvent("vl-arming", { detail: s }));
  }

  async function loadArming() {
    try {
      const r = await plainFetch("/api/arming");
      if (r.ok) drawArming(await r.json());
    } catch {}
  }

  window.VL.switchArming = async () => {
    const s = window.VL.arming;
    if (!s) return;
    const to = s.mode === "away" ? "home" : "away";
    const why = to === "away" ? "Alert rules marked \"only when Away\" will work." : "Alert rules marked \"only when Away\" will stop.";
    if (!confirm(`Switch to ${Nice(to)}?\n${why}`)) return;
    try {
      const r = await plainFetch("/api/arming", { method: "POST", headers: { "Content-Type": "application/json" },
                                                   body: JSON.stringify({ mode: to }) });
      if (r.ok) {
        drawArming(await r.json());
        if (to === "home") welcomeBack();
      } else alert("Could not switch: " + ((await r.json()).detail || r.status));
    } catch { alert("Could not switch: no answer from the app."); }
  };

  // After switching Away -> Home: a note under the header with what happened meanwhile and a link to the Summary
  // page (labeler/summary.py). Only for whoever switched; it never opens anything by itself; ✕ closes it.
  async function welcomeBack() {
    let s = null;
    try {
      const r = await plainFetch("/api/summary?period=away");
      if (r.ok) s = await r.json();
    } catch {}
    if (!s || s.none || !s.counts) return;
    const c = s.counts, parts = [];
    const add = (n, one, many) => { if (n) parts.push(`${n} ${n === 1 ? one : many}`); };
    add(c.people, "person", "people"); add(c.vehicles, "vehicle", "vehicles"); add(c.animals, "animal", "animals");
    add(c.alerts, "alert", "alerts"); add(c.scene_changes, "scene answer changed", "scene answers changed");
    document.getElementById("welcome")?.remove();
    const box = document.createElement("div");
    box.id = "welcome";
    box.className = "welcome";
    box.setAttribute("role", "status");
    box.innerHTML = `<span>Welcome back. ${parts.length ? "While you were away: " + parts.join(", ") + "." : "Nothing happened while you were away."}</span>
      <a href="/summary?period=away">What happened while you were away?</a><button aria-label="Close">✕</button>`;
    box.querySelector("button").addEventListener("click", () => box.remove());
    document.querySelector("header")?.after(box);
  }
  window.VL.welcomeBack = welcomeBack;

  function addArmingButton() {
    const header = document.querySelector("header");
    if (!header || document.getElementById("arming")) return;
    const b = document.createElement("button");
    b.id = "arming";
    b.className = "arming";
    b.textContent = "…";
    b.addEventListener("click", window.VL.switchArming);
    const nav = header.querySelector("nav");
    if (nav) nav.after(b); else header.append(b);
    loadArming();
    setInterval(loadArming, 30000);   // the schedule may switch it, or someone on another device
  }

  async function start() {
    let me = null;
    try {
      const r = await plainFetch("/api/me");
      if (r.ok) me = await r.json();
    } catch {}
    if (!me) return;
    const admin = me.role === "admin";
    addArmingButton();
    pollAlerts(admin);
    setInterval(() => pollAlerts(admin), 3000);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", show);
    document.addEventListener("DOMContentLoaded", start);
  } else {
    show();
    start();
  }
})();
