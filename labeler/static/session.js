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

  function showAlert(e, admin) {
    const t = document.createElement("div");
    const picture = e.crop ? `<img src="/crops/${encodeURIComponent(e.crop)}" alt="">` : "";
    const time = new Date(e.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    if (e.fall) {
      t.className = "toast fall";
      t.setAttribute("role", "alertdialog");
      t.innerHTML = `${picture}<div><b>FALL</b> · ${escHtml(e.camera)} · ${time}<br>Someone fell and has not got up
        (person #${e.track_id}).${admin ? ' <a href="/alerts">See the clip</a>' : ""}<br>
        <small class="sound-off">Click anywhere on this page to hear the alarm.</small><br>
        <button class="primary">OK</button></div>`;
      t.querySelector("button").addEventListener("click", () => { t.remove(); updateAlarm(); });
    } else {
      t.className = "toast";
      t.innerHTML = `${picture}<div><b>Alert</b> · ${escHtml(e.camera)}<br>${escHtml(e.message)}<br>
        <small>person #${e.track_id}</small></div>`;
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

  async function start() {
    let me = null;
    try {
      const r = await plainFetch("/api/me");
      if (r.ok) me = await r.json();
    } catch {}
    if (!me) return;
    const admin = me.role === "admin";
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
