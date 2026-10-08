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

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", show);
  else show();
})();
