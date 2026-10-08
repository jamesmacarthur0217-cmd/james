# Camera notice sign (template)

Put a sign where people can read it **before** they enter the camera's view: at the entrance, and next to a
kiosk camera at eye level. Large title, a camera symbol, short text. Fill in the [brackets]; pick the kiosk or
the security wording (or both, for a site with both kinds of camera). Check the wording against your local
rules: some countries require specific information or a standard sign. Not legal advice.

---

## Kiosk camera

> ### 📷 Camera with automatic analysis
>
> This kiosk has a camera. A computer estimates the **age group, gender and clothing** of people using it,
> to [understand who uses the kiosk / improve our service].
>
> - No face recognition, no names. Nothing is sent over the internet.
> - A picture and the estimates are kept for **[30] days**, then deleted.
> - Responsible: **[Organization]**. Questions or deletion requests: **[email / phone]**.
>
> More information: [link or "ask at the desk"]

## Security camera

> ### 📷 Video surveillance
>
> This area is monitored by cameras for **[the security of people and property]**. A computer describes
> people seen (clothing, bags, approximate age group) so recordings can be searched.
>
> - No face recognition, no names.
> - Kept for **[30] days**, then deleted automatically.
> - Responsible: **[Organization]**, **[address]**. Contact: **[email / phone]**.
>
> More information, including your rights: [link / QR code / "ask at reception"]

---

Notes for the person filling this in:

- The retention days must match `storage.retention_days` (or that camera's own `retention_days`) in
  `config.yaml`.
- "Nothing is sent over the internet" holds for pictures and estimates. The app does ask Hugging Face for model
  updates at startup (model names only); set `HF_HUB_OFFLINE=1` in `.env` to stop that.
- Give the longer explanation in [data-protection.md](data-protection.md), and link to it (a QR code to a page
  or a printed copy at the desk).
