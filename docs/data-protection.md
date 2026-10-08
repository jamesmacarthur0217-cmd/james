# Data protection summary (template)

Fill in the parts in [brackets] and remove what does not apply to your site. This describes what VisionLabeler
does with its settings as shipped; if you change `config.yaml`, update this text. It is not legal advice:
camera and data-protection rules differ by country (for example the GDPR in the EU and UK), so have it checked
for your situation, including whether you need a data protection impact assessment.

---

**Who is responsible:** [Organization name], [address]. Contact for questions and requests: [email / phone].

**Why we use it:** [e.g. to understand how many people use the kiosk and which groups / for the security of
the premises]. The system does not identify people by name and does not use face recognition.

## What is recorded

For each person seen by a camera:

| Data | Details |
|---|---|
| When and where | camera name, time first and last seen |
| One picture | the clearest frame of the person (whole body), kept on this computer |
| Estimated description | apparent age range or age group, apparent gender, clothing type and color, headwear, glasses, face mask, carried items (bag, phone, …). These are automatic estimates and can be wrong |
| Appearance vector | a list of numbers describing the person's look (mostly clothing), used to search ("man in a red jacket") and to find the same-looking person on other visits or cameras. It is not a face template |
| Corrections | labels a staff member checked or corrected, with their user name |

Not recorded: sound, names, ID numbers, face close-ups (off by default: `storage.save_face_crops`). Faces in the
saved pictures can also be blurred for everyone (`storage.blur_faces`): [on / off at this site].

Also recorded, about staff: an **audit log** of who logged in, watched the live video, searched, exported,
deleted or changed settings, with time and network address.

## Where it is processed

Everything runs on one computer at [location]: the camera picture, the analysis (local AI models) and the
database. No pictures, descriptions or search texts are sent to any online service. When the app starts, it
asks Hugging Face (a model library) whether newer versions of two of its models exist; only the model names
are sent. [Optional: we turned this off with `HF_HUB_OFFLINE=1` in `.env`.]

## How long it is kept

| Data | Kept |
|---|---|
| People (picture, description, appearance vector) | [30] days ([per camera: e.g. entrance 7 days]), then deleted automatically (checked every hour) |
| Audit log | [365] days |
| Backups of the database | the last 7 backups, at most one a day, so up to about [7–14] days after deletion |

## Who can see it

| Who | Can |
|---|---|
| Someone at the computer itself | everything (it is always administrator) — keep the computer locked: [where it is, who has access] |
| Administrator accounts ([names / roles]) | everything: live video, search, reports and exports, corrections, alert rules, deleting |
| Viewer accounts ([names / roles]) | live video and search only, [with faces blurred] (`web.blur_faces_for_viewers`) |

Other devices must log in; five wrong passwords block a device for 5 minutes. With `web.https: true` the
connection from other devices is encrypted. Nobody outside [Organization] has access.

## Your rights

You can ask what we hold about you, ask us to delete it, or object, by contacting [contact]. Because we do not
know your name, please tell us when and where you passed the camera and what you wore, so we can find your
records. We delete them with "Delete selected" (pictures, descriptions, history and alerts are removed; the
deletion itself is logged). Backups still contain them until they are rotated out (see above).
[You can also complain to the data protection authority: name, website.]

## Security measures

- Access limited to named accounts with roles; passwords stored only as salted hashes; audit log.
- Encrypted connections for other devices (HTTPS), when enabled.
- Faces blurred for viewer accounts.
- Automatic deletion after the retention time.
- **Encrypt the computer's disk** with BitLocker (Windows: Settings › Privacy & security › Device encryption,
  or Control Panel › BitLocker Drive Encryption), so the pictures, database and backups are unreadable if the
  computer or its disk is stolen: [enabled on (date)].
- [Physical security of the computer and cameras, who installs updates.]
