# 🚧 Safeguard AI

**Redefining construction site safety through Physical AI and smart surveillance.**

Safeguard transforms passive CCTV cameras into intelligent, proactive safety wardens. Built for the **Physical AI** track (Kaya AI Hackathon), it utilizes real-time computer vision to monitor construction sites, analyze worker interactions with their physical environment, and issue instant alerts to prevent accidents before they happen.

---

## 🎯 Problem & Inspiration

Construction sites are inherently dangerous environments. Despite strict safety protocols, human oversight is limited. Accidents involving heavy machinery, falls, and unauthorized access to hazardous zones are far too common. Safeguard acts as an automated, always-on safety officer that understands the spatial context of a construction site.

---

## 🧠 Physical AI Features

### 🏗️ Danger Zone Intrusion
Flags workers who linger near live machinery, crane swing radii, or excavation pits for too long. Uses a custom ID Relinking algorithm to maintain dwell-time continuity even through physical occlusions.

### 🎯 Heavy Machinery & Personnel Detection
Powered by YOLOv8, Safeguard identifies excavators, trucks, and site personnel in real-time, even in cluttered environments.

### 🤕 Fall & Posture Detection
Triggers emergency alerts if a worker falls from scaffolding or collapses on site, analyzing skeletal posture continuously to detect sudden vertical drops.

### 🔍 Real-Time Alerts
Sends instant SMS or WhatsApp notifications to site managers when critical safety thresholds are breached.

### 🌐 Command Dashboard
A sleek, real-time React dashboard featuring a modern "Trust Indigo & Cyber Cyan" design system for monitoring multiple camera feeds and safety metrics simultaneously.

---

## ⚙️ Tech Stack

- **Computer Vision:** YOLOv8, OpenCV, MediaPipe
- **Tracking & Logic:** BoT-SORT, Custom ID Relinker (Spatial-Temporal tracking fallback)
- **Backend:** Flask, Python 3.13
- **Frontend:** React, Vite, Vanilla CSS
- **Notifications:** Twilio, PyWhatKit

---

## 👷 Target Users

- **Site Managers & Foremen:** Maintain 24/7 safety oversight without manual patrolling.
- **Safety Officers (HSE):** Automate the logging of safety violations and restricted zone intrusions.
- **Construction Firms:** Lower insurance premiums and strive toward zero-harm work environments.

---

## 🏗️ How It Works

Safeguard bridges the gap between digital models and physical safety:
1. **Multi-threaded Pipeline:** Ensures high-FPS processing of heavy AI models alongside video I/O.
2. **YOLOv8 & Tracking:** Pinpoints human coordinates relative to mapped danger polygons using BoT-SORT.
3. **ID Relinker:** A robust post-processing layer that stitches broken tracker IDs together based on spatial and temporal proximity, preventing false resets of danger-zone dwell timers during occlusions.
4. **Posture Analysis:** MediaPipe tracks worker skeletons to detect sudden drops in vertical height indicative of a fall.

---

## 🧩 Overcoming Challenges

- **Tracker ID Instability (The Occlusion Problem):** In crowded construction sites or low-light conditions, workers constantly occlude each other, causing object trackers to lose IDs. We built a custom `IDRelinker` module that protects the critical "dwell-timer" by re-associating lost tracks within a 100px / 2.0s spatial-temporal window.
- **Real-World Clutter:** Differentiating workers from heavy machinery in chaotic, dusty construction footage required careful confidence thresholding and dedicated zone definitions.

---

> *"A smarter site isn’t just productive — it’s physically aware."*
