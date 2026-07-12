<div align="center">

  <h1>🚧 Safeguard AI 🚧</h1>
  <h3><em>Redefining construction site safety through Physical AI and smart surveillance.</em></h3>

</div>

<!-- Terminal Intro Animation -->
<div align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Outfit&size=18&duration=2000&pause=800&color=00FFFF&center=true&width=600&lines=Initializing+Safeguard+AI+Module...;Booting+Site+Surveillance...;Activating+Danger+Zone+Detection...;Enhancing+Construction+Safety...;System+Ready!" alt="Terminal Animation">
</div>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="Divider">

## 🎯 Problem & Inspiration

<table>
<tr>
<td>

Construction sites are inherently dangerous environments. Despite strict safety protocols, human oversight is limited, and accidents involving heavy machinery, falls, and unauthorized access to hazardous zones are far too common.

Built for the **Kaya AI "Physical AI" Track**, **Safeguard** transforms passive CCTV cameras into intelligent, proactive safety wardens. By utilizing real-time computer vision, Safeguard monitors construction sites, analyzes worker interactions with their physical environment, and issues instant alerts to prevent accidents before they happen.

</td>
<td width="40%">
<img src="images/surveillance-hero-bg.jpg" width="100%">
</td>
</tr>
</table>

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=100&section=divider">

## 🧠 What It Does (Physical AI Features)

<div align="center">
  <table>
    <tr>
      <td align="center"><h3>🏗️</h3><h4>Danger Zone Intrusion</h4><p>Flags workers lingering near live machinery, crane swing radii, or excavation pits.</p></td>
      <td align="center"><h3>🎯</h3><h4>YOLOv8 Detection</h4><p>Identifies heavy machinery, excavators, and site personnel in real-time.</p></td>
    </tr>
    <tr>
      <td align="center"><h3>🤕</h3><h4>Fall & Posture Detection</h4><p>Triggers emergency alerts if a worker falls from scaffolding or collapses on site.</p></td>
      <td align="center"><h3>🔍</h3><h4>Real-Time Alerts</h4><p>Instant SMS/WhatsApp notifications to site managers via Twilio & PyWhatKit.</p></td>
    </tr>
    <tr>
      <td align="center"><h3>👁️</h3><h4>Worker Tracking</h4><p>Assigns unique IDs to track movement through active hazard zones.</p></td>
      <td align="center"><h3>🌐</h3><h4>Command Dashboard</h4><p>A sleek, Cyber Cyan & Indigo React dashboard for monitoring multiple feeds.</p></td>
    </tr>
  </table>
</div>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## ⚙️ Tech Stack

<div align="center">

🐍 Python • 📹 OpenCV • 🧠 YOLOv8 • 🖐️ MediaPipe • 🌐 Flask Backend  
⚛️ React & Vite • 🌊 Tailwind CSS • 💬 Twilio • 🔥 Firebase • 🐘 PostgreSQL

</div>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:00e5ff,100:0044ff&height=100&section=divider&text=Target%20Users&fontColor=ffffff&fontSize=28&animation=twinkling">

## 👷 Target Users

- 🏗️ **Site Managers & Foremen** – Looking to maintain 24/7 safety oversight without manual patrolling.
- 🦺 **Safety Officers (HSE)** – Needing automated logging of safety violations (e.g., restricted zone intrusions).
- 🏢 **Construction Firms** – Aiming to lower insurance premiums and achieve zero-harm work environments.

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 🏗️ How We Built It

<table>
<tr>
<td>

Safeguard bridges the gap between digital models and physical safety:

- 🧵 **Multi-threaded Pipeline**: Ensures high-FPS processing of heavy AI models alongside video I/O.
- 🧠 **YOLOv8 & Centroid Tracking**: Pinpoints human coordinates relative to mapped danger polygons.
- 🖐️ **MediaPipe integration**: Tracks worker posture for critical health emergencies (falls/collapses).
- 🌐 **Modern React Frontend**: HMR-enabled Vite dashboards styled with a premium Indigo & Cyber Cyan glassmorphism aesthetic.

</td>
<td width="40%">
<img src="https://media.giphy.com/media/l2JHRhAtnJSDNJ2py/giphy.gif" width="100%">
</td>
</tr>
</table>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="Divider">

## 🧠 What We Learned

- Designing robust Physical AI pipelines that map 2D video coordinates to physical danger zones.
- Overcoming legacy dependency issues (e.g., dynamically handling legacy MediaPipe APIs on Python 3.13).
- Crafting premium, hardware-accelerated frontend UI architectures using React and Tailwind CSS.
- Reducing alert fatigue by tuning dwell-time thresholds for danger zones.

---

## 🧩 Challenges Faced

- ⚠️ **Dynamic Tracking:** Handling bounding-box flickering where the tracker would assign new IDs to workers, prematurely resetting their dwell-time in danger zones.
- 🔍 **Real-World Clutter:** Differentiating workers from heavy machinery in chaotic, dusty construction footage.
- 💾 **System Overhead:** Managing the memory footprint of YOLOv8 alongside pose estimation on standard hardware.

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="Divider">

> 🔐 *“A smarter site isn’t just productive — it’s physically aware.”*

<div align="center">
  <sub>Built for the Kaya AI Hackathon</sub>
</div>
