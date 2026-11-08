

 🐾 EcoSwarm: Autonomous Quadruped Swarm for Litter Collection & Hazard Quarantine

> **AgentVerse 2026 Hackathon @ UCL**
> **Track:** Embodied AI & Autonomous Systems / Robotics
> **Platform:** NVIDIA Isaac Sim 4.x + Dual Edge AI (PaliGemma-3B & Gemma-2-2B-IT)

An autonomous multi-agent fleet simulation featuring **5 Boston Dynamics Spot quadruped robots** collaborating to clean, classify, and isolate hazardous waste across a 44m × 44m public park environment.

## 🎯 Rubric Alignment

| Rubric Pillar | Score Focus | Implementation Highlights |
| --- | --- | --- |
| **Perception-Driven Action** | **25%** | Line-of-sight raycasting & occlusion estimation against trees, benches, and boulders. Closed-loop visual servoing with continuous bearing error correction (`vy`, `wz`). |
| **Adaptation to Change** | **25%** | Live 3D A* trajectory ribbons (`UsdGeom.BasisCurves`) rendered in the viewport. Dynamic target drift (wind gusts/pedestrian kicks) triggers instant mid-route re-planning. |
| **Operational Autonomy** | **15%** | Modeled within an onboard **NVIDIA Jetson Orin NX 16GB (25W)** budget: VRAM cap $\le 6.1\text{ GB}$, latency budget $\le 15.0\text{ ms}$. Uncertainty orbit protocol (42° repositioning) when confidence $< 0.78$. |
| **Relevance & Impact** | **20%** | Solves two real-world failure modes live on screen: (1) End-effector grasp slip detection and re-acquisition; (2) Swollen lithium-ion battery quarantine with amber roof beacon and 4.1m safety halo. |
| **Technical Quality** | **15%** | Hysteresis-stabilized ($1.85\text{m}$) Capacitated Hungarian algorithm, line-of-sight string-pulled A* waypoint smoothing, and full 2-link analytic leg IK gait generation. |

---

## 🏗️ System Architecture

```text
                    [Sensory Perception Tier]
        ┌───────────────────────┼───────────────────────┐
  74° Forward RGB        Omni 360° Cam Rig         Rotating 3D LiDAR
 (Inspection Frustum)   (Perimeter Awareness)    (Occlusion Raycasting)
        └───────────────────────┬───────────────────────┘
                                │
                 [Dual-Model Edge AI Bridge (Local)]
  ┌─────────────────────────────────────────────────────────────┐
  │ • Model A: PaliGemma-3B-mix-224 (4-bit Close-Range VLM)     │
  │ • Model B: Gemma-2-2B-IT (4-bit Fleet Zone Strategist)     │
  │ • Budget Profiler: Jetson Orin NX (25W, VRAM ≤ 6.1 GB)     │
  └─────────────────────────────┬───────────────────────────────┘
                                │
               [Centralized Fleet Orchestrator v7.0]
    • Hysteresis-Stabilized Hungarian Target Matching
    • Multi-Hazard Prioritization (Lithium Battery Isolation)
    • String-Pulled A* Obstacle Avoidance & Visual Servoing

```

## 🚦 Interactive Hotkeys (Isaac Sim)

When running inside the Isaac Sim viewport, use the following keys to demonstrate dynamic adaptation:

* **`[P]`** — **Toggle Fleet Mode:** Autonomous Clean-up vs. Hold / Idle.
* **`[C]`** — **Trigger Mid-Sequence Scene Perturbation:** Displaces an active target or crosses a pedestrian into a robot's corridor, showing live A* trajectory re-planning.
* **`[F]`** — **Inject Critical Failure Mode:** Simulates an end-effector grasp slip and flags a thermal-risk battery cell for isolated hazmat handling.
* **`[R]`** — **Reset Fleet:** Resets all 5 Spot quadrupeds to their charging pads.

---

## ⚖️ Ethics, Safety & Responsible AI

* **Pedestrian Safety Halo:**
Under nominal operations, robots enforce an active collision envelope of $2.85\text{m}$. Upon identifying an incendiary or thermal hazard (e.g., punctured battery cells), the safety envelope expands to $4.10\text{m}$, and the robot’s roof beacon switches to high-visibility amber.
* **Uncertainty-Aware Caution (No Blind Grasping):**
If ambient lighting, occlusion, or viewing distance drops VLM classification confidence below $78\%$, the robot does not guess. It transitions to `UNCERTAINTY_ORBIT_REINSPECT`, repositioning $42^\circ$ to inspect from an alternate viewpoint before interacting.
* **Privacy by Design:**
Onboard camera streams run entirely on simulated local edge hardware. No human facial data or identifying park imagery is transmitted to cloud APIs.

---

## 🚀 Quickstart Guide

### Prerequisites

* NVIDIA Omniverse / Isaac Sim 4.0+
* Python 3.10+
* **Optional:** CUDA-compatible GPU (RTX 3080 / 4080 or better recommended for local transformers inference)

### Running in Isaac Sim

1. Launch NVIDIA Isaac Sim.
2. Open the script console via **Window -> Script Editor**.
3. Paste the contents of `scripts/run_simulation.py` or load the file.
4. Click **Run** (`Ctrl+Enter`), then press **Play** on the left toolbar.
