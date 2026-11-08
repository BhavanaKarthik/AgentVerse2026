# 🐾 IsaacSpot Fleet: Perception-Driven Edge Litter Collection & Hazard Quarantine

> **AgentVerse 2026 Hackathon @ UCL**  
> **Track:** Embodied AI & Autonomous Systems / Robotics  
> **Platform:** NVIDIA Isaac Sim 4.x + Dual Edge AI (PaliGemma-3B & Gemma-2-2B-IT)

An autonomous multi-agent fleet simulation featuring **5 Boston Dynamics Spot quadruped robots** collaborating to clean, classify, and isolate hazardous waste across a 44m × 44m public park environment.

---

## 🎯 Rubric Alignment

| Rubric Pillar | Score Focus | Implementation Highlights |
| :--- | :--- | :--- |
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
  │ • Model A: PaliGemma-3B-mix-224 (4-bit Close-Range VLM)      │
  │ • Model B: Gemma-2-2B-IT (4-bit Fleet Zone Strategist)      │
  │ • Budget Profiler: Jetson Orin NX (25W, VRAM ≤ 6.1 GB)      │
  └─────────────────────────────┬───────────────────────────────┘
                                │
             [Centralized Fleet Orchestrator v7.0]
    • Hysteresis-Stabilized Hungarian Target Matching
    • Multi-Hazard Prioritization (Lithium Battery Isolation)
    • String-Pulled A* Obstacle Avoidance & Visual Servoing
