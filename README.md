# Neural World Model & Predictive Robot Intelligence

> A research platform for predictive robot navigation: the robot **learns a model of its world from its own experience, imagines multiple possible futures, quantifies its uncertainty, and chooses actions based on those imagined futures** — moving beyond the classical *sense → plan → act* loop toward *sense → learn world → imagine → act*.

![Python](https://img.shields.io/badge/Python-3.10%20tested-blue)
![Research](https://img.shields.io/badge/Research-World%20Models%20%7C%20Model--Based%20Control-green)
![CI](https://github.com/HARSHAVARDHAN-SEKAR/Predictive-Robot-Intelligence-with-Neural-World-Models/actions/workflows/ci.yml/badge.svg)
![Simulation](https://img.shields.io/badge/Validation-2D%20simulation-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Overview

Classical navigation reacts to where obstacles **are**. This project's robot reacts to where it **predicts they will be**. A neural world model — trained purely from logged experience, with no hand-written physics — imagines the next 3 seconds of the world (its own dynamics *and* every moving obstacle) 10 times per second. A sampling model-predictive planner (MPPI) evaluates 220 candidate action plans inside that imagination and executes the safest one. A bootstrapped ensemble extends the model to **multiple plausible futures**, providing an empirical disagreement-based uncertainty signal and enabling risk-aware planning.

Core simulation experiments run on a laptop CPU without ROS or a GPU. Reproducing the full benchmark requires generating model checkpoints and running the training and evaluation stages in order; runtime depends on the machine. PyTorch (Transformer / Dreamer-RSSM / VAE) and ROS 2 components are included as the scaling path.

## Live demo

Blue robot, red moving obstacles, green goal — **dashed lines are the robot's imagination**, recomputed live at 10 Hz. It steers around where obstacles *will* be:

![Demo](assets/demo_neural_mpc.gif)

---

## Features

**Learned world model** — MLP robot-dynamics model (body-frame, MSE ≈ 2×10⁻⁴); rotation-invariant, history-conditioned obstacle motion predictor with autoregressive rollout; bounce-discontinuity filtering.

**Uncertainty** — 5-member bootstrapped ensemble (PETS-style); inter-member disagreement as a proxy for epistemic uncertainty; multi-future imagination.

**Planning & control** — MPPI imagination planner (Neural-MPC); CVaR risk-aware variant; DWA-style constant-control and reactive baselines; the analytic and learned MPPI variants share the same planning objective.

**Simulation** — self-contained 2D dynamic world: unicycle robot, curved noisy moving obstacles, wall bounces, 24-beam lidar ray-caster.

**Benchmarking** — ADE/FDE prediction metrics; success/collision/smoothness/energy planning metrics; distribution-shift generalization tests; stress tests; JSON results + auto-generated figures + PDF technical report.

**Scale-up code (PyTorch)** — VAE sensor encoder, Transformer world model (past 10 s → future 5 s), Dreamer-style RSSM with stochastic latents.

**Deployment** — ROS 2 node (10 Hz): subscribes `/odom`, `/obstacles`, `/goal_pose`; publishes `/cmd_vel` and `/predicted_obstacles` for RViz visualization of the robot's imagination. Gazebo and Isaac Sim integration remain future validation tasks.

**Engineering** — Dockerfile, GitHub Actions CI (native + in-container smoke tests of every model and all five planners), Makefile, seeded determinism.

---

## Architecture

```
        Sensors (pose, obstacle tracks)
                     │
        Collect robot experience (s, a, s′)
                     │
        ┌────────────▼────────────┐
        │   NEURAL WORLD MODEL    │
        │  robot dynamics (MLP)   │
        │  obstacle motion (MLP,  │
        │  rotation-invariant)    │
        │  ×5 bootstrap ensemble  │
        └────────────┬────────────┘
                     │  imagine 220 plans × 5 futures × 1.8 s
        ┌────────────▼────────────┐
        │  IMAGINATION PLANNER    │
        │  MPPI · CVaR risk cost  │
        │  collision · goal ·     │
        │  smoothness · energy    │
        └────────────┬────────────┘
                     │  best first action (v, ω)
              ROS 2 /cmd_vel → Robot
                     │
        new experience → retrain  ↺
```

Full diagram: `assets/architecture.png` · Long-term plan: [`docs/ROADMAP.md`](docs/ROADMAP.md)

---

## Research Results

### Test 1 — Dynamic obstacle prediction (3 s horizon, 40 episodes)

| Predictor | ADE in-dist | FDE in-dist | ADE unseen env | FDE unseen env |
|---|--:|--:|--:|--:|
| Constant velocity | 0.190 m | 0.522 m | 0.448 m | 1.231 m |
| **Neural world model** | **0.121 m** | **0.335 m** | **0.187 m** | **0.522 m** |
| Improvement | −36 % | −36 % | **−58 %** | **−58 %** |

### Test 2 — Planning (30 simulated episodes)

| Planner | Success | Collision | Energy |
|---|--:|--:|--:|
| Reactive | 66.7 % | 33.3 % | 19.6 |
| DWA | 100 % | 0 % | 24.5 |
| MPPI (physics + const-vel) | 93.3 % | 0 % | 17.5 |
| **Neural-MPC (learned model)** | **100 %** | **0 %** | **17.2** |

### Test 3 — Zero-shot generalization (9 faster, sharper-turning obstacles)

Neural-MPC: 93.3 % success, **0 % collisions** — the only high-success planner with zero collisions in both environments.

### Test 4 — Uncertainty versus prediction error

Ensemble disagreement vs realized prediction error over 720 tracks: **r = 0.49**. This is a moderate positive correlation, not a calibrated uncertainty estimate.

### Test 5 — Multi-future planning under stress (25 simulated episodes per planner)

The stress environment contains 12 fast, noisy moving obstacles.

| Planner | Imagined futures | Success | Collision | Energy |
|---|--:|--:|--:|--:|
| Neural-MPC (ensemble mean) | 5 | 68% | 24% | 17.20 |
| Risk-MPC (ensemble CVaR) | 5 | 76% | 12% | 18.84 |

Risk-MPC showed an 8-percentage-point higher success rate and a 12-percentage-point lower collision rate in this run, with higher measured energy. These are empirical results from a small synthetic-simulation benchmark, not a safety guarantee or proof of statistical superiority.

![Risk-aware stress benchmark](assets/risk_aware_stress.png)

### Test 6 — Occupancy-grid prediction & planning (Stage 1 upgrade)

Predicting a full occupancy grid instead of per-obstacle points, with ensemble disagreement naturally widening uncertain cells (no separate uncertainty scalar needed):

| Metric | Constant velocity | Neural ensemble |
|---|--:|--:|
| Occupancy IoU (0.5–1.5s) | 0.569 | **0.656** |

| Planner (stress env, 1.5s horizon) | Success | Collision |
|---|--:|--:|
| Neural-MPC (points) | 45 % | 35 % |
| Risk-MPC (CVaR, points) | 40 % | 20 % |
| **Occupancy-MPC (grid)** | **60 %** | 25 % |

![Occupancy prediction](assets/occupancy_prediction.png)
![Occupancy planning](assets/occupancy_planning.png)

Reproduce: `python scripts/run_stage1_occupancy.py 1`, then `2` (or the chunked `scripts/bench_stage1_chunk.py`), then `3`.

### Test 7 — Transformer world model & autoregressive drift (Stage 2)

A Transformer world model (past context of [state, action] tokens -> future latents) was trained on logged experience and compared head-to-head against the MLP ensemble. The result is an **honest negative finding**, and a more interesting one than a tuned win:

| Predictor | 1-step val MSE | 3 s rollout ADE |
|---|--:|--:|
| MLP ensemble (rotation-inv. velocity) | — | **0.10 m** |
| Transformer (absolute latent) | **0.07** | 2.39 m |

The transformer is *excellent* one step ahead but drifts badly over a multi-step rollout: it predicts the **absolute next latent**, so small per-step errors compound as it feeds its own predictions back in. The MLP wins because it predicts **velocity in a heading-aligned canonical frame** — a well-conditioned, rollout-stable target. This reproduces the central stability challenge of learned world models (the reason DreamerV3 uses latent regularization and video world models drift).

![Autoregressive drift](assets/autoregressive_drift.png)

Reproduce: train with `scripts/collect_sequence_data.py` + `scripts/train_world_transformer.py`, then `scripts/diagnose_drift.py` (figure) and `scripts/run_stage2_benchmark.py 1` + `3` (prediction table + merge). The fix — multi-step rollout loss / scheduled sampling / rotation-invariant target — is logged as Stage 2c in [`docs/ROADMAP.md`](docs/ROADMAP.md).

All figures: `assets/` · Raw metrics: `results/benchmark_results.json`

---

## Installation

```bash
git clone https://github.com/HARSHAVARDHAN-SEKAR/Predictive-Robot-Intelligence-with-Neural-World-Models.git
cd Predictive-Robot-Intelligence-with-Neural-World-Models
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # or: make install
```

## Run

```bash
make test        # fast smoke test (<60 s): env, models, all 5 planners
make bench       # full pipeline: data → train → Tests 1–3 → all figures (~5 min, CPU)
make upgrade     # Tests 4–5 (requires datasets/world_model.pkl from make bench)
make demo        # regenerate the animated demo GIF
make occupancy   # Stage 1: occupancy-grid prediction + planning (Test 6)
make docker      # build image and run container smoke tests
```

## Repository structure

```
neural_world_model_robotics/
├── src/nwm/                # env/ simulator · models/ world model + ensemble · planners/
├── scripts/                # run_pipeline · run_upgrade · demo video · diagrams
├── models_pytorch/         # VAE encoder · Transformer world model · Dreamer RSSM
├── ros2_ws/world_model_node/  # 10 Hz Neural-MPC ROS 2 node (+ RViz imagination topic)
├── tests/                  # CI smoke tests
├── assets/  results/  datasets/  docs/
├── Dockerfile  Makefile  .github/workflows/ci.yml
└── results/technical_report.pdf
```

## Roadmap

**Done:** learned dynamics · rotation-invariant prediction · imagination planning · ensemble uncertainty · risk-aware MPC · full benchmark suite · Docker/CI · technical report.
**Done (Stage 1):** occupancy-grid prediction and planning (Test 6) — see results above.
**Done (Stage 2):** trained a Transformer world model and measured it fairly — it wins one-step but loses to the MLP ensemble on multi-step rollout due to autoregressive drift (Test 7 above). **Next (Stage 2c):** rollout-stable training (scheduled sampling / rotation-invariant target / latent regularization), then ROS 2 + Gazebo validation and real-robot deployment. Detailed staged plan: [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Related repositories

- **Learning-Based-Adaptive-Navigation-Controller-for-UGV** — classical navigation: localization, planning, control.
- **Neural World Model Robotics** (this repo) — predictive world models and robot intelligence.
- **CRIP** — embodied AI with LLMs, semantic memory, high-level task planning.

## Author

**Harshavardhan Sekar** — M.Sc. Advanced Robotics, École Centrale de Nantes

## License

MIT
