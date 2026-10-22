# Experiment log

Every model trained in this project and what it taught us. All policies are trained from scratch
(no warm starts) with seed 0. Run folders, configs and reproduction commands are in
[results/README.md](../results/README.md).

## Evaluation protocol

All evaluation runs in CPU MuJoCo, i.e. a different simulator from MJX training.

| Metric | How it's measured | Script |
|---|---|---|
| **Tracking** | RMS velocity error (vx m/s / vy m/s / yaw-rate rad/s), mean over 9 commands, nominal model, 15 s episodes | `scripts/evaluate.py` |
| **Push survival** | single planar kick at zero command, 16 directions, 0–40 ms actuation/IMU latency; success at 0.6 / 0.8 m/s | `scripts/push_recovery.py --latency` |
| **Reaction** | time from a 0.6 m/s push from behind to the first foot lift-off (median over 16 pushes in brackets) | `scripts/diagnose.py` |
| **Drift** | mean \|heading change\| over 10 s walking straight at 0.15 m/s, 12 episodes with randomized physics, latency and noise; distance covered in brackets | `scripts/drift_distribution.py` |
| **Head** | RMS error between head joints and commanded head pose (neck pitch + head yaw sweep) | |

## All models

| Model | Training (difference from previous version) | Tracking vx / vy / wz | Push survival 0.6 / 0.8 | Reaction | Drift [distance] | Head | Verdict |
|---|---|---|---|---|---|---|---|
| **Upstream** `BEST_WALK_ONNX` | Open Duck author's policy (tested only) | 0.03 / 0.08 / 0.12 | **0.75 / 0.56** | 0.02 s (median 0.12) | 43° [0.82 m] | 0.156 (overshoots +0.22) | reference |
| Upstream `BEST_WALK_ONNX_2` | the author's second published policy (tested only) | 0.04 / 0.07 / 0.16 | 0.62 / 0.44 | 0.08 s (median 0.06) | – | – | as push-robust as our best |
| Standing, CPU SB3 | PPO on CPU (Stable-Baselines3), stopped at 6M of 10M steps | – | – | – | – | – | superseded by the GPU pipeline, not evaluated |
| Standing v0 | first GPU standing run; resets left the feet inside the floor | – | – | – | – | – | 15 % nominal falls on CPU → found the reset bug |
| Standing v1 | v0 with grounded resets; head ranges ±0.3–0.8 rad, pushes ≤ 0.5 m/s | n/a | 0.50 / 0.31 | 0.54 s | n/a | 0.182 | 0 % falls without pushes |
| **Standing v2** | + full upstream head ranges, head-tracking cost, pushes ≤ 0.8 m/s, posture terms paused 1.5 s after pushes | n/a | 0.50 / 0.50 | 0.46 s | n/a | **0.070** | best standing |
| Walking v1 | upstream reward without the lateral tolerance (`y_tol` = 0) | 0.06 / 0.06 / 0.21 | 0.56 / 0.50 | 0.24 s | 62° [0.25 m] | – | barely walks |
| Walking v2 | v1 + training pushes ≤ 0.5 m/s | 0.06 / 0.04 / 0.17 | 0.62 / 0.38 | 0.32 s | 52° [0.10 m] | – | barely walks |
| Walking v3 | upstream reward exactly, on our corrected env | 0.04 / 0.07 / **0.11** | 0.50 / 0.44 | 0.36 s | 45° [0.93 m] | – | walks; ignores \|cmd\| ≤ 0.1 m/s |
| Walking v4 | v3 + 2× velocity tracking (2.5 → 5) | **0.03** / 0.07 / 0.24 | 0.50 / 0.38 | 0.26 s | 57° [**1.28 m**] | 0.155 | deadzone fixed, long strides |
| **Walking v5** | v4 + head-tracking cost, yaw weight 6 → 12, stand-still cost paused 1.5 s after pushes | 0.03 / 0.08 / 0.14 | 0.62 / 0.31 | 0.18 s | **24°** [1.22 m] | 0.081 | best walking |
| Walking v6 | v5 + heading-error cost, joint init ×[0.5, 1.5] (grounded) | 0.05 / 0.06 / 0.20 | 0.50 / 0.19 | 0.40 s | 35° [0.32 m] | – | regressed |
| Headless v1 | 10-servo robot, v5 recipe, upstream home pose | stands | 0.75 / 0.50 | – | – | n/a | doesn't walk |
| Headless v2 | v1 with home pose leaned 0.1 rad forward | stands | 0.75 / 0.50 | – | – | n/a | doesn't walk |
| **Headless v3** | v2 + 2× tracking, 3× imitation | shuffles ~0.02 m/s | 0.69 / **0.62** | 0.42 s | – | n/a | balances, doesn't walk |
| Walking v5, seeds 1 / 2 | v5 recipe, different random seeds | 0.03 / 0.06 / 0.10, 0.03 / 0.07 / 0.10 | 0.50 / 0.31, 0.50 / 0.38 | – | 24°, 45° (biased −45°) | – | seed spread: tracking stable, drift bias is seed-dependent |
| Walking replica (u1) | v3 reward in an env reproducing upstream's training conditions (see below) | 0.03 / 0.08 / **0.04** | 0.50 / 0.31 | 0.36 s (median 0.22) | **15°** [0.99 m] | – | best turning and straightest walking; no deadzone; pushes unchanged |
| Walking replica, seed 1 | replica, different seed | 0.02 / 0.12 / 0.05 | 0.50 / 0.31 | – | 15° [0.87 m] | – | reproduces seed 0 |

## Key findings

What worked:
- **Grounded resets** removed a sim-to-sim artifact: an MJX-trained policy went from 15 % to 0 % nominal falls in CPU MuJoCo.
- **Doubling the velocity-tracking weight** (v4) fixed the "standing is safer than walking" deadzone for small commands and gave longer strides.
- **A head-tracking cost** (v5, standing v2) made the head follow commands about 2× more closely than upstream. Upstream overshoots the neck command by about 0.2 rad because it uses the head as a balance weight.
- **Restoring the yaw:velocity weight ratio** (v5) gave the straightest walking under randomized conditions: about half of upstream's drift, at 1.5× the distance.
- **A forward-leaning home pose** made the headless robot stand passively. The upstream pose tips it over backward.

What didn't work:
- **Removing upstream's lateral-velocity tolerance** (v1, v2) hurt walking: the tolerance absorbs natural gait sway.
- **Capping training pushes** (v2) didn't help push survival or walking.
- **A heading-error cost plus a wider initial pose** (v6) regressed everything. The robot can't observe its absolute heading, so the cost is noise to it.
- **Reusing the full robot's recipe and reference gaits on the headless robot** (headless v1–v3): it balances but won't walk.
- **Reward tuning so far hasn't touched the push gap.** v1–v6 and both standing versions survive 0.6 m/s pushes 50–62 % of the time, against upstream's 75 %.

What the measurements say about the two open weaknesses:
- **The push gap comes from one exceptional upstream policy, not from our setup.** Pooled over 96 trials (32 directions × 0.4/0.6/0.8 m/s, with latency), all six of our runs score 0.55–0.59 (± 0.05) and the author's second policy BEST_WALK_ONNX_2 scores 0.59. Only BEST_WALK reaches 0.76 ± 0.04, about 3.5 standard errors above everything else, including its sibling. Seeds barely move our score. So the gap isn't the reward, the environment or luck of our seed. It's BEST_WALK itself: a selected outlier, or a run with an unpublished config. The 16-trial benchmark (± 0.12) was too coarse to show this.
- **Reaction timing is not the gap.** Over 16 pushes, our policies lift a foot as fast as upstream's (median 0.04–0.08 s vs 0.06–0.12 s). My first measurement (0.02 s vs 0.36 s) was one push direction and over-read. What differs is **step vigor**: on a push from behind, upstream's recovery step reaches 3.6–3.9 rad/s and 0.27 m of foot travel, ours 1.8–2.9 rad/s and 0.20–0.23 m.
- **The push gap is not explained by** the reward weights (v3 uses upstream's), nominal-model tuning (upstream still leads with randomized physics on every trial), evaluation latency, or upstream's training environment (the replica u1 reproduces it and pushes don't improve).
- **Seed-to-seed variance:** tracking and push survival are stable across seeds. Heading drift isn't: v5 seed 2 has a consistent −45° bias per 10 s while seeds 0 and 1 drift 24°. A seed-dependent bias is one more reason to train several seeds and select.
- **Drift is mostly random, not a fixed bias.** A single nominal-model episode is one sample from a wide spread: over randomized episodes upstream drifts 43° ± 51° per 10 s, v5 24° ± 25° and the replica 15° ± 18°. Policy mirror asymmetry doesn't predict drift.
- **Our harder environment caused the walking deadzone.** With the identical reward, v3 (our env) ignores 0.1 m/s commands and the replica (upstream-like env) doesn't. The extra randomization and latency make walking riskier, so PPO prefers to stand.

## Replica experiment (u1): reproducing upstream's training environment

Goal: same reward as v3, with everything else as close to upstream's Open_Duck_Playground as
our code allows (`configs/train_walking_upstream_replica.yaml`):

| Setting | Our default | Replica (= upstream) |
|---|---|---|
| domain randomization | full set (incl. calibration offsets, trunk mass/CoM, damping, torque limits) | `upstream_like`: friction, body masses, frictionloss, armature, kp only |
| IMU latency | 0–40 ms | none |
| initial joints | ×[0.8, 1.2], feet placed on the ground | ×[0.5, 1.5], base height fixed (feet may start in the floor) |
| head commands in training | added to head targets | not applied (upstream sim) |
| episode reset | full re-randomized reset | cached first state, info carried over, 10 full resets per eval |
| head command sampling | independent of velocity command | zeroed together with velocity 10 % of the time |
| action-history convention | robot runtime's | unchanged: it's part of the deployment contract |

Result: same push survival as our other policies (0.50 / 0.31), but the best turning (yaw-rate
error 0.04 vs upstream 0.12), the straightest walking (15° ± 18° per 10 s) and no deadzone.
Upstream's training environment therefore isn't what makes BEST_WALK more push-robust; it does
explain the deadzone. Next: more seeds, to measure run-to-run variance directly.

## Seed study

Pooled push survival, 96 trials per policy (32 directions × 0.4 / 0.6 / 0.8 m/s, 0–40 ms latency),
from `results/*/push_pooled.json`:

| policy | pooled survival | 0.4 | 0.6 | 0.8 |
|---|---|---|---|---|
| upstream BEST_WALK | **0.76 ± 0.04** | 0.94 | 0.72 | 0.62 |
| upstream BEST_WALK_ONNX_2 | 0.59 ± 0.05 | 0.81 | 0.66 | 0.31 |
| v3 | 0.57 ± 0.05 | 0.81 | 0.53 | 0.38 |
| v5 seed 0 / 1 / 2 | 0.56 / 0.58 / 0.55 | 0.72 / 0.94 / 0.84 | 0.59 / 0.50 / 0.50 | 0.38 / 0.31 / 0.31 |
| replica seed 0 / 1 | 0.59 / 0.56 | 0.91 / 0.84 | 0.50 / 0.50 | 0.38 / 0.34 |

Conclusion: with the configurations tried so far, push robustness is about 0.57 regardless of reward,
environment or seed, the same as upstream's second policy. Beating it needs a method aimed at push
recovery rather than more reward re-weighting. Candidates, in order:
1. **Bigger, more frequent pushes as a curriculum**, so recovery steps are trained explicitly (the
   current random pushes up to 1 m/s every 5–10 s are mostly survivable without stepping).
2. **Reward larger, faster recovery steps** after a push (swing-foot placement in the push
   direction), since step vigor is the measured difference from BEST_WALK.
3. **Train N seeds and select the best by pooled push survival**, which is plausibly how BEST_WALK
   was obtained.
