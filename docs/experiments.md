# Experiment log

Every model trained in this project and what it taught us. All policies are trained from scratch
with seed 0, except v7f (fine-tuned from v5) and the seed-study runs. Run folders, configs and reproduction commands are in
[results/README.md](../results/README.md).

## Evaluation protocol

All evaluation runs in CPU MuJoCo, i.e. a different simulator from MJX training.

| Metric | How it's measured | Script |
|---|---|---|
| **Tracking** | RMS velocity error (vx m/s / vy m/s / yaw-rate rad/s), mean over 9 commands, nominal model, 15 s episodes | `scripts/evaluate.py` |
| **Push survival** | single planar kick at zero command, 16 directions, 0–40 ms actuation/IMU latency; success at 0.6 / 0.8 m/s | `scripts/push_recovery.py --latency` |
| **Reaction** | time from a 0.6 m/s push from behind to the first foot lift-off (median over 16 pushes in brackets) | `scripts/diagnose.py` |
| **Pooled push** | 64 directions × 0.4 / 0.6 / 0.8 m/s at zero command (192 trials, 384 for finalists); *nominal* = latency only, *realistic* = + randomized physics and noise | `scripts/push_benchmark.py` |
| **Realistic tracking** | as Tracking, with randomized physics, sensor noise and 0–40 ms latency, no pushes; 6 episodes per command | `scripts/evaluate.py --realistic` |
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
| Walking replica (u1) | v3 reward in an env reproducing upstream's training conditions (see below) | 0.03 / 0.08 / **0.04** | 0.50 / 0.31 | 0.36 s (median 0.22) | **15°** [0.99 m] | – | best on the nominal model and in straight-line drift, but **falls 15 % under realistic conditions** (ours: 0–2 %); pushes unchanged |
| Walking replica, seed 1 | replica, different seed | 0.02 / 0.12 / 0.05 | 0.50 / 0.31 | – | 15° [0.87 m] | – | reproduces seed 0 |
| Walking v7a | v5 + pushes every 2–5 s, up to 1.2 m/s | 0.03 / 0.08 / 0.17 | 0.66 / 0.17 | – | – | – | pooled push unchanged |
| **Walking v7b** | v5 + action-rate, torque and imitation costs also paused 1.5 s after pushes | 0.03 / 0.07 / 0.14 | 0.58 / 0.31 | – | – | – | **pooled push 0.64 / 0.56 (v5 0.56 / 0.49); best push-robust walker** |
| Walking v7c | v5 + discount 0.99 | 0.07 / 0.05 / 0.37 | 0.52 / 0.33 | – | – | – | stops walking |
| Walking v7d | v5 + fall penalty, no reward clip | 0.04 / 0.06 / 0.14 | 0.61 / 0.34 | – | – | – | no change; deadzone back |
| Walking v7e | v5 + post-push velocity reward | 0.02 / 0.11 / 0.15 | 0.45 / 0.00 | – | – | – | worse |
| Walking v7f | v5 fine-tuned 50M steps with v7a's pushes | 0.03 / 0.08 / 0.15 | 0.56 / 0.31 | – | – | – | no change |
| Walking v7g | v5 + 50 % zero commands | 0.03 / 0.07 / 0.27 | 0.52 / 0.39 | – | – | – | realistic push 0.56, turning worse |
| Walking v8 | v7b + v7g | 0.05 / 0.04 / 0.15 | 0.58 / 0.39 | – | – | – | stops walking; push no better than v7b |

## Key findings

What worked:
- **Grounded resets** removed a sim-to-sim artifact: an MJX-trained policy went from 15 % to 0 % nominal falls in CPU MuJoCo.
- **Doubling the velocity-tracking weight** (v4) fixed the "standing is safer than walking" deadzone for small commands and gave longer strides.
- **A head-tracking cost** (v5, standing v2) made the head follow commands about 2× more closely than upstream. Upstream overshoots the neck command by about 0.2 rad because it uses the head as a balance weight.
- **Restoring the yaw:velocity weight ratio** (v5) gave the straightest walking under randomized conditions: about half of upstream's drift, at 1.5× the distance.
- **Pausing the movement costs (action rate, torques, imitation) after a push** (v7b) raised pooled push survival from 0.56 to 0.64 nominal and 0.49 to 0.56 realistic (~2 SE), all of it at 0.4 m/s, with walking intact.
- **A forward-leaning home pose** made the headless robot stand passively. The upstream pose tips it over backward.

What didn't work:
- **Removing upstream's lateral-velocity tolerance** (v1, v2) hurt walking: the tolerance absorbs natural gait sway.
- **Capping training pushes** (v2) didn't help push survival or walking.
- **A heading-error cost plus a wider initial pose** (v6) regressed everything. The robot can't observe its absolute heading, so the cost is noise to it.
- **Reusing the full robot's recipe and reference gaits on the headless robot** (headless v1–v3): it balances but won't walk.
- **Push-recovery methods other than v7b** (push series, below): more pushes (v7a), a push-heavy fine-tune (v7f), a fall penalty (v7d) and a longer horizon (v7c) didn't move pooled push survival; a shaped post-push reward (v7e) made it worse; more standing time (v7g, v8) traded walking for at most a small gain.
- **Reward tuning so far hasn't touched the push gap.** v1–v6 and both standing versions survive 0.6 m/s pushes 50–62 % of the time, against upstream's 75 %.

What the measurements say about the two open weaknesses:
- **The push gap comes from one exceptional upstream policy, not from our setup.** Pooled over 96 trials (32 directions × 0.4/0.6/0.8 m/s, with latency), all six of our runs score 0.55–0.59 (± 0.05) and the author's second policy BEST_WALK_ONNX_2 scores 0.59. Only BEST_WALK reaches 0.76 ± 0.04, about 3.5 standard errors above everything else, including its sibling. Seeds barely move our score. So the gap isn't the reward, the environment or luck of our seed. It's BEST_WALK itself: a selected outlier, or a run with an unpublished config. The 16-trial benchmark (± 0.12) was too coarse to show this.
- **Reaction timing is not the gap.** Over 16 pushes, our policies lift a foot as fast as upstream's (median 0.04–0.08 s vs 0.06–0.12 s). My first measurement (0.02 s vs 0.36 s) was one push direction and over-read. What differs is **step vigor**: on a push from behind, upstream's recovery step reaches 3.6–3.9 rad/s and 0.27 m of foot travel, ours 1.8–2.9 rad/s and 0.20–0.23 m.
- **The push gap is not explained by** the reward weights (v3 uses upstream's), nominal-model tuning (upstream still leads with randomized physics on every trial), evaluation latency, or upstream's training environment (the replica u1 reproduces it and pushes don't improve).
- **Seed-to-seed variance:** tracking and push survival are stable across seeds. Heading drift isn't: v5 seed 2 has a consistent −45° bias per 10 s while seeds 0 and 1 drift 24°. A seed-dependent bias is one more reason to train several seeds and select.
- **Drift is mostly random, not a fixed bias.** A single nominal-model episode is one sample from a wide spread: over randomized episodes upstream drifts 43° ± 51° per 10 s, v5 24° ± 25° and the replica 15° ± 18°. Policy mirror asymmetry doesn't predict drift.
- **Our harder environment caused the walking deadzone, and it also buys robustness.** With the identical reward, v3 (our env) ignores 0.1 m/s commands and the replica (upstream-like env) doesn't. The extra randomization and latency make walking riskier, so PPO prefers to stand. But under realistic conditions the replica falls 7–15 % of the time against v3's 0 %. No single imperfection causes it; their combination does.

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

Result on the nominal model: same push survival as our other policies (0.50 / 0.31), but the best
turning (yaw-rate error 0.04 vs upstream 0.12), the straightest walking (15° ± 18° per 10 s) and no
deadzone. Upstream's training environment therefore isn't what makes BEST_WALK more push-robust;
it does explain the deadzone.

**Under realistic conditions the replica is the most fragile policy** (`scripts/evaluate.py --realistic`;
randomized physics, sensor noise, 0–40 ms latency, no pushes; mean over 9 commands):

| | upstream | v3 | v5 | replica s0 | replica s1 |
|---|---|---|---|---|---|
| falls | 2 % | **0 %** | 2 % | 15 % | 7 % |
| vx error | 0.04 | 0.04 | 0.05 | 0.11 | 0.12 |
| yaw-rate error | 0.25 | 0.17 | 0.19 | 0.16 | 0.15 |

It falls in up to half of the turn-in-place episodes. One factor at a time (`scripts/realism_ablation.py`;
turn ±0.5 rad/s and forward 0.15 m/s, 30 episodes each), latency, sensor noise, calibration offsets,
trunk mass/CoM shifts, and damping + torque-limit changes each cause 0 falls. Only their
**combination** does: 20–30 % for the replica, 0 % for v3 and upstream. So the trade-off is real:
the easier environment buys smoother nominal walking and loses robustness to compounded
real-world imperfections, which is exactly what our harder environment was built for.
Upstream's BEST_WALK has both, one more sign that it's an exceptional run.

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

## Push series (v7): methods aimed at push recovery

Benchmark (`scripts/push_benchmark.py`, `*/push_bench.json`): standing robot, kicked at t = 3 s
in 64 directions at 0.4, 0.6 and 0.8 m/s (192 trials, ± 0.03–0.04); success = no fall in 3 s.
**Nominal** = nominal physics + 0–40 ms latency (the seed-study protocol). **Realistic** = also
randomized physics and sensor noise, re-sampled per trial.

Baselines, nominal / realistic:

| policy | nominal | realistic |
|---|---|---|
| upstream BEST_WALK (target) | **0.74** | **0.66** |
| upstream BEST_WALK_ONNX_2 | 0.64 | 0.55 |
| v3 | 0.57 | 0.54 |
| v4 | 0.60 | 0.55 |
| v5 seed 0 / 1 / 2 | 0.55 / 0.57 / 0.55 | 0.51 / 0.51 / 0.52 |
| v6 | 0.49 | 0.48 |
| replica seed 0 / 1 | 0.57 / 0.55 | 0.54 / 0.56 |

Where we lose: by direction (0.6 and 0.8 m/s, 4 seeds per direction), v3 and v5 fail
**every push from behind within ±45° of forward**, even at 0.6 m/s, while BEST_WALK survives most
of them. Both fail pushes from the front. On a 0.6 m/s push from behind, v5's hip-pitch joints move
only 0.06 rad before it falls (0.5 s); BEST_WALK swings them 0.17–0.31 rad and recovers. Knees sit
near their soft limit in both, so joint limits don't explain it. Push survival
of v3 and v5 is flat from 43M to 150M training steps (0.53–0.58), so longer training alone is
unlikely to help.

Each v7 run changes one thing from v5 (the hypothesis is in each config file):

| run | change vs. v5 | hypothesis |
|---|---|---|
| v7a | pushes every 2–5 s (was 5–10), 0.2–1.2 m/s (was 0.1–1.0) | recovery is under-trained data-wise |
| v7b | action-rate, torque and imitation costs also paused for 1.5 s after a push | costs make vigorous recovery steps too expensive |
| v7c | discount 0.99 (was 0.97) | a fall 0.5–1 s after a push is barely credited at a 0.7 s horizon |
| v7d | fall penalty −500 with the reward clip removed | falling only costs the lost alive bonus |
| v7e | new `push_recovery` reward (wide-kernel velocity tracking for 1.5 s after a push) | tracking reward has no gradient at push speeds |
| v7f | fine-tune v5 for 50M steps with v7a's pushes (`--init-from`) | a push-heavy finishing phase (curriculum) beats push-heavy training from scratch |
| v7g | 50 % of commands at zero velocity (was 10 %) | standing pushes get only ~10 % of the push data |

Results (192 trials per condition; v5, v7a, v7b and upstream and v7g re-run with 384 trials, ± 0.02–0.03).
Walking = mean over the 9 tracking commands (falls, RMS vx / wz error), nominal and `--realistic`:

| run | nominal push | realistic push | 0.4 / 0.6 / 0.8 nominal | walking nominal | walking realistic | verdict |
|---|---|---|---|---|---|---|
| upstream BEST_WALK | **0.74** | **0.69** | 0.94 / 0.68 / 0.60 | 0 %, 0.03 / 0.12 | 2 %, 0.04 / 0.25 | target |
| v5 (reference) | 0.56 | 0.49 | 0.72 / 0.61 / 0.34 | 0 %, 0.03 / 0.14 | 2 %, 0.05 / 0.19 | |
| v7a more pushes | 0.58 | 0.52 | 0.88 / 0.65 / 0.20 | 0 %, 0.03 / 0.17 | 0 %, 0.03 / 0.25 | no gain: better at 0.4–0.6, worse at 0.8 |
| v7b costs paused after push | **0.64** | **0.56** | **1.00** / 0.57 / 0.34 | 0 %, 0.03 / 0.14 | 2 %, 0.03 / 0.20 | **only gain (~2 SE)**, all of it at 0.4 m/s |
| v7c gamma 0.99 | 0.55 | 0.51 | 0.81 / 0.52 / 0.33 | 0 %, 0.07 / 0.37 | 0 %, 0.07 / 0.38 | no gain, walking much worse |
| v7d fall penalty | 0.56 | 0.54 | 0.73 / 0.61 / 0.34 | 0 %, 0.04 / 0.14 | 2 %, 0.04 / 0.19 | no change |
| v7e recovery reward | 0.41 | 0.42 | 0.77 / 0.45 / 0.00 | 0 %, 0.02 / 0.15 | 4 %, 0.07 / 0.18 | **worse**, collapses at 0.8 m/s |
| v7f fine-tune v5 with v7a pushes | 0.54 | 0.53 | 0.75 / 0.56 / 0.31 | 0 %, 0.03 / 0.15 | 2 %, 0.05 / 0.18 | no change |
| v7g 50 % zero commands | 0.60 | **0.56** | 0.88 / 0.54 / 0.38 | 0 %, 0.03 / 0.27 | 0 %, 0.04 / 0.30 | realistic gain (~2 SE), turning worse |
| v8 = v7b + v7g | 0.64 | 0.56 | 0.98 / 0.55 / 0.40 | 0 %, 0.05 / 0.15 | 0 %, 0.05 / 0.18 | **doesn't walk**; push no better than v7b |

Mean tracking error hides a policy that stands still: standing at a 0.1 m/s command costs only 0.1
m/s of error. Distance covered per command (`eval_walk_*.json`) shows that **v7c and v8 don't walk at
all** (≤ 0.08 m in 15 s at 0.1–0.15 m/s, v5: 1.3–1.9 m) and v7d stands at 0.1 m/s on the nominal
model. v7b, v7g and the others walk forward like v5.

What we learned:
- **Pausing the movement costs after a push (v7b) is the only change that helps.** It makes small
  pushes free to recover (0.4 m/s: 1.00), but doesn't move 0.6–0.8 m/s, where upstream's lead is.
- **More pushes while standing (v7g) helps a little under realistic conditions**, at the cost of
  turning (yaw-rate error doubles), presumably because less of training is spent turning.
- **More push data doesn't help**, whether from scratch (v7a) or as a fine-tuning curriculum (v7f).
  The policy sees enough pushes; it doesn't find the recovery behaviour.
- **Making falls costlier doesn't help**, whether by penalty (v7d) or by a longer horizon (v7c).
  A longer horizon also made walking much worse at this training budget.
- **Combining v7b and v7g (v8) collapsed walking**: with half the commands at zero and the gait
  costs paused after every push, PPO settled on standing, and push survival didn't improve over v7b.
- **Shaping the post-push velocity backfires (v7e):** the policy earns the reward without stepping
  and loses the 0.8 m/s pushes entirely.
- By direction (`dirs` check, 0.6 / 0.8 m/s), v7a and v7b gain at ±45° from behind, but **every
  policy still fails pushes straight from behind (0°, ±22.5°)**, which BEST_WALK survives.
- Ruled out as explanations of BEST_WALK's advantage: **marching in place** (every policy, upstream
  included, stands with both feet down at zero command) and **posture alone** (v7b and the replica
  stand with the CoM 1.3–1.5 cm ahead of the feet like BEST_WALK's 1.2 cm, and still fail; v5 leans
  most, 3.1 cm).
