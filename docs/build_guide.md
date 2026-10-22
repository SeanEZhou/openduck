# Build guide: a physical Open Duck Mini v2 that matches the simulator

This guide covers what to buy, print and check so the robot you build matches the robot the
policies are trained on (`assets/open_duck_mini_v2/`, Open Duck Mini v2 at the commits pinned in
`configs/robot.yaml`). For the mechanical assembly steps themselves, follow the upstream guides;
this document adds the sim-parity requirements and a bring-up sequence on top of them:

- Upstream assembly guide: <https://github.com/apirrone/Open_Duck_Mini/blob/v2/docs/assembly_guide.md>
- Step-by-step build on Tnkr: <https://tnkr.ai/explore/docs/open-duck-mini/open-duck-mini-v2>
- CAD (Onshape): <https://cad.onshape.com/documents/64074dfcfa379b37d8a47762>
- Runtime / Raspberry Pi setup: <https://github.com/apirrone/Open_Duck_Mini_Runtime>
- Community help: the Open Duck Discord (link in the upstream README)

## Is a policy ready to run on the robot?

**Not yet.** These gates have to be passed first:

1. **The robot exists and is characterized.** Every policy here was trained on upstream's
   identified STS3215 model, not on your servos. Build, then run the checks in
   [Bring-up](#bring-up-sequence) before any policy runs.
2. **A known-good policy runs first.** Start with upstream's `BEST_WALK_ONNX_2.onnx`, which has
   walked on real Open Ducks, on a tether. If it misbehaves, the problem is the build or the
   calibration, not our training.
3. **Our policies close the remaining sim gaps.** v4 drifts in yaw and is less push-robust than
   upstream (see the README). Run ours on hardware only once it matches upstream in simulation.
4. **The runtime matches training.** Our policies use the same 101→14 ONNX interface as
   upstream, so they run on the upstream runtime with one change: re-enable its motor-target
   rate limit, which is commented out in `scripts/v2_rl_walk_mujoco.py` (we train with it).
   Leave `--cutoff_frequency` unset (no action filter) and keep `--action_scale 0.25`.

## What must match the simulator

| Item | Simulator assumes | How to match / verify |
|---|---|---|
| Servos | 14 × Feetech **STS3215, 7.4 V version ("19 kg·cm")** | Buy only the 7.4 V model; the actuator parameters (kp 13.37, ±3.23 N·m, damping 0.56, friction 0.068, armature 0.027) were identified on it. The 12 V version is a different motor. |
| Servo IDs and joint order | `configs/robot.yaml` (legs 10–14 / 20–24, head 30–33) | Set each ID with upstream `configure_motor.py --id <id>` *before* assembly. |
| Servo gains | upstream runtime defaults (P = 30–32, head P = 8, D = 0) | Don't change them; the simulated actuator response assumes these. Verify with the step test below. |
| Supply voltage | 2S Li-ion (7.4 V nominal) | Use high-drain 18650 cells (upstream: Molicel P30B). Low batteries reduce torque, so charge before tests. |
| Printed parts mass | PLA, **15 % infill**; soles **TPU, 40 % infill** | Print with exactly these settings. The model's masses are slicer estimates for them. PETG or a different infill changes link masses. |
| Total mass | 2.11 kg (trunk 699 g, head 407 g, each leg ≈ 413 g) | Weigh sub-assemblies (table below). If anything is more than 5 % off, update the MJCF and retrain. |
| IMU | BNO055, on the trunk at x = −8 cm, z = +5 cm from the base frame, axes aligned with the trunk | Mount as in the upstream guide; set `imu_upside_down` in `duck_config.json` to match how you mounted it; verify with `imu_server.py` / `imu_client.py`. |
| Foot contacts | one switch per foot, part of the policy observation | Press-fit SS-10 switches so they trigger when the foot is loaded. Check that both toggle correctly. |
| Joint zeros | servo zero = MJCF zero; policies tolerate about ±0.03 rad (1.7°) residual offset | Align horns at zero during motor configuration, then measure residuals with `find_soft_offsets.py`. **Target < 1.7° per joint**. Larger offsets are outside the training distribution. |
| Floor | friction 0.6 nominal (trained over 0.5–1.0) | Test on a flat, hard, non-slippery surface. Avoid carpet and polished tiles. |
| Antennas | not simulated as actuated joints | Optional (2 × 9 g servos), but keep their mass: the head mass includes them. |

### Mass checklist (weigh before final assembly)

From the MJCF, each value is one moving body plus everything rigidly attached to it. Where a servo body sits on the parent or child link follows the CAD; check it in Onshape if your measured split differs:

| Sub-assembly | Sim mass |
|---|---:|
| Trunk (body shell, battery pack, electronics, IMU, the servos fixed to it) | 699 g |
| Head (head shell, Pi Zero 2W, antennas, expression parts) | 407 g |
| Neck pitch + head pitch + head yaw links | 66 + 17 + 92 g |
| Per leg: hip roll / roll-to-pitch / thigh / shin / foot | 67 / 75 / 124 / 73 / 75 g |
| **Total** | **2107 g** |

The upstream CAD export keeps the mass of electronics and expression parts (speaker, camera,
eyes) even though they're not drawn. If you skip the expression package, the head will be
lighter than 407 g. Either add equivalent mass or update the model.

## Shopping list

From the upstream BOM (prices in EUR, early 2025; re-check before ordering):
<https://docs.google.com/spreadsheets/d/1gq4iWWHEJVgAA_eemkTEsshXqrYlFxXAPwO515KpCJc>

| Part | Qty | ≈ € | Note |
|---|---:|---:|---|
| Feetech STS3215 **7.4 V** servo | 14 (+2–3 spares) | 14 each | buy 2 first and bench-test them (M0 gate) |
| Waveshare bus servo adapter (A) | 1 | 5 | USB ↔ servo bus |
| Raspberry Pi Zero 2W + microSD | 1 | 36 | runs the 50 Hz policy |
| BNO055 IMU breakout | 1 | 40 | same chip as the sim's IMU model assumes |
| SS-10 micro switches (foot contacts) | 2 (+2) | 1 each | |
| 9 g PWM servos (antennas) | 2 | 3 each | optional, but keep the mass |
| 18650 high-drain cells | 2 (+2) | 5 each | e.g. Molicel P30B |
| 2S 18650 holder, 2S BMS, 5 V UBEC, power switch, USB-C charger | 1 each | ≈ 32 | |
| XT30 connectors, 2.1 mm barrel jacks | 1 pair / 2 | ≈ 10 | |
| Bearings | 3 | 10 | |
| M3 heat-set inserts, M3 screws, blue threadlocker (Loctite 243) | 1 set | ≈ 15 | threadlocker on metal-to-metal screws only |
| PLA (~500 g), TPU (soles) | — | ≈ 20 | |
| Cables (micro-USB↔USB-C 15 cm and 50 cm), cable sheath | — | ≈ 22 | |

Add these for safe testing and model identification (not in the upstream BOM):

| Item | Why |
|---|---|
| Test stand or tether (aluminium extrusion gantry, or a strap from above) | first policy runs must be suspended, then tethered |
| Inline power switch / E-stop within reach | cut servo power instantly |
| 0.1 g kitchen scale + calipers | mass and geometry checks above |
| Spare charged cells, balance charger | consistent voltage across tests |
| Xbox controller (Bluetooth) | the upstream runtime's command input |

Core robot ≈ €400; with spares and test equipment ≈ €550–650.

## Print list

All PLA at 15 % infill except the TPU soles (40 %). STLs are in the upstream repo's `print/`
folder; the full list with quantities is in
[print_guide.md](https://github.com/apirrone/Open_Duck_Mini/blob/v2/docs/print_guide.md).
Structural parts: feet (top, side, PLA and TPU bottoms, ×2), knee-to-ankle sheets (left/right, ×4 each),
leg spacers (×4), roll-to-pitch (left, right), roll motor top/bottom (×2), trunk top/bottom,
neck sheets, head pitch-to-yaw, head yaw-to-roll, head roll mount, head and head bottom sheet,
body front/middle top/middle bottom/back, battery-pack lid, antenna holders, caches.
Expression parts (eyes, bulb, projector, speaker stand) affect head mass; see above.

Print orientation and supports per the upstream guide; inspect the knee and ankle sheets and
the hip parts after every test session. They take the highest loads.

## Bring-up sequence

Each step must pass before the next one starts. The robot is suspended or tethered until the last step.

1. **Bench-test 2 servos before ordering the rest.** Power one servo from the battery, run step and
   sine commands, and compare against the simulated response for the same test:
   `python scripts/calibrate_actuators.py --joint left_knee` (sim: about 100 ms rise to 90 %, about 17 %
   overshoot, about 60 ms lag on a chirp). Check heating at sustained load. Large differences mean the
   actuator model needs re-identification before training.
2. **Configure every motor** (ID, zero position) before assembly, then fit horns aligned at zero.
3. **Assemble** following the upstream guide. Note the hip-pitch motor orientation (it sets the zero)
   and use the mirrored left/right parts.
4. **Weigh sub-assemblies** against the mass checklist.
5. **Check joint directions against the sim**, robot suspended, one joint at a time. Conventions
   (tested in `tests/test_robot_model.py`): x points forward; **+left hip pitch** swings the left
   foot forward, **−right hip pitch** swings the right foot forward; hip roll moves the feet
   sideways in mirrored directions. A wrong sign here makes any policy fail immediately.
6. **Measure joint offsets** with upstream `find_soft_offsets.py` and write them to
   `duck_config.json`. Each should be under 1.7°.
7. **Check the IMU:** at rest, upright, the accelerometer reads about +9.8 m/s² on z and the gyro about 0;
   tilting the trunk forward should change gravity the same way it does in the sim viewer.
8. **Check the foot switches:** both read pressed when standing, released when lifted.
9. **Suspended policy test:** robot hanging with feet just off the ground, run upstream
   `BEST_WALK_ONNX_2.onnx` paused, then unpaused at zero command. The legs should hold the home pose
   and move smoothly. Kill power at the first violent motion.
10. **Tethered standing, then tethered walking** at small commands (0.05–0.1 m/s).
11. **Only then try our policies**, through the upstream runtime with the rate limit re-enabled, under
    the same suspended → tethered progression. Log everything (`--save_obs`) so the sim-to-real gap
    can be measured by replaying the logs against the simulator.

## After the build: updating the simulator (M4)

Once steps 1–8 are done, record in this repo:

- measured masses → update the MJCF inertials;
- joint offsets → `configs/calibration.yaml` (`biped.hardware.servos.ServoCalibration`);
- servo step and chirp logs → refit kp, damping, friction and armature, and adjust the randomization ranges.

Then retrain with the calibrated model. Comparing policies trained on nominal vs. calibrated
dynamics, with and without randomization, is the project's planned sim-to-real study.
