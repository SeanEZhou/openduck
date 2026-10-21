# M0 — Platform selection: Open Duck Mini v2

Status: **platform selected; M0 gate open pending two checks** (current prices, and
1–2 sample servos characterized). See checklist at the end.

## Locked revision

| Item | Value |
|---|---|
| Robot | Open Duck Mini v2 (~42 cm legs extended, ~2.1 kg in sim) |
| CAD | Onshape document `64074dfcfa379b37d8a47762` |
| Hardware repo | `apirrone/Open_Duck_Mini` @ `b23317a` (2026-01-31) |
| MJCF source | `apirrone/Open_Duck_Playground` @ `b9be205` (2025-08-05), vendored in `assets/open_duck_mini_v2/` unmodified |
| Runtime reference | `apirrone/Open_Duck_Mini_Runtime` @ `376de65` (2026-09-07) |
| License | Apache-2.0 (copied to `assets/open_duck_mini_v2/LICENSE`) |

All of this is recorded in `configs/robot.yaml → source` and asserted by tests.

## Joint configuration (adopted as-is, no 12-DOF variant)

14 position-controlled Feetech STS3215 7.4 V ("19 kg·cm") bus servos:

- per leg (5): hip yaw, hip roll, hip pitch, knee, ankle pitch — **no ankle roll**
- head (4): neck pitch, head pitch, head yaw, head roll
- 2 × 9 g PWM servos for antennas (cosmetic, not in the policy)

Joint order, servo bus IDs, limits and home pose: `configs/robot.yaml`. Joint order
equals actuator order in the MJCF equals the upstream runtime's servo order.

Actuator model: upstream identified the STS3215 with Rhoban BAM
(`kp 13.37, forcerange ±3.23 N·m, damping 0.56, frictionloss 0.068, armature 0.027`).
These are *their* servos; ours must be re-identified (Track B step 3).

## Sensors and compute

BNO055 IMU (I²C), 2 × SS-10 foot contact switches, Raspberry Pi Zero 2W,
Waveshare bus servo adapter (1 Mbps). Upstream runs the 50 Hz policy loop on the Pi.

## Cost (upstream BOM, EUR, not re-verified)

| Group | Items | € |
|---|---|---:|
| Servos | 14 × STS3215 7.4 V @ 14 | 196 |
| Compute/sensors | Pi Zero 2W, SD, bus adapter, BNO055, foot switches, 2 × 9 g servos | ~92 |
| Power | 2 × 18650 (high-drain), holder, 2S BMS, 5 V UBEC, switch, USB-C charger, connectors | ~49 |
| Mechanical/misc | M3 inserts, PLA (~500 g), TPU, 3 bearings, cables, sheath | ~58 |
| **Core robot** | | **~398** |
| Expression package (optional) | LEDs, speaker, amp, camera, mic | ~34 |

Recommended additions for this project (not in upstream BOM):

| Item | Why | € (est.) |
|---|---|---:|
| 2 sample STS3215 bought first | characterize before bulk order (gate) | (part of the 14 + spares) |
| 2–3 spare STS3215 | stripped gears / burnt servos during sim-to-real | 30–45 |
| Bench PSU or spare 18650s + charger | long characterization runs | 30–60 |
| Test stand / tether (aluminium extrusion or a simple gantry) | M5 safety gate | 30–60 |
| Physical E-stop / inline power switch | M5 safety gate | 10 |
| Kitchen scale (0.1 g) + calipers | mass/dimension identification (M4) | 20–30 |

Planning total ≈ **€550–650**, well under the $700–1,200 envelope.

## Fabrication notes

- Upstream prints **all structural parts in PLA at 15 % infill**, soles in TPU at 40 %.
  The MJCF masses are slicer estimates for those settings. Printing in PETG or with
  different infill changes link masses: either match the upstream print settings or
  weigh every printed part and update the model (M4).
- Use the published bearings and inserts; do not side-load bare servo shafts.

## Gaps found while integrating upstream (relevant to sim-to-real)

1. **Rate limit mismatch:** upstream training clips motor targets to 5.24 rad/s; the
   upstream runtime has that clip commented out.
2. **Head command mismatch:** the runtime adds head commands to head targets;
   training does not.
3. **Action-history off-by-one:** training puts a[t-1] in the "last action" slot
   when choosing a[t+1]; the runtime puts a[t].
4. **IMU latency randomization is inert:** the delayed buffer holds gravity,
   which is not in the observation; gyro/accelerometer are never delayed.
5. **Joint-offset randomization is inert:** it perturbs `qpos0`, which has no effect
   on hinge joints without springs.
6. **Torso mass/CoM randomization targets the massless `base` body**, not the trunk.

7. **Episodes don't really reset during training:** upstream trains with Playground's default
   auto-reset, which restores only `data`/`obs` to a cached first state. Every env restarts
   from the same initial sample, and `info` (action history, motor targets, push timer)
   carries across episodes.
8. **Lateral dead-band (kept, not a bug):** the velocity-tracking reward ignores |vy error| < 0.1 m/s.
   This explains the published policies' weak response to small lateral commands, but it also
   tolerates the robot's natural lateral sway while walking. Removing it (`y_tol: 0`) hurt
   forward walking (vx error 0.13 vs 0.05 at 0.15 m/s) without improving lateral tracking
   (`results/mjx_walking_ytol0_*`). Better lateral tracking needs a sway-insensitive target
   (e.g. velocity averaged over a gait period), not a smaller tolerance.
9. **Resets start the feet inside the floor:** upstream scales joint angles at reset while
   keeping the base height, so the feet start up to ~1 cm in the ground (more with the
   upstream ×[0.5, 1.5] scaling). MJX tolerates this; CPU MuJoCo ejects the robot. This caused a
   15 % nominal fall rate when an MJX-trained standing policy was run on CPU. Both of our envs
   now lower or raise the base so the lower foot rests on the ground.

This project fixes 1–3 by construction (a single `PolicyRuntime` class runs in
both sim and deployment), 4–6 in `biped/envs/` and `biped/mjx/`, 7 with `full_reset=True`,
and 9 with grounded resets.

## M0 gate checklist

- [x] Exact frame + revision locked (above)
- [x] Reference CAD and MJCF available; MJCF loads; joint layout verified by tests
- [x] Joint control feasible: position control over the Feetech bus at 50 Hz is
      demonstrated upstream; published policies walk in our simulator (baseline reproduced)
- [x] Compatible actuator list: STS3215 **7.4 V** variant only
- [ ] Re-check current prices/availability (upstream BOM is from early 2025)
- [ ] Buy 2 STS3215, run `scripts/calibrate_actuators.py`-equivalent bench tests
      (step, chirp, thermal at stall, bus latency) and compare against the sim
      response before ordering the remaining 12
