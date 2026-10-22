"""Side-by-side GIFs of two policies (CPU MuJoCo, nominal model) for the README.

    python scripts/make_gifs.py --ours-walk results/mjx_walking_lin5_seed0/policy.onnx \
        --ours-stand results/mjx_standing_seed0/policy.onnx --out docs/media

Writes standing.gif (head commands while standing), walking.gif (forward then turn)
and push.gif (identical kicks from four directions in sequence, outcome labelled).
Both policies always receive identical commands and pushes.
"""

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from biped.envs import make_env  # noqa: E402
from biped.policy import load_policy  # noqa: E402

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
UPSTREAM = "third_party/Open_Duck_Mini/BEST_WALK_ONNX.onnx"
GREEN, RED, INK, ACCENT = (20, 140, 60), (200, 40, 40), (30, 30, 30), (0, 90, 200)


def font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


class Recorder:
    def __init__(self, env, w, h, distance=0.9, azimuth=150, elevation=-12, track=True):
        self.env = env
        self.renderer = mujoco.Renderer(env.robot.model, h, w)
        self.cam = mujoco.MjvCamera()
        if track:
            self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.cam.trackbodyid = env.robot.root_body
        else:  # fixed camera on the start position, so falls stay in view
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.cam.lookat[:] = [0.0, 0.0, 0.2]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = distance, azimuth, elevation

    def frame(self):
        self.renderer.update_scene(self.env.robot.data, self.cam)
        return self.renderer.render()

    def close(self):
        # Must be closed before creating another Renderer: a second live GL context renders garbage.
        self.renderer.close()


def episode(env, rec, policy, seconds, command_fn, push=None, every=3, freeze_tilt_deg=50.0):
    """Run one episode; returns (frames, overlays per frame, fell).

    Only the soles collide with the floor in this model, so a falling body sinks through the
    ground. If the episode ends in a fall, the image is frozen at the first frame tilted past
    `freeze_tilt_deg` and labelled FELL from there.
    """
    env.reset(seed=0)
    env.command = command_fn(0.0)
    env.runtime.reset()
    obs = env.runtime.observe(env._measure(), env.command)
    r = env.robot
    frames, notes, tilts = [], [], []
    fell = False
    push_k = int(push[0] / env.dt) if push else None
    n = int(seconds / env.dt)
    for k in range(n):
        t = k * env.dt
        env.command = command_fn(t)
        if push and k == push_k:
            mag, theta = push[1], push[2]
            r.data.qvel[r.base_qvel_adr : r.base_qvel_adr + 2] += mag * np.array([np.cos(theta), np.sin(theta)])
        obs, _, term, _, _ = env.step(policy(obs))
        if k % every == 0:
            frames.append(rec.frame())
            tilts.append(np.degrees(np.arccos(np.clip(-r.gravity_in_imu()[2], -1, 1))))
            notes.append({"t": t, "pushing": push_k is not None and push_k <= k < push_k + 25,
                          "fell": False, "after_push": push_k is not None and k >= push_k})
        if term:
            fell = True
            break
    if fell:
        i0 = next((i for i, a in enumerate(tilts) if a > freeze_tilt_deg), len(frames) - 1)
        frames = frames[: i0 + 1]
        notes = notes[: i0 + 1]
        notes[-1]["fell"] = True
    # Pad (freeze last frame) so panels in one GIF have equal length.
    while len(frames) < -(-n // every):  # same length as a full episode (ceil(n / every))
        frames.append(frames[-1])
        notes.append(dict(notes[-1], pushing=False))
    return frames, notes, fell


def annotate(img, title, subtitle, note, push_label=None, verdict=True):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    w = im.width
    d.rectangle([0, 0, w, 40], fill=(255, 255, 255))
    d.text((8, 4), title, fill=INK, font=font(15))
    d.text((8, 23), subtitle, fill=(90, 90, 90), font=font(12))
    d.text((w - 52, 4), f"{note['t']:4.1f}s", fill=(90, 90, 90), font=font(12))
    if push_label and note["pushing"]:
        d.rectangle([4, im.height - 30, 70, im.height - 6], fill=(255, 255, 255))
        d.text((10, im.height - 27), push_label, fill=ACCENT, font=font(17))
    if verdict and note["fell"]:
        d.text((w - 70, im.height - 26), "FELL", fill=RED, font=font(18))
    elif verdict and push_label and note["after_push"] and not note["pushing"]:
        d.text((w - 118, im.height - 26), "RECOVERED", fill=GREEN, font=font(16))
    return im


def save_gif(rows, path, fps=17):
    """rows: list of rows, each a list of PIL frame sequences (equal length)."""
    n = min(len(seq) for row in rows for seq in row)
    canvases = []
    for i in range(n):
        tiles = [[seq[i] for seq in row] for row in rows]
        w = sum(t.width for t in tiles[0]) + 4 * (len(tiles[0]) - 1)
        h = sum(row[0].height for row in tiles) + 4 * (len(tiles) - 1)
        canvas = Image.new("RGB", (w, h), (255, 255, 255))
        y = 0
        for row in tiles:
            x = 0
            for t in row:
                canvas.paste(t, (x, y))
                x += t.width + 4
            y += row[0].height + 4
        canvases.append(canvas)
    # One shared palette from sampled frames (per-frame palettes + delta optimization corrupt GIFs).
    sample = canvases[:: max(1, n // 8)]
    strip = Image.new("RGB", (sample[0].width, sample[0].height * len(sample)))
    for k, c in enumerate(sample):
        strip.paste(c, (0, k * c.height))
    palette = strip.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    out = [c.quantize(palette=palette, dither=Image.Dither.NONE) for c in canvases]
    out[0].save(path, save_all=True, append_images=out[1:], duration=int(1000 / fps), loop=0, optimize=False)
    print(f"wrote {path} ({Path(path).stat().st_size / 1e6:.1f} MB, {n} frames)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ours-walk", default="results/mjx_walking_v5_seed0/policy.onnx")
    p.add_argument("--ours-stand", default="results/mjx_standing_v2_seed0/policy.onnx")
    p.add_argument("--ours-legs", default="results/mjx_walking_legs_v3_seed0/policy.onnx",
                   help="headless (legs-only) robot policy, for the headless_* GIFs")
    p.add_argument("--upstream", default=UPSTREAM)
    p.add_argument("--out", default="docs/media")
    p.add_argument("--only", nargs="*", default=["standing", "walking", "push", "headless"])
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    env = make_env("configs/eval_baseline.yaml", {"episode_length": 10**6, "commands": {"resample_steps": 0},
                                                  "push": {"enable": False}})
    up, ours_walk, ours_stand = load_policy(a.upstream), load_policy(a.ours_walk), load_policy(a.ours_stand)
    up_title, ours_title = "Upstream (Open Duck author)", "Ours (MJX-trained)"

    if "standing" in a.only:
        def head(t):
            c = np.zeros(7)
            c[3] = 0.35 + 0.35 * np.sin(2 * np.pi * t / 4)        # neck pitch
            c[5] = 0.7 * np.sin(2 * np.pi * t / 3)                  # head yaw
            return c
        rec = Recorder(env, 360, 270)
        panels = []
        for title, pol in ((up_title, up), (ours_title + " standing", ours_stand)):
            frames, notes, _ = episode(env, rec, pol, 8.0, head)
            panels.append([annotate(f, title, "stand, head commands", n, verdict=True) for f, n in zip(frames, notes)])
        rec.close()
        save_gif([panels], out / "standing.gif")

    if "walking" in a.only:
        def walk(t):
            c = np.zeros(7)
            if t < 5.0:
                c[0] = 0.15
            else:
                c[2] = 0.6
            return c
        rec = Recorder(env, 360, 270, distance=1.0)
        panels = []
        for title, pol in ((up_title, up), (ours_title + " walking", ours_walk)):
            frames, notes, _ = episode(env, rec, pol, 9.0, walk)
            subs = ["forward 0.15 m/s" if n["t"] < 5.0 else "turn 0.6 rad/s" for n in notes]
            panels.append([annotate(f, title, s, n) for f, s, n in zip(frames, subs, notes)])
        rec.close()
        save_gif([panels], out / "walking.gif")

    if "push" in a.only:
        # One large side-by-side segment per direction, played in sequence. Magnitude per direction:
        # the largest at which at least one policy recovers (from scripts/push_recovery.py sweeps),
        # so every segment needs an active stepping response.
        rec = Recorder(env, 400, 300, distance=1.3, elevation=-12, track=False)
        directions = [(0, "from behind", 0.8), (90, "from the right", 0.8), (180, "from the front", 0.4),
                      (270, "from the left", 0.8)]
        segments = []
        for deg, label, mag in directions:
            seg = []
            for title, pol in ((up_title, up), (ours_title, ours_walk)):
                frames, notes, fell = episode(env, rec, pol, 4.0, lambda t: np.zeros(7),
                                              push=(1.0, mag, np.radians(deg)))
                seg.append([annotate(f, title, f"push {label}, {mag:.1f} m/s", n, push_label="PUSH")
                            for f, n in zip(frames, notes)])
                print(f"{title:28s} push {label:15s} {mag:.1f} m/s: {'FELL' if fell else 'recovered'}")
            segments.append(seg)
        rec.close()
        frames_by_panel = [sum((seg[i] for seg in segments), []) for i in range(2)]
        save_gif([frames_by_panel], out / "push.gif")

    if "headless" in a.only:
        # Each panel may use a different robot model, so each episode gets its own env and renderer
        # (closed before the next one is created).
        ov = {"episode_length": 10**6, "commands": {"resample_steps": 0}, "push": {"enable": False}}
        setups = [("Upstream (with head)", "configs/eval_baseline.yaml", up),
                  ("Ours (with head)", "configs/eval_baseline.yaml", ours_walk),
                  ("Ours headless (10 servos)", "configs/eval_legs.yaml", load_policy(a.ours_legs))]
        envs = {c: make_env(c, ov) for _, c, _ in setups}

        def run(cfg, pol, w, h, seconds, command_fn, push=None, **rec_kw):
            rec = Recorder(envs[cfg], w, h, **rec_kw)
            try:
                return episode(envs[cfg], rec, pol, seconds, command_fn, push=push)
            finally:
                rec.close()

        directions = [(0, "from behind"), (45, "from behind-right"), (135, "from front-right"),
                      (225, "from front-left"), (315, "from behind-left")]
        segments = []
        for deg, label in directions:
            seg = []
            for title, cfg, pol in setups:
                frames, notes, fell = run(cfg, pol, 360, 270, 4.0, lambda t: np.zeros(7),
                                          push=(1.0, 0.8, np.radians(deg)), distance=1.3, track=False)
                seg.append([annotate(f, title, f"push {label}, 0.8 m/s", n, push_label="PUSH")
                            for f, n in zip(frames, notes)])
                print(f"{title:26s} push {label:18s}: {'FELL' if fell else 'recovered'}")
            segments.append(seg)
        save_gif([[sum((seg[i] for seg in segments), []) for i in range(len(setups))]],
                 out / "headless_push.gif")

        walk = lambda t: np.r_[0.15, 0, 0, 0, 0, 0, 0]  # noqa: E731
        panels = []
        for title, cfg, pol in setups[1:]:
            frames, notes, _ = run(cfg, pol, 360, 270, 8.0, walk, distance=1.0)
            panels.append([annotate(f, title, "forward 0.15 m/s", n) for f, n in zip(frames, notes)])
        save_gif([panels], out / "headless_walk.gif")


if __name__ == "__main__":
    main()
