# Motion Tracking and Kinematic Planner with Keyboard Controls

<figure style="margin: 1em 0;">
<video width="100%" autoplay loop muted playsinline style="border-radius: 8px;">
  <source src="../_static/kinematic_planner/Navigation.mp4" type="video/mp4">
</video>
<figcaption style="text-align: center; font-style: italic; margin-top: 0.5em;">In-the-wild navigation demo using the kinematic planner with keyboard controls.</figcaption>
</figure>

```{video} ../_static/Keyboard_Guidance.mp4
:width: 100%
```
*Video: Keyboard control walkthrough — starting the control system, playing reference motions, and using planner mode for real-time locomotion.*

Control the robot using keyboard commands for reference motion playback and planner-based locomotion (using `--input-type keyboard`).

```{admonition} Prerequisites
:class: note
Complete the [Quick Start](../getting_started/quickstart.md) to have the sim2sim loop running.
```

```{admonition} Emergency Stop
:class: danger
Press **`O`** at any time to immediately stop control and exit. Always keep a hand near the keyboard ready to press **`O`**.
```

## Launch

**Sim2Sim (MuJoCo):**

```bash
# Terminal 1 — MuJoCo simulator (from repo root)
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py

# Terminal 2 — C++ deployment (from gear_sonic_deploy/)
bash deploy.sh --input-type keyboard sim
```

**MuJoCo dynamics with Isaac Sim 5.0 camera rendering:**

Use the existing `gr00t-sim` conda environment (digits **00**); see
[installation instructions](../getting_started/installation_deploy.md#isaac-renderer).
Terminal 2 and all keyboard controls below remain the same.

```bash
# Terminal 1, repository root: original MuJoCo window and keyboard workflow
conda activate gr00t-sim
python gear_sonic/scripts/run_sim_loop.py

# For Isaac modes, after accepting NVIDIA's EULA (see installation instructions)
export OMNI_KIT_ACCEPT_EULA=YES

# Isaac RTX camera and image preview; automatically publish on port 5555
python gear_sonic/scripts/run_sim_loop.py --render-backend isaac

# Hide only the Isaac image preview; the MuJoCo control window remains available
python gear_sonic/scripts/run_sim_loop.py --render-backend isaac --isaac-headless

# Compare the same snapshot: Isaac on 5555, MuJoCo on 5559
python gear_sonic/scripts/run_sim_loop.py --render-backend both
```

Run one simulator command at a time. `--render-fps 30` sets the camera rate cap;
actual performance depends on the GPU. MuJoCo continues its 200 Hz dynamics/DDS
loop while the camera worker initializes, renders, or stops. Press **`9`** in the
**MuJoCo** window to release the suspension; the Isaac window only displays the
visual puppet. The Isaac window previews the exact published RTX image; Kit runs
offscreen and does not expose a scene editor. Closing the preview leaves MuJoCo
running and reports the worker exit. Focus the preview and press `Q` or `Esc` to
close it.

```bash
# Separate viewer terminals, using the existing camera client unchanged
python gear_sonic/scripts/run_camera_viewer.py --camera-host localhost --camera-port 5555
python gear_sonic/scripts/run_camera_viewer.py --camera-host localhost --camera-port 5559

# Optional camera publishing with the MuJoCo-only backend
python gear_sonic/scripts/run_sim_loop.py --enable-offscreen --enable-image-publish
```

`--camera-port` selects the main stream; `--comparison-camera-port` selects the
MuJoCo stream in `both` mode. Each stream contains RGB `images["ego_view"]`
at 640×480 and the state sampling wall time in `timestamps["ego_view"]`.
Both streams share the sampling timestamp in comparison mode. The existing
viewer, exporter and VLA camera subscriber use the same image protocol.
This change does not launch N1.7 inference or `launch_inference.py`.

For architecture and validation commands, see [independent rendering](isaac_rendering.md).

**Real Robot:**

```bash
# From gear_sonic_deploy/
bash deploy.sh --input-type keyboard real
```

## Step-by-Step: Normal Mode (Reference Motion Tracking)

Normal Mode plays back pre-loaded reference motions. This is the default mode when the program starts.

1. In Terminal 2, press **`]`** to start the control system.
2. In the MuJoCo window, press **`9`** to drop the robot to the ground.
3. Go back to Terminal 2, press **`T`** to play the current reference motion — the robot executes it to completion.
4. Press **`N`** to switch to the next motion sequence, or **`P`** for the previous one.
5. Press **`T`** again to play the new motion.
6. To replay the same motion, press **`T`** again after it finishes. To stop mid-motion and return to the first frame, press **`R`** — the robot pauses at the first frame without terminating the policy.
7. Use **`Q`** / **`E`** to nudge the heading left or right (±π/12 rad per press).
8. Press **`I`** to reinitialize the base quaternion and reset the heading to zero, i.e. robot will think the current facing is the facing at the first frame of the reference.
9. When done, press **`O`** to stop control and exit.

## Step-by-Step: Planner Mode (Real-time Motion Generation)

Planner Mode lets you control the robot in real time — choose a locomotion style, steer with WASD, and adjust speed and height on the fly.

1. From Normal Mode, press **`ENTER`** to switch to Planner Mode. The terminal will print `Planner enabled`.
2. The robot starts in the **Locomotion** motion set. Press **`1`** for slow walk, **`2`** for walk, or **`3`** for run.
3. Press **`W`** to walk forward. The robot uses a momentum system — holding a direction key sets momentum to full; releasing it lets the robot gradually decelerate and return to idle.
4. Steer with **`A`** / **`D`** (adjust heading and moving direction together) or turn in place with **`Q`** / **`E`** (±π/6 rad per press, only facing direction).
5. Press **`,`** / **`.`** to strafe left / right.
6. Press **`S`** to move backward.
7. Adjust speed with **`9`** (decrease) / **`0`** (increase). Speed ranges depend on the current mode (see tables below).
8. Press **`N`** to cycle to the next motion set (Locomotion → Squat → Boxing → Styled Walking → …). Use **`P`** to go back.
9. Within a motion set, press **`1`**–**`8`** to pick a specific mode (see the Motion Sets section below).
10. For squat-type modes, adjust body height with **`-`** (lower) / **`=`** (higher), clamped to 0.2–0.8 m.
11. If you need an immediate halt, press **`R`**, **`` ` ``**, or **`~`** — this resets movement momentum to zero instantly.
12. Press **`ENTER`** again to return to Normal Mode, or **`O`** to stop and exit.

## Control Reference

### System Controls (Both Modes)

| Key | Action |
|-----|--------|
| **]** | Start control system |
| **O** | Stop control and exit (emergency stop) |
| **ENTER** | Toggle between Normal / Planner modes |
| **I** | Reinitialize base quaternion and reset heading |
| **Z** | Toggle encoder mode (between mode 0 and mode 1, if encoder loaded) |
| **F** | Report motor temperatures (TTS voice alert) |

### Normal Mode Keys

| Key | Action |
|-----|--------|
| **T** | Play current motion to completion |
| **R** | Restart current motion from beginning (pause at frame 0) |
| **P** / **N** | Previous / Next motion sequence |
| **Q** / **E** | Adjust delta heading (at policy level) left / right (±π/12 rad) |

### Planner Mode Keys

**Movement:**

| Key | Action |
|-----|--------|
| **W** / **S** | Move forward / backward |
| **A** / **D** | Adjust heading slightly and move forward (left / right) |
| **,** / **.** | Strafe left / right |

**Heading:**

| Key | Action |
|-----|--------|
| **Q** / **E** | Adjust facing direction (at planner level) left / right (±π/6 rad) |
| **J** / **L** | Adjust delta heading (at policy level) left / right (±π/12 rad) |

**Mode & Speed:**

| Key | Action |
|-----|--------|
| **N** / **P** | Next / Previous motion set |
| **1**–**8** | Select mode within the current set |
| **9** / **0** | Decrease / Increase movement speed |
| **-** / **=** | Decrease / Increase height (non-standing sets, 0.2–0.8 m) |
| **T** | Play motion |

**Emergency:**

| Key | Action |
|-----|--------|
| **R** / **`** / **~** | Emergency stop (immediate momentum reset) |

## Motion Sets

A **motion set** is a group of related movement styles (e.g., locomotion, gestures, or crouching). Selecting a mode within a set makes the robot behave in that style. Each set contains up to eight selectable modes.

Cycle through motion sets with **`N`** (next) / **`P`** (previous). Within each set, press **`1`**–**`8`** to select a mode. For more details on the underlying planner model, mode indices, and input/output specifications, see the [Kinematic Planner ONNX Model Reference](../references/planner_onnx.md).

### Set 0 — Locomotion (Standing)

| Key | Mode | Speed Range |
|-----|------|-------------|
| **1** | Slow Walk | 0.2–0.8 m/s |
| **2** | Walk | — |
| **3** | Run | 1.5–3.0 m/s |
| **4** | Happy | — |
| **5** | Stealth | — |
| **6** | Injured | — |

```{tip}
For lateral (side-stepping) movement using **`,`** / **`.`**, we recommend keeping the target velocity at around **0.4 m/s**. Higher velocities during strafing can cause the robot's feet to collide due to the cross-legged foot placement required for lateral steps.
```

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1em; margin: 1em 0;">
<figure style="margin: 0;">
<video width="100%" autoplay loop muted playsinline style="border-radius: 8px;">
  <source src="../_static/kinematic_planner/planner_happy.mp4" type="video/mp4">
</video>
<figcaption style="text-align: center; font-style: italic; margin-top: 0.5em;">Happy styled walking.</figcaption>
</figure>
<figure style="margin: 0;">
<video width="100%" autoplay loop muted playsinline style="border-radius: 8px;">
  <source src="../_static/kinematic_planner/planner_stealth.mp4" type="video/mp4">
</video>
<figcaption style="text-align: center; font-style: italic; margin-top: 0.5em;">Stealth styled walking.</figcaption>
</figure>
<figure style="margin: 0;">
<video width="100%" autoplay loop muted playsinline style="border-radius: 8px;">
  <source src="../_static/kinematic_planner/planner_injured.mp4" type="video/mp4">
</video>
<figcaption style="text-align: center; font-style: italic; margin-top: 0.5em;">Injured styled walking.</figcaption>
</figure>
<figure style="margin: 0;">
<video width="100%" autoplay loop muted playsinline style="border-radius: 8px;">
  <source src="../_static/kinematic_planner/planner_run.mp4" type="video/mp4">
</video>
<figcaption style="text-align: center; font-style: italic; margin-top: 0.5em;">Running locomotion mode.</figcaption>
</figure>
</div>

### Set 1 — Squat / Ground

Height adjustable with **`-`** / **`=`** (0.2–0.8 m). Initial height defaults to 0.8 m when entering this set.

| Key | Mode | Speed Range |
|-----|------|-------------|
| **1** | Squat | static |
| **2** | Kneel (Two Legs) | static |
| **3** | Kneel (One Leg) | static |
| **4** | Hand Crawling | 0.4–1.0 m/s |
| **5** | Elbow Crawling | 0.7–1.0 m/s |

<div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1em; margin: 1em 0;">
<figure style="margin: 0;">
<video width="100%" autoplay loop muted playsinline style="border-radius: 8px;">
  <source src="../_static/kinematic_planner/planner_kneeling.mp4" type="video/mp4">
</video>
<figcaption style="text-align: center; font-style: italic; margin-top: 0.5em;">Kneeling mode with variable height control.</figcaption>
</figure>
<figure style="margin: 0;">
<video width="100%" autoplay loop muted playsinline style="border-radius: 8px;">
  <source src="../_static/kinematic_planner/hand_crawling.mp4" type="video/mp4">
</video>
<figcaption style="text-align: center; font-style: italic; margin-top: 0.5em;">Hand crawling locomotion.</figcaption>
</figure>
<figure style="margin: 0;">
<video width="100%" autoplay loop muted playsinline style="border-radius: 8px;">
  <source src="../_static/kinematic_planner/planner_elbow_crawling.mp4" type="video/mp4">
</video>
<figcaption style="text-align: center; font-style: italic; margin-top: 0.5em;">Elbow crawling locomotion.</figcaption>
</figure>
</div>

### Set 2 — Boxing

| Key | Mode | Speed Range |
|-----|------|-------------|
| **1** | Idle Boxing | static |
| **2** | Walk Boxing | 0.7–1.5 m/s |
| **3** | Left Jab | 0.7–1.5 m/s |
| **4** | Right Jab | 0.7–1.5 m/s |
| **5** | Random Punches | 0.7–1.5 m/s |
| **6** | Left Hook | 0.7–1.5 m/s |
| **7** | Right Hook | 0.7–1.5 m/s |

<figure style="margin: 1em 0;">
<video width="100%" autoplay loop muted playsinline style="border-radius: 8px;">
  <source src="../_static/kinematic_planner/planner_boxing.mp4" type="video/mp4">
</video>
<figcaption style="text-align: center; font-style: italic; margin-top: 0.5em;">Boxing mode demo.</figcaption>
</figure>

### Set 3 — Additional Styled Walking

| Key | Mode |
|-----|------|
| **1** | Careful |
| **2** | Object Carrying |
| **3** | Crouch |
| **4** | Happy Dance |
| **5** | Zombie |
| **6** | Point |
| **7** | Scared |

## Movement Momentum System

The planner usage in keyboard mode uses a momentum-based movement system:
- Pressing a direction key (**W/S/A/D/,/.**) sets momentum to **1.0** (full speed).
- Each frame without a direction key press, momentum decays multiplicatively (×0.999).
- When momentum drops below **0.1**, the robot transitions to idle (for the Locomotion set) or holds the current static pose (for Squat and Boxing sets).
- Emergency stop (**R/`/~**) instantly resets momentum to zero.

This means you don't need to hold a key down — a single press starts movement, and the robot coasts to a stop naturally.
