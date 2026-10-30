# Independent camera rendering

`run_sim_loop.py` owns the only dynamics loop. At each MuJoCo step it publishes
SONIC LowState through DDS and sends a single nonblocking ZMQ IPC message with
full `qpos`, simulation time, wall-clock sampling time and sequence number.
Each run creates its own temporary IPC directory and compiled model snapshot.

The camera worker is a fresh Python interpreter in the same environment.
It loads the compiled MuJoCo model and only calls `mj_forward`, then renders
with MuJoCo, Isaac Sim 5.0, or both. The publisher and subscriber conflate state;
the worker also drains transport bursts before selecting its next snapshot.
It skips duplicate sequences and never republishes a frame without new state.
Rendering, JPEG encoding and image ZMQ sends all happen in this worker.

Isaac references only `/g1_29dof_with_hand_rev_1_0` from the local `g1_43dof.usd`.
It excludes the asset's room and physics scene, disables rigid bodies, collisions,
articulations and joints, and binds local preview materials instead of remote MDL.
All 44 MuJoCo bodies use world transforms from forward kinematics; nine additional
USD links follow fixed-joint frames. Finger links are included. The ground uses
the MuJoCo floor's world pose and its compiled checker texture and repetition
settings. Local preview materials and lighting can produce different colors and
reflections from MuJoCo; comparison checks geometry and projection.

The USD camera uses MuJoCo `head_camera`'s world pose, vertical field of view,
aspect ratio and near/far distances. Both native camera frames look along -Z
with +Y up. Gf matrices use row vectors, so the MuJoCo rotation/translation matrix
is transposed when assigned to USD. No manual head offset is added.
Replicator captures synchronously with a stopped timeline and `delta_time=0.0`;
no Isaac physics step runs. In `both` mode both images use the same immutable
snapshot and sampling timestamp, although they arrive at their ports separately.
The first RTX capture warms shaders before subscribing to states, so shader
compilation cannot result in publishing a minute-old first snapshot. Kit runs
offscreen in both modes. Windowed mode opens a lightweight OpenCV preview of the
exact Isaac RTX RGB array being published; `--isaac-headless` hides that preview.
This is an image preview, not Kit's USD scene editor. The native Kit GUI stalled
repeatedly in this machine's simultaneous MuJoCo/Kit window setup, so it is not a
runtime dependency of this integration. The MuJoCo operation window and keyboard
controls remain available.

The small `isaac_renderer.kit` experience loads the viewport and Replicator APIs
without the full Isaac simulation/task application. In particular it omits
`isaacsim.core.throttling`, whose STOP/PAUSE callback re-enables Eco Mode and can
stop a camera whose timeline intentionally remains paused. NVIDIA PRIME variables
are set only for the child process (explicit environment overrides are preserved).
The preview window and image encoding also stay inside that renderer process.

The wire format is the existing base64 JPEG `ImageMessageSchema`. Both renderers
provide RGB arrays to its legacy string serializer, which preserves array channel
order across its existing OpenCV encode/decode pair. The worker reports measured
FPS and mean sampling-to-publish latency every five seconds. Add
`--verbose-timing` to report the physics loop frequency and maximum interval.

Initialization never blocks physics. Worker failure or window closure is reported
within five seconds while MuJoCo continues; there is no automatic restart or
backend switch. Exiting MuJoCo terminates and reaps the worker (including a paused
worker), then removes its IPC/model files. A worker also exits if its parent dies.

The state-sync approach and render synchronization settings are adapted from
SIMPLE (`src/simple/engines/isaacsim.py` and `isaac_app.py`, MIT,
Copyright 2025–2026 SIMPLE Authors). SIMPLE and Isaac Lab are not dependencies.

## Checks

```bash
conda activate gr00t-sim
# After reading and accepting NVIDIA's EULA; inherited by the camera worker
export OMNI_KIT_ACCEPT_EULA=YES
# Real MuJoCo child process, existing image schema, RGB channels, idle suppression,
# SIGSTOP >500ms, latest snapshot after resume, worker death and cleanup
MUJOCO_GL=egl python -m unittest gear_sonic.tests.test_state_render -v

# Actual USD transform and camera projection check, no VLA weights
python gear_sonic/scripts/check_render_alignment.py --headless
# Actual RGB marker pixels across four changing camera poses (detects old frames)
python gear_sonic/scripts/check_render_alignment.py --headless --render-check

# Live timing, Isaac startup and the existing DDS/keyboard deployment workflow
python gear_sonic/scripts/run_sim_loop.py --render-backend both --verbose-timing
```

The alignment check uses 20 deterministic root/joint poses and compares USD world
transforms against MuJoCo bodies, including the fixed-part relationships. It
projects markers through the USD camera and independently through MuJoCo's GL
camera. Limits are 1 mm, 0.1 degrees and 1 pixel. `--usd-only` runs the numerical
check with standalone USD bindings; it does not validate RTX-rendered pixels.
`--render-check` also compares red/green/blue marker centroids in actual RTX and
MuJoCo images immediately after each camera move, with the same 1-pixel limit.

For the live test, start `bash deploy.sh --input-type keyboard sim` in Terminal 2.
Verify standing, release with MuJoCo **9**, move and turn with the documented keys.
Use the reported `[Render] ... worker PID` to `kill -STOP PID` for over 500 ms,
then `kill -CONT PID`. Check DDS intervals and controller logs throughout startup,
rendering and the pause; `kill -TERM PID` checks worker exit independently.
Use the two viewers from the [keyboard tutorial](keyboard.md) to compare streams.
Record FPS, latency and `nvidia-smi` VRAM separately for windowed and headless
Isaac on the RTX 4060 Laptop 8 GB; 30 FPS is a cap, not a guaranteed rate.

N1.7 / `launch_inference.py` integration is outside this change. Its eventual camera
input can use the existing `ego_view` stream. The current exporter and VLA reader
continue to use `ComposedCameraClientSensor` without communication-layer changes.

## Local validation, 2026-10-29

Hardware: RTX 4060 Laptop GPU 8 GB, driver 580.95.05, Ubuntu 24.04.4,
Python 3.11.18, Isaac Sim 5.0.0.0, MuJoCo 3.17.0. The driver reported the
CPU powersave profile; no power, driver, BIOS or monitor settings were changed.

The following measurements used `--render-backend both --isaac-headless`, the
MuJoCo operation window enabled, a 30 FPS camera cap, and the existing SONIC
keyboard controller active on `lo`. They include actual client receive overhead.

| Measurement | Result |
|---|---|
| Isaac / MuJoCo image rates | 10.2 / 10.2 FPS |
| Mean sample-to-receive delay | 105.08 / 118.34 ms |
| Equal sampling timestamps observed | 101 image pairs |
| Total GPU memory during this control test | 3261 MiB |
| DDS rate during control | 200.11 Hz |
| Maximum DDS interval during control | 19.828 ms |
| Renderer stopped with SIGSTOP for 1 second | 200.67 Hz; maximum 8.954 ms |
| Renderer resumed | 200.12 Hz; maximum 15.296 ms |
| Renderer terminated | 200.09 Hz; maximum 11.854 ms |

The camera process also ran for over eight minutes before deliberate termination.
The final keyboard trial exercised start, planner idle/standing, suspension
release, slow walking, heading adjustment, movement stop and controller exit.
There were no fall resets or LowState-watchdog errors in this trial. After the
renderer exited with status 0, MuJoCo continued publishing and reported its exit.
These are local measurements, not a 30 FPS guarantee.

A separate **75-second preview-window trial**, with the MuJoCo operation window
enabled and SONIC inactive, measured 12.51 / 12.50 FPS, 70.49 / 80.50 ms mean
sample-to-receive delay, 938 equal-timestamp image pairs and 2354 MiB total GPU
memory. DDS measured 200.00 Hz; its maximum intervals during rendering, a
one-second renderer pause, resume and renderer exit were respectively 20.811,
7.976, 17.376 and 29.254 ms. This trial and the controller-active headless trial
have different GPU workloads, so their frame rates do not isolate UI overhead.

The final alignment check covered 20 root/joint poses, all 44 MuJoCo links and
nine extra USD fixed links. Maximum measured errors were below 0.000001 mm in
position, 0.000005 degrees in rotation and 0.000102 pixels for numerical camera
projection. Four changing camera poses with three actual RGB markers each had a
maximum RTX/MuJoCo centroid difference of **0.279 pixels**. The same check
confirmed **zero enabled Isaac rigid bodies and timeline time 0** after capture.

The existing camera viewer displayed the Isaac stream. The existing VLA
observation constructor consumed the live camera and C++ `g1_debug` state and
produced a `uint8` video tensor of shape `(1, 1, 480, 640, 3)`, without loading VLA
weights. The exporter uses the same validated camera reader; a complete LeRobot
dataset export was not run. The default MuJoCo process was checked and imported
no `isaacsim`, `omni` or `pxr` modules. The process/protocol integration tests also
passed RGB round trips, idle-frame suppression, latest-state resume and cleanup.

Detailed local artifacts are under `outputs/isaac_setup/`, including
`final-headless-result.json`, `sonic-headless-session.log`,
`working-camera-viewer.log`, `live-observation-result.log`,
`final-rendered-alignment.log` and package snapshots. That directory is ignored
by Git.

Pure Isaac mode was also checked for three 640×480 RGB frames with increasing
timestamps. Sending SIGTERM to the simulator while its Isaac worker was active
produced parent exit status 0, reaped the worker and removed its IPC/model
directory. Closing only the preview window produced worker exit 0 and left
MuJoCo running. An intentionally missing model produced a useful renderer error
and exit status **1**; Kit's fast shutdown did not mask that failure as success.
The final two process/protocol integration tests passed.

The runtime CLI conflict was fixed with `tyro==0.9.13` and `typeguard==4.4.2`.
`pip check` still reports the following pre-existing/newly intersecting metadata
constraints from sharing the simulation environment with Isaac's exact pins:
NumPy 1.26.4 vs 1.26.0, MuJoCo's websockets requirement vs 12.0,
huggingface-hub's click requirement vs 8.1.7, httpx2's idna requirement vs 3.10,
and wheel's packaging requirement vs 23.0. The simulation/camera paths above were
tested with the installed versions; these unrelated packages were not removed
or broadly upgraded to hide the conflicts.
