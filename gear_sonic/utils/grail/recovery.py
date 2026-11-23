"""Generate reachable, collision-checked starts for grasp recovery."""

import mujoco
import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.spatial.transform import Rotation
import yaml

from .checkpoints import resolve_checkpoint
from .policy import finger_primitive_targets


MIN_CLEARANCE = 0.05
MAX_CLEARANCE = 0.25
MAX_SAMPLES = 64


class RecoveryInitializer:
    def __init__(self, model, config, checkpoint):
        self.model = model
        self.pelvis = model.body("pelvis").id
        self.palm = model.body("right_hand_palm_link").id
        self.object = model.body("grail_object").id
        self.table = model.body("grail_table").id
        self.hand_frames = [model.body(name).id for name in config["hand_frame_links"]]
        arm_joints = [
            model.joint(name).id for name in config["robot"]["initial_body_joint_names"]
            if name.startswith(("right_shoulder_", "right_elbow_", "right_wrist_"))
        ]
        self.arm_qadr = model.jnt_qposadr[arm_joints]
        self.arm_lower, self.arm_upper = model.jnt_range[arm_joints].T
        robot_root = model.body_rootid[self.pelvis]
        self.robot_bodies = model.body_rootid == robot_root
        self.hand_geoms, self.object_geoms = [], []
        for geom, body in enumerate(model.geom_bodyid):
            if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
                continue
            if body == self.object:
                self.object_geoms.append(geom)
            elif model.body(body).name.startswith("right_hand_"):
                self.hand_geoms.append(geom)
        if not self.hand_geoms or not self.object_geoms:
            raise ValueError("Recovery requires right-hand and object collision geometry")

        path = resolve_checkpoint(checkpoint)
        policy_config = yaml.safe_load(path.with_name("config.yaml").read_text())
        self.finger_poses = []
        for closed in (False, True):
            targets = finger_primitive_targets(policy_config, "right", closed)
            qadr = [model.jnt_qposadr[model.joint(name).id] for name in targets]
            self.finger_poses.append((qadr, list(targets.values())))

    def clearance(self, data):
        """Minimum distance between the right-hand and object collision surfaces."""
        return min(
            mujoco.mj_geomDistance(self.model, data, hand, obj, 1.0, None)
            for hand in self.hand_geoms for obj in self.object_geoms
        )

    def _has_collision(self, data):
        for contact in data.contact:
            geom1, geom2 = contact.geom
            if min(geom1, geom2) < 0:
                if any(self.robot_bodies[self.model.geom_bodyid[g]] for g in contact.geom if g >= 0):
                    if contact.dist < -0.001:
                        return True
                continue
            a, b = self.model.geom_bodyid[[geom1, geom2]]
            if not (self.robot_bodies[a] or self.robot_bodies[b]):
                continue
            if self.object in (a, b) and contact.dist <= 0:
                return True
            if self.table in (a, b) and contact.dist < -0.001:
                return True
            # The reference stance has small floor overlaps. Adjacent robot
            # colliders can also overlap slightly in the original MJCF.
            if self.robot_bodies[a] and self.robot_bodies[b] and contact.dist < -0.003:
                return True
        return False

    def apply(self, data, rng):
        """Place the empty hand around or above the object using scratch data."""
        original = data.qpos.copy()
        forward = data.xmat[self.pelvis].reshape(3, 3)[:2, 0].copy()
        forward /= np.linalg.norm(forward)
        left = np.array([-forward[1], forward[0]])
        for attempt in range(1, MAX_SAMPLES + 1):
            azimuth = rng.uniform(-np.pi, np.pi)
            z = rng.uniform(0, 1)
            xy = np.sqrt(1 - z * z)
            direction = np.array([xy * np.cos(azimuth), xy * np.sin(azimuth), z])
            distance = rng.uniform(MIN_CLEARANCE, MAX_CLEARANCE)
            closed = bool(rng.integers(2))
            details = self._try_pose(data, original, forward, left, direction, distance, closed, rng)
            if details is not None:
                return dict(details, direction_world=direction.tolist(), sampling_attempts=attempt)
        data.qpos[:] = original
        mujoco.mj_forward(self.model, data)
        raise RuntimeError(f"Could not sample a collision-free recovery pose in {MAX_SAMPLES} attempts")

    def _try_pose(self, data, original, forward, left, direction, distance, closed, rng):
        baseline = original.copy()
        finger_qadr, finger_pose = self.finger_poses[closed]
        baseline[finger_qadr] = finger_pose
        data.qpos[:] = baseline
        mujoco.mj_forward(self.model, data)
        orientation = data.xmat[self.palm].reshape(3, 3).copy()
        nominal = baseline[self.arm_qadr].copy()
        object_position = data.xpos[self.object].copy()
        # Reference horizontal misses slightly above the object's center so
        # the hand can approach from the sides without clipping the table.
        anchor = data.xipos[self.object] + [0, 0, 0.045]
        center = data.xpos[self.hand_frames].mean(axis=0)

        def rigid_gap(radius):
            data.qpos[:3] = baseline[:3] + anchor + direction * radius - center
            mujoco.mj_kinematics(self.model, data)
            return self.clearance(data) - distance

        # Translate the hand geometry rigidly along the sampled ray to find a
        # target whose actual surface gap matches the sampled distance.
        previous = 0.45
        if rigid_gap(previous) <= 0:
            return None
        for radius in np.linspace(previous, 0, 91)[1:]:
            if rigid_gap(radius) <= 0:
                target = anchor + direction * brentq(rigid_gap, radius, previous, xtol=1e-7)
                break
            previous = radius
        else:
            return None

        def residual(joints):
            data.qpos[self.arm_qadr] = joints
            mujoco.mj_kinematics(self.model, data)
            position_error = data.xpos[self.hand_frames].mean(axis=0) - target
            palm_orientation = data.xmat[self.palm].reshape(3, 3)
            rotation_error = Rotation.from_matrix(orientation @ palm_orientation.T).as_rotvec()
            return np.r_[position_error, 0.12 * rotation_error, 0.0002 * (joints - nominal)]

        # Follow the hand target to keep it within reach on wider misses. A
        # minimum distance from the object towards the robot keeps the base
        # from following targets too far over the table.
        seed = np.clip(nominal, self.arm_lower, self.arm_upper)
        target_offset = target[:2] - object_position[:2]
        forward_offset = target_offset @ forward
        lateral_offset = target_offset @ left
        for stand_off in rng.uniform(0.28, 0.50, size=5):
            data.qpos[:] = baseline
            base_distance = max(0.28, stand_off - forward_offset)
            data.qpos[:2] = object_position[:2] - forward * base_distance + left * (0.18 + lateral_offset)
            solution = least_squares(
                residual, seed, bounds=(self.arm_lower, self.arm_upper),
                max_nfev=120, ftol=1e-9, xtol=1e-9, gtol=1e-9,
            )
            error = np.linalg.norm(residual(solution.x)[:3])
            if error >= 0.001:
                continue
            mujoco.mj_forward(self.model, data)
            clearance = self.clearance(data)
            if (abs(clearance - distance) < 0.001
                    and MIN_CLEARANCE <= clearance <= MAX_CLEARANCE and not self._has_collision(data)):
                hand_position = data.xpos[self.hand_frames].mean(axis=0)
                return dict(
                    hand="closed" if closed else "open", requested_clearance_m=distance,
                    clearance_m=clearance, hand_position=hand_position.tolist(),
                    hand_offset_world=(hand_position - object_position).tolist(),
                    robot_position=data.qpos[:3].tolist(),
                )
        return None
