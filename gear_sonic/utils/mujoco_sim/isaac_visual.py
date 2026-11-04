"""USD visual puppet driven exclusively by MuJoCo forward kinematics.

Synchronization/render settings follow SIMPLE's split engine design (MIT,
Copyright 2025-2026 SIMPLE Authors); this module has no SIMPLE/Isaac Lab dependency.
Import only after SimulationApp has been created in the render worker.
"""

from pathlib import Path

import mujoco
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

from .scene_visual import SceneVisual, add_gaussian_scene


ROBOT_SUBTREE = "/g1_29dof_with_hand_rev_1_0"


def world_matrix(position, rotation):
    """Column-vector SE(3); USD/Gf uses its transpose (row-vector convention)."""
    matrix = np.eye(4)
    matrix[:3, :3] = np.asarray(rotation).reshape(3, 3)
    matrix[:3, 3] = position
    return matrix


def _joint_frame(joint, side):
    position = getattr(joint, f"GetLocalPos{side}Attr")().Get()
    quaternion = getattr(joint, f"GetLocalRot{side}Attr")().Get()
    q = np.array([quaternion.GetReal(), *quaternion.GetImaginary()], dtype=float)
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, q)
    return world_matrix(position, rotation)


class VisualPuppet:
    """Pure USD setup/pose mapping, also usable for numerical alignment checks."""

    def __init__(self, stage, model, asset_path=None):
        self.stage = stage
        self.model = model
        if asset_path is None:
            asset_path = Path(__file__).resolve().parents[2] / "data/robots/g1/g1_43dof_s3.usda"
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        root = stage.DefinePrim("/Robot", "Xform")
        # Explicit reference excludes /SimpleRoom and /physicsScene from the asset.
        root.GetReferences().AddReference(str(asset_path), ROBOT_SUBTREE)
        for prim in Usd.PrimRange(root):
            if prim.IsInstance():
                prim.SetInstanceable(False)
        prims = list(Usd.PrimRange(root))
        links = {p.GetName(): p for p in root.GetChildren() if p.IsA(UsdGeom.Xform)}
        self.body_paths = {}
        robot_root = model.body("pelvis").id
        for index in range(1, model.nbody):
            ancestor = index
            while ancestor and ancestor != robot_root:
                ancestor = model.body_parentid[ancestor]
            if ancestor != robot_root:
                continue
            name = model.body(index).name
            if name not in links:
                raise ValueError(f"MuJoCo body {name!r} has no USD link")
            self.body_paths[str(links[name].GetPath())] = index

        # USD links are siblings, so fixed parts need explicit joint-frame following.
        fixed = []
        for prim in prims:
            if prim.IsA(UsdPhysics.FixedJoint):
                joint = UsdPhysics.FixedJoint(prim)
                parent = joint.GetBody0Rel().GetTargets()
                child = joint.GetBody1Rel().GetTargets()
                if parent and child:
                    offset = _joint_frame(joint, 0) @ np.linalg.inv(_joint_frame(joint, 1))
                    fixed.append((str(parent[0]), str(child[0]), offset))
        self.fixed = []
        known = set(self.body_paths)
        pending = [entry for entry in fixed if entry[1] not in known]
        while pending:
            ready = [entry for entry in pending if entry[0] in known]
            if not ready:
                raise ValueError(f"Unmapped fixed USD parts: {pending}")
            for entry in ready:
                self.fixed.append(entry)
                known.add(entry[1])
                pending.remove(entry)
        unmapped = [str(p.GetPath()) for p in links.values() if str(p.GetPath()) not in known]
        if unmapped:
            raise ValueError(f"Unmapped USD links: {unmapped}")

        self.ops = {}
        for path in known:
            xform = UsdGeom.Xformable(stage.GetPrimAtPath(path))
            xform.ClearXformOpOrder()
            op = xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, "mujoco")
            xform.SetResetXformStack(True)
            self.ops[path] = op

        for prim in prims:
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
                # GRAIL shares finger /visuals meshes with collision geometry.
                # Disable their physics without hiding the visible fingers.
                # Only dedicated /collisions* shapes are rendering helpers.
                if prim.GetName().startswith("collisions"):
                    UsdGeom.Imageable(prim).MakeInvisible()
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            if prim.IsA(UsdPhysics.Joint):
                UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
                prim.SetActive(False)  # disables drive APIs as well
        self._local_materials([prim for prim in prims if prim])

        self.camera = UsdGeom.Camera.Define(stage, "/EgoCamera")
        self.camera_op = self.camera.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)
        self.camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
        if self.camera_id < 0:
            raise ValueError("MuJoCo model must contain head_camera")
        projection = getattr(model, "cam_projection", getattr(model, "cam_orthographic", None))
        if (projection is not None and projection[self.camera_id] != 0) or np.any(model.cam_sensorsize[self.camera_id]):
            raise ValueError("This G1 renderer expects head_camera's perspective fovy projection")
        focal = 20.0  # USD lens units cancel in the aperture/focal ratio
        vertical = 2.0 * focal * np.tan(np.deg2rad(model.cam_fovy[self.camera_id]) / 2.0)
        self.camera.CreateFocalLengthAttr(focal)
        self.camera.CreateVerticalApertureAttr(float(vertical))
        self.camera.CreateHorizontalApertureAttr(float(vertical * 640 / 480))
        self.camera.CreateClippingRangeAttr(Gf.Vec2f(
            float(model.vis.map.znear * model.stat.extent),
            float(model.vis.map.zfar * model.stat.extent),
        ))
        self.camera.CreateProjectionAttr(UsdGeom.Tokens.perspective)
        self.transforms = {}

    def _local_materials(self, prims):
        # Rebind before deactivating Looks; no remote MDL or texture access at runtime.
        materials = {}
        for prim in prims:
            if not prim.IsA(UsdGeom.Gprim):
                continue
            bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            name = bound.GetPrim().GetName() if bound else "default"
            if name not in materials:
                lower = name.lower()
                color = (0.08, 0.08, 0.09) if any(x in lower for x in ("charcoal", "carpaint", "defaultmaterial_1")) else (0.65, 0.67, 0.7)
                material = UsdShade.Material.Define(self.stage, f"/LocalLooks/{name}")
                shader = UsdShade.Shader.Define(self.stage, f"/LocalLooks/{name}/Shader")
                shader.CreateIdAttr("UsdPreviewSurface")
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
                shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
                material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
                materials[name] = material
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(materials[name])
        self.stage.GetPrimAtPath("/Robot/Looks").SetActive(False)

    def update(self, data):
        transforms = {
            path: world_matrix(data.xpos[index], data.xmat[index])
            for path, index in self.body_paths.items()
        }
        for parent, child, offset in self.fixed:
            transforms[child] = transforms[parent] @ offset
        for path, matrix in transforms.items():
            self.ops[path].Set(Gf.Matrix4d(matrix.T.tolist()))
        # Both native USD and MuJoCo cameras look along -Z with +Y up.
        # Only transpose the matrix; no ROS/OpenCV axis correction is needed.
        matrix = world_matrix(data.cam_xpos[self.camera_id], data.cam_xmat[self.camera_id])
        self.camera_op.Set(Gf.Matrix4d(matrix.T.tolist()))
        self.transforms = transforms


class IsaacVisual:
    def __init__(self, app, model, texture_dir, scene_config=None,
                 object_asset_path=None):
        import omni.replicator.core as rep
        from omni.kit.viewport.utility import get_active_viewport_window
        import omni.timeline
        import omni.usd

        self.app = app
        self.rep = rep
        # The minimal renderer experience omits isaacsim.core.throttling, which
        # would re-enable eco mode whenever this stopped timeline is paused.
        omni.timeline.get_timeline_interface().stop()
        app.set_setting("/rtx/ecoMode/enabled", False)
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        self.puppet = VisualPuppet(stage, model)
        show_colliders = True
        if scene_config and scene_config.get("enabled", False):
            for name, value in add_gaussian_scene(stage, scene_config).items():
                app.set_setting("/" + name.replace(":", "/"), value)
            show_colliders = scene_config.get("show_colliders", False)
        self.scene = SceneVisual(stage, model, show_colliders=show_colliders,
                                 object_asset_path=object_asset_path)
        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor_id < 0 or model.geom_type[floor_id] != mujoco.mjtGeom.mjGEOM_PLANE:
            raise ValueError("Expected the current G1 floor plane")
        # Large visual plane at the MuJoCo floor transform; no collision/rigid body.
        floor = UsdGeom.Mesh.Define(stage, "/Ground")
        floor.CreatePointsAttr([(-100, -100, 0), (100, -100, 0), (100, 100, 0), (-100, 100, 0)])
        floor.CreateFaceVertexCountsAttr([4])
        floor.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        floor.CreateDisplayColorAttr([Gf.Vec3f(0.2, 0.3, 0.4)])
        if not show_colliders:
            floor.MakeInvisible()
        self._ground_material(floor, model, floor_id, texture_dir)
        initial = mujoco.MjData(model)
        mujoco.mj_forward(model, initial)
        floor.AddTransformOp().Set(Gf.Matrix4d(world_matrix(
            initial.geom_xpos[floor_id], initial.geom_xmat[floor_id]).T.tolist()))
        light = UsdLux.DomeLight.Define(stage, "/DomeLight")
        light.CreateIntensityAttr(1000)
        self.puppet.update(initial)
        rep.orchestrator.set_capture_on_play(False)
        self.product = rep.create.render_product("/EgoCamera", (640, 480))
        self.product_path = self.product.path
        self.rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb.attach([self.product_path])
        # Creating the stage re-enables Kit's default viewport after the app
        # disabled it. It renders an unused camera over the entire 3DGS room.
        # Only the dedicated /EgoCamera render product is needed, including
        # for the OpenCV preview. Destroy the unused window: merely disabling
        # updates is undone by its pending dock/focus callbacks during warm-up.
        viewport_window = get_active_viewport_window()
        if viewport_window is not None:
            viewport_window.destroy()
        # With the unused viewport disabled and presentation off, Hydra can
        # finish this frame before annotator readback without a GUI swapchain
        # wait. Keep Replicator's current-state capture barrier below as well.
        app.set_setting("/app/hydraEngine/waitIdle", True)
        # Isaac 5.0's throttling extension enables eco mode on STOP/PAUSE.
        # This puppet deliberately keeps time stopped, but must still render.
        app.set_setting("/rtx/ecoMode/enabled", False)
        # Compile shaders before subscribing to physics snapshots. The first RTX
        # capture can take a minute on a cold cache; never publish that old pose.
        print("[Render] Warming up RTX camera; physics/DDS continue.", flush=True)
        # A newly attached RGB annotator can return an empty first frame.
        # Warm only during initialization; an invalid runtime frame still fails.
        for _ in range(10):
            if self.render(initial, warming_up=True) is not None:
                break
        else:
            raise RuntimeError("Isaac RGB annotator did not initialize in 10 captures")

    def _ground_material(self, floor, model, floor_id, texture_dir):
        """Reuse the compiled MuJoCo checker pixels and repeats per metre."""
        import cv2

        material_id = int(model.geom_matid[floor_id])
        if material_id < 0:
            return
        texture_id = int(model.mat_texid[material_id, 1])  # RGB texture role
        if texture_id < 0:
            return
        height, width = int(model.tex_height[texture_id]), int(model.tex_width[texture_id])
        channels = int(model.tex_nchannel[texture_id])
        offset = int(model.tex_adr[texture_id])
        pixels = model.tex_data[offset:offset + height * width * channels].reshape(height, width, channels)
        path = Path(texture_dir) / "ground.png"
        if not cv2.imwrite(str(path), pixels[:, :, :3][:, :, ::-1]):
            raise RuntimeError("Could not write local ground texture")
        repeat = model.mat_texrepeat[material_id]
        if not model.mat_texuniform[material_id]:
            raise ValueError("Expected the current G1 ground's uniform texture mapping")
        uv = [(float(x * repeat[0]), float(y * repeat[1]))
              for x, y in ((-100, -100), (100, -100), (100, 100), (-100, 100))]
        UsdGeom.PrimvarsAPI(floor).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
        ).Set(uv)
        stage = floor.GetPrim().GetStage()
        material = UsdShade.Material.Define(stage, "/LocalLooks/Ground")
        reader = UsdShade.Shader.Define(stage, "/LocalLooks/Ground/UV")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        texture = UsdShade.Shader.Define(stage, "/LocalLooks/Ground/Texture")
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(path)))
        texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
        texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        surface = UsdShade.Shader.Define(stage, "/LocalLooks/Ground/Surface")
        surface.CreateIdAttr("UsdPreviewSurface")
        surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")
        surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
        material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(floor.GetPrim()).Bind(material)

    def render(self, data, *, warming_up=False):
        self.app.set_setting("/rtx/ecoMode/enabled", False)
        self.puppet.update(data)
        self.scene.update(data)
        # Synchronous capture prevents annotator data from belonging to the prior pose.
        # Timeline is stopped and delta_time=0: no Isaac physics is advanced.
        self.rep.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=True)
        self.app.set_setting("/rtx/ecoMode/enabled", False)
        rgba = np.asarray(self.rgb.get_data())
        if rgba.shape[:2] != (480, 640) or rgba.ndim != 3:
            if warming_up:
                return None
            raise RuntimeError(f"Isaac returned an invalid camera frame: {rgba.shape}")
        return np.ascontiguousarray(rgba[:, :, :3])

    def close(self):
        # Isaac 5.0 attach accepts HydraTexture objects; detach expects USD paths.
        self.rgb.detach([self.product_path])
        if self.product is not None:
            self.product.destroy()
