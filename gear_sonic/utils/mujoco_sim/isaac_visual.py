"""USD visual puppet driven exclusively by MuJoCo forward kinematics.

Synchronization/render settings follow SIMPLE's split engine design (MIT,
Copyright 2025-2026 SIMPLE Authors); this module has no SIMPLE/Isaac Lab dependency.
Import only after SimulationApp has been created in the render worker.
"""

from pathlib import Path
import time

import mujoco
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

from .scene_visual import SceneVisual, add_gaussian_scene, add_occlusion_proxy


ROBOT_SUBTREE = "/g1_29dof_with_hand_rev_1_0"
EGO_SIZE = (640, 480)


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


def _centered_capture(focal_px, principal_offset_px):
    """Return the centered frustum and output-to-capture pixel transform."""
    size = np.asarray(EGO_SIZE)
    focal = float(max(focal_px))
    scale = focal / focal_px
    # Cover the offset window at native pixel density, with dimensions aligned
    # to 32 pixels. Use square pixels so Hydra need not conform the aperture.
    capture = 32 * np.ceil(scale * (size + 2 * np.abs(principal_offset_px)) / 32).astype(int)
    principal = (size - 1) / 2 + principal_offset_px
    transform = np.column_stack((np.diag(scale), (capture - 1) / 2 - scale * principal))
    return (int(capture[0]), int(capture[1])), focal, transform


class EgoCamera:
    def __init__(self, stage, model, *, centered_camera=False):
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        self.camera = UsdGeom.Camera.Define(stage, "/EgoCamera")
        self.camera_op = self.camera.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)
        self.camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
        if self.camera_id < 0:
            raise ValueError("MuJoCo model must contain head_camera")
        if model.cam_projection[self.camera_id] != 0:
            raise ValueError("This G1 renderer expects a perspective head_camera")
        focal = 20.0  # USD lens units cancel in the aperture/focal ratio
        self.camera.CreateFocalLengthAttr(focal)
        self.capture_size = EGO_SIZE
        self.output_to_capture = None
        sensor = model.cam_sensorsize[self.camera_id]
        if sensor.any():
            fx, fy, offset_x, offset_y = model.cam_intrinsic[self.camera_id]
            if centered_camera:
                # Keep the NuRec frustum symmetric. Restore the calibrated
                # principal point after composing both layers.
                pixels_per_unit = np.asarray(EGO_SIZE) / sensor
                focal_px = pixels_per_unit * (fx, fy)
                principal_offset_px = pixels_per_unit * (-offset_x, -offset_y)
                self.capture_size, capture_focal, self.output_to_capture = _centered_capture(
                    focal_px, principal_offset_px,
                )
                horizontal, vertical = focal / capture_focal * np.asarray(self.capture_size)
            else:
                horizontal, vertical = focal * sensor / (fx, fy)
                self.camera.CreateHorizontalApertureOffsetAttr(float(focal * offset_x / fx))
                self.camera.CreateVerticalApertureOffsetAttr(float(-focal * offset_y / fy))
        else:
            vertical = 2.0 * focal * np.tan(np.deg2rad(model.cam_fovy[self.camera_id]) / 2.0)
            horizontal = vertical * EGO_SIZE[0] / EGO_SIZE[1]
        self.camera.CreateVerticalApertureAttr(float(vertical))
        self.camera.CreateHorizontalApertureAttr(float(horizontal))
        self.camera.CreateClippingRangeAttr(Gf.Vec2f(
            float(model.vis.map.znear * model.stat.extent),
            float(model.vis.map.zfar * model.stat.extent),
        ))
        self.camera.CreateProjectionAttr(UsdGeom.Tokens.perspective)

    def set_pose(self, position, rotation):
        # Both native USD and MuJoCo cameras look along -Z with +Y up.
        matrix = world_matrix(position, rotation)
        self.camera_op.Set(Gf.Matrix4d(matrix.T.tolist()))

    def update(self, data):
        self.set_pose(data.cam_xpos[self.camera_id], data.cam_xmat[self.camera_id])


class VisualPuppet(EgoCamera):
    """Pure USD setup/pose mapping, also usable for numerical alignment checks."""

    def __init__(self, stage, model, asset_path=None, *, centered_camera=False):
        super().__init__(stage, model, centered_camera=centered_camera)
        self.stage = stage
        self.model = model
        if asset_path is None:
            asset_path = Path(__file__).resolve().parents[2] / "data/robots/g1/g1_43dof_s3.usda"
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
        super().update(data)
        self.transforms = transforms


class IsaacVisual:
    def __init__(self, app, model, texture_dir, scene_config=None,
                 object_asset_path=None, *, layer="combined"):
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
        has_gaussians = bool(scene_config and scene_config.get("enabled", False))
        camera_type = EgoCamera if layer == "background" else VisualPuppet
        self.puppet = camera_type(stage, model, centered_camera=has_gaussians)
        if has_gaussians and layer != "foreground":
            for name, value in add_gaussian_scene(stage, scene_config).items():
                app.set_setting("/" + name.replace(":", "/"), value)
        initial = mujoco.MjData(model)
        mujoco.mj_forward(model, initial)
        self.scene = None
        if layer != "background":
            show_colliders = not has_gaussians or scene_config.get("show_colliders", False)
            self.scene = SceneVisual(stage, model, show_colliders=show_colliders,
                                     object_asset_path=object_asset_path)
            if layer == "combined":
                self._add_ground(stage, model, initial, texture_dir, visible=show_colliders)
        if layer == "foreground":
            add_occlusion_proxy(stage, scene_config)
        light = UsdLux.DomeLight.Define(stage, "/DomeLight")
        light.CreateIntensityAttr(1000)
        self.puppet.update(initial)
        rep.orchestrator.set_capture_on_play(False)
        self.product = rep.create.render_product("/EgoCamera", self.puppet.capture_size)
        self.product_path = self.product.path
        self.rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb.attach([self.product_path])
        self.ids = None
        if layer == "foreground":
            self.ids = rep.AnnotatorRegistry.get_annotator(
                "instance_id_segmentation_fast", init_params={"colorize": False},
            )
            self.ids.attach([self.product_path])
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
        self.submit_s = self.capture_s = self.readback_s = 0.0

    def _add_ground(self, stage, model, initial, texture_dir, *, visible):
        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor_id < 0 or model.geom_type[floor_id] != mujoco.mjtGeom.mjGEOM_PLANE:
            raise ValueError("Expected the current G1 floor plane")
        floor = UsdGeom.Mesh.Define(stage, "/Ground")
        floor.CreatePointsAttr([(-100, -100, 0), (100, -100, 0), (100, 100, 0), (-100, 100, 0)])
        floor.CreateFaceVertexCountsAttr([4])
        floor.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        floor.CreateDisplayColorAttr([Gf.Vec3f(0.2, 0.3, 0.4)])
        if not visible:
            floor.MakeInvisible()
        self._ground_material(floor, model, floor_id, texture_dir)
        floor.AddTransformOp().Set(Gf.Matrix4d(world_matrix(
            initial.geom_xpos[floor_id], initial.geom_xmat[floor_id]).T.tolist()))

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

    def capture_rgb(self, *, warming_up=False):
        self.app.set_setting("/rtx/ecoMode/enabled", False)
        # Synchronous capture prevents annotator data from belonging to the prior pose.
        # Timeline is stopped and delta_time=0: no Isaac physics is advanced.
        self.rep.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=True)
        self.app.set_setting("/rtx/ecoMode/enabled", False)
        captured = time.perf_counter()
        rgba = np.asarray(self.rgb.get_data())
        if rgba.ndim != 3 or rgba.shape[:2] != self.puppet.capture_size[::-1] or rgba.shape[2] < 3:
            if warming_up:
                return None
            raise RuntimeError(f"Isaac returned an invalid camera frame: {rgba.shape}")
        return np.ascontiguousarray(rgba[:, :, :3]), captured

    def render(self, data, *, warming_up=False, background=None):
        started = time.perf_counter()
        with Sdf.ChangeBlock():
            self.puppet.update(data)
            if self.scene is not None:
                self.scene.update(data)
        submitted = time.perf_counter()
        capture = self.capture_rgb(warming_up=warming_up)
        if capture is None:
            return None
        rgb, captured = capture
        if self.ids is not None:
            segmentation = self.ids.get_data()
            ids = np.asarray(segmentation["data"]).squeeze()
            if ids.shape != rgb.shape[:2]:
                if warming_up:
                    return None
                raise RuntimeError(f"Isaac returned an invalid foreground mask: {ids.shape}")
            if not warming_up:
                # Only the robot and task object supply color. The tabletop
                # participates in the depth test through the same render pass.
                visible_ids = [int(key) for key, path in segmentation["info"]["idToLabels"].items()
                               if path.startswith(("/Robot/", "/Task/grail_object/"))]
                mesh_visible = np.isin(ids, visible_ids)
                rgb = np.where(mesh_visible[..., None], rgb, background())
        if self.puppet.output_to_capture is not None:
            import cv2

            rgb = cv2.warpAffine(rgb, self.puppet.output_to_capture, EGO_SIZE,
                                 flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
        if not warming_up:
            self.submit_s += submitted - started
            self.capture_s += captured - submitted
            self.readback_s += time.perf_counter() - captured
        return rgb

    def close(self):
        # Isaac 5.0 attach accepts HydraTexture objects; detach expects USD paths.
        if self.ids is not None:
            self.ids.detach([self.product_path])
        self.rgb.detach([self.product_path])
        if self.product is not None:
            self.product.destroy()
