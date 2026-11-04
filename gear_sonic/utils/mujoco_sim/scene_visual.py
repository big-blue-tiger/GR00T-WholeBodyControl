"""MuJoCo task poses with original USD object materials and disabled USD physics."""

from pathlib import Path

import mujoco
import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics


class SceneVisual:
    def __init__(self, stage, model, show_colliders=True, object_asset_path=None):
        from .isaac_visual import world_matrix

        self.model, self.ops = model, {}
        self.world_matrix = world_matrix
        self.object_body_id = None
        if object_asset_path is not None:
            self._reference_object(stage, object_asset_path)
        for i in range(model.ngeom):
            name = model.geom(i).name
            body_name = model.body(int(model.geom_bodyid[i])).name
            if not body_name.startswith("grail_") or name.endswith("_collision"):
                continue
            if body_name == "grail_object":
                # Use the authored USD mesh/UVs/OmniPBR, without also drawing
                # the converted MuJoCo mesh at the same location.
                continue
            path = "/Task/" + name
            kind = model.geom_type[i]
            if kind == mujoco.mjtGeom.mjGEOM_BOX:
                geom = UsdGeom.Cube.Define(stage, path)
                geom.CreateSizeAttr(2.0)
            elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                geom = UsdGeom.Sphere.Define(stage, path)
                geom.CreateRadiusAttr(float(model.geom_size[i, 0]))
            else:
                raise ValueError(f"Unsupported task geometry {name}")
            geom.CreateDisplayColorAttr([Gf.Vec3f(*map(float, model.geom_rgba[i, :3]))])
            if name == "grail_table_geom" and not show_colliders:
                geom.MakeInvisible()
            transform = UsdGeom.Xformable(geom).AddTransformOp(UsdGeom.XformOp.PrecisionDouble)
            self.ops[i] = transform

    def _reference_object(self, stage, asset_path):
        asset = Path(asset_path).resolve()
        if not asset.is_file():
            raise FileNotFoundError(f"Isaac object USD does not exist: {asset}")
        source = Usd.Stage.Open(str(asset))
        if not source or not source.GetDefaultPrim():
            raise ValueError(f"Object USD requires a default prim: {asset}")
        if UsdGeom.GetStageUpAxis(source) != UsdGeom.Tokens.z:
            raise ValueError("Expected the prepared object's Z-up USD")
        self.object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "grail_object")
        if self.object_body_id < 0:
            raise ValueError("Object USD requires the MuJoCo grail_object body")
        self.object_unit = UsdGeom.GetStageMetersPerUnit(source)
        root = UsdGeom.Xform.Define(stage, "/Task/grail_object")
        self.object_op = root.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)
        # Keep the source root transform below the pose, just as the converter
        # bakes it into mesh vertices. A reference preserves relative texture
        # resolution, shader inputs, bindings and indexed UVs in their layers.
        # In particular, MuJoCo flips OBJ V coordinates at compile time; those
        # compiled UVs cannot be used unchanged by a USD texture shader.
        prim = stage.DefinePrim("/Task/grail_object/Asset")
        if not prim.GetReferences().AddReference(str(asset)):
            raise ValueError(f"Could not reference object USD: {asset}")
        for child in Usd.PrimRange(prim):
            if child.IsInstance():
                child.SetInstanceable(False)
        for child in Usd.PrimRange(prim):
            if child.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(child).CreateRigidBodyEnabledAttr(False)
            if child.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(child).CreateCollisionEnabledAttr(False)
            if child.HasAPI(UsdPhysics.ArticulationRootAPI):
                child.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            if child.IsA(UsdPhysics.Joint):
                child.SetActive(False)

    def update(self, data):
        if self.object_body_id is not None:
            index = self.object_body_id
            matrix = self.world_matrix(data.xpos[index], data.xmat[index])
            matrix[:3, :3] *= self.object_unit
            self.object_op.Set(Gf.Matrix4d(matrix.T.tolist()))
        for index, op in self.ops.items():
            matrix = self.world_matrix(data.geom_xpos[index], data.geom_xmat[index])
            if self.model.geom_type[index] == mujoco.mjtGeom.mjGEOM_BOX:
                matrix[:3, :3] = matrix[:3, :3] @ np.diag(self.model.geom_size[index])
            op.Set(Gf.Matrix4d(matrix.T.tolist()))


def add_gaussian_scene(stage, config):
    """Reference the visual background and return its render settings."""
    from .isaac_visual import world_matrix

    asset = Path(config["asset_path"])
    if not asset.is_file():
        raise FileNotFoundError(f"Isaac 3DGS background does not exist: {asset}")
    source = Usd.Stage.Open(str(asset))
    if source is None or not source.GetDefaultPrim():
        raise ValueError(f"3DGS USDZ requires a default prim: {asset}")
    if not np.isclose(UsdGeom.GetStageMetersPerUnit(source), 1.0):
        raise ValueError("3DGS background must use metres; automatic scaling is disabled")
    root = UsdGeom.Xform.Define(stage, "/GaussianScene")
    if not root.GetPrim().GetReferences().AddReference(str(asset)):
        raise ValueError(f"Could not reference 3DGS background: {asset}")
    # Gf row-vector convention. The referenced child's authored transform
    # composes underneath this placement; there is deliberately no scale op.
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, np.asarray(config["quaternion_wxyz"], dtype=float))
    matrix = world_matrix(config["translation"], rotation)
    root.ClearXformOpOrder()
    root.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Matrix4d(matrix.T.tolist())
    )
    # Layer metadata does not compose through a prim reference. Preserve the
    # scan's tonemapping/compositing settings explicitly in the render app.
    return source.GetRootLayer().customLayerData.get("renderSettings", {})
