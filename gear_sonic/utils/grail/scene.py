"""Build the G1 model with the task scene and a free RoboCasa body."""

from pathlib import Path
import re
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from .assets import load_assets


def numbers(values):
    return " ".join(str(float(x)) for x in values)


def load_scene_config(path):
    """Load scene parameters independently of the selected object asset."""
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg["version"] != 1:
        raise ValueError("Scene config version must be 1")
    table = cfg["table"]
    if table["size"] != [2.0, 0.6, 0.04] or cfg["object"]["mass"] != 0.1:
        raise ValueError("This demo requires a 2x0.6x0.04 m table frame and a 0.1 kg object")
    if "collision_bounds_xy" in table:
        bounds = np.asarray(table["collision_bounds_xy"], dtype=float)
        if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
            raise ValueError("table.collision_bounds_xy requires increasing XY bounds")
        if not np.isclose(bounds[0, 1], -table["size"][1] / 2):
            raise ValueError("The tabletop collider must start at the observed front edge")
    if (
        len(cfg["robot"]["observation_joint_names"]) != 43
        or len(set(cfg["robot"]["observation_joint_names"])) != 43
    ):
        raise ValueError("Expected 43 unique named robot DOFs")
    for quaternion in (cfg["table"]["quaternion_wxyz"], cfg["object"]["quaternion_wxyz"]):
        if not np.isclose(np.linalg.norm(quaternion), 1, atol=1e-3):
            raise ValueError("Expected unit wxyz quaternion")
    if "ego_camera" in cfg:
        fov = float(cfg["ego_camera"]["vertical_fov_deg"])
        if not np.isfinite(fov) or not 0 < fov < 180:
            raise ValueError("ego_camera.vertical_fov_deg must be finite and between 0 and 180")
    background = cfg.get("isaac_scene", {})
    edge = background.get("front_edge_xy")
    if edge is not None or background.get("enabled", False):
        # Resolve once in the physics process; the worker may start elsewhere.
        asset = Path(background["asset_path"])
        if not asset.is_absolute():
            asset = Path(__file__).resolve().parents[3] / asset
        background["asset_path"] = str(asset.resolve())
        if not isinstance(background.get("show_colliders", False), bool):
            raise ValueError("isaac_scene.show_colliders must be a boolean")
    if edge is not None:
        edge = np.asarray(edge, dtype=float)
        if edge.shape != (2, 2) or not np.isfinite(edge).all() or np.array_equal(edge[0], edge[1]):
            raise ValueError("isaac_scene.front_edge_xy requires two distinct finite XY points")
        align_scanned_scene(background, table, edge)
    elif background.get("enabled", False):
        translation = np.asarray(background["translation"], dtype=float)
        quaternion = np.asarray(background["quaternion_wxyz"], dtype=float)
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError("isaac_scene.translation requires three finite metre values")
        if (quaternion.shape != (4,) or not np.isfinite(quaternion).all()
                or not np.isclose(np.linalg.norm(quaternion), 1, atol=1e-6)):
            raise ValueError("isaac_scene.quaternion_wxyz requires a unit quaternion")
    position = cfg["robot"].get("initial_position_xy")
    if position is not None:
        position = np.asarray(position, dtype=float)
        if position.shape != (2,) or not np.isfinite(position).all():
            raise ValueError("robot.initial_position_xy requires two finite metre values")
    return cfg


def align_scanned_scene(background, table, edge):
    table_rotation = Rotation.from_quat(table["quaternion_wxyz"], scalar_first=True)
    tangent, inward, up = table_rotation.as_matrix().T
    if not np.allclose(up, [0, 0, 1]):
        raise ValueError("The scanned scene requires an upright table frame")
    source_direction = edge[1] - edge[0]
    source_yaw = np.arctan2(source_direction[1], source_direction[0])
    yaw = np.arctan2(tangent[1], tangent[0]) - source_yaw
    rotation = Rotation.from_euler("z", yaw)
    # Map the scan's front edge to the existing table-local -Y edge, keeping
    # the robot approach, object sampling frame and ground elevation fixed.
    target_midpoint = np.r_[table["position_xy"], 0] - 0.5 * table["size"][1] * inward
    translation = target_midpoint - rotation.apply(np.r_[edge.mean(axis=0), 0])
    background["translation"] = translation.tolist()
    background["quaternion_wxyz"] = rotation.as_quat(scalar_first=True).tolist()


def add_room_collisions(root, cfg):
    background = cfg["isaac_scene"]
    asset = Path(background["asset_path"])
    rotation = Rotation.from_quat(background["quaternion_wxyz"], scalar_first=True)
    translation = np.asarray(background["translation"], dtype=float)
    layer = asset.read_text()
    deformable = ET.SubElement(root, "deformable")
    # The deployed USDA bakes the mesh points into the scene frame. Static
    # flexes preserve those triangles, including concavities and holes.
    for name in ("SceneGeometry", "Tabletop"):
        mesh = re.search(rf'def Mesh "{name}"\s*\([^)]*\)\s*\{{([^}}]+)\}}', layer)
        if mesh is None or "bool physics:collisionEnabled = 1" not in mesh[1]:
            raise ValueError(f"{asset} requires an enabled {name} collision mesh")
        arrays = dict(re.findall(
            r"(?:point3f|int)\[\]\s+(points|faceVertexIndices|faceVertexCounts)\s*=\s*\[([^\]]+)\]",
            mesh[1],
        ))
        if len(arrays) != 3 or not np.all(np.fromstring(arrays["faceVertexCounts"], sep=",") == 3):
            raise ValueError(f"{asset}:{name} requires a triangulated mesh")
        points = arrays["points"].translate(str.maketrans("(),", "   "))
        vertices = rotation.apply(np.fromstring(points, sep=" ").reshape(-1, 3)) + translation
        flex = ET.SubElement(
            deformable, "flex", name=name, body="world", dim="2", radius="0.0001",
            vertex=numbers(vertices.ravel()), element=arrays["faceVertexIndices"].replace(",", " "),
        )
        ET.SubElement(flex, "contact", selfcollide="none", margin="0", friction="1 .005 .0001")
        if name == "Tabletop":
            height = float(vertices[0, 2])
            if not np.allclose(vertices[:, 2], height):
                raise ValueError(f"{asset}:Tabletop must be horizontal")
            cfg["table"]["top_height_range"] = [height, height]


def add_grasp_contacts(root, parameters):
    """Override only hand/object pairs, keeping other material contacts intact."""
    if not parameters or not parameters.get("enabled", True):
        return
    dimensions = int(parameters["condim"])
    if dimensions not in (3, 4, 6):
        raise ValueError("grasp_contact.condim must be 3, 4 or 6")
    friction = np.asarray(parameters["friction"], dtype=float)
    if friction.shape != (5,) or not np.isfinite(friction).all() or np.any(friction < 0):
        raise ValueError("grasp_contact.friction requires five finite nonnegative values")
    attributes = dict(condim=str(dimensions), friction=numbers(friction),
                      solref=numbers(parameters["solref"]),
                      solimp=numbers(parameters["solimp"]))
    obj = root.find(".//body[@name='grail_object']")
    objects = [geom.get("name") for geom in obj.findall("geom")
               if geom.get("name", "").endswith("_collision")]
    hands = []
    for body in root.iter("body"):
        if not re.fullmatch(r"(left|right)_hand_(palm|thumb_[012]|index_[01]|middle_[01])_link",
                            body.get("name", "")):
            continue
        for index, geom in enumerate(body.findall("geom")):
            if int(geom.get("contype", "1")) == 0 and int(geom.get("conaffinity", "1")) == 0:
                continue
            if not geom.get("name"):
                geom.set("name", f"grail_grasp_{body.get('name')}_{index}")
            hands.append(geom.get("name"))
    if not hands or not objects:
        raise ValueError("grasp_contact requires hand and object collision geometry")
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for index, (hand, obj) in enumerate((h, o) for h in hands for o in objects):
        ET.SubElement(contact, "pair", name=f"grail_grasp_pair_{index}",
                      geom1=hand, geom2=obj, **attributes)


def build_scene(robot_scene, scene_config, dataset_dir, object_key, cache_dir):
    cfg = load_scene_config(scene_config)
    manifest_path, manifest = load_assets(dataset_dir, object_key, cache_dir)
    root = ET.parse(robot_scene).getroot()
    # Expand this repository's single robot include, resolving mesh paths at
    # their source. Never modify the existing robot MJCF on disk.
    for include in list(root.findall("include")):
        source = Path(robot_scene).parent / include.attrib["file"]
        robot = ET.parse(source).getroot()
        compiler = robot.find("compiler")
        meshdir = source.parent / compiler.get("meshdir", "")
        compiler.attrib.pop("meshdir", None)
        for mesh in robot.findall("asset/mesh"):
            mesh.set("file", str((meshdir / mesh.get("file")).resolve()))
        index = list(root).index(include)
        root.remove(include)
        for element in list(robot):
            root.insert(index, element)
            index += 1
    # The stock model merges palms into wrist yaw. Split only those fixed
    # visuals/colliders into named massless children for contact filtering.
    for side in ("left", "right"):
        wrist = root.find(f".//body[@name='{side}_wrist_yaw_link']")
        palm = ET.SubElement(
            wrist,
            "body",
            name=f"{side}_hand_palm_link",
            pos=f"0.0415 {0.003 if side == 'left' else -0.003} 0",
        )
        for geom in list(wrist.findall("geom")):
            if geom.get("mesh") == f"{side}_hand_palm_link":
                wrist.remove(geom)
                geom.set("pos", "0 0 0")
                geom.set("density", "0")
                palm.append(geom)
    assets = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    table = ET.SubElement(world, "body", name="grail_table", mocap="true")
    table_config = cfg["table"]
    bounds = table_config.get("collision_bounds_xy")
    if bounds is None and cfg.get("isaac_scene", {}).get("front_edge_xy") is not None:
        add_room_collisions(root, cfg)
    else:
        half_size = np.asarray(table_config["size"], dtype=float) / 2
        center = np.zeros(3)
        if bounds is not None:
            lower, upper = np.asarray(bounds)
            half_size[:2] = (upper - lower) / 2
            center[:2] = (lower + upper) / 2
        ET.SubElement(
            table,
            "geom",
            name="grail_table_geom",
            type="box",
            size=numbers(half_size),
            pos=numbers(center),
            rgba=".5 .35 .2 1",
            friction="1 .005 .0001",
        )
    ET.SubElement(
        world,
        "camera",
        name="grail_overview",
        mode="targetbody",
        target="grail_table",
        pos="3 4 2.5",
        fovy="55",
    )
    obj = ET.SubElement(world, "body", name="grail_object", pos="0 0 1")
    ET.SubElement(obj, "freejoint", name="grail_object_joint")
    ET.SubElement(
        obj,
        "inertial",
        mass="0.1",
        pos=numbers(manifest["center_of_mass"]),
        fullinertia=numbers(manifest["fullinertia"]),
    )
    for record in manifest["meshes"]:
        name = record["name"]
        ET.SubElement(assets, "mesh", name=name, file=str(manifest_path.parent / record["file"]))
        if record["texture"]:
            ET.SubElement(
                assets,
                "texture",
                name=name,
                type="2d",
                file=str(manifest_path.parent / record["texture"]),
            )
            ET.SubElement(
                assets, "material", name=name, texture=name, texuniform="false", rgba="1 1 1 1"
            )
        visual = ET.SubElement(
            obj,
            "geom",
            name=name + "_visual",
            type="mesh",
            mesh=name,
            contype="0",
            conaffinity="0",
            group="1",
            density="0",
        )
        if record["texture"]:
            visual.set("material", name)
        else:
            visual.set("rgba", numbers(record["rgba"]))
        if record["collision"]:
            ET.SubElement(
                obj,
                "geom",
                name=name + "_collision",
                type="mesh",
                mesh=name,
                group="3",
                rgba="0 0 0 0",
                friction=numbers(cfg["object"]["friction"]),
                condim="4",
                density="0",
                solref=".005 1",
            )
    for name, color in (("carry", ".1 .4 1 .7"), ("release", ".2 1 .2 .7")):
        marker = ET.SubElement(world, "body", name="grail_" + name, mocap="true")
        ET.SubElement(
            marker,
            "geom",
            name="grail_" + name + "_marker",
            type="sphere",
            size=".035",
            rgba=color,
            contype="0",
            conaffinity="0",
            group="2",
        )
    add_grasp_contacts(root, cfg.get("grasp_contact"))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    if "ego_camera" in cfg:
        # Custom scene configs can still select a symmetric perspective camera.
        camera_id = model.camera("head_camera").id
        model.cam_sensorsize[camera_id] = 0
        model.cam_fovy[camera_id] = float(cfg["ego_camera"]["vertical_fov_deg"])
    model.opt.timestep = 0.005
    # G1_43DOF_CFG sets armature explicitly and has no passive dry friction.
    # The legacy MJCF's .01 armature/.05 damping/.1-.2 friction otherwise
    # changes the plant before any C++ gains are applied.
    for name in cfg["robot"]["observation_joint_names"]:
        joint = model.joint(name).id
        parameters = [
            value
            for pattern, value in cfg["robot"]["dynamics"].items()
            if re.fullmatch(pattern, name)
        ]
        if len(parameters) != 1:
            raise ValueError(f"Expected one dynamics entry for {name}")
        parameters = parameters[0]
        dof = model.jnt_dofadr[joint]
        model.dof_armature[dof] = parameters["armature"]
        model.dof_damping[dof] = 0
        model.dof_frictionloss[dof] = 0
        effort = parameters["effort"]
        model.jnt_actfrcrange[joint] = [-effort, effort]
        model.jnt_actfrclimited[joint] = 1
        for actuator in np.flatnonzero(model.actuator_trnid[:, 0] == joint):
            model.actuator_ctrlrange[actuator] = [-effort, effort]
    return model, cfg, manifest
