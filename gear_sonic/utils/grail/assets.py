"""RoboCasa USD conversion. Run preparation with usd-core; runtime needs no USD.

Meshes are baked into the USD default-prim parent's coordinate system, exactly
the frame to which Isaac's rigid root pose is applied. The current bottle uses
SDF in PhysX; MuJoCo uses its convex hull, recorded explicitly in the manifest.
"""

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

CONVERTER_VERSION = 1


def asset_paths(dataset_dir, object_key):
    if Path(object_key).name != object_key:
        raise ValueError("object_key must be an asset basename")
    root = Path(dataset_dir).resolve()
    usd, bps = root / "object_usd" / (object_key + ".usd"), root / "bps" / (object_key + ".npy")
    for path in (usd, bps):
        if not path.is_file():
            raise FileNotFoundError(path)
    values = np.load(bps, allow_pickle=False)
    if values.shape != (10,) or not np.isfinite(values).all():
        raise ValueError(f"Expected finite 10D BPS: {bps}")
    return usd, bps


def prepare_assets(dataset_dir, object_key, cache_dir):
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    import trimesh

    usd, bps = asset_paths(dataset_dir, object_key)
    stage = Usd.Stage.Open(str(usd))
    if not stage or not stage.GetDefaultPrim():
        raise ValueError(f"USD has no default prim: {usd}")
    if UsdGeom.GetStageUpAxis(stage) != "Z":
        raise ValueError("Only this batch's Z-up RoboCasa USD is supported")
    # Fingerprint every used layer and resolved asset input, including textures.
    dependencies = {usd, bps}
    dependencies.update(Path(layer.realPath) for layer in stage.GetUsedLayers() if layer.realPath)
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            value = attr.Get()
            if isinstance(value, Sdf.AssetPath) and value.path:
                # Built-in OmniPBR is translated to a local preview surface;
                # it is a shader module, not a missing dataset texture.
                if value.path == "OmniPBR.mdl":
                    continue
                resolved = (
                    Path(value.resolvedPath) if value.resolvedPath else usd.parent / value.path
                )
                if not resolved.is_file():
                    raise FileNotFoundError(f"Unresolved USD asset {prim.GetPath()} {value.path}")
                dependencies.add(resolved.resolve())
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(dependencies)}
    digest = hashlib.sha256(
        json.dumps([CONVERTER_VERSION, hashes, "sdf-to-convex-hull", 0.1], sort_keys=True).encode()
    ).hexdigest()
    destination = Path(cache_dir).resolve() / object_key / digest[:20]
    manifest_path = destination / "manifest.json"
    if manifest_path.is_file():
        (destination.parent / "latest.json").write_text(
            json.dumps({"manifest": str(manifest_path)})
        )
        return manifest_path
    destination.mkdir(parents=True, exist_ok=True)
    cache = UsdGeom.XformCache()
    unit = UsdGeom.GetStageMetersPerUnit(stage)
    records, clouds, solids = [], [], []
    for prim in Usd.PrimRange(stage.GetDefaultPrim()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        matrix = np.asarray(cache.GetLocalToWorldTransform(prim))
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        vertices = (np.column_stack((points, np.ones(len(points)))) @ matrix)[:, :3] * unit
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
        uv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
        uv_values = np.asarray(uv.ComputeFlattened()) if uv and uv.HasValue() else None
        uv_mode = uv.GetInterpolation() if uv_values is not None else None
        if uv_mode not in (None, "vertex", "faceVarying"):
            raise ValueError(f"Unsupported UV interpolation {uv_mode}")
        faces, uv_faces = [], []
        offset = 0
        for count in counts:
            for j in range(1, count - 1):
                corners = [offset, offset + j, offset + j + 1]
                faces.append(indices[corners])
                uv_faces.append(corners if uv_mode == "faceVarying" else indices[corners])
            offset += count
        faces = np.asarray(faces)
        if (mesh.GetOrientationAttr().Get() == "leftHanded") != (np.linalg.det(matrix[:3, :3]) < 0):
            faces = faces[:, ::-1]
            uv_faces = np.asarray(uv_faces)[:, ::-1]
        name = f"object_mesh_{len(records)}"
        obj = destination / (name + ".obj")
        with obj.open("w") as stream:
            for v in vertices:
                stream.write("v " + " ".join(map(str, v)) + "\n")
            if uv_values is not None:
                for v in uv_values:
                    stream.write(f"vt {v[0]} {v[1]}\n")
            for face, uv_face in zip(faces, uv_faces):
                entries = [
                    f"{v + 1}/{u + 1}" if uv_values is not None else str(v + 1)
                    for v, u in zip(face, uv_face)
                ]
                stream.write("f " + " ".join(entries) + "\n")
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        texture, color = None, [0.8, 0.8, 0.8, 1.0]
        if material:
            for shader in Usd.PrimRange(material.GetPrim()):
                if not shader.IsA(UsdShade.Shader):
                    continue
                for attr in shader.GetAttributes():
                    value = attr.Get()
                    if isinstance(value, Sdf.AssetPath) and "diffuse" in attr.GetName():
                        source = (
                            Path(value.resolvedPath)
                            if value.resolvedPath
                            else usd.parent / value.path
                        )
                        texture = name + source.suffix
                        shutil.copyfile(source, destination / texture)
        collision = (
            prim.HasAPI(UsdPhysics.CollisionAPI)
            and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False
        )
        approximation = (
            UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() if collision else None
        )
        if approximation not in (None, "convexHull", "sdf"):
            raise ValueError(f"Unsupported collision declaration {approximation}")
        records.append(
            dict(
                name=name,
                file=obj.name,
                texture=texture,
                rgba=color,
                collision=collision,
                source_approximation=approximation,
                mujoco_approximation="convexHull" if collision else None,
            )
        )
        clouds.append(vertices)
        if collision:
            solids.append(trimesh.Trimesh(vertices=vertices, faces=faces, process=True).convex_hull)
    if not records or not solids:
        raise ValueError("No supported visual/collision meshes in asset")
    # A uniform-density convex hull gives a positive definite inertia in the
    # original rigid frame; mass override is exactly 0.1 kg, as in training.
    solid = trimesh.util.concatenate(solids)
    solid.density = 0.1 / solid.volume
    inertia = solid.moment_inertia
    manifest = dict(
        version=CONVERTER_VERSION,
        object_key=object_key,
        source=str(usd),
        dependencies=hashes,
        digest=digest,
        bps=np.load(bps).tolist(),
        meshes=records,
        vertices=np.concatenate(clouds).tolist(),
        mass=0.1,
        center_of_mass=solid.center_mass.tolist(),
        fullinertia=[
            inertia[0, 0],
            inertia[1, 1],
            inertia[2, 2],
            inertia[0, 1],
            inertia[0, 2],
            inertia[1, 2],
        ],
        collision_note="PhysX SDF replaced by MuJoCo mesh convex hull; no attachment constraints",
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))
    # Stable pointer permits the physics environment to use a prepared cache
    # without loading USD bindings, while still verifying its source hashes.
    (destination.parent / "latest.json").write_text(json.dumps({"manifest": str(manifest_path)}))
    return manifest_path


def load_assets(dataset_dir, object_key, cache_dir):
    usd, _ = asset_paths(dataset_dir, object_key)
    pointer = Path(cache_dir).resolve() / object_key / "latest.json"
    if not pointer.is_file():
        raise FileNotFoundError(
            "Prepare USD cache first: python -m gear_sonic.scripts.prepare_grail_assets "
            f"--dataset-dir {dataset_dir} --object-key {object_key} --cache-dir {cache_dir}"
        )
    path = Path(json.loads(pointer.read_text())["manifest"])
    manifest = json.loads(path.read_text())
    if manifest["object_key"] != object_key or Path(manifest["source"]) != usd:
        raise ValueError("Cache belongs to a different object/dataset; prepare it again")
    for filename, digest in manifest["dependencies"].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Asset cache is stale: {filename}; prepare again")
    return path, manifest
