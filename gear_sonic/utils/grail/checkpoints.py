"""Checkpoint selection shared by the live actor and offline collector."""

from pathlib import Path


DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[3] / "gear_sonic_deploy/policy/grail_grasp_policy"
    / "1119robocasa_carry_table-randomized-20261119_081005/last.pt"
)


def resolve_checkpoint(path):
    """Accept a .pt file or choose the most recently modified complete .pt."""
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        candidates = [p for p in path.glob("*.pt") if p.is_file() and not p.name.startswith(".")]
        if not candidates:
            raise FileNotFoundError(f"No complete .pt checkpoint in {path}")
        path = max(candidates, key=lambda p: (p.stat().st_mtime_ns, p.name))
    if path.suffix != ".pt" or not path.is_file():
        raise FileNotFoundError(path)
    return path
