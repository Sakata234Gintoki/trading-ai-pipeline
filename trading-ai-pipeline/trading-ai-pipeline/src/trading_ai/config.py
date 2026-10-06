from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None, synthetic=False):
    """Load YAML config. Paths are resolved against the project root.

    With synthetic=True, all data goes under data/synthetic/... so fake data
    never mixes with real data.
    """
    path = Path(path) if path else ROOT / "config" / "config.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)

    resolved = {}
    for key, rel in cfg["paths"].items():
        rel_path = Path(rel)
        if synthetic:
            rel_path = Path(rel_path.parts[0]) / "synthetic" / Path(*rel_path.parts[1:])
        resolved[key] = ROOT / rel_path
        resolved[key].mkdir(parents=True, exist_ok=True)
    cfg["paths"] = resolved
    cfg["synthetic_mode"] = synthetic
    return cfg
