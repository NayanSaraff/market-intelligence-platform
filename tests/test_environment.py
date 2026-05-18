import os
from pathlib import Path
from src.data.preprocessor import SELECTED


def test_config_and_raw_files_exist():
    root = Path(os.getcwd())
    cfg = root / "configs" / "project_config.yaml"
    assert cfg.exists(), "configs/project_config.yaml is missing"

    raw_dir = root / "data" / "raw"
    assert raw_dir.exists(), "data/raw directory missing"

    missing = []
    for t in SELECTED:
        if not (raw_dir / f"{t}.csv").exists():
            missing.append(t)
    assert not missing, f"Missing raw CSVs for: {missing}"
