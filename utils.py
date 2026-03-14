from pathlib import Path


def check_local_model(model_dir: str):
    p = Path(model_dir)

    if not p.exists():
        return False, f"Directory does not exist: {p}"

    if not p.is_dir():
        return False, f"Directory invalid: {p}"

    config_file = p / "config.json"
    if not config_file.exists():
        return False, f"Absent config.json into {p}"

    has_weights = (
        (p / "pytorch_model.bin").exists()
        or (p / "model.safetensors").exists()
        or (p / "tf_model.h5").exists()
        or (p / "flax_model.msgpack").exists()
    )

    if not has_weights:
        return False, f"Weights file was not found in: {p}"

    return True, f"Local model is ready for loading from: {p}"
