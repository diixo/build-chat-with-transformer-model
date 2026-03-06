from pathlib import Path

def check_local_model_dir(model_dir: str):
    p = Path(model_dir)

    if not p.exists():
        return False, f"Директория не существует: {p}"

    if not p.is_dir():
        return False, f"Это не директория: {p}"

    config_file = p / "config.json"
    if not config_file.exists():
        return False, f"Нет config.json в {p}"

    has_weights = (
        (p / "pytorch_model.bin").exists()
        or (p / "model.safetensors").exists()
        or (p / "tf_model.h5").exists()
        or (p / "flax_model.msgpack").exists()
    )

    if not has_weights:
        return False, f"Не найден файл весов в {p}"

    return True, f"Похоже, локальная модель готова к загрузке: {p}"
