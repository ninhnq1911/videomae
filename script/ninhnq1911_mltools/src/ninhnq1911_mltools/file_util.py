import os
from pathlib import Path

DEFAULT_COLAB_MOUNTED_FOLDER = os.getenv("DEFAULT_COLAB_MOUNTED_FOLDER", "DATA")
PROJECT_NAME = os.getenv("PROJECT_NAME", "VIDEOMAE")


def make_dir(name: Path | str):
    if isinstance(name, str):
        dir_path = Path(name)
    if not dir_path.is_dir():
        dir_path = dir_path.parent
    dir_path.mkdir(parents=True, exist_ok=True)
    return Path(name).as_posix()


def get_resource(name: str) -> str:
    return make_dir(os.path.join(DEFAULT_COLAB_MOUNTED_FOLDER, PROJECT_NAME, name))
