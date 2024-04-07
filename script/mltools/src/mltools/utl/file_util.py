import os
import shutil
import glob
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


def resolve_path(path: str | Path) -> str:
    return Path(path).resolve().as_posix()


def get_subfolders(dir_path: str | Path) -> list[str]:
    if dir_path is None:
        return []
    return [f.name for f in Path(resolve_path(dir_path)).iterdir() if f.is_dir()]


def count_files(dir_path: str | Path, ext: str = "*") -> int:
    return len(get_file_list(dir_path, ext))


def get_file_list(dir_path: str | Path, ext: str = "*") -> list[str]:
    if dir_path is None:
        return []
    dir_path = Path(dir_path).resolve().as_posix()
    return glob.glob(f"{dir_path}/**/*.{ext}", recursive=True)


def copy_file(src: str | Path, dst: str | Path) -> None:
    src = resolve_path(src)
    dst = resolve_path(dst)
    make_dir(dst)
    print(f"Copying {src} to {dst}")
    shutil.copy(src, dst)
