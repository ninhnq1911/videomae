import os
from typing import Dict, List

import pandas as pd
from mltools.utl.file_util import copy_file, count_files, get_file_list, get_subfolders, make_dir
from mltools.utl.list_utl import custom_size_chunking


def ucf101_subset_info(root_dir: str) -> None:
    subset = get_subfolders(root_dir)
    cls_name = get_subfolders(os.path.join(root_dir, subset[0]))
    print("Total classes: ", len(cls_name))
    print("class: ", cls_name)
    for folder in subset:
        print(f"{folder}: {count_files(os.path.join(root_dir, folder), "avi")}")


def split_ucf101(data_files: Dict[str, str], split_frac: List, ) -> List[pd.DataFrame]:
    df = pd.DataFrame(data_files)
    df = df.sample(frac=1).reset_index(drop=True)
    print("Dataset size: ", len(df))
    return list(custom_size_chunking(df, split_frac))


def install_subset(ouput_dir: str, subset: str, subset_files: pd.DataFrame) -> None:
    subset_dir = make_dir(os.path.join(ouput_dir, subset))
    cls_name = list(subset_files.columns)
    for cls in cls_name:
        dir_path = make_dir(os.path.join(subset_dir, cls))
        for file in subset_files[cls]:
            copy_file(file, os.path.join(dir_path, os.path.basename(file)))
    print("Subset installed: ", subset)

def process_data_bkx(videos_dir: str, output_dir: str) -> None:
    cls_name = get_subfolders(videos_dir)
    data_files = {
        cls: get_file_list(os.path.join(videos_dir, cls), "avi") for cls in cls_name
    }
    train, test, val = split_ucf101(data_files, [0.7, 0.2, 0.1])
    print("Split sizes: ", len(train), len(test), len(val))
    
    install_subset(output_dir, "train", train)
    install_subset(output_dir, "test", test)
    install_subset(output_dir, "val", val)
    
    
    
