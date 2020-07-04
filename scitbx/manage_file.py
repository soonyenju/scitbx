from pathlib import Path
import zipfile
import sys
import numbers

# if parent or grand parent dirs not exist, 
# make them including current dir (if its not a file)
def create_all_parents(directory, flag = "a"):
    # flag: if directory is a dir ("d") or file ("f") or automatically desice ("a")
    if not isinstance(directory, Path):
        directory = Path(directory)
    parents = list(directory.parents)
    parents.reverse()
    # NOTICE: sometimes is_dir returns false, e.g., a dir of onedrive
    if flag == "a":
        if not directory.is_file():
            parents.append(directory)
    elif flag == "d":
        parents.append(directory)
    else:
        pass
    for p in parents:
      if not p.exists():
        p.mkdir()

# search current dir and list the subdirs
def searching_all_files(directory):
    if not isinstance(directory, Path):
        dirpath = Path(directory)
    assert(dirpath.is_dir())
    file_list = []
    for x in dirpath.iterdir():
        if x.is_file():
            file_list.append(x)
        elif x.is_dir():
            file_list.extend(searching_all_files(x))
    return file_list

# extract a .zip (p) into a folder at tar_dir of a name p.stem
def unzip(p, tar_dir, new_folder = True, folder_name = None, delete = False):
    # print(p)
    if not isinstance(p, Path):
        p = Path(p)
    if not isinstance(tar_dir, Path):
        tar_dir = Path(tar_dir)
    if new_folder:
        if not folder_name:
            out_dir = tar_dir.joinpath(p.stem)
        else:
            out_dir = tar_dir.joinpath(folder_name)
    else:
        out_dir = tar_dir
    # if not out_dir.exists(): out_dir.mkdir()
    create_all_parents(out_dir)
    with zipfile.ZipFile(p, "r") as zip_ref:
        zip_ref.extractall(out_dir)
    if delete:
        p.unlink()

# print progress bar at one line.
def pbar(idx, total, auto_check = False):
    if auto_check:
        if not isinstance(idx, numbers.Number):
            idx = float(idx)
        if not isinstance(total, numbers.Number):
            total = float(total)
    if (100*(idx + 1)/total).is_integer():
        sys.stdout.write(f"progress reaches {idx + 1} of {total}, {100*(idx + 1)/total}% ...\r")