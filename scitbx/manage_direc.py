from pathlib import Path

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