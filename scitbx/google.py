# mount the drive
def mount_drive():
    from google.colab import drive
    from pathlib import Path
  
    drive.mount('/content/drive')
    return Path.cwd().joinpath('drive/My Drive')

# download from colab
def download_file(src, filename, **kwargs):
    """
    src: data source, dataframe, figures, etc.
    filename: directory to save file (e.g. fig.png)
    """
    import pickle
    from google.colab import files
    # save figures:
    if filename.split(".")[-1] in ["jpeg", "jpg", "png"]:
        fig = src
        fig.savefig(filename, dpi = 300, bbox_inches = "tight", **kwargs)
    # csv
    elif filename.split(".")[-1] == "csv":
        df = src
        df.to_csv(filename)
    # xlsx
    elif filename.split(".")[-1] == "xlsx":
        df = src
        df.to_excel(filename)
    # pickle
    else:
        with open(filename, "wb") as f:
            pickle.dump(src, f)
    files.download(filename)