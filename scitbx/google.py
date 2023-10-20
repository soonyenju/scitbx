# mount the drive
def mount_drive(force_remount = False):
    from google.colab import drive
    from pathlib import Path
  
    drive.mount('/content/drive', force_remount = force_remount)
    return Path.cwd().joinpath('drive/My Drive')

def unmount_drive():
    from google.colab import drive
    drive.flush_and_unmount()

def init_gee():
    import ee

    # Trigger the authentication flow.
    ee.Authenticate()

    # Initialize the library.
    ee.Initialize()

# download from colab
def download_file(src, filename, **kwargs):
    """
    src: data source, dataframe, figures, etc.
    filename: directory to save file (e.g. fig.png)
    """
    import pickle
    from google.colab import files
    # save figures:
    if filename.split(".")[-1] in ["jpeg", "jpg", "png", "pdf"]:
        fig = src
        if 'dpi' in kwargs:
            dpi = kwargs['dpi']
        else:
            dpi = 300
        fig.savefig(filename, dpi = dpi, bbox_inches = "tight", **kwargs)
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

def check_colab_gpu():
    # This function is deprecated on 25/10/2022
    import os
    if int(os.environ["COLAB_GPU"]) > 0: # TF_FORCE_GPU_ALLOW_GROWTH: True
        print("A GPU is connected.")
        gpu = 1
    elif "COLAB_TPU_ADDR" in os.environ and os.environ["COLAB_TPU_ADDR"]: # XRT_TPU_CONFIG or COLAB_TPU_ADDR or TPU_NAME in os.environ
        print("A TPU is connected.")
        gpu = 2
    else:
        print("No accelerator is connected.")
        gpu = 0
    return gpu