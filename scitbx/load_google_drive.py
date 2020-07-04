# mount the drive
def mount_drive():
    from google.colab import drive
    from pathlib import Path
  
    drive.mount('/content/drive')
    return Path.cwd().joinpath('drive/My Drive')