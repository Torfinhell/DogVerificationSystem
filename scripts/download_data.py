import kagglehub
import shutil
import os
downloaded_path = kagglehub.dataset_download("nikitasolonitsyn/barkopedia")
target_path = "."
if not os.path.exists(target_path):
    shutil.copytree(downloaded_path, target_path)
print("Dataset available at:", target_path)