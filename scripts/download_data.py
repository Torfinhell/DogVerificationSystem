import kagglehub
import shutil
import os
cached_path = kagglehub.dataset_download("nikitasolonitsyn/barkopedia/versions/6")
target_path = "."
os.makedirs(target_path, exist_ok=True)
for item in os.listdir(cached_path):
    src = os.path.join(cached_path, item)
    dst = os.path.join(target_path, item)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
print(f"Dataset available at: {target_path}")