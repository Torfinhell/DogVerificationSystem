from pathlib import Path
import gdown
GDRIVE_FILE_ID = "1q0iDiGDCPbPThtL8zICkIS2jwHDDCM7c"
OUTPUT_PATH = "saved/models/dog2vec.pth"

if __name__ == "__main__":
    Path(OUTPUT_PATH).parent.mkdir(exist_ok=True, parents=True)
    gdown.download(id=GDRIVE_FILE_ID, output=OUTPUT_PATH, quiet=False)
    print(f"Downloaded to {OUTPUT_PATH}")
