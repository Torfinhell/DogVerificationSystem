from pathlib import Path
import json
from .data_utils import youtube_download
from .data_utils import FILEDownloader, CsvChunkDownloader
import pandas as pd
from collections import defaultdict
import ast
BREED_JSONS = [
        "chihuahua.json",
        "german_shepherd.json",
        "husky.json",
        "labrador.json",
        "pitbull.json",
        "shiba_inu.json",
    ]

BASE_URL = "https://raw.githubusercontent.com/fispresent/dog2vec/main/data/150h_data/"


DATA_DIR = Path("data/dog2vec_dataset")
RAW_DIR = DATA_DIR / "raw"
FILTERED_CSV = DATA_DIR / "filtered.csv"
FINAL_JSON = DATA_DIR / "filtered_index.json"


def download_raw_jsons():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for json_file in BREED_JSONS:
        local_path = RAW_DIR / json_file
        if local_path.exists():
            continue
        url = BASE_URL + json_file
        import requests
        r = requests.get(url)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)


def filter_info(entry) -> bool:
    # TODO: improve later
    return True


def filter_audio(entry) -> bool:
    return True


def download_info_youtube():
    processed_ids = set()

    with CsvChunkDownloader(
        FILTERED_CSV,
        columns=["video_id", "breed", "segments"],
    ) as csv_writer:

        for json_file in BREED_JSONS:
            breed = json_file.replace(".json", "")
            path = RAW_DIR / json_file

            data = json.load(open(path))

            for video_id, segments in data.items():
                if video_id in processed_ids:
                    continue

                entry = {
                    "video_id": video_id,
                    "breed": breed,
                    "segments": segments,
                }

                if filter_info(entry):
                    csv_writer.update_csv(
                        pd.Series(entry)
                    )

                processed_ids.add(video_id)


def final_filter_result(audio_only: bool = True):
    df = pd.read_csv(FILTERED_CSV)

    grouped = defaultdict(lambda: {"segments": [], "breed": None})

    for _, row in df.iterrows():
        video_id = row["video_id"]
        breed = row["breed"]

        segments = ast.literal_eval(row["segments"])

        grouped[video_id]["segments"].extend(segments)
        grouped[video_id]["breed"] = breed

    final_entries = []

    for video_id, data in grouped.items():
        entry = {
            "video_id": video_id,
            "breed": data["breed"],
            "segments": data["segments"],
        }

        outputs = youtube_download(entry, audio_only=audio_only)

        for e in outputs:
            if filter_audio(e):
                final_entries.append(e)

    with FILEDownloader(FINAL_JSON) as fd:
        with open(FINAL_JSON, "w") as f:
            json.dump(final_entries, f, indent=2)