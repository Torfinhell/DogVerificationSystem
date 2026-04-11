from pathlib import Path
import json
from .yandex_utils import FILEDownloader, CsvChunkDownloader, FILETracker
from .youtube_utils import youtube_download
import pandas as pd
from collections import defaultdict
import ast
from pytubefix import YouTube
from tqdm import tqdm
import requests
import argparse
import os
import time
BREED_JSONS = [
        "chihuahua.json",
        "german_shepherd.json",
        "husky.json",
        "labrador.json",
        "pitbull.json",
        "shiba_inu.json",
    ]

BASE_URL = "https://raw.githubusercontent.com/fispresent/dog2vec/main/data/150h_data/"


DATA_DIR = Path("data/datasets/dog2vec_dataset")
RAW_DIR = DATA_DIR / "raw"
FILTERED_CSV = DATA_DIR / "filtered.csv"
FINAL_JSON = DATA_DIR / "filtered_index.json"
TRACKER_JSON = DATA_DIR / "download_tracker.json"
CATEGORY_JSON = DATA_DIR / "filtered_categories.json"


def download_raw_jsons():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for json_file in tqdm(BREED_JSONS, desc="Downloading raw JSON files"):
        local_path = RAW_DIR / json_file
        if local_path.exists():
            continue
        url = BASE_URL + json_file
        r = requests.get(url)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)


def filter_info(entry) -> bool:
    # TODO: improve later
    return True


def filter_audio(entry) -> bool:
    return True


def infer_context(entry) -> str:
    # Get the combined title and description text from the context field
    text = str(entry.get("context", "")).lower()

    if "home" in text or "indoors" in text or "indoor" in text:
        return "home"
    if "outdoor" in text or "outdoors" in text or "street" in text or "park" in text or "yard" in text:
        return "outdoors"
    if "night" in text or "dark" in text:
        return "night"
    if "car" in text or "road" in text or "traffic" in text:
        return "road"
    if "beach" in text or "water" in text or "river" in text:
        return "water"
    return "unknown"


def filter_context(entry, allowed_contexts: list[str] | None = None) -> bool:
    if allowed_contexts is None:
        return True
    return entry.get("context") in allowed_contexts


def download_info_youtube():
    processed_ids = set()
    breed_counts: dict[str, int] = {}

    with FILETracker(TRACKER_JSON) as tracker:
        with CsvChunkDownloader(
            FILTERED_CSV,
            columns=["video_id", "breed", "segments", "context"],
        ) as csv_writer:

            for json_file in tqdm(BREED_JSONS, desc="Processing breed files"):
                breed = json_file.replace(".json", "")
                path = RAW_DIR / json_file

                data = json.load(open(path))

                for video_id in tqdm(data.keys(), desc=f"Processing {breed} videos", leave=False):
                    if video_id in processed_ids:
                        continue
                    status = tracker.get_status(video_id)
                    if status in ("completed", "failed", "skipped"):
                        processed_ids.add(video_id)
                        continue


                    segments = data[video_id]
                    title = ""
                    description = ""
                    try:
                        url = f"https://www.youtube.com/watch?v={video_id}"
                        yt = YouTube(url)
                        title = yt.title or ""
                        description = yt.description or ""
                    except Exception as e:
                        error_msg = str(e).lower()
                        is_bot = "bot" in error_msg or "captcha" in error_msg

                        print(f"Error fetching metadata for video {video_id}: {e}")

                        if is_bot:
                            raise RuntimeError(
                                f"Bot detection encountered for video {video_id}. "
                                f"Use proxy / cookies / PO token."
                            )
                        else:
                            tracker.mark_failed(video_id, reason=str(e))
                            continue

                    entry = {
                        "video_id": video_id,
                        "breed": breed,
                        "segments": segments,
                        "context": f"{title} {description}".strip(),
                    }
                    csv_writer.update_csv(pd.Series(entry))
                    breed_counts[breed] = breed_counts.get(breed, 0) + 1
                    tracker.mark_done(video_id, {"breed": breed, "context": entry["context"]})
                    processed_ids.add(video_id)
                    time.sleep(1)  
    category_info = {
        "num_breeds": len(breed_counts),
        "breed_video_counts": breed_counts,
        "max_videos_per_breed_context": None,
    }
    CATEGORY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with FILEDownloader(CATEGORY_JSON) as fd:
        with open(CATEGORY_JSON, "w") as f:
            json.dump(category_info, f, indent=2)


def final_filter_result(
    audio_only: bool = True,
    max_videos_per_breed_context: int | None = None,
    allowed_contexts: list[str] | None = None,
):
    df = pd.read_csv(FILTERED_CSV)

    grouped = defaultdict(lambda: {"segments": [], "breed": None, "context": ""})

    for _, row in tqdm(df.iterrows(), desc="Grouping videos", total=len(df)):
        video_id = row["video_id"]
        breed = row["breed"]
        context_text = row.get("context", "")
        
        segments = ast.literal_eval(row["segments"])

        grouped[video_id]["segments"].extend(segments)
        grouped[video_id]["breed"] = breed
        grouped[video_id]["context"] = context_text

    selected_counts: dict[tuple[str, str], int] = {}
    final_entries = []
    context_counts: dict[str, int] = {}

    for video_id in tqdm(grouped.keys(), desc="Processing videos"):
        data = grouped[video_id]
        
        # Infer context category from the combined text
        context_category = infer_context({"context": data["context"]})
        
        entry = {
            "video_id": video_id,
            "breed": data["breed"],
            "context": context_category,  # Now contains the inferred category
            "segments": data["segments"],
        }

        if not filter_context(entry, allowed_contexts):
            continue

        context_counts[context_category] = context_counts.get(context_category, 0) + 1

        if max_videos_per_breed_context is not None:
            key = (entry["breed"], context_category)
            if selected_counts.get(key, 0) >= max_videos_per_breed_context:
                continue
            selected_counts[key] = selected_counts.get(key, 0) + 1

        outputs = youtube_download(entry, audio_only=audio_only)

        for e in outputs:
            if filter_audio(e):
                e["context"] = context_category
                final_entries.append(e)

    # Update category info with context counts
    category_info = {
        "num_breeds": len(set(data["breed"] for data in grouped.values())),
        "breed_video_counts": {breed: sum(1 for data in grouped.values() if data["breed"] == breed) for breed in set(data["breed"] for data in grouped.values())},
        "context_counts": context_counts,
        "max_videos_per_breed_context": max_videos_per_breed_context,
        "selected_counts": selected_counts,
    }
    CATEGORY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with FILEDownloader(CATEGORY_JSON) as fd:
        with open(CATEGORY_JSON, "w") as f:
            json.dump(category_info, f, indent=2)

    with FILEDownloader(FINAL_JSON) as fd:
        with open(FINAL_JSON, "w") as f:
            json.dump(final_entries, f, indent=2)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dog2Vec dataset preparation pipeline")
    parser.add_argument(
        "--max_videos_per_breed_context",
        type=int,
        default=5,
        help="Maximum number of videos per breed-context combination (default: 5)"
    )
    parser.add_argument(
        "--yandex_token",
        type=str,
        default=None,
        help="Yandex Disk token for uploading/downloading tracker and CSV files"
    )
    args = parser.parse_args()

    if args.yandex_token:
        os.environ["YANDEX_TOKEN"] = args.yandex_token

    print("Running Dog2Vec pipeline...")
    download_raw_jsons()
    download_info_youtube()
    # final_filter_result(
    #     max_videos_per_breed_context=args.max_videos_per_breed_context
    # )