"""Filter VSI-Bench to 3 spatial categories, sample N per category, download videos."""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from src.config import (CATEGORIES, DATA_DIR, N_PER_CATEGORY, SEED,
                        SUBSET_FILE, VIDEO_DIR)

REPO = "nyu-visionx/VSI-Bench"

# VSI-Bench column names. Adjust here if 01_inspect_dataset.py shows different names.
CATEGORY_KEY = "question_type"
QUESTION_KEY = "question"
GT_KEY = "ground_truth"
OPTIONS_KEY = "options"
DATASET_KEY = "dataset"
SCENE_KEY = "scene_name"
ID_KEY = "id"


def try_download_video(dataset_name: str, scene_name: str) -> str | None:
    """Download a single video from the VSI-Bench HF repo. Returns local path."""
    candidates = [
        f"{dataset_name}/{scene_name}.mp4",
        f"videos/{dataset_name}/{scene_name}.mp4",
        f"{dataset_name}/videos/{scene_name}.mp4",
        f"{scene_name}.mp4",
    ]
    for path in candidates:
        try:
            local = hf_hub_download(
                repo_id=REPO,
                repo_type="dataset",
                filename=path,
                cache_dir=str(VIDEO_DIR),
            )
            return local
        except Exception:
            continue
    return None


def main():
    print(f"Loading {REPO} ...")
    ds = load_dataset(REPO, split="test")

    rng = random.Random(SEED)
    sampled = []
    for cat in CATEGORIES:
        cat_items = [dict(x) for x in ds if str(x[CATEGORY_KEY]) == cat]
        print(f"  {cat}: {len(cat_items)} available")
        rng.shuffle(cat_items)
        sampled.extend(cat_items[: N_PER_CATEGORY * 2])  # oversample in case downloads fail

    enriched = []
    per_cat = {c: 0 for c in CATEGORIES}
    print("\nDownloading videos ...")
    for item in tqdm(sampled):
        cat = str(item[CATEGORY_KEY])
        if per_cat[cat] >= N_PER_CATEGORY:
            continue
        ds_name = str(item[DATASET_KEY])
        scene = str(item[SCENE_KEY])
        video_path = try_download_video(ds_name, scene)
        if video_path is None:
            continue

        enriched.append({
            "id": str(item.get(ID_KEY, f"{ds_name}_{scene}_{cat}_{per_cat[cat]}")),
            "category": cat,
            "question": item[QUESTION_KEY],
            "ground_truth": item[GT_KEY],
            "options": item.get(OPTIONS_KEY),
            "video_path": video_path,
            "dataset": ds_name,
            "scene_name": scene,
        })
        per_cat[cat] += 1

    print(f"\nFinal sample sizes: {per_cat}")
    SUBSET_FILE.write_text(json.dumps(enriched, indent=2, default=str))
    print(f"Wrote {len(enriched)} items -> {SUBSET_FILE}")


if __name__ == "__main__":
    main()
