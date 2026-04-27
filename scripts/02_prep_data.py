"""Filter VSI-Bench to 3 spatial categories, sample N per category, extract videos.

VSI-Bench stores videos in 3 zip files (arkitscenes.zip, scannet.zip, scannetpp.zip)
in the HF dataset repo. We download whichever zip(s) are needed by our sample, then
extract only the specific scene videos we need.
"""
import json
import random
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from src.config import (CATEGORIES, N_PER_CATEGORY, SEED, SUBSET_FILE,
                        VIDEO_DIR)

REPO = "nyu-visionx/VSI-Bench"
DATASET_ZIPS = {
    "arkitscenes": "arkitscenes.zip",
    "scannet": "scannet.zip",
    "scannetpp": "scannetpp.zip",
}


def ensure_zip_downloaded(dataset_name: str) -> str:
    return hf_hub_download(
        repo_id=REPO, repo_type="dataset",
        filename=DATASET_ZIPS[dataset_name], cache_dir=str(VIDEO_DIR),
    )


def extract_video_for_scene(zip_path: str, dataset_name: str, scene_name: str,
                            extract_to: Path) -> str | None:
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        # Match files containing the scene name
        candidates = [n for n in names if scene_name in n and not n.endswith("/")]
        # Prefer video extensions
        videos = [n for n in candidates
                  if n.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm"))]
        target = videos[0] if videos else (candidates[0] if candidates else None)
        if not target:
            return None

        out_name = f"{dataset_name}__{scene_name}__{Path(target).name}"
        out_path = extract_to / out_name
        if out_path.exists() and out_path.stat().st_size > 0:
            return str(out_path)
        with zf.open(target) as src, open(out_path, "wb") as dst:
            dst.write(src.read())
        return str(out_path)


def main():
    print(f"Loading {REPO} ...")
    ds = load_dataset(REPO, split="test")

    rng = random.Random(SEED)
    sampled = []
    for cat in CATEGORIES:
        cat_items = [dict(x) for x in ds if str(x["question_type"]) == cat]
        print(f"  {cat}: {len(cat_items)} available")
        rng.shuffle(cat_items)
        sampled.extend(cat_items[: N_PER_CATEGORY * 2])

    needed_datasets = sorted({str(x["dataset"]) for x in sampled})
    print(f"\nNeeded zip(s): {needed_datasets}")
    print("(Each zip is multi-GB; ensure your container disk has space.)")

    zip_paths = {}
    for ds_name in needed_datasets:
        if ds_name not in DATASET_ZIPS:
            print(f"  WARN: unknown dataset {ds_name}, skipping")
            continue
        print(f"  downloading {DATASET_ZIPS[ds_name]} ...")
        zip_paths[ds_name] = ensure_zip_downloaded(ds_name)

    extract_dir = VIDEO_DIR / "extracted"
    enriched = []
    per_cat = {c: 0 for c in CATEGORIES}

    print("\nExtracting videos ...")
    for item in tqdm(sampled):
        cat = str(item["question_type"])
        if per_cat[cat] >= N_PER_CATEGORY:
            continue
        ds_name = str(item["dataset"])
        scene = str(item["scene_name"])
        if ds_name not in zip_paths:
            continue
        try:
            video_path = extract_video_for_scene(zip_paths[ds_name], ds_name, scene, extract_dir)
        except Exception as e:
            print(f"  WARN extract failed for {ds_name}/{scene}: {e}")
            video_path = None
        if not video_path:
            continue

        opts = item.get("options")
        if opts is not None and not isinstance(opts, list):
            try:
                opts = list(opts)
            except Exception:
                opts = None

        enriched.append({
            "id": str(item.get("id", f"{ds_name}_{scene}_{cat}_{per_cat[cat]}")),
            "category": cat,
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "options": opts,
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
