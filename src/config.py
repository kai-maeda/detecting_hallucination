from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
VIDEO_DIR = DATA_DIR / "videos"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

for d in [DATA_DIR, VIDEO_DIR, RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CATEGORIES = ["object_counting", "relative_distance", "relative_direction"]
N_PER_CATEGORY = 50
N_PARAPHRASES = 4
SEED = 42

VLM_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
PARAPHRASE_MODEL = "gpt-4o-mini"
N_VIDEO_FRAMES = 8

SUBSET_FILE = DATA_DIR / "vsi_subset.json"
PARAPHRASES_FILE = DATA_DIR / "paraphrases.json"
INFERENCE_FILE = RESULTS_DIR / "inference.json"
SCORES_FILE = RESULTS_DIR / "scores.json"
