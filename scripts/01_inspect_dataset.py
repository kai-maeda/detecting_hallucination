"""Inspect VSI-Bench schema so 02_prep_data.py uses the right field names.

Run first on RunPod to confirm field names match what 02_prep_data.py expects.
If field names differ, update src/config.py or the relevant prep script.
"""
from datasets import load_dataset
from collections import Counter
import json

REPO = "nyu-visionx/VSI-Bench"


def main():
    print(f"Loading {REPO} ...")
    ds = load_dataset(REPO, split="test")
    print(f"\nTotal rows: {len(ds)}")
    print(f"Columns: {ds.column_names}")
    print("\nFirst row (truncated):")
    row = ds[0]
    pretty = {k: (str(v)[:200] + "...") if len(str(v)) > 200 else v for k, v in row.items()}
    print(json.dumps(pretty, indent=2, default=str))

    # Heuristically find the category field
    candidate_cat_keys = [k for k in ds.column_names
                          if "type" in k.lower() or "categ" in k.lower() or "task" in k.lower()]
    print(f"\nLikely category field(s): {candidate_cat_keys}")
    for k in candidate_cat_keys:
        c = Counter(str(x[k]) for x in ds)
        print(f"\nDistribution of '{k}':")
        for v, n in c.most_common():
            print(f"  {v}: {n}")


if __name__ == "__main__":
    main()
