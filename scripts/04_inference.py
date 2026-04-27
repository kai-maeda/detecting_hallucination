"""Run Qwen2.5-VL-7B on every (video, prompt) pair. Saves raw answers.

Resumable: skips items already in the output file.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from src.config import (INFERENCE_FILE, N_VIDEO_FRAMES, PARAPHRASES_FILE,
                        VLM_MODEL)

INSTRUCTION = {
    "object_counting": "Answer with a single integer count and nothing else.",
    "relative_distance": "Answer with the letter (A, B, C, or D) of the correct option and nothing else.",
    "relative_direction": "Answer with the letter (A, B, C, or D) of the correct option and nothing else.",
}


def format_prompt(question: str, options: list | None, category: str) -> str:
    instr = INSTRUCTION.get(category, "Answer concisely.")
    if options:
        letters = ["A", "B", "C", "D", "E", "F"]
        opts_str = "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(options))
        return f"{question}\n\nOptions:\n{opts_str}\n\n{instr}"
    return f"{question}\n\n{instr}"


def load_model():
    print(f"Loading {VLM_MODEL} ...")
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            VLM_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="flash_attention_2",
        )
    except Exception as e:
        print(f"  flash_attention_2 unavailable ({e}); falling back to sdpa")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            VLM_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="sdpa",
        )
    processor = AutoProcessor.from_pretrained(VLM_MODEL)
    model.eval()
    return model, processor


def run_one(model, processor, video_path: str, prompt_text: str) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": video_path, "nframes": N_VIDEO_FRAMES,
             "max_pixels": 360 * 420},
            {"type": "text", "text": prompt_text},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    gen_ids = gen_ids[:, inputs.input_ids.shape[1]:]
    out = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
    return out.strip()


def main():
    items = json.loads(PARAPHRASES_FILE.read_text())

    done = {}
    if INFERENCE_FILE.exists():
        for r in json.loads(INFERENCE_FILE.read_text()):
            done[r["id"]] = r
        print(f"Resuming: {len(done)} items already done")

    model, processor = load_model()

    results = list(done.values())
    try:
        for item in tqdm(items, desc="inference"):
            if item["id"] in done:
                continue
            answers = []
            t0 = time.time()
            for prompt in item["prompts"]:
                text = format_prompt(prompt, item.get("options"), item["category"])
                try:
                    ans = run_one(model, processor, item["video_path"], text)
                except Exception as e:
                    ans = f"<ERROR: {e}>"
                answers.append(ans)

            results.append({
                "id": item["id"],
                "category": item["category"],
                "ground_truth": item["ground_truth"],
                "options": item.get("options"),
                "prompts": item["prompts"],
                "answers": answers,
                "video_path": item["video_path"],
                "elapsed_sec": round(time.time() - t0, 2),
            })

            # Checkpoint every item to make this fully resumable
            INFERENCE_FILE.write_text(json.dumps(results, indent=2, default=str))
    finally:
        INFERENCE_FILE.write_text(json.dumps(results, indent=2, default=str))
        print(f"Wrote {len(results)} items -> {INFERENCE_FILE}")


if __name__ == "__main__":
    main()
