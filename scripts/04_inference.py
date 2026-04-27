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
    "object_rel_distance": "Answer with the letter (A, B, C, or D) of the correct option and nothing else.",
    "object_rel_direction": "Answer with the letter (A, B, C, or D) of the correct option and nothing else.",
}


def _instruction_for(category: str) -> str:
    if category in INSTRUCTION:
        return INSTRUCTION[category]
    if category.startswith("object_rel_direction"):
        return INSTRUCTION["object_rel_direction"]
    return "Answer concisely."


def format_prompt(question: str, options: list | None, category: str) -> str:
    instr = _instruction_for(category)
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
    # CRITICAL for batched generation on decoder-only models: pad on the LEFT.
    # Right-padding makes the model attend to pad tokens during generation and
    # produces garbage for all sequences in the batch except the longest one.
    processor.tokenizer.padding_side = "left"
    if hasattr(processor.tokenizer, "pad_token") and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model.eval()
    return model, processor


def run_item_batched(model, processor, video_path: str, prompt_texts: list[str]) -> list[str]:
    """Run all prompts for a single video in one batched forward pass.

    Same video is reused across all prompts, so the vision encoder + decoder
    forward runs as batch_size=len(prompts), keeping the GPU continuously busy
    instead of idle between sequential single-prompt calls.
    """
    messages_list = [[{
        "role": "user",
        "content": [
            {"type": "video", "video": video_path, "nframes": N_VIDEO_FRAMES,
             "max_pixels": 360 * 420},
            {"type": "text", "text": pt},
        ],
    }] for pt in prompt_texts]

    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
             for m in messages_list]
    image_inputs, video_inputs = process_vision_info(messages_list)
    inputs = processor(text=texts, images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    gen_ids = gen_ids[:, inputs.input_ids.shape[1]:]
    outs = processor.batch_decode(gen_ids, skip_special_tokens=True)
    return [o.strip() for o in outs]


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
            t0 = time.time()
            prompt_texts = [format_prompt(p, item.get("options"), item["category"])
                            for p in item["prompts"]]
            try:
                answers = run_item_batched(model, processor, item["video_path"], prompt_texts)
            except torch.cuda.OutOfMemoryError:
                # Fall back to one-at-a-time if the batch doesn't fit
                torch.cuda.empty_cache()
                answers = []
                for pt in prompt_texts:
                    try:
                        a = run_item_batched(model, processor, item["video_path"], [pt])[0]
                    except Exception as e:
                        a = f"<ERROR: {e}>"
                    answers.append(a)
            except Exception as e:
                answers = [f"<ERROR: {e}>"] * len(prompt_texts)

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

            # Free GPU cache between items so it doesn't bloat over time.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        INFERENCE_FILE.write_text(json.dumps(results, indent=2, default=str))
        print(f"Wrote {len(results)} items -> {INFERENCE_FILE}")


if __name__ == "__main__":
    main()
