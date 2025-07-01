# Preprocessing Script for the CLEVR Dataset JSON file
import json
import argparse
from pathlib import Path
from tqdm import tqdm

# === Constants ===
BASE_DIR = Path("/share/data/speech/txu/vlm_semantics/data")
CLEVR_DIR = BASE_DIR / "CLEVR_v1.0"


def build_clevr(split="train") -> None:
    print("[*] Downloading and Formatting `CLEVR` Dataset!")
    CLEVR_QUESTIONS_FILE = CLEVR_DIR / f"../generated_CLEVR/CLEVR_{split}_qa_mixed.json"
    CLEVR_IMG_DIR = CLEVR_DIR / f"images/{split}"
    CLEVR_JSON_FILE = BASE_DIR / f"preprocessed_CLEVR/clevr_{split}_qa_mixed_preprocessed.json"

    # Open CLEVR Questions JSON File
    with open(CLEVR_QUESTIONS_FILE, "r") as f:
        clevr_data = json.load(f)
        clevr_examples = clevr_data
        # clevr_examples = clevr_data["questions"]

    # Verify Image Existence
    for example in tqdm(clevr_examples, desc="[*] Verifying all CLEVR Images"):
        image_filename = example["image_filename"]
        assert (CLEVR_IMG_DIR / image_filename).exists(), f"Missing Image `{image_filename}`"

    # Reformat CLEVR Examples as LLaVa "Chat" Style
    clevr_chat_json = []
    for i, clevr_example in enumerate(tqdm(clevr_examples, desc="[*] Converting CLEVR Examples to LLaVa Format")):
        clevr_chat_json.append(
            {
                "id": f"clevr_{i}",
                "image": f"{split}/{clevr_example['image_filename']}",
                "conversations": [
                    {"from": "human", "value": f"<image>\n{clevr_example['question'].strip()}"},
                    {"from": "gpt", "value": clevr_example["answer"].strip()},
                ],
            }
        )

    with open(CLEVR_JSON_FILE, "w") as f:
        json.dump(clevr_chat_json, f)
    
    print(f"[*] Successfully created dataset with {len(clevr_chat_json)} examples!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess CLEVR dataset for LLaVa.")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], help="Dataset split to preprocess.")
    args = parser.parse_args()

    build_clevr(args.split)
