# Preprocessing Script for the CLEVR Dataset JSON file
import json
from pathlib import Path
from tqdm import tqdm

# === Constants ===
BASE_DIR = Path("/share/data/speech/txu/vlm_semantics/data")
CLEVR_DIR = BASE_DIR / "CLEVR_v1.0"
CLEVR_QUESTIONS_FILE = CLEVR_DIR / "questions/simple_CLEVR_val_questions.json"
CLEVR_IMG_DIR = CLEVR_DIR / "images/val"

# Output JSON file
CLEVR_JSON_FILE = BASE_DIR / "simple_clevr_val_preprocessed.json"


def build_clevr() -> None:
    print("[*] Downloading and Formatting `CLEVR` Dataset!")

    # Open CLEVR Questions JSON File
    with open(CLEVR_QUESTIONS_FILE, "r") as f:
        clevr_data = json.load(f)
        clevr_examples = clevr_data["questions"]

    # Verify Image Existence
    for example in tqdm(clevr_examples, desc="[*] Verifying all CLEVR Images"):
        image_filename = example["image_filename"]
        assert (CLEVR_IMG_DIR / image_filename).exists(), f"Missing Image `{image_filename}`"

    # Reformat CLEVR Examples as LLaVa "Chat" Style
    clevr_chat_json = []
    for clevr_example in tqdm(clevr_examples, desc="[*] Converting CLEVR Examples to LLaVa Format"):
        clevr_chat_json.append(
            {
                "id": f"clevr_{clevr_example['question_index']}",
                "image": f"val/{clevr_example['image_filename']}",
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
    build_clevr()
