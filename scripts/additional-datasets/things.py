# Preprocessing Script for the THINGS Dataset JSON file
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# === Constants ===
BASE_DIR = Path("/share/data/speech/txu/vlm_semantics/data")
THINGS_DIR = BASE_DIR / "hypernymy_THINGS"
TRAIN_SPLIT = 0.7
VALID_SPLIT = 0.05

def extract_things_metadata():
    print(f"[*] Writing `THINGS` Metadata!")
    THINGS_QUESTIONS_FILE = THINGS_DIR / "things.jsonl"
    THINGS_METADATA_FILE = THINGS_DIR / "things_metadata.json"
    
    things_metadata = defaultdict(dict)
    
    with open(THINGS_QUESTIONS_FILE, "r") as f:
        for line in f:
            item = json.loads(line)
            concept_id = item["concept_id"]

            if item["q_type"] == "positive_sample":
                things_metadata[concept_id]["id"] = concept_id
                things_metadata[concept_id]["concept"] = item["concept"]
            elif item["q_type"] == "negative_sample":
                things_metadata[concept_id]["negative_concept"] = item["concept"]
            else:
                raise ValueError(f"Unknown q_type: {item['q_type']}")

    # Convert defaultdict to list of dicts
    things_metadata = [things_metadata[k] for k in things_metadata.keys()]
    
    with open(THINGS_METADATA_FILE, "w") as f:
        json.dump(things_metadata, f)
    
    print(f"[*] Successfully created THINGS metadata with {len(things_metadata)} entries!")


def build_things():
    print(f"[*] Downloading and Formatting `THINGS` Dataset!")
    # THINGS_HYP_QUESTIONS_FILE = THINGS_DIR / f"things-hyp.jsonl"
    THINGS_QUESTIONS_FILE = THINGS_DIR / f"things.jsonl"
    THINGS_IMG_DIR = THINGS_DIR / "images"
    THINGS_METADATA_FILE = THINGS_DIR / "things_metadata.json"

    # Open THINGS Questions JSONL File (read line by line)
    things_examples = []
    with open(THINGS_QUESTIONS_FILE, "r") as f:
        for line in f:
            things_examples.append(json.loads(line))

    # Verify Subdir Existence
    for example in tqdm(things_examples, desc="[*] Verifying all THINGS Categories"):
        image_dir = example["concept"]
        img_path = THINGS_IMG_DIR / image_dir
        if not img_path.exists():
            print(f"Warning: Missing Image Category `{image_dir}`")
    
    # Load THINGS Metadata
    metadata = json.load(open(THINGS_METADATA_FILE, "r"))
    neg_concept_map = {item["negative_concept"]: item["concept"] for item in metadata}

    # Reformat THINGS Examples as LLaVa "Chat" Style
    things_train_json = []
    things_val_json = []
    things_test_json = []
    for _, things_example in enumerate(tqdm(things_examples, desc="[*] Converting THINGS Examples to LLaVa Format")):
        concept = things_example["concept"] if things_example["q_type"] == "positive_sample" else neg_concept_map[things_example["concept"]]
        images_filenames = os.listdir(THINGS_IMG_DIR / concept)
        for fname in images_filenames:
            if not fname.endswith(('.jpg', '.jpeg', '.png')):
                print(f"Warning: Skipping non-image file `{fname}`")
                images_filenames.remove(fname)
        num_images = len(images_filenames) # Count number of images in concept directory
        split_indices = [int(num_images * TRAIN_SPLIT), int(num_images * (TRAIN_SPLIT + VALID_SPLIT))] # (txu =>>) This is a wrong line as some categories don't exist in val split
        # For each question, create entries for train/val/test splits with all the images in the concept directory
        for img_idx in range(split_indices[0]):
            things_train_json.append(
                {
                    "id": things_example["concept_id"],
                    "image": f"{concept}/{images_filenames[img_idx]}",
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{things_example['question'].strip()}"},
                        {"from": "gpt", "value": things_example['label'].strip()}
                    ],
                }
            )
        for img_idx in range(split_indices[0], split_indices[1]):
            things_val_json.append(
                {
                    "id": things_example["concept_id"],
                    "image": f"{concept}/{images_filenames[img_idx]}",
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{things_example['question'].strip()}"},
                        {"from": "gpt", "value": things_example['label'].strip()}
                    ],
                }
            )
        for img_idx in range(split_indices[1], num_images):
            things_test_json.append(
                {
                    "id": things_example["concept_id"],
                    "image": f"{concept}/{images_filenames[img_idx]}",
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{things_example['question'].strip()}"},
                        {"from": "gpt", "value": things_example['label'].strip()}
                    ],
                }
            )

    with open(BASE_DIR / "preprocessed_THINGS" / "train.json", "w") as f:
        json.dump(things_train_json, f)
    with open(BASE_DIR / "preprocessed_THINGS" / "val.json", "w") as f:
        json.dump(things_val_json, f)
    with open(BASE_DIR / "preprocessed_THINGS" / "test.json", "w") as f:
        json.dump(things_test_json, f)

    print(f"[*] Successfully created dataset with {len(things_train_json)} train, {len(things_val_json)} val, and {len(things_test_json)} test examples!")


def build_things_hyp():
    print(f"[*] Downloading and Formatting `THINGS_HYP` Dataset!")
    # THINGS_HYP_QUESTIONS_FILE = THINGS_DIR / f"things-hyp.jsonl"
    THINGS_QUESTIONS_FILE = THINGS_DIR / f"things-hyp.jsonl"
    THINGS_IMG_DIR = THINGS_DIR / "images"

    # Open THINGS Questions JSONL File (read line by line)
    things_examples = []
    with open(THINGS_QUESTIONS_FILE, "r") as f:
        for line in f:
            things_examples.append(json.loads(line))

    # Verify Subdir Existence
    for example in tqdm(things_examples, desc="[*] Verifying all THINGS Categories"):
        image_dir = example["hypernym"]
        img_path = THINGS_IMG_DIR / image_dir
        if not img_path.exists():
            print(f"Warning: Missing Hypernym Category `{image_dir}`")

    # Reformat THINGS Examples as LLaVa "Chat" Style
    things_hyp_train_json = []
    things_hyp_val_json = []
    things_hyp_test_json = []
    for _, things_example in enumerate(tqdm(things_examples, desc="[*] Converting THINGS Examples to LLaVa Format")):
        concept = things_example["concept"]
        images_filenames = os.listdir(THINGS_IMG_DIR / concept)
        for fname in images_filenames:
            if not fname.endswith(('.jpg', '.jpeg', '.png')):
                print(f"Warning: Skipping non-image file `{fname}`")
                images_filenames.remove(fname)
        num_images = len(images_filenames) # Count number of images in concept directory
        split_indices = [int(num_images * TRAIN_SPLIT), int(num_images * (TRAIN_SPLIT + VALID_SPLIT))]
        for img_idx in range(split_indices[0]):
            things_hyp_train_json.append(
                {
                    "id": things_example["concept_id"],
                    "hypernym_id": things_example["hypernym_id"],
                    "image": f"{concept}/{images_filenames[img_idx]}",
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{things_example['question'].strip()}"},
                        {"from": "gpt", "value": things_example['label'].strip()}
                    ],
                }
            )
        for img_idx in range(split_indices[0], split_indices[1]):
            things_hyp_val_json.append(
                {
                    "id": things_example["concept_id"],
                    "hypernym_id": things_example["hypernym_id"],
                    "image": f"{concept}/{images_filenames[img_idx]}",
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{things_example['question'].strip()}"},
                        {"from": "gpt", "value": things_example['label'].strip()}
                    ],
                }
            )
        for img_idx in range(split_indices[1], num_images):
            things_hyp_test_json.append(
                {
                    "id": things_example["concept_id"],
                    "hypernym_id": things_example["hypernym_id"],
                    "image": f"{concept}/{images_filenames[img_idx]}",
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{things_example['question'].strip()}"},
                        {"from": "gpt", "value": things_example['label'].strip()}
                    ],
                }
            )

    with open(BASE_DIR / "preprocessed_THINGS" / "train_hyp.json", "w") as f:
        json.dump(things_hyp_train_json, f)
    with open(BASE_DIR / "preprocessed_THINGS" / "val_hyp.json", "w") as f:
        json.dump(things_hyp_val_json, f)
    with open(BASE_DIR / "preprocessed_THINGS" / "test_hyp.json", "w") as f:
        json.dump(things_hyp_test_json, f)

    print(f"[*] Successfully created dataset with {len(things_hyp_train_json)} train, {len(things_hyp_val_json)} val, and {len(things_hyp_test_json)} test examples!")


if __name__ == "__main__":
    # extract_things_metadata()
    build_things_hyp()