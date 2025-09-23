import base64
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from num2words import num2words
from PIL import Image

from src.paths import DOTENV_PATH, VIST_IMAGE_ROOT

if DOTENV_PATH.exists():
    load_dotenv(DOTENV_PATH)


def env(key: str, default: str | None = None, required: bool = False) -> str | None:
    value = os.getenv(key, default)
    if required and value is None:
        raise RuntimeError(f"Required environment variable '{key}' not set.")
    return value


def load_image(image_id):
    for ext in ("jpg", "png"):
        img_path = VIST_IMAGE_ROOT / "test" / f"{image_id}.{ext}"
        if img_path.exists():
            return Image.open(img_path)
    raise FileNotFoundError(f"No image found for {image_id} with .jpg or .png")


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def convert_numbers_to_words(text: str) -> str:
    # 連続する数字をまとめて変換
    return re.sub(r"\d+", lambda m: num2words(int(m.group()), lang="en"), text)


def encode_image_to_base64(image_path: Path) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def convert_to_conversation(sample):
    instruction = "You are given a sequence of images that tell a story, but one image is missing. Choose only the number of the most appropriate option that describes what should happen in the missing image location. Respond with just the index (e.g., 0, 1, 2, ...)."
    content = [
        {"type": "text", "text": instruction},
        {"type": "text", "text": "\n\nSequence of images:\n"},
    ]

    image_ids = list(sample["question"].keys())
    original_length = len(image_ids)

    # 問題文の構成
    is_missing_inserted = False
    for i in range(original_length + 1):
        if i == sample["drop_pos"] and not is_missing_inserted:
            content.append({"type": "text", "text": "[Missing Image Position]"})
            is_missing_inserted = True
        else:
            if is_missing_inserted:
                idx = i - 1
            else:
                idx = i

            # 変更: load_image を使って拡張子をフォールバック
            image = load_image(image_ids[idx])
            content.append({"type": "image", "image": image})

    # 選択肢の構成 テキスト
    options = list(sample["option"].values())
    content.append({"type": "text", "text": "\n\nOptions:"})
    for i, option in enumerate(options):
        content.append({"type": "text", "text": f"\n{i}. {option}"})

    conversation = [
        {"role": "user", "content": content},
    ]
    return conversation


def convert_to_messages(sample):
    instruction = "You are given a sequence of images that tell a story, but one image is missing. Choose only the number of the most appropriate option that describes what should happen in the missing image location. Respond with just the index (e.g., 0, 1, 2, ...)."
    content = [
        {"type": "text", "text": instruction},
        {"type": "text", "text": "\n\nSequence of images:\n"},
    ]

    image_ids = list(sample["question"].keys())
    original_length = len(image_ids)

    is_missing_inserted = False
    for i in range(original_length + 1):
        if i == sample["drop_pos"] and not is_missing_inserted:
            content.append({"type": "text", "text": "[Missing Image Position]"})
            is_missing_inserted = True
        else:
            if is_missing_inserted:
                idx = i - 1
            else:
                idx = i

    options = list(sample["option"].values())
    content.append({"type": "text", "text": "\n\nOptions:"})
    for i, option in enumerate(options):
        content.append({"type": "text", "text": f"\n{i}. {option}"})

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer the following question.",
        },
        {"role": "user", "content": content},
    ]

    return messages
