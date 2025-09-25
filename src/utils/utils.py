import base64
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from num2words import num2words
from PIL import Image

from utils.paths import DOTENV_PATH, VIST_IMAGE_ROOT

if DOTENV_PATH.exists():
    load_dotenv(DOTENV_PATH)


def env(key: str, default: str | None = None, required: bool = False) -> str | None:
    value = os.getenv(key, default)
    if required and value is None:
        raise RuntimeError(f"Required environment variable '{key}' not set.")
    return value


IMAGE_EXTS = ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG")


class ImageProcessor:
    pass


def id_to_path(image_id: str) -> Path:
    test_dir = VIST_IMAGE_ROOT / "test"
    for ext in IMAGE_EXTS:
        image_path = test_dir / f"{image_id}.{ext}"
        if image_path.exists():
            return image_path
        raise FileNotFoundError(
            f"No image found for {image_id} with extensions: {IMAGE_EXTS}"
        )


def load_image(image_path: Path) -> Image.Image:
    return Image.open(image_path)


def encode_image_to_base64(image_path: Path) -> str:
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
        base64_string = base64.b64encode(image_bytes).decode("utf-8")
        return base64_string


def encode_image_to_url(image_path: Path) -> str:
    with open(image_path, "rb") as image_file:
        image = Image.open(image_path)
        format = image.format.lower()
        mime = f"image/{'jpeg' if format in ['jpg', 'jpeg'] else format}"

        image_bytes = image_file.read()
        base64_string = base64.b64encode(image_bytes).decode("utf-8")
        url = f"data:{mime};base64,{base64_string}"
        return url


def load_json(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


class TextProcessor:
    pass


def convert_numbers_to_words(text: str) -> str:
    # 連続する数字をまとめて変換
    return re.sub(r"\d+", lambda m: num2words(int(m.group()), lang="en"), text)


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
            image_path = id_to_path(image_id=image_ids[idx])
            image = load_image(image_path=image_path)
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
            idx = i - 1 if is_missing_inserted else i
            image_path = id_to_path(image_id=image_ids[idx])
            url = encode_image_to_url(image_path=image_path)
            content.append(
                {"type": "image_url", "image_url": {"url": url, "detail": "low"}}
            )

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
