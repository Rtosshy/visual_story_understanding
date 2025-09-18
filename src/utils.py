from pathlib import Path

from PIL import Image


def convert_to_conversation(sample):
    def load_image(image_id):
        for ext in ("jpg", "png"):
            img_path = test_image_path / f"{image_id}.{ext}"
            if img_path.exists():
                return Image.open(img_path)
        raise FileNotFoundError(f"No image found for {image_id} with .jpg or .png")

    instruction = "You are given a sequence of images that tell a story, but one image is missing. Choose only the number of the most appropriate option that describes what should happen in the missing image location. Respond with just the index (e.g., 0, 1, 2, ...)."
    content = [
        {"type": "text", "text": instruction},
        {"type": "text", "text": "\n\nSequence of images:\n"},
    ]

    image_ids = list(sample["question"].keys())
    original_length = len(image_ids)
    test_image_path = Path(".") / "dataset" / "vist" / "image" / "test"

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
    return {"messages": conversation}
