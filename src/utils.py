from pathlib import Path

from PIL import Image


def convert_to_conversation(sample):
    instruction = "You are given a sequence of images that tell a story, but one image is missing from the sequence. Please select the most appropriate text that describes what should happen in the missing image location. Consider the visual narrative flow and context of the surrounding images to choose the best option."
    content = [
        {"type": "text", "text": instruction},
        {"type": "text", "text": "\nSequence of images"},
    ]
    image_ids = list(sample["question"].keys())
    original_length = len(image_ids)
    test_image_path = Path(".") / "dataset" / "vist" / "image" / "test"

    # 問題文の構成
    is_missing_inserted = False
    for i in range(original_length + 1):
        if i == sample["drop_pos"] and not is_missing_inserted:
            # drop_posの位置でテキストを挿入
            content.append({"type": "text", "text": "[Missing Image Position]"})
            is_missing_inserted = True
            # drop_pos以降はインデックスをデクリメント
        elif is_missing_inserted:
            i -= 1
            image_path = test_image_path / f"{image_ids[i]}.jpg"
            image = Image.open(image_path)
            content.append({"type": "image", "image": image})
        else:
            # drop_posまでは普通に挿入
            image_path = test_image_path / f"{image_ids[i]}.jpg"
            image = Image.open(image_path)
            content.append({"type": "image", "image": image})

    # TODO:選択肢の構成
    conversation = [
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": "Answer"}]},
    ]
    return {"messages": conversation}
