import json
from pathlib import Path

import numpy as np

from src.utils.paths import ORIGINAL_ROOT


def build_shuffle_text_data(base_path: Path, output_path: Path) -> None:
    data = []

    with open(base_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            texts = item.get("texts")

            indices = np.arange(len(texts))
            # index for shuffle
            shuffled_indices = np.random.permutation(indices)

            shuffled_texts = [texts[i] for i in shuffled_indices]

            # Index for restoration
            restore_indices = np.argsort(shuffled_indices)

            # print("index before shuffle:", indices)
            # print("index after shuffle:", shuffled_indices)
            # print("index for restoration:", restore_indices)
            # restored_texts = [shuffled_texts[i] for i in restore_indices]
            # confirmation
            # print(texts==restored_texts)

            data.append(
                {
                    "story_id": item.get("story_id"),
                    "image_ids": item.get("image_ids"),
                    "texts": texts,
                    "shuffled_texts": shuffled_texts,
                    "answer": restore_indices.tolist(),
                }
            )

    with open(output_path, "w", encoding="utf-8") as f_out:
        for item in data:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_shuffle_image_data(base_path: Path, output_path: Path) -> None:
    data = []
    with open(base_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            image_ids = item.get("image_ids")

            indices = np.arange(len(image_ids))
            # index for shuffle
            shuffled_indices = np.random.permutation(indices)

            shuffled_image_ids = [image_ids[i] for i in shuffled_indices]

            # Index for restoration
            restore_indices = np.argsort(shuffled_indices)

            # print("index before shuffle:", indices)
            # print("index after shuffle:", shuffled_indices)
            # print("index for restoration:", restore_indices)
            # restored_image_ids = [shuffled_image_ids[i] for i in restore_indices]
            # confirmation
            # print(image_ids == restored_image_ids)

            data.append(
                {
                    "story_id": item.get("story_id"),
                    "image_ids": image_ids,
                    "texts": item.get("texts"),
                    "shuffled_image_ids": shuffled_image_ids,
                    "answer": restore_indices.tolist(),
                }
            )

    with open(output_path, "w", encoding="utf-8") as f_out:
        for item in data:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    base_path = ORIGINAL_ROOT / "sis_base.jsonl"
    text_output_path = ORIGINAL_ROOT / "shuffle" / "text_option" / "shuffle_data.jsonl"
    image_output_path = (
        ORIGINAL_ROOT / "shuffle" / "image_option" / "shuffle_data.jsonl"
    )
    build_shuffle_text_data(base_path=base_path, output_path=text_output_path)
    build_shuffle_image_data(base_path=base_path, output_path=image_output_path)
