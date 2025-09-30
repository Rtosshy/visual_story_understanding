import json
from pprint import pprint

import pandas as pd

from src.utils.paths import ORIGINAL_ROOT, VIST_JSON_ROOT
from src.utils.text_processor import TextProcessor as tp
from src.utils.utils import load_json

if __name__ == "__main__":
    sis_test = load_json(VIST_JSON_ROOT / "sis" / "test.story-in-sequence.json")
    sis_test_annotations_df = pd.DataFrame(sis_test["annotations"])
    sis_test_annotations_df = sis_test_annotations_df[0].apply(pd.Series)

    sis_test_annotations_df["text"] = sis_test_annotations_df["text"].apply(
        tp.convert_numbers_to_words
    )

    ordered_annotations = (
        sis_test_annotations_df.sort_values(["story_id", "worker_arranged_photo_order"])
        .groupby("story_id")
        .apply(
            lambda df: {
                "image_ids": df["photo_flickr_id"].tolist(),
                "texts": df["text"].tolist(),
            }
        )
        .to_dict()
    )

    pprint(ordered_annotations)

    output_path = ORIGINAL_ROOT / "sis_base.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for story_id, rec in ordered_annotations.items():
            record = {"story_id": story_id, **rec}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
