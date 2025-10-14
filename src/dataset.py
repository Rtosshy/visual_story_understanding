import json
import random

import pandas as pd
from torch.utils.data import Dataset

from src.utils.paths import VIST_JSON_ROOT
from src.utils.text_processor import TextProcessor as tp
from src.utils.utils import load_json


class Seq2optDataset(Dataset):
    def __init__(self, jsonl_path):
        self.data = []
        # JSONL を１行ずつ読み込む
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                self.data.append(
                    {
                        "story_id": item.get("story_id"),
                        "question": item.get("question"),
                        "answer": item.get("answer"),
                        "option": item.get("option"),
                        "answer_idx": item.get("answer_idx"),
                        "drop_pos": item.get("drop_pos"),
                    }
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_seq2opt_dataset(jsonl_path):
    return Seq2optDataset(jsonl_path)


def build_seq2opt_dataset(drop_pos: int = 2, n_options: int = 4):
    sis_test = load_json(VIST_JSON_ROOT / "sis" / "test.story-in-sequence.json")
    sis_test_annotations_df = pd.DataFrame(sis_test["annotations"])
    sis_test_annotations_df = sis_test_annotations_df[0].apply(pd.Series)

    sis_test_annotations_df["text"] = sis_test_annotations_df["text"].apply(
        tp.convert_numbers_to_words
    )

    answer_dict = {}
    modified_dict = {}

    for story_id, group in sis_test_annotations_df.groupby("story_id"):
        group = group.sort_values("worker_arranged_photo_order")
        # 答え（order==drop_posのもの）
        answer_row = group[group["worker_arranged_photo_order"] == drop_pos]
        if not answer_row.empty:
            ans_photo_id = answer_row.iloc[0]["photo_flickr_id"]
            ans_text = answer_row.iloc[0]["text"]
            answer_dict[story_id] = {ans_photo_id: ans_text}
        # modified_dictにはdrop_posのものを入れない
        photo_text_dict = {
            row["photo_flickr_id"]: row["text"]
            for _, row in group.iterrows()
            if row["worker_arranged_photo_order"] != drop_pos
        }
        modified_dict[story_id] = photo_text_dict

    options_dict = {}
    answer_index_dict = {}

    n_options = 4  # 選択肢数
    seed = 42
    random.seed(seed)  # 一度だけseedを設定

    # 全storyのphoto_flickr_id-textペアを集める（重複除外）
    all_photo_text = set()
    for d in modified_dict.values():
        all_photo_text.update(d.items())
    all_photo_text = list(all_photo_text)

    for story_id in answer_dict.keys():
        correct_pair = list(answer_dict[story_id].items())[0]  # (photo_flickr_id, text)
        # 不正解候補（正解と同じphoto_flickr_id/textは除外）
        negatives = [
            item
            for item in all_photo_text
            if item[0] != correct_pair[0] and item[1] != correct_pair[1]
        ]
        sampled_neg = random.sample(negatives, min(n_options - 1, len(negatives)))
        # optionsリスト作成
        options = [correct_pair] + sampled_neg
        random.shuffle(options)
        # photo_flickr_id: text の辞書に変換
        options_dict[story_id] = {k: v for k, v in options}
        # 正解の位置をstory_id: indexで記録
        correct_index = [i for i, (k, _) in enumerate(options) if k == correct_pair[0]][
            0
        ]
        answer_index_dict[story_id] = correct_index


class ShuffledTextDataset(Dataset):
    def __init__(self, jsonl_path):
        self.data = []
        # JSONL を１行ずつ読み込む
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                self.data.append(
                    {
                        "story_id": item.get("story_id"),
                        "image_ids": item.get("image_ids"),
                        "texts": item.get("texts"),
                        "shuffled_texts": item.get("shuffled_texts"),
                        "answer": item.get("answer"),
                    }
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_shuffled_text_dataset(jsonl_path):
    return ShuffledTextDataset(jsonl_path)


class ShuffledImageDataset(Dataset):
    def __init__(self, jsonl_path):
        self.data = []
        # JSONL を１行ずつ読み込む
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                self.data.append(
                    {
                        "story_id": item.get("story_id"),
                        "image_ids": item.get("image_ids"),
                        "texts": item.get("texts"),
                        "shuffled_image_ids": item.get("shuffled_image_ids"),
                        "answer": item.get("answer"),
                    }
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_shuffled_image_dataset(jsonl_path):
    return ShuffledImageDataset(jsonl_path)
