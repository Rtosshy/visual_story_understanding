import json

from torch.utils.data import Dataset


class VistDataset(Dataset):
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


def get_vist_dataset(jsonl_path):
    return VistDataset(jsonl_path)
