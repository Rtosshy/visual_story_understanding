import json

from torch.utils.data import Dataset


class VistDataset(Dataset):
    def __init__(self, json_path):
        self.data = []
        with open(json_path, "r", encoding="utf-8") as f:
            all_data = json.load(f)
        for story_id, item in all_data.items():
            self.data.append(
                {
                    "story_id": story_id,
                    "question": item["question"],
                    "answer": item["answer"],
                    "option": item["option"],
                    "answer_idx": item["answer_idx"],
                    "drop_pos": item["drop_pos"],
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_vist_dataset(json_path):
    dataset = VistDataset(json_path)
    return dataset
