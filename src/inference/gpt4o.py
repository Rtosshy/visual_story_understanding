import json
import re

from openai import OpenAI
from tqdm import tqdm

from src.dataset import get_vist_dataset
from src.paths import ORIGINAL_ROOT, OUTPUT_ROOT
from src.utils import convert_to_messages, env

MODEL="gpt-4o"
OPENAI_API_KEY = env(key="OPENAI_API_KEY", required=True)
client = OpenAI(api_key=OPENAI_API_KEY)

# response = client.chat.completions.create(
#     model=MODEL,
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant. Answer the following question."},
#         {"role": "user", "content": [
#             {"type": "text", "text": "What is the women wearing?"},
#             {
#                 "type": "image_url", "image_url": {
#                     "url": f"data:image/jpg;base64,{base64_image}"
#                 }
#             }
#         ]}
#     ],
# )

# print("Assistant: " + response.choices[0].message.content)

if __name__ == "__main__":
    dataset_path = ORIGINAL_ROOT / "seq2opt.jsonl"
    dataset = get_vist_dataset(dataset_path)

    out_path = OUTPUT_ROOT / "gpt4o.jsonl"
    f_out = out_path.open("w", encoding="utf-8")
    
    for data in tqdm(dataset):
        msg = convert_to_messages(data)

        response = client.chat.completions.create(
            model=MODEL,
            messages=msg,
        )
        
        generated = response.choices[0].message.content

        m = re.search(r"\d+", generated)
        if m:
            pred = m.group()
        else:
            pred = None

        record = {
            "story_id": data["story_id"],
            "question": data["question"],
            "answer": data["answer"],
            "option": data["option"],
            "answer_idx": data["answer_idx"],
            "drop_pos": data["drop_pos"],
            "generated": generated,
            "pred": pred,
        }
        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    f_out.close()