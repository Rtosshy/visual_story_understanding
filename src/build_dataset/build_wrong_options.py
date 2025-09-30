import json
import sys
from time import sleep

from openai import OpenAI
from tqdm import tqdm

from src.utils.paths import ORIGINAL_ROOT
from src.utils.text_processor import TextProcessor as tp
from src.utils.utils import env

MODEL = "gpt-4o"
OPENAI_API_KEY = env(key="OPENAI_API_KEY", required=True)
client = OpenAI(api_key=OPENAI_API_KEY)


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: uv run -m src.build_dataset.build_wrong_options <num_wrong_options> <target_pos>",
            file=sys.stderr,
        )
        sys.exit(1)
    num_wrong_options = int(sys.argv[1])
    target_pos = int(sys.argv[2])
    sis_base_path = ORIGINAL_ROOT / "sis_base.jsonl"
    data = []
    with open(sis_base_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            data.append(item)

    out_path = (
        ORIGINAL_ROOT
        / "text_option"
        / f"pos{target_pos}"
        / f"wrong_options_{num_wrong_options}.jsonl"
    )
    f_out = out_path.open("a", encoding="utf-8")

    for sample in tqdm(data):
        messages = tp.convert_to_build_wrong_option_template(
            sample=sample, num_wrong_options=num_wrong_options, target_pos=target_pos
        )

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0,
            max_tokens=512,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        generated = response.choices[0].message.content
        generated = generated.split("json")[1] if "json" in generated else generated
        generated = generated.split("```")[0] if "```" in generated else generated
        generated = generated.strip()

        record = {
            "story_id": sample["story_id"],
            "image_ids": sample["image_ids"],
            "texts": sample["texts"],
            "target_pos": target_pos,
            "num_wrong_options": num_wrong_options,
            "wrong_options": json.loads(generated),
        }
        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("sleeping 10 seconds to avoid rate limits...")
        sleep(10)

    f_out.close()


if __name__ == "__main__":
    main()
