import json
from datetime import datetime
from time import sleep
from typing import Any

from openai import OpenAI
from tqdm import tqdm

from src.dataset import get_shuffled_text_dataset
from src.utils.paths import ORIGINAL_ROOT, OUTPUT_ROOT
from src.utils.text_processor import TextProcessor as tp
from src.utils.utils import env

MODEL = "gpt-4o"
OPENAI_API_KEY = env(key="OPENAI_API_KEY", required=True)
client = OpenAI(api_key=OPENAI_API_KEY)

PAYLOAD_LOG_PATH = OUTPUT_ROOT / "shuffled_text" / "payload_stats.txt"


def _ensure_payload_log_header():
    if not PAYLOAD_LOG_PATH.exists():
        with PAYLOAD_LOG_PATH.open("w", encoding="utf-8") as f:
            f.write("timestamp\tstory_id\tbytes\tmb\tsuccess\terror\n")


def _append_payload_log(
    story_id: Any,
    bytes_len: int,
    success: bool,
    error: str | None = None,
):
    mb = bytes_len / (1024 * 1024)
    ts = datetime.now().isoformat()
    with PAYLOAD_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(
            f"{ts}\t{story_id}\t{bytes_len}\t{mb:.2f}\t{success}\t{(error or '-')}\n"
        )


if __name__ == "__main__":
    _ensure_payload_log_header()
    dataset_path = ORIGINAL_ROOT / "shuffle" / "text_option" / "shuffle_data.jsonl"

    dataset = get_shuffled_text_dataset(dataset_path)
    output_path = OUTPUT_ROOT / "shuffled_text" / "gpt4o.jsonl"
    f_out = output_path.open("a", encoding="utf-8")

    max_success_bytes = 0
    max_success_story_id = None

    for data in tqdm(dataset[:100]):
        messages = tp.convert_to_shuffled_text_template(sample=data)

        try:
            payload_bytes = len(json.dumps({"model": MODEL, "messages": messages}))
        except Exception:
            payload_bytes = -1  # 失敗時の保険

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
            )
            generated = response.choices[0].message.content

            # ログ: 成功
            _append_payload_log(
                story_id=data["story_id"],
                bytes_len=payload_bytes,
                success=True,
                error=None,
            )
            if payload_bytes >= 0 and payload_bytes > max_success_bytes:
                max_success_bytes = payload_bytes
                max_success_story_id = data["story_id"]

        except Exception as e:
            # ログ: 失敗
            _append_payload_log(
                story_id=data["story_id"],
                bytes_len=payload_bytes,
                success=False,
                error=str(e),
            )
            print(f"[error] story_id {data['story_id']} failed: {e}")
            generated = None

        record = {
            "story_id": data["story_id"],
            "image_ids": data["image_ids"],
            "texts": data["texts"],
            "shuffled_texts": data["shuffled_texts"],
            "answer": data["answer"],
            "generated": generated,
        }
        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
        print("sleeping 20 seconds to avoid rate limits...")
        sleep(20)

    f_out.close()

    with PAYLOAD_LOG_PATH.open("a", encoding="utf-8") as f:
        if max_success_bytes > 0:
            f.write(
                f"# MAX_SUCCESS\t{datetime.now().isoformat()}\t{max_success_story_id}\t"
                f"{max_success_bytes}\t{max_success_bytes / 1024 / 1024:.2f}MB\n"
            )
        else:
            f.write(f"# MAX_SUCCESS\t{datetime.now().isoformat()}\t-\t0\t0.00MB\n")
