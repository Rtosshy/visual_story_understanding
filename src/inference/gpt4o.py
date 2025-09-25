import json
import re
from datetime import datetime  # 追加
from typing import Any  # 追加

from openai import OpenAI
from tqdm import tqdm

from src.dataset import get_seq2opt_dataset
from src.utils import convert_to_messages, env
from utils.paths import ORIGINAL_ROOT, OUTPUT_ROOT

MODEL = "gpt-4o"
OPENAI_API_KEY = env(key="OPENAI_API_KEY", required=True)
client = OpenAI(api_key=OPENAI_API_KEY)

# 追加: ペイロードログのパスとヘッダー初期化
PAYLOAD_LOG_PATH = OUTPUT_ROOT / "payload_stats.txt"


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
    _ensure_payload_log_header()  # 追加

    dataset_path = ORIGINAL_ROOT / "seq2opt.jsonl"
    dataset = get_seq2opt_dataset(dataset_path)

    out_path = OUTPUT_ROOT / "gpt4o.jsonl"
    f_out = out_path.open("a", encoding="utf-8")

    dataset = dataset[195:]

    max_success_bytes = 0
    max_success_story_id = None

    for data in tqdm(dataset):
        messages = convert_to_messages(data)

        # 追加: このリクエストのペイロードサイズ（概算）
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
            m = re.search(r"\d+", generated)
            pred = m.group() if m else None

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

    # 追加: 最大成功ペイロードのサマリを追記
    with PAYLOAD_LOG_PATH.open("a", encoding="utf-8") as f:
        if max_success_bytes > 0:
            f.write(
                f"# MAX_SUCCESS\t{datetime.now().isoformat()}\t{max_success_story_id}\t"
                f"{max_success_bytes}\t{max_success_bytes / 1024 / 1024:.2f}MB\n"
            )
        else:
            f.write(f"# MAX_SUCCESS\t{datetime.now().isoformat()}\t-\t0\t0.00MB\n")
