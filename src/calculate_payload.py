import json
from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from src.dataset import get_seq2opt_dataset
from src.utils import convert_to_messages
from utils.paths import ORIGINAL_ROOT, OUTPUT_ROOT

MODEL = "gpt-4o"
THRESHOLD_MB = 10.0  # 判定しきい値


@dataclass
class PayloadRecord:
    story_id: Any
    bytes_len: int
    mb: float
    ok: bool
    error: Optional[str] = None


def calc_bytes(messages: List[Dict[str, Any]]) -> int:
    # 実際の送信に近い形：UTF-8バイト数を計測
    payload_obj = {"model": MODEL, "messages": messages}
    payload_str = json.dumps(payload_obj, ensure_ascii=False, separators=(",", ":"))
    return len(payload_str.encode("utf-8"))


def main():
    # 出力先を準備
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_ROOT / "payload_report.tsv"

    dataset_path = ORIGINAL_ROOT / "seq2opt.jsonl"
    dataset = get_seq2opt_dataset(dataset_path)

    records: List[PayloadRecord] = []
    exceed_count = 0
    ok_sizes: List[int] = []

    for data in tqdm(dataset, desc="calc payload"):
        story_id = data.get("story_id")
        try:
            messages = convert_to_messages(data)
            bytes_len = calc_bytes(messages)
            mb = bytes_len / (1024 * 1024)
            ok = True
            err = None
            if mb >= THRESHOLD_MB:
                exceed_count += 1
            else:
                ok_sizes.append(bytes_len)
        except Exception as e:
            bytes_len = -1
            mb = -1.0
            ok = False
            err = str(e)

        records.append(
            PayloadRecord(
                story_id=story_id, bytes_len=bytes_len, mb=mb, ok=ok, error=err
            )
        )

    # TSVに保存
    with report_path.open("w", encoding="utf-8") as f:
        f.write("story_id\tbytes\tmb\tok\terror\n")
        for r in records:
            f.write(
                f"{r.story_id}\t{r.bytes_len}\t{r.mb:.2f}\t{r.ok}\t{(r.error or '-')}\n"
            )

        # サマリも末尾に追記
        total = len(records)
        ok_count = sum(1 for r in records if r.ok)
        fail_count = total - ok_count
        max_rec = max(
            (r for r in records if r.ok), key=lambda x: x.bytes_len, default=None
        )
        min_rec = min(
            (r for r in records if r.ok), key=lambda x: x.bytes_len, default=None
        )
        avg_mb = (mean([b for b in ok_sizes]) / (1024 * 1024)) if ok_sizes else 0.0
        f.write("# SUMMARY\n")
        f.write(f"# total\t{total}\n")
        f.write(f"# ok\t{ok_count}\n")
        f.write(f"# fail\t{fail_count}\n")
        f.write(f"# exceed_{int(THRESHOLD_MB)}MB\t{exceed_count}\n")
        if max_rec:
            f.write(
                f"# max_ok\t{max_rec.story_id}\t{max_rec.bytes_len}\t{max_rec.mb:.2f}MB\n"
            )
        if min_rec:
            f.write(
                f"# min_ok\t{min_rec.story_id}\t{min_rec.bytes_len}\t{min_rec.mb:.2f}MB\n"
            )
        f.write(f"# avg_ok\t{avg_mb:.2f}MB\n")

    # コンソールにも主要サマリを表示
    total = len(records)
    ok_count = sum(1 for r in records if r.ok)
    fail_count = total - ok_count
    print("==== SUMMARY ====")
    print(f"total: {total}")
    print(f"ok: {ok_count}")
    print(f"fail: {fail_count}")
    print(f"exceed >= {THRESHOLD_MB:.0f}MB: {exceed_count}")
    if records and any(r.ok for r in records):
        max_rec = max((r for r in records if r.ok), key=lambda x: x.bytes_len)
        print(f"max_ok: story_id={max_rec.story_id} size={max_rec.mb:.2f}MB")


if __name__ == "__main__":
    main()
