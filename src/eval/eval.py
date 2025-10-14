import json
from pathlib import Path

from src.utils.paths import OUTPUT_ROOT


def evaluate(jsonl_path: Path) -> None:
    total = 0  # 総データ数
    correct = 0  # 正しく予測できた数
    wrong = 0  # 誤予測の数（pred=None も含む）
    none_pred = 0  # pred が None の数
    wrong_non_none = 0  # pred が None 以外で誤った数
    correct_indices = []  # 正解した行番号
    none_pred_indices = []  # pred=None の行番号
    wrong_non_none_indices = []  # pred があり非Noneで誤った行番号

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            answer_idx = data.get("answer_idx")
            pred = data.get("pred")

            total += 1

            # pred が None の場合は必ず誤りとしてカウント
            if pred is None:
                none_pred += 1
                wrong += 1
                none_pred_indices.append(idx)
                continue

            # pred が文字列の数値で来ている想定なので int に変換
            try:
                pred_idx = int(pred)
            except (ValueError, TypeError):
                none_pred += 1
                wrong += 1
                none_pred_indices.append(idx)
                continue

            # 正誤判定
            if pred_idx == answer_idx:
                correct += 1
                correct_indices.append(idx)
            else:
                wrong += 1
                wrong_non_none += 1
                wrong_non_none_indices.append(idx)

    # 結果出力
    accuracy = correct / total if total > 0 else 0.0
    print(f"総データ数           : {total}")
    print(f"正解数               : {correct}")
    print(f"誤り数               : {wrong} (うち pred=None: {none_pred})")
    print(f"正答率               : {accuracy:.2%}")
    print(f"正解の行番号          : {correct_indices}")
    print(f"pred=None の行番号    : {none_pred_indices}")
    print(f"誤答(非None)の行番号   : {wrong_non_none_indices}")


if __name__ == "__main__":
    # command line argument
    jsonl_path = OUTPUT_ROOT / "qwen2-5-VL-72B.jsonl"
    evaluate(jsonl_path)
