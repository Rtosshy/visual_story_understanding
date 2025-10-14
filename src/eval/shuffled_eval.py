import json
import re
from pathlib import Path
from typing import List

import numpy as np
from scipy.stats import kendalltau

from src.utils.paths import OUTPUT_ROOT


def extract_array_str(text: str) -> str:
    if not text:
        return ""

    match = re.search(r"\[\s*-?\d+(?:\s*,\s*-?\d+)*\s*\]", text)
    return match.group(0) if match else ""


def convert_to_list(array_str) -> List[int]:
    if not array_str:
        return []

    try:
        data = json.loads(array_str)
        if isinstance(data, list) and all(isinstance(item, int) for item in data):
            return data
    except json.JSONDecodeError:
        pass

    return [int(num) for num in re.findall(r"-?\d+", array_str)]


def evaluate(jsonl_path: Path) -> None:
    total = 0
    num_exact_match = 0
    error_samples = 0
    error_indices: List[int] = []
    tau_list: List[float] = []
    pvalue_list: List[float] = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            answer = data.get("answer")
            generated = data.get("generated")
            array_str = extract_array_str(generated)
            pred = convert_to_list(array_str)

            total += 1
            if pred is None:
                error_samples += 1
                error_indices.append(idx)
                continue

            if len(pred) < 2 or not isinstance(answer, list) or len(answer) < 2:
                continue

            if pred == answer:
                num_exact_match += 1

            tau, pvalue = kendalltau(pred, answer)
            tau_list.append(tau)
            pvalue_list.append(pvalue)

    # 結果出力
    valid_tau = [tau for tau in tau_list if not np.isnan(tau)]
    mean_tau = np.mean(valid_tau) if valid_tau else float("nan")
    tau_nan_count = int(np.sum(np.isnan(tau_list)))
    print(f"総データ数           : {total}")
    print(f"完全一致数           : {num_exact_match}")
    print(f"平均tau             : {mean_tau}")
    print(f"tauのNaN数          : {tau_nan_count}")
    print(f"エラーサンプル数      : {error_samples}")
    print(f"エラーサンプルインデックス: {error_indices}")


if __name__ == "__main__":
    jsonl_path = OUTPUT_ROOT / "shuffled_text" / "gpt4o.jsonl"
    evaluate(jsonl_path)
