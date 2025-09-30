import json
import re

from tqdm import tqdm
from unsloth import FastVisionModel

from src.dataset import get_seq2opt_dataset
from src.utils.paths import ORIGINAL_ROOT, OUTPUT_ROOT
from src.utils.text_processor import TextProcessor as tp

num_param = 72


def main():
    model, tokenizer = FastVisionModel.from_pretrained(
        f"unsloth/Qwen2.5-VL-{num_param}B-Instruct-bnb-4bit",
        load_in_4bit=True,  # Use 4bit to reduce memory use. False for 16bit LoRA.
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for long context
    )

    FastVisionModel.for_inference(model)

    dataset_path = ORIGINAL_ROOT / "seq2opt.jsonl"
    dataset = get_seq2opt_dataset(dataset_path)

    out_path = OUTPUT_ROOT / f"qwen2-5-VL-{num_param}B.jsonl"
    f_out = out_path.open("a", encoding="utf-8")

    dataset = dataset[16:]

    for data in tqdm(dataset):
        message, images = tp.convert_to_qwen_template(data)

        input_text = tokenizer.apply_chat_template(message, add_generation_prompt=True)

        inputs = tokenizer(
            images=images,
            text=input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")

        output = model.generate(**inputs, max_new_tokens=16, use_cache=True)

        response = tokenizer.batch_decode(output, skip_special_tokens=True)
        response = response[0]

        generated = response.split("assistant")[-1]

        if generated is not None:
            m = re.search(r"\d+", generated)
            pred = m.group() if m else None

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


if __name__ == "__main__":
    main()
