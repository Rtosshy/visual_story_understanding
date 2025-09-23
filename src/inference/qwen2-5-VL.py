from tqdm import tqdm
from unsloth import FastVisionModel

from src.dataset import get_seq2opt_dataset
from src.paths import ORIGINAL_ROOT
from src.utils import convert_to_conversation

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
    load_in_4bit=True,  # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for long context
)

FastVisionModel.for_inference(model)

if __name__ == "__main__":
    dataset_path = ORIGINAL_ROOT / "seq2opt.jsonl"
    dataset = get_seq2opt_dataset(dataset_path)

    # out_path = OUTPUT_ROOT / "qwen2-5-VL.jsonl"
    # f_out = out_path.open("w", encoding="utf-8")

    for data in tqdm(dataset):
        conv = convert_to_conversation(data)

        input_text = tokenizer.apply_chat_template(conv, add_generation_prompt=True)

        print(input_text)

        inputs = tokenizer(
            images=None,
            text=input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")

        output = model.generate(**inputs, max_new_tokens=256, use_cache=True)

        response = tokenizer.batch_decode(output, skip_special_tokens=True)
        response = response[0]

        generated = response.split("assistant")[-1]

        break

    # m = re.search(r"\d+", generated)
    # if m:
    #     pred = m.group()  # 抽出した数字文字列
    # else:
    #     pred = None  # 見つからなかった場合

    # record = {
    #     "story_id": data["story_id"],
    #     "question": data["question"],
    #     "answer": data["answer"],
    #     "option": data["option"],
    #     "answer_idx": data["answer_idx"],
    #     "drop_pos": data["drop_pos"],
    #     "generated": generated,
    #     "pred": pred,
    # }
    # f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    # f_out.close()
