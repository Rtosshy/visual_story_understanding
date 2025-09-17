from pathlib import Path
from pprint import pprint

from ..dataset import get_vist_dataset
from ..utils import convert_to_conversation

# import torch
# from unsloth import FastVisionModel


# model, tokenizer = FastVisionModel.from_pretrained(
#     "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
#     load_in_4bit=True,  # Use 4bit to reduce memory use. False for 16bit LoRA.
#     use_gradient_checkpointing="unsloth",  # True or "unsloth" for long context
# )

# model = FastVisionModel.get_peft_model(
#     model,
#     finetune_vision_layers=True,  # False if not finetuning vision layers
#     finetune_language_layers=True,  # False if not finetuning language layers
#     finetune_attention_modules=True,  # False if not finetuning attention layers
#     finetune_mlp_modules=True,  # False if not finetuning MLP layers
#     r=16,  # The larger, the higher the accuracy, but might overfit
#     lora_alpha=16,  # Recommended alpha == r at least
#     lora_dropout=0,
#     bias="none",
#     random_state=3407,
#     use_rslora=False,  # We support rank stabilized LoRA
#     loftq_config=None,  # And LoftQ
#     # target_modules = "all-linear", # Optional now! Can specify a list if needed
# )

if __name__ == "__main__":
    dataset_path = Path(".") / "notebook" / "eda" / "vist_final.json"
    dataset = get_vist_dataset(dataset_path)

    for data in dataset:
        msg = convert_to_conversation(data)
        pprint(msg)
        break
