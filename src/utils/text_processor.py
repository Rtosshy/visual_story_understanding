import re

from num2words import num2words

from src.utils.image_processor import ImageProcessor as ip


class TextProcessor:
    @staticmethod
    def convert_numbers_to_words(text: str) -> str:
        return re.sub(r"\d+", lambda m: num2words(int(m.group()), lang="en"), text)

    @staticmethod
    def convert_to_openai_template(sample):
        instruction = "You are given a sequence of images that tell a story, but one image is missing. Choose only the number of the most appropriate option that describes what should happen in the missing image location. Respond with just the index (e.g., 0, 1, 2, ...)."

        content = [
            {"type": "text", "text": instruction},
            {"type": "text", "text": "\n\nSequence of images:\n"},
        ]

        image_ids = list(sample["question"].keys())
        original_length = len(image_ids)

        is_missing_inserted = False
        for i in range(original_length + 1):
            if i == sample["drop_pos"] and not is_missing_inserted:
                content.append({"type": "text", "text": "[Missing Image Position]"})
                is_missing_inserted = True
            else:
                idx = i - 1 if is_missing_inserted else i
                base64_string = ip.encode_image_to_base64(image_id=image_ids[idx])
                url = ip.encode_base64_to_url(base64_string=base64_string)
                content.append(
                    {"type": "image_url", "image_url": {"url": url, "detail": "low"}}
                )

        options = list(sample["option"].values())
        content.append({"type": "text", "text": "\n\nOptions:"})
        for i, option in enumerate(options):
            content.append({"type": "text", "text": f"\n{i}. {option}"})

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer the following question.",
            },
            {"role": "user", "content": content},
        ]

        return messages

    @staticmethod
    def convert_to_qwen_template(sample):
        instruction = "You are given a sequence of images that tell a story, but one image is missing. Choose only the number of the most appropriate option that describes what should happen in the missing image location. Respond with just the index (e.g., 0, 1, 2, ...)."

        content = [
            {"type": "text", "text": instruction},
            {"type": "text", "text": "\n\nSequence of images:\n"},
        ]

        image_ids = list(sample["question"].keys())
        original_length = len(image_ids)

        is_missing_inserted = False
        images = []
        for i in range(original_length + 1):
            if i == sample["drop_pos"] and not is_missing_inserted:
                content.append({"type": "text", "text": "[Missing Image Position]"})
                is_missing_inserted = True
            else:
                idx = i - 1 if is_missing_inserted else i
                image = ip.load_image(image_id=image_ids[idx])
                images.append(image)
                content.append({"type": "image"})

        options = list(sample["option"].values())
        content.append({"type": "text", "text": "\n\nOptions:"})
        for i, option in enumerate(options):
            content.append({"type": "text", "text": f"\n{i}. {option}"})

        messages = [
            {"role": "user", "content": content},
        ]

        return messages, images
