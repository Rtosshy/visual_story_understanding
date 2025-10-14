import re

from num2words import num2words

from src.utils.image_processor import ImageProcessor as ip


class TextProcessor:
    @staticmethod
    def convert_numbers_to_words(text: str) -> str:
        return re.sub(r"\d+", lambda m: num2words(int(m.group()), lang="en"), text)

    @staticmethod
    def convert_to_build_wrong_option_template(sample, target_pos, num_wrong_options):
        """
        sample:
        {
            "story_id": story_id, int
            "image_ids": [id0, id1, id2, ...], List[str]
            "texts": [text0, text1, text2, ...], List[str]
        }
        target_pos: int
        num_wrong_options: int
        return: massage
        """
        image_ids = sample["image_ids"]
        texts = sample["texts"]
        assert len(image_ids) == len(texts), "image_ids and texts length mismatch"
        assert 0 <= target_pos <= len(image_ids), "target_pos out of range"

        instruction = "You are given a coherent story represented by a sequence of images and their captions. "

        content = [
            {"type": "text", "text": instruction},
            {"type": "text", "text": "\n\nStory sequence (index, image, caption):\n"},
        ]

        for idx, (image_id, text) in enumerate(zip(image_ids, texts)):
            marker = " <== TAGET POSITION" if idx == target_pos else ""
            content.append({"type": "text", "text": f"({idx}){marker}"})
            base64_string = ip.encode_image_to_base64(image_id=image_id)
            url = ip.encode_base64_to_url(base64_string=base64_string)
            content.append(
                {"type": "image_url", "image_url": {"url": url, "detail": "low"}}
            )
            content.append({"type": "text", "text": text})

        content.append(
            {
                "type": "text",
                "text": "",
            }
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content},
        ]

        return messages

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

    @staticmethod
    def convert_to_shuffled_text_template(sample):
        instruction = "You are given five images representing a sequence of events. You are also given five sentences that describe the same sequence of events, but their order has been shuffled. Rearrange them into the correct sequence. Your answer must be provided in array format only. Example: [1, 3, 4, 0, 2]"

        content = [
            {"type": "text", "text": instruction},
            {"type": "text", "text": "\n\nSequence of images:\n"},
        ]

        image_ids = list(sample["image_ids"])

        for image_id in image_ids:
            base64_string = ip.encode_image_to_base64(image_id=image_id)
            url = ip.encode_base64_to_url(base64_string=base64_string)
            content.append(
                {"type": "image_url", "image_url": {"url": url, "detail": "low"}}
            )

        content.append({"type": "text", "text": "\n\nSentences:"})
        shuffled_texts = list(sample["shuffled_texts"])
        for i, text in enumerate(shuffled_texts):
            content.append({"type": "text", "text": f"\n{i}. {text}"})

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer the following question.",
            },
            {"role": "user", "content": content},
        ]

        return messages

    @staticmethod
    def convert_to_shuffled_image_template(sample):
        instruction = "You are given five sentences representing a sequence of events. You are also given five images that describe the same sequence of events, but their order has been shuffled. Rearrange them into the correct sequence. Your answer must be provided in array format only. Example: [1, 3, 4, 0, 2]"

        content = [
            {"type": "text", "text": instruction},
            {"type": "text", "text": "\n\nSequence of sentences:\n"},
        ]

        texts = list(sample["texts"])
        for text in texts:
            content.append({"type": "text", "text": f"\n{text}"})

        shuffled_image_ids = list(sample["shuffled_image_ids"])
        content.append({"type": "text", "text": "\n\nImages:"})
        for i, image_id in enumerate(shuffled_image_ids):
            base64_string = ip.encode_image_to_base64(image_id=image_id)
            url = ip.encode_base64_to_url(base64_string=base64_string)
            content.append({"type": "text", "text": f"\n{i}\n"})
            content.append(
                {"type": "image_url", "image_url": {"url": url, "detail": "low"}}
            )

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer the following question.",
            },
            {"role": "user", "content": content},
        ]

        return messages
