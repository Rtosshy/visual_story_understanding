import base64
import sys
import tarfile
from io import BytesIO
from pathlib import Path

from PIL import Image

from src.utils.paths import VIST_IMAGE_ROOT


class ImageProcessor:
    image_root_path = VIST_IMAGE_ROOT / "test"

    @staticmethod
    def extract_tar_gz(tar_gz_path: Path, extract_path: Path) -> None:
        try:
            with tarfile.open(tar_gz_path, "r:gz") as tar:
                tar.extractall(path=extract_path, filter="data")
                print(f"Extracted {tar_gz_path} to {extract_path}")
        except Exception as e:
            print(f"Error extracting {tar_gz_path}: {e}", file=sys.stderr)

    @staticmethod
    def convert_to_jpg(extract_path: Path) -> None:
        if not extract_path.exists():
            print(f"Extract path {extract_path} does not exist.", file=sys.stderr)
            return

        for png_file in extract_path.rglob("*png"):
            try:
                with Image.open(png_file) as image_file:
                    if image_file.mode != "RGB":
                        image = image_file.convert("RGB")
                    jpg_file = png_file.with_suffix(".jpg")

                    image.save(jpg_file, "JPEG")
                    png_file.unlink()
                    print(f"Converted {png_file} to {jpg_file}")
            except Exception as e:
                print(f"Error processing {png_file}: {e}", file=sys.stderr)

    @staticmethod
    def get_reduced_jpg_quality_if_large(
        image_path: Path, max_file_size_mb: float = 10.0, quality: int = 40
    ) -> bytes:
        if not image_path.exists():
            print(f"Image path {image_path} does not exist.", file=sys.stderr)
            return b""

        try:
            file_size_mb = image_path.stat().st_size / (1024 * 1024)
            with Image.open(image_path) as image_file:
                if file_size_mb > max_file_size_mb:
                    buffer = BytesIO()
                    image_file.save(buffer, format="JPEG", quality=quality)
                    buffer.seek(0)
                    reduced_bytes = buffer.getvalue()
                    print(
                        f"Created reduced quality bytes for {image_path} (size: {file_size_mb:.2f}MB > {max_file_size_mb}MB) at quality {quality}"
                    )
                    return reduced_bytes
                else:
                    buffer = BytesIO()
                    image_file.save(buffer, format="JPEG", quality=100)
                    buffer.seek(0)
                    raw_bytes = buffer.getvalue()
                    print(
                        f"Image {image_path} size {file_size_mb:.2f}MB is within limit, returning raw bytes"
                    )
                    return raw_bytes
        except Exception as e:
            print(f"Error processing {image_path}: {e}", file=sys.stderr)
            return b""

    @staticmethod
    def get_image_path(image_id: str) -> Path:
        return ImageProcessor.image_root_path / f"{image_id}.jpg"

    @staticmethod
    def load_image(image_id: str) -> Image.Image:
        image_path = ImageProcessor.get_image_path(image_id)
        return Image.open(image_path)

    @staticmethod
    def encode_image_to_base64(image_id: str) -> str:
        image_path = ImageProcessor.get_image_path(image_id)
        image_bytes = ImageProcessor.get_reduced_jpg_quality_if_large(image_path)
        base64_string = base64.b64encode(image_bytes).decode("utf-8")
        return base64_string

    @staticmethod
    def encode_base64_to_url(base64_string: str) -> str:
        url = f"data:image/jpeg;base64,{base64_string}"
        return url
