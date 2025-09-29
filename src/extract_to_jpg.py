import sys

from src.utils.image_processor import ImageProcessor as ip
from src.utils.paths import VIST_IMAGE_ROOT


def main():
    if len(sys.argv) != 2:
        print(
            "Usage: uv run -m src.utils.extract_to_jpg <tar_gz_file_name>",
            file=sys.stderr,
        )
        sys.exit(1)

    tar_gz_file_name = sys.argv[1]

    tar_gz_path = VIST_IMAGE_ROOT / tar_gz_file_name
    extract_path = VIST_IMAGE_ROOT

    ip.extract_tar_gz(tar_gz_path=tar_gz_path, extract_path=extract_path)

    ip.convert_to_jpg(extract_path=extract_path)


if __name__ == "__main__":
    main()
