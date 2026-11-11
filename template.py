import logging
import os
from pathlib import Path


list_of_files = [
    "src/__init__.py",
    "src/helper.py",
    "src/prompt.py",
    "templates/index.html",
    "static/style.css",
    "app.py",
    "setup.py",
    "requirements.txt",
    ".env",
]


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s]: %(message)s",
    )

    for filepath in list_of_files:
        filepath = Path(filepath)
        filedir, filename = os.path.split(filepath)

        if filedir != "":
            os.makedirs(filedir, exist_ok=True)
            logging.info("Directory ready: %s", filedir)

        try:
            with open(filepath, "x", encoding="utf-8"):
                pass
            logging.info("Created file: %s", filepath)
        except FileExistsError:
            logging.info("File already exists: %s", filepath)


if __name__ == "__main__":
    main()