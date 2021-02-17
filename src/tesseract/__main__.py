from pathlib import Path
import os
import cv2
import pytesseract
from src.tesseract.helpers import get_grayscale, thresholding, opening, canny


def main():
    files = os.listdir(Path("..", "..", "data", "cv-bilder"))
    for file in files:
        cv_page_jpg = cv2.imread(Path("..", "..", "data", "cv-bilder", file).__str__())
        img = get_grayscale(cv_page_jpg)
        img = thresholding(img)
        img = opening(img)
        img = canny(img)

        custom_config = r"--oem 3 --psm 6"
        pytesseract.image_to_string(img, config=custom_config)


if __name__ == "__main__":
    main()