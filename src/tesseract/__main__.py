from pathlib import Path
import os
import cv2
import pytesseract
from src.tesseract.helpers import get_grayscale, thresholding, opening, canny, read_image
from src.tesseract.wrappers import read_text_from_image, draw_boxes_around_characters, draw_boxes_around_words


def main_get_text():
    cv_path = Path("..", "..", "data", "cv-bilder")
    files = os.listdir(cv_path)
    for file in files:
        img = read_image(Path(cv_path, file))
        img = get_grayscale(img)
        ocr_text = read_text_from_image(img)
        print(ocr_text)


def main_get_bounding_boxes():
    files = os.listdir(Path("..", "..", "data", "cv-bilder"))
    for file in files:
        img = cv2.imread(Path("..", "..", "data", "cv-bilder", file).__str__())
        img = draw_boxes_around_words(img)
        cv2.imshow("img", img)
        cv2.waitKey(0)


if __name__ == "__main__":
    #main_get_text()
    main_get_bounding_boxes()