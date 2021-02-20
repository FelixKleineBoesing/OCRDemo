from pathlib import Path
import os
import cv2
import pytesseract
from src.tesseract.helpers import get_grayscale, thresholding, opening, canny, read_image
from src.tesseract.wrappers import read_text_from_image, draw_boxes_around_characters, draw_boxes_around_words


def main_get_text():
    invoice_path = Path("..", "..", "data", "invoice.jpg")
    img = read_image(invoice_path)
    img = get_grayscale(img)
    ocr_text = read_text_from_image(img)
    print(ocr_text)


def main_get_bounding_boxes():
    invoice_path = Path("..", "..", "data", "invoice2.png")
    img = read_image(invoice_path)
    img = draw_boxes_around_words(img, confidence=40)
    cv2.imshow("img", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    #main_get_text()
    main_get_bounding_boxes()