import cv2
import pytesseract
from pytesseract import Output


def read_text_from_image(img):
    custom_config = r"--oem 3 --psm 6"
    ocr_text = pytesseract.image_to_string(img, config=custom_config)
    return ocr_text


def draw_boxes_around_characters(img):
    h, w, c = img.shape
    boxes = pytesseract.image_to_boxes(img)
    for b in boxes.splitlines():
        b = b.split(" ")
        img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)
    return img


def draw_boxes_around_words(img, confidence: float = 60):
    data = pytesseract.image_to_data(img, output_type=Output.DICT)
    n_boxes = len(data['text'])
    for i in range(n_boxes):
        if int(data['conf'][i]) > confidence:
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img