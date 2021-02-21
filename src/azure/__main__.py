import os

from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import FormRecognizerClient
from dotenv.main import load_dotenv
from cv2 import imread, rectangle
import cv2


def draw_blocks(img, res, confidence: float = 0.9):

    for block in res[0].lines:
        if any(word.confidence > confidence for word in block.words):
            x_coords, y_coords = [int(b.x) for b in block.bounding_box], [int(b.y) for b in block.bounding_box]
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            img = rectangle(img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
    return img


def main():
    load_dotenv("../../.env")
    credential = AzureKeyCredential(os.environ.get("COGNITIVE_SERVICE_KEY"))

    form_recognizer_client = FormRecognizerClient(
        endpoint="https://ocrdemo1.cognitiveservices.azure.com/",
        credential=credential
    )
    with open("../../data/invoice2.png", "rb") as f:
        invoice = f.read()
    poller = form_recognizer_client.begin_recognize_content(invoice)
    page = poller.result()

    img = imread("../../data/invoice2.png")
    img = draw_blocks(img, page)
    cv2.imshow("img", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()