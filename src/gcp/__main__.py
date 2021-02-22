import os
from pathlib import Path
import cv2

from src.gcp.helpers import upload_blob, get_parsed_document, print_texts, draw_blocks

creds = Path("../../.creds/My Project 71810-5db423e4bba0.json").absolute().__str__()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds


def main(project_id: str):
    file_path = Path("..", "..", "data", "invoice2.tif")
    upload_blob("tests_demos", file_path.__str__(), "invoice2.tif")
    document = get_parsed_document(project_id, input_uri="gs://tests_demos/invoice2.tif")

    img = cv2.imread(file_path.__str__())
    img = draw_blocks(img, document)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    #print_texts(document)


if __name__ == "__main__":
    main(project_id="white-flame-244921")