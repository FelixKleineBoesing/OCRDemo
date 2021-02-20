# Displays information about a block returned by text detection and text analysis
import cv2

from src.aws.helpers import process_text_analysis
from dotenv.main import load_dotenv


load_dotenv("../../.env")


def main():
    bucket = 'felix-ml-sagemaker'
    document = 'invoice2.png'
    img, block_count = process_text_analysis(bucket, document)
    print("Blocks detected: " + str(block_count))
    cv2.imshow("img", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()