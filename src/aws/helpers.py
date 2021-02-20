import cv2
import boto3
import numpy as np


def draw_bounding_box(img, box, width, height, box_color = None):
    if box_color is None:
        box_color = (0, 255, 0)
    x, y, w, h = get_coordinate_information(box, width, height)
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img


def show_selected_element(img, box, width, height, box_color=None):
    if box_color is None:
        box_color = (0, 255, 0)
    x, y, w, h = get_coordinate_information(box, width, height)
    img = cv2.rectangle(img, (x, y), (x + w, y + h), box_color, -1)
    return img


def get_coordinate_information(box, width, height):
    left = width * box['Left']
    top = height * box['Top']
    (x, y, w, h) = (int(left), int(top), int(width * box["Width"]), int(height * box["Height"]))
    return x, y, w, h


def display_block_information(block):
    print('Id: {}'.format(block['Id']))
    if 'Text' in block:
        print('    Detected: ' + block['Text'])
    print('    Type: ' + block['BlockType'])

    if 'Confidence' in block:
        print('    Confidence: ' + "{:.2f}".format(block['Confidence']) + "%")

    if block['BlockType'] == 'CELL':
        print("    Cell information")
        print("        Column:" + str(block['ColumnIndex']))
        print("        Row:" + str(block['RowIndex']))
        print("        Column Span:" + str(block['ColumnSpan']))
        print("        RowSpan:" + str(block['ColumnSpan']))

    if 'Relationships' in block:
        print('    Relationships: {}'.format(block['Relationships']))
    print('    Geometry: ')
    print('        Bounding Box: {}'.format(block['Geometry']['BoundingBox']))
    print('        Polygon: {}'.format(block['Geometry']['Polygon']))

    if block['BlockType'] == "KEY_VALUE_SET":
        print('    Entity Type: ' + block['EntityTypes'][0])

    if block['BlockType'] == 'SELECTION_ELEMENT':
        print('    Selection element detected: ', end='')

        if block['SelectionStatus'] == 'SELECTED':
            print('Selected')
        else:
            print('Not selected')

    if 'Page' in block:
        print('Page: ' + block['Page'])
    print()


def process_text_analysis(bucket, document):
    # Get the document from S3
    s3_connection = boto3.resource('s3')

    s3_object = s3_connection.Object(bucket, document)
    s3_response = s3_object.get()

    stream = s3_response['Body'].read()
    nparr = np.fromstring(stream, np.uint8)
    image_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Analyze the document
    client = boto3.client('textract', region_name="eu-west-1")

    response = client.analyze_document(Document={'Bytes': stream},
                                       FeatureTypes=["TABLES", "FORMS"])

    # Alternatively, process using S3 object
    # response = client.analyze_document(
    #    Document={'S3Object': {'Bucket': bucket, 'Name': document}},
    #    FeatureTypes=["TABLES", "FORMS"])

    # Get the text blocks
    blocks = response['Blocks']
    h, w, c = image_cv.shape

    # Create image showing bounding box/polygon the detected lines/text
    for block in blocks:

        display_block_information(block)

        if block['BlockType'] == "KEY_VALUE_SET":
            if block['EntityTypes'][0] == "KEY":
                image_cv = draw_bounding_box(image_cv, block['Geometry']['BoundingBox'], w, h, 'red')
            else:
                image_cv = draw_bounding_box(image_cv, block['Geometry']['BoundingBox'], w, h, 'green')

        if block['BlockType'] == 'TABLE':
            image_cv = draw_bounding_box(image_cv, block['Geometry']['BoundingBox'], w, h, 'blue')

        if block['BlockType'] == 'CELL':
            image_cv = draw_bounding_box(image_cv, block['Geometry']['BoundingBox'], w, h, 'yellow')
        if block['BlockType'] == 'SELECTION_ELEMENT':
            if block['SelectionStatus'] == 'SELECTED':
                image_cv = show_selected_element(image_cv, block['Geometry']['BoundingBox'], w, h, 'blue')

                # uncomment to draw polygon for all Blocks
            # points=[]
            # for polygon in block['Geometry']['Polygon']:
            #    points.append((w * polygon['X'], h * polygon['Y']))
            # image_cv.polygon((points), outline='blue')

    # Display the image
    return image_cv, len(blocks)