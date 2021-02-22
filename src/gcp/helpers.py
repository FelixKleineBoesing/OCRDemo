from google.cloud import storage
from google.cloud import documentai_v1beta2 as documentai
from cv2 import rectangle

def upload_blob(bucket_name, source_file_name, destination_blob_name, overwrite: bool = False):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    if not blob.exists() or overwrite:
        blob.upload_from_filename(source_file_name)

        print(
            "File {} uploaded to {}.".format(
                source_file_name, destination_blob_name
            )
        )


def get_parsed_document(
    project_id="white-flame-244921",
    input_uri="gs://cloud-samples-data/documentai/invoice.pdf",
):
    """Process a single document with the Document AI API, including
    text extraction and entity extraction."""

    client = documentai.DocumentUnderstandingServiceClient()

    gcs_source = documentai.types.GcsSource(uri=input_uri)

    # mime_type can be application/pdf, image/tiff,
    # and image/gif, or application/json
    input_config = documentai.types.InputConfig(
        gcs_source=gcs_source,
        mime_type="image/tiff",

    )

    # Location can be 'us' or 'eu'
    parent = "projects/{}/locations/eu".format(project_id)
    request = documentai.types.ProcessDocumentRequest(
        parent=parent, input_config=input_config
    )

    document = client.process_document(request=request)
    return document


def print_texts(document):
    def _get_text(el):
        """Convert text offset indexes into text snippets."""
        response = ""
        # If a text segment spans several lines, it will
        # be stored in different text segments.
        for segment in el.text_anchor.text_segments:
            start_index = segment.start_index
            end_index = segment.end_index
            response += document.text[start_index:end_index]
        return response

    for entity in document.entities:
        print("Entity type: {}".format(entity.type_))
        print("Text: {}".format(_get_text(entity)))
        print("Mention text: {}\n".format(entity.mention_text))


def draw_blocks(img, res, confidence: float = 0.9):

    for block in res.pages[0].lines:
        if block.layout.confidence > confidence:
            x_coords, y_coords = \
                [int(b.x) for b in block.layout.bounding_poly.vertices], \
                [int(b.y) for b in block.layout.bounding_poly.vertices]
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            img = rectangle(img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
    return img

