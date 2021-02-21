import os
from pathlib import Path

from google.cloud import documentai_v1beta2 as documentai

from src.gcp.helpers import upload_blob

creds = Path("../../.creds/My Project 71810-5db423e4bba0.json").absolute().__str__()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds


def main(
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
        gcs_source=gcs_source, mime_type="image/gifc"
    )

    # Location can be 'us' or 'eu'
    parent = "projects/{}/locations/eu".format(project_id)
    request = documentai.types.ProcessDocumentRequest(
        parent=parent, input_config=input_config
    )

    document = client.process_document(request=request)

    # All text extracted from the document
    print("Document Text: {}".format(document.text))

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


if __name__ == "__main__":
    upload_blob("tests_demos", "../../data/invoice2.png", "invoice2.png")
    main(input_uri="gs://tests_demos/invoice2.png")