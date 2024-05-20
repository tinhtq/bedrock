import boto3  # import aws sdk and supporting libraries
import json
import base64
import logging

from io import BytesIO


class ImageError(Exception):
    "Custom exception for errors returned by Amazon Titan Image Generator G1"

    def __init__(self, message):
        self.message = message


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

session = boto3.Session()

bedrock = session.client(service_name="bedrock-runtime")  # creates a Bedrock client

bedrock_model_id = "amazon.titan-image-generator-v1"  # use the Stable Diffusion model


def get_response_image_from_payload(
    response,
):  # returns the image bytes from the model response payload

    payload = json.loads(
        response.get("body").read()
    )  # load the response body into a json object

    base64_image = payload.get("images")[0]
    base64_bytes = base64_image.encode("ascii")
    image_bytes = base64.b64decode(base64_bytes)
    finish_reason = payload.get("error")

    if finish_reason is not None:
        raise ImageError(f"Image generation error. Error is {finish_reason}")

    logger.info(
        "Successfully generated image with Amazon Titan Image Generator G1 model"
    )

    return BytesIO(image_bytes)


def get_image_response(prompt_content):  # text-to-text client function

    request_body = json.dumps(
        {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {"text": prompt_content},
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "height": 1024,
                "width": 1024,
                "cfgScale": 8.0,
                "seed": 0,
            },
        }
    )  # number of diffusion steps to perform

    response = bedrock.invoke_model(
        body=request_body, modelId=bedrock_model_id
    )  # call the Bedrock endpoint

    output = get_response_image_from_payload(
        response
    )  # convert the response payload to a BytesIO object for the client to consume

    return output
