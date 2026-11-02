import os
import base64
import requests
import json
from io import BytesIO
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# Get OpenAI API Key from environment variable
api_key = os.environ["OPENAI_API_KEY"]
headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

# Structured Output Schema
response_schema = ResponseSchema(
    name="mapped_ids",
    description="List of numeric IDs corresponding to the identified objects in the image.",
)

parser = StructuredOutputParser.from_response_schemas([response_schema])

metaprompt = """
- For any marks mentioned in your answer, please highlight them with [].
- Ensure the output is in structured JSON format, returning only the `mapped_ids` list.
"""


# Function to encode the image
def encode_image_from_pil(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def prepare_inputs(message, image):
    base64_image = encode_image_from_pil(image)

    system_instruction = parser.get_format_instructions() + "\n\n" + metaprompt

    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": system_instruction},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": message},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            },
        ],
        "max_tokens": 800,
    }
    return payload


def request_gpt4v(message, image):

    payload = prepare_inputs(message, image)
    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )

    # Extract response content
    res = response.json()["choices"][0]["message"]["content"]
    print("res", res)
    # Parse response using structured output parser
    parsed_output = parser.parse(res)
    print(parsed_output)

    # Convert to a list of integers
    mapped_ids = parsed_output.get("mapped_ids", [])
    if isinstance(mapped_ids, list):
        try:
            return [int(i) for i in mapped_ids]  # Convert to integers
        except ValueError:
            raise ValueError("Unexpected non-numeric values in output.")
    else:
        raise ValueError("Output is not a valid list.")


# Pass an image (PIL format) when calling `request_gpt4v(message, image)`


def get_prompt(obj_list, depth_caption):
    prompt = (
        "Each object in the image has a bright numeric ID at its center.\n\n"
        "Identify the numeric ID for each of the following objects and return them in a Python list in the same order:\n\n"
    )

    for i in obj_list:
        prompt += f"- {i}\n"

    prompt += (
        f"\nImage Caption about depth: \n{depth_caption}\n\n"  # Added depth caption to prompt
        f"\nThere are exactly {len(obj_list)} objects to identify, so your answer must contain {len(obj_list)} numeric IDs.\n"
        "If the object is present in the image, return the ID you see, even if uncertain.\n"
        "Do not guess or assume an ID—only use IDs visible in the image.\n"
        "try to answer confidently with what you have"
        "If the object is completely missing from the image, return `-1`.\n"
        "Return only a Python list of IDs, nothing else. Example: `[1, 2, -1, 4]`"
    )

    return prompt


def get_id_res_list_(
    ds, index, obj_text="obj_text", image="image", depth_caption="depth_caption"
):

    img_path = ds["train"][index][image]
    obj_list = ds["train"][index][obj_text]
    depth_caption = ds["train"][index][depth_caption]
    # print(obj_list)
    message = get_prompt(obj_list, depth_caption)
    print(message)

    return request_gpt4v(message, img_path)
