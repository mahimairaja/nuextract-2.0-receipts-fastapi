import json
from typing import Optional

import modal
from PIL import Image

# Create a Modal app for the function
app = modal.App("nuextract-2.0-function")


CUDA_BASE = "nvidia/cuda:12.2.0-devel-ubuntu22.04"

# Build the Modal image with required dependencies
inference_image = (
    modal.Image.from_registry(CUDA_BASE, add_python="3.12")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .uv_pip_install(
        "torch",
    )
    .uv_pip_install(
        "pillow",
        "torchvision",
        "accelerate",
        "huggingface_hub[hf_transfer]",
        "transformers",
        "qwen-vl-utils[decord]",
    )
    .uv_pip_install(
        "flash-attn",
        extra_options="--no-build-isolation",
    )
)

MODEL_NAME = "numind/NuExtract-2.0-2B"
MODEL_REVISION = "c2ccd5d6b32c3fc70873153df6065b201cbd4bae"


def setup():
    import warnings

    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    with warnings.catch_warnings():  # filter noisy warnings from GOT modeling code
        warnings.simplefilter("ignore")
        model_name = "numind/NuExtract-2.0-2B"

        # Load the model with flash attention and device mapping
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )

        # Load the processor for the model
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",
            use_fast=True,
        )

    return model, processor


# Use a Modal volume for model cache
model_cache = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

MODEL_CACHE_PATH = "/root/models"
inference_image = inference_image.env(
    {"HF_HUB_CACHE": MODEL_CACHE_PATH, "HF_HUB_ENABLE_HF_TRANSFER": "1"}
)


@app.function(
    gpu="l40s",
    retries=3,
    volumes={MODEL_CACHE_PATH: model_cache},
    image=inference_image,
)
def parse_receipt(image: bytes) -> str:
    from tempfile import NamedTemporaryFile

    model, processor = setup()

    # Define the extraction template
    TEMPLATE = json.dumps(
        {
            "vendor": "verbatim-string",
            "purchase_date": "date-time",
            "total": "number",
            "currency": ["USD", "CAD"],
            "items": [
                {
                    "description": "string",
                    "quantity": "integer",
                    "unit_price": "number",
                    "line_total": "number",
                }
            ],
        },
        indent=4,
    )

    # Save image to a temp file for processing
    with NamedTemporaryFile(delete=False, mode="wb+") as temp_img_file:
        temp_img_file.write(image)
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": f"file://{temp_img_file.name}"}],
            }
        ]
        image = Image.open(temp_img_file.name).convert("RGB")

    # Prepare prompt using the processor's chat template
    prompt_text = processor.tokenizer.apply_chat_template(
        messages,
        template=TEMPLATE,  # template is specified here
        tokenize=False,
        add_generation_prompt=True,
    )

    # Prepare model inputs
    inputs = processor(
        text=[prompt_text],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # Run model inference
    out_ids = model.generate(
        **inputs,
        do_sample=False,  # greedy works well for extraction
        num_beams=1,
        max_new_tokens=512,
    )

    # Remove prompt tokens from output
    out_trimmed = [o[len(i) :] for i, o in zip(inputs.input_ids, out_ids)]
    extracted_json = processor.batch_decode(
        out_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    print("Result: ", extracted_json)

    return extracted_json


@app.local_entrypoint()
def main(receipt_filename: Optional[str] = None):
    import urllib.request
    from pathlib import Path

    if receipt_filename is None:
        receipt_filename = Path(__file__).parent / "receipt.png"
    else:
        receipt_filename = Path(receipt_filename)

    if receipt_filename.exists():
        image = receipt_filename.read_bytes()
        print(f"running OCR on {receipt_filename}")
    else:
        # Download a sample receipt if not found locally
        receipt_url = "https://modal-cdn.com/cdnbot/Brandys-walmart-receipt-8g68_a_hk_f9c25fce.webp"
        request = urllib.request.Request(receipt_url)
        with urllib.request.urlopen(request) as response:
            image = response.read()
        print(f"running OCR on sample from URL {receipt_url}")
    print(parse_receipt.remote(image))
