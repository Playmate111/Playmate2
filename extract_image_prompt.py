import os
import cv2
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from src.utils.util import get_file_list

make_abs_path = lambda fn: os.path.join(os.path.dirname(os.path.realpath(__file__)), fn)


@torch.no_grad()
def extract_image_prompt(image_path, prompt_path, idx=0):
    device = 'cuda'
    question = 'Suppose the character in the picture is speaking. Describe main subjects and it is actions, background and environment,lighting condition and atmosphere ,camera movement and image style of image, the image. Replace the word "image" with "picture" in the description'
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            f"content": [
                {"type": "image", "image": {"image_path": image_path}},
                {"type": "text", "text": question},
            ]
        },
    ]

    with torch.no_grad():
        inputs = vlm_processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        output_ids = vlm_model.generate(**inputs, max_new_tokens=1024)
        prompt = vlm_processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    prompt = prompt.replace('\n', ' ')

    with open(prompt_path, "w") as f:
        f.write(prompt)
    f.close()

    print(f"prompt extracted successfully: {prompt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--prompt_path", type=str, required=True)

    args = parser.parse_args()
    image_path = args.image_path
    prompt_path = args.prompt_path

    vlm_model_path = './pretrained_weights/VideoLLaMA3-7B'
    vlm_model = AutoModelForCausalLM.from_pretrained(
        vlm_model_path,
        trust_remote_code=True,
        device_map={"": 'cuda'},
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    vlm_model.eval()
    vlm_processor = AutoProcessor.from_pretrained(vlm_model_path, trust_remote_code=True)

    os.makedirs(os.path.dirname(prompt_path), exist_ok=True)
    extract_image_prompt(image_path, prompt_path)
