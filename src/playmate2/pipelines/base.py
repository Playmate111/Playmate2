import numpy as np
import torch
from PIL import Image


class BasePipeline(torch.nn.Module):
    def __init__(
        self,
        height_division_factor=64,
        width_division_factor=64,
    ):
        super().__init__()
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.cpu_offload = False
        self.model_names = []

    def check_resize_height_width(self, height, width):
        if height % self.height_division_factor != 0:
            height = (
                (height + self.height_division_factor - 1)
                // self.height_division_factor
                * self.height_division_factor
            )
            print(
                f"The height cannot be evenly divided by {self.height_division_factor}. We round it up to {height}."
            )
        if width % self.width_division_factor != 0:
            width = (
                (width + self.width_division_factor - 1)
                // self.width_division_factor
                * self.width_division_factor
            )
            print(
                f"The width cannot be evenly divided by {self.width_division_factor}. We round it up to {width}."
            )
        return height, width

    def preprocess_image(self, image):
        image = (
            torch.Tensor(np.array(image, dtype=np.float32) * (2 / 255) - 1)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        return image

    def preprocess_images(self, images):
        return [self.preprocess_image(image) for image in images]

    def vae_output_to_image(self, vae_output):
        image = vae_output[0].cpu().float().permute(1, 2, 0).numpy()
        image = Image.fromarray(((image / 2 + 0.5).clip(0, 1) * 255).astype("uint8"))
        return image

    def vae_output_to_video(self, vae_output):
        video = vae_output.cpu().permute(1, 2, 0).numpy()
        video = [
            Image.fromarray(((image / 2 + 0.5).clip(0, 1) * 255).astype("uint8"))
            for image in video
        ]
        return video

    def enable_cpu_offload(self):
        self.cpu_offload = True

    def generate_noise(self, shape, seed=None, device="cpu", dtype=torch.float16):
        generator = None if seed is None else torch.Generator(device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        return noise
