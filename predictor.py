from typing import List, Optional
import os

from dotenv import load_dotenv
import time
import subprocess
import random
import logging
from datetime import datetime

# load the environment variables
load_dotenv()

from cog import Path, Input
import torch
from PIL import Image
import PIL
from omegaconf import OmegaConf

from src.flux.xflux_pipeline_dreamfit import XFluxPipeline

MODEL_CACHE = "pretrained_models/FLUX.1-dev"
MODEL_URL = (
    "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar"
)

TRYON_LORA_CACHE = "pretrained_models/flux_tryon.bin"
TRYON_LORA_URL = "https://huggingface.co/bytedance-research/Dreamfit/resolve/main/flux_tryon.bin"

os.environ["FLUX_DEV"] = "pretrained_models/FLUX.1-dev/flux1-dev.safetensors"
os.environ["AE"] = "pretrained_models/FLUX.1-dev/ae.safetensors"


def concatenate_images(cloth, pose):
    width1, height1 = pose.size
    width2, height2 = cloth.size

    new_width = width1 + width2
    new_height = max(height1, height2)
    concat_image = Image.new('RGB', (new_width, new_height))

    concat_image.paste(pose, (0, 0))
    concat_image.paste(cloth, (width1, 0))

    return concat_image


def download_weights(url, dest, file=False):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    if not file:
        subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    else:
        subprocess.check_call(["pget", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class CogFluxDreamFitPredictor:
    def setup(self) -> None:
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)
        if not os.path.exists(TRYON_LORA_CACHE):
            download_weights(TRYON_LORA_URL, TRYON_LORA_CACHE, file=True)

        config_path = "configs/inference/inference_dreamfit_flux_tryon.yaml"
        output_path = "outputs"
        os.makedirs(output_path, exist_ok=True)

        self.config = OmegaConf.load(config_path)

        date_str = datetime.now().strftime("%Y%m%d")
        time_str = datetime.now().strftime("%H%M%S")
        LOG_FORMAT = (
            "%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s"
        )
        logging.basicConfig(
            filename=f"run_{date_str}_{time_str}.log",
            level=logging.INFO,
            datefmt="%a, %d %b %Y %H:%M:%S",
            format=LOG_FORMAT,
            filemode="w",
        )

        self.xflux_pipeline = XFluxPipeline(
            self.config.model_type,
            self.config.device,
            self.config.offload,
            image_encoder_path=self.config.image_encoder_path,
            lora_path=self.config.lora_local_path,
            model_path=self.config.model_path,
        )

        if self.config.use_lora:
            print(
                ">>> load lora:",
                self.config.lora_local_path,
                self.config.lora_repo_id,
                self.config.lora_name,
            )
            self.xflux_pipeline.set_lora(
                self.config.lora_local_path,
                self.config.lora_repo_id,
                self.config.lora_name,
                self.config.lora_weight,
                self.config.network_alpha,
                self.config.double_blocks,
                self.config.single_blocks,
            )


    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            "Prompt for the image generation", default="A woman wearing a white Bape T-shirt with a colorful ape graphic and bold text and a blue jeans."
        ),
        cloth_image: Path = Input(description="Input image of garment.", default=None),
        keep_image: Path = Input(
            description="Input keep image for editing. Area with black color and with open pose skeleton. "
            "No editing will take place for black color area",
            default=None,
        ),
        seed: Optional[int] = Input(
            description="Random seed. Set for reproducible generation", default=16414308815
        ),
        num_inference_steps: int = Input(
            description="Number of steps for the prediction",
            default=28,
            ge=1,
            le=50,
        ),
        guidance_scale: float = Input(
            description="Guidance scale value", ge=0, le=10, default=4
        )
    ) -> List[Path]:
        if not seed:
            seed = random.randint(0, 100000)
            print("No seed provided. Generating a random seed. Seed : {seed}")
        """Run a single prediction on the model"""

        height = self.config.inference_params.height
        width = self.config.inference_params.width
        ref_height = self.config.inference_params.ref_height
        ref_width = self.config.inference_params.ref_width
        init_seed = seed
        ref_size = (ref_width, ref_height)

        save_dir = "outputs"
        os.makedirs(save_dir, exist_ok=True)
        cloth_path = cloth_image
        keep_image_path = keep_image

        ref_text = "Two reference image. [IMAGE1] keep image and pose. [IMAGE2] cloth."

        cloth = PIL.Image.open(cloth_path).convert("RGB").resize(ref_size)
        keep_image = PIL.Image.open(keep_image_path).convert("RGB").resize(ref_size)
        ref_image = concatenate_images(cloth, keep_image)
        output_files = []
        for ind in range(self.config.num_images_per_prompt):
            result = self.xflux_pipeline(
                prompt=prompt,
                controlnet_image=None,
                width=width,
                height=height,
                guidance=guidance_scale,
                num_steps=num_inference_steps,
                true_gs=self.config.true_gs,
                control_weight=self.config.control_weight,
                neg_prompt=self.config.neg_prompt,
                timestep_to_start_cfg=self.config.timestep_to_start_cfg,
                ref_img=ref_image,
                ref_prompt=ref_text,
                neg_image_prompt=None,
                ip_scale=self.config.ip_scale,
                neg_ip_scale=self.config.neg_ip_scale,
            )

            filename = cloth_path.split("/")[-1]
            file_path = f"{save_dir}/{ind}_{filename}"

            result.save(file_path)
            print("save to ", file_path)
            output_files.append(Path(file_path))
            init_seed = init_seed + 1

        return output_files
