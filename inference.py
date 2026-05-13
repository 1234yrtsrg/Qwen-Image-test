import argparse
import math
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImagePipeline


MODEL_ID = "Qwen/Qwen-Image-2512"
LIGHTNING_LORA_ID = "lightx2v/Qwen-Image-2512-Lightning"
LIGHTNING_LORA_WEIGHT = "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors"


def parse_args():
    parser = argparse.ArgumentParser(description="Run Qwen-Image text-to-image inference.")
    parser.add_argument("--gpu", type=int, required=True, help="GPU id to use, for example: 0 or 7.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt used to generate the image.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the generated image.")
    return parser.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please run this script on a server with an NVIDIA GPU.")

    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise ValueError(
            f"Invalid GPU id {args.gpu}. This server has {torch.cuda.device_count()} CUDA device(s)."
        )

    torch.cuda.set_device(args.gpu)
    device = f"cuda:{args.gpu}"

    scheduler_config = {
        "base_image_seq_len": 256,
        "base_shift": math.log(3),
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

    pipe = QwenImagePipeline.from_pretrained(
        MODEL_ID,
        scheduler=scheduler,
        torch_dtype=torch.bfloat16,
    )
    pipe.load_lora_weights(LIGHTNING_LORA_ID, weight_name=LIGHTNING_LORA_WEIGHT)
    pipe.enable_sequential_cpu_offload(gpu_id=args.gpu)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if hasattr(pipe, "vae"):
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    image = pipe(
        prompt=args.prompt,
        negative_prompt="low resolution, low quality, blurry, distorted face, deformed body, bad anatomy, artificial looking skin",
        width=1024,
        height=1024,
        num_inference_steps=4,
        true_cfg_scale=1.0,
        generator=torch.Generator(device=device).manual_seed(42),
    ).images[0]

    image.save(args.output)
    print(f"Image saved to {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
