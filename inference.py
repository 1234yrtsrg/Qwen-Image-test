import argparse
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from diffusers import QwenImagePipeline


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

    pipe = QwenImagePipeline.from_pretrained(
        "Qwen/Qwen-Image-2512",
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload(gpu_id=args.gpu)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    image = pipe(
        prompt=args.prompt,
        negative_prompt="low resolution, low quality, blurry, distorted face, deformed body, bad anatomy, artificial looking skin",
        width=1328,
        height=1328,
        num_inference_steps=50,
        true_cfg_scale=4.0,
        generator=torch.Generator(device=device).manual_seed(42),
    ).images[0]

    image.save(args.output)
    print(f"Image saved to {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
