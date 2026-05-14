import argparse
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from diffusers import QwenImageEditPlusPipeline
from PIL import Image, ImageOps


def parse_args():
    parser = argparse.ArgumentParser(description="Run Qwen-Image image editing inference.")
    parser.add_argument("--gpu", type=int, required=True, help="GPU id to use, for example: 0 or 7.")
    parser.add_argument("--image", type=str, required=True, help="Input image path.")
    parser.add_argument("--prompt", type=str, required=True, help="Edit instruction.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the edited image.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen-Image-Edit-2511", help="Model id or local model path.")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--true-cfg-scale", type=float, default=4.0, help="True CFG scale.")
    parser.add_argument("--guidance-scale", type=float, default=1.0, help="Guidance scale.")
    parser.add_argument(
        "--offload",
        choices=["model", "sequential", "none"],
        default="model",
        help="Memory strategy. Use 'none' for speed on large GPUs, 'sequential' if 'model' OOMs.",
    )
    return parser.parse_args()


def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input image does not exist: {path}")

    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


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

    input_image = load_image(args.image)

    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
    )
    pipeline.set_progress_bar_config(disable=None)

    if args.offload == "none":
        pipeline.to(device)
    elif args.offload == "model":
        pipeline.enable_model_cpu_offload(gpu_id=args.gpu)
    else:
        pipeline.enable_sequential_cpu_offload(gpu_id=args.gpu)

    if hasattr(pipeline, "vae"):
        pipeline.vae.enable_slicing()
        pipeline.vae.enable_tiling()

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    inputs = {
        "image": [input_image],
        "prompt": args.prompt,
        "negative_prompt": " ",
        "num_inference_steps": args.steps,
        "true_cfg_scale": args.true_cfg_scale,
        "guidance_scale": args.guidance_scale,
        "num_images_per_prompt": 1,
        "generator": torch.Generator(device=device).manual_seed(args.seed),
    }

    with torch.inference_mode():
        output = pipeline(**inputs)

    output_image = output.images[0]
    output_image.save(args.output)
    print(f"Image saved to {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
