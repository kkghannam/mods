#!/usr/bin/env python3
"""
Image Generator CLI

Uses Claude (Opus 4.6) to enhance prompts and Replicate (FLUX) to generate images.

Usage:
    python generate.py "a cat in space"
    python generate.py "a cat in space" --no-enhance
    python generate.py "a cat in space" --output my_image.png
    python generate.py "a cat in space" --width 1024 --height 768
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path

import anthropic
import replicate


def enhance_prompt(client: anthropic.Anthropic, prompt: str) -> str:
    """Use Claude to enhance the image generation prompt."""
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=512,
        system=(
            "You are an expert at writing prompts for AI image generators. "
            "Given a simple description, enhance it into a detailed, vivid image generation prompt. "
            "Focus on visual details: lighting, style, composition, mood, colors, textures. "
            "Return ONLY the enhanced prompt, nothing else — no explanation, no preamble."
        ),
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def generate_image(
    prompt: str,
    output_path: Path,
    width: int = 1024,
    height: int = 1024,
) -> None:
    """Generate an image using FLUX via Replicate and save it."""
    print(f"Generating image ({width}x{height})...")

    output = replicate.run(
        "black-forest-labs/flux-schnell",
        input={
            "prompt": prompt,
            "num_outputs": 1,
            "aspect_ratio": f"{width}:{height}" if width != height else "1:1",
            "output_format": "png",
            "output_quality": 90,
        },
    )

    # output is a list of URLs or file-like objects
    image_url = output[0] if isinstance(output[0], str) else output[0].url

    print(f"Downloading image to {output_path}...")
    urllib.request.urlretrieve(image_url, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate images from text prompts using Claude + FLUX"
    )
    parser.add_argument("prompt", help="Text description of the image to generate")
    parser.add_argument(
        "--no-enhance",
        action="store_true",
        help="Skip Claude prompt enhancement, use prompt as-is",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output file path (default: generated_image.png)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width in pixels (default: 1024)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height in pixels (default: 1024)",
    )
    args = parser.parse_args()

    # Validate API keys
    if not os.environ.get("ANTHROPIC_API_KEY") and not args.no_enhance:
        print("Error: ANTHROPIC_API_KEY not set. Use --no-enhance to skip prompt enhancement.")
        sys.exit(1)
    if not os.environ.get("REPLICATE_API_TOKEN"):
        print("Error: REPLICATE_API_TOKEN not set.")
        sys.exit(1)

    output_path = Path(args.output) if args.output else Path("generated_image.png")
    prompt = args.prompt

    # Enhance prompt with Claude
    if not args.no_enhance:
        client = anthropic.Anthropic()
        print(f"Original prompt: {prompt}")
        print("Enhancing prompt with Claude...")
        prompt = enhance_prompt(client, prompt)
        print(f"Enhanced prompt: {prompt}")
    else:
        print(f"Prompt: {prompt}")

    # Generate image
    generate_image(prompt, output_path, width=args.width, height=args.height)
    print(f"Image saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
