#!/usr/bin/env python3
"""
Image Generator CLI

Uses Claude (Opus 4.6) to enhance prompts and Replicate (FLUX) or Gemini to generate images.

Usage:
    python generate.py "a cat in space"
    python generate.py "a cat in space" --no-enhance
    python generate.py "a cat in space" --output my_image.png
    python generate.py "a cat in space" --width 1024 --height 768
    python generate.py "a cat in space" --gemini
    python generate.py "a cat in space" --gemini --styles "watercolor,oil-painting"
"""

import argparse
import base64
import os
import sys
import urllib.request
from pathlib import Path

import anthropic
import replicate
from google import genai
from google.genai import types


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


def generate_image_gemini(
    prompt: str,
    output_path: Path,
    styles: str = None,
) -> None:
    """Generate an image using Gemini (nano-banana) via the google-genai SDK."""
    api_key = os.environ.get("NANOBANANA_API_KEY")
    if not api_key:
        print("Error: NANOBANANA_API_KEY not set.")
        sys.exit(1)

    model = os.environ.get("NANOBANANA_MODEL", "gemini-3.1-flash-image-preview")
    print(f"Generating image with Gemini ({model})...")

    full_prompt = prompt
    if styles:
        full_prompt = f"{prompt} in the style of: {styles}"

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=full_prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )

    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            image_data = base64.b64decode(part.inline_data.data) if isinstance(part.inline_data.data, str) else bytes(part.inline_data.data)
            output_path.write_bytes(image_data)
            print(f"Image saved to: {output_path.resolve()}")
            return

    print("Error: No image data in Gemini response.")
    sys.exit(1)


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
    parser.add_argument(
        "--gemini",
        action="store_true",
        help="Use Gemini (nano-banana) for image generation instead of Replicate/FLUX",
    )
    parser.add_argument(
        "--styles",
        default=None,
        help='Comma-separated styles for Gemini generation (e.g. "watercolor,oil-painting")',
    )
    args = parser.parse_args()

    # Validate API keys
    if not os.environ.get("ANTHROPIC_API_KEY") and not args.no_enhance:
        print("Error: ANTHROPIC_API_KEY not set. Use --no-enhance to skip prompt enhancement.")
        sys.exit(1)
    if args.gemini:
        if not os.environ.get("NANOBANANA_API_KEY"):
            print("Error: NANOBANANA_API_KEY not set.")
            sys.exit(1)
    elif not os.environ.get("REPLICATE_API_TOKEN"):
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
    if args.gemini:
        generate_image_gemini(prompt, output_path, styles=args.styles)
    else:
        generate_image(prompt, output_path, width=args.width, height=args.height)
        print(f"Image saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
