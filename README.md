# Stable Diffusion - README

## Introduction
Stable Diffusion is a powerful, open-source text-to-image generation model. While there exist multiple open-source implementations that allow you to easily create images from textual prompts, KerasCV's implementation offers a few distinct advantages. These include XLA compilation and mixed precision support, which together achieve state-of-the-art generation speed.

This guide will walk you through KerasCV's Stable Diffusion implementation, showcasing how to leverage these powerful performance boosts and exploring the benefits they offer.

**Note:** To run this guide on the torch backend, please set `jit_compile=False` everywhere. XLA compilation for Stable Diffusion does not currently work with torch.

## Dependencies and Setup
To get started, let's install a few dependencies and sort out some imports:
```python
!pip install -q --upgrade keras-cv
!pip install -q --upgrade keras
import time
import keras_cv
import keras
import matplotlib.pyplot as plt

## Getting Started
First, let's explore how to use KerasCV's Stable Diffusion implementation and generate images from textual prompts.\
model = keras_cv.models.StableDiffusion(
    img_width=512, img_height=512, jit_compile=False
)

## How It Works
Stable Diffusion operates on the concept of latent diffusion, where a model denoises noise patches to generate images. Here's a brief overview of how it works:

1. **Text Encoding**: The prompt is encoded into a latent vector.\
2. **Diffusion Model**: A diffusion model denoises a 64x64 latent image patch over multiple steps.\
3. **Decoder**: The final 64x64 latent patch is decoded into a higher-resolution 512x512 image.\

## Advantages of KerasCV
KerasCV's Stable Diffusion model offers several advantages over other implementations, including:
- Graph mode execution
- XLA compilation through `jit_compile=True`
- Support for mixed precision computation

By combining these features, KerasCV's Stable Diffusion model achieves significantly faster generation speeds. See the benchmarks below for comparison.

## Benchmarks
We conducted benchmarks comparing KerasCV's Stable Diffusion with other implementations. Here are the results on a Tesla T4 GPU:\

| GPU        | Model                    | Runtime (s) |\
|------------|--------------------------|-------------|\
| Tesla T4   | KerasCV (Warm Start)     | 28.97       |\
| Tesla T4   | Other (Warm Start)       | 41.33       |\
| Tesla V100 | KerasCV (Warm Start)     | 12.45       |\
| Tesla V100 | Other (Warm Start)       | 12.72       |\

As seen from the results, KerasCV's implementation outperforms others in terms of runtime.\

## Conclusion
In conclusion, KerasCV's Stable Diffusion model offers state-of-the-art text-to-image generation with superior performance. By leveraging XLA compilation and mixed precision support, it achieves remarkable speed improvements over other implementations. Get started with KerasCV today to experience faster and more efficient image generation!\

## License
By using this model checkpoint, you acknowledge that its usage is subject to the terms of the CreativeML Open RAIL-M license at [CreativeML License](https://raw.githubusercontent.com/CompVis/stable-diffusion/main/LICENSE).\
}
