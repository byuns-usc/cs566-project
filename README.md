# SegOne: Reinventing Text-Based Image Semantic Segmentation with Only 1D Channel Convolutions

Many modern computer vision architectures leverage `diffusion` for its ease of feature extraction However, these models are often heavy due to their gradual noise generation and removal processes. We propose to lighten the model by generating an open-vocabulary segmentation mask with diffusion using only `1D channel convolutions`. The `pixel-unshuffle` operation is utilized to capture spatial relationships in the channel dimension so that 2D information is distributed to channels. The architecture comprises a `1D diffusion` and a `CLIP encoder`, with text embedding augmented to latent space and/or decoder skip-connection layers. To further enhance performance, output image segments are compared to text prompts with a teacher model (SAM2) to ensure consistency during training. For the sake of simplicity, we focus on `MRI image segmentation` task.

### About
This repository contains code base for project titled __`SegOne: Reinventing Text-Based Image Semantic Segmentation with Only 1D Channel Convolutions`__ developed during the `CSCI 566: Deep Learning & Applications` course, Fall 2024, at the `University of Southern California` (USC).

## Authors
1. [Sanghyun Byun](https://github.com/) | `MS in Computer Science` | `USC`
3. [Kayvan Shah](https://github.com/KayvanShah1) | `MS in Applied Data Science` | `USC`
2. [Ayushi Gang](https://github.com/) | `MS in Computer Science` | `USC`
4. [Chistopher Anton](https://github.com/) | `MS in Computer Science` | `USC`

#### LICENSE
This project is licensed under the `MIT` License. See the [LICENSE](LICENSE) file for details.

#### Disclaimer

<sub>
The content and code provided in this repository are for educational and demonstrative purposes only. The project may contain experimental features, and the code might not be optimized for production environments. The authors and contributors are not liable for any misuse, damages, or risks associated with the use of this code. Users are advised to review, test, and modify the code to suit their specific use cases and requirements. By using any part of this project, you agree to these terms.
</sub>