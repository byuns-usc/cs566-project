# SegOne: Reinventing Semantic Segmentation with Channel-Wise 1D Convolutions

Many modern computer vision architectures leverage diffusion for its ease of feature extraction and generation. However, these models are often heavy due to their gradual noise generation and removal processes, negatively impacting their usability in power-limited edge devices. In this paper, we show diffusion models can be reduced to 1D convolution architecture without significantly impacting the accuracy while increasing deployability. We propose to make a move toward a lighter diffusion pipeline that only requires 1D calculations by generating semantic segmentation masks with only channel-wise 1D convolutions. In many state-of-the-art super-resolution methods, PixelShuffle operations have been proven to increase accuracy while decreasing computational overhead. Following suit, SegOne incorporates PixelShuffle operations into diffusion to effectively capture spatial relations without 2D kernels. Although we mainly focus on MRI image segmentation for the sake of specialization, the proposed method is also rigorously tested against 2D convolution UNet baselines with varying depths, evaluated on multiple datasets such as COCO, NYUv2, and Diode. 

### About
This repository serves as the official codebase for the paper __`SegOne: Reinventing Semantic Segmentation with Channel-Wise 1D Convolutions`__.

## Authors
1. [Sanghyun Byun](https://shbyun080.github.io/) | `MS in Computer Science @ USC`, `AI Partner @ LG Electronics`
3. [Kayvan Shah](https://github.com/KayvanShah1) | `MS in Applied Data Science @ USC`
2. [Ayushi Gang](https://github.com/) | `MS in Computer Science @ USC`
4. [Christopher Apton](https://github.com/chrisapton) | `MS in Applied Data Science @ USC`

## Citation
If you find our work useful in your research, please consider citing our paper:
```
@misc{segone-2024,
  title={SegOne: Reinventing Semantic Segmentation with Channel-Wise 1D Convolutions},
  author={Sanghyun Byun and Kayvan Shah and Ayushi Gang and Christopher Apton},
  archivePrefix={arXiv:},
  eprint={},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/}, 
  month={November},
  year={2024},
}
```

#### LICENSE
This project is licensed under the `MIT` License. See the [LICENSE](LICENSE) file for details.

#### Disclaimer
<sub>
The content and code provided in this repository are for educational and demonstrative purposes only. The project may contain experimental features, and the code might not be optimized for production environments. The authors and contributors are not liable for any misuse, damages, or risks associated with the use of this code. Users are advised to review, test, and modify the code to suit their specific use cases and requirements. By using any part of this project, you agree to these terms.
</sub>
