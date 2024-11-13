# OneNet: A Channel-Wise 1D Convolutional U-Net

This repository serves as the official codebase for the paper:
> **OneNet: A Channel-Wise 1D Convolutional U-Net**
>
> [Sanghyun Byun](https://shbyun080.github.io/), [Kayvan Shah](https://github.com/KayvanShah1), [Ayushi Gang](https://github.com/ayu-04), [Christopher Apton](https://github.com/chrisapton)
>
> [arXiv:](/)

### About
Many state-of-the-art computer vision architectures leverage U-Net for its adaptability and efficient feature extraction. However, the multi-resolution convolutional design often leads to significant computational demands, limiting deployment on edge devices. We present a streamlined alternative: a 1D convolutional encoder that retains U-Net’s accuracy while enhancing its suitability for edge applications. Our novel encoder architecture achieves semantic segmentation through channel-wise 1D convolutions combined with pixel-unshuffle operations. By incorporating PixelShuffle, known for improving accuracy in super-resolution tasks while reducing computational load, OneNet captures spatial relationships without requiring 2D convolutions, reducing parameters by up to 47%. Additionally, we explore a fully 1D encoder-decoder that achieves a 70% reduction in size, albeit with some accuracy loss. We benchmark our approach against U-Net variants across diverse mask-generation tasks, demonstrating that it preserves accuracy effectively. Although focused on image segmentation, this architecture is adaptable to other convolutional applications. 

[Secondary Repository](https://github.com/shbyun080/OneNet)

## 🚧 Roadmap
10/01/2024: Project Repo Initialized

10/22/2024: Initial Model and Training Code Uploaded

## ⚙️ Installation
Environment (model has not been tested on other environments)
- Linux
- Python 3.12
- CUDA 12.1

Please set the environment with
```bash
export VENV_DIR=<YOUR-VENV>
export NAME=SegOne

python -m venv $VENV_DIR/$NAME
source $VENV_DIR/$NAME/bin/activate
```

For general use
```bash
pip install -r requirements.txt
```

For development use, do an editable installation locally to avoid importing issues
```bash
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu121
```

## 🤖 Prediction
WIP

## 🦾 Training

### Config Options
Before training, the below config parameters must be changed in config accordingly:
```
model:
  name: ONENET | SEGONE | RESNET | UNET | MOBILENET
  channel_in: 1 (GREY) | 3 (RGB) | 4(RGBD)
  channel_out: <FOLLOW DATASET CLASS NUM>

data:
  name: COCO | VOC | PET | PET2 | BRAIN | HEART
  datapath: <PATH TO DATA FOLDER>
```

### Train
```
python train.py --cfg <PATH TO CONFIG FILE> --cuda <CUDA ID>
```

## 📦 Model Zoo
We currently release the following weights:

|Model         |Input Size|Dataset   |Download    |
|--------------|----------|----------|------------|
|OneNet-e4     |256x256   |MSD BRAIN |[Link]()    |
|OneNet-e4     |256x256   |MSD HEART |[Link]()    |
|OneNet-e4     |256x256   |PET_S     |[Link]()    |
|OneNet-e4     |256x256   |PET_F     |[Link]()    |
|OneNet-e4     |256x256   |PASCAL VOC|[Link]()    |
|--------------|----------|----------|------------|
|OneNet-ed4    |256x256   |MSD BRAIN |[Link]()    |
|OneNet-ed4    |256x256   |MSD HEART |[Link]()    |
|OneNet-ed4    |256x256   |PET_S     |[Link]()    |
|OneNet-ed4    |256x256   |PET_F     |[Link]()    |
|OneNet-ed4    |256x256   |PASCAL VOC|[Link]()    |

### Dataset
- MSD Brain
- MSD Heart
- Oxford PET
- VOC
- COCO

## 📜 Citation
If you find our work useful in your research, please consider citing our paper:
```bibtex
@misc{onenet-2024,
  title={OneNet: A Channel-Wise 1D Convolutional U-Net},
  author={Sanghyun Byun and Kayvan Shah and Ayushi Gang and Christopher Apton and Jacob Song and Woo Seong Chung},
  archivePrefix={arXiv:},
  eprint={},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/}, 
  month={November},
  year={2024},
}
```

### Authors
1. [Sanghyun Byun](https://shbyun080.github.io/) | `MS in Computer Science @ USC`, `AI Partner @ LG Electronics`
3. [Kayvan Shah](https://github.com/KayvanShah1) | `MS in Applied Data Science @ USC`
2. [Ayushi Gang](https://github.com/ayu-04) | `MS in Computer Science @ USC`
4. [Christopher Apton](https://github.com/chrisapton) | `MS in Applied Data Science @ USC`

### Acknowledgement
We thank `Professor Yan Liu` at the `University of Southern California` for guidance.

## 🪪 LICENSE
This project is licensed under the `MIT` License. See the [LICENSE](LICENSE) file for details.

#### Disclaimer
<sub>
The content and code provided in this repository are for demonstrative purposes only. The project may contain experimental features, and the code might not be optimized for production environments. The authors and contributors are not liable for any misuse, damages, or risks associated with the use of this code. Users are advised to review, test, and modify the code to suit their specific use cases and requirements. By using any part of this project, you agree to these terms.
</sub>
