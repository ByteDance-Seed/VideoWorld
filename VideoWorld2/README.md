# VideoWorld 2: Learning Transferable Knowledge from Real-world Video
> #### Zhongwei Ren, Yunchao Wei<sup>&dagger;</sup>, Xiao Yu, Guixun Luo, Yao Zhao, Bingyi Kang, Jiashi Feng, and Xiaojie Jin<sup>&dagger;</sup><sup>&ddagger;</sup>
> <sup>&dagger;</sup>Correspondence, <sup>&ddagger;</sup>Project Lead

> Beijing Jiaotong University, ByteDance Seed

<font size=7><div align='center' > <a href='https://arxiv.org/pdf/2501.09781'>**Paper**</a> | <a href="https://maverickren.github.io/VideoWorld2.github.io/">**Project Page**</a> | <a href="#installation">**Installation**</a> | <a href="#training">**Training**</a> | <a href="#inference">**Inference**</a> | <a href="https://huggingface.co/maverickrzw/Video-CraftBench_v0.1/tree/main">**Video-CraftBench**</a></div></font>

<img width="1000" alt="image" src='assets/readme_figs/Fig1_final.png'>

## :fire: News
* **[2026.02]** We release the code and dataset.

# Highlight

👉 We are the first to explore how to learn transferable world knowledge for complex long-horizon tasks directly from raw real-world videos, and we reveal that disentangling action dynamics from visual appearance is essential for successful knowledge learning.

👉 We propose VideoWorld 2, whose core is a dynamic-enhanced Latent Dynamics Model (dLDM) that decouples task-relevant dynamics from visual appearance, enhancing the quality and transferability of learned knowledge.

👉 We construct Video-CraftBench to address the rarely explored challenge of fine-grained, long-horizon visual reasoning through real-world handicraft tasks. This benchmark facilitates future research on learning transferable knowledge from raw videos.

# Introduction
Learning transferable knowledge from unlabeled video data and applying it in new environments is a fundamental capability of intelligent agents. This work presents VideoWorld 2, which extends VideoWorld and offers the first investigation into learning transferable knowledge directly from raw real-world videos. At its core, VideoWorld 2 introduces a dynamic-enhanced Latent Dynamics Model (dLDM) that decouples action dynamics from visual appearance: a pretrained video diffusion model handles visual appearance modeling, enabling the dLDM to learn latent codes that focus on compact and meaningful task-related dynamics. These latent codes are then modeled autoregressively to learn task policies and support long-horizon reasoning. We evaluate VideoWorld 2 on challenging real-world handcraft making tasks, where prior video generation and latent-dynamics models struggle to operate reliably. Remarkably, VideoWorld 2 achieves up to \textbf{70\% improvement in task success rate} and produces coherent long execution videos. In robotics, we show that VideoWorld 2 can acquire effective manipulation knowledge from the Open-X dataset, which substantially improves task performance on CALVIN. This study reveals the potential of learning transferable world knowledge directly from raw videos, with all code, data, and models open-sourced for further research.

# Video
[![IMAGE ALT TEXT](assets/readme_figs/Thumbnail.png)](https://youtu.be/2qz7_iLV7qk "VideoWorld 2 Demo")

# Architecture

<img width="1000" alt="image" src='assets/readme_figs/method_final.png'>

Overview of the VideoWorld 2 model architecture. (Left) First, a dLDM compresses future visual changes into compact and generalizable latent codes. These codes are then modeled by an autoregressive transformer. 
(Right) In inference, the transformer predicts latent codes for a new, unseen environment from the input image, which are subsequently decoded into task execution videos.


# Installation

### Setup Environment
```
conda create -n videoworld2 python=3.11 -y
conda activate videoworld2
pip install --upgrade pip  

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```
### Install VideoWorld
```
git clone https://github.com/bytedance/VideoWorld2.git
cd VideoWorld2

bash install.sh
```


# Inference
### VideoCraft-Bench
We provide the capability to generate videos from input images using dLDM latent codes. Users can select an initial scene image from `asset/videocraft_example` and visualize the complete task execution using latent codes predicted by our AR Transformer.

```
cd VideoWorld2 # This VideoWorld is located in a subdirectory.
bash scripts/test.sh
```
We provide the [weight files](https://huggingface.co/maverickrzw/VideoWorld2_dLDM_2B/tree/main) for testing on VideoCraft-Bench. Please place the various weights according to the following directory structure to start the test. VideoWorld2 will generate the imagined video in `./infer_output`
```
├── VideoWorld2
│   ├── checkpoints
│   │   └── VideoWorld2_dLDM_2B
│   │       │── tokenizer
│   │       │── default_neg.pickle
│   │       │── VideoWorld2_dLDM_DiT.pth
│   │       │── VideoCraft-dLDM-codes.pt
│   └──     └── VideoWorld2_dLDM_VAE.pt
```

# Training
We provide the procedure for training the VideoWorld 2 dLDM. 


## Dataset
The model is trained on the Video-CraftBench and OpenX datasets.

Video-CraftBench: Please download the data from [Video-CraftBench](https://huggingface.co/maverickrzw/Video-CraftBench_v0.1/tree/main). We provide the sliced data used for training (`Paper_and_Block_clips.tar.gz`), as well as the original MP4 files and keyframe annotations.
OpenX: Our [OpenX data](https://huggingface.co/datasets/jxu124/OpenX-Embodiment/tree/main) is downloaded from Hugging Face. After downloading, please extract the tar archives. Each sample is stored in a pickle file, and the extracted file structure must be consistent with the original paths.


Please organize the downloaded data into the following directory structure:

```
├── VideoWorld2
│   ├── datasets
│   │   │── Video-CraftBench
│   │   │   └── Paper_and_Block_clips
│   │   │      │── buildingblock_horse_0_0.mp4
│   │   │      └── ...
│   │   │── openx_untar
│   │   │   │── asu_table_top_converted_externally_to_rlds
│   │   │   │── ...
│   │   │   └── viola
│   │   │── openx_videocraft_cache.json
│   │   └── openx_videocraft_meta.json
```

`openx_videocraft_cache.json` pre-stores information for all samples from the OpenX and Video-CraftBench datasets. `openx_videocraft_meta.json` stores the proportional distribution of samples across different sub-datasets. Together, they can accelerate the loading speed during mixed data training.

## Checkpoints

You will need to download the [`Cosmos-Predict2-2B-Video2World`](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World) pre-trained weights beforehand. Please place them under `checkpoints/Cosmos-Predict2-2B-Video2World/`

```
├── VideoWorld2
│   ├── checkpoints
│   │   │── VideoWorld2_dLDM_2B
│   │   └── Cosmos-Predict2-2B-Video2World
```

## Start training

Training the dLDM consists of two steps. First:

```
cd VideoWorld2 # This VideoWorld is located in a subdirectory.
bash scripts/train_dldm_warmup.sh
```
Then, run:

```
bash scripts/train_dldm.sh
bash scripts/inference_dldm_codes.sh
```
The script will automatically load the weights saved from the previous step.

After training is complete, the model will automatically generate the latent codes for the full video. These codes are created by sequentially merging the clips from the training set and are saved to `./latent_code_infos.pt`. This file is provided for users to train their AR models.


# Citation
If you find this project useful in your research, please consider citing:
```
@misc{ren2025videoworldexploringknowledgelearning,
      title={VideoWorld: Exploring Knowledge Learning from Unlabeled Videos}, 
      author={Zhongwei Ren and Yunchao Wei and Xun Guo and Yao Zhao and Bingyi Kang and Jiashi Feng and Xiaojie Jin},
      year={2025},
      eprint={2501.09781},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.09781}, 
}
```