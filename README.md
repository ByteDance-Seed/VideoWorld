# VideoWorld Series: Learning World Models from Unlabeled Videos

This repository hosts the **VideoWorld** research project series, exploring how deep generative models can learn complex world knowledge, physics, and dynamics solely from visual inputs.

This project encompasses two main iterations:
- **[VideoWorld (CVPR 2025)](./VideoWorld)**: The first generation model using Latent Dynamics Model (LDM) for knowledge acquisition.
- **[VideoWorld 2](./VideoWorld2)**: The second generation focusing on *transferable* knowledge using disentangled Latent Dynamics Model (dLDM).

---

## 📂 Projects Overview

### 1. [VideoWorld: Exploring Knowledge Learning from Unlabeled Videos](./VideoWorld)
**Accepted by CVPR 2025**

> **Highlights:**
> *   demonstrates that video generation models can learn complex rules (e.g., Go game) without reward signals.
> *   Introduces the **Latent Dynamics Model (LDM)** to compress visual changes into informative latent codes.
> *   Achieves 5-dan professional level in Go and strong performance in robotic control tasks (CALVIN).

*   **Code:** [./VideoWorld](./VideoWorld)
*   **Paper:** [arXiv](https://arxiv.org/pdf/2501.09781)

### 2. [VideoWorld 2: Learning Transferable Knowledge from Real-world Video](./VideoWorld2)

> **Highlights:**
> *   Focuses on **transferable knowledge** and long-horizon tasks in real-world settings.
> *   Proposes the **disentangled Latent Dynamics Model (dLDM)** to decouple action dynamics from visual appearance.
> *   Significant improvements in task success rates (up to 70%) on challenging handcraft benchmarks.

*   **Code:** [./VideoWorld2](./VideoWorld2)

---

## 🚀 Getting Started

Since the two projects were developed with slightly different dependencies to maintain reproducibility, we recommend using separate environments for each.

### Prerequisites
*   Anaconda or Miniconda
*   NVIDIA GPU with CUDA support

### Setup for VideoWorld 
Please refer to [VideoWorld/README.md](./VideoWorld/README.md) for detailed instructions.
```bash
conda create -n videoworld python=3.10 -y
conda activate videoworld
# Detailed install steps inside the sub-folder
```

### Setup for VideoWorld 2
Please refer to [VideoWorld2/README.md](./VideoWorld2/README.md) for detailed instructions.
```bash
conda create -n videoworld2 python=3.11 -y
conda activate videoworld2
# Detailed install steps inside the sub-folder
```

---

## 📚 Citation

If you use this codebase or models in your research, please cite our papers:

**VideoWorld (CVPR 2025)**
```bibtex
@article{ren2025videoworld,
  title={VideoWorld: Exploring Knowledge Learning from Unlabeled Videos},
  author={Ren, Zhongwei and Wei, Yunchao and Guo, Xun and Zhao, Yao and Kang, Bingyi and Feng, Jiashi and Jin, Xiaojie},
  journal={arXiv preprint arXiv:2501.09781},
  year={2025}
}
```

**VideoWorld 2**
```bibtex
@article{ren2025videoworld2,
  title={VideoWorld 2: Learning Transferable Knowledge from Real-world Video},
  author={Ren, Zhongwei and Wei, Yunchao and Yu, Xiao and Luo, Guixun and Zhao, Yao and Feng, Jiashi and Jin, Xiaojie},
  journal={arXiv preprint},
  year={2025}
}
```

## 📄 License
This project is released under the [LICENSE](./VideoWorld/LICENSE) terms.
