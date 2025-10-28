<div align="center">

# Epipolar Geometry Improves Video Generation Models

**Orest Kupyn**<sup>1</sup> · **Fabian Manhardt**<sup>2</sup> · **Federico Tombari**<sup>2,3</sup> · **Christian Rupprecht**<sup>1</sup>

<sup>1</sup>University of Oxford · <sup>2</sup>Google · <sup>3</sup>TU Munich

<a href='https://arxiv.org/abs/2510.21615'><img src='https://img.shields.io/badge/arXiv-2510.21615-b31b1b.svg'></a>
<a href='https://epipolar-dpo.github.io/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

**This repository is a work in progress** and demonstrates how to align video diffusion models with epipolar geometry constraints using preference-based optimization. We bridge modern video generation with classical computer vision by using epipolar geometry as a reward signal in a Flow-DPO framework to improve 3D consistency in generated videos.

![Epipolar Alignment](images/architecture.svg)

## News
- [2025/01] 🔥 Initial code release - Training pipeline and evaluation metrics
- [2025/01] 📄 Paper accepted to arXiv

## Overview

Video generation models struggle with geometric inconsistencies despite massive training data. This work explores how simple epipolar geometry constraints can improve modern video diffusion models. We demonstrate that aligning diffusion models through preference-based optimization using pairwise epipolar geometry constraints yields videos with superior visual quality, enhanced 3D consistency, and significantly improved motion stability.

### Key Features

- **Flow-DPO Training**: Adaptation of DPO for flow-matching video diffusion models
- **Epipolar Geometry Constraints**: Classical computer vision as reward signals
- **3D Consistency Metrics**: Epipolar error, motion dynamics, perspective fields, and depth estimation
- **Modular Pipeline**: Complete four-step workflow from generation to evaluation
- **LoRA Adaptation**: Efficient fine-tuning of large video models

### Why Epipolar Geometry?

Epipolar geometry provides fundamental mathematical constraints for 3D consistency in videos. By computing the fundamental matrix between frame pairs and measuring Sampson distance for matched keypoints, we quantify how well generated videos adhere to rigid scene structure. Lower epipolar error indicates better 3D consistency and more realistic camera motion.

## Installation

### Requirements

- Python 3.9+
- CUDA 11.8+ (for GPU support)
- 24GB+ VRAM recommended for training

### Setup

```bash
git clone https://github.com/yourusername/synth_3d.git
cd synth_3d
pip install -r model_training/requirements.txt
```

### Model Dependencies

You'll need access to:
- **Wan Video Model**: Text-to-video or image-to-video foundation model
- **Metric Checkpoints** (optional): Pre-trained models for additional evaluations

## Workflow

The complete pipeline consists of four main steps:

### Step 1: Generate Videos with Latents

Generate videos from the same prompts using different seeds, saving both video outputs and latent representations.

```bash
python video_generation/generate_videos.py \
    --data_path /path/to/dataset \
    --json_path /path/to/captions.json \
    --output_dir /path/to/output \
    --model_path /path/to/wan/model
```

### Step 2: Evaluate and Annotate Metrics

Run evaluation on generated videos to compute 3D consistency metrics.

```bash
python metrics/run_evaluation.py \
    --video_dir /path/to/videos \
    --metadata_path /path/to/metadata.json \
    --output_path /path/to/annotated_metadata.json \
    --config metrics/config/evaluators.yaml
```

**Available Metrics:**
- **Epipolar Consistency** (primary): Measures 3D geometric consistency using fundamental matrix
- **Motion Dynamics**: Detects static vs. dynamic content
- **Perspective Fields**: Validates camera perspective
- **MET3R**: Depth-based consistency check

### Step 3: Train Reward LoRA

Train a LoRA adapter using Flow-DPO on the annotated dataset.

```bash
cd model_training/reward_lora
python train.py \
    data.metadata_path=/path/to/annotated_metadata.json \
    data.metric_name=epipolar_consistency \
    data.metric_mode=min \
    model.dit_path=/path/to/wan/model/diffusion_pytorch_model.safetensors \
    logging.output_path=/path/to/checkpoints
```

**Key Configuration:**
- Videos from the same prompt are grouped and paired (best vs. worst) for DPO training
- Training uses Flow-DPO loss to prefer geometrically consistent outputs
- See `config/train.yaml` for full configuration options

### Step 4: Generate and Evaluate with Trained LoRA

Generate videos using the trained LoRA and evaluate improvements.

```bash
python model_training/reward_lora/generate.py \
    lora_path=/path/to/checkpoint.ckpt \
    model_path=/path/to/wan/model \
    output_dir=/path/to/results

python model_training/reward_lora/evaluate.py \
    output_dir=/path/to/results
```

## Configuration

Training and evaluation are configured via YAML files in `model_training/reward_lora/config/` and `metrics/config/`.

**Training (`train.yaml`):**
```yaml
training:
  learning_rate: 5e-6
  train_strategy: dpo
  beta: 500

data:
  metric_name: "epipolar_consistency"
  metric_mode: "min"
  metric_threshold: 8.0

lora:
  rank: 64
  alpha: 128.0
```

**Evaluation (`evaluators.yaml`):**
```yaml
evaluators:
  - _target_: metrics.video_evaluation.EpipolarEvaluator
    sampling_rate: 15
    descriptor_type: "sift"
  - _target_: metrics.video_evaluation.DynamicsEvaluator
```

## Results

Our epipolar-aligned model significantly reduces artifacts and enhances motion smoothness, resulting in more geometrically consistent 3D scenes. Visit our [project page](https://epipolar-dpo.github.io/) for video comparisons and detailed results.


## Citation

If you use this code in your research, please cite:

```bibtex
@article{kupyn2025epipolar,
  title={Epipolar Geometry Improves Video Generation Models},
  author={Kupyn, Orest and Manhardt, Fabian and Tombari, Federico and Rupprecht, Christian},
  journal={arXiv preprint arXiv:2510.21615},
  year={2025},
  url={https://arxiv.org/abs/2510.21615}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work builds on:
- **DiffSynth**: Video generation framework
- **Wan Video Models**: Foundation video diffusion models
- **DPO**: Direct Preference Optimization framework
- **DeepLSD**: Line segment detection for geometry evaluation
- **Perspective Fields**: Camera geometry estimation
