# VNL-STES: A Benchmark Dataset and Model for Spatiotemporal Event Spotting in Volleyball Analytics

> **🔧 This is a modified fork for GPU portability**
>
> Original repository: [jhong93/spot](https://github.com/jhong93/spot)
> 
> **Modifications**: Added support for CUDA, MPS (Apple Metal), and CPU devices. See [Modifications](#modifications-for-gpu-portability) section below.

**CVPR Workshop 2025**

Hoang Quoc Nguyen, Ankhzaya Jamsrandorj, Vanyi Chao, Yin May Oo, Muhammad Amrulloh Robbani, Kyung-Ryoul Mun, Jinwook Kim

[[Project Page]](https://hoangqnguyen.github.io/stes/) [[Paper]](https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/papers/Nguyen_VNL-STES_A_Benchmark_Dataset_and_Model_for_Spatiotemporal_Event_Spotting_CVPRW_2025_paper.pdf)

We introduce **Precise Spatiotemporal Event Spotting**, a novel task that jointly determines *when* and *where* key events occur in volleyball videos at single-frame temporal precision. We present the **VNL Dataset** (1,028 rally videos, 251,110 frames, 6,137 annotated events with temporal and spatial labels) and the **Spatiotemporal Event Spotter (STES)** model.

STES outperforms the state-of-the-art in temporal action spotting by **+9.86 mTAP** and achieves **80.21 mSAP@2-6P** for spatial localization, pinpointing event locations within a 2-6 pixel range.

## Architecture

STES comprises three components:

- **(A) Feature Extractor**: RegNet-Y 800MF backbone with Gated Shift Modules (GSM) to capture spatiotemporal features
- **(B) Temporal Aggregator**: Multi-layer bidirectional GRU for long-term temporal modeling and event classification
- **(C) Spatial Predictor**: MLP head that regresses normalized (x, y) event coordinates from backbone features

## Results

### Temporal Event Spotting (mTAP)

| Model | @0F | @1F | @2F | @4F | @0-4F |
|-------|-----|-----|-----|-----|-------|
| E2E Spot | 32.37 | 57.95 | 63.37 | 67.58 | 55.32 |
| T-DEED | 30.41 | 63.39 | 69.00 | 71.51 | 58.58 |
| E2E Spatial | 44.56 | 70.03 | 72.15 | 72.51 | 64.81 |
| **STES (Ours)** | **46.76** | **73.64** | **76.29** | **77.06** | **68.44** |

### Spatial Event Spotting (mSAP)

| Metric | E2E Spatial | STES (Ours) |
|--------|-------------|-------------|
| mSAP@2P | 57.16 | **69.63** |
| mSAP@4P | 79.86 | **84.52** |
| mSAP@6P | 83.82 | **86.47** |
| mSAP@2-6P | 73.61 | **80.21** |

## VNL Dataset

The dataset contains 8 full VNL matches (2022-2023 seasons), split into 1,028 rally clips with 6 event classes:

| Action | Count | % |
|--------|-------|---|
| Serve | 1,071 | 17.45 |
| Receive | 1,558 | 25.39 |
| Set | 1,393 | 22.70 |
| Spike | 1,321 | 21.53 |
| Block | 550 | 8.96 |
| Score | 244 | 3.97 |

**Splits**: 811 train / 102 val / 115 test rallies.

Download the dataset (resized 398x224, 13GB) at: https://hoangqnguyen.github.io/stes/

## Setup

```bash
pip install -r requirements.txt
```

Tested on Linux with Python 3.10+ and PyTorch 2.x with CUDA.

## Data Preparation

### Frame extraction

Extract video frames resized to 224px height:

```
<frame_dir>/
├── video1/
│   ├── 000000.jpg
│   ├── 000001.jpg
│   └── ...
├── video2/
│   └── ...
```

### Dataset metadata

Place dataset files under `data/<dataset_name>/`:

- **`class.txt`** — one class name per line
- **`train.json`**, **`val.json`**, **`test.json`** — annotation files

Annotation format:
```json
[
    {
        "video": "video_id",
        "num_frames": 3600,
        "num_events": 12,
        "events": [
            {
                "frame": 525,
                "label": "spike",
                "comment": "",
                "x": 0.45,
                "y": 0.62
            }
        ],
        "fps": 25,
        "width": 1920,
        "height": 1080
    }
]
```

## Training

### Train from scratch

```bash
python train_e2e_spatial.py <dataset> <frame_dir> \
  -m rny008_gsm -t gru \
  --clip_len 64 --batch_size 16 \
  --num_epochs 150 \
  -s exp/<experiment_name> \
  --predict_location \
  --num_workers 4
```

### Pretrain + Finetune

```bash
# Stage 1: Pretrain
python train_e2e_spatial.py vnl_1.5 data/vnl_1.5/frames_224p \
  -m rny008_gsm -t gru --clip_len 64 --batch_size 16 \
  --num_epochs 100 -s exp/pretrain_finetune --predict_location

# Stage 2: Finetune (resumes from last checkpoint)
python train_e2e_spatial.py vnl_2.0 data/vnl_2.0/frames_224p \
  -m rny008_gsm -t gru --clip_len 64 --batch_size 16 \
  --num_epochs 200 -s exp/pretrain_finetune --predict_location --resume
```

See `run.sh` and `pretrain_finetune.sh` for ready-to-use scripts.

## Evaluation

```bash
python eval.py <model_dir_or_pred_file> -s <split>
```

Predictions are saved as `pred-{split}.{epoch}.recall.json.gz` during training.

## Inference on Video

```bash
python inference_on_mp4.py --checkpoint_path <path_to_checkpoint> --video <path_to_video>
```

## Demo App

Gradio web interface for uploading videos and visualizing detected events:

```bash
python app.py
```

## Project Structure

```
├── train_e2e_spatial.py    # Main training script
├── eval.py                 # Evaluation with mean-AP
├── inference_on_mp4.py     # Inference on raw video files
├── app.py                  # Gradio demo app
├── model/
│   ├── common.py           # Base model classes, MLP
│   ├── modules.py          # GRU head, location predictor, channel attention
│   ├── shift.py            # TSM and GSM temporal shift modules
│   └── min_gru.py          # Minimal GRU variant
├── dataset/
│   ├── frame.py            # Frame-based dataset classes
│   └── transform.py        # Data augmentation transforms
├── util/
│   ├── score.py            # mAP computation and location-aware metrics
│   ├── eval.py             # Frame prediction post-processing
│   ├── dataset.py          # Dataset registry and helpers
│   └── io.py               # JSON/GZ I/O utilities
├── run.sh                  # Training script
├── pretrain_finetune.sh    # Two-stage pretrain+finetune script
└── requirements.txt
```
## Modifications for GPU Portability

This fork adds **automatic device selection** to support CUDA, MPS (Apple Metal), and CPU without changing the main training code.

### New Files

- **`util/device.py`**: Device selection utilities
  - `select_device(prefer: str)` — Auto-select best device (CUDA > MPS > CPU)
  - `get_autocast_context(device)` — Device-aware mixed precision
  - `get_grad_scaler(device)` — Device-aware gradient scaling

### Modified Files

| File | Changes |
|------|---------|
| `train_e2e_spatial.py` | Import device utilities; replace `torch.cuda.amp.autocast()` with `get_autocast_context()` |
| `model/common.py` | Use `get_grad_scaler()` instead of checking `device == "cuda"` |
| `model/shift.py` | Replace CUDA-specific `FloatTensor` with device-agnostic `x.new_zeros()` |
| `requirements.txt` | Added `transformers>=4.30.0` (for ConvNeXt V2) |

### Usage

The code automatically detects and uses:
- **NVIDIA CUDA** on GPU machines
- **Apple Metal (MPS)** on M1/M2/M3+ Macs
- **CPU** as fallback

No code changes needed. Just run training/inference as before:

```bash
python train_e2e_spatial.py <dataset> <frame_dir> -m rny008_gsm -t gru ...
# Automatically selects the best device
```

### Original Repository

- **Source**: [jhong93/spot](https://github.com/jhong93/spot)
- **License**: BSD 2-Clause (see [LICENSE](LICENSE))
- **Citation**: See citation section below
## Citation

```bibtex
@inproceedings{nguyen2025vnlstes,
    author={Nguyen, Hoang Quoc and Jamsrandorj, Ankhzaya and Chao, Vanyi and Oo, Yin May and Robbani, Muhammad Amrulloh and Mun, Kyung-Ryoul and Kim, Jinwook},
    title={VNL-STES: A Benchmark Dataset and Model for Spatiotemporal Event Spotting in Volleyball Analytics},
    booktitle={CVPR Workshops},
    year={2025}
}
```

## Acknowledgments

This work was supported by Athletes' training/matches data management and AI-based performance enhancement solution technology development Project (No.1375027432) and Korea Institute of Science and Technology (KIST) Institutional Program (No. 2E33841).

This project builds on [E2E-Spot](https://github.com/jhong93/spot) by Hong et al. (ECCV 2022).

## License

BSD-3 — see [LICENSE](LICENSE).
