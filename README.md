# vision-rt

Real-time object detection across multiple video streams, with every stage on the GPU:
NVDEC decode, a fused CUDA preprocessing kernel, TensorRT (FP16 / INT8), and GPU NMS.
The point is the end-to-end number, broken down per stage, not model-only FPS.

The scenario is livestock monitoring: YOLOX detects cows (a COCO class, so no fine-tuning)
in public-domain USDA dairy footage.

> Work in progress. M0 (baseline) is done; the optimized pipeline comes next.

## Results so far

Naive PyTorch pipeline (the "before"): OpenCV CPU decode, CPU letterbox, PyTorch eager FP32,
CPU NMS, one frame at a time. RTX 5070 Ti, WSL2. Numbers come from
[`bench/results/`](bench/results); conditions and caveats are in each JSON file.
"fp32-tf32" is PyTorch's default, which lets cuDNN use TF32 for convolutions; with TF32 off,
mAP is the same to within 0.01.

One forward pass of YOLOX-S launches 451 CUDA kernels that add up to 2.55 ms of GPU time, but
the GPU is busy for only about a third of the forward pass (YOLOX-M: 612 kernels, 57% busy);
the rest is waiting on Python kernel launches
([`eager_profile_yolox_s.json`](bench/results/eager_profile_yolox_s.json)). That's why YOLOX-M,
with ~2.8x the FLOPs (73.8 vs 26.8 GFLOPs), is only 27% slower end to end (mean). Most of the time is overhead around the model,
which is what the rest of this project removes.

<!-- baseline:begin -->
| Model | Precision | mAP | decode | preprocess | H2D + infer + D2H | NMS | **end-to-end p50 / p99** | throughput |
|---|---|---|---|---|---|---|---|---|
| YOLOX-S | fp32-tf32 | 40.3 | 1.08 | 1.24 | 10.90 | 0.54 | **10.8 / 32.3** | 73 fps |
| YOLOX-M | fp32-tf32 | 46.6 | 1.02 | 1.26 | 14.67 | 0.55 | **13.2 / 41.5** | 57 fps |

Stage columns are mean ms per frame (host wall-clock). Clip: `barn022.mp4`, 1920x1080 h264, 30 fps; batch 1, one stream. mAP is COCO val2017 mAP@0.5:0.95 with the same pre/postprocessing.
<!-- baseline:end -->

## Plan

| Milestone | Deliverable | Status |
|---|---|---|
| M0 Baseline | Toolchain verified on Blackwell; naive PyTorch pipeline with per-stage timing; baseline COCO mAP | done |
| M1 TensorRT FP16 | ONNX export, FP16 engine, C++ inference app, mAP matches baseline | |
| M2 GPU pipeline | NVDEC decode, fused preprocessing kernel, GPU NMS, streams and pinned memory, Nsight timeline | |
| M3 INT8 + multi-stream | INT8 (explicit Q/DQ), mAP per precision, multi-stream scaling curve | |
| M4 Ship | Demo video, accuracy-vs-latency plot, write-up | |

## Setup

Linux (tested on WSL2 Ubuntu 24.04) with an NVIDIA GPU.

| Component | Version |
|---|---|
| CUDA toolkit | 13.3 |
| TensorRT | 11.3.0 (tar install) |
| FFmpeg | 8.1, [BtbN](https://github.com/BtbN/FFmpeg-Builds) GPL shared build (has the NVDEC decoders) |
| PyTorch | 2.14.1 + CUDA 13.0 wheels |
| Python | 3.12 |

```bash
git clone --recurse-submodules https://github.com/mann-uofg/vision-rt.git
cd vision-rt
python3 -m venv ~/venvs/vision-rt
source scripts/env.sh            # TensorRT, FFmpeg, CUDA paths and the venv; edit paths if yours differ
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt -e .
pip install "$TRT_ROOT"/python/tensorrt-*-cp312-*.whl

scripts/fetch_models.sh          # YOLOX-S / YOLOX-M COCO weights, hash-checked
scripts/fetch_coco.sh            # COCO val2017 (~1 GB)
scripts/fetch_footage.sh         # test clips, transcoded to 1080p H.264
```

## Run

```bash
python -m pytest -q                                           # CPU-side tests
python tools/eval_coco.py --size s                            # COCO val2017 mAP
python tools/baseline.py --video data/footage/barn022.mp4     # naive pipeline, per-stage timing
bench/run_baseline.sh                                         # both sizes, refuses a busy GPU
```

## Footage and model licences

- Footage: USDA "Dairy Cows (20220209-FPAC-RockHillDairy…)" on Wikimedia Commons, public
  domain. Downloaded by the script, not stored in the repo.
- Model: [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (Apache-2.0), pinned as a
  submodule in `third_party/YOLOX`, with the official COCO checkpoints.
