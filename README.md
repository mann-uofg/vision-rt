# vision-rt

Real-time object detection across multiple video streams, with every stage on the GPU:
NVDEC decode, a fused CUDA preprocessing kernel, TensorRT (FP16 / INT8), and GPU NMS.
The point is the end-to-end number, broken down per stage, not model-only FPS.

The scenario is livestock monitoring: YOLOX detects cows (a COCO class, so no fine-tuning)
in public-domain USDA dairy footage.

> Work in progress. M0 (baseline) and M1 (TensorRT + C++) are done; M2 moves decode,
> preprocessing and NMS onto the GPU.

## Results so far

RTX 5070 Ti, WSL2, YOLOX at 640x640, batch 1, one stream. Every number comes from a file in
[`bench/results/`](bench/results), which records the conditions and versions.

### Latency, per stage

<!-- latency:begin -->
| Model | Pipeline | Precision | decode | preprocess | H2D + infer + D2H | NMS | **end-to-end mean / p50 / p99** | throughput | vs M0 |
|---|---|---|---|---|---|---|---|---|---|
| YOLOX-S | PyTorch eager (M0) | fp32-tf32 | 1.08 | 1.24 | 10.90 | 0.54 | **13.75 / 10.79 / 32.32** | 73 fps | 1.00x |
| YOLOX-S | C++ + TensorRT (M1) | fp32 | 0.55 | 1.83 | 4.14 | 0.36 | **6.87 / 6.17 / 10.27** | 146 fps | 2.00x |
| YOLOX-S | C++ + TensorRT (M1) | tf32 | 0.54 | 1.82 | 3.80 | 0.36 | **6.53 / 5.93 / 9.16** | 153 fps | 2.11x |
| YOLOX-S | C++ + TensorRT (M1) | fp16 | 0.54 | 1.95 | 3.56 | 0.36 | **6.41 / 5.93 / 9.83** | 156 fps | 2.14x |
| YOLOX-M | PyTorch eager (M0) | fp32-tf32 | 1.02 | 1.26 | 14.67 | 0.55 | **17.51 / 13.17 / 41.50** | 57 fps | 1.00x |
| YOLOX-M | C++ + TensorRT (M1) | fp32 | 0.55 | 1.85 | 5.44 | 0.36 | **8.20 / 7.78 / 11.47** | 122 fps | 2.14x |
| YOLOX-M | C++ + TensorRT (M1) | tf32 | 0.54 | 1.84 | 5.22 | 0.36 | **7.96 / 7.50 / 10.60** | 126 fps | 2.20x |
| YOLOX-M | C++ + TensorRT (M1) | fp16 | 0.54 | 1.55 | 4.06 | 0.36 | **6.50 / 5.39 / 10.46** | 154 fps | 2.69x |

Stage columns are mean ms per frame, host wall-clock; they add up to end-to-end. Clip `barn022.mp4`, 1920x1080 h264, 1406 timed frames after 100 warm-up; batch 1, one stream.
<!-- latency:end -->

- **M0, naive PyTorch:** OpenCV decode, CPU letterbox, eager PyTorch, CPU NMS. One forward pass
  of YOLOX-S launches 451 CUDA kernels worth 2.55 ms of GPU time, but the GPU is busy for only a
  third of it; the rest is waiting on Python kernel launches
  ([`eager_profile_yolox_s.json`](bench/results/eager_profile_yolox_s.json)). That's why
  YOLOX-M, with ~2.8x the FLOPs, is only 27% slower.
- **M1, C++ + TensorRT:** the same stages and the same pageable copies, with TensorRT in place
  of PyTorch and C++ in place of Python, so the change isolates the engine. The C++ letterbox
  produces bit-identical input to the Python one on the benchmark clip, and the detections
  match the baseline's on 99.98-100% of boxes
  ([`compare_cpp_sync_*.json`](bench/results)).
- What M1 shows: FP16 makes the engine ~3x faster than FP32 on its own, but end-to-end barely
  moves. Decode, the CPU letterbox, NMS and per-kernel launch overhead now dominate. That's
  what M2 is for.

### Accuracy per precision

<!-- accuracy:begin -->
| Model | Precision | mAP, PyTorch | mAP, TensorRT | mAP, C++ app | TensorRT GPU compute (model only) |
|---|---|---|---|---|---|
| YOLOX-S | fp32 | 40.26 | 40.26 | 40.23 | 1.82 ms |
| YOLOX-S | tf32 | 40.26 | 40.26 | — | 1.48 ms |
| YOLOX-S | fp16 | — | 40.25 | 40.24 | 0.57 ms |
| YOLOX-M | fp32 | 46.63 | 46.63 | — | 3.90 ms |
| YOLOX-M | tf32 | 46.64 | 46.63 | — | 3.49 ms |
| YOLOX-M | fp16 | — | 46.63 | 46.65 | 1.25 ms |

COCO val2017 mAP@0.5:0.95, all 5,000 images, score threshold 0.01, NMS IoU 0.65. "C++ app" runs the whole C++ path, including its own FFmpeg JPEG decode. The last column is trtexec's mean GPU time for the engine alone at batch 1: no decode, no pre/postprocessing, no copies. It is not an end-to-end number.
<!-- accuracy:end -->

The published YOLOX numbers are 40.5 / 46.9. With score threshold 0.001 instead of 0.01, this
pipeline gives 40.45 for YOLOX-S, so the small gap is the threshold, not the pipeline.

## Plan

| Milestone | Deliverable | Status |
|---|---|---|
| M0 Baseline | Toolchain verified on Blackwell; naive PyTorch pipeline with per-stage timing; baseline COCO mAP | done |
| M1 TensorRT FP16 | ONNX export, FP16 engine, C++ inference app, mAP matches baseline | done |
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

cmake --preset release && cmake --build --preset release && ctest --preset release
bench/build_engines.sh                                        # ONNX export + fp32/tf32/fp16 engines
python tools/eval_coco.py --size s --engine models/yolox_s_fp16.engine
bench/run_cpp_sync.sh                                         # C++ pipeline, all engines, vs baseline
python bench/make_tables.py                                   # regenerate the tables above
```

## Footage and model licences

- Footage: USDA "Dairy Cows (20220209-FPAC-RockHillDairy…)" on Wikimedia Commons, public
  domain. Downloaded by the script, not stored in the repo.
- Model: [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (Apache-2.0), pinned as a
  submodule in `third_party/YOLOX`, with the official COCO checkpoints.
