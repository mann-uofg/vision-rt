# vision-rt

Real-time object detection across multiple video streams, with every stage on the GPU:
NVDEC decode, a fused CUDA preprocessing kernel, TensorRT (FP16 / INT8), and GPU NMS.
The point is the end-to-end number, broken down per stage, not model-only FPS.

The scenario is livestock monitoring: YOLOX detects cows (a COCO class, so no fine-tuning)
in public-domain USDA dairy footage.

> Work in progress. M0-M3 are done: the whole per-frame path runs on the GPU, INT8 is measured,
> and one GPU serves 32 live 1080p cameras. Next: the demo video and write-up (M4).

## Results so far

RTX 5070 Ti, WSL2, YOLOX at 640x640. Every number comes from a file in
[`bench/results/`](bench/results), which records the conditions and versions.

### Where the time goes, step by step (YOLOX-S)

<!-- latency:begin -->
| Pipeline | Precision | decode | preprocess | inference | NMS | copies | **latency** mean / p50 / p99 | **fps** | vs M0 | CPU ms/frame |
|---|---|---|---|---|---|---|---|---|---|---|
| M0: PyTorch eager, CPU decode/pre/NMS | fp32-tf32 | 0.98 | 1.24 | 9.74 | 0.56 | 0.80 | **13.73** / 10.67 / 34.62 | **73** | 1.0x | 49.6 |
| M1: C++ + TensorRT, CPU decode/pre/NMS | fp16 | 0.56 | 1.60 | 2.80 | 0.32 | 0.50 | **6.02** / 5.06 / 9.51 | **166** | 2.3x | 10.3 |
| M1 + CUDA graph for the engine | fp16 | 0.56 | 1.59 | 0.60 | 0.33 | 0.51 | **3.71** / 3.64 / 4.62 | **270** | 3.7x | 7.6 |
| M2: NVDEC + GPU pre/NMS, one frame at a time | fp16 | 0.97 | 0.03 | 0.59 | 0.15 | 0.01 | **1.80** / 1.75 / 2.93 | **548** | 7.5x | 1.1 |
| M2, decode overlapped with GPU work | fp16 | 1.04 | 0.04 | 0.59 | 0.16 | 0.01 | **2.55** / 2.51 / 4.00 | **775** | 10.6x | 0.5 |

Mean ms per frame on `barn022.mp4` (1920x1080 H.264), batch 1, one stream, 1406 timed frames after 100 warm-up. CPU stages are host wall-clock; GPU stages (inference everywhere, and preprocess/NMS/copies in M2) are CUDA event times. Latency is from asking for a frame to having its detections in host memory. In the pipelined run stages overlap, so they don't add up to latency, and latency includes waiting behind the next frame's decode. CPU ms/frame is process CPU time (all threads) over the timed frames.
<!-- latency:end -->

![M1 timeline](docs/timeline_m1.png)
![M2 timeline](docs/timeline_m2.png)

Four frames of each, drawn from Nsight Systems traces
([`bench/profile_timelines.sh`](bench/profile_timelines.sh)). In M1 the CPU thread never
stops: decode, letterbox, then waiting while ~160 kernels are launched one by one, then NMS;
the GPU works in sparse bursts. In M2 the CPU thread decodes (NVDEC) and queues work; the GPU
runs each frame as one dense block of inference, a few NMS kernels and a tiny result copy,
and is idle about half the time. That headroom is what multiple camera streams will use (M3).

How each step earned its gain:

- **M0 → M1 (TensorRT, C++): 2.3x.** One YOLOX-S forward pass in eager PyTorch launches 451
  kernels and keeps the GPU busy only a third of the time
  ([`eager_profile_yolox_s.json`](bench/results/eager_profile_yolox_s.json)). The TensorRT
  engine computes the same thing in 0.57 ms (trtexec), and the FP16 engine costs 0.01 mAP.
- **M1 → CUDA graph: 1.6x.** Nsight showed `enqueueV3` spending 1.6 ms median of CPU per frame
  issuing ~160 launches through WSL's driver. Replaying a captured graph made the inference
  stage 0.59 ms, matching trtexec, with identical detections.
- **→ M2 (frames stay on the GPU): 2.1x.** NVDEC decodes into GPU memory, one fused kernel turns
  NV12 into the FP16 input tensor (letterbox, colour conversion, layout) in ~0.03 ms instead of
  1.6 ms on the CPU, and NMS runs on the GPU. Only ~2.4 KB of detections cross PCIe per frame.
  CPU time drops from 49.6 ms/frame (M0, all threads) to about 1.
- **→ pipelined: 1.4x more throughput** by decoding frame n+1 while the GPU runs frame n, at the
  cost of ~0.75 ms more latency per frame. Throughput is now capped by NVDEC (~1.1 ms per
  1080p frame through FFmpeg).

Correctness: every variant's detections are compared frame by frame against the M0 baseline
([`compare_*.json`](bench/results)). M1 matches on 99.98-100% of boxes. M2 matches on 99.8%
(mean IoU 0.993): its kernel uses the exact BT.709 conversion, while OpenCV's fast path
rounds about one level dark. The preprocessing kernel and GPU NMS are bit-identical to the
C++ reference in the tests ([`tests/test_gpu_kernels.py`](tests/test_gpu_kernels.py)).

### Every model and precision on the M2 pipeline

<!-- m2:begin -->
| Model | Precision | mAP (TensorRT) | M2 sync latency mean / p99 | M2 sync fps | M2 pipelined fps | GPU time per frame | vs M0 (fps) |
|---|---|---|---|---|---|---|---|
| YOLOX-S | fp32 | 40.26 | 3.13 / 4.35 ms | 317 | 480 | 1.98 ms | 6.6x |
| YOLOX-S | tf32 | 40.26 | 2.86 / 4.01 ms | 346 | 583 | 1.62 ms | 8.0x |
| YOLOX-S | fp16 | 40.25 | 1.80 / 2.93 ms | 548 | 775 | 0.77 ms | 10.6x |
| YOLOX-S | int8 (entropy) | 39.61 | 1.93 / 3.14 ms | 511 | 776 | 0.75 ms | 10.7x |
| YOLOX-M | fp32 | 46.63 | 5.32 / 6.50 ms | 187 | 244 | 4.06 ms | 4.0x |
| YOLOX-M | tf32 | 46.63 | 5.07 / 6.36 ms | 196 | 258 | 3.63 ms | 4.2x |
| YOLOX-M | fp16 | 46.63 | 2.74 / 3.95 ms | 361 | 645 | 1.43 ms | 10.5x |
| YOLOX-M | int8, fp32 elsewhere | 45.54 | 2.74 / 3.85 ms | 361 | 624 | 1.47 ms | 10.2x |
<!-- m2:end -->

### Many cameras on one GPU

Each camera stream gets its own NVDEC decoder, CUDA stream, TensorRT execution context and
worker thread; the engine itself is loaded once and shared
([`apps/vrt_multistream.cpp`](apps/vrt_multistream.cpp), [`bench/run_multistream.sh`](bench/run_multistream.sh)).

![Scaling](docs/scaling.png)

With every stream running flat out, YOLOX-S FP16 tops out at about **1,700 fps across all
streams** from 6 streams on. The GPU is ~91% busy there while NVDEC sits at ~65%, so the limit
is GPU compute, not decode; past the knee, throughput stays flat and latency grows linearly
with the number of streams (that's queueing). Concurrent streams overlap each other's kernels,
which is why the total beats the ~1,300 fps a single stream's 0.77 ms of GPU work would allow.
YOLOX-M saturates the GPU at 3 streams (~1,000 fps).

![Real-time cameras](docs/realtime.png)

Paced like real cameras (30 fps each, phases spread across the frame interval, one frame at a
time), one RTX 5070 Ti runs **32 cameras with YOLOX-S and every stream's p99 under 5.4 ms**,
or 24 with YOLOX-M (p99 under 5.6 ms). One step further, the tail blows up: at 40 cameras
YOLOX-S demand is ~83% of what the GPU serves one frame at a time (~1,450 fps), and while the
median stays at 2.7 ms, p99 jumps to ~0.7 s. At 32 cameras YOLOX-M needs more than the GPU can
give and falls behind.

### Accuracy vs latency

![Pareto](docs/pareto.png)

FP16 is the frontier for both sizes. TF32 and FP32 add latency for no accuracy. INT8
(post-training, explicit Q/DQ from TensorRT Model Optimizer, entropy calibration on COCO
*train* images) costs YOLOX-S 0.64 mAP and gains nothing end to end: at batch 1 the engine is
only 6% faster and the pipeline is decode-bound. YOLOX-M with INT8 + FP16 gives wrong output
on TensorRT 11.3; the bisection and what was ruled out are in
[`int8_yolox_m_investigation.json`](bench/results/int8_yolox_m_investigation.json).

### TensorRT's built-in NMS layer vs mine

TensorRT 11 replaced the EfficientNMS plugin with a built-in NMS layer. Appending ONNX
NonMaxSuppression to the graph ([`tools/add_nms.py`](tools/add_nms.py)) and running it against
my NMS on the same raw predictions over COCO val
([`tools/compare_nms_layer.py`](tools/compare_nms_layer.py)): identical mAP (40.248), identical
boxes on 4,991 of 5,000 images at deploy thresholds. But the layer's output size is
data-dependent, so inference has to sync with the host mid-run and the engine can no longer be
captured as a CUDA graph: it adds ~0.46 ms per frame on real images and takes the engine from
0.57 to ~2.5 ms in trtexec. My kernels write a fixed-size result block and cost ~0.15 ms.

### Accuracy per precision

<!-- accuracy:begin -->
| Model | Precision | mAP, PyTorch | mAP, TensorRT | mAP, C++ app | TensorRT GPU compute (model only) |
|---|---|---|---|---|---|
| YOLOX-S | fp32 | 40.26 | 40.26 | 40.23 | 1.82 ms |
| YOLOX-S | tf32 | 40.26 | 40.26 | — | 1.48 ms |
| YOLOX-S | fp16 | — | 40.25 | 40.24 | 0.57 ms |
| YOLOX-S | int8 (entropy) | — | 39.61 | — | 0.54 ms |
| YOLOX-S | int8 (max calib.) | — | 38.57 | — | 0.56 ms |
| YOLOX-M | fp32 | 46.63 | 46.63 | — | 3.90 ms |
| YOLOX-M | tf32 | 46.64 | 46.63 | — | 3.49 ms |
| YOLOX-M | fp16 | — | 46.63 | 46.65 | 1.25 ms |
| YOLOX-M | int8 (entropy) ⚠ | — | 18.29 | — | 1.04 ms |
| YOLOX-M | int8, fp32 elsewhere | — | 45.54 | — | 1.25 ms |

COCO val2017 mAP@0.5:0.95, all 5,000 images, score threshold 0.01, NMS IoU 0.65. "C++ app" runs the whole C++ path, including its own FFmpeg JPEG decode. The last column is trtexec's mean GPU time for the engine alone at batch 1: no decode, no pre/postprocessing, no copies. It is not an end-to-end number. INT8 is post-training quantization with explicit Q/DQ (Model Optimizer, entropy or max calibration on 512 COCO train images). ⚠ YOLOX-M with INT8 + FP16 gives wrong output on TensorRT 11.3; keeping the non-quantized layers in FP32 restores accuracy but is no faster than plain FP16, so YOLOX-M deploys as FP16.
<!-- accuracy:end -->

The published YOLOX numbers are 40.5 / 46.9. With score threshold 0.001 instead of 0.01, this
pipeline gives 40.45 for YOLOX-S, so the small gap is the threshold, not the pipeline.

## Plan

| Milestone | Deliverable | Status |
|---|---|---|
| M0 Baseline | Toolchain verified on Blackwell; naive PyTorch pipeline with per-stage timing; baseline COCO mAP | done |
| M1 TensorRT FP16 | ONNX export, FP16 engine, C++ inference app, mAP matches baseline | done |
| M2 GPU pipeline | NVDEC decode, fused preprocessing kernel, GPU NMS, streams and pinned memory, Nsight timeline | done |
| M3 INT8 + multi-stream | INT8 (explicit Q/DQ), mAP per precision, multi-stream scaling curve | done |
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
bench/run_all.sh                                              # decode, baseline, every pipeline
bench/run_multistream.sh                                      # scaling: capacity and 30 fps cameras
python tools/quantize_int8.py --size s                        # INT8 Q/DQ ONNX (needs make_calib.py)
python bench/make_tables.py                                   # regenerate the tables above
```

## Footage and model licences

- Footage: USDA "Dairy Cows (20220209-FPAC-RockHillDairy…)" on Wikimedia Commons, public
  domain. Downloaded by the script, not stored in the repo.
- Model: [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (Apache-2.0), pinned as a
  submodule in `third_party/YOLOX`, with the official COCO checkpoints.
