# vision-rt

Real-time object detection across multiple video streams, with every stage on the GPU:
NVDEC decode, a fused CUDA preprocessing kernel, TensorRT (FP16 / INT8), and GPU NMS.
The point is the end-to-end number, broken down per stage, not model-only FPS.

> Work in progress. Results, architecture and build instructions will land here as each
> milestone is measured.

## Plan

| Milestone | Deliverable |
|---|---|
| M0 Baseline | Toolchain verified on Blackwell; naive PyTorch pipeline with per-stage timing; baseline COCO mAP |
| M1 TensorRT FP16 | ONNX export, FP16 engine, C++ inference app, mAP matches baseline |
| M2 GPU pipeline | NVDEC decode, fused preprocessing kernel, GPU NMS, streams and pinned memory, Nsight timeline |
| M3 INT8 + multi-stream | INT8 calibration, mAP per precision, multi-stream scaling curve |
| M4 Ship | Demo video, accuracy-vs-latency plot, write-up |
