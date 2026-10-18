"""Run a serialized TensorRT engine from Python, with torch tensors as the device buffers.

Used for accuracy checks (COCO mAP per precision) and the NMS-layer comparison. The C++ app is
the real pipeline.
"""

from pathlib import Path

import tensorrt as trt
import torch

_DTYPES = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT8: torch.int8,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT64: torch.int64,
    trt.DataType.BOOL: torch.bool,
}


class _GrowingAllocator(trt.IOutputAllocator):
    """Output whose size TensorRT only knows after running (e.g. the NMS layer's [K, 3])."""

    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.dtype = dtype
        self.buffer = torch.empty(0, dtype=torch.uint8, device="cuda")
        self.shape: tuple[int, ...] = ()

    def reallocate_output_async(self, tensor_name, memory, size, alignment, stream):
        if size > self.buffer.numel():
            self.buffer = torch.empty(max(size, 1), dtype=torch.uint8, device="cuda")
        return self.buffer.data_ptr()

    def notify_shape(self, tensor_name, shape):
        self.shape = tuple(shape)

    def tensor(self) -> torch.Tensor:
        count = 1
        for d in self.shape:
            count *= d
        itemsize = torch.empty(0, dtype=self.dtype).element_size()
        return self.buffer[: count * itemsize].view(self.dtype).view(self.shape)


class TrtModel:
    """Single-input engine. Static outputs get fixed buffers; data-dependent ones an allocator."""

    def __init__(self, engine_path: str | Path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize {engine_path}")
        self.context = self.engine.create_execution_context()

        self.buffers: dict[str, torch.Tensor] = {}
        self.allocators: dict[str, _GrowingAllocator] = {}
        inputs, self.output_names = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = _DTYPES[self.engine.get_tensor_dtype(name)]
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            (inputs if is_input else self.output_names).append(name)
            if not is_input and any(d < 0 for d in shape):
                self.allocators[name] = _GrowingAllocator(dtype)
                self.context.set_output_allocator(name, self.allocators[name])
            else:
                self.buffers[name] = torch.empty(shape, dtype=dtype, device="cuda")
                self.context.set_tensor_address(name, self.buffers[name].data_ptr())
        if len(inputs) != 1:
            raise ValueError(f"expected 1 input, got {inputs}")
        self.input_name = inputs[0]

    @property
    def input_dtype(self) -> torch.dtype:
        return self.buffers[self.input_name].dtype

    def run(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run on torch's current stream; returns every output by name (engine-owned buffers,
        overwritten by the next call)."""
        self.buffers[self.input_name].copy_(x)
        stream = torch.cuda.current_stream().cuda_stream
        if not self.context.execute_async_v3(stream):
            raise RuntimeError("execute_async_v3 failed")
        if self.allocators:
            torch.cuda.current_stream().synchronize()  # shapes are known only after the run
        out = {n: self.buffers[n] for n in self.output_names if n in self.buffers}
        out.update({n: a.tensor() for n, a in self.allocators.items()})
        return out

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """The first output (the head's predictions), for engines used like the PyTorch model."""
        return self.run(x)[self.output_names[0]]
