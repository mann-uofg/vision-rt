"""Run a serialized TensorRT engine from Python, with torch tensors as the device buffers.

Used for accuracy checks (COCO mAP per precision). The C++ app is the real pipeline.
"""

from pathlib import Path

import tensorrt as trt
import torch

_DTYPES = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT8: torch.int8,
    trt.DataType.INT32: torch.int32,
    trt.DataType.BOOL: torch.bool,
}


class TrtModel:
    """Single-input, single-output engine with static shapes."""

    def __init__(self, engine_path: str | Path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize {engine_path}")
        self.context = self.engine.create_execution_context()

        self.buffers: dict[str, torch.Tensor] = {}
        inputs, outputs = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = _DTYPES[self.engine.get_tensor_dtype(name)]
            self.buffers[name] = torch.empty(shape, dtype=dtype, device="cuda")
            self.context.set_tensor_address(name, self.buffers[name].data_ptr())
            mode = self.engine.get_tensor_mode(name)
            (inputs if mode == trt.TensorIOMode.INPUT else outputs).append(name)
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(f"expected 1 input and 1 output, got {inputs} -> {outputs}")
        self.input_name, self.output_name = inputs[0], outputs[0]

    @property
    def input_dtype(self) -> torch.dtype:
        return self.buffers[self.input_name].dtype

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Copy x into the input buffer, run on torch's current stream, return the output.

        The returned tensor is the engine's own output buffer; it's overwritten by the next call.
        """
        self.buffers[self.input_name].copy_(x)
        stream = torch.cuda.current_stream().cuda_stream
        if not self.context.execute_async_v3(stream):
            raise RuntimeError("execute_async_v3 failed")
        return self.buffers[self.output_name]
