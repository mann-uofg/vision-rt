"""NV12 -> BGR reference, the maths the GPU preprocessing kernel implements.

Chroma is 4:2:0 and is read nearest-neighbour (each UV pair covers a 2x2 block of luma), which
is what swscale's unscaled yuv420p -> bgr24 path does. Each pixel is rounded to uint8 after
the matrix, like a decoded BGR frame.
"""

import numpy as np

# (Kr, Kb) luma coefficients.
_K = {"bt709": (0.2126, 0.0722), "bt601": (0.299, 0.114)}


def yuv_to_bgr_coeffs(matrix: str, full_range: bool) -> dict:
    kr, kb = _K[matrix]
    kg = 1.0 - kr - kb
    # Limited ("TV") range: Y in [16, 235], UV in [16, 240].
    y_scale = 1.0 if full_range else 255.0 / 219.0
    c_scale = 1.0 if full_range else 255.0 / 224.0
    return {
        "y_off": 0.0 if full_range else 16.0,
        "y_scale": y_scale,
        "r_v": 2.0 * (1.0 - kr) * c_scale,
        "g_u": -2.0 * (1.0 - kb) * kb / kg * c_scale,
        "g_v": -2.0 * (1.0 - kr) * kr / kg * c_scale,
        "b_u": 2.0 * (1.0 - kb) * c_scale,
    }


def nv12_to_bgr(nv12: np.ndarray, width: int, height: int, matrix: str = "bt709",
                full_range: bool = False) -> np.ndarray:
    """Tightly packed NV12 (Y plane then interleaved UV) -> HxWx3 uint8 BGR."""
    y = nv12[: width * height].reshape(height, width).astype(np.float32)
    uv = nv12[width * height :].reshape(height // 2, width // 2, 2).astype(np.float32)
    u = np.repeat(np.repeat(uv[..., 0], 2, axis=0), 2, axis=1) - 128.0
    v = np.repeat(np.repeat(uv[..., 1], 2, axis=0), 2, axis=1) - 128.0
    c = yuv_to_bgr_coeffs(matrix, full_range)
    yy = (y - c["y_off"]) * c["y_scale"]
    r = yy + c["r_v"] * v
    g = yy + c["g_u"] * u + c["g_v"] * v
    b = yy + c["b_u"] * u
    bgr = np.stack([b, g, r], axis=-1)
    return np.clip(np.floor(bgr + 0.5), 0, 255).astype(np.uint8)
