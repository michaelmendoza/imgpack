# utils_bin.py
from __future__ import annotations
import json, struct
from typing import Dict, Tuple, Any
import numpy as np

MAGIC = b"IMPK"   # 4 bytes
VERSION = 1       # 1 byte

def encode_data(
    data: np.ndarray,
    vmin: float,
    vmax: float,
    dtype: str = "uint16",
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Quantize a 2D numpy array into uint8/uint16 and return (blob, header).
    Blob is C-contiguous little-endian for easy JS consumption.
    """
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("encode_data expects a 2D array (HxW).")

    if dtype == "uint8":
        resolution = 255
        out_dt = np.dtype("<u1")
        clip_hi = 255.0
        dtype_str = "uint8"
    else:
        resolution = 4096
        out_dt = np.dtype("<u2")
        clip_hi = 4096.0
        dtype_str = "uint16"

    a = np.asarray(data, dtype=np.float32)
    denom = float(vmax - vmin)

    if not np.isfinite(denom) or denom <= 0.0:
        scaled = np.zeros_like(a, dtype=out_dt)
    else:
        scaled_f = (a - vmin) * (resolution / denom)
        scaled_f[~np.isfinite(scaled_f)] = 0.0
        scaled_f = np.clip(scaled_f, 0.0, clip_hi)
        scaled = scaled_f.astype(out_dt, copy=False)

    scaled = np.ascontiguousarray(scaled, dtype=out_dt)
    blob = scaled.tobytes(order="C")
    size_bytes = len(blob)      
    size_mb = size_bytes / (1024*1024)

    header: Dict[str, Any] = {
        "version": VERSION,
        "dtype": dtype_str,          # "uint8" | "uint16"
        "endianness": "LE",          # payload endianness
        "shape": list(scaled.shape), # [H, W]
        "order": "C",
        "size": size_mb,
        "resolution": resolution,    # 255 or 4096
        "vmin": float(vmin),
        "vmax": float(vmax),
    }
    return blob, header

def pack_envelope(blob: bytes, header: Dict[str, Any]) -> bytes:
    """
    MAGIC(4) | VERSION(1) | HEADER_LEN(4, BE) | HEADER_JSON(utf-8) | PADDING(0 or 1) | BLOB
    """
    header_json = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    header_len = len(header_json)

    # Offset right after HEADER_JSON:
    off = 4 + 1 + 4 + header_len  # MAGIC + VERSION + HEADER_LEN + HEADER_JSON

    # Pad to 2-byte boundary so BLOB is aligned for Uint16Array
    pad_len = (-off) & 1  # = 1 if off is odd, else 0
    pad = b"\x00" * pad_len

    return b"".join([
        MAGIC,
        struct.pack("B", VERSION),
        struct.pack(">I", header_len),
        header_json,
        pad,
        blob
    ])

def unpack_envelope(buf: bytes) -> Tuple[Dict[str, Any], bytes]:
    """
    Parse envelope → (header, blob)
    """
    if len(buf) < 9:
        raise ValueError("Buffer too small.")
    off = 0
    magic = buf[off:off+4]; off += 4
    if magic != MAGIC:
        raise ValueError(f"Bad MAGIC {magic!r}")
    version = buf[off]; off += 1
    if version < 1:
        raise ValueError(f"Unsupported version {version}")
    (header_len,) = struct.unpack(">I", buf[off:off+4]); off += 4
    header_json = buf[off:off+header_len]; off += header_len
    header = json.loads(header_json.decode("utf-8"))
    blob = buf[off:]
    return header, blob

def blob_to_ndarray(header: Dict[str, Any], blob: bytes) -> np.ndarray:
    """
    Reconstruct numpy array from header + blob.
    """
    dtype = header["dtype"]
    shape = tuple(header["shape"])
    endianness = header.get("endianness", "LE")
    order = header.get("order", "C")

    if dtype == "uint8":
        dt = np.dtype("u1")
    elif dtype == "uint16":
        dt = np.dtype("<u2") if endianness == "LE" else np.dtype(">u2")
    else:
        raise ValueError(f"Unsupported dtype {dtype!r}")

    arr = np.frombuffer(blob, dtype=dt)
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise ValueError(f"Blob size {arr.size} != {shape}")
    return arr.reshape(shape, order=order)
