# /imgpack/utils.py
from __future__ import annotations
import json
import struct
from typing import Dict, Tuple, Any
import numpy as np

# ---- Envelope: HEADER_LEN(4, BE) | HEADER_JSON(utf-8) | PADDING | BLOB ----
# Header schema:
#   format, version, dtype, endianness, shape, order, size, bits,
#   resolution, vmin, vmax
#
# Notes:
# - dtype: "packed" | "uint8" | "uint16" | "uint32" | "float16" | "float32" | "float64"
# - size: payload element size in bits for typed containers (8/16/32/64),
#         or BLOB byte size in MB (MiB) for reporting (we store MB in header.size)
#   -> Per your request, header.size is the BLOB size in MB (MiB).
# - bits: quantization precision for "packed" and integer containers; ignored for floats
# - endianness: "LE"/"BE" for integer/float containers; None for "packed"
# - packed payload is byte-stream; no endianness

_INT_DT_SIZES = {"uint8": 8, "uint16": 16, "uint32": 32}
_FLOAT_DT_SIZES = {"float16": 16, "float32": 32, "float64": 64}
_SUPPORTED_DTYPES = set(list(_INT_DT_SIZES.keys()) + list(_FLOAT_DT_SIZES.keys()) + ["packed"])

def _np_dtype(dtype: str, endianness: str = "LE") -> np.dtype:
    d = dtype.lower()
    if d == "uint8":
        return np.dtype("u1")
    if d == "uint16":
        return np.dtype("<u2" if endianness == "LE" else ">u2")
    if d == "uint32":
        return np.dtype("<u4" if endianness == "LE" else ">u4")
    if d == "float16":
        return np.dtype("<f2" if endianness == "LE" else ">f2")
    if d == "float32":
        return np.dtype("<f4" if endianness == "LE" else ">f4")
    if d == "float64":
        return np.dtype("<f8" if endianness == "LE" else ">f8")
    raise ValueError(f"Unsupported dtype {dtype!r}")

def _quantize_nbit(data: np.ndarray, vmin: float, vmax: float, bits: int) -> np.ndarray:
    levels = 1 << bits
    a = np.asarray(data, dtype=np.float32)
    denom = float(vmax - vmin)
    if not np.isfinite(denom) or denom <= 0.0:
        q = np.zeros_like(a, dtype=np.uint32)
    else:
        scaled = (a - vmin) * ((levels - 1) / denom)
        scaled[~np.isfinite(scaled)] = 0.0
        q = np.clip(scaled, 0.0, float(levels - 1)).astype(np.uint32, copy=False)
    return q

def _pack_nbits(values: np.ndarray, nbits: int) -> bytes:
    vals = np.asarray(values, dtype=np.uint32).ravel()
    out = bytearray(); acc = 0; acc_bits = 0; mask = (1 << nbits) - 1
    for v in vals:
        acc |= (int(v) & mask) << acc_bits
        acc_bits += nbits
        while acc_bits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8; acc_bits -= 8
    if acc_bits:
        out.append(acc & 0xFF)
    return bytes(out)

def _unpack_nbits(blob: bytes, nbits: int, count: int) -> np.ndarray:
    out = np.empty(count, dtype=np.uint32)
    acc = 0; acc_bits = 0; mask = (1 << nbits) - 1; bi = 0; b = blob
    for i in range(count):
        while acc_bits < nbits:
            if bi >= len(b): raise ValueError("Packed data underrun")
            acc |= b[bi] << acc_bits
            acc_bits += 8; bi += 1
        out[i] = acc & mask
        acc >>= nbits; acc_bits -= nbits
    return out

def _align_pad_len(offset: int, elem_size: int) -> int:
    # pad so (offset + pad) % elem_size == 0
    if elem_size <= 1:
        return 0
    return (-offset) & (elem_size - 1)

def encode(
    data: np.ndarray,
    vmin: float,
    vmax: float,
    dtype: str = "uint16",
    bits: int = 12,
    *,
    endianness: str = "LE",
    version: int = 1,
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Encode 2D array using:
      - dtype="packed": true n-bit bitstream (bits=1..32), endianness=None
      - dtype in {"uint8","uint16","uint32"}: quantize to `bits` (1..container size) and store in that container
      - dtype in {"float16","float32","float64"}: raw floats; bits ignored

    Header keys: format, version, dtype, endianness, shape, order, size, bits, resolution, vmin, vmax
      - size (header) = BLOB byte size in MB (MiB)
    """
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("encode expects a 2D array (H, W).")

    H, W = data.shape
    dtype_l = dtype.lower()
    if dtype_l not in _SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported dtype {dtype!r}")

    if dtype_l == "packed":
        if not (1 <= bits <= 32):
            raise ValueError("For dtype='packed', bits must be 1..32")
        q = _quantize_nbit(data, vmin, vmax, bits)
        blob = _pack_nbits(q, bits)
        size_mb = len(blob) / 1048576.0
        header: Dict[str, Any] = {
            "format": "imgpack",
            "version": version,
            "dtype": "packed",
            "endianness": None,
            "shape": [H, W],
            "order": "C",
            "size": size_mb,            # MB of BLOB
            "bits": bits,               # quantization precision
            "resolution": 1 << bits,    # 2**bits
            "vmin": float(vmin),
            "vmax": float(vmax),
        }
        return blob, header

    if dtype_l in _INT_DT_SIZES:
        container_bits = _INT_DT_SIZES[dtype_l]
        if not (1 <= bits <= container_bits):
            raise ValueError(f"bits must be 1..{container_bits} for dtype={dtype_l}, got {bits}")
        q = _quantize_nbit(data, vmin, vmax, bits)
        dt = _np_dtype(dtype_l, endianness)
        arr = np.ascontiguousarray(q.astype(dt, copy=False))
        blob = arr.tobytes(order="C")
        size_mb = len(blob) / 1048576.0
        header = {
            "format": "imgpack",
            "version": version,
            "dtype": dtype_l,
            "endianness": endianness,   # ignored by uint8
            "shape": [H, W],
            "order": "C",
            "size": size_mb,            # MB of BLOB
            "bits": bits,               # quantization precision
            "resolution": 1 << bits,    # 2**bits (effective precision)
            "vmin": float(vmin),
            "vmax": float(vmax),
        }
        return blob, header

    # floats
    dt = _np_dtype(dtype_l, endianness)
    arr = np.ascontiguousarray(data, dtype=dt)
    blob = arr.tobytes(order="C")
    size_mb = len(blob) / 1048576.0
    header = {
        "format": "imgpack",
        "version": version,
        "dtype": dtype_l,
        "endianness": endianness,
        "shape": [H, W],
        "order": "C",
        "size": size_mb,            # MB of BLOB
        "bits": None,
        "resolution": None,
        "vmin": float(vmin),
        "vmax": float(vmax),
    }
    return blob, header

def pack_envelope(blob: bytes, header: Dict[str, Any]) -> bytes:
    """
    HEADER_LEN(4, BE) | HEADER_JSON | PADDING | BLOB
    Padding aligns to element size: packed=1, uint8=1, uint16/float16=2, uint32/float32=4, float64=8
    """
    header_json = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    header_len = len(header_json)

    # Determine element size for alignment
    dtype_l = str(header.get("dtype", "")).lower()
    if dtype_l == "packed":
        elem_size = 1
    elif dtype_l in _INT_DT_SIZES:
        elem_size = _INT_DT_SIZES[dtype_l] // 8
    elif dtype_l in _FLOAT_DT_SIZES:
        elem_size = _FLOAT_DT_SIZES[dtype_l] // 8
    else:
        elem_size = 1

    off = 4 + header_len
    pad_len = _align_pad_len(off, elem_size)
    pad = b"\x00" * pad_len

    return b"".join([struct.pack(">I", header_len), header_json, pad, blob])

def unpack_envelope(buf: bytes) -> Tuple[Dict[str, Any], bytes]:
    if len(buf) < 4:
        raise ValueError("Buffer too small for HEADER_LEN.")
    (header_len,) = struct.unpack(">I", buf[:4])
    if header_len < 0 or 4 + header_len > len(buf):
        raise ValueError("Invalid HEADER_LEN.")
    off = 4
    header_json = buf[off:off + header_len]; off += header_len

    try:
        header_str = header_json.decode("utf-8")
        header = json.loads(header_str)
    except Exception as e:
        raise ValueError("Failed to parse HEADER_JSON") from e

    dtype_l = str(header.get("dtype", "")).lower()
    if dtype_l == "packed":
        elem_size = 1
    elif dtype_l in _INT_DT_SIZES:
        elem_size = _INT_DT_SIZES[dtype_l] // 8
    elif dtype_l in _FLOAT_DT_SIZES:
        elem_size = _FLOAT_DT_SIZES[dtype_l] // 8
    else:
        elem_size = 1

    pad_len = _align_pad_len(4 + header_len, elem_size)
    off += pad_len
    blob = buf[off:]
    return header, blob

def decode(header: Dict[str, Any], blob: bytes) -> np.ndarray:
    dtype_l = header["dtype"].lower()
    shape = tuple(header["shape"])
    order = header.get("order", "C")

    if dtype_l == "packed":
        bits = int(header["bits"])
        flat = _unpack_nbits(blob, bits, count=int(np.prod(shape)))
        # place into smallest uint container that can hold the range
        if bits <= 8:
            arr = flat.astype(np.uint8, copy=False)
        elif bits <= 16:
            arr = flat.astype(np.uint16, copy=False)
        else:
            arr = flat.astype(np.uint32, copy=False)
        return arr.reshape(shape, order=order)

    dt = _np_dtype(dtype_l, header.get("endianness", "LE"))
    arr = np.frombuffer(blob, dtype=dt)
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise ValueError(f"Blob size {arr.size} does not match header shape {shape}.")
    return arr.reshape(shape, order=order)
