# /test/test.py
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')
import numpy as np
from imgpack import (
    encode, encode_data, pack_data,
    decode, decode_data, unpack_data
)

def show(hdr, name):
    print(f"\n=== {name} ===")
    for k in ["format","version","dtype","endianness","shape","order","size","bits","resolution","vmin","vmax"]:
        print(f"{k}: {hdr.get(k)}")

def test_low_level():
    H, W = 128, 192
    img = np.linspace(0, 1, H*W, dtype=np.float32).reshape(H, W)

    # 1) True 12-bit packed
    blob, hdr = encode_data(img, vmin=0.0, vmax=1.0, dtype="packed", bits=12)
    env = pack_data(blob, hdr)
    h, b = unpack_data(env)
    arr = decode_data(h, b)
    assert arr.shape == (H, W)
    assert arr.dtype in (np.uint16, np.uint32)  # smallest container that fits 12 bits => uint16
    show(h, "packed-12bit")

    # 2) Quantized into uint16 (12-bit precision, 16-bit container)
    blob, hdr = encode_data(img, 0.0, 1.0, dtype="uint16", bits=12)
    env = pack_data(blob, hdr)
    h, b = unpack_data(env)
    arr = decode_data(h, b)
    assert arr.shape == (H, W)
    assert arr.dtype == np.uint16
    show(h, "uint16-12bit")

    # 3) Quantized into uint8 (8-bit precision, 8-bit container)
    blob, hdr = encode_data(img, 0.0, 1.0, dtype="uint8", bits=8)
    env = pack_data(blob, hdr)
    h, b = unpack_data(env)
    arr = decode_data(h, b)
    assert arr.shape == (H, W)
    assert arr.dtype == np.uint8
    show(h, "uint8-8bit")

    # 4) Float32 (no quantization)
    blob, hdr = encode_data(img, 0.0, 1.0, dtype="float32", bits=0)
    env = pack_data(blob, hdr)
    h, b = unpack_data(env)
    arr = decode_data(h, b)
    assert arr.shape == (H, W)
    assert arr.dtype == np.float32
    show(h, "float32")

def test_api():
    H, W = 128, 192
    img = np.linspace(0, 1, H*W, dtype=np.float32).reshape(H, W)

    # Primary API: encode → bytes, decode → (header, ndarray)
    env = encode(img, vmin=0.0, vmax=1.0, dtype="packed", bits=12)
    hdr, arr = decode(env)
    assert arr.shape == (H, W)
    show(hdr, "encode/decode (packed 12-bit)")

    # Lower-level control: encode_data + pack_data
    blob, hdr = encode_data(img, 0.0, 1.0, dtype="uint16", bits=12)
    env = pack_data(blob, hdr)
    hdr2, blob2 = unpack_data(env)
    arr2 = decode_data(hdr2, blob2)
    assert arr2.dtype == np.uint16
    show(hdr2, "encode_data/pack_data then unpack_data/decode_data (uint16 12-bit)")

    # Float32
    env = encode(img, 0.0, 1.0, dtype="float32", bits=0)
    hdr3, arr3 = decode(env)
    assert arr3.dtype == np.float32
    show(hdr3, "encode/decode (float32)")

if __name__ == "__main__":
    try:
        test_low_level()
        test_api()
        print("\nAll tests passed.")
    except Exception as e:
        print(f"\n\n*** ERROR: {e}")