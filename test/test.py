# /test/test.py
import numpy as np
from imgpack.utils import encode, pack_envelope, unpack_envelope, decode

def show(hdr, name):
    print(f"\n=== {name} ===")
    for k in ["format","version","dtype","endianness","shape","order","size","bits","resolution","vmin","vmax"]:
        print(f"{k}: {hdr.get(k)}")

def main():
    H, W = 128, 192
    img = np.linspace(0, 1, H*W, dtype=np.float32).reshape(H, W)

    # 1) True 12-bit packed
    blob, hdr = encode(img, vmin=0.0, vmax=1.0, dtype="packed", bits=12)
    env = pack_envelope(blob, hdr)
    h, b = unpack_envelope(env)
    arr = decode(h, b)
    assert arr.shape == (H, W)
    assert arr.dtype in (np.uint16, np.uint32)  # smallest container that fits 12 bits => uint16
    show(h, "packed-12bit")

    # 2) Quantized into uint16 (12-bit precision, 16-bit container)
    blob, hdr = encode(img, 0.0, 1.0, dtype="uint16", bits=12)
    env = pack_envelope(blob, hdr)
    h, b = unpack_envelope(env)
    arr = decode(h, b)
    assert arr.shape == (H, W)
    assert arr.dtype == np.uint16
    show(h, "uint16-12bit")

    # 3) Quantized into uint8 (8-bit precision, 8-bit container)
    blob, hdr = encode(img, 0.0, 1.0, dtype="uint8", bits=8)
    env = pack_envelope(blob, hdr)
    h, b = unpack_envelope(env)
    arr = decode(h, b)
    assert arr.shape == (H, W)
    assert arr.dtype == np.uint8
    show(h, "uint8-8bit")

    # 4) Float32 (no quantization)
    blob, hdr = encode(img, 0.0, 1.0, dtype="float32", bits=0)
    env = pack_envelope(blob, hdr)
    h, b = unpack_envelope(env)
    arr = decode(h, b)
    assert arr.shape == (H, W)
    assert arr.dtype == np.float32
    show(h, "float32")

    print("\nAll tests passed.")

if __name__ == "__main__":
    main()
