# ImgPack

A simple binary format for packing and streaming 2D NumPy arrays (images) from Python to the browser. Built with python with zero dependencies.

ImgPack supports multiple transport modes:

* **Packed**: true **n-bit bitstream** (e.g. 12-bit packed into 1.5 bytes per pixel).
* **Integer containers**: quantized to `bits` and stored in `uint8`, `uint16`, or `uint32`.
* **Float containers**: raw `float16`, `float32`, or `float64` with no quantization.

---

## Install

```bash
git clone https://github.com/michaelmendoza/imgpack
```

---

## 🚀 Quick Start with Demo

### 1. Install dependencies

```bash
pip install fastapi uvicorn numpy
```

### 2. Run the demo server

```bash
python demo/run.py
```

Server runs at [http://localhost:8765](http://localhost:8765).

### 3. Open the browser

Visit [http://localhost:8765](http://localhost:8765) and you’ll see a grayscale image rendered from a NumPy array streamed over a WebSocket.

---

## 🖼 Examples

### Python (encode + pack)

```python
import numpy as np
from imgpack.utils import encode, pack_envelope

arr = np.linspace(0, 1, 64*64, dtype=np.float32).reshape(64, 64)

# Example 1: true 12-bit packed stream
blob, header = encode(arr, vmin=0.0, vmax=1.0, dtype="packed", bits=12)
envelope = pack_envelope(blob, header)

# Example 2: 12-bit quantization into uint16 container
blob, header = encode(arr, vmin=0.0, vmax=1.0, dtype="uint16", bits=12)
envelope = pack_envelope(blob, header)

# Example 3: raw float32 transport (no quantization)
blob, header = encode(arr, vmin=0.0, vmax=1.0, dtype="float32", bits=0)
envelope = pack_envelope(blob, header)

# send `envelope` over a WebSocket:
# await ws.send_bytes(envelope)
```

### Python (decode)

```python
from imgpack.utils import unpack_envelope, decode

header, blob = unpack_envelope(envelope)
arr_decoded = decode(header, blob)

print("Decoded shape:", arr_decoded.shape)
print("Decoded dtype:", arr_decoded.dtype)
print("Header:", header)
```

Output (example):

```
Decoded shape: (64, 64)
Decoded dtype: uint16
Header: {
  'format': 'imgpack',
  'version': 1,
  'dtype': 'uint16',
  'endianness': 'LE',
  'shape': [64, 64],
  'order': 'C',
  'size': 0.016,          # BLOB size in MB
  'bits': 12,
  'resolution': 4096,
  'vmin': 0.0,
  'vmax': 1.0
}
```

### JavaScript (browser decode)

```js
import { readHeaderAndBlob, blobToTypedArray } from "/decode.js";

ws.onmessage = async (evt) => {
  const ab = evt.data instanceof Blob ? await evt.data.arrayBuffer() : evt.data;
  const { header, blobBytes } = readHeaderAndBlob(ab);
  const typed = blobToTypedArray(header, blobBytes);
  renderGrayscale(canvas, typed, header);
};
```

---

## 📦 Binary Format

```
HEADER_LEN(4, BE) | HEADER_JSON(UTF-8) | PAD | BLOB
```

* **HEADER\_LEN**: JSON header size (uint32, big-endian)
* **HEADER\_JSON**: metadata (`format`, `version`, `dtype`, `shape`, `bits`, `resolution`, etc.)
* **PAD**: zero bytes to align BLOB start to element size (1, 2, 4, or 8)
* **BLOB**: raw packed/quantized/float data

---

## ⚠️ Limitations

* Only 2D arrays are supported right now.
* Quantization is **lossy** for `packed` and integer encodings (values outside `[vmin, vmax]` are clipped).
* For floats, values are stored losslessly but still aligned to chosen precision (`float16/32/64`).
